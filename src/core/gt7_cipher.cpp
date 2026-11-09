// SPDX-License-Identifier: GPL-2.0-or-later
//
// GT7 v1.00's file cipher, faster. Every file the game reads goes through
// ContextChaCha20::decrypt (eboot 0x3a26500):
//
//   void decrypt(Context* this, const u8* src, s64 size, u64 offset, u8* dst)
//     this+0x10  cipher: vtable (slot 1 seeks to a block, slot 2 steps one
//                block), then the ChaCha20 state, 16 words (DJB layout: 64-bit
//                counter in words 12-13, nonce in 14-15)
//     dst null or equal to src: in place
//   seek(offset / 64); keystream = block(state); pos = offset % 64
//   for each byte: dst = src ^ keystream[pos++];
//                  at pos 64: step(); keystream = block(state); pos = 0
//
// One byte at a time with a vtable call per block, ~3.5x slower than needed
// under Rosetta. Here the game's own seek still runs; with the default step
// (0x325e140: 64-bit counter + 1) blocks are computed four at once (SSE) and
// XORed 16 bytes at a time; any other step runs the game's loop as written
// (Generic). Same idea as shadgt7's PERF-GT7-113 for its executable.

#include "core/gt7_cipher.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include <sys/mman.h>
#include <tmmintrin.h>

#include "common/log.h"

namespace Core::Gt7Cipher {

namespace {

constexpr u64 kDecrypt = 0x3a26500;
constexpr u64 kStep = 0x325e140;   // counter + 1
constexpr u64 kVtable = 0x5bd27c0; // a cipher vtable with seek 0x325e130 and that step (self-test)

constexpr u8 kDecryptHead[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48,
                               0x83, 0xec, 0x58, 0x48, 0x89, 0xcb, 0x48, 0x8b, 0x0d, 0xa5, 0xef, 0x22, 0x02, 0x49,
                               0x89, 0xfe, 0x48, 0x85, 0xf6, 0x48, 0x8b, 0x01, 0x48, 0x89, 0x45, 0xd0};
constexpr u8 kStepCode[] = {0xff, 0x47, 0x38, 0x48, 0x89, 0xf8, 0x75, 0x03, 0xff, 0x40, 0x3c, 0xc3};

using DecryptFn = void (*)(u8*, const u8*, s64, u64, u8*);
using SeekFn = void* (*)(u8*, u64);
using StepFn = void* (*)(u8*);

u64 g_step = 0; // the default Step's address
DecryptFn g_game = nullptr; // the game's function, through a trampoline (its prologue, then a jump back)
enum class Mode { Fast, Game, Verify } g_mode = Mode::Fast;
std::atomic<u64> g_mismatch{0};

std::atomic<u64> g_calls{0}, g_bytes{0}, g_ns{0};
bool g_diag = false;

inline u32 Rotl(u32 v, int n) {
    return (v << n) | (v >> (32 - n));
}

// One block: out = rounds(in) + in.
void Block(const u32 in[16], u8 out[64]) {
    u32 x[16];
    std::memcpy(x, in, 64);
    auto q = [&](int a, int b, int c, int d) {
        x[a] += x[b]; x[d] = Rotl(x[d] ^ x[a], 16);
        x[c] += x[d]; x[b] = Rotl(x[b] ^ x[c], 12);
        x[a] += x[b]; x[d] = Rotl(x[d] ^ x[a], 8);
        x[c] += x[d]; x[b] = Rotl(x[b] ^ x[c], 7);
    };
    for (int r = 0; r < 10; ++r) {
        q(0, 4, 8, 12); q(1, 5, 9, 13); q(2, 6, 10, 14); q(3, 7, 11, 15);
        q(0, 5, 10, 15); q(1, 6, 11, 12); q(2, 7, 8, 13); q(3, 4, 9, 14);
    }
    for (int i = 0; i < 16; ++i) {
        const u32 w = x[i] + in[i];
        std::memcpy(out + i * 4, &w, 4);
    }
}

// Four consecutive blocks (counter, +1, +2, +3) into out[256].
__attribute__((target("ssse3"))) void Block4(const u32 in[16], u8 out[256]) {
    const __m128i rot16 = _mm_setr_epi8(2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13);
    const __m128i rot8 = _mm_setr_epi8(3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14);
    __m128i s[16], x[16];
    for (int i = 0; i < 16; ++i) {
        s[i] = _mm_set1_epi32(static_cast<int>(in[i]));
    }
    // 64-bit counter per lane.
    const u64 c = u64(in[12]) | (u64(in[13]) << 32);
    s[12] = _mm_setr_epi32(int(u32(c)), int(u32(c + 1)), int(u32(c + 2)), int(u32(c + 3)));
    s[13] = _mm_setr_epi32(int(u32(c >> 32)), int(u32((c + 1) >> 32)), int(u32((c + 2) >> 32)),
                           int(u32((c + 3) >> 32)));
    for (int i = 0; i < 16; ++i) {
        x[i] = s[i];
    }
    auto q = [&](int a, int b, int cc, int d) {
        x[a] = _mm_add_epi32(x[a], x[b]); x[d] = _mm_shuffle_epi8(_mm_xor_si128(x[d], x[a]), rot16);
        x[cc] = _mm_add_epi32(x[cc], x[d]); x[b] = _mm_xor_si128(x[b], x[cc]);
        x[b] = _mm_or_si128(_mm_slli_epi32(x[b], 12), _mm_srli_epi32(x[b], 20));
        x[a] = _mm_add_epi32(x[a], x[b]); x[d] = _mm_shuffle_epi8(_mm_xor_si128(x[d], x[a]), rot8);
        x[cc] = _mm_add_epi32(x[cc], x[d]); x[b] = _mm_xor_si128(x[b], x[cc]);
        x[b] = _mm_or_si128(_mm_slli_epi32(x[b], 7), _mm_srli_epi32(x[b], 25));
    };
    for (int r = 0; r < 10; ++r) {
        q(0, 4, 8, 12); q(1, 5, 9, 13); q(2, 6, 10, 14); q(3, 7, 11, 15);
        q(0, 5, 10, 15); q(1, 6, 11, 12); q(2, 7, 8, 13); q(3, 4, 9, 14);
    }
    for (int i = 0; i < 16; ++i) {
        x[i] = _mm_add_epi32(x[i], s[i]);
    }
    // Transpose each group of four words: lane b of words 4g..4g+3 -> block b.
    for (int g = 0; g < 4; ++g) {
        __m128i a0 = x[4 * g], a1 = x[4 * g + 1], a2 = x[4 * g + 2], a3 = x[4 * g + 3];
        const __m128i t0 = _mm_unpacklo_epi32(a0, a1), t1 = _mm_unpacklo_epi32(a2, a3);
        const __m128i t2 = _mm_unpackhi_epi32(a0, a1), t3 = _mm_unpackhi_epi32(a2, a3);
        a0 = _mm_unpacklo_epi64(t0, t1);
        a1 = _mm_unpackhi_epi64(t0, t1);
        a2 = _mm_unpacklo_epi64(t2, t3);
        a3 = _mm_unpackhi_epi64(t2, t3);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 0 * 64 + g * 16), a0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 1 * 64 + g * 16), a1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 2 * 64 + g * 16), a2);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 3 * 64 + g * 16), a3);
    }
}

void XorBytes(u8* d, const u8* k, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(d + i));
        const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(k + i));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(d + i), _mm_xor_si128(a, b));
    }
    for (; i < n; ++i) {
        d[i] ^= k[i];
    }
}

void XorInto(u8* dst, const u8* src, const u8* k, size_t n) {
    if (dst == src) {
        XorBytes(dst, k, n);
        return;
    }
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(k + i));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), _mm_xor_si128(a, b));
    }
    for (; i < n; ++i) {
        dst[i] = src[i] ^ k[i];
    }
}

// After the seek, with the default step: four blocks at a time. The counter
// ends where the game's leaves it (one step for each block it runs to the end).
void Fast(u8* cipher, const u8* src, u64 size, u64 offset, u8* dst) {
    u32* st = reinterpret_cast<u32*>(cipher + 8);
    alignas(16) u8 ks[256];
    u64 pos = offset & 63;
    while (size) {
        const u64 need = (pos + size + 63) / 64; // blocks this call touches from here
        const u64 blocks = std::min<u64>(need, 4);
        if (blocks == 4) {
            Block4(st, ks);
        } else {
            u32 tmp[16];
            std::memcpy(tmp, st, 64);
            for (u64 b = 0; b < blocks; ++b) {
                Block(tmp, ks + b * 64);
                if (++tmp[12] == 0) ++tmp[13];
            }
        }
        const u64 n = std::min<u64>(size, blocks * 64 - pos);
        XorInto(dst, src, ks + pos, n);
        const u64 done = (pos + n) / 64; // blocks used to their end: stepped past
        const u64 c = (u64(st[12]) | (u64(st[13]) << 32)) + done;
        st[12] = u32(c);
        st[13] = u32(c >> 32);
        pos = (pos + n) & 63;
        src += n;
        dst += n;
        size -= n;
    }
}

// Any other step: the game's loop, its step called at each block end.
void Generic(u8* cipher, const u8* src, u64 size, u64 offset, u8* dst, StepFn step) {
    const u32* st = reinterpret_cast<const u32*>(cipher + 8);
    alignas(16) u8 ks[64];
    u64 pos = offset & 63;
    Block(st, ks);
    while (size) {
        const u64 n = std::min<u64>(size, 64 - pos);
        XorInto(dst, src, ks + pos, n);
        pos += n;
        src += n;
        dst += n;
        size -= n;
        if (pos == 64) {
            step(cipher);
            Block(st, ks);
            pos = 0;
        }
    }
}

void Run(u8* self, const u8* src, s64 size, u64 offset, u8* dst);
void RunFast(u8* self, const u8* src, s64 size, u64 offset, u8* dst);

void Hook(u8* self, const u8* src, s64 size, u64 offset, u8* dst) {
    if (g_mode == Mode::Verify && src && size > 0 && size <= (64 << 20)) {
        // The game's function on copies of the context and data, ours on the real ones.
        alignas(16) u8 ctx[0x80];
        std::memcpy(ctx, self, sizeof(ctx));
        std::vector<u8> in(src, src + size), out(static_cast<size_t>(size));
        u8* game_dst = (!dst || dst == src) ? in.data() : out.data();
        g_game(ctx, in.data(), size, offset, (!dst || dst == src) ? nullptr : out.data());
        Run(self, src, size, offset, dst);
        const u8* ours = (!dst || dst == src) ? src : dst;
        if (std::memcmp(game_dst, ours, static_cast<size_t>(size)) != 0 || std::memcmp(ctx + 0x10, self + 0x10, 0x48) != 0) {
            if (g_mismatch.fetch_add(1) < 8) {
                DC_ERROR("gt7 cipher: VERIFY mismatch (size %lld, offset %#llx)", static_cast<long long>(size),
                         static_cast<unsigned long long>(offset));
            }
        }
        return;
    }
    Run(self, src, size, offset, dst);
}

void Run(u8* self, const u8* src, s64 size, u64 offset, u8* dst) {
    if (g_mode == Mode::Game) {
        const auto t0 = g_diag ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        g_game(self, src, size, offset, dst);
        if (g_diag && src && size > 0) {
            g_calls.fetch_add(1, std::memory_order_relaxed);
            g_bytes.fetch_add(static_cast<u64>(size), std::memory_order_relaxed);
            g_ns.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                std::chrono::steady_clock::now() - t0)
                                                .count()),
                           std::memory_order_relaxed);
        }
        return;
    }
    RunFast(self, src, size, offset, dst);
}

// The game's seek, then Fast (default step) or Generic.
void RunFast(u8* self, const u8* src, s64 size, u64 offset, u8* dst) {
    if (!src || size == 0) {
        return; // the game logs these and returns
    }
    const auto t0 = g_diag ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    u8* cipher = self + 0x10;
    const u64 vtbl = *reinterpret_cast<const u64*>(cipher);
    reinterpret_cast<SeekFn>(*reinterpret_cast<const u64*>(vtbl + 0x08))(cipher, offset >> 6);
    if (size > 0) {
        if (!dst) {
            dst = const_cast<u8*>(src);
        }
        const u64 step = *reinterpret_cast<const u64*>(vtbl + 0x10);
        if (step == g_step) {
            Fast(cipher, src, static_cast<u64>(size), offset, dst);
        } else {
            Generic(cipher, src, static_cast<u64>(size), offset, dst, reinterpret_cast<StepFn>(step));
        }
    }
    if (g_diag) {
        g_calls.fetch_add(1, std::memory_order_relaxed);
        g_bytes.fetch_add(static_cast<u64>(size), std::memory_order_relaxed);
        g_ns.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now() - t0)
                                            .count()),
                       std::memory_order_relaxed);
    }
}

// The game's function against Hook on the same contexts and data: in place
// and out of place, offsets inside a block, sizes around block and 4-block
// edges, a counter about to carry into word 13.
bool SelfTest(u64 base) {
    const auto original = reinterpret_cast<DecryptFn>(base + kDecrypt);
    std::mt19937_64 rng(7);
    const u64 sizes[] = {1, 13, 63, 64, 65, 191, 255, 256, 257, 1000, 4096, 65536 + 77};
    const u64 offsets[] = {0, 1, 63, 64, 100, 0x10000 + 5, 0x3fffffffc0ull};
    for (const u64 size : sizes) {
        for (const u64 offset : offsets) {
            for (const bool in_place : {true, false}) {
                alignas(16) u8 ctx_a[0x60] = {}, ctx_b[0x60] = {};
                const u64 vtbl = base + kVtable;
                std::memcpy(ctx_a + 0x10, &vtbl, 8);
                for (int i = 0x18; i < 0x58; ++i) ctx_a[i] = static_cast<u8>(rng());
                std::memcpy(ctx_b, ctx_a, sizeof(ctx_a));
                std::vector<u8> src(size + 7), out_a(size + 7, 0xcc), out_b(size + 7, 0xcc);
                for (auto& v : src) v = static_cast<u8>(rng());
                std::vector<u8> src_a = src, src_b = src;
                original(ctx_a, src_a.data() + 3, static_cast<s64>(size), offset,
                         in_place ? nullptr : out_a.data() + 1);
                RunFast(ctx_b, src_b.data() + 3, static_cast<s64>(size), offset, in_place ? nullptr : out_b.data() + 1);
                if (src_a != src_b || out_a != out_b || std::memcmp(ctx_a, ctx_b, sizeof(ctx_a)) != 0) {
                    DC_WARN("gt7 cipher: self-test failed (size %llu, offset %#llx, %s)",
                            static_cast<unsigned long long>(size), static_cast<unsigned long long>(offset),
                            in_place ? "in place" : "out of place");
                    return false;
                }
            }
        }
    }
    return true;
}

} // namespace

void Install(u64 base) {
    if (const char* e = std::getenv("DC_FAST_CIPHER")) {
        if (e[0] == '0') {
            return;
        }
        if (std::strcmp(e, "game") == 0) {
            g_mode = Mode::Game; // the game's code, behind the hook (DC_DIAG_CIPHER measures it)
        }
    }
    if (const char* v = std::getenv("DC_FAST_CIPHER_VERIFY"); v && v[0] && v[0] != '0') {
        g_mode = Mode::Verify;
    }
    auto* code = reinterpret_cast<u8*>(base + kDecrypt);
    if (std::memcmp(code, kDecryptHead, sizeof(kDecryptHead)) != 0 ||
        std::memcmp(reinterpret_cast<const void*>(base + kStep), kStepCode, sizeof(kStepCode)) != 0 ||
        *reinterpret_cast<const u64*>(base + kVtable + 0x10) != base + kStep) {
        DC_WARN("gt7 cipher: function not as expected, left as it is");
        return;
    }
    g_step = base + kStep;
    const auto t0 = std::chrono::steady_clock::now();
    if (!SelfTest(base)) {
        return;
    }
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    // 8 MB through each: the speed-up on this machine (under Rosetta here).
    double ms_game = 0, ms_fast = 0;
    {
        std::vector<u8> buf(8u << 20, 0x5a);
        alignas(16) u8 ctx[0x60] = {};
        const u64 vtbl = base + kVtable;
        std::memcpy(ctx + 0x10, &vtbl, 8);
        auto time = [&](auto&& fn) {
            const auto s = std::chrono::steady_clock::now();
            fn();
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count();
        };
        const s64 n = static_cast<s64>(buf.size());
        ms_game = time([&] { reinterpret_cast<DecryptFn>(base + kDecrypt)(ctx, buf.data(), n, 0, nullptr); });
        ms_fast = time([&] { RunFast(ctx, buf.data(), n, 0, nullptr); });
    }
    // Trampoline: the 12 prologue bytes the jump replaces, then movabs rax, kDecrypt+12 ; jmp rax.
    void* tramp = mmap(nullptr, 0x1000, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (tramp == MAP_FAILED) {
        return;
    }
    {
        u8* t = static_cast<u8*>(tramp);
        std::memcpy(t, code, 12);
        const u8 jmp[12] = {0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xe0};
        std::memcpy(t + 12, jmp, 12);
        const u64 back = base + kDecrypt + 12;
        std::memcpy(t + 14, &back, 8);
        mprotect(tramp, 0x1000, PROT_READ | PROT_EXEC);
        g_game = reinterpret_cast<DecryptFn>(tramp);
    }
    const u64 page = reinterpret_cast<u64>(code) & ~0xFFFull;
    mprotect(reinterpret_cast<void*>(page), 0x2000, PROT_READ | PROT_WRITE | PROT_EXEC);
    // movabs rax, Hook ; jmp rax (12 bytes: the prologue's first instructions; never returned to)
    u8 jump[12] = {0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xe0};
    const u64 target = reinterpret_cast<u64>(&Hook);
    std::memcpy(jump + 2, &target, 8);
    std::memcpy(code, jump, sizeof(jump));
    mprotect(reinterpret_cast<void*>(page), 0x2000, PROT_READ | PROT_EXEC);
    DC_INFO("gt7 cipher: %s (self-test passed in %.1f ms; 8 MB: game %.1f ms, fast %.1f ms)",
            g_mode == Mode::Game ? "game's code (DC_FAST_CIPHER=game)"
            : g_mode == Mode::Verify ? "fast path, each call checked against the game's (DC_FAST_CIPHER_VERIFY)"
                                     : "fast path on",
            ms, ms_game, ms_fast);

    g_diag = std::getenv("DC_DIAG_CIPHER") != nullptr;
    if (g_diag) {
        std::thread([] {
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                const u64 calls = g_calls.exchange(0), bytes = g_bytes.exchange(0), ns = g_ns.exchange(0);
                if (calls) {
                    DC_INFO("gt7 cipher: %llu calls, %.1f MB, %.1f ms (verify mismatches %llu)", static_cast<unsigned long long>(calls),
                            bytes / 1048576.0, ns / 1e6,
                            static_cast<unsigned long long>(g_mismatch.load()));
                }
            }
        }).detach();
    }
}

} // namespace Core::Gt7Cipher
