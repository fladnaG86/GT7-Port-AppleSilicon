// SPDX-License-Identifier: GPL-2.0-or-later
//
// GT7 v1.00's Adhoc script loads, seen (and changed) as the game reads them.
//
//   u64 load(void* self, Stream** in)   eboot 0x2febb30: checks "ADCH" and a version
//                                       below 15, then reads the script from *in
//   Stream: vtable, read at slot +0x268 (the buffered stream's is 0x3017b70; the
//           loader and the game's byte/word readers call it directly when the
//           vtable has it, through the vtable otherwise), end flag +0x48
//           (readers fail without calling read once it is set), unread
//           pushback +0x58..+0x60
//
// While a load runs, its stream gets a copy of its vtable whose read serves a
// copy of the script: on its first call it takes the whole script through the
// stream's own read, then dumps it (DC_DUMP_SCRIPTS) and lets gt7_arcade.cpp
// change it. The stream gets its vtable back when the load returns. Version 15
// (later game versions) encrypts the body behind another stream; v1.00 reads
// version 14 in clear, so the copy is the script itself.

#include "core/gt7_scripts.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <sys/mman.h>

#include "common/log.h"
#include "core/gt7_arcade.h"

namespace Core::Gt7Scripts {

namespace {

constexpr u64 kLoad = 0x2febb30;
// push rbp; mov rbp, rsp; push r15; push r14; push r13; push r12 (12 bytes
// without RIP-relative operands, moved to the trampoline as they are), push rbx;
// sub rsp, 0xa8.
constexpr u8 kLoadHead[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54,
                            0x53, 0x48, 0x81, 0xec, 0xa8, 0x00, 0x00, 0x00};
constexpr u64 kReadSlot = 0x268 / 8;
constexpr u64 kEofFlag = 0x48;

using LoadFn = u64 (*)(u64, u64, u64, u64, u64, u64);
using ReadFn = u64 (*)(void*, u8*, u64);
LoadFn g_load = nullptr;

std::string g_dump_dir;
bool g_arcade = false;
std::mutex g_dump_lock;
int g_dump_count = 0;

u32& EofFlag(void* stream) {
    return *reinterpret_cast<u32*>(static_cast<u8*>(stream) + kEofFlag);
}

// A load's stream state: the vtable copy it reads through (two words before
// the table, as the original has: offset to top and type info) and the script.
struct Load {
    u64 head[2];
    u64 vtable[256];
    void* stream = nullptr;
    u64 original = 0; // the stream's own vtable
    ReadFn read = nullptr;
    bool taken = false;
    std::vector<u8> data;
    size_t served = 0;
    u32 eof = 0; // the end flag after the whole read
};

// The root frame's source name: the first symbol written in full (0, length,
// bytes) that looks like a path, near the start.
std::string RootName(const std::vector<u8>& d) {
    for (size_t at = 8; at + 2 < d.size() && at < 64; ++at) {
        if (d[at] == 0 && d[at + 1] > 4 && d[at + 1] < 0x80 && at + 2 + d[at + 1] <= d.size()) {
            std::string s(reinterpret_cast<const char*>(d.data() + at + 2), d[at + 1]);
            if (s.find('/') != std::string::npos || s.find(".ad") != std::string::npos) {
                return s;
            }
        }
    }
    return {};
}

void Dump(const std::vector<u8>& d) {
    std::scoped_lock lk(g_dump_lock);
    std::string name = RootName(d);
    std::replace(name.begin(), name.end(), '/', '_');
    char file[64];
    std::snprintf(file, sizeof(file), "%04d_", g_dump_count++);
    const std::filesystem::path path = std::filesystem::path(g_dump_dir) / (file + name + ".adc");
    if (FILE* f = std::fopen(path.c_str(), "wb")) {
        std::fwrite(d.data(), 1, d.size(), f);
        std::fclose(f);
    }
    DC_INFO("gt7 scripts: %s (%zu bytes)", path.c_str(), d.size());
}

// The whole script through the stream's own read, then dumped and changed.
void Take(Load& l) {
    l.taken = true;
    static constexpr u64 kStep = 4096; // small reads, as the loader makes
    for (;;) {
        const size_t have = l.data.size();
        l.data.resize(have + kStep);
        // A read can return less than asked before the end (one buffer's worth):
        // only 0 is the end.
        const u64 got = l.read(l.stream, l.data.data() + have, kStep);
        l.data.resize(have + std::min<u64>(got, kStep));
        if (got == 0 || got > kStep) {
            break;
        }
    }
    l.eof = EofFlag(l.stream);
    EofFlag(l.stream) = 0; // the copy has bytes for the readers
    // On a host thread: this one may run on a small guest stack (fibers), too
    // small for file and log calls or the patcher.
    std::thread([&l] {
        if (!g_dump_dir.empty()) {
            Dump(l.data);
        }
        if (g_arcade && Gt7Arcade::Patch(l.data) && !g_dump_dir.empty()) {
            Dump(l.data); // the changed script too
        }
    }).join();
}

// The vtable copy's read: the Load is found from the vtable the stream holds.
u64 CopyRead(void* stream, u8* to, u64 size) {
    const u64 vt = *static_cast<u64*>(stream);
    Load& l = *reinterpret_cast<Load*>(vt - offsetof(Load, vtable));
    if (!l.taken) {
        Take(l);
    }
    const u64 n = std::min<u64>(size, l.data.size() - l.served);
    if (n && to) {
        std::memcpy(to, l.data.data() + l.served, n);
    }
    l.served += n;
    if (l.served >= l.data.size()) {
        EofFlag(stream) = l.eof;
    }
    return n;
}

u64 LoadHook(u64 a, u64 b, u64 c, u64 d, u64 e, u64 f) {
    void* stream = b ? *reinterpret_cast<void**>(b) : nullptr;
    if (!stream) {
        return g_load(a, b, c, d, e, f);
    }
    auto l = std::make_unique<Load>();
    l->stream = stream;
    l->original = *static_cast<u64*>(stream);
    const auto* src = reinterpret_cast<const u64*>(l->original);
    std::memcpy(l->head, src - 2, sizeof(l->head));
    std::memcpy(l->vtable, src, sizeof(l->vtable));
    l->read = reinterpret_cast<ReadFn>(l->vtable[kReadSlot]);
    l->vtable[kReadSlot] = reinterpret_cast<u64>(&CopyRead);
    *static_cast<u64*>(stream) = reinterpret_cast<u64>(l->vtable);
    const u64 r = g_load(a, b, c, d, e, f);
    *static_cast<u64*>(stream) = l->original; // the caller still holds the stream
    return r;
}

// movabs rax, target; jmp rax over `code`; returns a trampoline running the
// 12 bytes it replaced, then jumping back.
void* Hook(u64 code, void* target) {
    u8* tramp = static_cast<u8*>(mmap(nullptr, 0x1000, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0));
    if (tramp == MAP_FAILED) {
        return nullptr;
    }
    u8 jmp[12] = {0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xe0};
    std::memcpy(tramp, reinterpret_cast<const void*>(code), 12);
    const u64 back = code + 12;
    std::memcpy(jmp + 2, &back, 8);
    std::memcpy(tramp + 12, jmp, 12);
    mprotect(tramp, 0x1000, PROT_READ | PROT_EXEC);
    const u64 page = code & ~0xFFFull;
    mprotect(reinterpret_cast<void*>(page), 0x2000, PROT_READ | PROT_WRITE | PROT_EXEC);
    const u64 to = reinterpret_cast<u64>(target);
    std::memcpy(jmp + 2, &to, 8);
    std::memcpy(reinterpret_cast<void*>(code), jmp, 12);
    mprotect(reinterpret_cast<void*>(page), 0x2000, PROT_READ | PROT_EXEC);
    return tramp;
}

} // namespace

void Install(u64 base) {
    if (const char* d = std::getenv("DC_DUMP_SCRIPTS"); d && d[0]) {
        g_dump_dir = d;
        std::error_code ec;
        std::filesystem::create_directories(g_dump_dir, ec);
    }
    const char* arcade = std::getenv("DC_ARCADE_MEMORY");
    g_arcade = !(arcade && arcade[0] == '0');
    if (!g_arcade && g_dump_dir.empty()) {
        return;
    }
    if (std::memcmp(reinterpret_cast<const void*>(base + kLoad), kLoadHead, sizeof(kLoadHead)) != 0) {
        DC_WARN("gt7 scripts: loader not as expected, left as it is");
        return;
    }
    g_load = reinterpret_cast<LoadFn>(Hook(base + kLoad, reinterpret_cast<void*>(&LoadHook)));
    DC_INFO("gt7 scripts: loads hooked%s%s", g_dump_dir.empty() ? "" : " (dumping)",
            g_arcade ? ", Arcade remembers its choices" : "");
}

} // namespace Core::Gt7Scripts
