#include "gpu/liverpool/capture.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include "common/log.h"
#include "core/memory.h"
#include "gpu/gds.h"
#include "gpu/renderer.h"

namespace Gpu::Capture {

namespace {

namespace Mem = Core::Memory;

constexpr u64 kPg = 0x4000;           // tracking granule (guest mapping granularity)
constexpr u64 kChunk = 0x10000;       // snapshot chunk; all-zero chunks are skipped
constexpr u32 kVmaMagic = 0x56434731; // "VCG1"

// kPage: a whole page. kDiff: the dwords of a page the frame changed, with
// their value before and after: the replay writes a dword only where memory
// still holds the "before" value, so results its own GPU work already wrote
// there (counters, labels sharing a page with CPU data) are kept.
enum Event : u32 { kEnd = 0, kPage = 1, kGraphics = 2, kCompute = 3, kDiff = 4 };

// ---------------------------------------------------------------- files ----
void W(FILE* f, const void* p, size_t n) {
    std::fwrite(p, 1, n, f);
}
template <typename T>
void W(FILE* f, const T& v) {
    W(f, &v, sizeof(T));
}
bool R(FILE* f, void* p, size_t n) {
    return std::fread(p, 1, n, f) == n;
}
template <typename T>
bool R(FILE* f, T& v) {
    return R(f, &v, sizeof(T));
}

FILE* Open(const std::string& dir, const char* name, const char* mode) {
    const std::string path = dir + "/" + name;
    FILE* f = std::fopen(path.c_str(), mode);
    if (!f) {
        DC_ERROR("capture: cannot open %s", path.c_str());
        return nullptr;
    }
    std::setvbuf(f, nullptr, _IOFBF, 8 << 20);
    return f;
}

bool IsZero(const u8* p, size_t n) {
    const u64* q = reinterpret_cast<const u64*>(p);
    for (size_t i = 0; i < n / 8; ++i) {
        if (q[i]) {
            return false;
        }
    }
    return true;
}

// Writes the non-zero chunks of [src, src+size) as {u64 key, u32 len, bytes};
// `key` is the address the chunk is restored at (phys or VA). Returns bytes.
u64 WriteChunks(FILE* f, const u8* src, u64 key, u64 size) {
    u64 written = 0;
    for (u64 off = 0; off < size; off += kChunk) {
        const u32 len = u32(std::min(kChunk, size - off));
        if (IsZero(src + off, len)) {
            continue;
        }
        W(f, key + off);
        W(f, len);
        W(f, src + off, len);
        written += len;
    }
    return written;
}

// Host-readable parts of [start, end).
std::vector<std::pair<u64, u64>> ReadableRanges(u64 start, u64 end) {
    std::vector<std::pair<u64, u64>> out;
    mach_vm_address_t a = start;
    while (a < end) {
        mach_vm_size_t size = 0;
        vm_region_basic_info_data_64_t info{};
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object = MACH_PORT_NULL;
        if (mach_vm_region(mach_task_self(), &a, &size, VM_REGION_BASIC_INFO_64,
                           reinterpret_cast<vm_region_info_t>(&info), &count, &object) != KERN_SUCCESS ||
            a >= end) {
            break;
        }
        const u64 s = std::max<u64>(a, start), e = std::min<u64>(a + size, end);
        if ((info.protection & VM_PROT_READ) && e > s) {
            out.emplace_back(s, e);
        }
        a += size;
    }
    return out;
}

int HostProt(int prot) {
    int host = PROT_NONE;
    if (prot & Mem::kProtCpuRead) host |= PROT_READ;
    if (prot & Mem::kProtCpuWrite) host |= PROT_READ | PROT_WRITE;
    if (prot & Mem::kProtCpuExec) host |= PROT_READ | PROT_EXEC;
    if (prot & (Mem::kProtGpuRead | Mem::kProtGpuWrite)) host |= PROT_READ | PROT_WRITE;
    return host;
}

// ------------------------------------------------------------- capture ----
struct Range {
    u64 start, end;
    int prot;    // host protection to restore
    size_t bit0; // first page's bit
};

std::mutex g_mutex;
std::atomic<bool> g_active{false}; // pages are protected: the fault handler records writes
std::atomic<bool> g_armed{false};  // starting or recording: submits take the lock
std::atomic<u64> g_flips{0};
enum class State { Idle, Recording, Done } g_state = State::Idle;
std::vector<Range> g_ranges; // sorted by start, fixed while recording
std::unique_ptr<std::atomic<u64>[]> g_bits;
size_t g_words = 0;
// Shadow copies: a page's content when the CPU first wrote it (then as of the
// last flush), so a flush records only the dwords that changed.
constexpr u64 kShadowBytes = 4ull << 30; // virtual; only touched slots are backed
constexpr u32 kSlotBusy = 0xFFFFFFFFu;
u8* g_shadow = nullptr;
std::unique_ptr<std::atomic<u32>[]> g_slot; // per page bit: slot + 1, 0 = none
std::atomic<u32> g_next_slot{0};
u64 g_diff_dwords = 0;
std::string g_dir;
FILE* g_events = nullptr;
FlipInfo g_flip{};
u64 g_pages = 0, g_submits = 0, g_faults = 0;
std::chrono::steady_clock::time_point g_t0;

const char* Dir() {
    static const char* d = std::getenv("DC_CAPTURE");
    return d;
}
u64 TargetFlip() {
    static const u64 n =
        std::getenv("DC_CAPTURE_FLIP") ? std::strtoull(std::getenv("DC_CAPTURE_FLIP"), nullptr, 0) : 0;
    return n;
}

// DC_CAPTURE_FRAMES=<k>: frames recorded (default 2: GT7 renders a frame's
// scene during the previous one, so the first replayed frame lacks it).
u64 Frames() {
    static const u64 n = std::getenv("DC_CAPTURE_FRAMES") ? std::max<u64>(1, std::strtoull(std::getenv("DC_CAPTURE_FRAMES"), nullptr, 0)) : 2;
    return n;
}

const Range* FindRange(u64 a) {
    auto it = std::upper_bound(g_ranges.begin(), g_ranges.end(), a,
                               [](u64 v, const Range& r) { return v < r.start; });
    if (it == g_ranges.begin()) {
        return nullptr;
    }
    --it;
    return a < it->end ? &*it : nullptr;
}

// Called with the page still write-protected: copies it before the first write.
void EnsureShadow(size_t bit, u64 page) {
    std::atomic<u32>& slot = g_slot[bit];
    u32 v = slot.load(std::memory_order_acquire);
    if (v == 0) {
        u32 expected = 0;
        if (slot.compare_exchange_strong(expected, kSlotBusy, std::memory_order_acq_rel)) {
            const u32 n = g_next_slot.fetch_add(1, std::memory_order_relaxed);
            if (u64(n + 1) * kPg <= kShadowBytes) {
                std::memcpy(g_shadow + u64(n) * kPg, reinterpret_cast<const void*>(page), kPg);
                slot.store(n + 1, std::memory_order_release);
            } else {
                slot.store(kSlotBusy - 1, std::memory_order_release); // arena full: whole pages
            }
            return;
        }
        v = expected;
    }
    while (slot.load(std::memory_order_acquire) == kSlotBusy) {
        // another thread is copying the page: it must not change under the copy
    }
}

void MarkDirty(const Range& r, u64 page) {
    const size_t bit = r.bit0 + (page - r.start) / kPg;
    EnsureShadow(bit, page);
    g_bits[bit / 64].fetch_or(1ull << (bit % 64), std::memory_order_relaxed);
}

// Records every page written since the last flush and protects it again.
void Flush() {
    for (size_t w = 0; w < g_words; ++w) {
        u64 x = g_bits[w].exchange(0, std::memory_order_acq_rel);
        while (x) {
            const size_t bit = w * 64 + size_t(__builtin_ctzll(x));
            x &= x - 1;
            auto it = std::upper_bound(g_ranges.begin(), g_ranges.end(), bit,
                                       [](size_t v, const Range& r) { return v < r.bit0; });
            const Range& r = *std::prev(it);
            const u64 page = r.start + (bit - r.bit0) * kPg;
            // Protect before copying: a write after this faults and is recorded next time.
            mprotect(reinterpret_cast<void*>(page), kPg, r.prot & ~PROT_WRITE);
            ++g_pages;
            const u32 slot = g_slot[bit].load(std::memory_order_acquire);
            if (slot == 0 || slot >= kSlotBusy - 1) {
                W(g_events, u32(kPage));
                W(g_events, page);
                W(g_events, u32(kPg));
                W(g_events, reinterpret_cast<const void*>(page), kPg);
                continue;
            }
            u32* shadow = reinterpret_cast<u32*>(g_shadow + u64(slot - 1) * kPg);
            const u32* cur = reinterpret_cast<const u32*>(page);
            // Runs of changed dwords: {u16 first, u16 count, old[count], new[count]}.
            static thread_local std::vector<u32> runs;
            runs.clear();
            u32 nruns = 0;
            constexpr u32 kDw = kPg / 4;
            for (u32 i = 0; i < kDw;) {
                if (cur[i] == shadow[i]) {
                    ++i;
                    continue;
                }
                u32 j = i;
                while (j < kDw && cur[j] != shadow[j]) {
                    ++j;
                }
                runs.push_back(i | ((j - i) << 16));
                runs.insert(runs.end(), shadow + i, shadow + j);
                runs.insert(runs.end(), cur + i, cur + j);
                std::memcpy(shadow + i, cur + i, (j - i) * 4);
                g_diff_dwords += j - i;
                ++nruns;
                i = j;
            }
            if (!nruns) {
                continue;
            }
            W(g_events, u32(kDiff));
            W(g_events, page);
            W(g_events, nruns);
            W(g_events, runs.data(), runs.size() * 4);
        }
    }
}

void Start() {
    auto& cp = Liverpool::Instance();
    Renderer* renderer = cp.GetRenderer();
    const auto t0 = std::chrono::steady_clock::now();
    DC_INFO("capture: frame %llu starts: waiting for the GPU", (unsigned long long)TargetFlip());
    if (!cp.WaitIdleFor(5000)) {
        DC_WARN("capture: the command processor did not go idle in 5 s; capturing anyway");
    }
    if (renderer) {
        renderer->WaitIdle();
    }
    g_dir = Dir();
    std::error_code ec;
    std::filesystem::create_directories(g_dir, ec);

    const std::vector<Mem::Vma> vmas = Mem::AllVmas();
    if (FILE* f = Open(g_dir, "vmas.bin", "wb")) {
        W(f, kVmaMagic);
        W(f, u32(vmas.size()));
        for (const Mem::Vma& v : vmas) {
            W(f, v.start);
            W(f, v.size);
            W(f, s32(v.prot));
            W(f, u32(v.type));
            W(f, v.phys);
            W(f, u32(v.name.size()));
            W(f, v.name.data(), v.name.size());
        }
        std::fclose(f);
    }
    u64 direct_bytes = 0, va_bytes = 0;
    if (FILE* f = Open(g_dir, "direct.bin", "wb")) {
        const u8* view = static_cast<const u8*>(Mem::DirectHostView());
        for (const auto& [phys, size] : Mem::DirectAllocations()) {
            direct_bytes += WriteChunks(f, view + phys, phys, size);
        }
        std::fclose(f);
    }
    if (FILE* f = Open(g_dir, "va.bin", "wb")) {
        for (const Mem::Vma& v : vmas) {
            if (v.type == Mem::VmaType::Direct) {
                continue; // in direct.bin, by physical address
            }
            for (const auto& [s, e] : ReadableRanges(v.start, v.start + v.size)) {
                va_bytes += WriteChunks(f, reinterpret_cast<const u8*>(s), s, e - s);
            }
        }
        std::fclose(f);
    }
    if (renderer) {
        renderer->SaveTargets((g_dir + "/targets.bin").c_str());
    }
    if (FILE* f = Open(g_dir, "cp.bin", "wb")) {
        const std::vector<u8> state = cp.SaveState();
        W(f, state.data(), state.size());
        std::fclose(f);
    }
    if (renderer) {
        if (FILE* f = Open(g_dir, "gds.bin", "wb")) {
            W(f, renderer->Gds(), size_t(kGdsPartition) * kGdsPartitions);
            std::fclose(f);
        }
    }
    g_events = Open(g_dir, "events.bin", "wb");
    if (!g_events) {
        g_state = State::Done;
        return;
    }

    // Track GPU-visible memory: the GPU can only read what the CPU wrote there.
    g_ranges.clear();
    size_t bits = 0;
    for (const Mem::Vma& v : vmas) {
        const bool gpu = v.type == Mem::VmaType::Direct ||
                         (v.prot & (Mem::kProtGpuRead | Mem::kProtGpuWrite));
        const int host = HostProt(v.prot);
        if (!gpu || !(host & PROT_WRITE) || v.type == Mem::VmaType::Reserved) {
            continue;
        }
        g_ranges.push_back({v.start, v.start + v.size, host, bits});
        bits += v.size / kPg;
    }
    g_words = (bits + 63) / 64;
    g_bits.reset(new std::atomic<u64>[g_words]);
    for (size_t w = 0; w < g_words; ++w) {
        g_bits[w].store(0, std::memory_order_relaxed);
    }
    g_slot.reset(new std::atomic<u32>[bits]);
    for (size_t b = 0; b < bits; ++b) {
        g_slot[b].store(0, std::memory_order_relaxed);
    }
    if (!g_shadow) {
        void* p = mmap(nullptr, kShadowBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        g_shadow = p == MAP_FAILED ? nullptr : static_cast<u8*>(p);
    }
    if (!g_shadow) {
        g_next_slot.store(u32(kShadowBytes / kPg), std::memory_order_relaxed); // whole pages only
    }
    g_pages = g_submits = g_faults = 0;
    g_active.store(true, std::memory_order_release);
    u32 protect_failed = 0;
    for (const Range& r : g_ranges) {
        protect_failed += mprotect(reinterpret_cast<void*>(r.start), r.end - r.start, r.prot & ~PROT_WRITE) != 0;
    }
    if (protect_failed) {
        DC_WARN("capture: %u ranges could not be protected", protect_failed);
    }
    g_state = State::Recording;
    g_t0 = std::chrono::steady_clock::now();
    DC_INFO("capture: snapshot in %s: %zu mappings, direct %llu MB, other %llu MB, %zu tracked ranges "
            "(%llu MB), %.1f s",
            g_dir.c_str(), vmas.size(), (unsigned long long)(direct_bytes >> 20),
            (unsigned long long)(va_bytes >> 20), g_ranges.size(), (unsigned long long)(bits * kPg >> 20),
            std::chrono::duration<double>(g_t0 - t0).count());
}

void Finish() {
    W(g_events, u32(kEnd));
    std::fclose(g_events);
    g_events = nullptr;
    // Unprotect first: a write in between still finds the handler active.
    for (const Range& r : g_ranges) {
        mprotect(reinterpret_cast<void*>(r.start), r.end - r.start, r.prot);
    }
    g_active.store(false, std::memory_order_release);
    g_armed.store(false, std::memory_order_release);
    g_state = State::Done;
    DC_INFO("capture: done in %.2f s: %llu submits, %llu page flushes, %llu MB of changed dwords, "
            "%llu faults. Replay: DC_FRAME_REPLAY=%s",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - g_t0).count(),
            (unsigned long long)g_submits, (unsigned long long)g_pages,
            (unsigned long long)(g_diff_dwords * 4 >> 20), (unsigned long long)g_faults, g_dir.c_str());
    if (std::getenv("DC_CAPTURE_EXIT")) {
        std::fflush(nullptr);
        std::_Exit(0);
    }
}

} // namespace

void NoteFlip(const FlipInfo& flip) {
    if (!Dir()) {
        return;
    }
    std::scoped_lock lk(g_mutex);
    g_flip = flip;
}

void BeforeGraphics(const std::vector<Buffers>& bufs, bool flip) {
    if (!g_armed.load(std::memory_order_acquire)) {
        return;
    }
    std::scoped_lock lk(g_mutex);
    if (g_state != State::Recording) {
        return;
    }
    Flush();
    W(g_events, u32(kGraphics));
    W(g_events, u32(bufs.size()));
    for (const Buffers& b : bufs) {
        W(g_events, u64(reinterpret_cast<uintptr_t>(b.dcb.data())));
        W(g_events, u32(b.dcb.size()));
        W(g_events, u64(reinterpret_cast<uintptr_t>(b.ccb.data())));
        W(g_events, u32(b.ccb.size()));
    }
    W(g_events, u8(flip));
    if (flip) {
        W(g_events, g_flip);
    }
    ++g_submits;
}

void AfterGraphics(bool flip) {
    if (!flip || !Dir() || TargetFlip() < 2) {
        return;
    }
    const u64 n = g_flips.fetch_add(1) + 1;
    std::scoped_lock lk(g_mutex);
    if (g_state == State::Idle && n == TargetFlip() - 1) {
        g_armed.store(true, std::memory_order_release); // other submitters now wait for the snapshot
        Start();
    } else if (g_state == State::Recording && n >= TargetFlip() - 1 + Frames()) {
        Finish();
    }
}

void BeforeCompute(u32 queue, std::span<const u32> acb) {
    if (!g_armed.load(std::memory_order_acquire)) {
        return;
    }
    std::scoped_lock lk(g_mutex);
    if (g_state != State::Recording) {
        return;
    }
    Flush();
    W(g_events, u32(kCompute));
    W(g_events, queue);
    W(g_events, u64(reinterpret_cast<uintptr_t>(acb.data())));
    W(g_events, u32(acb.size()));
    ++g_submits;
}

bool HandleWriteFault(void* addr) {
    if (!g_active.load(std::memory_order_acquire)) {
        return false;
    }
    const u64 a = reinterpret_cast<u64>(addr);
    const Range* r = FindRange(a);
    if (!r) {
        char msg[128];
        const int n = std::snprintf(msg, sizeof(msg), "capture: fault at %#llx outside the %zu tracked ranges\n",
                                    (unsigned long long)a, g_ranges.size());
        write(2, msg, size_t(n));
        return false;
    }
    const u64 page = a & ~(kPg - 1);
    MarkDirty(*r, page);
    if (mprotect(reinterpret_cast<void*>(page), kPg, r->prot) != 0) {
        char msg[128];
        const int n = std::snprintf(msg, sizeof(msg), "capture: mprotect %#llx failed: %d\n",
                                    (unsigned long long)page, errno);
        write(2, msg, size_t(n));
    }
    ++g_faults; // approximate (not atomic): statistics only
    return true;
}

void BeforeHostWrite(void* addr, u64 size) {
    if (!g_active.load(std::memory_order_acquire) || !size) {
        return;
    }
    const u64 a = reinterpret_cast<u64>(addr);
    for (u64 page = a & ~(kPg - 1); page < a + size; page += kPg) {
        if (const Range* r = FindRange(page)) {
            MarkDirty(*r, page);
            mprotect(reinterpret_cast<void*>(page), kPg, r->prot);
        }
    }
}

// --------------------------------------------------------------- replay ----
namespace {

struct ChunkRef {
    u64 key;
    u32 len;
    long offset; // of the bytes in the file
};
std::vector<ChunkRef> g_direct_chunks, g_va_chunks;

std::vector<ChunkRef> IndexChunks(const std::string& dir, const char* name) {
    std::vector<ChunkRef> out;
    FILE* f = Open(dir, name, "rb");
    if (!f) {
        return out;
    }
    for (;;) {
        ChunkRef c{};
        if (!R(f, c.key) || !R(f, c.len)) {
            break;
        }
        c.offset = std::ftell(f);
        out.push_back(c);
        std::fseek(f, c.len, SEEK_CUR);
    }
    std::fclose(f);
    return out;
}

void LoadChunks(const std::string& dir, const char* name, const std::vector<ChunkRef>& chunks, u8* base) {
    const std::string path = dir + "/" + name;
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        DC_ERROR("capture: cannot open %s", path.c_str());
        return;
    }
    for (const ChunkRef& c : chunks) {
        pread(fd, base + c.key, c.len, c.offset);
    }
    close(fd);
}

void LoadContents(const std::string& dir) {
    LoadChunks(dir, "direct.bin", g_direct_chunks, static_cast<u8*>(Mem::DirectHostView()));
    LoadChunks(dir, "va.bin", g_va_chunks, nullptr);
}

std::vector<u8> ReadAll(const std::string& dir, const char* name) {
    std::vector<u8> out;
    FILE* f = Open(dir, name, "rb");
    if (!f) {
        return out;
    }
    std::fseek(f, 0, SEEK_END);
    out.resize(size_t(std::ftell(f)));
    std::fseek(f, 0, SEEK_SET);
    R(f, out.data(), out.size());
    std::fclose(f);
    return out;
}

} // namespace

bool LoadMemory(const char* dir_c) {
    const std::string dir = dir_c;
    FILE* f = Open(dir, "vmas.bin", "rb");
    if (!f) {
        return false;
    }
    u32 magic = 0, count = 0;
    if (!R(f, magic) || magic != kVmaMagic || !R(f, count)) {
        DC_ERROR("capture: %s/vmas.bin is not a capture", dir_c);
        std::fclose(f);
        return false;
    }
    std::vector<Mem::Vma> vmas(count);
    for (Mem::Vma& v : vmas) {
        s32 prot = 0;
        u32 type = 0, name_len = 0;
        R(f, v.start);
        R(f, v.size);
        R(f, prot);
        R(f, type);
        R(f, v.phys);
        R(f, name_len);
        v.name.resize(name_len);
        R(f, v.name.data(), name_len);
        v.prot = prot;
        v.type = static_cast<Mem::VmaType>(type);
    }
    std::fclose(f);
    g_direct_chunks = IndexChunks(dir, "direct.bin");
    g_va_chunks = IndexChunks(dir, "va.bin");

    u32 mapped = 0, failed = 0;
    for (const Mem::Vma& v : vmas) {
        VAddr a = v.start;
        int err = 0;
        if (v.type == Mem::VmaType::Direct) {
            Mem::AllocateDirectAt(v.phys, v.size);
            err = Mem::MapDirect(&a, v.size, v.prot, Mem::kMapFixed, v.phys, 0, v.name.c_str());
        } else if (v.type == Mem::VmaType::Flexible) {
            err = Mem::MapFlexible(&a, v.size, v.prot, Mem::kMapFixed, v.name.c_str());
        } else {
            // Reserved: module images and reservations. Back only those with data.
            const bool data = std::any_of(g_va_chunks.begin(), g_va_chunks.end(), [&](const ChunkRef& c) {
                return c.key >= v.start && c.key < v.start + v.size;
            });
            if (!data) {
                continue;
            }
            void* p = mmap(reinterpret_cast<void*>(v.start), v.size, PROT_READ | PROT_WRITE,
                           MAP_FIXED | MAP_PRIVATE | MAP_ANON, -1, 0);
            err = p == MAP_FAILED ? errno : 0;
        }
        if (err) {
            ++failed;
            DC_WARN("capture: cannot map %#llx +%#llx (type %u): %d", (unsigned long long)v.start,
                    (unsigned long long)v.size, u32(v.type), err);
            continue;
        }
        // Loading writes everything: writable whatever the game asked for.
        mprotect(reinterpret_cast<void*>(v.start), v.size, PROT_READ | PROT_WRITE);
        ++mapped;
    }
    const auto t0 = std::chrono::steady_clock::now();
    LoadContents(dir);
    u64 bytes = 0;
    for (const auto& c : g_direct_chunks) bytes += c.len;
    for (const auto& c : g_va_chunks) bytes += c.len;
    DC_INFO("capture: replay of %s: %u mappings (%u failed), %llu MB loaded in %.1f s", dir_c, mapped, failed,
            (unsigned long long)(bytes >> 20),
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

void Replay(const char* dir_c, Renderer* renderer) {
    const std::string dir = dir_c;
    auto& cp = Liverpool::Instance();
    const std::vector<u8> cp_state = ReadAll(dir, "cp.bin");
    const std::vector<u8> gds = ReadAll(dir, "gds.bin");
    const std::vector<u8> ev = ReadAll(dir, "events.bin");
    const int loops = std::getenv("DC_REPLAY_LOOPS") ? std::atoi(std::getenv("DC_REPLAY_LOOPS")) : 4;
    for (int loop = 0; loop < loops; ++loop) {
        const auto t0 = std::chrono::steady_clock::now();
        if (loop > 0) {
            LoadContents(dir);
        }
        cp.LoadState(cp_state);
        if (renderer) {
            renderer->LoadTargets((dir + "/targets.bin").c_str());
        }
        if (renderer && gds.size() == size_t(kGdsPartition) * kGdsPartitions) {
            std::memcpy(renderer->Gds(), gds.data(), gds.size());
        }
        size_t at = 0;
        auto get = [&](void* p, size_t n) {
            if (at + n <= ev.size()) {
                std::memcpy(p, ev.data() + at, n);
            }
            at += n;
        };
        bool idle = true;
        u32 submits = 0, pages = 0;
        u64 kept = 0; // dwords the replay's GPU had already changed
        while (at < ev.size()) {
            u32 type = kEnd;
            get(&type, 4);
            if (type == kEnd) {
                break;
            }
            if (type == kPage) {
                u64 va = 0;
                u32 len = 0;
                get(&va, 8);
                get(&len, 4);
                // The CPU wrote these while the GPU ran the earlier submits, as here.
                // DC_REPLAY_SYNC=1: wait for those submits first (they then never see
                // the writes; a GPU wait for a CPU-written label times out).
                static const bool sync = std::getenv("DC_REPLAY_SYNC") != nullptr;
                if (sync && !idle) {
                    cp.WaitIdleFor(1000);
                    if (renderer) {
                        renderer->WaitIdle();
                    }
                    idle = true;
                }
                if (at + len <= ev.size()) {
                    std::memcpy(reinterpret_cast<void*>(va), ev.data() + at, len);
                }
                at += len;
                ++pages;
            } else if (type == kDiff) {
                u64 va = 0;
                u32 nruns = 0;
                get(&va, 8);
                get(&nruns, 4);
                static const bool overwrite = std::getenv("DC_REPLAY_OVERWRITE") != nullptr;
                u32* mem = reinterpret_cast<u32*>(va);
                for (u32 k = 0; k < nruns && at < ev.size(); ++k) {
                    u32 head = 0;
                    get(&head, 4);
                    const u32 first = head & 0xFFFF, count = head >> 16;
                    const u32* old_v = reinterpret_cast<const u32*>(ev.data() + at);
                    const u32* new_v = old_v + count;
                    for (u32 i = 0; i < count; ++i) {
                        if (overwrite || mem[first + i] == old_v[i]) {
                            mem[first + i] = new_v[i];
                        } else {
                            ++kept;
                        }
                    }
                    at += size_t(count) * 8;
                }
                ++pages;
            } else if (type == kGraphics) {
                u32 n = 0;
                get(&n, 4);
                std::vector<Buffers> bufs(n);
                for (Buffers& b : bufs) {
                    u64 dcb = 0, ccb = 0;
                    u32 dcb_dw = 0, ccb_dw = 0;
                    get(&dcb, 8);
                    get(&dcb_dw, 4);
                    get(&ccb, 8);
                    get(&ccb_dw, 4);
                    b.dcb = {reinterpret_cast<const u32*>(dcb), dcb_dw};
                    b.ccb = {reinterpret_cast<const u32*>(ccb), ccb_dw};
                }
                u8 flip = 0;
                get(&flip, 1);
                std::function<void()> after;
                if (flip) {
                    FlipInfo info{};
                    get(&info, sizeof(info));
                    if (info.address && renderer) {
                        after = [renderer, info, &cp] {
                            static u64 last_draws = 0, last_disp = 0;
                            const u64 d = cp.GetStats().draws.load(), c = cp.GetStats().dispatches.load();
                            DC_INFO("capture: replay flip: %llu draws, %llu dispatches", (unsigned long long)(d - last_draws),
                                    (unsigned long long)(c - last_disp));
                            last_draws = d;
                            last_disp = c;
                            renderer->Flip(info);
                        };
                    }
                }
                cp.SubmitGraphics(std::move(bufs), std::move(after));
                idle = false;
                ++submits;
            } else if (type == kCompute) {
                u32 queue = 0, dw = 0;
                u64 addr = 0;
                get(&queue, 4);
                get(&addr, 8);
                get(&dw, 4);
                cp.SubmitCompute(queue, {reinterpret_cast<const u32*>(addr), dw});
                idle = false;
                ++submits;
            } else {
                DC_ERROR("capture: events.bin: unknown event %u at %zu", type, at - 4);
                break;
            }
        }
        const bool done = cp.WaitIdleFor(10000);
        if (renderer) {
            renderer->WaitIdle();
        }
        DC_INFO("capture: replay loop %d: %u submits, %u pages (%llu dwords kept), %.2f s%s", loop + 1, submits,
                pages, (unsigned long long)kept,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                done ? "" : " (GPU still busy after 10 s)");
    }
}

} // namespace Gpu::Capture
