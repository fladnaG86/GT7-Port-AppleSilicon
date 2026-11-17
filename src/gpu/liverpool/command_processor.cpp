#include "core/guest_profile.h"
#include "gpu/liverpool/command_processor.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <x86intrin.h>

#include "common/log.h"
#include "core/memory.h"
#include "gpu/label_trace.h"
#include "gpu/liverpool/capture.h"
#include "gpu/renderer.h"

namespace Gpu::Liverpool {

namespace {
// DMA_DATA copies of group counts into a DISPATCH_DIRECT of the command buffer
// being parsed: packet body address -> source of the counts.
thread_local std::unordered_map<u64, u64> t_dispatch_args;
} // namespace

namespace {

// EVENT_WRITE_EOP / RELEASE_MEM data and interrupt selects.
enum DataSel : u32 { kDataNone = 0, kData32 = 1, kData64 = 2, kGpuClock = 3, kPerfCounter = 4 };
// NOP payload markers the Gnm driver uses for flip requests.
constexpr u32 kNopPrepareFlipLabel = 0x68750778;
constexpr u32 kNopPrepareFlipInterrupt = 0x68750780;
constexpr u32 kNopPrepareFlipInterruptLabel = 0x68750781;

enum IntSel : u32 { kIntNone = 0, kIntOnly = 1, kIntWriteConfirm = 2, kIntUndocumented = 3 };

u64 GpuClock() {
    // The GPU clock runs at 100 MHz on PS4; derive it from the host clock.
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 10;
}

// True if [addr, addr + size) is guest memory (one mapping).
bool Mapped(const void* p, u64 size) {
    const u64 addr = reinterpret_cast<u64>(p);
    Core::Memory::Vma vma;
    return addr && Core::Memory::Query(addr, false, &vma) && vma.type != Core::Memory::VmaType::Free &&
           addr + size <= vma.start + vma.size;
}

template <typename T>
T* Ptr(u32 lo, u32 hi) {
    return reinterpret_cast<T*>(u64(lo) | (u64(hi) << 32));
}

void WriteFence(u32 data_sel, u32 lo, u32 hi, u32 data_lo, u32 data_hi) {
    switch (data_sel) {
    case kData32:
        *Ptr<u32>(lo, hi) = data_lo;
        break;
    case kData64:
        *Ptr<u64>(lo, hi) = u64(data_lo) | (u64(data_hi) << 32);
        break;
    case kGpuClock:
        *Ptr<u64>(lo, hi) = GpuClock();
        break;
    case kPerfCounter:
        *Ptr<u64>(lo, hi) = __rdtsc();
        break;
    default:
        break;
    }
}

bool Compare(u32 func, u32 value, u32 ref) {
    switch (func) {
    case 0: return true;
    case 1: return value < ref;
    case 2: return value <= ref;
    case 3: return value == ref;
    case 4: return value != ref;
    case 5: return value >= ref;
    case 6: return value > ref;
    default: return true;
    }
}

} // namespace

namespace {
// DC_LABEL_HIST=1 (debug): keeps the last 1M CP events in memory. On a
// WaitRegMem timeout it dumps the submits/jobs and the events near the label
// (then polls the label for 60 s); on a chain/IB to nowhere it dumps the
// command buffer and the history of the chain and of the ring's last wait on
// CPU memory. DC_POLL_CPU_WAITS=1 also polls those CPU flags (~10 us),
// DC_HIST_DUMP_SUBMIT=<n> dumps such a history once past gfx submit n.
// Kinds: R fence recorded, W fence written, D WRITE_DATA, F/C CP DMA fill/copy,
// f/c host GPU DMA fill/copy, S wait start, T wait timeout, P/p wait passed on a
// pending fence, Q gfx wait on CPU memory (value seen), V polled value change,
// G gfx submit (dwords; ring 256 = flip), K compute submit, J/X ring job
// start/end (packets so far), I/i (chained) indirect buffer.
struct LabelEvent {
    u64 t;
    u64 addr;
    u32 value;
    u32 ring;
    char kind;
};
struct LabelHist {
    std::mutex lock;
    std::vector<LabelEvent> ev = std::vector<LabelEvent>(1u << 20);
    u64 n = 0;
};
bool HistOn() {
    static const bool on = std::getenv("DC_LABEL_HIST") != nullptr;
    return on;
}
LabelHist& Hist() {
    static LabelHist h;
    return h;
}
thread_local u64 t_hist_packets = 0;
thread_local u64 t_last_q = 0;
u64 HistNow() {
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}
void HistAdd(char kind, u64 addr, u32 value, u32 ring) {
    if (!HistOn()) {
        return;
    }
    LabelHist& h = Hist();
    std::scoped_lock lk(h.lock);
    h.ev[h.n++ & (h.ev.size() - 1)] = {HistNow(), addr, value, ring, kind};
}
// DC_POLL_CPU_WAITS=1 (debug): a thread polls (every ~10 us) each CPU-memory
// address the graphics ring has waited on, and logs every change ('V').
std::atomic<u64> g_poll_addrs[16];
void HistPoll(u64 a) {
    static const bool on = std::getenv("DC_POLL_CPU_WAITS") != nullptr;
    if (!on || !HistOn()) {
        return;
    }
    static std::once_flag once;
    std::call_once(once, [] {
        std::thread([] {
            u32 last[16] = {};
            u64 seen[16] = {};
            for (;;) {
                for (int k = 0; k < 16; ++k) {
                    const u64 a = g_poll_addrs[k].load(std::memory_order_relaxed);
                    if (!a) {
                        break;
                    }
                    const u32 v = *reinterpret_cast<const volatile u32*>(a);
                    if (seen[k] != a || v != last[k]) {
                        seen[k] = a;
                        last[k] = v;
                        HistAdd('V', a, v, 0);
                    }
                }
                usleep(10);
            }
        }).detach();
    });
    for (int k = 0; k < 16; ++k) {
        u64 cur = g_poll_addrs[k].load();
        if (cur == a) {
            return;
        }
        if (!cur && g_poll_addrs[k].compare_exchange_strong(cur, a)) {
            return;
        }
        if (cur == a) {
            return;
        }
    }
}
// Every history event at exactly `addr` (last `max`).
void HistDumpAddr(u64 addr, size_t max, const char* why) {
    LabelHist& h = Hist();
    std::vector<LabelEvent> out;
    {
        std::scoped_lock lk(h.lock);
        const u64 cnt = std::min<u64>(h.n, h.ev.size());
        for (u64 k = h.n - cnt; k < h.n; ++k) {
            const LabelEvent& e = h.ev[k & (h.ev.size() - 1)];
            if (((e.kind == 'J' || e.kind == 'X') && e.ring == 0) || e.kind == 'Q' || e.kind == 'V' || (e.addr <= addr && addr < e.addr + ((e.kind == 'c' || e.kind == 'C' || e.kind == 'f' || e.kind == 'F') ? std::max<u64>(e.value, 4) : 4)) ||
                e.kind == 'G') {
                out.push_back(e);
            }
        }
    }
    if (out.size() > max) {
        out.erase(out.begin(), out.end() - max);
    }
    const u64 now = HistNow();
    DC_WARN("gpu: history of %#llx (%s), with gfx submits:", (unsigned long long)addr, why);
    for (const auto& e : out) {
        DC_WARN("gpu:   %c %#llx = %#x ring %u, %.3f ms ago", e.kind, (unsigned long long)e.addr, e.value, e.ring,
                (now - e.t) / 1e6);
    }
}
void HistDump(u64 addr, u32 ring) {
    if (!HistOn()) {
        return;
    }
    LabelHist& h = Hist();
    std::vector<LabelEvent> out, subs, tail;
    {
        std::scoped_lock lk(h.lock);
        const u64 cnt = std::min<u64>(h.n, h.ev.size());
        for (u64 k = h.n - cnt; k < h.n; ++k) {
            const LabelEvent& e = h.ev[k & (h.ev.size() - 1)];
            if (e.addr + 0x100 >= addr && e.addr <= addr + 0x100) {
                out.push_back(e);
            } else if (e.kind == 'G' || e.kind == 'I' || e.kind == 'i' || e.kind == 'Q' || e.kind == 'V' || ((e.kind == 'K' || e.kind == 'J' || e.kind == 'X') && e.ring < 3)) {
                subs.push_back(e);
            }
            if (k + 40 >= h.n) {
                tail.push_back(e);
            }
        }
    }
    if (subs.size() > 300) {
        subs.erase(subs.begin(), subs.end() - 300);
    }
    const u64 now0 = HistNow();
    DC_WARN("gpu: last submits/jobs (G gfx submit: dwords, ring 256 = flip; K compute: dwords; J/X job start/end: "
            "packets so far):");
    for (const auto& e : subs) {
        DC_WARN("gpu:   %c %#llx = %#x ring %u, %.3f s ago", e.kind, (unsigned long long)e.addr, e.value, e.ring,
                (now0 - e.t) / 1e9);
    }
    const u64 now = HistNow();
    DC_WARN("gpu: label hist for %#llx (ring %u): %zu events near it", (unsigned long long)addr, ring, out.size());
    if (out.size() > 400) {
        out.erase(out.begin(), out.end() - 400);
    }
    for (const auto& e : out) {
        DC_WARN("gpu:   %c %#llx = %#x ring %u, %.3f s ago", e.kind, (unsigned long long)e.addr, e.value, e.ring,
                (now - e.t) / 1e9);
    }
    DC_WARN("gpu: last label events:");
    for (const auto& e : tail) {
        DC_WARN("gpu:   %c %#llx = %#x ring %u, %.3f s ago", e.kind, (unsigned long long)e.addr, e.value, e.ring,
                (now - e.t) / 1e9);
    }
    std::thread([addr] {
        const volatile u32* p = reinterpret_cast<const volatile u32*>(addr);
        u32 last = *p;
        const auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(60)) {
            if (*p != last) {
                DC_WARN("gpu: label %#llx changed %#x -> %#x %.3f s after its timeout", (unsigned long long)addr,
                        last, *p,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                last = *p;
            }
            usleep(200);
        }
    }).detach();
}
} // namespace

__attribute__((noinline)) CommandProcessor::Ring*& CommandProcessor::TRing() {
    static thread_local Ring* ring = nullptr;
    return ring;
}
__attribute__((noinline)) RegFile*& CommandProcessor::TRegs() {
    static thread_local RegFile* r = nullptr;
    return r;
}
#define t_ring TRing()
#define t_regs TRegs()

namespace {
// Pipe record kinds (low byte of a record's first dword).
enum : u32 { kPipeRegs = 1, kPipeDraw, kPipeDispatch, kPipeCtx, kPipeMem, kPipeFn };
constexpr u32 kPipeBatchDraws = 32;   // draws per batch handed to the worker
constexpr size_t kPipeMaxQueued = 8;  // batches in flight before the ring thread waits
constexpr u32 kDrawCallWords = (sizeof(DrawCall) + 3) / 4;
} // namespace


CommandProcessor::CommandProcessor() {
    trace = std::getenv("DC_CP_TRACE") != nullptr;
    ClearState();
    if (const char* e = std::getenv("DC_CP_PIPE"); e && std::atoi(e) != 0) {
        pipe = std::make_unique<Pipe>();
        pipe->regs = regs;
        pipe->thread = std::jthread([this](std::stop_token st) { PipeWorker(st); });
        DC_INFO("gpu: graphics CP pipelined (parse and render on two threads)");
    }
}

// CLEAR_STATE: context registers back to their power-on values. Most are 0;
// these are the ones whose default matters (from AMD's CIK clear-state table).
void CommandProcessor::ClearState() {
    std::memset(regs.Data() + kContextRegBase, 0, 0x400 * 4);
    regs[Reg::PA_SC_SCREEN_SCISSOR_BR] = 0x40004000;
    regs[Reg::PA_SC_WINDOW_SCISSOR_TL] = 0x80000000;
    regs[Reg::PA_SC_WINDOW_SCISSOR_BR] = 0x40004000;
    regs[0xA083] = 0xFFFF; // PA_SC_CLIPRECT_RULE
    for (u32 k = 0; k < 4; ++k) {
        regs[0xA085 + 2 * k] = 0x20002000; // PA_SC_CLIPRECT_n_BR
    }
    regs[0xA08C] = 0xAA99AAAA; // PA_SC_EDGERULE
    regs[Reg::CB_TARGET_MASK] = 0xFFFFFFFF;
    regs[0xA08F] = 0xFFFFFFFF; // CB_SHADER_MASK
    regs[Reg::PA_SC_GENERIC_SCISSOR_TL] = 0x80000000;
    regs[Reg::PA_SC_GENERIC_SCISSOR_BR] = 0x40004000;
    for (u32 k = 0; k < 16; ++k) {
        regs[Reg::PA_SC_VPORT_SCISSOR_0_TL + 2 * k] = 0x80000000;
        regs[Reg::PA_SC_VPORT_SCISSOR_0_TL + 2 * k + 1] = 0x40004000;
        regs[Reg::PA_SC_VPORT_ZMIN_0 + 2 * k + 1] = 0x3F800000; // ZMAX = 1.0
    }
    if (Piped()) {
        PipeRegs(kContextRegBase, regs.Data() + kContextRegBase, 0x400);
    }
}

CommandProcessor::~CommandProcessor() {
    if (pipe) {
        pipe->thread.request_stop();
        pipe->cv.notify_all();
    }
    for (auto& r : rings) {
        if (r) {
            r->thread.request_stop();
            r->cv.notify_all();
        }
    }
}

CommandProcessor::Ring& CommandProcessor::GetRing(u32 index) {
    std::scoped_lock lk(rings_lock);
    if (rings.size() <= index) {
        rings.resize(index + 1);
    }
    if (!rings[index]) {
        auto ring = std::make_unique<Ring>();
        ring->index = index;
        Ring* r = ring.get();
        r->thread = std::jthread([this, r](std::stop_token st) { Run(*r, st); });
        rings[index] = std::move(ring);
    }
    return *rings[index];
}

void CommandProcessor::Push(u32 index, Job job) {
    Ring& r = GetRing(index);
    {
        std::scoped_lock lk(r.lock);
        r.jobs.push_back(std::move(job));
    }
    r.cv.notify_one();
}

void CommandProcessor::SubmitGraphics(std::vector<CmdBuffers> bufs, std::function<void()> after) {
    ++stats.submits;
    if (LtOn()) {
        DC_INFO("lt %.0f submit %zu bufs", LtUs(LtNow()), bufs.size());
    }
    const bool flip = static_cast<bool>(after);
    if (HistOn()) {
        u32 dw = 0;
        for (const auto& b : bufs) {
            dw += u32(b.dcb.size());
        }
        HistAdd('G', bufs.empty() ? 0 : reinterpret_cast<u64>(bufs[0].dcb.data()), dw, flip ? 256 : 0);
        // GT7 frame heads: 0x116 dwords whose chain lands on the 4 dwords right after them.
        static std::atomic<u64> heads{0}, unpatched{0};
        for (const auto& b : bufs) {
            if (b.dcb.size() == 0x116) {
                const u32* tail = b.dcb.data() + 0x116;
                const u64 n = ++heads;
                if ((tail[3] & 0xFFFFF) == 0 || tail[1] == 0) {
                    const u64 u = ++unpatched;
                    if (u <= 20 || u % 100 == 0) {
                        DC_WARN("gpu: frame head %p submitted with its tail chain unpatched (%08x %08x %08x %08x), "
                                "%llu of %llu heads", (const void*)b.dcb.data(), tail[0], tail[1], tail[2], tail[3],
                                (unsigned long long)u, (unsigned long long)n);
                    }
                }
            }
        }
    }
    Capture::BeforeGraphics(bufs, flip);
    Push(0, {std::move(bufs), std::move(after)});
    Capture::AfterGraphics(flip);
}

void CommandProcessor::SubmitCompute(u32 queue, std::span<const u32> acb) {
    HistAdd('K', reinterpret_cast<u64>(acb.data()), u32(acb.size()), queue);
    Capture::BeforeCompute(queue, acb);
    Push(queue, {{CmdBuffers{acb, {}}}, {}});
}

void CommandProcessor::WaitIdle() {
    std::vector<Ring*> all;
    {
        std::scoped_lock lk(rings_lock);
        for (auto& r : rings) {
            if (r) {
                all.push_back(r.get());
            }
        }
    }
    for (Ring* r : all) {
        std::unique_lock lk(r->lock);
        r->idle_cv.wait(lk, [&] { return r->jobs.empty() && !r->busy; });
    }
    if (pipe) { // an idle ring has flushed its last batch
        std::unique_lock lk(pipe->lock);
        pipe->done_cv.wait(lk, [&] { return pipe->queue.empty() && !pipe->busy; });
    }
}

bool CommandProcessor::WaitIdleFor(int ms) {
    std::vector<Ring*> all;
    {
        std::scoped_lock lk(rings_lock);
        for (auto& r : rings) {
            if (r) {
                all.push_back(r.get());
            }
        }
    }
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    bool ok = true;
    for (Ring* r : all) {
        std::unique_lock lk(r->lock);
        ok &= r->idle_cv.wait_until(lk, until, [&] { return r->jobs.empty() && !r->busy; });
    }
    if (pipe) {
        std::unique_lock lk(pipe->lock);
        ok &= pipe->done_cv.wait_until(lk, until, [&] { return pipe->queue.empty() && !pipe->busy; });
    }
    return ok;
}

bool CommandProcessor::WaitQuiescentFor(int ms) {
    std::vector<Ring*> all;
    {
        std::scoped_lock lk(rings_lock);
        for (auto& r : rings) {
            if (r) {
                all.push_back(r.get());
            }
        }
    }
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    for (;;) {
        bool quiet = true;
        for (Ring* r : all) {
            std::scoped_lock lk(r->lock);
            quiet &= (r->jobs.empty() && !r->busy) || r->waiting.load(std::memory_order_acquire);
        }
        if (pipe) {
            std::scoped_lock lk(pipe->lock);
            quiet &= pipe->queue.empty() && !pipe->busy;
        }
        if (quiet) {
            return true;
        }
        if (std::chrono::steady_clock::now() > until) {
            return false;
        }
        sched_yield();
    }
}

namespace {
template <typename T>
void Put(std::vector<u8>& out, const T& v) {
    const auto* p = reinterpret_cast<const u8*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}
void PutBytes(std::vector<u8>& out, const void* data, size_t n) {
    const auto* p = static_cast<const u8*>(data);
    out.insert(out.end(), p, p + n);
}
template <typename T>
T Get(const std::vector<u8>& in, size_t& at) {
    T v{};
    if (at + sizeof(T) <= in.size()) {
        std::memcpy(&v, in.data() + at, sizeof(T));
    }
    at += sizeof(T);
    return v;
}
void GetBytes(const std::vector<u8>& in, size_t& at, void* data, size_t n) {
    if (at + n <= in.size()) {
        std::memcpy(data, in.data() + at, n);
    }
    at += n;
}
constexpr u32 kStateMagic = 0x43505331; // "CPS1"
} // namespace

std::vector<u8> CommandProcessor::SaveState() {
    std::vector<u8> out;
    Put(out, kStateMagic);
    PutBytes(out, regs.Data(), RegFile::kSize * 4);
    Put(out, ce.ce_counter);
    Put(out, ce.de_counter);
    PutBytes(out, ce.ram.data(), ce.ram.size() * 4);
    Put(out, index_base);
    Put(out, index_size);
    Put(out, num_instances);
    Put(out, flip_label.addr);
    Put(out, flip_label.value);
    Put(out, u8(flip_label.interrupt));
    std::scoped_lock lk(rings_lock);
    u32 n = 0;
    for (auto& r : rings) {
        n += r ? 1 : 0;
    }
    Put(out, n);
    for (auto& r : rings) {
        if (!r) {
            continue;
        }
        Put(out, r->index);
        Put(out, r->indirect_base);
        PutBytes(out, r->regs.Data(), RegFile::kSize * 4);
    }
    return out;
}

void CommandProcessor::LoadState(const std::vector<u8>& in) {
    size_t at = 0;
    if (Get<u32>(in, at) != kStateMagic) {
        DC_ERROR("capture: CP state has the wrong magic");
        return;
    }
    GetBytes(in, at, regs.Data(), RegFile::kSize * 4);
    ce.stack.clear();
    ce.ce_counter = Get<u64>(in, at);
    ce.de_counter = Get<u64>(in, at);
    GetBytes(in, at, ce.ram.data(), ce.ram.size() * 4);
    index_base = Get<u64>(in, at);
    index_size = Get<u32>(in, at);
    num_instances = Get<u32>(in, at);
    flip_label.addr = Get<u64>(in, at);
    flip_label.value = Get<u32>(in, at);
    flip_label.interrupt = Get<u8>(in, at) != 0;
    {
        std::scoped_lock lk(fence_lock);
        pending_fences.clear();
        if (pipe) {
            pipe->pending.clear();
        }
    }
    if (pipe) { // idle: the worker replays nothing meanwhile
        pipe->regs = regs;
    }
    const u32 n = Get<u32>(in, at);
    for (u32 k = 0; k < n && at < in.size(); ++k) {
        const u32 index = Get<u32>(in, at);
        Ring& r = GetRing(index);
        std::scoped_lock lk(r.lock);
        r.indirect_base = Get<u64>(in, at);
        GetBytes(in, at, r.regs.Data(), RegFile::kSize * 4);
    }
}

void CommandProcessor::Run(Ring& ring, std::stop_token stop) {
    char name[32];
    std::snprintf(name, sizeof(name), ring.index == 0 ? "GPU gfx" : "GPU compute %u", ring.index);
    pthread_setname_np(name);
    Core::GuestProfile::RegisterThread("CP");
    // The CP threads are on every frame's critical path, among dozens of
    // guest threads (some spin-waiting) at the default QoS: ask for P-cores and
    // priority. DC_NO_QOS=1 leaves the default (A/B).
    if (!std::getenv("DC_NO_QOS")) {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
    t_ring = &ring;
    t_regs = ring.index == 0 ? &regs : &ring.regs;
    Gpu::t_queue = ring.index;
    for (;;) {
        Job job;
        {
            std::unique_lock lk(ring.lock);
            ring.cv.wait(lk, [&] { return !ring.jobs.empty() || stop.stop_requested(); });
            if (stop.stop_requested()) {
                return;
            }
            job = std::move(ring.jobs.front());
            ring.jobs.pop_front();
            ring.busy = true;
        }
        HistAdd('J', 0, u32(t_hist_packets), ring.index);
        const bool lt = ring.index == 0 && LtOn();
        double lt_user = 0, lt_sys = 0;
        if (lt) {
            LtThreadCpu(&lt_user, &lt_sys);
            lt_lock_ns = 0;
            DC_INFO("lt %.0f job start (%zu queued)", LtUs(LtNow()), ring.jobs.size());
        }
        for (const auto& buf : job.buffers) {
            if (ring.index == 0) {
                ce.stack.clear();
                if (!buf.ccb.empty()) {
                    ce.stack.push_back(buf.ccb);
                }
                ce.ce_counter = ce.de_counter = 0;
            }
            Execute(buf.dcb, ring.index != 0, 0);
            if (ring.index == 0) {
                RunConstantEngine([] { return false; }); // whatever the CE has left
            }
        }
        if (job.after) {
            if (Piped()) {
                PipeFn(std::move(job.after));
            } else {
                job.after();
            }
        }
        if (flip_label.addr || flip_label.interrupt) {
            const FlipLabel label = flip_label;
            if (lt) {
                DC_INFO("lt %.0f fliplabel rec %#llx=%#x", LtUs(LtNow()), (unsigned long long)label.addr, label.value);
            }
            AfterGpu([label, this] {
                if (LtOn()) {
                    DC_INFO("lt %.0f fliplabel write %#x", LtUs(LtNow()), label.value);
                }
                if (label.addr) {
                    *Ptr<u32>(static_cast<u32>(label.addr), static_cast<u32>(label.addr >> 32)) = label.value;
                }
                if (label.interrupt) {
                    Interrupt(kIrqGfxEop);
                }
            });
            flip_label = {};
        }
        if (Piped()) {
            PipeFn([this] { renderer->OnSubmit(); });
            PipeFlush();
        } else if (renderer) {
            renderer->OnSubmit(); // hand everything, signals included, to the host GPU
        }
        if (lt) {
            double user = 0, sys = 0;
            LtThreadCpu(&user, &sys);
            DC_INFO("lt %.0f job end cpu user %.0f sys %.0f lock %.0f us", LtUs(LtNow()), user - lt_user, sys - lt_sys,
                    lt_lock_ns / 1e3);
        }
        HistAdd('X', 0, u32(t_hist_packets), ring.index);
        {
            std::scoped_lock lk(ring.lock);
            ring.busy = false;
        }
        ring.idle_cv.notify_all();
    }
}

RegFile& CommandProcessor::CurRegs() {
    return t_regs ? *t_regs : regs;
}

void CommandProcessor::SetRegs(u32 index, const u32* values, u32 count) {
    if (index >= RegFile::kSize || count > RegFile::kSize - index) {
        return;
    }
    std::memcpy(CurRegs().Data() + index, values, count * 4);
}

void CommandProcessor::AfterGpu(std::function<void()> fn) {
    if (Piped()) {
        PipeFn([this, fn = std::move(fn)]() mutable { renderer->Signal(std::move(fn)); });
    } else if (renderer) {
        renderer->Signal(std::move(fn));
    } else {
        fn();
    }
}

void CommandProcessor::FenceWrite(u32 data_sel, u64 addr, u32 lo, u32 hi, u32 irq) {
    const u32 hist_ring = t_ring ? t_ring->index : 99;
    HistAdd('R', addr, lo, hist_ring);
    const bool value_write = data_sel == kData32 || data_sel == kData64;
    const bool piped = Piped();
    if (value_write) {
        std::scoped_lock lk(fence_lock);
        auto& m = piped ? pipe->pending : pending_fences;
        m[addr] = lo;
        if (data_sel == kData64) {
            m[addr + 4] = hi;
        }
    }
    const bool lt = addr == Lt().addr && LtOn();
    if (lt) {
        DC_INFO("lt %.0f label rec %#x%s", LtUs(LtNow()), lo, t_ring && t_ring->index ? " (compute)" : "");
    }
    auto write = [=, this] {
        if (lt) {
            DC_INFO("lt %.0f label write %#x", LtUs(LtNow()), lo);
        }
        WriteFence(data_sel, static_cast<u32>(addr), static_cast<u32>(addr >> 32), lo, hi);
        HistAdd('W', addr, lo, hist_ring);
        if (data_sel == kData32 || data_sel == kData64) {
            std::scoped_lock lk(fence_lock);
            // A later write to the same label may still be on its way.
            auto it = pending_fences.find(addr);
            if (it != pending_fences.end() && it->second == lo) {
                pending_fences.erase(it);
            }
            if (data_sel == kData64) {
                it = pending_fences.find(addr + 4);
                if (it != pending_fences.end() && it->second == hi) {
                    pending_fences.erase(it);
                }
            }
        }
        if (irq) {
            Interrupt(irq);
        }
    };
    if (!piped) {
        AfterGpu(std::move(write));
        return;
    }
    // Now the renderer gets it: pending for every ring from here on.
    PipeFn([=, this, write = std::move(write)]() mutable {
        if (value_write) {
            std::scoped_lock lk(fence_lock);
            for (u32 k = 0; k < (data_sel == kData64 ? 2u : 1u); ++k) {
                const u64 a = addr + 4 * k;
                const u32 v = k ? hi : lo;
                pending_fences[a] = v;
                auto it = pipe->pending.find(a);
                if (it != pipe->pending.end() && it->second == v) {
                    pipe->pending.erase(it);
                }
            }
        }
        renderer->Signal(std::move(write));
    });
}

bool CommandProcessor::PendingFence(u64 addr, u32* value) {
    std::scoped_lock lk(fence_lock);
    auto it = pending_fences.find(addr);
    if (it != pending_fences.end()) {
        *value = it->second;
        return true;
    }
    if (Piped()) { // recorded by this ring: ahead of anything it records next
        it = pipe->pending.find(addr);
        if (it != pipe->pending.end()) {
            *value = it->second;
            return true;
        }
    }
    return false;
}

void CommandProcessor::SyncRenderer() {
    if (Piped()) {
        PipeDrain();
    }
    if (renderer) {
        renderer->WaitIdle();
    }
}

void CommandProcessor::RunConstantEngine(const std::function<bool()>& done) {
    while (!ce.stack.empty() && !done()) {
        std::span<const u32>& cur = ce.stack.back();
        if (cur.empty()) {
            ce.stack.pop_back();
            continue;
        }
        const u32 header = cur[0];
        const u32 type = HeaderType(header);
        if (header == 0 || type == 2) {
            cur = cur.subspan(1);
            continue;
        }
        if (type != 3) {
            DC_WARN("gpu: CE: unsupported packet type %u", type);
            ce.stack.clear();
            return;
        }
        const u32 n = HeaderBodyDwords(header);
        if (n + 1 > cur.size()) {
            ce.stack.pop_back();
            continue;
        }
        const u32* body = cur.data() + 1;
        switch (HeaderOpcode(header)) {
        case Op::WriteConstRam: {
            const u32 off = (body[0] & 0xFFFF) / 4;
            for (u32 k = 0; k + 1 < n && off + k < ce.ram.size(); ++k) {
                ce.ram[off + k] = body[1 + k];
            }
            break;
        }
        case Op::LoadConstRam: {
            const u32* src = Ptr<const u32>(body[0], body[1]);
            const u32 count = body[2] & 0x7FFF;
            const u32 off = (body[3] & 0xFFFF) / 4;
            if (src && off + count <= ce.ram.size()) {
                std::memcpy(ce.ram.data() + off, src, count * 4);
            }
            break;
        }
        case Op::DumpConstRam: {
            const u32 off = (body[0] & 0xFFFF) / 4;
            const u32 count = body[1] & 0x7FFF;
            u32* dst = Ptr<u32>(body[2], body[3]);
            if (!dst || off + count > ce.ram.size()) {
                break;
            }
            if (Piped()) {
                const u64 a = reinterpret_cast<u64>(dst);
                const u32 hdr[3] = {u32(a), u32(a >> 32), 1};
                PipeWords(kPipeMem, hdr, 3, ce.ram.data() + off, count);
                break;
            }
            std::memcpy(dst, ce.ram.data() + off, count * 4);
            if (renderer) {
                renderer->NoteConstDump(reinterpret_cast<u64>(dst), count * 4);
            }
            break;
        }
        case Op::WriteData: {
            static const bool log_ce = std::getenv("DC_LOG_CE") != nullptr;
            if (log_ce) {
                const u32 dst_sel = (body[0] >> 8) & 0xF;
                DC_INFO("gpu: CE WRITE_DATA %#llx %u dwords ctl %#x dst_sel %u",
                        (unsigned long long)(u64(body[1]) | (u64(body[2]) << 32)), n > 3 ? n - 3 : 0, body[0], dst_sel);
            }
            break;
        }
        case Op::DmaData: {
            static const bool log_ce = std::getenv("DC_LOG_CE") != nullptr;
            if (log_ce) {
                const u32 ctl = body[0];
                const u32 cmd = body[5];
                const u32 bytes = cmd & 0x1FFFFF;
                const u32 src_sel = (ctl >> 29) & 3;
                const u32 dst_sel = (ctl >> 20) & 3;
                const u64 dst_addr = dst_sel == 1 ? (body[3] & 0xFFFF) : u64(body[3]) | (u64(body[4]) << 32);
                const u64 src_addr = src_sel == 2 ? body[1]
                                    : src_sel == 1 ? (body[1] & 0xFFFF) : u64(body[1]) | (u64(body[2]) << 32);
                DC_INFO("gpu: CE DMA_DATA dst %#llx src %#llx %u bytes (ctl %#x src_sel %u dst_sel %u)",
                        (unsigned long long)dst_addr, (unsigned long long)src_addr, bytes, ctl, src_sel, dst_sel);
            }
            break;
        }
        case Op::IncrementCeCounter:
            ++ce.ce_counter;
            break;
        case Op::WaitOnDeCounterDiff:
            // Signed: a DE that is ahead (it can count a draw before the CE starts)
            // leaves the CE free to run, not blocked on an underflowed difference.
            if (static_cast<s64>(ce.ce_counter - ce.de_counter) >= static_cast<s64>(body[0])) {
                return; // blocked until the DE catches up; retried later
            }
            break;
        case Op::IndirectBufferConst: {
            const u32* ib = Ptr<const u32>(body[0], body[1] & 0xFFFF);
            const u32 size = body[2] & 0xFFFFF;
            cur = cur.subspan(n + 1);
            if (ib && size) {
                ce.stack.push_back({ib, size});
            }
            continue;
        }
        default:
            break;
        }
        cur = cur.subspan(n + 1);
    }
}

void CommandProcessor::Draw(bool indexed, u32 count, u32 index_offset, u32 first_vertex, u32 instances) {
    ++stats.draws;
    if (!renderer || count == 0) {
        return;
    }
    DrawCall dc;
    dc.indexed = indexed;
    dc.count = count;
    dc.instances = draw_indirect_args ? instances : instances ? instances : num_instances;
    dc.indirect_args = draw_indirect_args;
    dc.index_base = index_base;
    dc.index_offset = index_offset;
    dc.index_size = index_size;
    dc.first_vertex = first_vertex;
    if (Piped()) {
        u32 w[kDrawCallWords] = {};
        std::memcpy(w, &dc, sizeof(dc));
        PipeWords(kPipeDraw, w, kDrawCallWords);
        return;
    }
    renderer->Draw(CurRegs(), dc);
}

// Sanity limits for indirect args (a 1080p frame never needs more).
constexpr u32 kMaxIndirectCount = 1u << 24;
constexpr u64 kMaxIndirectGroups = 1ull << 20;

// Indirect draw/dispatch arguments live at the SET_BASE address + offset.
const u32* CommandProcessor::IndirectArgs(u32 offset, u32 dwords) {
    const Ring* ring = t_ring;
    const u64 addr = (ring ? ring->indirect_base : 0) + offset;
    if (!addr) {
        return nullptr;
    }
    (void)dwords;
    return reinterpret_cast<const u32*>(addr);
}

void CommandProcessor::Interrupt(u32 irq) {
    if (on_irq) {
        on_irq(irq);
    }
}

// DC_FIND_UD=<dword>: log dispatches whose compute user data holds that
// dword (the low half of a buffer address: finds the shader writing it).
void CommandProcessor::FindUserData(u32 x, u32 y, u32 z) {
    static const char* env = std::getenv("DC_FIND_UD");
    if (!env) {
        return;
    }
    static const u32 needle = u32(std::strtoul(env, nullptr, 16));
    const u32* ud = regs.UserData(Reg::COMPUTE_USER_DATA_0);
    for (u32 k = 0; k < 16; ++k) {
        if (ud[k] == needle) {
            std::string all;
            for (u32 j = 0; j < 16; ++j) {
                char w[12];
                std::snprintf(w, sizeof(w), " %08x", ud[j]);
                all += w;
            }
            DC_INFO("gpu: find ud %#x: cs %#llx ud[%u] %ux%ux%u:%s", needle,
                    (unsigned long long)regs.ProgramAddr(Reg::COMPUTE_PGM_LO), k, x, y, z, all.c_str());
        }
    }
    // One level down: tables the user data points to.
    static const u64 dump_cs = std::getenv("DC_FIND_UD_CS") ? std::strtoull(std::getenv("DC_FIND_UD_CS"), nullptr, 16) : 0;
    if (dump_cs && regs.ProgramAddr(Reg::COMPUTE_PGM_LO) == dump_cs) {
        std::string all;
        for (u32 j = 0; j < 16; ++j) {
            char w[12];
            std::snprintf(w, sizeof(w), " %08x", ud[j]);
            all += w;
        }
        DC_INFO("gpu: ud of cs %#llx %ux%ux%u:%s", (unsigned long long)dump_cs, x, y, z, all.c_str());
        const u64 tp = u64(ud[12]) | (u64(ud[13] & 0xFFFF) << 32);
        if (tp && Mapped(reinterpret_cast<const u8*>(tp), 64)) {
            const u32* t = reinterpret_cast<const u32*>(tp);
            DC_INFO("gpu:   ud[12]-> %08x %08x %08x %08x | %08x %08x %08x %08x | %08x %08x %08x %08x", t[0], t[1], t[2],
                    t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11]);
        }
    }
    for (u32 k = 0; k + 1 < 16; ++k) {
        const u64 ptr = u64(ud[k]) | (u64(ud[k + 1] & 0xFFFF) << 32);
        if (ptr < 0x7000000000ull || ptr >= 0x8000000000ull || (ptr & 3) ||
            !Mapped(reinterpret_cast<const u8*>(ptr), 256)) {
            continue;
        }
        const u32* t = reinterpret_cast<const u32*>(ptr);
        for (u32 j = 0; j < 64; ++j) {
            if (t[j] == needle) {
                DC_INFO("gpu: find ud %#x: cs %#llx ud[%u]->[%u] %ux%ux%u: %08x %08x %08x %08x", needle,
                        (unsigned long long)regs.ProgramAddr(Reg::COMPUTE_PGM_LO), k, j, x, y, z, t[j],
                        j + 1 < 64 ? t[j + 1] : 0, j + 2 < 64 ? t[j + 2] : 0, j + 3 < 64 ? t[j + 3] : 0);
            }
        }
    }
}

void CommandProcessor::Execute(std::span<const u32> cmds, bool compute, int depth) {
    RegFile& regs = CurRegs();
    const bool piped = Piped();
    // SET_*_REG: the registers here, and in the pipe when recording into it.
    auto set_regs = [&](u32 index, const u32* values, u32 count) {
        SetRegs(index, values, count);
        if (piped && index < RegFile::kSize && count <= RegFile::kSize - index) {
            PipeRegs(index, values, count);
        }
    };
    const u32* p = cmds.data();
    const u32* end = p + cmds.size();

    const u32* recent[4] = {}; // last packets, for desync reports
    while (p < end) {
        const u32 header = *p;
        const u32 type = HeaderType(header);
        if (type == 3) {
            std::memmove(recent, recent + 1, sizeof(recent) - sizeof(recent[0]));
            recent[3] = p;
        }
        if (header == 0 || type == 2) { // filler / trailing padding
            ++p;
            continue;
        }
        if (type == 0) {
            const u32 base_reg = header & 0xFFFF;
            const u32 count = ((header >> 16) & 0x3FFF) + 1;
            if (p + 1 + count > end) {
                break;
            }
            for (u32 i = 0; i < count; ++i) {
                if (base_reg + i < RegFile::kSize) {
                    regs[base_reg + i] = p[1 + i];
                }
            }
            if (Piped() && base_reg < RegFile::kSize) {
                PipeRegs(base_reg, p + 1, std::min(count, RegFile::kSize - base_reg));
            }
            p += 1 + count;
            continue;
        }
        if (type != 3) {
            DC_WARN("gpu: unsupported PM4 packet type %u (header %#x) at +%#zx of %#zx dwords", type, header,
                    static_cast<size_t>(p - cmds.data()), cmds.size());
            DC_WARN("gpu:   around: %08x %08x %08x %08x [%08x] %08x %08x %08x", p[-4], p[-3], p[-2], p[-1], p[0], p[1],
                    p[2], p[3]);
            for (const u32* q : recent) {
                if (q) {
                    DC_WARN("gpu:   after +%#zx: %08x %08x %08x %08x %08x %08x", static_cast<size_t>(q - cmds.data()),
                            q[0], q[1], q[2], q[3], q[4], q[5]);
                }
            }
            return;
        }
        const u32 op = HeaderOpcode(header);
        const u32 body_n = HeaderBodyDwords(header);
        const u32* body = p + 1;
        ++stats.packets;
        ++t_hist_packets;

        // DC_FIND_DWORD=<value>[,<value>]: log PM4 packets whose body contains
        // one of these dwords. Used to find where a resource address is set.
        static const char* find_env = std::getenv("DC_FIND_DWORD");
        static u32 find_vals[4] = {};
        static u32 find_count = 0;
        static bool find_init = false;
        if (!find_init && find_env) {
            const char* q = find_env;
            while (*q && find_count < 4) {
                find_vals[find_count++] = u32(std::strtoul(q, const_cast<char**>(&q), 16));
                if (*q == ',') ++q;
                else break;
            }
            find_init = true;
        }
        for (u32 fi = 0; fi < find_count; ++fi) {
            const u32 needle = find_vals[fi];
            for (u32 bi = 0; bi < body_n; ++bi) {
                if (body[bi] != needle) continue;
                DC_INFO("gpu: find dword %#x op %#x body[%u] at cmd+%#zx; body", needle, op, bi,
                        static_cast<size_t>(body - cmds.data()));
                for (u32 bj = 0; bj < body_n && bj < 12; ++bj) {
                    DC_INFO("gpu:   [%u] %08x", bj, body[bj]);
                    if (bj == 11 && body_n > 12) DC_INFO("gpu:   ... (%u dwords)", body_n);
                }
            }
        }

        // DC_WATCH_GPU=<addr hex>[,<addr>...] (debug): after each dispatch, draw or
        // memory packet, waits for the host GPU and logs which one changed a dword.
        struct GpuWatch {
            CommandProcessor* cp;
            u32 op;
            bool compute;
            ~GpuWatch() {
                static const char* env = std::getenv("DC_WATCH_GPU");
                if (!env) {
                    return;
                }
                static std::vector<u64> addrs = [] {
                    std::vector<u64> v;
                    const char* q = env;
                    while (*q) {
                        char* e = nullptr;
                        v.push_back(std::strtoull(q, &e, 16));
                        q = *e == ',' ? e + 1 : e;
                        if (e == q && *e != ',') break;
                    }
                    return v;
                }();
                static std::vector<u32> last(addrs.size(), 0xdeadbeef);
                const bool interesting = op == Op::DispatchDirect || op == Op::DispatchIndirect ||
                                         op == Op::DmaData || op == Op::WriteData || op == Op::ReleaseMem ||
                                         op == Op::DrawIndex2 || op == Op::DrawIndexAuto ||
                                         op == Op::DrawIndexOffset2 || op == Op::DrawIndirect ||
                                         op == Op::DrawIndexIndirect;
                if (!interesting) {
                    return;
                }
                cp->SyncRenderer();
                for (size_t i = 0; i < addrs.size(); ++i) {
                    const u32 v = *Ptr<const u32>(static_cast<u32>(addrs[i]), static_cast<u32>(addrs[i] >> 32));
                    if (v != last[i]) {
                        DC_INFO("gpu watch %#llx: %#x -> %#x after op %#x%s cs %#llx ps %#llx",
                                (unsigned long long)addrs[i], last[i], v, op, compute ? " (compute)" : "",
                                (unsigned long long)cp->CurRegs().ProgramAddr(Reg::COMPUTE_PGM_LO),
                                (unsigned long long)cp->CurRegs().ProgramAddr(Reg::SPI_SHADER_PGM_LO_PS));
                        last[i] = v;
                    }
                }
            }
        } gpu_watch{this, op, compute};

        switch (op) {
        case Op::Nop:
            // The Gnm driver hides flip requests in NOPs: "when this frame is
            // shown, write `value` to `label` and/or raise the EOP interrupt".
            if (body_n >= 4 && (body[0] == kNopPrepareFlipLabel || body[0] == kNopPrepareFlipInterruptLabel)) {
                flip_label = {u64(body[1] & ~3u) | (u64(body[2]) << 32), body[3], body[0] == kNopPrepareFlipInterruptLabel};
            } else if (body_n >= 1 && body[0] == kNopPrepareFlipInterrupt) {
                flip_label = {0, 0, true};
            }
            break;
        case Op::ClearState:
            ClearState();
            break;

        case Op::SetContextReg:
            set_regs(kContextRegBase + body[0], body + 1, body_n ? body_n - 1 : 0);
            if (Piped()) {
                PipeWords(kPipeCtx, body, body_n);
            } else if (renderer) {
                renderer->OnContextRegs(body, body_n);
            }
            break;
        case Op::SetShReg:
            set_regs(kShRegBase + body[0], body + 1, body_n ? body_n - 1 : 0);
            break;
        case Op::SetUconfigReg:
            set_regs(kUconfigRegBase + body[0], body + 1, body_n ? body_n - 1 : 0);
            break;
        case Op::SetConfigReg:
            set_regs(kConfigRegBase + body[0], body + 1, body_n ? body_n - 1 : 0);
            break;

        case Op::IndirectBuffer: {
            constexpr u32 kIbChain = 1u << 20;
            // GT7 submits a frame's head (0x116 dwords) whose chain leads, through
            // a stub, to the 4 dwords right after it, a chain the game fills in
            // with the rest of the frame. It is almost always written before the
            // submit, but now and then a little after (0.06-0.5 ms seen): a real
            // CP is still busy with the previous frame then, ours is idle when
            // nothing compiles (render pipeline prewarm) and read zeros. The frame
            // then ended there, its end-of-frame labels were never written, async
            // compute waited for them (WaitRegMem timeout) and the game stopped
            // submitting. A chain to nowhere (no size, or no address above 4 GB)
            // is a packet still being written: wait for the game to finish it.
            // DC_NO_CHAIN_WAIT=1: old behavior (end the command buffer there).
            if ((body[2] & kIbChain) && ((body[2] & 0xFFFFF) == 0 || (body[1] & 0xFFFF) == 0)) {
                static const bool no_wait = std::getenv("DC_NO_CHAIN_WAIT") != nullptr;
                const volatile u32* vb = body;
                const auto t0 = std::chrono::steady_clock::now();
                while (!no_wait && ((vb[2] & 0xFFFFF) == 0 || (vb[1] & 0xFFFF) == 0) &&
                       std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(500)) {
                    sched_yield();
                }
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                static std::atomic<u32> reports{0};
                if (!no_wait && reports.fetch_add(1) < 20) {
                    DC_INFO("gpu: chain at %p was still being written: %s after %.3f ms (-> %#llx, %u dwords)",
                            (const void*)p, ((vb[2] & 0xFFFFF) && (vb[1] & 0xFFFF)) ? "done" : "gave up", ms,
                            (unsigned long long)(u64(vb[0]) | (u64(vb[1] & 0xFFFF) << 32)), vb[2] & 0xFFFFF);
                }
                std::atomic_thread_fence(std::memory_order_acquire);
            }
            const u32* ib = Ptr<const u32>(body[0], body[1] & 0xFFFF);
            const u32 size = body[2] & 0xFFFFF;
            if (!compute && depth < 2) {
                HistAdd(depth ? 'i' : 'I', reinterpret_cast<u64>(ib), size | (body[2] & kIbChain ? 0x80000000u : 0),
                        0);
            }
            if ((!ib || !size) && HistOn()) {
                const u32* pkt = p;
                DC_WARN("gpu: INDIRECT_BUFFER%s to %p size %u at %p (+%#zx of %zu dwords, depth %d%s)",
                        body[2] & kIbChain ? " chain" : "", (const void*)ib, size, (const void*)pkt,
                        static_cast<size_t>(pkt - cmds.data()), cmds.size(), depth, compute ? ", compute" : "");
                for (size_t k = 0; k + 8 <= cmds.size() && k < 320; k += 8) {
                    const u32* q = cmds.data() + k;
                    DC_WARN("gpu:   +%02zx: %08x %08x %08x %08x %08x %08x %08x %08x", k, q[0], q[1], q[2], q[3], q[4],
                            q[5], q[6], q[7]);
                }
                const u32 snap[4] = {pkt[0], pkt[1], pkt[2], pkt[3]};
                DC_WARN("gpu:   packet now %08x %08x %08x %08x", snap[0], snap[1], snap[2], snap[3]);
                if (t_last_q) {
                    HistDumpAddr(t_last_q, 200, "last CPU-memory wait of this ring");
                }
                HistDumpAddr(reinterpret_cast<u64>(pkt + 2), 200, "the chain size dword");
                {
                    LabelHist& h = Hist();
                    std::scoped_lock lk(h.lock);
                    const u64 cnt = std::min<u64>(h.n, h.ev.size());
                    int shown = 0;
                    const u64 now = HistNow();
                    for (u64 k = h.n; k > h.n - cnt && shown < 12; --k) {
                        const LabelEvent& e = h.ev[(k - 1) & (h.ev.size() - 1)];
                        if ((e.kind == 'P' || e.kind == 'p' || e.kind == 'S') && e.ring == (t_ring ? t_ring->index : 99)) {
                            DC_WARN("gpu:   wait %c %#llx = %#x ring %u, %.3f ms ago", e.kind,
                                    (unsigned long long)e.addr, e.value, e.ring, (now - e.t) / 1e6);
                            ++shown;
                        }
                    }
                }
                std::thread([pkt, snap] {
                    u32 last[4] = {snap[0], snap[1], snap[2], snap[3]};
                    const auto t0 = std::chrono::steady_clock::now();
                    while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5)) {
                        if (std::memcmp(last, pkt, 16) != 0) {
                            std::memcpy(last, pkt, 16);
                            DC_WARN("gpu: IB packet at %p now %08x %08x %08x %08x, %.3f ms after it ran",
                                    (const void*)pkt, last[0], last[1], last[2], last[3],
                                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                                        .count());
                        }
                        usleep(20);
                    }
                }).detach();
            }
            // A chained buffer replaces the current one, however many links there are.
            if (body[2] & kIbChain) {
                p = ib;
                end = ib + size;
                continue;
            }
            if (depth < 4) {
                Execute({ib, size}, compute, depth + 1);
            }
            break;
        }

        case Op::EventWriteEop: {
            const u32 data_control = body[2];
            const u32 int_sel = (data_control >> 24) & 3;
            const bool irq = int_sel == kIntOnly || int_sel == kIntWriteConfirm;
            FenceWrite((data_control >> 29) & 7, u64(body[1]) | (u64(data_control & 0xFFFF) << 32), body[3], body[4],
                       irq ? kIrqGfxEop : 0);
            break;
        }
        case Op::ReleaseMem: {
            const u32 dw2 = body[1];
            const u32 int_sel = (dw2 >> 24) & 7;
            const bool irq = int_sel == kIntWriteConfirm || int_sel == kIntUndocumented;
            FenceWrite((dw2 >> 29) & 7, u64(body[2]) | (u64(body[3]) << 32), body[4], body[5],
                       !irq ? 0 : compute ? kIrqCompute + t_ring->index : kIrqGfxEop);
            break;
        }
        case Op::EventWrite: { // no-op; each event type logged once (ZPASS_DONE = 0x15 needs results)
            static std::atomic<u64> seen{0};
            const u32 type = body_n ? body[0] & 0x3F : 0;
            if (!(seen.fetch_or(1ull << type) & (1ull << type))) {
                DC_INFO("gpu: CP%s: EVENT_WRITE type %#x (%u dwords)", compute ? " compute" : "", type, body_n);
            }
            break;
        }
        case Op::EventWriteEos: {
            const u32 cmd = (body[2] >> 29) & 7;
            if (cmd == 2) { // signal fence
                FenceWrite(kData32, u64(body[1]) | (u64(body[2] & 0xFFFF) << 32), body[3], 0, 0);
            }
            break;
        }

        case Op::WriteData: {
            const u32 dst_sel = (body[0] >> 8) & 0xF;
            if (dst_sel == 1 || dst_sel == 2 || dst_sel == 5) {
                static const bool log = std::getenv("DC_LOG_WRITEDATA") != nullptr;
                if (log) {
                    DC_INFO("gpu: WRITE_DATA %#llx %u dwords ctl %#x val %#x",
                            (unsigned long long)(u64(body[1]) | (u64(body[2]) << 32)), body_n - 3, body[0],
                            body_n > 3 ? body[3] : 0);
                }
                u32* dst = Ptr<u32>(body[1], body[2]);
                HistAdd('D', reinterpret_cast<u64>(dst), body_n > 3 ? body[3] : 0, t_ring ? t_ring->index : 99);
                // A write into the packets still to be parsed must land now.
                const u64 wa = reinterpret_cast<u64>(dst);
                const bool patches_ib = wa < reinterpret_cast<u64>(end) &&
                                        wa + (body_n - 3) * 4 > reinterpret_cast<u64>(cmds.data());
                if (Piped() && !patches_ib) {
                    const u64 a = wa;
                    const u32 hdr[3] = {u32(a), u32(a >> 32), body_n - 3 > 2 ? 1u : 0u};
                    PipeWords(kPipeMem, hdr, 3, body + 3, body_n - 3);
                    break;
                }
                std::memcpy(dst, body + 3, (body_n - 3) * 4);
                // Like a CE dump: draws recorded before this one must still read the
                // old data when the host GPU runs them, later ones the new data.
                if (renderer && body_n - 3 > 2) {
                    renderer->NoteConstDump(reinterpret_cast<u64>(dst), (body_n - 3) * 4);
                }
            }
            break;
        }

        case Op::DmaData: {
            // CP DMA: copies/fills between memory, GDS and immediate data.
            const u32 ctl = body[0];
            const u32 cmd = body[5];
            const u32 bytes = cmd & 0x1FFFFF;
            const u32 src_sel = (ctl >> 29) & 3; // 0/3 memory, 1 GDS, 2 data
            const u32 dst_sel = (ctl >> 20) & 3; // 0/3 memory, 1 GDS
            if ((cmd >> 27) & 1) {
                break; // destination is register space: cache prefetch hints
            }
            if (renderer) {
                const u32 dsel = dst_sel == 1 ? 1 : 0;
                const u32 ssel = src_sel == 3 ? 0 : src_sel;
                const u64 dst_addr = dsel ? (body[3] & 0xFFFF) : u64(body[3]) | (u64(body[4]) << 32);
                const u64 src_addr = ssel == 2 ? body[1]
                                   : ssel == 1 ? (body[1] & 0xFFFF) : u64(body[1]) | (u64(body[2]) << 32);
                // The game patches packets further down this very command buffer
                // (e.g. DISPATCH_DIRECT sizes): we parse it now, so such a copy
                // cannot wait on the host GPU queue.
                const u64 ib_lo = reinterpret_cast<u64>(cmds.data());
                const u64 ib_hi = reinterpret_cast<u64>(end);
                const bool patches_ib = !dsel && dst_addr < ib_hi && dst_addr + bytes > ib_lo;
                // A copy of the group counts into a DISPATCH_DIRECT further down:
                // the host GPU reads them itself when that dispatch runs, no wait.
                if (patches_ib && ssel == 0 && bytes == 12 && dst_addr >= ib_lo + 4) {
                    const u32 h = *reinterpret_cast<const u32*>(dst_addr - 4);
                    if (HeaderType(h) == 3 && HeaderOpcode(h) == Op::DispatchDirect) {
                        t_dispatch_args[dst_addr] = src_addr;
                        break;
                    }
                }
                if (bytes && !patches_ib && !dsel) {
                    HistAdd(ssel == 2 ? 'f' : 'c', dst_addr, bytes, t_ring ? t_ring->index : 99);
                }
                if (Piped() && !patches_ib) {
                    u32 b[6];
                    std::memcpy(b, body, sizeof(b));
                    PipeFn([=, this] {
                        if (bytes && renderer->DmaData(dsel, dst_addr, ssel, src_addr, bytes)) {
                            return;
                        }
                        DmaCopy(b, false);
                    });
                    break;
                }
                if (bytes && !patches_ib && renderer->DmaData(dsel, dst_addr, ssel, src_addr, bytes)) {
                    break;
                }
            }
            DmaCopy(body, compute);
            break;
        }

        case Op::WaitRegMem: {
            const u32 func = body[0] & 7;
            const bool memory = (body[0] >> 4) & 1;
            const u32 ref = body[3];
            const u32 mask = body[4];
            if (memory) {
                const volatile u32* addr = Ptr<const u32>(body[1] & ~3u, body[2]);
                while (Capture::ReplayApplying()) {
                    sched_yield(); // frame replay: the label may come before its data
                }
                if (HistOn() && !compute && reinterpret_cast<u64>(addr) >= 0x7000000000ull) {
                    HistAdd('Q', reinterpret_cast<u64>(addr), *addr, 0);
                    HistPoll(reinterpret_cast<u64>(addr));
                    t_last_q = reinterpret_cast<u64>(addr);
                    static std::atomic<int> dumps{0};
                    static const u64 dump_at = std::getenv("DC_HIST_DUMP_SUBMIT")
                                                   ? std::strtoull(std::getenv("DC_HIST_DUMP_SUBMIT"), nullptr, 10)
                                                   : 0;
                    if (dump_at && stats.submits.load() >= dump_at && dumps.fetch_add(1) < 3) {
                        HistDumpAddr(t_last_q, 200, "sample");
                    }
                }
                if (Compare(func, *addr & mask, ref)) {
                    break;
                }
                u32 pending = 0;
                if (PendingFence(reinterpret_cast<u64>(addr), &pending) && Compare(func, pending & mask, ref)) {
                    HistAdd('P', reinterpret_cast<u64>(addr), pending, t_ring ? t_ring->index : 99);
                    break;
                }
                // Whatever produces the value may sit in recorded host work.
                if (Piped()) {
                    PipeFn([this] { renderer->OnSubmit(); });
                    PipeFlush();
                } else if (renderer) {
                    renderer->OnSubmit();
                }
                HistAdd('S', reinterpret_cast<u64>(addr), ref, t_ring ? t_ring->index : 99);
                const u64 t = LtOn() ? LtNow() : 0;
                struct Report {
                    u64 t;
                    const volatile u32* a;
                    bool c;
                    ~Report() {
                        if (t) {
                            DC_INFO("lt %.0f waitregmem %p %.0f us%s", LtUs(LtNow()), (const void*)a,
                                    (LtNow() - t) / 1e3, c ? " (compute)" : "");
                        }
                    }
                } report{t, addr, compute};
                // DC_WAIT_SLEEP=1: yield for the first kWaitSpin, then sleep, so long
                // waits (async compute, ~9 ms a frame) stop burning a core. Off by
                // default: ~10% less process CPU on an idle Mac, fps not proven.
                static const bool no_sleep = std::getenv("DC_WAIT_SLEEP") == nullptr;
                constexpr auto kWaitSpin = std::chrono::microseconds(200);
                const auto t0 = std::chrono::steady_clock::now();
                struct Waiting {
                    std::atomic<bool>* w;
                    ~Waiting() {
                        if (w) w->store(false, std::memory_order_release);
                    }
                } waiting{t_ring ? &t_ring->waiting : nullptr};
                if (waiting.w) {
                    waiting.w->store(true, std::memory_order_release);
                }
                while (Capture::ReplayApplying() || !Compare(func, *addr & mask, ref)) {
                    // Another queue (async compute) may record the write meanwhile:
                    // all queues share one host command stream, so recorded work
                    // runs before anything this queue records after the wait.
                    if (!Capture::ReplayApplying() && PendingFence(reinterpret_cast<u64>(addr), &pending) &&
                        Compare(func, pending & mask, ref)) {
                        HistAdd('p', reinterpret_cast<u64>(addr), pending, t_ring ? t_ring->index : 99);
                            break;
                    }
                    const auto waited = std::chrono::steady_clock::now() - t0;
                    if (waited > std::chrono::seconds(1)) {
                        DC_WARN("gpu: WaitRegMem timeout (addr=%p val=%#x ref=%#x mask=%#x func=%u)",
                                addr, *addr, ref, mask, func);
                        HistAdd('T', reinterpret_cast<u64>(addr), *addr, t_ring ? t_ring->index : 99);
                        HistDump(reinterpret_cast<u64>(addr), t_ring ? t_ring->index : 99);
                        if (HistOn()) {
                            std::scoped_lock rl(rings_lock);
                            for (auto& r : rings) {
                                if (r) {
                                    DC_WARN("gpu:   ring %u: busy %d waiting %d jobs queued %zu", r->index,
                                            int(r->busy), int(r->waiting.load()), r->jobs.size());
                                }
                            }
                            DC_WARN("gpu:   submits %llu draws %llu", (unsigned long long)stats.submits.load(),
                                    (unsigned long long)stats.draws.load());
                        }
                        break;
                    }
                    if (no_sleep || waited < kWaitSpin) {
                        sched_yield();
                    } else {
                        usleep(50);
                    }
                }
            }
            break;
        }

        case Op::WaitOnCeCounter:
            if (!compute) {
                RunConstantEngine([&] { return ce.ce_counter > ce.de_counter; });
            }
            break;
        case Op::IncrementDeCounter:
            ++ce.de_counter;
            break;

        case Op::IndexType:
            index_size = (body[0] & 3) == 1 ? 4 : 2;
            break;
        case Op::IndexBase:
            index_base = u64(body[0] & ~1u) | (u64(body[1] & 0xFFFF) << 32);
            break;
        case Op::NumInstances:
            num_instances = body[0] ? body[0] : 1;
            break;

        case Op::DrawIndex2:
            index_base = u64(body[1]) | (u64(body[2] & 0xFFFF) << 32);
            Draw(true, body[3], 0, 0, 0);
            break;
        case Op::DrawIndexOffset2:
            Draw(true, body[2], body[1], 0, 0);
            break;
        case Op::DrawIndexAuto:
            Draw(false, body[0], 0, 0, 0);
            break;
        case Op::SetBase:
            if (t_ring) {
                t_ring->indirect_base = u64(body[1] & ~7u) | (u64(body[2] & 0xFFFF) << 32);
            }
            break;
        case Op::DrawIndirect:
        case Op::DrawIndexIndirect: {
            // DrawIndirect args: vertex count, instances, first vertex, first instance.
            // DrawIndexIndirect: index count, instances, first index, base vertex, first instance.
            const bool indexed = op == Op::DrawIndexIndirect;
            // DC_INDIRECT_SYNC=1: wait for the host GPU before reading the args
            // (they may be written by a culling compute still in flight).
            // DC_INDIRECT_LOG=1: log every indirect draw's CP-read counts.
            static const bool indirect_sync = std::getenv("DC_INDIRECT_SYNC") != nullptr;
            static const bool indirect_log = std::getenv("DC_INDIRECT_LOG") != nullptr;
            if (indirect_sync) {
                SyncRenderer();
            }
            const u32* a = IndirectArgs(body[0], indexed ? 5 : 4);
            if (!a) {
                break;
            }
            if (indirect_log) {
                DC_INFO("gpu: indirect draw count=%u instances=%u args %p (vs %#llx ps %#llx)", a[0], a[1],
                        (const void*)a, (unsigned long long)regs.ProgramAddr(Reg::SPI_SHADER_PGM_LO_VS),
                        (unsigned long long)regs.ProgramAddr(Reg::SPI_SHADER_PGM_LO_PS));
            }
            const u32 base_vtx_loc = body[1] & 0xFFFF;
            const u32 start_inst_loc = body[2] & 0xFFFF;
            if (base_vtx_loc) {
                set_regs(kShRegBase + base_vtx_loc, indexed ? &a[3] : &a[2], 1);
            }
            if (start_inst_loc) {
                set_regs(kShRegBase + start_inst_loc, indexed ? &a[4] : &a[3], 1);
            }
            // The args may be GPU-written and not landed yet; garbage counts
            // would make a draw that hangs the GPU. Instance counts usually
            // come from a compute shader (particles) that has not run when we
            // get here: the host GPU reads them from the args when it draws.
            // DC_GPU_DRAW_ARGS (dangerous, see the Metal renderer): insane CP-read
            // instance counts are left to the host GPU's own read of the args.
            static const bool gpu_args = std::getenv("DC_GPU_DRAW_ARGS") != nullptr;
            if (a[0] > kMaxIndirectCount || (!gpu_args && a[1] > kMaxIndirectCount)) {
                DC_WARN("gpu: skipping indirect draw with count=%u instances=%u args %p (vs %#llx ps %#llx)", a[0],
                        a[1], (const void*)a, (unsigned long long)regs.ProgramAddr(Reg::SPI_SHADER_PGM_LO_VS),
                        (unsigned long long)regs.ProgramAddr(Reg::SPI_SHADER_PGM_LO_PS));
                break;
            }
            const u32 instances = a[1] > kMaxIndirectCount ? 0 : a[1];
            draw_indirect_args = (t_ring ? t_ring->indirect_base : 0) + body[0];
            if (indexed) {
                Draw(true, a[0], a[2], 0, instances);
            } else {
                Draw(false, a[0], 0, a[2], instances);
            }
            draw_indirect_args = 0;
            break;
        }

        case Op::DispatchDirect:
            ++stats.dispatches;
            FindUserData(body[0], body[1], body[2]);
            if (renderer) {
                if (Piped()) {
                    if (auto it = t_dispatch_args.find(reinterpret_cast<u64>(body)); it != t_dispatch_args.end()) {
                        const u64 args = it->second;
                        t_dispatch_args.erase(it);
                        PipeFn([this, args] {
                            if (renderer->DispatchIndirect(pipe->regs, args)) {
                                return;
                            }
                            renderer->WaitIdle();
                            const u32* a = reinterpret_cast<const u32*>(args);
                            renderer->Dispatch(pipe->regs, a[0], a[1], a[2]);
                        });
                        break;
                    }
                    PipeWords(kPipeDispatch, body, 3);
                    break;
                }
                if (auto it = t_dispatch_args.find(reinterpret_cast<u64>(body)); it != t_dispatch_args.end()) {
                    const u64 args = it->second;
                    t_dispatch_args.erase(it);
                    if (renderer->DispatchIndirect(regs, args)) {
                        break;
                    }
                    SyncRenderer(); // no GPU-side counts: read them once they are there
                    const u32* a = reinterpret_cast<const u32*>(args);
                    renderer->Dispatch(regs, a[0], a[1], a[2]);
                    break;
                }
                renderer->Dispatch(regs, body[0], body[1], body[2]);
            }
            break;
        case Op::DispatchIndirect: {
            ++stats.dispatches;
            const u32* a = IndirectArgs(body[0], 3);
            if (a) {
                FindUserData(a[0], a[1], a[2]);
            }
            // The group counts are usually written by an earlier dispatch the
            // host GPU has not run yet (Driveclub's particle sort sizes its
            // passes from the particle count): the GPU reads them itself.
            if (Piped()) {
                if (a) {
                    PipeFn([this, a] {
                        if (renderer->DispatchIndirect(pipe->regs, reinterpret_cast<u64>(a))) {
                            return;
                        }
                        if (u64(a[0]) * a[1] * a[2] > kMaxIndirectGroups) {
                            DC_WARN("gpu: skipping indirect dispatch %ux%ux%u", a[0], a[1], a[2]);
                            return;
                        }
                        renderer->Dispatch(pipe->regs, a[0], a[1], a[2]);
                    });
                }
                break;
            }
            if (renderer && a && renderer->DispatchIndirect(regs, reinterpret_cast<u64>(a))) {
                break;
            }
            if (a && u64(a[0]) * a[1] * a[2] > kMaxIndirectGroups) {
                DC_WARN("gpu: skipping indirect dispatch %ux%ux%u", a[0], a[1], a[2]);
                break;
            }
            if (renderer && a) {
                renderer->Dispatch(regs, a[0], a[1], a[2]);
            }
            break;
        }

        default:
            { // once per opcode: a skipped packet can drop draws silently
                static std::atomic<u64> seen[4]{};
                const u64 bit = 1ull << (op & 63);
                if (!(seen[op >> 6].fetch_or(bit) & bit)) {
                    DC_INFO("gpu: CP%s: unhandled PM4 opcode %#x (%u dwords)", compute ? " compute" : "", op, body_n);
                }
            }
            break;
        }
        if (trace && (op == Op::EventWriteEop || op == Op::ReleaseMem || op == Op::EventWriteEos ||
                      op == Op::WriteData || op == Op::DmaData || op == Op::WaitRegMem)) {
            DC_INFO("gpu: CP%s %#x: %08x %08x %08x %08x %08x %08x", compute ? " compute" : "", op, body[0],
                    body_n > 1 ? body[1] : 0, body_n > 2 ? body[2] : 0, body_n > 3 ? body[3] : 0,
                    body_n > 4 ? body[4] : 0, body_n > 5 ? body[5] : 0);
        }
        p += body_n + 1;
    }
}

void CommandProcessor::DmaCopy(const u32 body[6], bool compute) {
    const u32 ctl = body[0];
    const u32 cmd = body[5];
    const u32 bytes = cmd & 0x1FFFFF;
    const u32 src_sel = (ctl >> 29) & 3;
    const u32 dst_sel = (ctl >> 20) & 3;
    u8* gds = renderer ? renderer->Gds() : nullptr;
    const u64 dst_off = body[3] & (Gpu::kGdsPartition - 1);
    u8* dst = dst_sel == 1
                  ? (gds && dst_off + bytes <= Gpu::kGdsPartition ? gds + dst_off : nullptr)
                  : Ptr<u8>(body[3], body[4]);
    if (!dst || bytes == 0 || (dst_sel != 1 && !Mapped(dst, bytes)) ||
        (src_sel != 1 && src_sel != 2 && !Mapped(Ptr<const u8>(body[1], body[2]), bytes))) {
        if (bytes) {
            DC_WARN("gpu: DMA_DATA to/from unmapped memory skipped (ctl %#x dst %p, %u bytes)", ctl, dst, bytes);
        }
        return;
    }
    if (src_sel != 2) {
        const u64 t = LtOn() ? LtNow() : 0;
        SyncRenderer(); // the GPU may still be producing what we copy
        if (t) {
            DC_INFO("lt %.0f dma sync %.0f us (ctl %#x %u bytes)%s", LtUs(LtNow()), (LtNow() - t) / 1e3, ctl,
                    bytes, compute ? " (compute)" : "");
        }
    }
    HistAdd(src_sel == 2 ? 'F' : 'C', reinterpret_cast<u64>(dst), bytes,
            t_ring ? t_ring->index : 99);
    if (src_sel == 2) {
        for (u32 k = 0; k + 4 <= bytes; k += 4) {
            std::memcpy(dst + k, &body[1], 4);
        }
    } else {
        const u64 src_off = body[1] & (Gpu::kGdsPartition - 1);
        const u8* src = src_sel == 1
                            ? (gds && src_off + bytes <= Gpu::kGdsPartition ? gds + src_off : nullptr)
                            : Ptr<const u8>(body[1], body[2]);
        if (src) {
            std::memmove(dst, src, bytes);
        }
    }
}

void CommandProcessor::PipeWords(u32 kind, const u32* payload, u32 n, const u32* extra, u32 extra_n) {
    Pipe::Batch& b = pipe->cur;
    b.words.push_back(kind | ((n + extra_n) << 8));
    b.words.insert(b.words.end(), payload, payload + n);
    if (extra_n) {
        b.words.insert(b.words.end(), extra, extra + extra_n);
    }
    if (kind == kPipeDraw && ++b.draws >= kPipeBatchDraws) {
        PipeFlush();
    } else if (b.words.size() > 64 * 1024) {
        PipeFlush();
    }
}

void CommandProcessor::PipeRegs(u32 index, const u32* values, u32 count) {
    if (!count) {
        return;
    }
    Pipe::Batch& b = pipe->cur;
    b.words.push_back(kPipeRegs | ((count + 1) << 8));
    b.words.push_back(index);
    b.words.insert(b.words.end(), values, values + count);
}

void CommandProcessor::PipeFn(std::function<void()> fn) {
    const u32 index = static_cast<u32>(pipe->cur.fns.size());
    pipe->cur.fns.push_back(std::move(fn));
    PipeWords(kPipeFn, &index, 1);
}

void CommandProcessor::PipeFlush() {
    if (pipe->cur.words.empty()) {
        return;
    }
    {
        std::unique_lock lk(pipe->lock);
        pipe->done_cv.wait(lk, [&] { return pipe->queue.size() < kPipeMaxQueued; });
        pipe->queue.push_back(std::move(pipe->cur));
        if (!pipe->spare.empty()) {
            pipe->cur = std::move(pipe->spare.back());
            pipe->spare.pop_back();
        } else {
            pipe->cur = {};
        }
    }
    pipe->cv.notify_one();
}

void CommandProcessor::PipeDrain() {
    PipeFlush();
    std::unique_lock lk(pipe->lock);
    pipe->done_cv.wait(lk, [&] { return pipe->queue.empty() && !pipe->busy; });
}

void CommandProcessor::PipeWorker(std::stop_token stop) {
    pthread_setname_np("GPU gfx render");
    Core::GuestProfile::RegisterThread("CPr");
    if (!std::getenv("DC_NO_QOS")) {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
    t_ring = nullptr; // SyncRenderer & co. act directly here
    t_regs = &pipe->regs;
    Gpu::t_queue = 0;
    for (;;) {
        Pipe::Batch b;
        {
            std::unique_lock lk(pipe->lock);
            pipe->cv.wait(lk, [&] { return !pipe->queue.empty() || stop.stop_requested(); });
            if (stop.stop_requested()) {
                return;
            }
            b = std::move(pipe->queue.front());
            pipe->queue.pop_front();
            pipe->busy = true;
        }
        pipe->done_cv.notify_all(); // room for the ring thread
        PipeReplay(b);
        b.words.clear();
        b.fns.clear();
        b.draws = 0;
        {
            std::scoped_lock lk(pipe->lock);
            pipe->busy = false;
            if (pipe->spare.size() < kPipeMaxQueued) {
                pipe->spare.push_back(std::move(b));
            }
        }
        pipe->done_cv.notify_all();
    }
}

void CommandProcessor::PipeReplay(Pipe::Batch& b) {
    RegFile& r = pipe->regs;
    const u32* w = b.words.data();
    const u32* end = w + b.words.size();
    while (w < end) {
        const u32 kind = *w & 0xFF;
        const u32 n = *w >> 8;
        const u32* p = w + 1;
        w = p + n;
        switch (kind) {
        case kPipeRegs:
            std::memcpy(r.Data() + p[0], p + 1, (n - 1) * 4);
            break;
        case kPipeDraw: {
            DrawCall dc;
            std::memcpy(&dc, p, sizeof(dc));
            renderer->Draw(r, dc);
            break;
        }
        case kPipeDispatch:
            renderer->Dispatch(r, p[0], p[1], p[2]);
            break;
        case kPipeCtx:
            renderer->OnContextRegs(p, n);
            break;
        case kPipeMem: {
            const u64 a = u64(p[0]) | (u64(p[1]) << 32);
            std::memcpy(reinterpret_cast<void*>(a), p + 3, (n - 3) * 4);
            if (p[2]) {
                renderer->NoteConstDump(a, (n - 3) * 4);
            }
            break;
        }
        case kPipeFn:
            b.fns[p[0]]();
            break;
        default:
            DC_ERROR("gpu: pipe record kind %u", kind);
            return;
        }
    }
}

CommandProcessor& Instance() {
    static CommandProcessor cp;
    return cp;
}

} // namespace Gpu::Liverpool
