#pragma once

// The guest GPU front end: executes PM4 command streams the game submits.
//
// Submissions are queued and run on a dedicated GPU thread, like the real
// command processor, so the game keeps running while "the GPU" works. Sync
// packets (fences, waits, memory writes, interrupts) are executed here;
// draws and dispatches are handed to a Gpu::Renderer.

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "gpu/liverpool/pm4.h"
#include "gpu/liverpool/registers.h"

namespace Gpu {
class Renderer;
}

namespace Gpu::Liverpool {

class CommandProcessor {
public:
    // Interrupt sources the game can listen to (sceGnmAddEqEvent ids).
    static constexpr u32 kIrqGfxEop = 0x40;
    // kIrqCompute + vqid: end of pipe on that compute queue.
    static constexpr u32 kIrqCompute = 0x100;

    struct Stats {
        std::atomic<u64> packets{0};
        std::atomic<u64> draws{0};
        std::atomic<u64> dispatches{0};
        std::atomic<u64> submits{0};
    };

    CommandProcessor();
    ~CommandProcessor();

    // One draw command buffer and the constant command buffer that feeds it
    // (run by the constant engine, CE, alongside the draw engine, DE).
    struct CmdBuffers {
        std::span<const u32> dcb;
        std::span<const u32> ccb;
    };

    // Graphics ring. `after` runs on the GPU thread once the buffers are done
    // (used for flips).
    void SubmitGraphics(std::vector<CmdBuffers> bufs, std::function<void()> after = {});
    // Compute ring `queue` (1-based Gnm vqid).
    void SubmitCompute(u32 queue, std::span<const u32> acb);

    // Blocks until every queued submission has executed.
    void WaitIdle();
    // WaitIdle with a limit; false if some ring was still busy after `ms`.
    bool WaitIdleFor(int ms);
    // Frame replay: waits (at most `ms`) until every ring is idle or blocked in
    // a WAIT_REG_MEM, i.e. has consumed all it can without new memory.
    bool WaitQuiescentFor(int ms);

    // Frame capture/replay (capture.cpp): the state that carries over from
    // one frame to the next (registers, constant engine, ring state). Only
    // meaningful while idle.
    std::vector<u8> SaveState();
    void LoadState(const std::vector<u8>& state);

    void SetInterruptHandler(std::function<void(u32 irq)> handler) { on_irq = std::move(handler); }
    void SetRenderer(Gpu::Renderer* r) { renderer = r; }
    Gpu::Renderer* GetRenderer() const { return renderer; }

    const Stats& GetStats() const { return stats; }
    const RegFile& Regs() const { return regs; }

private:
    struct Job {
        std::vector<CmdBuffers> buffers;
        std::function<void()> after;
    };

    // Constant engine: fills its 48 KB RAM from the CCB and dumps it to
    // memory (descriptor tables) for the DE. It runs ahead of the DE, synced
    // through two counters; here it runs on demand whenever the DE waits.
    struct ConstantEngine {
        std::vector<std::span<const u32>> stack; // CCB + IB_CONST nesting
        u64 ce_counter = 0;
        u64 de_counter = 0;
        std::vector<u32> ram = std::vector<u32>(0xC000 / 4, 0);
    };

    // One hardware ring: the graphics ring or a compute queue. Rings run in
    // parallel, as on the GPU, so a wait on one can be satisfied by another.
    struct Ring {
        u32 index = 0; // 0 = graphics, n = compute vqid n
        std::mutex lock;
        std::condition_variable cv;
        std::condition_variable idle_cv;
        std::deque<Job> jobs;
        bool busy = false;
        RegFile regs;           // compute rings: their own SH state
        std::atomic<bool> waiting{false}; // inside a WAIT_REG_MEM on memory
        u64 indirect_base = 0;  // SET_BASE
        std::jthread thread;
    };

    Ring& GetRing(u32 index);
    void Push(u32 ring, Job job);
    void Run(Ring& ring, std::stop_token stop);
    void Execute(std::span<const u32> cmds, bool compute, int depth);
    void Interrupt(u32 irq);
    void SetRegs(u32 index, const u32* values, u32 count);
    void Draw(bool indexed, u32 count, u32 index_offset, u32 first_vertex, u32 instances);
    RegFile& CurRegs();
    // Ring and registers of the ring the calling thread executes.
    // Never inlined: Apple clang 21 sometimes passes a thread-local's TLV
    // descriptor in a register other than rdi while rdi is live, and the read
    // then returns garbage (tlvcheck.py finds such call sites in a binary).
    static Ring*& TRing();
    static Liverpool::RegFile*& TRegs();
    const u32* IndirectArgs(u32 offset, u32 dwords);
    void SyncRenderer();
    // Runs `fn` once the renderer has executed everything recorded so far.
    void AfterGpu(std::function<void()> fn);
    // AfterGpu for a fence/label write (`data_sel` as in EVENT_WRITE_EOP),
    // remembered as pending until it lands; then `irq` if non-zero.
    void FenceWrite(u32 data_sel, u64 addr, u32 lo, u32 hi, u32 irq);
    // Value a recorded but not yet executed fence write will put at `addr`.
    bool PendingFence(u64 addr, u32* value);
    void ClearState();
    // Runs the CE until `done()` holds, it blocks on the DE, or its CCB ends.
    void RunConstantEngine(const std::function<bool()>& done);

    RegFile regs;
    ConstantEngine ce;
    // Draw state set by packets rather than registers.
    u64 index_base = 0;
    u32 index_size = 2;
    u32 num_instances = 1;
    void FindUserData(u32 x, u32 y, u32 z); // DC_FIND_UD
    u64 draw_indirect_args = 0; // set around Draw() for DRAW_(INDEX_)INDIRECT
    bool warned_indirect = false;
    bool trace = false; // DC_CP_TRACE
    // Flip label requested by the frame being executed; written once it is shown.
    struct FlipLabel {
        u64 addr = 0;
        u32 value = 0;
        bool interrupt = false;
    } flip_label;
    Gpu::Renderer* renderer = nullptr;
    Stats stats;
    std::function<void(u32)> on_irq;

    // The host GPU runs everything recorded so far in order, so a ring that
    // waits for a value one of these writes will produce only waits for work
    // that is already ahead of it: the wait is satisfied without a round trip.
    std::mutex fence_lock;
    std::unordered_map<u64, u32> pending_fences; // dword address -> value

    std::mutex rings_lock;
    std::vector<std::unique_ptr<Ring>> rings;

    // Graphics ring pipeline (DC_CP_PIPE=1). The ring thread only parses PM4:
    // register writes, draws and every memory side effect are recorded in
    // order into batches, and a second thread replays them against its own
    // register file and calls the renderer. Parsing and translation then
    // overlap instead of taking turns on one thread.
    struct Pipe {
        struct Batch {
            std::vector<u32> words; // [kind | dwords << 8] payload...
            std::vector<std::function<void()>> fns;
            u32 draws = 0;
        };
        std::mutex lock;
        std::condition_variable cv;      // worker: a batch arrived
        std::condition_variable done_cv; // ring thread: room, or drained
        std::deque<Batch> queue;
        std::vector<Batch> spare;
        bool busy = false; // worker is replaying a batch
        Batch cur;         // being recorded (ring thread only)
        RegFile regs;      // the worker's registers
        // Fence writes recorded into batches the renderer has not seen yet
        // (guarded by fence_lock): the graphics ring may treat its own as
        // pending, other rings only once the renderer has them.
        std::unordered_map<u64, u32> pending;
        std::jthread thread;
    };
    std::unique_ptr<Pipe> pipe;
    // True on the graphics ring thread when it records into the pipe.
    bool Piped() const { return pipe && renderer && TRing() && TRing()->index == 0; }
    void PipeWords(u32 kind, const u32* payload, u32 n, const u32* extra = nullptr, u32 extra_n = 0);
    void PipeRegs(u32 index, const u32* values, u32 count);
    void PipeFn(std::function<void()> fn);
    void PipeFlush(); // hands the recorded batch to the worker
    void PipeDrain(); // PipeFlush, then waits until the worker has replayed everything
    void PipeWorker(std::stop_token stop);
    void PipeReplay(Pipe::Batch& b);
    // DMA_DATA executed by the CP itself (memory, GDS or immediate data).
    void DmaCopy(const u32 body[6], bool compute);
};

CommandProcessor& Instance();

} // namespace Gpu::Liverpool
