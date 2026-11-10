#pragma once

// Background Metal shader compilation.
//
// Metal compiles MSL, and again per pipeline, synchronously and slowly
// (seconds for large shaders on AMD). The renderer hands that work to a few
// worker threads so the GPU thread keeps going; Metal's compiler service runs
// the requests in parallel.

#import <Metal/Metal.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/types.h"

namespace Gpu::Metal {

class ShaderCompiler {
public:
    ShaderCompiler(id<MTLDevice> device, u32 threads);
    ~ShaderCompiler();

    // The "main0" function of MSL `src`, cached by `key`; nil if it does not
    // compile. Thread-safe. `prewarm`: a pipeline prewarm compile, which
    // DC_MSL_SALT salts at any time (it stands for a cold first run).
    id<MTLFunction> Function(u64 key, const std::string& src, u64 address, bool prewarm = false);

    // Whether Function(key, ...) is already cached (it then ignores `src`).
    // Entries are never removed, so a true answer stays true.
    bool Has(u64 key);

    // Runs `job` on a worker thread. Urgent jobs (the GPU thread is about to
    // wait for them) go before every queued normal job, and one worker serves
    // only them, so they never sit behind long pipeline compiles. `key`
    // (0: none) names the job for Touch and RunNow.
    // `background` (prewarm): once a course has loaded, a normal worker runs
    // it at utility QoS so it yields the cores to the game (DC_PREWARM_QOS).
    void Submit(std::function<void()> job, bool urgent = false, u64 key = 0, bool background = false);

    // A queued job `key` is wanted again (its draw was skipped once more):
    // normal jobs run most recently wanted first, so what is on screen now
    // compiles before what was on screen long ago. DC_COMPILE_FIFO=1: FIFO.
    void Touch(u64 key);

    // Moves queued normal job `key` to the urgent queue (a prewarm job that a
    // draw now needs: the urgent worker takes it next instead of it waiting
    // behind long background compiles). False if it is not queued.
    bool Promote(u64 key);

    // Takes job `key` out of the queues and runs it on the calling thread;
    // false if it is not queued (done, running, or never submitted). The
    // thread about to wait for a job runs it itself rather than wait for a
    // worker to reach it.
    bool RunNow(u64 key);

    // While a thread waits for a job, workers start no new normal job: the
    // compiles they would start compete with the one being waited for in
    // Metal's compiler service. DC_COMPILE_NO_PAUSE=1: off.
    void BeginWait();
    void EndWait();

    size_t Queued();

    // DC_COMPILE_STATS=1: where the time of a pipeline compile goes, summed
    // over all threads since the last perf line (which prints and clears it).
    // "hit": the call took under 5 ms, so Metal's own cache answered.
    struct Stats {
        std::atomic<u64> msl_ns{0}, msl_n{0};                         // GenerateMsl (translator, under Rosetta)
        std::atomic<u64> lib_ns{0}, lib_n{0}, lib_hit{0}, lib_bytes{0}; // newLibraryWithSource (MSL frontend)
        std::atomic<u64> pso_ns{0}, pso_n{0}, pso_hit{0};             // new*PipelineState (AIR -> GPU backend)
        std::atomic<u64> job_ns{0}, job_n{0};                         // workers inside a job (saturation)
        std::atomic<u64> queue_ns{0};                                 // Submit -> a worker takes the job
        std::atomic<u64> paused_ns{0};                                // GPU thread waiting: workers start no normal job
        std::atomic<u32> inflight{0}, peak{0};                        // Metal calls running at once
    };
    Stats stats;
    static bool StatsOn();
    static u64 NowNs() {
        return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count());
    }
    // Times one Metal compile call into (ns, n, hit).
    struct Timed {
        Stats& s;
        std::atomic<u64>&ns, &n, &hit;
        const u64 t0 = NowNs();
        Timed(Stats& s, std::atomic<u64>& ns, std::atomic<u64>& n, std::atomic<u64>& hit) : s(s), ns(ns), n(n), hit(hit) {
            const u32 now = ++s.inflight;
            u32 p = s.peak.load();
            while (now > p && !s.peak.compare_exchange_weak(p, now)) {
            }
        }
        ~Timed() {
            const u64 d = NowNs() - t0;
            --s.inflight;
            ns += d;
            ++n;
            hit += d < 5'000'000;
        }
    };
    u32 Workers() const { return static_cast<u32>(threads.size()); } // with the urgent one

private:
    struct Job {
        u64 key = 0;
        u64 stamp = 0; // Touch: newest first
        std::function<void()> fn;
        bool background = false;
        u64 submit_ns = 0; // DC_COMPILE_STATS
    };
    void Worker(bool urgent_only);

    id<MTLDevice> device;
    std::mutex lock;
    std::condition_variable cv;
    std::deque<Job> jobs;
    std::deque<Job> urgent_jobs;
    std::vector<std::thread> threads;
    bool stop = false;
    u32 waiters = 0;
    u64 wait_t0 = 0; // DC_COMPILE_STATS: when waiters left 0
    u64 next_stamp = 0;
    std::unordered_map<u64, u64> touched; // queued normal job key -> its newest stamp

    std::mutex fn_lock;
    std::unordered_map<u64, id<MTLFunction>> functions;
};

} // namespace Gpu::Metal
