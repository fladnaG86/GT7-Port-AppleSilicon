#pragma once

// Background Metal shader compilation.
//
// Metal compiles MSL, and again per pipeline, synchronously and slowly
// (seconds for large shaders on AMD). The renderer hands that work to a few
// worker threads so the GPU thread keeps going; Metal's compiler service runs
// the requests in parallel.

#import <Metal/Metal.h>

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
    // compile. Thread-safe.
    id<MTLFunction> Function(u64 key, const std::string& src, u64 address);

    // Whether Function(key, ...) is already cached (it then ignores `src`).
    // Entries are never removed, so a true answer stays true.
    bool Has(u64 key);

    // Runs `job` on a worker thread. Urgent jobs (the GPU thread is about to
    // wait for them) go before every queued normal job, and one worker serves
    // only them, so they never sit behind long pipeline compiles. `key`
    // (0: none) names the job for Touch and RunNow.
    void Submit(std::function<void()> job, bool urgent = false, u64 key = 0);

    // A queued job `key` is wanted again (its draw was skipped once more):
    // normal jobs run most recently wanted first, so what is on screen now
    // compiles before what was on screen long ago. DC_COMPILE_FIFO=1: FIFO.
    void Touch(u64 key);

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

private:
    struct Job {
        u64 key = 0;
        u64 stamp = 0; // Touch: newest first
        std::function<void()> fn;
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
    u64 next_stamp = 0;
    std::unordered_map<u64, u64> touched; // queued normal job key -> its newest stamp

    std::mutex fn_lock;
    std::unordered_map<u64, id<MTLFunction>> functions;
};

} // namespace Gpu::Metal
