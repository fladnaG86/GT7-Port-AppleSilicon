#include "gpu/metal/mtl_compiler.h"

#include <pthread.h>
#include <pthread/qos.h>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>

#include "common/log.h"

// Set by the game log hook once GT7 loads a course (DC_MSL_SALT).
std::atomic<bool> g_course_loaded{false};

namespace Gpu::Metal {

bool ShaderCompiler::StatsOn() {
    static const bool on = std::getenv("DC_COMPILE_STATS") != nullptr;
    return on;
}

ShaderCompiler::ShaderCompiler(id<MTLDevice> device, u32 count) : device(device) {
    for (u32 k = 0; k < count; ++k) {
        threads.emplace_back([this] { Worker(false); });
    }
    threads.emplace_back([this] { Worker(true); });
}

ShaderCompiler::~ShaderCompiler() {
    {
        std::scoped_lock lk(lock);
        stop = true;
    }
    cv.notify_all();
    for (auto& t : threads) {
        t.join();
    }
}

void ShaderCompiler::Submit(std::function<void()> job, bool urgent, u64 key, bool background) {
    // DC_NO_URGENT_SHADERS: one FIFO queue for everything (A/B).
    static const bool no_urgent = std::getenv("DC_NO_URGENT_SHADERS") != nullptr;
    urgent = urgent && !no_urgent;
    {
        std::scoped_lock lk(lock);
        (urgent ? urgent_jobs : jobs)
            .push_back(Job{key, ++next_stamp, std::move(job), background, StatsOn() ? NowNs() : 0});
        if (key && !urgent) {
            touched[key] = next_stamp;
        }
    }
    // Wake everyone: a waiting worker may be the urgent-only one.
    cv.notify_all();
}

void ShaderCompiler::Touch(u64 key) {
    static const bool fifo = std::getenv("DC_COMPILE_FIFO") != nullptr;
    if (fifo || !key) {
        return;
    }
    std::scoped_lock lk(lock);
    if (auto it = touched.find(key); it != touched.end()) {
        it->second = ++next_stamp;
    }
}

bool ShaderCompiler::Promote(u64 key) {
    if (!key) {
        return false;
    }
    {
        std::scoped_lock lk(lock);
        auto it = std::find_if(jobs.begin(), jobs.end(), [&](const Job& j) { return j.key == key; });
        if (it == jobs.end()) {
            return false;
        }
        urgent_jobs.push_back(std::move(*it));
        jobs.erase(it);
        touched.erase(key);
    }
    cv.notify_all();
    return true;
}

bool ShaderCompiler::RunNow(u64 key) {
    static const bool off = std::getenv("DC_COMPILE_NO_STEAL") != nullptr;
    if (off || !key) {
        return false;
    }
    std::function<void()> job;
    {
        std::scoped_lock lk(lock);
        for (auto* q : {&urgent_jobs, &jobs}) {
            for (auto it = q->begin(); it != q->end(); ++it) {
                if (it->key == key) {
                    job = std::move(it->fn);
                    q->erase(it);
                    touched.erase(key);
                    break;
                }
            }
            if (job) {
                break;
            }
        }
    }
    if (!job) {
        return false;
    }
    @autoreleasepool {
        job();
    }
    return true;
}

void ShaderCompiler::BeginWait() {
    // Without RunNow a waited-for normal job needs the workers: never pause them.
    static const bool off = std::getenv("DC_COMPILE_NO_PAUSE") != nullptr ||
                            std::getenv("DC_COMPILE_NO_STEAL") != nullptr;
    if (off) {
        return;
    }
    std::scoped_lock lk(lock);
    if (!waiters++ && StatsOn()) {
        wait_t0 = NowNs();
    }
}

void ShaderCompiler::EndWait() {
    // Without RunNow a waited-for normal job needs the workers: never pause them.
    static const bool off = std::getenv("DC_COMPILE_NO_PAUSE") != nullptr ||
                            std::getenv("DC_COMPILE_NO_STEAL") != nullptr;
    if (off) {
        return;
    }
    {
        std::scoped_lock lk(lock);
        if (!--waiters && StatsOn()) {
            stats.paused_ns += NowNs() - wait_t0;
        }
    }
    cv.notify_all();
}

size_t ShaderCompiler::Queued() {
    std::scoped_lock lk(lock);
    return jobs.size() + urgent_jobs.size();
}

void ShaderCompiler::Worker(bool urgent_only) {
    pthread_setname_np(urgent_only ? "GPU shader compiler (urgent)" : "GPU shader compiler");
    static const bool fifo = std::getenv("DC_COMPILE_FIFO") != nullptr;
    // Prewarm jobs run at utility QoS only once a course has loaded: before
    // that nothing competes, and utility QoS puts the compiles (Metal's
    // compiler service inherits it) on the efficiency cores. Cold prewarm,
    // M1 Max, pipelines done at ~355 s: utility 4 / 8 threads 2088 / 1390,
    // default QoS 4 / 8 threads 2736 / 3994; menu stayed at 60 fps.
    // DC_PREWARM_QOS=default: never utility; =utility: always (the old way).
    static const int qos_mode = [] {
        const char* q = std::getenv("DC_PREWARM_QOS");
        if (q && std::strcmp(q, "default") == 0) return 0;
        if (q && std::strcmp(q, "utility") == 0) return 2;
        return 1;
    }();
    while (true) {
        std::function<void()> job;
        bool background = false;
        u64 submit_ns = 0;
        {
            std::unique_lock lk(lock);
            cv.wait(lk, [&] {
                return stop || !urgent_jobs.empty() ||
                       (!urgent_only && !waiters && !jobs.empty());
            });
            if (stop) {
                return;
            }
            if (!urgent_jobs.empty()) {
                submit_ns = urgent_jobs.front().submit_ns;
                job = std::move(urgent_jobs.front().fn);
                urgent_jobs.pop_front();
            } else {
                auto best = jobs.begin();
                if (!fifo) {
                    u64 best_stamp = 0;
                    for (auto it = jobs.begin(); it != jobs.end(); ++it) {
                        auto t = it->key ? touched.find(it->key) : touched.end();
                        const u64 st = t != touched.end() ? t->second : it->stamp;
                        if (st > best_stamp) {
                            best_stamp = st;
                            best = it;
                        }
                    }
                }
                if (best->key) {
                    touched.erase(best->key);
                }
                background = best->background &&
                             (qos_mode == 2 ||
                              (qos_mode == 1 && g_course_loaded.load(std::memory_order_relaxed)));
                submit_ns = best->submit_ns;
                job = std::move(best->fn);
                jobs.erase(best);
            }
        }
        if (background) {
            pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
        }
        const u64 job_t0 = StatsOn() ? NowNs() : 0;
        if (submit_ns) {
            stats.queue_ns += job_t0 - submit_ns;
        }
        @autoreleasepool {
            job();
        }
        if (job_t0) {
            stats.job_ns += NowNs() - job_t0;
            ++stats.job_n;
        }
        if (background) {
            pthread_set_qos_class_self_np(QOS_CLASS_DEFAULT, 0);
        }
    }
}

bool ShaderCompiler::Has(u64 key) {
    std::scoped_lock lk(fn_lock);
    return functions.count(key) != 0;
}

id<MTLFunction> ShaderCompiler::Function(u64 key, const std::string& src, u64 address, bool prewarm) {
    {
        std::scoped_lock lk(fn_lock);
        if (auto it = functions.find(key); it != functions.end()) {
            return it->second;
        }
    }
    MTLCompileOptions* opts = [MTLCompileOptions new];
    // Relaxed keeps Inf/NaN semantics but lets the compiler fuse and reorder:
    // ~5% less GPU time per frame in a race. DC_MATH_MODE=safe|fast for A/B.
    static const MTLMathMode math_mode = [] {
        const char* m = std::getenv("DC_MATH_MODE");
        if (m && std::strcmp(m, "safe") == 0) {
            return MTLMathModeSafe;
        }
        if (m && std::strcmp(m, "fast") == 0) {
            return MTLMathModeFast;
        }
        return MTLMathModeRelaxed;
    }();
    opts.mathMode = math_mode;
    // 3.1 only where needed (texture atomics): other shaders keep their cached binaries.
    opts.languageVersion = src.find(".atomic_fetch_") != std::string::npos ? MTLLanguageVersion3_1
                                                                           : MTLLanguageVersion3_0;
    NSError* err = nil;
    // DC_MSL_SALT=<text> (benchmarks): from course load on (DC_GAME_LOG; with
    // DC_MSL_SALT_EARLY=1 from the start), a comment line before every source,
    // so Metal's shader cache misses as on a first run after a translator
    // change. Not from the start by default: the slow cold startup throws the
    // key replay off.
    static const char* salt_env = std::getenv("DC_MSL_SALT");
    static const bool salt_early = std::getenv("DC_MSL_SALT_EARLY") != nullptr;
    const char* salt = salt_env && (prewarm || salt_early || g_course_loaded.load(std::memory_order_relaxed)) ? salt_env
                                                                                                  : nullptr;
    std::string salted = salt ? "// " + std::string(salt) + "\n" + src : std::string();
    // DC_MSL_SALT_CODE=1: the salt also changes main0's code (a volatile store
    // of its hash), so the AIR differs and Metal's backend cache misses too.
    static const bool salt_code = std::getenv("DC_MSL_SALT_CODE") != nullptr;
    if (salt && salt_code) {
        const size_t m = salted.find(" main0(");
        const size_t b = m == std::string::npos ? m : salted.find('{', m);
        if (b != std::string::npos) {
            salted.insert(b + 1, " { volatile uint dc_salt_v = " +
                                     std::to_string(std::hash<std::string>{}(salt) & 0x7fffffffu) +
                                     "u; (void)dc_salt_v; }");
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    id<MTLLibrary> lib = nil;
    {
        Timed timed(stats, stats.lib_ns, stats.lib_n, stats.lib_hit);
        stats.lib_bytes += src.size();
        lib = [device newLibraryWithSource:@((salt ? salted : src).c_str()) options:opts error:&err];
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (secs > 0.5) {
        DC_WARN("gpu: MSL of shader %#llx (key %016llx, %zu KB) took %.1f s to compile", address,
                (unsigned long long)key, src.size() >> 10, secs);
    }
    id<MTLFunction> fn = lib ? [lib newFunctionWithName:@"main0"] : nil;
    if (!fn) {
        DC_ERROR("gpu: MSL compile failed for shader %#llx: %s", address,
                 err ? err.localizedDescription.UTF8String : "?");
    }
    std::scoped_lock lk(fn_lock);
    return functions.emplace(key, fn).first->second;
}

} // namespace Gpu::Metal
