#pragma once

// DC_LABEL_TRACE=<label addr hex>,<start s>,<duration s>: logs a timeline
// ("lt <us> ...") of guest submits, CP jobs, command buffers and writes of one
// label, to see where a frame waits. Times are CLOCK_UPTIME_RAW.

#include <cstdlib>
#include <ctime>

#include <mach/mach.h>

#include "common/types.h"

namespace Gpu {

inline u64 LtNow() {
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

struct LabelTrace {
    u64 addr = 0;
    u64 t0 = 0, t1 = 0;
};

inline const LabelTrace& Lt() {
    static const LabelTrace lt = [] {
        LabelTrace l;
        const char* v = std::getenv("DC_LABEL_TRACE");
        if (!v) {
            return l;
        }
        char* end = nullptr;
        l.addr = std::strtoull(v, &end, 16);
        double start = 60, dur = 5;
        if (end && *end == ',') {
            start = std::strtod(end + 1, &end);
            if (end && *end == ',') {
                dur = std::strtod(end + 1, &end);
            }
        }
        l.t0 = LtNow() + static_cast<u64>(start * 1e9);
        l.t1 = l.t0 + static_cast<u64>(dur * 1e9);
        return l;
    }();
    return lt;
}

inline bool LtOn() {
    const LabelTrace& l = Lt();
    if (!l.t1) {
        return false;
    }
    const u64 n = LtNow();
    return n >= l.t0 && n < l.t1;
}

// Microseconds since the trace window opened.
inline double LtUs(u64 ns) {
    return (static_cast<double>(ns) - static_cast<double>(Lt().t0)) / 1e3;
}

// User and system CPU time of the calling thread, microseconds.
inline void LtThreadCpu(double* user_us, double* sys_us) {
    thread_basic_info_data_t info{};
    mach_msg_type_number_t n = THREAD_BASIC_INFO_COUNT;
    const mach_port_t self = mach_thread_self();
    *user_us = *sys_us = 0;
    if (thread_info(self, THREAD_BASIC_INFO, reinterpret_cast<thread_info_t>(&info), &n) == KERN_SUCCESS) {
        *user_us = info.user_time.seconds * 1e6 + info.user_time.microseconds;
        *sys_us = info.system_time.seconds * 1e6 + info.system_time.microseconds;
    }
    mach_port_deallocate(mach_task_self(), self);
}

// Time the calling thread spent blocked on the renderer lock (trace only).
inline thread_local u64 lt_lock_ns = 0;

// The graphics ring takes the renderer lock once per draw (~15 us each) and
// the compute rings once per dispatch or fence (a few us each). A thread that
// sleeps on the lock costs its owner a kernel wake at unlock and runs again
// tens of microseconds later, far longer than the work it waited for: in a
// GT7 race the graphics thread spent ~15% of its time in that sleep and ~5%
// waking sleepers that lost the lock again. Waiters poll for this long first.
// Every thread polls: with the graphics thread alone the compute rings starve
// and the guest waits for their fences (21.8 vs 24 fps). Polling compute
// rings win the lock between two draws far more often, which is only cheap
// because their dispatches no longer end the open render pass (see
// DeferredDispatch in mtl_renderer.mm). Together: +4-5% fps in a race.
// DC_LOCK_SPIN_US=<n> overrides, 0 sleeps at once (old behavior).
inline u64 LtSpinNs() {
    static const u64 ns = [] {
        const char* e = std::getenv("DC_LOCK_SPIN_US");
        return (e ? std::strtoull(e, nullptr, 0) : 60ull) * 1000;
    }();
    return ns;
}

// Takes `m` the way LtLock does (poll, then sleep).
template <typename M>
inline void LtAcquire(M& m) {
    if (m.try_lock()) {
        return;
    }
    const u64 t = LtNow();
    if (const u64 spin = LtSpinNs()) {
        do {
            for (int k = 0; k < 8; ++k) {
                __builtin_ia32_pause();
            }
            if (m.try_lock()) {
                lt_lock_ns += LtNow() - t;
                return;
            }
        } while (LtNow() - t < spin);
    }
    m.lock();
    lt_lock_ns += LtNow() - t;
}

template <typename M>
struct LtLock {
    M& m;
    explicit LtLock(M& mu) : m(mu) {
        if (m.try_lock()) {
            return;
        }
        const u64 t = LtNow();
        if (const u64 spin = LtSpinNs()) {
            do {
                for (int k = 0; k < 8; ++k) {
                    __builtin_ia32_pause();
                }
                if (m.try_lock()) {
                    lt_lock_ns += LtNow() - t;
                    return;
                }
            } while (LtNow() - t < spin);
        }
        m.lock();
        lt_lock_ns += LtNow() - t;
    }
    ~LtLock() { m.unlock(); }
    LtLock(const LtLock&) = delete;
    LtLock& operator=(const LtLock&) = delete;
};

} // namespace Gpu
