#include "core/guest_profile.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <mach/mach.h>

#include "common/log.h"
#include "common/types.h"
#include "core/loader/linker.h"
#include "hle/kernel/kernel.h"

namespace Core::GuestProfile {

namespace {

constexpr int kFramesKept = 10;
constexpr u64 kMaxFrameSpan = 1 << 20;
constexpr u32 kMaxSamples = 4096;
constexpr auto kPeriod = std::chrono::milliseconds(10);

struct Sample {
    u64 frames[kFramesKept];
};

struct ProfiledThread {
    mach_port_t mach;
    std::string name;
    std::vector<Sample> samples;
    std::atomic<u32> count{0};
};

std::mutex g_lock;
std::vector<ProfiledThread*> g_threads;
std::atomic<u32> g_state_errors{0};

bool IsGuestCode(u64 pc) {
    constexpr u64 kHostImage = 0x700000000000; // our own code, for host threads
    return (pc >= 0x400000 && pc < 0x10000000) ||
           (pc >= kHostImage && pc < kHostImage + 0x4000000);
}

// Keeps the guest return addresses nearest the interrupted code: a thread
// blocked inside the host still shows which guest call it is in. The thread is
// suspended meanwhile. Signals would not do: they wake host condition waits,
// and the game does not always re-check its predicate.
void SampleThread(ProfiledThread* t) {
    const u32 n = t->count.load(std::memory_order_relaxed);
    if (n >= kMaxSamples || thread_suspend(t->mach) != KERN_SUCCESS) {
        return;
    }
    x86_thread_state64_t ss{};
    mach_msg_type_number_t state_count = x86_THREAD_STATE64_COUNT;
    const kern_return_t got = thread_get_state(
        t->mach, x86_THREAD_STATE64, reinterpret_cast<thread_state_t>(&ss), &state_count);
    if (got == KERN_SUCCESS) {
        Sample& s = t->samples[n];
        int kept = 0;
        std::fill(std::begin(s.frames), std::end(s.frames), u64{0});
        if (IsGuestCode(ss.__rip)) {
            s.frames[kept++] = ss.__rip;
        }
        u64 fp = ss.__rbp;
        for (int depth = 0; depth < 32 && kept < kFramesKept; ++depth) {
            if (fp < ss.__rsp || fp > ss.__rsp + kMaxFrameSpan || (fp & 7)) {
                break;
            }
            const auto* frame = reinterpret_cast<const u64*>(fp);
            if (IsGuestCode(frame[1])) {
                s.frames[kept++] = frame[1];
            }
            if (frame[0] <= fp) {
                break;
            }
            fp = frame[0];
        }
        t->count.store(n + 1, std::memory_order_relaxed);
    } else {
        g_state_errors.fetch_add(1, std::memory_order_relaxed);
    }
    thread_resume(t->mach);
}

void Report() {
    auto* linker = Loader::Linker::Instance();
    DC_INFO("profile: %u failed state reads", g_state_errors.load());
    Hle::Kernel::DumpFibers();
    std::scoped_lock lk(g_lock);
    for (ProfiledThread* t : g_threads) {
        const u32 n = t->count.load();
        std::map<std::vector<u64>, u32> stacks;
        for (u32 i = 0; i < n; ++i) {
            const Sample& s = t->samples[i];
            ++stacks[std::vector<u64>(std::begin(s.frames), std::end(s.frames))];
        }
        std::vector<std::pair<u32, std::vector<u64>>> sorted;
        for (const auto& [frames, count] : stacks) {
            sorted.push_back({count, frames});
        }
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        for (size_t i = 0; i < std::min<size_t>(sorted.size(), t->name == "Job#0" ? 14 : 2); ++i) {
            std::string line;
            for (const u64 pc : sorted[i].second) {
                if (pc) {
                    line += " < " + linker->Describe(pc);
                }
            }
            thread_basic_info_data_t info{};
            mach_msg_type_number_t info_count = THREAD_BASIC_INFO_COUNT;
            thread_info(t->mach, THREAD_BASIC_INFO, reinterpret_cast<thread_info_t>(&info),
                        &info_count);
            DC_INFO("profile %-22s cpu %3d%% %4u/%u%s", t->name.c_str(),
                    info.cpu_usage * 100 / TH_USAGE_SCALE, sorted[i].first, n, line.c_str());
        }
    }
}

} // namespace

void RegisterThread(const char* name) {
    if (!std::getenv("DC_GUEST_PROFILE")) {
        return;
    }
    auto* t = new ProfiledThread();
    t->mach = mach_thread_self();
    t->name = name;
    t->samples.resize(kMaxSamples);
    std::scoped_lock lk(g_lock);
    g_threads.push_back(t);
}

void Start() {
    const char* env = std::getenv("DC_GUEST_PROFILE");
    if (!env) {
        return;
    }
    double start = 0, seconds = 0;
    std::sscanf(env, "%lf,%lf", &start, &seconds);
    std::thread([start, seconds] {
        std::this_thread::sleep_for(std::chrono::duration<double>(start));
        const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
        while (std::chrono::steady_clock::now() < end) {
            {
                std::scoped_lock lk(g_lock);
                for (ProfiledThread* t : g_threads) {
                    SampleThread(t);
                }
            }
            std::this_thread::sleep_for(kPeriod);
        }
        Report();
    }).detach();
}

} // namespace Core::GuestProfile
