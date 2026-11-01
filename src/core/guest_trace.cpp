#include "core/guest_trace.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/ucontext.h>

#include "common/log.h"

namespace Core::GuestTrace {

namespace {

constexpr u8 kPushRbp = 0x55;
constexpr u8 kInt3 = 0xCC;
constexpr size_t kMaxPoints = 32;
constexpr size_t kRingSize = 32;

struct Event {
    std::atomic<u32> point;
    std::atomic<u64> thread;
};

u64 g_base;
size_t g_count;
std::array<u64, kMaxPoints> g_offsets;
std::array<std::atomic<u64>, kMaxPoints> g_hits;
std::array<Event, kRingSize> g_ring;
std::atomic<u64> g_next;

void OnTrap(int, siginfo_t*, void* uc) {
    auto& ss = static_cast<ucontext_t*>(uc)->uc_mcontext->__ss;
    const u64 offset = ss.__rip - 1 - g_base;
    for (size_t i = 0; i < g_count; ++i) {
        if (g_offsets[i] != offset) {
            continue;
        }
        g_hits[i].fetch_add(1, std::memory_order_relaxed);
        Event& e = g_ring[g_next.fetch_add(1, std::memory_order_relaxed) % kRingSize];
        e.point.store(static_cast<u32>(i), std::memory_order_relaxed);
        e.thread.store(reinterpret_cast<u64>(pthread_self()), std::memory_order_relaxed);
        // Runs the `push rbp` the breakpoint replaced.
        ss.__rsp -= sizeof(u64);
        *reinterpret_cast<u64*>(ss.__rsp) = ss.__rbp;
        return;
    }
    signal(SIGTRAP, SIG_DFL);
}

std::string Counts() {
    std::string out;
    char buf[48];
    for (size_t i = 0; i < g_count; ++i) {
        std::snprintf(buf, sizeof(buf), " %llx=%llu", g_offsets[i], g_hits[i].load());
        out += buf;
    }
    return out;
}

void Report() {
    std::string last;
    bool told = false;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::string now = Counts();
        if (now != last) {
            DC_INFO("trace:%s", now.c_str());
            last = std::move(now);
            told = false;
            continue;
        }
        if (told) {
            continue;
        }
        told = true;
        const u64 end = g_next.load();
        for (u64 n = end > kRingSize ? end - kRingSize : 0; n < end; ++n) {
            const Event& e = g_ring[n % kRingSize];
            DC_INFO("trace last: %llx on %#llx", g_offsets[e.point.load()], e.thread.load());
        }
    }
}

} // namespace

void Start(u64 image_base) {
    const char* spec = std::getenv("DC_TRACE");
    if (!spec) {
        return;
    }
    g_base = image_base;
    for (const char* p = spec; *p && g_count < kMaxPoints;) {
        char* end = nullptr;
        const u64 offset = std::strtoull(p, &end, 16);
        p = *end ? end + 1 : end;
        auto* code = reinterpret_cast<u8*>(image_base + offset);
        if (*code != kPushRbp) {
            DC_WARN("trace: %llx is not a push rbp", offset);
            continue;
        }
        g_offsets[g_count++] = offset;
    }
    struct sigaction sa {};
    sa.sa_sigaction = OnTrap;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGTRAP, &sa, nullptr);
    for (size_t i = 0; i < g_count; ++i) {
        auto* code = reinterpret_cast<u8*>(image_base + g_offsets[i]);
        const u64 page = reinterpret_cast<u64>(code) & ~0x3FFFull;
        mprotect(reinterpret_cast<void*>(page), 0x8000, PROT_READ | PROT_WRITE | PROT_EXEC);
        *code = kInt3;
        mprotect(reinterpret_cast<void*>(page), 0x8000, PROT_READ | PROT_EXEC);
    }
    std::thread(Report).detach();
}

} // namespace Core::GuestTrace
