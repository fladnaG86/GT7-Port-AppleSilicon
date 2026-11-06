// SPDX-License-Identifier: GPL-2.0-or-later

#include <unordered_map>
#include <string>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include "hle/pad/pad.h"

#include <atomic>
#include <cstring>

#include "common/log.h"
#include "hle/hle.h"

namespace Hle::Pad {

namespace {

constexpr s32 kPadHandle = 1;
constexpr s32 kErrorInvalidPortType = -2137980926; // 0x80920002
constexpr s32 kErrorInvalidHandle = static_cast<s32>(0x80920003);

s32 scePadInit() {
    DC_INFO("scePadInit()");
    return 0;
}

s32 scePadOpen(s32 userId, s32 type, s32 index, const void* param) {
    if (type != 0) {
        return kErrorInvalidPortType;
    }
    DC_INFO("scePadOpen(userId=%#x, type=%d, index=%d) -> handle %d", userId, type, index, kPadHandle);
    return kPadHandle;
}

s32 scePadOpenExt(s32 userId, s32 type, s32 index, const void* param) {
    if (type != 0) {
        return kErrorInvalidPortType;
    }
    DC_INFO("scePadOpenExt(userId=%#x, type=%d, index=%d) -> handle %d", userId, type, index, kPadHandle);
    return kPadHandle;
}

s32 scePadClose(s32 handle) {
    DC_INFO("scePadClose(handle=%d)", handle);
    return 0;
}

static std::atomic<u32> g_buttons{0};
static std::atomic<u8> g_left_x{128};
static std::atomic<u8> g_left_y{128};
static std::atomic<u8> g_right_x{128};
static std::atomic<u8> g_right_y{128};
static std::atomic<u8> g_l2{0};
static std::atomic<u8> g_r2{0};

// Microseconds, like sceKernelGetProcessTime: games skip samples whose time
// did not move.
static u64 PadTimestamp() {
    using namespace std::chrono;
    static const auto t0 = steady_clock::now();
    return duration_cast<microseconds>(steady_clock::now() - t0).count();
}

// Only the handle scePadOpen gave out is a pad: logiWheel.prx probes with -1
// and, served data, reports a wheel the game then gives player 1.
s32 scePadReadState(s32 handle, OrbisPadData* data) {
    if (handle != kPadHandle) {
        return kErrorInvalidHandle;
    }
    if (data) {
        std::memset(data, 0, sizeof(*data));
        data->buttons = g_buttons.load();
        data->leftStick = {g_left_x.load(), g_left_y.load()};
        data->rightStick = {g_right_x.load(), g_right_y.load()};
        data->analogButtons.l2 = g_l2.load();
        data->analogButtons.r2 = g_r2.load();
        data->orientation[3] = 1.0f;
        data->acceleration[2] = 9.8f;
        data->connected = true;
        data->connectedCount = 1;
        data->timestamp = PadTimestamp();
    }
    return 0;
}

s32 scePadRead(s32 handle, OrbisPadData* data, s32 count) {
    if (handle != kPadHandle) {
        return kErrorInvalidHandle;
    }
    if (data && count > 0) {
        for (s32 i = 0; i < count; ++i) {
            std::memset(&data[i], 0, sizeof(OrbisPadData));
            data[i].buttons = g_buttons.load();
            data[i].leftStick = {g_left_x.load(), g_left_y.load()};
            data[i].rightStick = {g_right_x.load(), g_right_y.load()};
            data[i].analogButtons.l2 = g_l2.load();
            data[i].analogButtons.r2 = g_r2.load();
            data[i].orientation[3] = 1.0f;
            data[i].acceleration[2] = 9.8f;
            data[i].connected = true;
            data[i].connectedCount = 1;
            data[i].timestamp = PadTimestamp();
        }
    }
    return count;
}

s32 scePadGetControllerInformation(s32 handle, OrbisPadControllerInformation* info) {
    if (handle != kPadHandle) {
        return kErrorInvalidHandle;
    }
    if (info) {
        std::memset(info, 0, sizeof(*info));
        info->touchPadInfo.pixelDensity = 1.0f;
        info->touchPadInfo.resX = 1920;
        info->touchPadInfo.resY = 942;
        info->connected = true;
        info->connectedCount = 1;
    }
    return 0;
}

s32 scePadGetExtControllerInformation(s32 handle, void* pInfo) {
    return scePadGetControllerInformation(handle, reinterpret_cast<OrbisPadControllerInformation*>(pInfo));
}

s32 scePadSetTiltCorrectionState(s32 handle, bool bEnable) {
    return 0;
}

s32 scePadSetVibration(s32 handle, const void* param) {
    return 0;
}

s32 scePadResetOrientation(s32 handle) {
    return 0;
}

} // namespace

// DC_PAD_RECORD=<file>: every input with its time since start, as text.
// DC_PAD_REPLAY=<file> plays such a file back with the same timing.
//   B <sec> <hex mask> <0|1>   button
//   T <sec> <right> <value>    analog trigger
//   S <sec> <right> <y> <value> stick axis
static std::chrono::steady_clock::time_point g_input_t0 = std::chrono::steady_clock::now();

static double InputTime() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - g_input_t0).count();
}

// ---- Anchors: game log events (file:line, n-th occurrence) ----
// A recorded input names the latest *rare* event before it (an occurrence
// number <= kRareOccurrence) and its delay after it: "B <sec> <mask> <0|1> @
// <file:line> <n> <delay>". Replay waits for that event, then the delay.
namespace {
constexpr u32 kRareOccurrence = 2;
std::mutex g_anchor_lock;
std::condition_variable g_anchor_cv;
std::unordered_map<std::string, u32> g_anchor_count;           // file:line -> occurrences
std::unordered_map<std::string, double> g_anchor_time;         // "file:line#n" -> time
std::string g_last_anchor;                                     // "file:line n"
double g_last_anchor_t = 0;
} // namespace

static double InputTime();

void OnGameLog(const char* file, int line) {
    const double t = InputTime();
    std::string key = std::string(file) + ":" + std::to_string(line);
    std::scoped_lock lk(g_anchor_lock);
    const u32 n = ++g_anchor_count[key];
    if (n <= kRareOccurrence) {
        g_anchor_time[key + "#" + std::to_string(n)] = t;
        g_last_anchor = key + " " + std::to_string(n);
        g_last_anchor_t = t;
        g_anchor_cv.notify_all();
    }
}

// " @ <file:line> <n> <delay>" for the current time, or "".
static std::string AnchorSuffix() {
    std::scoped_lock lk(g_anchor_lock);
    if (g_last_anchor.empty()) {
        return "";
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), " %.3f", InputTime() - g_last_anchor_t);
    return " @ " + g_last_anchor + buf;
}

// Waits (at most `timeout` s) for the n-th occurrence of `key`; its time, or <0.
static double WaitAnchor(const std::string& key, u32 n, double timeout) {
    const std::string id = key + "#" + std::to_string(n);
    std::unique_lock lk(g_anchor_lock);
    const bool seen = g_anchor_cv.wait_for(lk, std::chrono::milliseconds(static_cast<s64>(timeout * 1000)),
                                           [&] { return g_anchor_time.count(id) != 0; });
    return seen ? g_anchor_time[id] : -1.0;
}

static std::FILE* RecordFile() {
    static std::FILE* f = [] {
        const char* e = std::getenv("DC_PAD_RECORD");
        return e ? std::fopen(e, "w") : nullptr;
    }();
    return f;
}

void SetButton(u32 button, bool pressed) {
    const u32 before = g_buttons.load();
    if (std::FILE* f = RecordFile(); f && ((before & button) != 0) != pressed) {
        std::fprintf(f, "B %.3f %x %d%s\n", InputTime(), button, pressed ? 1 : 0, AnchorSuffix().c_str());
        std::fflush(f);
    }
    if (pressed) {
        g_buttons.fetch_or(button);
    } else {
        g_buttons.fetch_and(~button);
    }
}

void SetStick(bool right, u8 x, u8 y) {
    if (std::FILE* f = RecordFile()) {
        const u8 ox = (right ? g_right_x : g_left_x).load(), oy = (right ? g_right_y : g_left_y).load();
        if (ox != x) {
            std::fprintf(f, "S %.3f %d 0 %u%s\n", InputTime(), right, x, AnchorSuffix().c_str());
        }
        if (oy != y) {
            std::fprintf(f, "S %.3f %d 1 %u%s\n", InputTime(), right, y, AnchorSuffix().c_str());
        }
        std::fflush(f);
    }
    if (right) {
        g_right_x.store(x);
        g_right_y.store(y);
    } else {
        g_left_x.store(x);
        g_left_y.store(y);
    }
}

void SetStickAxis(bool right, bool y, u8 value) {
    if (std::FILE* f = RecordFile()) {
        std::fprintf(f, "S %.3f %d %d %u%s\n", InputTime(), right, y, value, AnchorSuffix().c_str());
        std::fflush(f);
    }
    (right ? (y ? g_right_y : g_right_x) : (y ? g_left_y : g_left_x)).store(value);
}

void SetTrigger(bool right, u8 value) {
    if (std::FILE* f = RecordFile(); f && (right ? g_r2 : g_l2).load() != value) {
        std::fprintf(f, "T %.3f %d %u%s\n", InputTime(), right, value, AnchorSuffix().c_str());
        std::fflush(f);
    }
    (right ? g_r2 : g_l2).store(value);
    SetButton(right ? kR2 : kL2, value > 30);
}

// DC_AUTOPRESS=<sec>[:<hex mask>],...: presses Cross (or that button) at
// those times after start (unattended runs that need to get past "Start
// Event", or to switch the camera).
static void StartAutoPress() {
    const char* e = std::getenv("DC_AUTOPRESS");
    if (!e) {
        return;
    }
    // DC_AUTOPRESS_BUTTON=<hex mask> (default Cross), DC_AUTOPRESS_HOLD=<ms>.
    const char* be = std::getenv("DC_AUTOPRESS_BUTTON");
    const u32 button = be ? static_cast<u32>(std::strtoul(be, nullptr, 16)) : 0x4000;
    const char* he = std::getenv("DC_AUTOPRESS_HOLD");
    const s64 hold = he ? std::strtoll(he, nullptr, 10) : 250;
    std::vector<std::pair<double, u32>> presses;
    for (const char* p = e; *p;) {
        char* q = nullptr;
        const double t = std::strtod(p, &q);
        if (q == p) break;
        u32 mask = button;
        if (*q == ':') {
            mask = static_cast<u32>(std::strtoul(q + 1, &q, 16));
        }
        presses.push_back({t, mask});
        p = *q == ',' ? q + 1 : q;
    }
    std::thread([presses, hold] {
        const auto t0 = std::chrono::steady_clock::now();
        for (auto [t, mask] : presses) {
            std::this_thread::sleep_until(t0 + std::chrono::milliseconds(static_cast<s64>(t * 1000)));
            DC_INFO("pad: auto-press %#x at %.1f s", mask, t);
            SetButton(mask, true);
            std::this_thread::sleep_for(std::chrono::milliseconds(hold));
            SetButton(mask, false);
        }
    }).detach();
}

static void StartReplay() {
    const char* e = std::getenv("DC_PAD_REPLAY");
    if (!e) {
        return;
    }
    struct Event {
        char kind;
        double t;
        u32 a, b, c;
        std::string anchor; // file:line, empty = wall-clock time only
        u32 occurrence = 0;
        double delay = 0;
    };
    std::vector<Event> events;
    if (std::FILE* f = std::fopen(e, "r")) {
        char line[256];
        while (std::fgets(line, sizeof(line), f)) {
            Event ev{};
            if (std::sscanf(line, "B %lf %x %u", &ev.t, &ev.a, &ev.b) == 3) {
                ev.kind = 'B';
            } else if (std::sscanf(line, "T %lf %u %u", &ev.t, &ev.a, &ev.b) == 3) {
                ev.kind = 'T';
            } else if (std::sscanf(line, "S %lf %u %u %u", &ev.t, &ev.a, &ev.b, &ev.c) == 4) {
                ev.kind = 'S';
            } else {
                continue;
            }
            if (const char* at = std::strstr(line, " @ ")) {
                char key[160];
                if (std::sscanf(at + 3, "%159s %u %lf", key, &ev.occurrence, &ev.delay) == 3) {
                    ev.anchor = key;
                }
            }
            events.push_back(ev);
        }
        std::fclose(f);
    }
    DC_INFO("pad: replaying %zu inputs from %s", events.size(), e);
    std::thread([events = std::move(events)] {
        // DC_PAD_ANCHOR_TIMEOUT=<s>: longest wait for an anchor (default 180).
        const double timeout =
            std::getenv("DC_PAD_ANCHOR_TIMEOUT") ? std::strtod(std::getenv("DC_PAD_ANCHOR_TIMEOUT"), nullptr) : 180.0;
        for (const Event& ev : events) {
            double at = ev.t;
            if (!ev.anchor.empty()) {
                const double t = WaitAnchor(ev.anchor, ev.occurrence, timeout);
                if (t < 0) {
                    DC_WARN("pad: replay anchor %s #%u not seen in %.0f s: input at %.1f s goes now", ev.anchor.c_str(),
                            ev.occurrence, timeout, ev.t);
                    at = InputTime();
                } else {
                    at = t + ev.delay;
                }
            }
            std::this_thread::sleep_until(g_input_t0 + std::chrono::milliseconds(static_cast<s64>(at * 1000)));
            if (ev.kind == 'B') {
                SetButton(ev.a, ev.b != 0);
            } else if (ev.kind == 'T') {
                SetTrigger(ev.a != 0, static_cast<u8>(ev.b));
            } else {
                SetStickAxis(ev.a != 0, ev.b != 0, static_cast<u8>(ev.c));
            }
        }
    }).detach();
}

// DC_AUTO_THROTTLE=<sec>,<ms> / DC_AUTO_BRAKE=<sec>,<ms>: full R2 / L2 for
// <ms> from <sec> after start.
static void StartAutoTrigger(const char* var, bool right) {
    const char* e = std::getenv(var);
    if (!e) {
        return;
    }
    double at = 0;
    s64 ms = 0;
    std::sscanf(e, "%lf,%lld", &at, &ms);
    std::thread([at, ms, right] {
        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<s64>(at * 1000)));
        DC_INFO("pad: auto-%s for %lld ms", right ? "throttle" : "brake", ms);
        SetTrigger(right, 255);
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
        SetTrigger(right, 0);
    }).detach();
}

void RegisterPad() {
    g_input_t0 = std::chrono::steady_clock::now();
    StartAutoPress();
    StartReplay();
    StartAutoTrigger("DC_AUTO_THROTTLE", true);
    StartAutoTrigger("DC_AUTO_BRAKE", false);
    constexpr const char* lib = "libScePad";
    DC_HLE(lib, scePadInit);
    DC_HLE(lib, scePadOpen);
    DC_HLE(lib, scePadOpenExt);
    DC_HLE(lib, scePadClose);
    DC_HLE(lib, scePadReadState);
    DC_HLE(lib, scePadRead);
    DC_HLE(lib, scePadGetControllerInformation);
    DC_HLE(lib, scePadGetExtControllerInformation);
    DC_HLE(lib, scePadSetTiltCorrectionState);
    DC_HLE(lib, scePadSetVibration);
    DC_HLE(lib, scePadResetOrientation);
}

} // namespace Hle::Pad
