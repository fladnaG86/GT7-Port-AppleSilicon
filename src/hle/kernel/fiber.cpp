// SPDX-License-Identifier: GPL-2.0-or-later
// libSceFiber: cooperative contexts that may resume on a different thread.

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>

#include <sys/mman.h>
#include <xmmintrin.h>

#include "common/log.h"
#include "core/loader/linker.h"
#include "hle/hle.h"
#include "hle/kernel/kernel.h"

extern "C" void DcFiberSwitchStack(void** save, void* load);
extern "C" void DcFiberTrampoline();

// Saves the System V callee-saved state on the current stack, then restores
// it from `load`. The frame layout is InitialFrame below.
asm(R"(
    .text
    .globl _DcFiberSwitchStack
_DcFiberSwitchStack:
    push %rbp
    push %rbx
    push %r12
    push %r13
    push %r14
    push %r15
    sub $8, %rsp
    stmxcsr 0(%rsp)
    fnstcw 4(%rsp)
    mov %rsp, (%rdi)
    mov %rsi, %rsp
    ldmxcsr 0(%rsp)
    fldcw 4(%rsp)
    add $8, %rsp
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %rbx
    pop %rbp
    ret

    .globl _DcFiberTrampoline
_DcFiberTrampoline:
    mov %r12, %rdi
    and $-16, %rsp
    call _DcFiberMain
    ud2
)");

namespace Hle::Kernel {

namespace {

constexpr s32 kFiberErrorNull = static_cast<s32>(0x80590001);
constexpr s32 kFiberErrorAlignment = static_cast<s32>(0x80590002);
constexpr s32 kFiberErrorRange = static_cast<s32>(0x80590003);
constexpr s32 kFiberErrorInvalid = static_cast<s32>(0x80590004);
constexpr s32 kFiberErrorPermission = static_cast<s32>(0x80590005);
constexpr s32 kFiberErrorState = static_cast<s32>(0x80590006);

constexpr u64 kFiberMagic = 0x4342494644434647ull;
constexpr u64 kMinContextSize = 512;
// The game sizes a context for its own code; HLE calls run on it too.
constexpr u64 kStackHeadroom = 512 * 1024;
constexpr u64 kObjectSize = 0x100;

using FiberEntry = void (*)(u64 arg_on_initialize, u64 arg_on_run);

// A fiber switching away stays Suspending until its stack pointer is saved;
// whatever runs next on that thread publishes Suspended, so a resumer on
// another thread never reads a stale stack pointer.
enum class FiberState : u32 { Idle = 1, Running, Suspending, Suspended };

struct Fiber {
    u64 magic;
    std::atomic<FiberState> state;
    FiberEntry entry;
    u64 arg_on_initialize;
    u8* stack;
    u64 stack_size;
    void* saved_stack;
    char name[32];
};
static_assert(sizeof(Fiber) <= kObjectSize);

struct InitialFrame {
    u32 mxcsr;
    u16 fpu_control;
    u16 pad;
    u64 r15, r14, r13, r12, rbx, rbp;
    u64 return_address;
};

struct ThreadFiberState {
    Fiber* current = nullptr;
    void* thread_stack = nullptr;
    u64 transfer = 0;
    Fiber* pending_suspend = nullptr;
};

thread_local ThreadFiberState g_thread;

// Fibers migrate between threads, so this must be looked up again after every
// stack switch: the compiler may not cache the TLS address across one.
__attribute__((noinline)) ThreadFiberState& ThreadState() {
    asm volatile("" ::: "memory");
    return g_thread;
}

void CompletePendingSuspend() {
    auto& t = ThreadState();
    if (t.pending_suspend) {
        t.pending_suspend->state.store(FiberState::Suspended, std::memory_order_release);
        t.pending_suspend = nullptr;
    }
}

std::mutex g_fibers_lock;
std::set<Fiber*> g_fibers; // every fiber ever initialized, for DumpFibers

// One host stack per guest context, so re-initializing over a context reuses it.
u8* HostStack(void* context, u64 size) {
    struct Stack {
        u8* base = nullptr;
        u64 size = 0;
    };
    static std::mutex lock;
    static std::map<void*, Stack> stacks;
    std::scoped_lock lk(lock);
    Stack& s = stacks[context];
    if (s.size < size) {
        if (s.base) {
            munmap(s.base, s.size);
        }
        void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        s.base = p == MAP_FAILED ? nullptr : static_cast<u8*>(p);
        s.size = s.base ? size : 0;
    }
    return s.base;
}

void PrepareInitialStack(Fiber* f) {
    const u64 top = reinterpret_cast<u64>(f->stack + f->stack_size) & ~u64{15};
    auto* frame = reinterpret_cast<InitialFrame*>(top - 256);
    std::memset(frame, 0, sizeof(*frame));
    frame->mxcsr = _mm_getcsr();
    asm volatile("fnstcw %0" : "=m"(frame->fpu_control));
    frame->r12 = reinterpret_cast<u64>(f);
    frame->return_address = reinterpret_cast<u64>(&DcFiberTrampoline);
    f->saved_stack = frame;
}

bool AcquireForResume(Fiber* f) {
    for (;;) {
        FiberState s = f->state.load(std::memory_order_acquire);
        if (s == FiberState::Suspending) {
            std::this_thread::yield();
            continue;
        }
        if (s != FiberState::Idle && s != FiberState::Suspended) {
            return false;
        }
        if (f->state.compare_exchange_weak(s, FiberState::Running, std::memory_order_acq_rel)) {
            if (s == FiberState::Idle) {
                PrepareInitialStack(f);
            }
            return true;
        }
    }
}

void Resume(Fiber* target, void** save, u64 arg) {
    auto& t = ThreadState();
    t.current = target;
    t.transfer = arg;
    DcFiberSwitchStack(save, target->saved_stack);
}

Fiber* AsFiber(void* object) {
    auto* f = static_cast<Fiber*>(object);
    return f && f->magic == kFiberMagic ? f : nullptr;
}

s32 BadFiber(void* object) {
    return object ? kFiberErrorInvalid : kFiberErrorNull;
}

s32 _sceFiberInitializeImpl(void* object, const char* name, FiberEntry entry, u64 arg_on_initialize,
                            void* context, u64 context_size, const void* /*opt*/,
                            u32 /*build_version*/) {
    if (!object || !name || !entry) {
        return kFiberErrorNull;
    }
    if (reinterpret_cast<u64>(object) & 7) {
        return kFiberErrorAlignment;
    }
    if (!context) {
        return kFiberErrorInvalid;
    }
    if ((reinterpret_cast<u64>(context) & 15) || (context_size & 15)) {
        return kFiberErrorAlignment;
    }
    if (context_size < kMinContextSize) {
        return kFiberErrorRange;
    }
    const u64 stack_size = context_size + kStackHeadroom;
    u8* stack = HostStack(context, stack_size);
    if (!stack) {
        return kFiberErrorRange;
    }
    std::memset(object, 0, kObjectSize);
    auto* f = static_cast<Fiber*>(object);
    f->magic = kFiberMagic;
    f->state.store(FiberState::Idle, std::memory_order_relaxed);
    f->entry = entry;
    f->arg_on_initialize = arg_on_initialize;
    f->stack = stack;
    f->stack_size = stack_size;
    std::strncpy(f->name, name, sizeof(f->name) - 1);
    {
        std::scoped_lock lk(g_fibers_lock);
        g_fibers.insert(f);
    }
    return ORBIS_OK;
}

s32 sceFiberOptParamInitialize(void* param) {
    if (!param) {
        return kFiberErrorNull;
    }
    std::memset(param, 0, 0x80);
    return ORBIS_OK;
}

s32 sceFiberFinalize(void* object) {
    Fiber* f = AsFiber(object);
    if (!f) {
        return BadFiber(object);
    }
    const FiberState s = f->state.load(std::memory_order_acquire);
    if (s == FiberState::Running || s == FiberState::Suspending) {
        return kFiberErrorState;
    }
    f->magic = 0;
    {
        std::scoped_lock lk(g_fibers_lock);
        g_fibers.erase(f);
    }
    return ORBIS_OK;
}

s32 sceFiberRun(void* object, u64 arg_on_run, u64* arg_on_return) {
    Fiber* f = AsFiber(object);
    if (!f) {
        return BadFiber(object);
    }
    if (ThreadState().current) {
        return kFiberErrorPermission;
    }
    if (!AcquireForResume(f)) {
        return kFiberErrorState;
    }
    Resume(f, &ThreadState().thread_stack, arg_on_run);
    CompletePendingSuspend();
    if (arg_on_return) {
        *arg_on_return = ThreadState().transfer;
    }
    return ORBIS_OK;
}

s32 sceFiberSwitch(void* object, u64 arg_on_run, u64* arg_on_run_out) {
    Fiber* target = AsFiber(object);
    if (!target) {
        return BadFiber(object);
    }
    Fiber* self = ThreadState().current;
    if (!self) {
        return kFiberErrorPermission;
    }
    if (target == self || !AcquireForResume(target)) {
        return kFiberErrorState;
    }
    self->state.store(FiberState::Suspending, std::memory_order_relaxed);
    ThreadState().pending_suspend = self;
    Resume(target, &self->saved_stack, arg_on_run);
    CompletePendingSuspend();
    if (arg_on_run_out) {
        *arg_on_run_out = ThreadState().transfer;
    }
    return ORBIS_OK;
}

s32 sceFiberReturnToThread(u64 arg_on_return, u64* arg_on_run) {
    Fiber* self = ThreadState().current;
    if (!self) {
        return kFiberErrorPermission;
    }
    self->state.store(FiberState::Suspending, std::memory_order_relaxed);
    auto& t = ThreadState();
    t.pending_suspend = self;
    t.current = nullptr;
    t.transfer = arg_on_return;
    DcFiberSwitchStack(&self->saved_stack, t.thread_stack);
    CompletePendingSuspend();
    if (arg_on_run) {
        *arg_on_run = ThreadState().transfer;
    }
    return ORBIS_OK;
}

s32 sceFiberGetSelf(void** fiber) {
    if (!fiber) {
        return kFiberErrorNull;
    }
    Fiber* self = ThreadState().current;
    if (!self) {
        return kFiberErrorPermission;
    }
    *fiber = self;
    return ORBIS_OK;
}

s32 sceFiberRename(void* object, const char* name) {
    Fiber* f = AsFiber(object);
    if (!f || !name) {
        return object && name ? kFiberErrorInvalid : kFiberErrorNull;
    }
    std::memset(f->name, 0, sizeof(f->name));
    std::strncpy(f->name, name, sizeof(f->name) - 1);
    // DC_LOG_FIBERS: games name their tasks through this.
    static const bool log = std::getenv("DC_LOG_FIBERS") != nullptr;
    if (log) {
        DC_INFO("fiber rename: %s", name);
    }
    return ORBIS_OK;
}

} // namespace

extern "C" [[noreturn]] void DcFiberMain(Fiber* fiber) {
    CompletePendingSuspend();
    fiber->entry(fiber->arg_on_initialize, ThreadState().transfer);
    DC_ERROR("fiber '%s': entry function returned", fiber->name);
    std::abort();
}

void DumpFibers() {
    std::scoped_lock lk(g_fibers_lock);
    auto* linker = Core::Loader::Linker::Instance();
    for (Fiber* f : g_fibers) {
        if (f->state.load() != FiberState::Suspended) {
            DC_INFO("fiber %s: state %u", f->name, static_cast<u32>(f->state.load()));
            continue;
        }
        const auto* frame = static_cast<const InitialFrame*>(f->saved_stack);
        std::string line = " < " + linker->Describe(frame->return_address);
        u64 fp = frame->rbp;
        for (int depth = 0; depth < 24; ++depth) {
            if (fp < reinterpret_cast<u64>(f->stack) ||
                fp + 16 > reinterpret_cast<u64>(f->stack) + f->stack_size || (fp & 7)) {
                break;
            }
            const auto* links = reinterpret_cast<const u64*>(fp);
            line += " < " + linker->Describe(links[1]);
            if (links[0] <= fp) {
                break;
            }
            fp = links[0];
        }
        DC_INFO("fiber %s suspended%s", f->name, line.c_str());
    }
}

void RegisterFiber() {
    constexpr const char* k = "libSceFiber";
    DC_HLE(k, _sceFiberInitializeImpl);
    DC_HLE(k, sceFiberOptParamInitialize);
    DC_HLE(k, sceFiberFinalize);
    DC_HLE(k, sceFiberRun);
    DC_HLE(k, sceFiberSwitch);
    DC_HLE(k, sceFiberReturnToThread);
    DC_HLE(k, sceFiberGetSelf);
    DC_HLE(k, sceFiberRename);
}

} // namespace Hle::Kernel
