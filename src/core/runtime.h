#pragma once

#include "common/types.h"

// Runs the guest: a dedicated thread with guest TLS starts the preloaded Sony
// modules, then jumps to the executable's entry point. The host main thread
// stays free for the window and presentation (Cocoa requires it).

namespace Core::Loader {
class Linker;
class Module;
}

namespace Core::Runtime {

// Starts the guest main thread and returns immediately.
void StartGame(Loader::Linker& linker);

// Calls a module's init (module_start) once; returns its result.
int StartModule(const char* name, u64 args = 0, const void* argp = nullptr);

// True if the loaded eboot is a version this runtime patches: GT7 v1.00
// (CUSA24769), or Driveclub v1.28 (CUSA00003), the game the runtime started
// from, when every fixed-heap site holds the expected instruction.
bool IsSupportedExecutable(const Loader::Module& exe);

} // namespace Core::Runtime
