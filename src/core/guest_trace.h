#pragma once

#include "common/types.h"

// DC_TRACE=<offset>,<offset>,...: counts calls of guest functions, given as
// offsets into the executable. Every offset must be a `push rbp`. Logs the
// counts while they change, then the last calls once they stop: where a chain
// of waits broke, which static reading cannot tell.

namespace Core::GuestTrace {

void Start(u64 image_base);

} // namespace Core::GuestTrace
