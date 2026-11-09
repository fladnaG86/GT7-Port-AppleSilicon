#pragma once

#include "common/types.h"

namespace Core::Gt7Scripts {

// GT7 v1.00 loads its Adhoc scripts ("ADCH" + version 14) through one loader
// (eboot 0x2febb30) that reads them from a stream. Install() wraps it: while a
// script loads, its stream reads through a copy of its vtable whose read takes
// the whole script on the first call and serves it from a copy, which can be
// dumped (DC_DUMP_SCRIPTS=<dir>) and changed (Arcade remembering its choices,
// gt7_arcade.cpp; DC_ARCADE_MEMORY=0 turns that off). With neither, nothing is
// hooked.
void Install(u64 eboot_base);

} // namespace Core::Gt7Scripts
