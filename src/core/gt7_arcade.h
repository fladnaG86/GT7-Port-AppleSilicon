#pragma once

#include <vector>

#include "common/types.h"

namespace Core::Gt7Arcade {

// Changes `script` (a whole Adhoc script as GT7 v1.00 reads it) when it is the
// one with Arcade's menu, so that Arcade remembers its course, weather and car
// across launches. Any other script, or one that differs from what is
// expected, stays as it is. Returns true when changed.
bool Patch(std::vector<u8>& script);

} // namespace Core::Gt7Arcade
