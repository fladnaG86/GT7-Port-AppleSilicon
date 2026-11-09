#pragma once

#include "common/types.h"

namespace Hle::Net {

void RegisterNet();

// The game's Vegas keys as base64 text in its executable (cipher key: 32
// bytes, HMAC key: 64 bytes); invalid text leaves the asset server's files
// unencoded.
void SetVegasKeys(const char* cipher_b64, const char* hmac_b64);

} // namespace Hle::Net
