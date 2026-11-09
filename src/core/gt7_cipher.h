#pragma once

#include "common/types.h"

namespace Core::Gt7Cipher {

// GT7 v1.00 decrypts every file it reads with ChaCha20, one 64-byte block at a
// time, then XORs byte by byte. Install() replaces that function with one that
// computes four blocks at once (SSE) and XORs 16 bytes at a time, after a
// self-test against the game's own function. DC_FAST_CIPHER=0: off;
// DC_FAST_CIPHER=game: the game's code behind the hook (to measure it);
// DC_FAST_CIPHER_VERIFY=1: every call also runs the game's code on copies and
// compares. DC_DIAG_CIPHER=1: calls, MB and time each second.
void Install(u64 eboot_base);

} // namespace Core::Gt7Cipher
