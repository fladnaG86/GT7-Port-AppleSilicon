#pragma once

// DC_GUEST_PROFILE=<start sec>,<seconds>: samples every guest thread and logs where each one spent its time. `sample` cannot see guest code under
// Rosetta; Mach thread state can.

namespace Core::GuestProfile {

// Call on the thread itself, before it runs guest code.
void RegisterThread(const char* name);
// Starts the sampler if the environment asks for it.
void Start();

} // namespace Core::GuestProfile
