#pragma once

// GPU frame capture and offline replay.
//
// Capture (DC_CAPTURE=<dir> DC_CAPTURE_FLIP=<n>): right after the game queues
// flip n-1, the submitting thread waits for the command processor and the
// host GPU to go idle, then writes every guest mapping (direct memory by
// physical address, the rest by virtual address), the CP's carried-over state
// and the GDS. GPU-visible memory is then write-protected: each later submit
// first records the pages the CPU wrote since the previous one, then the
// submit itself (command buffer addresses), for DC_CAPTURE_FRAMES frames
// (default 2: flips n and n+1; GT7 builds a frame's scene during the frame
// before, so look at the last replayed flip). DC_CAPTURE_EXIT=1
// ends the process once the files are written.
//
// Replay (DC_FRAME_REPLAY=<dir>): the gt7 binary loads no game. It rebuilds
// the mappings, loads the memory, and plays the frame's submits in order
// DC_REPLAY_LOOPS times (default 4; the first passes compile pipelines),
// restoring memory and CP state before each. Frame and pass dumps, traces and
// every other renderer switch work as in a game run; flip k = loop k.

#include <cstdint>
#include <span>
#include <vector>

#include "common/types.h"
#include "gpu/liverpool/command_processor.h"

namespace Gpu {
struct FlipInfo;
class Renderer;
} // namespace Gpu

namespace Gpu::Capture {

// ---- Capture side (called from the command processor and HLE) ----
using Buffers = Liverpool::CommandProcessor::CmdBuffers;
// Before the graphics submit is queued (`flip`: it carries a flip).
void BeforeGraphics(const std::vector<Buffers>& bufs, bool flip);
// After it is queued: counts flips, starts or ends a capture.
void AfterGraphics(bool flip);
void BeforeCompute(u32 queue, std::span<const u32> acb);
// What the next flip submit shows (gnmdriver, before it submits).
void NoteFlip(const FlipInfo& flip);
// Signal handler: a write to a page the capture protected. True if handled.
bool HandleWriteFault(void* addr);
// The host kernel is about to write guest memory (read/pread): pages the
// capture protected are opened first (a protected page would fail with EFAULT).
void BeforeHostWrite(void* addr, u64 size);

// ---- Replay side ----
// Recreates the capture's mappings and memory. Call after Memory::Init.
bool LoadMemory(const char* dir);
// Plays the captured frame; returns when the loops are done.
void Replay(const char* dir, Renderer* renderer);

} // namespace Gpu::Capture
