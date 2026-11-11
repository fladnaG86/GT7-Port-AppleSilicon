// SPDX-License-Identifier: GPL-2.0-or-later
//
// libSceAvPlayer on AVFoundation. The game hands the player its own file
// callbacks (GT7 reads /movie/... out of its .vol archives); the whole file is
// copied to a host temp file, then one AVAssetReader per track decodes it
// (VideoToolbox does H.264 in hardware, also under Rosetta). Video frames are
// NV12 in buffers the game allocates through its texture allocator (pitch
// aligned to 64, height to 16, chroma after the padded luma plane), audio is
// interleaved s16 in chunks of 1024 samples. Frames are handed out against a
// wall clock started by Start. Struct layouts and the frame format follow
// PS4Recomp (wie-project/PS4Recomp, GPL-2.0-or-later) and shadPS4.
//
// If the source cannot be read or decoded the player keeps its old
// behaviour: StateReady at once, StateStop 300 ms after Start.
// DC_NO_AVPLAYER=1 forces that fallback.

#include "hle/avplayer/avplayer.h"

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/log.h"
#include "common/types.h"
#include "core/memory.h"
#include "core/tls.h"
#include "hle/hle.h"

namespace Hle::AvPlayer {

namespace {

using Clock = std::chrono::steady_clock;

using EventCallback = void (*)(void* p, s32 event, s32 src_id, void* data);
using AllocateFn = void* (*)(void* p, u32 align, u32 size);
using DeallocateFn = void (*)(void* p, void* mem);
using OpenFn = s32 (*)(void* p, const char* filename);
using CloseFn = s32 (*)(void* p);
using ReadOffsetFn = s32 (*)(void* p, u8* buf, u64 pos, u32 len);
using SizeFn = u64 (*)(void* p);

constexpr s32 kEventStop = 1;
constexpr s32 kEventReady = 2;
constexpr s32 kEventPlay = 3;
constexpr s32 kEventPause = 4;

constexpr u32 kStreamVideo = 0;
constexpr u32 kStreamAudio = 1;

constexpr u32 kAudioChunkSamples = 1024;
constexpr u32 kVideoBuffers = 6;
constexpr u32 kAudioBuffers = 32;

// SceAvPlayerInitData / InitDataEx (the Ex one has a leading size field).
struct MemAllocator {
    void* object;
    AllocateFn allocate;
    DeallocateFn deallocate;
    AllocateFn allocate_texture;
    DeallocateFn deallocate_texture;
};
struct FileReplacement {
    void* object;
    OpenFn open;
    CloseFn close;
    ReadOffsetFn read_offset;
    SizeFn size;
};
struct EventReplacement {
    void* object;
    EventCallback callback;
};
struct InitData {
    MemAllocator memory;
    FileReplacement file;
    EventReplacement event;
};
static_assert(sizeof(InitData) == 96);

u32 AlignUp(u32 v, u32 a) { return (v + a - 1) & ~(a - 1); }

struct VideoFrame {
    u8* buf;
    u64 ts_ms;
};
struct AudioFrame {
    u8* buf;
    u32 bytes;
    u64 ts_ms;
};

struct Player {
    InitData init{};
    bool ready_on_add = false; // InitEx games wait for StateReady before Start
    bool loaded = false;       // a decodable source is present
    bool looping = false;
    std::string temp_path;

    // Source description, from the asset.
    bool has_video = false, has_audio = false;
    bool video_enabled = false, audio_enabled = false;
    u32 width = 0, height = 0, pitch = 0, aligned_height = 0;
    float fps = 30.0f;
    u32 channels = 0, sample_rate = 0;
    u64 duration_ms = 0;

    // Playback.
    std::mutex mu;
    std::condition_variable cv;
    std::deque<VideoFrame> video_ready;
    std::vector<u8*> video_free, video_all;
    u8* video_current = nullptr;
    std::deque<AudioFrame> audio_ready;
    std::vector<u8*> audio_free, audio_all;
    u8* audio_current = nullptr;
    std::thread video_thread, audio_thread, control_thread;
    std::atomic<bool> stop_workers{false};
    std::atomic<bool> playing{false};
    std::atomic<bool> paused{false};
    std::atomic<bool> video_eof{false}, audio_eof{false};
    std::atomic<bool> stop_sent{false};
    std::atomic<u32> generation{0};
    Clock::time_point start_time{};
    Clock::time_point pause_time{};
    Clock::duration paused_total{};

    // Statistics, logged at the end.
    u64 video_decoded = 0, video_shown = 0, video_dropped = 0, audio_chunks = 0;
};

Player g_player;

bool Disabled() {
    static const bool off = [] {
        const char* v = std::getenv("DC_NO_AVPLAYER");
        return v && v[0] == '1';
    }();
    return off;
}

void Emit(s32 event) {
    if (g_player.init.event.callback) {
        g_player.init.event.callback(g_player.init.event.object, event, 0, nullptr);
    }
}

// Emits from a host thread: guest callbacks need a guest TLS block.
void EmitFromHost(s32 event) {
    Core::Tls::SetupThread();
    Emit(event);
    Core::Tls::TeardownThread();
}

u64 NowMs() {
    Player& p = g_player;
    if (!p.playing.load()) {
        return 0;
    }
    Clock::time_point now = p.paused.load() ? p.pause_time : Clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - p.start_time - p.paused_total).count();
}

u64 ToMs(CMTime t) {
    if (!CMTIME_IS_NUMERIC(t)) {
        return 0;
    }
    const double s = CMTimeGetSeconds(t);
    return s > 0 ? static_cast<u64>(s * 1000.0) : 0;
}

// Copies the source to a host file through the game's file callbacks.
// Returns false if it cannot be read.
bool FetchSource(const char* filename) {
    Player& p = g_player;
    const FileReplacement& f = p.init.file;
    if (!f.open || !f.read_offset || !f.size) {
        DC_WARN("avplayer: no file callbacks for %s, not supported", filename);
        return false;
    }
    const auto t0 = Clock::now();
    if (f.open(f.object, filename) < 0) {
        DC_WARN("avplayer: game open failed for %s", filename);
        return false;
    }
    const u64 size = f.size(f.object);
    if (size == 0 || size > (u64(4) << 30)) {
        DC_WARN("avplayer: %s has size %llu", filename, (unsigned long long)size);
        if (f.close) {
            f.close(f.object);
        }
        return false;
    }
    static std::atomic<u32> serial{0};
    char path[512];
    const char* tmp = std::getenv("TMPDIR");
    std::snprintf(path, sizeof(path), "%s/gt7-avplayer-%d-%u.mp4", tmp ? tmp : "/tmp", getpid(), ++serial);
    FILE* out = std::fopen(path, "wb");
    if (!out) {
        if (f.close) {
            f.close(f.object);
        }
        return false;
    }
    constexpr u32 kChunk = 1u << 20;
    std::vector<u8> buf(kChunk);
    u64 pos = 0;
    bool ok = true;
    while (pos < size) {
        const u32 want = static_cast<u32>(std::min<u64>(kChunk, size - pos));
        const s32 got = f.read_offset(f.object, buf.data(), pos, want);
        if (got <= 0) {
            ok = false;
            break;
        }
        std::fwrite(buf.data(), 1, got, out);
        pos += got;
    }
    std::fclose(out);
    if (f.close) {
        f.close(f.object);
    }
    if (!ok) {
        DC_WARN("avplayer: read of %s stopped at %llu of %llu", filename, (unsigned long long)pos,
                (unsigned long long)size);
        unlink(path);
        return false;
    }
    p.temp_path = path;
    DC_INFO("avplayer: %s, %.1f MB copied in %.0f ms", filename, size / 1048576.0,
            std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    return true;
}

AVURLAsset* OpenAsset() {
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:g_player.temp_path.c_str()]];
    return [AVURLAsset URLAssetWithURL:url options:@{AVURLAssetPreferPreciseDurationAndTimingKey : @YES}];
}

// Reads the stream description out of the asset. Synchronous track loading is
// deprecated but still works, and keeps the HLE call blocking as on a console.
bool Probe() {
    Player& p = g_player;
    @autoreleasepool {
        AVURLAsset* asset = OpenAsset();
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        NSArray<AVAssetTrack*>* video = [asset tracksWithMediaType:AVMediaTypeVideo];
        NSArray<AVAssetTrack*>* audio = [asset tracksWithMediaType:AVMediaTypeAudio];
        p.duration_ms = ToMs(asset.duration);
        if (video.count) {
            AVAssetTrack* t = video[0];
            const CGSize s = t.naturalSize;
            p.width = static_cast<u32>(s.width);
            p.height = static_cast<u32>(s.height);
            p.fps = t.nominalFrameRate > 0 ? t.nominalFrameRate : 30.0f;
            p.pitch = AlignUp(p.width, 64);
            p.aligned_height = AlignUp(p.height, 16);
            p.has_video = p.width && p.height;
            if (t.formatDescriptions.count) {
                auto desc = (__bridge CMFormatDescriptionRef)t.formatDescriptions[0];
                const FourCharCode c = CMFormatDescriptionGetMediaSubType(desc);
                CFTypeRef tf = CMFormatDescriptionGetExtension(desc, kCMFormatDescriptionExtension_TransferFunction);
                CFTypeRef bits = CMFormatDescriptionGetExtension(desc, kCMFormatDescriptionExtension_Depth);
                DC_INFO("avplayer: codec %c%c%c%c transfer %s depth %s", char(c >> 24), char(c >> 16), char(c >> 8),
                        char(c), tf ? [(__bridge NSString*)tf UTF8String] : "-",
                        bits ? [[(__bridge NSNumber*)bits stringValue] UTF8String] : "-");
            }
        }
        if (audio.count) {
            AVAssetTrack* t = audio[0];
            if (t.formatDescriptions.count) {
                auto desc = (__bridge CMAudioFormatDescriptionRef)t.formatDescriptions[0];
                if (const AudioStreamBasicDescription* asbd = CMAudioFormatDescriptionGetStreamBasicDescription(desc)) {
                    p.channels = std::min<u32>(asbd->mChannelsPerFrame, 8);
                    p.sample_rate = static_cast<u32>(asbd->mSampleRate);
                    p.has_audio = p.channels && p.sample_rate;
                }
            }
        }
#pragma clang diagnostic pop
    }
    DC_INFO("avplayer: video %s %ux%u %.2f fps (pitch %u), audio %s %u ch %u Hz, %llu ms",
            p.has_video ? "yes" : "no", p.width, p.height, p.fps, p.pitch, p.has_audio ? "yes" : "no",
            p.channels, p.sample_rate, (unsigned long long)p.duration_ms);
    return p.has_video || p.has_audio;
}

AVAssetReader* MakeReader(AVMediaType type, NSDictionary* settings, AVAssetReaderTrackOutput** out) {
    AVURLAsset* asset = OpenAsset();
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    NSArray<AVAssetTrack*>* tracks = [asset tracksWithMediaType:type];
#pragma clang diagnostic pop
    if (!tracks.count) {
        return nil;
    }
    NSError* err = nil;
    AVAssetReader* reader = [AVAssetReader assetReaderWithAsset:asset error:&err];
    if (!reader) {
        DC_WARN("avplayer: reader: %s", err.localizedDescription.UTF8String);
        return nil;
    }
    AVAssetReaderTrackOutput* o = [AVAssetReaderTrackOutput assetReaderTrackOutputWithTrack:tracks[0]
                                                                             outputSettings:settings];
    o.alwaysCopiesSampleData = NO;
    [reader addOutput:o];
    if (![reader startReading]) {
        DC_WARN("avplayer: startReading: %s", reader.error.localizedDescription.UTF8String);
        return nil;
    }
    *out = o;
    return reader;
}

// Waits for a free buffer; nullptr when the workers are told to stop.
u8* TakeFree(std::vector<u8*>& pool) {
    Player& p = g_player;
    std::unique_lock lk(p.mu);
    p.cv.wait(lk, [&] { return p.stop_workers.load() || !pool.empty(); });
    if (p.stop_workers.load()) {
        return nullptr;
    }
    u8* b = pool.back();
    pool.pop_back();
    return b;
}

void GiveBack(std::vector<u8*>& pool, u8* b) {
    std::scoped_lock lk(g_player.mu);
    pool.push_back(b);
}

void VideoWorker() {
    Player& p = g_player;
    @autoreleasepool {
        AVAssetReaderTrackOutput* out = nil;
        NSDictionary* settings = @{
            (id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
        };
        AVAssetReader* reader = MakeReader(AVMediaTypeVideo, settings, &out);
        while (reader && !p.stop_workers.load()) {
            @autoreleasepool {
                CMSampleBufferRef sb = [out copyNextSampleBuffer];
                if (!sb) {
                    break;
                }
                CVImageBufferRef img = CMSampleBufferGetImageBuffer(sb);
                if (!img) {
                    CFRelease(sb);
                    continue;
                }
                u8* dst = TakeFree(p.video_free);
                if (!dst) {
                    CFRelease(sb);
                    break;
                }
                CVPixelBufferLockBaseAddress(img, kCVPixelBufferLock_ReadOnly);
                const u32 w = std::min<u32>(static_cast<u32>(CVPixelBufferGetWidth(img)), p.pitch);
                const u32 h = std::min<u32>(static_cast<u32>(CVPixelBufferGetHeight(img)), p.aligned_height);
                const u8* y = static_cast<const u8*>(CVPixelBufferGetBaseAddressOfPlane(img, 0));
                const u8* uv = static_cast<const u8*>(CVPixelBufferGetBaseAddressOfPlane(img, 1));
                const size_t ys = CVPixelBufferGetBytesPerRowOfPlane(img, 0);
                const size_t uvs = CVPixelBufferGetBytesPerRowOfPlane(img, 1);
                u8* dst_uv = dst + size_t(p.pitch) * p.aligned_height;
                for (u32 r = 0; r < h; ++r) {
                    std::memcpy(dst + size_t(r) * p.pitch, y + r * ys, w);
                }
                for (u32 r = 0; r < (h + 1) / 2; ++r) {
                    std::memcpy(dst_uv + size_t(r) * p.pitch, uv + r * uvs, w);
                }
                CVPixelBufferUnlockBaseAddress(img, kCVPixelBufferLock_ReadOnly);
                const u64 ts = ToMs(CMSampleBufferGetPresentationTimeStamp(sb));
                CFRelease(sb);
                std::scoped_lock lk(p.mu);
                p.video_ready.push_back({dst, ts});
                ++p.video_decoded;
            }
        }
        if (reader) {
            [reader cancelReading];
        }
    }
    p.video_eof = true;
}

void AudioWorker() {
    Player& p = g_player;
    @autoreleasepool {
        AVAssetReaderTrackOutput* out = nil;
        NSDictionary* settings = @{
            AVFormatIDKey : @(kAudioFormatLinearPCM),
            AVLinearPCMBitDepthKey : @16,
            AVLinearPCMIsFloatKey : @NO,
            AVLinearPCMIsBigEndianKey : @NO,
            AVLinearPCMIsNonInterleaved : @NO,
            AVSampleRateKey : @(p.sample_rate),
            AVNumberOfChannelsKey : @(p.channels),
        };
        AVAssetReader* reader = MakeReader(AVMediaTypeAudio, settings, &out);
        const u32 frame_bytes = p.channels * 2;
        const u32 chunk_bytes = kAudioChunkSamples * frame_bytes;
        std::vector<u8> pending; // PCM not yet cut into chunks
        u64 pending_start = 0;   // sample index of pending[0]
        bool first = true;
        auto flush = [&](bool all) {
            while (pending.size() >= chunk_bytes || (all && !pending.empty())) {
                u8* dst = TakeFree(p.audio_free);
                if (!dst) {
                    return false;
                }
                const u32 n = static_cast<u32>(std::min<size_t>(chunk_bytes, pending.size()));
                std::memcpy(dst, pending.data(), n);
                std::memset(dst + n, 0, chunk_bytes - n);
                const u64 ts = pending_start * 1000 / p.sample_rate;
                pending.erase(pending.begin(), pending.begin() + n);
                pending_start += n / frame_bytes;
                std::scoped_lock lk(p.mu);
                p.audio_ready.push_back({dst, chunk_bytes, ts});
                ++p.audio_chunks;
            }
            return true;
        };
        while (reader && !p.stop_workers.load()) {
            @autoreleasepool {
                CMSampleBufferRef sb = [out copyNextSampleBuffer];
                if (!sb) {
                    break;
                }
                if (first) {
                    first = false;
                    pending_start = ToMs(CMSampleBufferGetPresentationTimeStamp(sb)) * p.sample_rate / 1000;
                }
                if (CMBlockBufferRef bb = CMSampleBufferGetDataBuffer(sb)) {
                    const size_t len = CMBlockBufferGetDataLength(bb);
                    const size_t at = pending.size();
                    pending.resize(at + len);
                    CMBlockBufferCopyDataBytes(bb, 0, len, pending.data() + at);
                }
                CFRelease(sb);
                if (!flush(false)) {
                    break;
                }
            }
        }
        if (reader && !p.stop_workers.load()) {
            flush(true);
        }
        if (reader) {
            [reader cancelReading];
        }
    }
    p.audio_eof = true;
}

bool Drained() {
    Player& p = g_player;
    std::scoped_lock lk(p.mu);
    return (!p.video_enabled || (p.video_eof && p.video_ready.empty())) &&
           (!p.audio_enabled || (p.audio_eof && p.audio_ready.empty()));
}

void StopWorkers() {
    Player& p = g_player;
    p.stop_workers = true;
    p.cv.notify_all();
    for (std::thread* t : {&p.video_thread, &p.audio_thread}) {
        if (t->joinable()) {
            t->join();
        }
    }
}

void StartWorkers() {
    Player& p = g_player;
    p.stop_workers = false;
    {
        std::scoped_lock lk(p.mu);
        for (const VideoFrame& f : p.video_ready) {
            p.video_free.push_back(f.buf);
        }
        p.video_ready.clear();
        for (const AudioFrame& f : p.audio_ready) {
            p.audio_free.push_back(f.buf);
        }
        p.audio_ready.clear();
    }
    p.video_eof = !p.video_enabled;
    p.audio_eof = !p.audio_enabled;
    p.start_time = Clock::now();
    p.paused_total = {};
    if (p.video_enabled) {
        p.video_thread = std::thread(VideoWorker);
    }
    if (p.audio_enabled) {
        p.audio_thread = std::thread(AudioWorker);
    }
}

// Ends playback at the end of the file (or loops), from a host thread.
void ControlWorker(u32 generation) {
    Player& p = g_player;
    while (generation == p.generation.load() && p.playing.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (generation != p.generation.load() || !Drained()) {
            continue;
        }
        if (p.looping) {
            StopWorkers();
            StartWorkers();
            continue;
        }
        if (!p.stop_sent.exchange(true)) {
            DC_INFO("avplayer: end of movie at %llu ms: %llu frames decoded, %llu shown, %llu dropped, "
                    "%llu audio chunks",
                    (unsigned long long)NowMs(), (unsigned long long)p.video_decoded,
                    (unsigned long long)p.video_shown, (unsigned long long)p.video_dropped,
                    (unsigned long long)p.audio_chunks);
            p.playing = false;
            EmitFromHost(kEventStop);
        }
        break;
    }
}

// The game's Stop callback may call Stop or Close on the control thread itself.
void JoinControl() {
    Player& p = g_player;
    if (!p.control_thread.joinable()) {
        return;
    }
    if (p.control_thread.get_id() == std::this_thread::get_id()) {
        p.control_thread.detach();
    } else {
        p.control_thread.join();
    }
}

void* GuestAlloc(bool texture, u32 align, u32 size) {
    const MemAllocator& m = g_player.init.memory;
    AllocateFn fn = texture && m.allocate_texture ? m.allocate_texture : m.allocate;
    if (fn) {
        return fn(m.object, align, size);
    }
    VAddr addr = 0;
    using namespace Core::Memory;
    if (MapFlexible(&addr, size, kProtCpuRead | kProtCpuWrite | kProtGpuRead | kProtGpuWrite, 0, "avplayer") != 0) {
        return nullptr;
    }
    return reinterpret_cast<void*>(addr);
}

void GuestFree(bool texture, void* mem, u32 size) {
    const MemAllocator& m = g_player.init.memory;
    AllocateFn afn = texture && m.allocate_texture ? m.allocate_texture : m.allocate;
    DeallocateFn fn = texture && m.allocate_texture ? m.deallocate_texture : m.deallocate;
    if (afn) {
        if (fn) {
            fn(m.object, mem);
        }
        return;
    }
    Core::Memory::Unmap(reinterpret_cast<VAddr>(mem), size);
}

u32 VideoBytes() { return g_player.pitch * g_player.aligned_height * 3 / 2; }
u32 AudioBytes() { return kAudioChunkSamples * g_player.channels * 2; }

bool AllocateBuffers() {
    Player& p = g_player;
    if (p.video_enabled && p.video_all.empty()) {
        for (u32 i = 0; i < kVideoBuffers; ++i) {
            void* b = GuestAlloc(true, 0x100, VideoBytes());
            if (!b) {
                DC_WARN("avplayer: video buffer allocation failed");
                return false;
            }
            std::memset(b, 0, VideoBytes());
            p.video_all.push_back(static_cast<u8*>(b));
        }
        p.video_free = p.video_all;
    }
    if (p.audio_enabled && p.audio_all.empty()) {
        for (u32 i = 0; i < kAudioBuffers; ++i) {
            void* b = GuestAlloc(false, 0x10, AudioBytes());
            if (!b) {
                DC_WARN("avplayer: audio buffer allocation failed");
                return false;
            }
            p.audio_all.push_back(static_cast<u8*>(b));
        }
        p.audio_free = p.audio_all;
    }
    return true;
}

void ReleaseAll() {
    Player& p = g_player;
    ++p.generation;
    p.playing = false;
    StopWorkers();
    JoinControl();
    for (u8* b : p.video_all) {
        GuestFree(true, b, VideoBytes());
    }
    for (u8* b : p.audio_all) {
        GuestFree(false, b, AudioBytes());
    }
    p.video_all.clear();
    p.video_free.clear();
    p.video_ready.clear();
    p.video_current = nullptr;
    p.audio_all.clear();
    p.audio_free.clear();
    p.audio_ready.clear();
    p.audio_current = nullptr;
    if (!p.temp_path.empty()) {
        unlink(p.temp_path.c_str());
        p.temp_path.clear();
    }
    p.loaded = p.has_video = p.has_audio = false;
    p.video_enabled = p.audio_enabled = false;
}

void ResetPlayer() {
    ReleaseAll();
    Player& p = g_player;
    p.init = {};
    p.ready_on_add = false;
    p.looping = false;
    p.paused = false;
    p.video_decoded = p.video_shown = p.video_dropped = p.audio_chunks = 0;
}

void LogInit(const char* what) {
    const InitData& i = g_player.init;
    DC_INFO("%s: event_cb=%p file_open=%p alloc=%p alloc_tex=%p", what, reinterpret_cast<void*>(i.event.callback),
            reinterpret_cast<void*>(i.file.open), reinterpret_cast<void*>(i.memory.allocate),
            reinterpret_cast<void*>(i.memory.allocate_texture));
}

void* sceAvPlayerInit(void* init_data) {
    ResetPlayer();
    if (init_data) {
        std::memcpy(&g_player.init, init_data, sizeof(InitData));
    }
    LogInit("sceAvPlayerInit");
    return &g_player;
}

s32 sceAvPlayerInitEx(const void* p_data, void** p_player) {
    ResetPlayer();
    if (p_data) {
        std::memcpy(&g_player.init, static_cast<const u8*>(p_data) + 8, sizeof(InitData));
        g_player.ready_on_add = true;
    }
    LogInit("sceAvPlayerInitEx");
    if (p_player) {
        *p_player = &g_player;
    }
    return 0;
}

s32 sceAvPlayerClose(void* handle) {
    DC_INFO("sceAvPlayerClose(handle=%p)", handle);
    if (handle == &g_player) {
        ReleaseAll();
    }
    return 0;
}

s32 sceAvPlayerAddSource(void* handle, const char* filename) {
    DC_INFO("sceAvPlayerAddSource(handle=%p, file=%s)", handle, filename ? filename : "null");
    Player& p = g_player;
    if (handle != &p) {
        return 0;
    }
    if (!Disabled() && filename && FetchSource(filename)) {
        p.loaded = Probe();
        p.video_enabled = p.has_video;
        p.audio_enabled = p.has_audio;
    }
    if (!p.loaded) {
        DC_WARN("avplayer: %s not decodable, movie skipped", filename ? filename : "null");
    }
    Emit(kEventReady);
    return 0;
}

s32 sceAvPlayerAddSourceEx(void* handle, u32 uri_type, void* source_details) {
    // SceAvPlayerSourceDetails starts with SceAvPlayerUri {const char* name; u32 length}.
    const char* name = source_details ? *static_cast<const char* const*>(source_details) : nullptr;
    DC_INFO("sceAvPlayerAddSourceEx(type=%u)", uri_type);
    return sceAvPlayerAddSource(handle, name);
}

// Old behaviour, for sources that cannot be played.
void FakePlayback() {
    Player& p = g_player;
    Emit(kEventPlay);
    if (p.ready_on_add) {
        // On a console it comes from the player's own thread, after Start
        // returns, and such games then only wait for it.
        std::thread([generation = ++p.generation] {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            const bool current = generation == g_player.generation.load();
            DC_INFO("avplayer: fallback movie over (%s)", current ? "Stop sent" : "superseded");
            if (current) {
                EmitFromHost(kEventStop);
            }
        }).detach();
    } else {
        Emit(kEventStop);
    }
}

s32 sceAvPlayerStart(void* handle) {
    DC_INFO("sceAvPlayerStart(handle=%p)", handle);
    Player& p = g_player;
    if (handle != &p) {
        return 0;
    }
    if (!p.loaded || (!p.video_enabled && !p.audio_enabled) || !AllocateBuffers()) {
        FakePlayback();
        return 0;
    }
    const u32 generation = ++p.generation;
    p.playing = false;
    StopWorkers();
    JoinControl();
    p.paused = false;
    p.stop_sent = false;
    p.video_decoded = p.video_shown = p.video_dropped = p.audio_chunks = 0;
    p.playing = true;
    StartWorkers();
    p.control_thread = std::thread(ControlWorker, generation);
    Emit(kEventPlay);
    return 0;
}

s32 sceAvPlayerStartEx(void* handle) { return sceAvPlayerStart(handle); }

s32 sceAvPlayerStop(void* handle) {
    DC_INFO("sceAvPlayerStop(handle=%p)", handle);
    Player& p = g_player;
    if (handle != &p) {
        return 0;
    }
    ++p.generation;
    const u64 at_ms = NowMs();
    const bool was_playing = p.playing.exchange(false);
    StopWorkers();
    JoinControl();
    if (was_playing && !p.stop_sent.exchange(true)) {
        DC_INFO("avplayer: stopped by the game at %llu ms: %llu frames decoded, %llu shown, %llu dropped, "
                "%llu audio chunks",
                (unsigned long long)at_ms, (unsigned long long)p.video_decoded,
                (unsigned long long)p.video_shown, (unsigned long long)p.video_dropped,
                (unsigned long long)p.audio_chunks);
        Emit(kEventStop);
    }
    return 0;
}

s32 sceAvPlayerPause(void* handle) {
    Player& p = g_player;
    if (handle == &p && p.playing.load() && !p.paused.exchange(true)) {
        p.pause_time = Clock::now();
        Emit(kEventPause);
    }
    return 0;
}

s32 sceAvPlayerResume(void* handle) {
    Player& p = g_player;
    if (handle == &p && p.paused.exchange(false)) {
        p.paused_total += Clock::now() - p.pause_time;
        Emit(kEventPlay);
    }
    return 0;
}

bool sceAvPlayerIsActive(void* handle) {
    Player& p = g_player;
    return handle == &p && p.playing.load() && !Drained();
}

u64 sceAvPlayerCurrentTime(void* handle) { return handle == &g_player ? NowMs() : 0; }

// Hands out the latest due video frame; earlier due ones are dropped, and the
// frame handed out by the previous call goes back to the decoder.
bool NextVideo(VideoFrame* out) {
    Player& p = g_player;
    if (!p.playing.load() || p.paused.load() || !p.video_enabled) {
        return false;
    }
    const u64 now = NowMs();
    std::scoped_lock lk(p.mu);
    if (p.video_ready.empty() || p.video_ready.front().ts_ms > now + 40) {
        return false;
    }
    while (p.video_ready.size() > 1 && p.video_ready[1].ts_ms <= now) {
        p.video_free.push_back(p.video_ready.front().buf);
        p.video_ready.pop_front();
        ++p.video_dropped;
    }
    *out = p.video_ready.front();
    p.video_ready.pop_front();
    if (p.video_current) {
        p.video_free.push_back(p.video_current);
    }
    p.video_current = out->buf;
    ++p.video_shown;
    p.cv.notify_all();
    return true;
}

// SceAvPlayerVideo (16 bytes) or SceAvPlayerVideoEx (80 bytes) at `v`.
void WriteVideoDetails(u8* v, bool ex) {
    const Player& p = g_player;
    const u32 w = AlignUp(p.width, 16), h = p.aligned_height;
    const float aspect = float(p.width) / float(p.height);
    std::memcpy(v + 0, &w, 4);
    std::memcpy(v + 4, &h, 4);
    std::memcpy(v + 8, &aspect, 4);
    std::memcpy(v + 12, "eng", 4);
    if (!ex) {
        return;
    }
    // SceAvPlayerVideoEx as GT7 reads it (layout from ryGT7, which checked it
    // against the game): reserved u32 at 16, crop offsets 20..35, pitch 36,
    // bit depths 40/41, full range 42, frame rate as a double at 48, colour
    // primaries 56 and transfer 60 (zero: BT.709 SDR).
    const double fps = p.fps;
    const u32 crop_right = w - p.width, crop_bottom = h - p.height;
    std::memcpy(v + 24, &crop_right, 4);
    std::memcpy(v + 32, &crop_bottom, 4);
    std::memcpy(v + 36, &p.pitch, 4);
    v[40] = 8; // luma bit depth
    v[41] = 8; // chroma bit depth
    v[42] = 0; // video range
    std::memcpy(v + 48, &fps, 8);
}

// SceAvPlayerAudio, also the head of SceAvPlayerAudioEx.
void WriteAudioDetails(u8* a, u32 bytes) {
    const Player& p = g_player;
    const u16 ch = static_cast<u16>(p.channels);
    std::memcpy(a + 0, &ch, 2);
    std::memcpy(a + 4, &p.sample_rate, 4);
    std::memcpy(a + 8, &bytes, 4);
    std::memcpy(a + 12, "eng", 4);
}

// SceAvPlayerFrameInfo(Ex): pData, reserved[4], timestamp, details.
void WriteFrameHead(u8* o, const u8* data, u64 ts, size_t details) {
    std::memset(o, 0, 24 + details);
    const u64 ptr = reinterpret_cast<u64>(data);
    std::memcpy(o + 0, &ptr, 8);
    std::memcpy(o + 16, &ts, 8);
}

bool sceAvPlayerGetVideoDataEx(void* handle, void* p_info) {
    VideoFrame f;
    if (handle != &g_player || !p_info || !NextVideo(&f)) {
        return false;
    }
    u8* o = static_cast<u8*>(p_info);
    WriteFrameHead(o, f.buf, f.ts_ms, 80);
    WriteVideoDetails(o + 24, true);
    return true;
}

bool sceAvPlayerGetVideoData(void* handle, void* p_info) {
    VideoFrame f;
    if (handle != &g_player || !p_info || !NextVideo(&f)) {
        return false;
    }
    u8* o = static_cast<u8*>(p_info);
    WriteFrameHead(o, f.buf, f.ts_ms, 16);
    WriteVideoDetails(o + 24, false);
    return true;
}

bool sceAvPlayerGetAudioData(void* handle, void* p_info) {
    Player& p = g_player;
    if (handle != &p || !p_info || !p.playing.load() || p.paused.load() || !p.audio_enabled) {
        return false;
    }
    AudioFrame f;
    {
        std::scoped_lock lk(p.mu);
        if (p.audio_ready.empty()) {
            return false;
        }
        f = p.audio_ready.front();
        p.audio_ready.pop_front();
        if (p.audio_current) {
            p.audio_free.push_back(p.audio_current);
        }
        p.audio_current = f.buf;
        p.cv.notify_all();
    }
    u8* o = static_cast<u8*>(p_info);
    WriteFrameHead(o, f.buf, f.ts_ms, 16);
    WriteAudioDetails(o + 24, f.bytes);
    return true;
}

s32 StreamType(u32 stream_id) {
    const Player& p = g_player;
    if (p.has_video && stream_id == kStreamVideo) {
        return 0;
    }
    if (p.has_audio && stream_id == (p.has_video ? kStreamAudio : 0)) {
        return 1;
    }
    return -1;
}

constexpr s32 kErrorInvalidParams = static_cast<s32>(0x806A0001);

s32 sceAvPlayerGetStreamInfo(void* handle, u32 stream_id, void* p_info) {
    DC_INFO("sceAvPlayerGetStreamInfo(%u)", stream_id);
    (void)handle;
    if (!p_info) {
        return 0;
    }
    const Player& p = g_player;
    // SceAvPlayerStreamInfo: type, reserved, details (16), duration, start time.
    u8* o = static_cast<u8*>(p_info);
    std::memset(o, 0, 40);
    if (!p.loaded) { // placeholder for the fallback
        const u32 type = 0, width = 1920, height = 1080;
        const float aspect = 16.0f / 9.0f;
        const u64 duration_ms = 1000;
        std::memcpy(o + 0, &type, 4);
        std::memcpy(o + 8, &width, 4);
        std::memcpy(o + 12, &height, 4);
        std::memcpy(o + 16, &aspect, 4);
        std::memcpy(o + 24, &duration_ms, 8);
        return 0;
    }
    const s32 type = StreamType(stream_id);
    if (type < 0) {
        return kErrorInvalidParams;
    }
    std::memcpy(o + 0, &type, 4);
    if (type == 0) {
        WriteVideoDetails(o + 8, false);
    } else {
        WriteAudioDetails(o + 8, AudioBytes());
    }
    std::memcpy(o + 24, &p.duration_ms, 8);
    return 0;
}

s32 sceAvPlayerGetStreamInfoEx(void* handle, u32 stream_id, void* p_info) {
    DC_INFO("sceAvPlayerGetStreamInfoEx(%u)", stream_id);
    (void)handle;
    const Player& p = g_player;
    if (!p_info || !p.loaded) {
        return 0;
    }
    const s32 type = StreamType(stream_id);
    if (type < 0) {
        return kErrorInvalidParams;
    }
    // SceAvPlayerStreamInfoEx (104 bytes): u64 size (kept), type, reserved,
    // details (80) at 16, duration at 96.
    u8* o = static_cast<u8*>(p_info);
    u64 size = 0;
    std::memcpy(&size, o, 8);
    std::memset(o, 0, 104);
    std::memcpy(o, &size, 8);
    std::memcpy(o + 8, &type, 4);
    if (type == 0) {
        WriteVideoDetails(o + 16, true);
    } else {
        WriteAudioDetails(o + 16, 0);
    }
    std::memcpy(o + 96, &p.duration_ms, 8);
    return 0;
}

s32 sceAvPlayerStreamCount(void* handle) {
    (void)handle;
    const Player& p = g_player;
    if (!p.loaded) {
        return 1;
    }
    return (p.has_video ? 1 : 0) + (p.has_audio ? 1 : 0);
}

s32 sceAvPlayerEnableStream(void* handle, u32 stream_id) {
    (void)handle;
    // Every stream is decoded from AddSource on, so games that never call
    // this still get both. Kept as a log point.
    DC_INFO("sceAvPlayerEnableStream(%u)", stream_id);
    return 0;
}

s32 sceAvPlayerDisableStream(void* handle, u32 stream_id) {
    (void)handle;
    DC_INFO("sceAvPlayerDisableStream(%u)", stream_id);
    Player& p = g_player;
    if (!p.playing.load()) {
        const s32 type = StreamType(stream_id);
        if (type == 0) {
            p.video_enabled = false;
        } else if (type == 1) {
            p.audio_enabled = false;
        }
    }
    return 0;
}

s32 sceAvPlayerChangeStream(void* handle) {
    (void)handle;
    return 0;
}

s32 sceAvPlayerSetLooping(void* handle, bool loop) {
    if (handle == &g_player) {
        g_player.looping = loop;
    }
    return 0;
}

s32 sceAvPlayerSetAvSyncMode(void* handle, u32 mode) {
    (void)handle;
    (void)mode;
    return 0;
}

s32 sceAvPlayerPostInit(void* handle, void* data) {
    (void)handle;
    (void)data;
    return 0;
}

s32 sceAvPlayerSetLogCallback(void* cb, void* user_data) {
    (void)cb;
    (void)user_data;
    return 0;
}

s32 sceAvPlayerSetTrickSpeed(void* handle, s32 speed) {
    (void)handle;
    (void)speed;
    return 0;
}

s32 sceAvPlayerJumpToTime(void* handle, u64 time) {
    (void)handle;
    (void)time;
    return 0;
}

} // namespace

void RegisterAvPlayer() {
    constexpr const char* lib = "libSceAvPlayer";
    DC_HLE(lib, sceAvPlayerInit);
    DC_HLE(lib, sceAvPlayerInitEx);
    DC_HLE(lib, sceAvPlayerClose);
    DC_HLE(lib, sceAvPlayerAddSource);
    DC_HLE(lib, sceAvPlayerAddSourceEx);
    DC_HLE(lib, sceAvPlayerStart);
    DC_HLE(lib, sceAvPlayerStartEx);
    DC_HLE(lib, sceAvPlayerStop);
    DC_HLE(lib, sceAvPlayerPause);
    DC_HLE(lib, sceAvPlayerResume);
    DC_HLE(lib, sceAvPlayerIsActive);
    DC_HLE(lib, sceAvPlayerCurrentTime);
    DC_HLE(lib, sceAvPlayerGetAudioData);
    DC_HLE(lib, sceAvPlayerGetVideoData);
    DC_HLE(lib, sceAvPlayerGetVideoDataEx);
    DC_HLE(lib, sceAvPlayerGetStreamInfo);
    DC_HLE(lib, sceAvPlayerGetStreamInfoEx);
    DC_HLE(lib, sceAvPlayerEnableStream);
    DC_HLE(lib, sceAvPlayerDisableStream);
    DC_HLE(lib, sceAvPlayerChangeStream);
    DC_HLE(lib, sceAvPlayerSetLooping);
    DC_HLE(lib, sceAvPlayerSetAvSyncMode);
    DC_HLE(lib, sceAvPlayerPostInit);
    DC_HLE(lib, sceAvPlayerSetLogCallback);
    DC_HLE(lib, sceAvPlayerSetTrickSpeed);
    DC_HLE(lib, sceAvPlayerStreamCount);
    DC_HLE(lib, sceAvPlayerJumpToTime);
}

} // namespace Hle::AvPlayer
