#include "gpu/metal/mtl_renderer.h"

#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#include <mach/mach.h>
#include <nmmintrin.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/log.h"
#include "core/memory.h"
#include "gpu/label_trace.h"
#include "launcher.h"
#include "gpu/liverpool/pm4.h"
#include "gpu/liverpool/registers.h"
#include "gpu/liverpool/tiling.h"
#include "gpu/metal/mtl_compiler.h"
#include "gpu/metal/mtl_device.h"
#include "gpu/metal/mtl_formats.h"
#include "gpu/shader/msl.h"
#include "gpu/shader/msl_prelude.h"
#include "gpu/shader/program.h"

namespace Gpu::Metal {

using namespace Liverpool;
using Shader::OutType;
using Shader::Program;
using Shader::Stage;
using Shader::TexDim;
using Shader::TexSample;
using Shader::TextureType;

namespace {
std::atomic<bool> g_trace_request{false};
std::atomic<bool> g_capture_request{false};

constexpr u64 kUploadChunk = 4ull << 20;
constexpr u64 kGpuAddrMask =
    (1ull << 40) - 1; // Liverpool virtual addresses are 40 bits
constexpr u32 kNopSizeHint = 0xC0001000; // one-dword NOP carrying a target size

// GPU execution time of a completed command buffer, 0 if Metal gave no
// timestamps: GPUStartTime can be 0 (seen on M5), and End - 0 would be the
// system uptime.
double GpuSeconds(id<MTLCommandBuffer> c) {
  const double start = c.GPUStartTime, end = c.GPUEndTime;
  return start > 0 && end >= start ? end - start : 0;
}

// Key hash for caches: eight bytes per step (keys are mostly descriptor and
// register arrays; a byte loop was a visible share of the CP thread).
u64 Fnv(const void *data, size_t size, u64 h = 0xcbf29ce484222325ull) {
  const auto *p = static_cast<const u8 *>(data);
  size_t i = 0;
  for (; i + 8 <= size; i += 8) {
    u64 v;
    std::memcpy(&v, p + i, 8);
    h = (h ^ v) * 0x100000001b3ull;
    h ^= h >> 32;
  }
  for (; i < size; ++i) {
    h = (h ^ p[i]) * 0x100000001b3ull;
  }
  return h;
}

std::string Hex(u64 v) {
  char b[24];
  std::snprintf(b, sizeof(b), "%#llx", static_cast<unsigned long long>(v));
  return b;
}

template <typename T> u64 Hash(const T &v, u64 seed = 0xcbf29ce484222325ull) {
  return Fnv(&v, sizeof(T), seed);
}

// DC_UNTRACKED=1 (experiment): the guest-memory buffers skip Metal's hazard
// tracking, which serializes every pass that binds them. No fences yet: an
// upper bound for the gain, output may race where the GPU writes guest memory.
static MTLResourceOptions SharedOpts() {
  static const bool untracked = std::getenv("DC_UNTRACKED") != nullptr;
  return MTLResourceStorageModeShared | (untracked ? MTLResourceHazardTrackingModeUntracked : 0);
}

// The range is guest memory the host can read. It may span several adjacent
// mappings: GT7 maps its PRT textures (tile mode 16) one 64 KB page at a time.
bool Readable(u64 addr, u64 size) {
  if (!addr) {
    return false;
  }
  for (u64 at = addr, end = addr + size; at < end;) {
    Core::Memory::Vma vma;
    if (!Core::Memory::Query(at, false, &vma)) {
      return false;
    }
    if (at != addr && vma.type != Core::Memory::VmaType::Direct &&
        vma.type != Core::Memory::VmaType::Flexible) {
      return false;
    }
    at = vma.start + vma.size;
  }
  return true;
}

// Scissor registers: x in [14:0], y in [30:16].
struct Rect {
  s32 x0, y0, x1, y1;
};
Rect ScissorReg(u32 tl, u32 br) {
  return {static_cast<s32>(tl & 0x7FFF), static_cast<s32>((tl >> 16) & 0x7FFF),
          static_cast<s32>(br & 0x7FFF), static_cast<s32>((br >> 16) & 0x7FFF)};
}
Rect Intersect(Rect a, Rect b) {
  return {std::max(a.x0, b.x0), std::max(a.y0, b.y0), std::min(a.x1, b.x1),
          std::min(a.y1, b.y1)};
}

float Half(u32 h) {
  const u32 e = (h >> 10) & 0x1F, m = h & 0x3FF;
  const float v = e == 0    ? std::ldexp(static_cast<float>(m), -24)
                  : e == 31 ? (m ? NAN : INFINITY)
                            : std::ldexp(static_cast<float>(m | 0x400),
                                         static_cast<int>(e) - 25);
  return (h & 0x8000) ? -v : v;
}

// Unsigned small float with a 5-bit exponent and `mbits` of mantissa
// (R11G11B10).
float SmallFloat(u32 v, u32 mbits) {
  const u32 e = (v >> mbits) & 0x1F, m = v & ((1u << mbits) - 1);
  if (e == 0) {
    return std::ldexp(static_cast<float>(m), -14 - static_cast<int>(mbits));
  }
  return e == 31
             ? INFINITY
             : std::ldexp(static_cast<float>(m | (1u << mbits)),
                          static_cast<int>(e) - 15 - static_cast<int>(mbits));
}

// CB_COLOR_CLEAR_WORD0/1 (the target's packed format) as a Metal clear color.
MTLClearColor ClearColor(u32 info, u32 w0, u32 w1) {
  const u32 fmt = ColorInfo::Format(info), nt = ColorInfo::NumberType(info);
  const bool snorm = nt == NumFormat::Snorm,
             integer = nt == NumFormat::Uint || nt == NumFormat::Sint;
  auto n8 = [&](u32 v) {
    return integer ? double(v & 0xFF)
           : snorm ? std::max(-1.0, s8(v) / 127.0)
                   : (v & 0xFF) / 255.0;
  };
  auto n16 = [&](u32 v) {
    v &= 0xFFFF;
    return nt == NumFormat::Float ? double(Half(v))
           : integer              ? double(v)
           : snorm                ? std::max(-1.0, s16(v) / 32767.0)
                                  : v / 65535.0;
  };
  auto f32 = [&](u32 v) {
    float f;
    std::memcpy(&f, &v, 4);
    return integer ? double(v) : double(f);
  };
  double c[4] = {0, 0, 0, integer ? 1.0 : 1.0};
  switch (fmt) {
  case ColorFormat::C8:
    c[0] = n8(w0);
    break;
  case ColorFormat::C8_8:
    c[0] = n8(w0);
    c[1] = n8(w0 >> 8);
    break;
  case ColorFormat::C8_8_8_8:
    for (u32 k = 0; k < 4; ++k) {
      c[k] = n8(w0 >> (8 * k));
    }
    if (ColorInfo::CompSwap(info) == 1) { // BGRA in memory
      std::swap(c[0], c[2]);
    }
    break;
  case ColorFormat::C16:
    c[0] = n16(w0);
    break;
  case ColorFormat::C16_16:
    c[0] = n16(w0);
    c[1] = n16(w0 >> 16);
    break;
  case ColorFormat::C16_16_16_16:
    c[0] = n16(w0);
    c[1] = n16(w0 >> 16);
    c[2] = n16(w1);
    c[3] = n16(w1 >> 16);
    break;
  case ColorFormat::C32:
    c[0] = f32(w0);
    break;
  case ColorFormat::C32_32:
    c[0] = f32(w0);
    c[1] = f32(w1);
    break;
  case ColorFormat::C10_11_11:
    c[0] = SmallFloat(w0 & 0x7FF, 6);
    c[1] = SmallFloat((w0 >> 11) & 0x7FF, 6);
    c[2] = SmallFloat(w0 >> 22, 5);
    break;
  case ColorFormat::C11_11_10:
    c[0] = SmallFloat(w0 & 0x3FF, 5);
    c[1] = SmallFloat((w0 >> 10) & 0x7FF, 6);
    c[2] = SmallFloat(w0 >> 21, 6);
    break;
  case ColorFormat::C2_10_10_10:
    c[0] = (w0 & 0x3FF) / 1023.0;
    c[1] = ((w0 >> 10) & 0x3FF) / 1023.0;
    c[2] = ((w0 >> 20) & 0x3FF) / 1023.0;
    c[3] = (w0 >> 30) / 3.0;
    break;
  default:
    break;
  }
  return MTLClearColorMake(c[0], c[1], c[2], c[3]);
}

struct PipelineKey {
  u64 vs = 0;
  u64 ps = 0;
  u32 color_fmt[8] = {};
  u32 blend[8] = {};
  u32 write_mask = 0;
  u32 depth_fmt = 0;
  u32 shader_mask = 0;
};

constexpr const char *kCopyShader = R"(
#include <metal_stdlib>
using namespace metal;
struct V { float4 pos [[position]]; float2 uv; };
vertex V copy_vs(uint vid [[vertex_id]]) {
    V o;
    o.uv = float2((vid << 1) & 2, vid & 2);
    o.pos = float4(o.uv.x * 2.0 - 1.0, 1.0 - o.uv.y * 2.0, 0.0, 1.0);
    return o;
}
fragment float4 debug_fs() { return float4(1.0, 0.0, 1.0, 1.0); }
fragment float4 dump_fs(V in [[stage_in]], texture2d<float> src [[texture(0)]], constant float2& opt [[buffer(1)]]) {
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    float3 c = (opt.x != 0.0 ? float3(src.sample(s, in.uv).a) * 0.5 : src.sample(s, in.uv).rgb) * opt.y;
    if (any(isnan(c)) || any(isinf(c))) return float4(1.0, 0.0, 1.0, 1.0);
    return float4(max(c, 0.0) / (1.0 + max(c, 0.0)) * 2.0, 1.0);
}
// A depth surface the game wrote as a color image (sample 0 in .r).
struct ZOut { float d [[depth(any)]]; };
fragment ZOut zcopy_fs(V in [[stage_in]], texture2d<float> src [[texture(0)]]) {
    ZOut o;
    o.d = src.read(uint2(in.pos.xy)).r;
    return o;
}
// EDR: the value the game's tonemap feeds its LUT, which clamps it to 1.
kernel void edr_peak(texture2d<float> scene [[texture(0)]], texture2d<float> bloom [[texture(1)]],
                     texture2d<float, access::write> out [[texture(2)]],
                     constant float& exposure [[buffer(0)]], uint2 id [[thread_position_in_grid]]) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float2 uv = (float2(id) + 0.5) / float2(out.get_width(), out.get_height());
    float3 c = bloom.sample(s, uv, level(0)).rgb + exposure * scene.sample(s, uv, level(0)).rgb;
    out.write(float4(max(c.r, max(c.g, c.b))), id);
}
// EDR: keep the peak only where nothing was drawn over the tonemapped image.
kernel void edr_mask(texture2d<float, access::read> base [[texture(0)]],
                     texture2d<float, access::read> final_frame [[texture(1)]],
                     texture2d<float, access::read> peak [[texture(2)]],
                     texture2d<float, access::write> out [[texture(3)]],
                     uint2 id [[thread_position_in_grid]]) {
    float3 d = abs(base.read(id).rgb - final_frame.read(id).rgb);
    out.write(float4(all(d < 2.0 / 255.0) ? peak.read(id).r : 0.0), id);
}
kernel void fill32(device uint* dst [[buffer(0)]], constant uint2& vc [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    if (i < vc.y) dst[i] = vc.x;
}
// HTILE (depth with stencil) from the real depth: per 8x8 tile, ZRANGE base
// (min Z, 14 bits, [31:18]) and a 6-bit code ([17:12]) whose delta table is
// the one the game decodes (max Z = base + delta + 1); ZMASK expanded. The
// address is the tile swizzle the game's light culling computes.
// p: width and height in tiles, dwords available, clear depth bits when the
// surface reads as its clear value (else ~0).
constant uint kZDelta[64] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    17, 19, 21, 23, 25, 27, 29, 31, 35, 39, 43, 47, 51, 55, 59, 63, 79, 95, 111,
    127, 159, 191, 223, 255, 319, 383, 447, 511, 639, 767, 895, 1023, 1279, 1535,
    1791, 2047, 2559, 3071, 3583, 4095, 5119, 6143, 7167, 8191, 10239, 12287, 14335,
    16383};
kernel void htile_build(depth2d<float> z [[texture(0)]], device uint* dst [[buffer(0)]],
                        constant uint4& p [[buffer(1)]], uint2 t [[thread_position_in_grid]]) {
    if (t.x >= p.x || t.y >= p.y) return;
    float lo = 1.0, hi = 0.0;
    if (p.w != 0xffffffffu) {
        lo = hi = as_type<float>(p.w);
    } else {
        uint2 last = uint2(z.get_width() - 1, z.get_height() - 1);
        for (uint j = 0; j < 8; ++j) {
            for (uint i = 0; i < 8; ++i) {
                float d = z.read(min(t * 8u + uint2(i, j), last));
                lo = min(lo, d);
                hi = max(hi, d);
            }
        }
    }
    uint base = uint(floor(saturate(lo) * 16383.0));
    uint need = uint(ceil(saturate(hi) * 16383.0));
    uint c = 0;
    while (c < 63 && base + kZDelta[c] + 1 < need) ++c;
    uint x = t.x, y = t.y, mw = (p.x + 63) >> 6;
    uint low = ((32 * ((y >> 2) & 15) + 2 * ((x >> 2) & 15)) & ~3u) | (((x >> 1) ^ y) & 1) |
               (((x >> 1) & 1) << 1);
    uint a = 4 * (((y >> 6) * mw + (x >> 6)) * 512 + low);
    uint pipe = ((x ^ y ^ (x >> 1)) & 1) | ((((x >> 1) ^ (y >> 1)) & 1) << 1) |
                ((((x >> 2) ^ (y >> 2)) & 1) << 2);
    uint i = ((a & 0xff) + pipe * 256 + (a >> 8) * 2048) >> 2;
    if (i < p.z) dst[i] = (base << 18) | (c << 12) | 0xFu;
}
// DC_SCAN_MAX: the largest finite-or-not channel of a target (all slices and
// levels), then where it is. o: [max bits, x, y, slice, level, r, g, b].
static inline uint scan_bits(float4 c) {
    float m = max(c.x, max(c.y, c.z));
    return isnan(m) || isinf(m) ? 0x7f800000u : (m > 0.0 ? as_type<uint>(m) : 0u);
}
kernel void scan_val(texture2d_array<float, access::read> t [[texture(0)]],
                     device atomic_uint* o [[buffer(0)]], constant uint& lv [[buffer(1)]],
                     uint3 g [[thread_position_in_grid]]) {
    if (g.x >= t.get_width(lv) || g.y >= t.get_height(lv) || g.z >= t.get_array_size()) return;
    atomic_fetch_max_explicit(&o[0], scan_bits(t.read(g.xy, g.z, lv)), memory_order_relaxed);
}
kernel void scan_loc(texture2d_array<float, access::read> t [[texture(0)]],
                     device uint* o [[buffer(0)]], constant uint& lv [[buffer(1)]],
                     uint3 g [[thread_position_in_grid]]) {
    if (g.x >= t.get_width(lv) || g.y >= t.get_height(lv) || g.z >= t.get_array_size()) return;
    float4 c = t.read(g.xy, g.z, lv);
    uint b = scan_bits(c);
    if (b != 0u && b == o[0]) {
        o[1] = g.x; o[2] = g.y; o[3] = g.z; o[4] = lv;
        o[5] = as_type<uint>(c.x); o[6] = as_type<uint>(c.y); o[7] = as_type<uint>(c.z);
    }
}
fragment float4 copy_fs(V in [[stage_in]], texture2d<float> src [[texture(0)]],
                        constant float2& scale [[buffer(0)]]) {
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    return float4(src.sample(s, in.uv * scale).rgb, 1.0);
}
)";

} // namespace

struct MetalRenderer::Impl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  std::mutex lock;

  // ---- guest memory view ----
  id<MTLBuffer> mem[2] = {nil, nil};
  u64 mem_split = 0;
  id<MTLBuffer> mem_map = nil;
  u64 mem_map_gen = ~0ull;
  std::vector<Core::Memory::DirectRange> mem_ranges; // sorted by start
  id<MTLComputePipelineState> fill_pipeline = nil;
  id<MTLComputePipelineState> htile_pipeline = nil;
  id<MTLComputePipelineState> scan_val_pipeline = nil, scan_loc_pipeline = nil;
  // DC_SCAN_MAX=<addr>[,<addr>...] (or all targets when F12 traces a frame):
  // in the traced frame, after every pass or dispatch writing a color target,
  // log its largest value and where it is (finds where a blown-up value appears).
  std::vector<u64> scan_addrs;
  bool scan_all = false;
  std::unordered_map<const void *, u64> scan_seen; // target -> written seq
  std::string scan_label;

  // ---- recording ----
  id<MTLCommandBuffer> cmd = nil;
  id<MTLCommandBuffer> last_committed = nil;
  id<MTLRenderCommandEncoder> enc = nil;
  // Per-draw scratch reused across draws (no allocation per draw).
  std::vector<Shader::MslOptions::VertexAttr> draw_vattrs;
  // What `enc` has bound, so draws skip Objective-C calls (slow under
  // Rosetta) that would not change anything. Reset with each encoder.
  struct BoundState {
    __unsafe_unretained id<MTLBuffer> buf[2][31] = {};
    u64 off[2][31] = {};
    __unsafe_unretained id<MTLTexture> tex[2][32] = {};
    __unsafe_unretained id<MTLSamplerState> smp[2][16] = {};
    u8 bytes[2][3][64] = {}; // user data, draw constants (vertex), linear bases per stage
    bool has_bytes[2][3] = {};
    u32 bytes_len[2][3] = {};
    // Fixed-function state. The encoder retains what it was given, so a
    // pointer match cannot be a freed object's reused address.
    __unsafe_unretained id<MTLRenderPipelineState> pso = nil;
    __unsafe_unretained id<MTLDepthStencilState> depth_state = nil;
    u32 viewport[2] = {}; // width, height
    bool has_viewport = false;
    MTLScissorRect scissor = {};
    bool has_scissor = false;
    u32 stencil_ref[2] = {};
    bool has_stencil_ref = false;
    int clip = -1, cull = -1, winding = -1;
    float depth_bias[3] = {};
    bool has_depth_bias = false;
    float blend[4] = {};
    bool has_blend = false;
  } bound;
  // DC_NO_STATE_CACHE=1: set the fixed-function state on every draw (A/B).
  const bool state_cache = std::getenv("DC_NO_STATE_CACHE") == nullptr;
  // stage 0 = vertex, 1 = fragment
  void BindBuffer(u32 st, id<MTLBuffer> b, u64 off, u32 idx) {
    if (bound.buf[st][idx] == b && bound.off[st][idx] == off) {
      return;
    }
    bound.buf[st][idx] = b;
    bound.off[st][idx] = off;
    if (st) {
      [enc setFragmentBuffer:b offset:off atIndex:idx];
    } else {
      [enc setVertexBuffer:b offset:off atIndex:idx];
    }
  }
  void BindTexture(u32 st, id<MTLTexture> t, u32 idx) {
    if (idx < 32 && bound.tex[st][idx] == t && t) {
      return;
    }
    if (idx < 32) {
      bound.tex[st][idx] = t;
    }
    if (st) {
      [enc setFragmentTexture:t atIndex:idx];
    } else {
      [enc setVertexTexture:t atIndex:idx];
    }
  }
  void BindSampler(u32 st, id<MTLSamplerState> sm, u32 idx) {
    if (bound.smp[st][idx] == sm && sm) {
      return;
    }
    bound.smp[st][idx] = sm;
    if (st) {
      [enc setFragmentSamplerState:sm atIndex:idx];
    } else {
      [enc setVertexSamplerState:sm atIndex:idx];
    }
  }
  // Bytes slot `which` (0 user data, 1 draw constants) at buffer index idx.
  void BindBytes(u32 st, u32 which, const void *data, u32 len, u32 idx) {
    if (len <= 64 && bound.has_bytes[st][which] && bound.bytes_len[st][which] == len &&
        !std::memcmp(bound.bytes[st][which], data, len)) {
      return;
    }
    if (len <= 64) {
      std::memcpy(bound.bytes[st][which], data, len);
    }
    bound.has_bytes[st][which] = len <= 64;
    bound.bytes_len[st][which] = len;
    bound.buf[st][idx] = nil; // bytes replace any buffer there
    if (st) {
      [enc setFragmentBytes:data length:len atIndex:idx];
    } else {
      [enc setVertexBytes:data length:len atIndex:idx];
    }
  }
  struct PassState {
    u64 color[8] = {};
    u32 slice[8] = {};
    u64 depth = 0;
    u32 depth_slice = 0;
    u32 layers = 0;
    bool operator==(const PassState &) const = default;
  } pass;
  u32 draw_layers = 0; // slices the VS of the current draw picks among; 0 = one
  u32 pass_draws = 0;  // draws recorded in the open render pass
  u32 pass_width = 0, pass_height = 0;
  std::vector<id<MTLBuffer>> chunks;
  id<MTLBuffer> chunk = nil;
  u64 chunk_off = 0;
  u32 work_since_commit = 0; // draws + dispatches in the open command buffer
  // Draws + dispatches per command buffer. The guest waits for the frame's
  // flip label before submitting the next frame, so the work recorded after
  // the last commit is pure latency: 400 left a ~5 ms GPU tail per frame
  // (frames over 33 ms, ~27 fps), 60 leaves ~2 ms (~29.4 fps) at no GPU cost.
  // DC_COMMIT_EVERY=<n> overrides.
  const u32 kCommitEvery = [] {
    const char *v = std::getenv("DC_COMMIT_EVERY");
    return v ? static_cast<u32>(std::atoi(v)) : 60u;
  }();

  // Signals: an event value encoded between passes, so fences do not cut
  // command buffers. Callbacks run in order on the listener's queue.
  id<MTLSharedEvent> signal_event = nil;
  MTLSharedEventListener *signal_listener = nil;
  u64 signal_value = 0;
  u64 signal_pending = 0; // value of callbacks waiting for FlushSignal, 0: none
  bool signal_batch = true;
  std::mutex signal_lock;
  std::deque<std::pair<u64, std::function<void()>>> signal_queue;

  // ---- caches ----
  // A program is reused only while its whole code (and fetch subroutine) is
  // unchanged: streaming puts other shaders at freed addresses, and GCN
  // programs often share their first instructions (shadPS4 keys shaders by a
  // hash of all of the code).
  struct ShaderEntry {
    std::unique_ptr<Program> prog;
    std::vector<u32> code;
    std::vector<u32> fetch;
    u64 checked_frame = ~0ull; // frame in which code and fetch last matched memory
  };
  std::unordered_map<u64, std::vector<ShaderEntry>> shaders;
  std::unordered_map<u64, id<MTLRenderPipelineState>> pipelines;

  // Pipelines compile on `compiler`'s threads; draws whose pipeline is not
  // ready yet are skipped (DC_SYNC_SHADERS waits instead), dispatches wait.
  std::unique_ptr<ShaderCompiler> compiler;
  bool sync_shaders = false;
  std::unordered_set<u64> pending;
  std::mutex done_lock;
  std::condition_variable done_cv;
  std::vector<std::pair<u64, id<MTLRenderPipelineState>>> done_render;
  std::vector<std::pair<u64, id<MTLComputePipelineState>>> done_compute;
  std::unordered_map<u64, id<MTLDepthStencilState>> depth_states;
  std::unordered_map<u64, id<MTLSamplerState>> samplers;

  // Guest surfaces rendered on the host GPU stay there. The same address can
  // be bound with different sizes or formats (aliased memory); each variant
  // keeps its own texture, and readers get the one written last.
  struct RenderTarget {
    id<MTLTexture> tex = nil;
    u64 addr = 0;
    u32 width = 0, height = 0;
    u32 depth_or_layers = 1;
    TexDim dim = TexDim::Tex2D;
    bool initialized = false;
    id<MTLTexture> bounds_copy =
        nil; // depth as of bounds_seq, for the depth bounds test
    u64 bounds_seq = 0;
    bool clear_pending = false; // fast-cleared (CMASK/HTILE): next pass clears
    MTLClearColor clear_color{};
    double clear_depth = 1.0;
    u32 clear_stencil = 0;
    u64 written = 0;    // write_seq of the last pass/dispatch writing it
    u32 bind_slice = 0; // array slice the CB registers point at (CB_COLOR_VIEW)
    u32 bind_level = 0; // mip level, when the CB address is a level of this one
    u32 levels = 1; // mip levels (compute shaders build mip chains in place)
    std::unordered_map<u64, id<MTLTexture>>
        views; // (base, count, format) -> view
    RenderTarget *view_of = nullptr; // sRGB/linear twin whose texture this views
    bool has_crops = false;          // a shorter T# reads it (see CropOf)
    u32 comp_swap = 0;               // CB_INFO.COMP_SWAP it was last bound with
    bool three_channels = false;     // its format has no alpha (10_11_11, 11_11_10)
  };
  using TargetMap =
      std::unordered_map<u64, std::vector<std::unique_ptr<RenderTarget>>>;
  TargetMap color_targets; // by address
  TargetMap depth_targets;
  u64 write_seq = 0;
  // Metal allows 2048 array layers / 3D depth; more comes from a garbage T#.
  // DC_NO_LAYER_CLAMP=1: pass the count through (old behaviour, for A/B runs).
  static u32 LayerLimit(u32 layers) {
    static const bool no_clamp = std::getenv("DC_NO_LAYER_CLAMP") != nullptr;
    if (layers > 2048) {
      static bool warned = false;
      if (!warned) {
        warned = true;
        DC_WARN("gpu: texture with %u layers clamped to 2048%s", layers, no_clamp ? " (DC_NO_LAYER_CLAMP: not)" : "");
      }
      if (!no_clamp) {
        return 2048;
      }
    }
    return layers;
  }
  // DC_LOG_RESTORED=1 (frame replay): write_seq range LoadTargets handed out;
  // a sampled target still inside it was never rewritten in this loop.
  u64 restored_lo = 0, restored_hi = 0, dbg_shader = 0;
  std::set<std::tuple<u64, u32, u32, u64>> restored_logged;
  void NoteRestoredRead(const RenderTarget &rt, u64 addr, bool depth) {
    static const bool on = std::getenv("DC_LOG_RESTORED") != nullptr;
    if (!on || rt.written <= restored_lo || rt.written > restored_hi ||
        !restored_logged.insert({addr, rt.width, rt.height, dbg_shader}).second) {
      return;
    }
    std::string others;
    if (auto it = (depth ? depth_targets : color_targets).find(addr);
        it != (depth ? depth_targets : color_targets).end()) {
      for (auto &o : it->second) {
        if (o.get() == &rt) continue;
        others += " " + std::to_string(o->width) + "x" + std::to_string(o->height) + "/f" +
                  std::to_string(u32(o->tex.pixelFormat)) +
                  (o->written > restored_hi ? "/NEW" : "/old");
      }
    }
    DC_INFO("restored-read: %s %#llx %ux%u f%u by shader %#llx; others:%s", depth ? "depth" : "color",
            (unsigned long long)addr, rt.width, rt.height, u32(rt.tex.pixelFormat),
            (unsigned long long)dbg_shader, others.c_str());
  }
  std::unordered_map<u64, u32> storage_swizzles; // address -> selectors of its storage writes
  // Crops of color targets read through a shorter T#, by (address, height).
  std::unordered_map<u64, std::unique_ptr<RenderTarget>> target_crops;

  // GetTexture results for one T# descriptor. The answer changes only when a
  // render target is created, grown or replaced, or a mip alias appears; the
  // same descriptors come back for thousands of draws per frame. Writes to a
  // target keep the same texture object, so they leave the cache alone.
  // DC_NO_TEX_CACHE: resolve every descriptor (A/B).
  bool tex_cache = true;
  // DC_TEX_CACHE_CHECK: on a cache hit, also resolve the descriptor the slow
  // way and report a mismatch. Deterministic, unlike comparing two runs.
  bool tex_cache_check = false;
  u64 tex_cache_mismatches = 0, tex_cache_checks = 0;
  struct TextureCacheEntry {
    id<MTLTexture> tex = nil;
    TextureType type;
    u64 epoch = 0;
  };
  std::unordered_map<u64, TextureCacheEntry> texture_cache;
  u64 texture_cache_frame = ~0ull;
  u64 texture_epoch = 0;

  // A target was written: Latest() picks the most recently written surface at
  // an address, so when several share one the answer changes. With one target
  // there it cannot, and the cache stays valid.
  // Target address -> write_seq of a compute memset over it. GT7 clears transient
  // images that way, and a host target there keeps the pixels of the last surface.
  std::unordered_map<u64, u64> memory_fills;
  // Linear images in GT7's per-frame memory, filled from the guest memory
  // buffer on the GPU timeline at each bind (see GetTexture).
  std::unordered_map<u64, id<MTLTexture>> gpu_linear_tex;
  // dispatch_seq when each of them was last copied: no compute since, no copy
  // (a copy per bind split the render pass at every draw: 25 -> 77 ms/frame).
  std::unordered_map<u64, std::pair<u64, u64>> gpu_linear_copied; // key -> (dispatch_seq, frame)
  u64 dispatch_seq = 0;

  void NoteTargetWritten(const RenderTarget &rt) {
    auto c = color_targets.find(rt.addr);
    if (c == color_targets.end() || c->second.size() > 1) {
      ++texture_epoch;
      return;
    }
    auto d = depth_targets.find(rt.addr);
    if (d != depth_targets.end() && d->second.size() > 1) {
      ++texture_epoch;
    }
  }

  // Fast-clear metadata surfaces (CMASK / HTILE) -> the surface they describe.
  struct Meta {
    u64 target = 0;
    bool depth = false;
    u32 width = 0, height = 0; // of the surface: one address can hold several
  };
  std::unordered_map<u64, Meta> metas;

  // Render targets whose address is a mip level of a texture's memory (a T#
  // with levels): the game renders straight into that level, e.g. the 256x256
  // cubes of level 1 of the 512x512 environment cube array. Like shadPS4's
  // ImageInfo::MipOf, such a target is that level of the larger image.
  struct MipAlias {
    u64 parent = 0;
    u32 level = 0;
    u32 slice = 0; // array slice (cube face) of that level the address starts
    u32 width = 0, height = 0;
    MTLPixelFormat fmt = MTLPixelFormatInvalid;
  };
  std::unordered_map<u64, MipAlias> mip_alias; // by level address
  u64 resolving_shader = 0; // shader whose resources are being resolved (logs)
  std::vector<std::unique_ptr<RenderTarget>> retired; // merged into a parent

  // Render target sizes the Gnm driver hides in a NOP after the CB/DB
  // register writes (width | height << 16); 0 when absent.
  u32 cb_extent[8] = {};
  u32 db_extent = 0;

  // Guest textures are re-uploaded when their memory changes. Each keeps
  // hashes of its 64 KB chunks; a texture that never changed has one chunk
  // checked per frame in turn, one that did (glyph caches, streamed or
  // CPU-animated textures) is checked in full every frame it is used.
  static constexpr u64 kChunk = 64 << 10;
  struct TextureEntry {
    id<MTLTexture> tex = nil;
    id<MTLTexture> base_view = nil; // levels from the T# BASE_LEVEL
    std::vector<u64> chunks;
    u32 next_chunk = 0;
    bool hot = false;
    u64 frame = 0;
    u64 addr = 0;
    u64 size = 0;
  };
  std::unordered_map<u64, TextureEntry> textures;
  std::unordered_map<u32, id<MTLTexture>> dummy_textures;

  // Streaming reuses texture memory with new descriptors, and each one gets
  // its own entry: without eviction every track load kept the last one's
  // textures (37 GB after a few events). Entries unused for kTextureMaxAge
  // frames go; a later use uploads again. Command buffers retain what they
  // bind, so dropping ours is safe while the GPU still reads it.
  // DC_TEX_MAX_AGE=<frames> overrides (0 keeps everything, the old way).
  // Track loads create thousands of textures within seconds (15 GB peaks),
  // so past kTextureBudget anything unused for a couple of frames goes at
  // once. The budget is 1/8 of RAM, at least 1 GB (a race uses ~1.2 GB);
  // DC_TEX_BUDGET_MB=<MB> overrides.
  static constexpr u64 kTextureSweepEvery = 150;
  static constexpr u64 kTextureMinAge = 2;
  const u64 kTextureMaxAge = [] {
    const char *v = std::getenv("DC_TEX_MAX_AGE");
    return v ? std::strtoull(v, nullptr, 10) : 600ull;
  }();
  // DC_TEX_HOT_MB=<MB>: textures at least this big are hashed in full on
  // every use from the start (debug: GPU-or-CPU rewritten pools such as the
  // GT7 lighting probe atlases would otherwise lag by size/64 KB frames).
  const u64 kTextureHotBytes = [] {
    const char *v = std::getenv("DC_TEX_HOT_MB");
    return v ? std::strtoull(v, nullptr, 10) << 20 : ~0ull;
  }();
  const u64 kTextureBudget = [] {
    if (const char *v = std::getenv("DC_TEX_BUDGET_MB")) {
      return std::strtoull(v, nullptr, 10) << 20;
    }
    return std::max<u64>(NSProcessInfo.processInfo.physicalMemory / 8, 1ull << 30);
  }();
  u64 texture_bytes = 0;
  u64 texture_bytes_peak = 0; // since the last "mem:" line
  u64 texture_pressure_frame = ~0ull;
  // Freeing a Metal texture is a kernel call: the sweep that drops a track
  // load's textures (~3 GB) took 100-175 ms on the graphics thread, twice
  // per race start. The entries die on this queue instead.
  // DC_NO_TEX_REAPER=1 frees them in place (A/B).
  dispatch_queue_t texture_reaper = dispatch_queue_create(
      "gpu texture reaper",
      dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL,
                                              QOS_CLASS_UTILITY, 0));
  const bool texture_reaper_off = std::getenv("DC_NO_TEX_REAPER") != nullptr;
  void EvictTextures(u64 max_age) {
    if (kTextureMaxAge == 0) {
      return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const u64 bytes_before = texture_bytes;
    auto dead = std::make_shared<std::vector<TextureEntry>>();
    for (auto it = textures.begin(); it != textures.end();) {
      if (frame_count - it->second.frame > max_age) {
        texture_bytes -= it->second.tex ? it->second.tex.allocatedSize : 0;
        dead->push_back(std::move(it->second));
        it = textures.erase(it);
      } else {
        ++it;
      }
    }
    const size_t evicted = dead->size();
    if (texture_reaper_off) {
      dead->clear();
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    if (ms > 2) {
      DC_INFO("gpu: evicted %zu textures (%llu MB) in %.1f ms", evicted,
              (bytes_before - texture_bytes) >> 20, ms);
    }
    if (!dead->empty()) {
      dispatch_async(texture_reaper, ^{
        dead->clear();
      });
    }
  }
  // After a new texture: over budget, drop what the last frames did not use
  // (once per frame; what is left is in use and stays).
  void TexturePressure() {
    if (texture_bytes > kTextureBudget && texture_pressure_frame != frame_count) {
      texture_pressure_frame = frame_count;
      EvictTextures(kTextureMinAge);
    }
  }
  id<MTLSamplerState> default_sampler = nil;
  id<MTLBuffer> quad_indices = nil;
  id<MTLBuffer> gds = nil; // 64 KB global data share

  // ---- constant-engine snapshots ----
  // The CE dumps resource tables into a small ring that the guest rewrites
  // while draws that read it may still be queued on the host GPU. A draw
  // whose user data points into a dump gets a pointer to a copy in `arena`
  // (guest memory the renderer owns) taken when the draw is recorded.
  struct ConstDump {
    u32 bytes = 0;
    u64 gen = 0;
  };
  // const_dumps and dump_gen have their own lock: the CE notes dumps without
  // waiting for `lock`, which the compute queues hold while they record.
  std::mutex dump_lock;
  std::map<u64, ConstDump> const_dumps; // by guest address
  u64 dump_gen = 0;
  struct Snapshot {
    u64 gen = 0;
    u64 va = 0;
    u64 start = 0; // arena_head where the copy begins
  };
  std::unordered_map<u64, Snapshot>
      snapshots; // dump address -> its latest copy
  static constexpr u64 kArenaSize = 16ull << 20;
  u64 arena_va = 0;
  u64 arena_head = 0; // bytes handed out so far (wraps modulo kArenaSize)
  std::atomic<u64> arena_released{0}; // head values the GPU is done with
  u64 arena_commit_head = 0; // head when the last command buffer was committed

  // ---- presentation ----
  id<MTLRenderPipelineState> copy_pipeline = nil;
  id<MTLRenderPipelineState> dump_pipeline =
      nil; // DC_DUMP_PASSES: x / (1 + x), NaN/Inf magenta
  // DepthFromColor: per depth format (Depth32Float, Depth32Float_Stencil8).
  id<MTLRenderPipelineState> zcopy_pipeline = nil, zcopy_pipeline_s8 = nil;
  id<MTLDepthStencilState> zcopy_ds = nil;
  id<MTLTexture> frames[3] = {nil, nil, nil};
  u32 frame_index = 0;
  std::atomic<void *> latest_frame{nullptr};
  // EDR highlights, see EdrCapture. Set once before the renderer is in use.
  bool edr_on = false;
  bool edr_armed = false, edr_have = false;
  u64 edr_final = 0, edr_scene = 0, edr_bloom = 0;
  float edr_exposure = 0;
  id<MTLComputePipelineState> edr_peak_pipeline = nil, edr_mask_pipeline = nil;
  id<MTLTexture> edr_base = nil, edr_base_view = nil, edr_peak = nil;
  id<MTLTexture> edr_out[3] = {nil, nil, nil};
  std::atomic<void *> latest_boost{nullptr};
  u64 frame_count = 0;

  // ---- diagnostics ----
  const char *dump_dir = nullptr;
  const char *dump_frames = nullptr; // DC_DUMP_FRAMES=dir: flips as PPMs
  u64 dump_every = 1;                // DC_DUMP_EVERY=n: only every n-th flip
  bool key_trace_only = false;       // F12 picked the dump directory: dump the traced frame only
  std::string key_trace_dir;
  u64 capture_end = 0; // F11: flip that stops the running GPU capture
  u64 commits = 0;     // command buffers committed (user data patches go stale)
  bool debug_clear = false; // DC_DEBUG_CLEAR: new targets start dark blue
  u64 dump_texture =
      0; // DC_DUMP_TEXTURE=<addr>: write its uploads next to the frame dumps
  u64 debug_vsout = 0; // DC_DEBUG_VSOUT=<vs address>: dump its first positions
  id<MTLBuffer> debug_vs_buf = nil;
  bool debug_vs_domain = false;
  id<MTLFunction> debug_fs =
      nil; // DC_DEBUG_FLAT: every pixel shader outputs magenta
  bool trace = false;
  bool gds_readback = false; // DC_GDS_READBACK: log GDS->memory copies in the traced frame
  // Linear per-frame images bound in the traced flip: re-counted at the next
  // flips (does their producer run after their consumer?).
  struct LinearWatch { const u8 *src; u64 n; u64 addr; };
  std::vector<LinearWatch> linear_watch;
  std::vector<std::pair<u64, std::vector<std::tuple<u32, u64, u64>>>> peek_cs_table_pending;
  u64 find_addr = 0;         // DC_FIND_ADDR (see FindRefs)
  u64 find_size_min = 0, find_size_max = 0; // DC_FIND_SIZE (see FindRefs)
  u64 peek_vs = 0;           // DC_PEEK_VS (see Draw)
  // DC_RAIN_DIAG: one F12/P in the rain traces two flips. The first peeks
  // the rain drop VS; its buffer holding only FLT_MAX (the drop colors, white
  // screen) becomes DC_FIND_ADDR for the second, which lists every draw and
  // dispatch referencing it.
  bool rain_diag = false;
  static constexpr u64 kRainDropVs = 0x7003086d00;
  static constexpr u64 kRainDropLightCs = 0x7003086100; // writes the drop colors
  u64 trace_frame = 0; // DC_TRACE_FRAME=n: DC_GPU_TRACE for flip n only
  bool dump_passes =
      false; // DC_DUMP_PASSES: with DC_TRACE_FRAME, every pass's targets
  struct PassDump {
    id<MTLBuffer> buf;
    u32 w, h;
    std::string name;
  };
  std::vector<PassDump> pass_dumps;
  std::vector<u64> peek_addrs; // DC_PEEK_CS=<cs>: its V# buffers, printed after
                               // the traced frame
  // DC_PROBE=<shader>:<pc>:<x>,<y>: registers of one fragment (compute: group
  // x, thread y) before instruction pc, for every draw/dispatch of the traced
  // frame using that shader.
  struct Probe {
    u64 shader = 0;
    u32 pc = ~0u, x = 0, y = 0;
  } probe;
  id<MTLBuffer> probe_buf = nil;
  std::vector<std::string>
      probe_slots; // what each slot of probe_buf belongs to
  static constexpr u32 kProbeSlots = 4096;
  id<MTLTexture> pass_tex[9] = {};
  u64 pass_addr[9] = {};
  bool map_hoist = true; // DC_NO_MAP_HOIST: shaders read the page-table header per access
  bool stats = false;   // DC_GPU_STATS: per-second GPU time and submit counts
  bool profile = false; // DC_GPU_PROFILE: GPU time per shader (one command
                        // buffer per call)
  struct ProfileEntry {
    double seconds = 0;
    u64 calls = 0;
    u64 verts = 0; // pass timing: vertices (indices) x instances drawn
  };
  std::mutex profile_lock;
  std::unordered_map<std::string, ProfileEntry> profile_data;
  // DC_PASS_TIMING (with DC_GPU_STATS): GPU timestamps at the start and end of
  // every render pass and dispatch, from stage-boundary counters, so nothing
  // is split or committed early. Reported per second as ms per frame.
  bool pass_timing = false;
  static constexpr u32 kTsPerBuf = 4096; // Metal caps a sample buffer at 32 KB
  static constexpr u32 kTsBufs = 8;
  id<MTLCounterSampleBuffer> ts_buf[kTsBufs] = {};
  u32 ts_next = 0; // next group of 4 samples in the ring
  struct TsPass {
    u32 sample;
    u64 shader; // first pixel shader of a render pass, or the compute shader
    u32 draws;
    u64 verts;
    u32 w, h;
    bool compute;
  };
  std::vector<TsPass> ts_cmd; // passes recorded into `cmd`
  std::mutex ts_lock;
  std::unordered_map<std::string, ProfileEntry> ts_data;
  double ts_vs_s = 0, ts_fs_s = 0, ts_cs_s = 0;
  MTLTimestamp ts_cpu0 = 0, ts_gpu0 = 0;
  double ts_ns_per_tick = 1.0;

  u32 TsBegin(u64 shader, u32 w, u32 h, bool compute) {
    const u32 sample = ts_next;
    ts_next = (ts_next + 4) % (kTsPerBuf * kTsBufs);
    ts_cmd.push_back({sample, shader, 0, 0, w, h, compute});
    return sample;
  }
  std::atomic<u64> gpu_ns{0};
  // Frame rate and GPU time logged every 10 s, for bug reports.
  std::atomic<u64> perf_gpu_ns{0};
  u64 perf_frames = 0;
  std::chrono::steady_clock::time_point perf_t0{};

  // "mem:" line next to "perf:": what the process and the host caches hold,
  // in MB, so a log shows which one grows.
  void LogMemory() {
    task_vm_info_data_t vm{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    const u64 footprint =
        task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&vm), &count) == KERN_SUCCESS
            ? vm.phys_footprint
            : 0;
    const u64 tex_bytes = texture_bytes;
    const u64 tex_peak = std::max(texture_bytes_peak, texture_bytes);
    texture_bytes_peak = 0;
    auto targets = [](const TargetMap &m, u64 &n) {
      u64 bytes = 0;
      for (const auto &[a, v] : m) {
        for (const auto &rt : v) {
          ++n;
          if (rt->tex && !rt->view_of) {
            bytes += rt->tex.allocatedSize;
          }
          bytes += rt->bounds_copy ? rt->bounds_copy.allocatedSize : 0;
        }
      }
      return bytes;
    };
    u64 n_color = 0, n_depth = 0;
    const u64 color_bytes = targets(color_targets, n_color);
    const u64 depth_bytes = targets(depth_targets, n_depth);
    u64 retired_bytes = 0;
    for (const auto &rt : retired) {
      retired_bytes += rt->tex && !rt->view_of ? rt->tex.allocatedSize : 0;
    }
    size_t n_shaders = 0;
    for (const auto &[a, v] : shaders) {
      n_shaders += v.size();
    }
    constexpr double kMB = 1.0 / (1 << 20);
    DC_INFO("mem: footprint %.0f, metal %.0f, textures %zu/%.0f (peak %.0f), color %llu/%.0f, depth %llu/%.0f, "
            "retired %zu/%.0f, shaders %zu, pipelines %zu",
            footprint * kMB, device.currentAllocatedSize * kMB, textures.size(), tex_bytes * kMB, tex_peak * kMB,
            (unsigned long long)n_color, color_bytes * kMB, (unsigned long long)n_depth, depth_bytes * kMB,
            retired.size(), retired_bytes * kMB, n_shaders, pipelines.size());
  }
  mach_port_t draw_thread = 0; // DC_GPU_STATS: whose CPU time to report
  u64 stat_hashed = 0;
  u64 stat_commits = 0, stat_waits = 0, stat_dispatches = 0, stat_draws = 0;
  double stat_wait_s = 0;
  // DC_CPU_PROFILE: where the CPU time of a frame goes.
  bool cpu_profile = false;
  double cpu_draw_s = 0, cpu_resolve_s = 0, cpu_hash_s = 0, cpu_upload_s = 0;
  // DC_CPU_PROFILE: time of each section of Draw (see DrawLap).
  static constexpr const char *kDrawSections[] = {"entry", "attachments", "shaders", "resources",
                                                 "functions", "pipeline", "fixed", "bindings"};
  double cpu_section_s[8] = {};
  std::chrono::steady_clock::time_point draw_lap;
  u32 draw_section = 0;
  void DrawLap(u32 next) {
    if (!cpu_profile) {
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (next) {
      cpu_section_s[draw_section] += std::chrono::duration<double>(now - draw_lap).count();
    }
    draw_lap = now;
    draw_section = next;
  }
  double cpu_other_s = 0; // CPU time outside draws (CP parsing, submits, flips)
  double cpu_gettex_s = 0; // part of resolve spent in GetTexture
  double cpu_sampler_s = 0; // part of resolve spent in GetSampler
  double cpu_sgpr_s = 0;    // part of resolve spent resolving S#/T# dwords
  u64 cpu_gettex_calls = 0, cpu_gettex_hits = 0;
  double cpu_flip_s = 0, cpu_commit_s = 0, cpu_dma_s = 0;
  double cpu_linear_s = 0; // LinearTable
  u64 cpu_resolve_calls = 0, cpu_resolve_hits = 0;
  u64 cpu_linear_calls = 0, cpu_linear_bases = 0;
  double cpu_notework_s = 0;
  // Times a scope into `slot` (no-op unless DC_CPU_PROFILE).
  struct CpuTimer {
    double *slot;
    std::chrono::steady_clock::time_point t0;
    explicit CpuTimer(double *s) : slot(s) {
      if (slot) {
        t0 = std::chrono::steady_clock::now();
      }
    }
    ~CpuTimer() {
      if (slot) {
        *slot += std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t0)
                     .count();
      }
    }
  };
  std::chrono::steady_clock::time_point stat_t0 =
      std::chrono::steady_clock::now();
  u64 draw_count = 0;
  std::set<std::string> reported;

  void ReportOnce(const std::string &what) {
    if (reported.insert(what).second) {
      DC_WARN("gpu: %s", what.c_str());
    }
  }

  // ================= setup =================

  void Init() {
    dump_dir = std::getenv("DC_DUMP_SHADERS");
    dump_frames = std::getenv("DC_DUMP_FRAMES");
    if (const char *e = std::getenv("DC_DUMP_EVERY")) {
      dump_every = std::max<u64>(1, std::strtoull(e, nullptr, 10));
    }
    debug_clear = std::getenv("DC_DEBUG_CLEAR") != nullptr;
    if (const char *v = std::getenv("DC_DEBUG_VSOUT")) {
      debug_vsout = std::strtoull(v, nullptr, 16);
    }
    trace = std::getenv("DC_GPU_TRACE") != nullptr;
    gds_readback = std::getenv("DC_GDS_READBACK") != nullptr;
    if (const char *e = std::getenv("DC_FIND_ADDR")) {
      find_addr = std::strtoull(e, nullptr, 16);
    }
    if (const char *e = std::getenv("DC_FIND_SIZE")) { // <min>-<max> bytes
      char *q = nullptr;
      find_size_min = std::strtoull(e, &q, 0);
      find_size_max = *q == '-' ? std::strtoull(q + 1, nullptr, 0) : find_size_min;
    }
    rain_diag = std::getenv("DC_RAIN_DIAG") != nullptr;
    if (const char *e = std::getenv("DC_PEEK_VS")) {
      peek_vs = std::strtoull(e, nullptr, 16);
    } else if (rain_diag) {
      peek_vs = kRainDropVs;
    }
    dump_passes = std::getenv("DC_DUMP_PASSES") != nullptr;
    if (const char *e = std::getenv("DC_TRACE_FRAME")) {
      trace_frame = std::strtoull(e, nullptr, 10);
    }
    if (const char *e = std::getenv("DC_DUMP_TEXTURE")) {
      dump_texture = std::strtoull(e, nullptr, 16);
    }
    sync_shaders = std::getenv("DC_SYNC_SHADERS") != nullptr;
    if (const char *e = std::getenv("DC_PROBE")) {
      char *q = nullptr;
      probe.shader = std::strtoull(e, &q, 16);
      if (!std::strncmp(e, "*:any:", 6)) { // every color export at pixel x,y
        probe.shader = ~0ull;
        probe.pc = Shader::kProbeExportAny;
        q = const_cast<char *>(e + std::strlen(e));
      } else if (!std::strcmp(
              e,
              "*:nan")) { // every pixel shader's first non-finite color export
        probe.shader = ~0ull;
        probe.pc = Shader::kProbeExportNan;
        probe.x = ~0u;
        q = const_cast<char *>(e + std::strlen(e));
      }
      if (*q == ':') {
        probe.pc = std::strtoul(q + 1, &q, 16);
      }
      // '*': any pixel; 'nan<k>': the first whose vk is NaN/Inf; 'big<k>': the
      // first whose vk is above 10000 or not finite
      if (*q == ':' && q[1] == '*') {
        probe.x = ~0u;
      } else if (*q == ':' && !std::strncmp(q + 1, "nan", 3)) {
        probe.x = ~1u;
        probe.y = std::strtoul(q + 4, &q, 10);
      } else if (*q == ':' && !std::strncmp(q + 1, "big", 3)) { // |vk| > 10000
        probe.x = ~2u;
        probe.y = std::strtoul(q + 4, &q, 10);
      } else {
        probe.x = *q == ':' ? std::strtoul(q + 1, &q, 10) : 0;
      }
      if (*q == ',') {
        probe.y = std::strtoul(q + 1, &q, 10);
      }
      if (!std::strncmp(e, "*:any:", 6)) { // x,y (the parsing above is for <pc>:<x>,<y>)
        probe.x = std::strtoul(e + 6, &q, 10);
        probe.y = *q == ',' ? std::strtoul(q + 1, &q, 10) : 0;
      }
      probe_buf =
          [device newBufferWithLength:kProbeSlots * Shader::kProbeWords * 4
                              options:MTLResourceStorageModeShared];
      std::memset(probe_buf.contents, 0, probe_buf.length);
    }
    stats = std::getenv("DC_GPU_STATS") != nullptr;
    cpu_profile = std::getenv("DC_CPU_PROFILE") != nullptr;
    tex_cache = std::getenv("DC_NO_TEX_CACHE") == nullptr;
    tex_cache_check = std::getenv("DC_TEX_CACHE_CHECK") != nullptr;
    map_hoist = std::getenv("DC_NO_MAP_HOIST") == nullptr;
    signal_batch = std::getenv("DC_NO_SIGNAL_BATCH") == nullptr;
    profile = std::getenv("DC_GPU_PROFILE") != nullptr;
    if (std::getenv("DC_PASS_TIMING") &&
        [device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
      for (id<MTLCounterSet> set in device.counterSets) {
        if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) {
          MTLCounterSampleBufferDescriptor *d = [MTLCounterSampleBufferDescriptor new];
          d.counterSet = set;
          d.storageMode = MTLStorageModeShared;
          d.sampleCount = kTsPerBuf;
          pass_timing = true;
          for (u32 k = 0; k < kTsBufs; ++k) {
            NSError *err = nil;
            ts_buf[k] = [device newCounterSampleBufferWithDescriptor:d error:&err];
            pass_timing = pass_timing && ts_buf[k];
          }
        }
      }
      [device sampleTimestamps:&ts_cpu0 gpuTimestamp:&ts_gpu0];
      DC_INFO("gpu: pass timing %s", pass_timing ? "on" : "unavailable");
    }
    compiler = std::make_unique<ShaderCompiler>(device, 4);
    PrewarmStart();
    signal_event = [device newSharedEvent];
    signal_listener = [[MTLSharedEventListener alloc]
        initWithDispatchQueue:dispatch_queue_create("gpu signals",
                                                    DISPATCH_QUEUE_SERIAL)];

    void *view = Core::Memory::DirectHostView();
    const u64 total = Core::Memory::kDirectSize;
    mem_split = (total / 2) & ~0xFFFFull;
    if (view) {
      mem[0] = [device newBufferWithBytesNoCopy:view
                                         length:mem_split
                                        options:SharedOpts()
                                    deallocator:nil];
      mem[1] =
          [device newBufferWithBytesNoCopy:static_cast<u8 *>(view) + mem_split
                                    length:total - mem_split
                                   options:SharedOpts()
                               deallocator:nil];
    }
    if (!mem[0] || !mem[1]) {
      DC_ERROR("gpu: cannot expose guest memory to Metal");
    }

    std::vector<u32> quads(0x10000 * 6);
    for (u32 q = 0; q < 0x10000; ++q) {
      const u32 b = q * 4;
      const u32 idx[6] = {b, b + 1, b + 2, b, b + 2, b + 3};
      std::memcpy(&quads[q * 6], idx, sizeof(idx));
    }
    quad_indices = [device newBufferWithBytes:quads.data()
                                       length:quads.size() * 4
                                      options:SharedOpts()];

    const NSUInteger gds_bytes = Gpu::kGdsPartition * Gpu::kGdsPartitions;
    gds = [device newBufferWithLength:gds_bytes
                              options:SharedOpts()];
    std::memset(gds.contents, 0, gds_bytes);

    MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
    sd.minFilter = MTLSamplerMinMagFilterLinear;
    sd.magFilter = MTLSamplerMinMagFilterLinear;
    default_sampler = [device newSamplerStateWithDescriptor:sd];

    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:@(kCopyShader)
                                              options:nil
                                                error:&err];
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    if (std::getenv("DC_DEBUG_FLAT") || std::getenv("DC_TESS_FLAT") || std::getenv("DC_DEBUG_FLAT_RT")) { // DC_TESS_FLAT: tess draws only
      debug_fs = [lib newFunctionWithName:@"debug_fs"];
    }
    pd.vertexFunction = [lib newFunctionWithName:@"copy_vs"];
    pd.fragmentFunction = [lib newFunctionWithName:@"copy_fs"];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    copy_pipeline = [device newRenderPipelineStateWithDescriptor:pd error:&err];
    pd.fragmentFunction = [lib newFunctionWithName:@"dump_fs"];
    dump_pipeline = [device newRenderPipelineStateWithDescriptor:pd error:&err];
    {
      MTLRenderPipelineDescriptor *zd = [MTLRenderPipelineDescriptor new];
      zd.vertexFunction = pd.vertexFunction;
      zd.fragmentFunction = [lib newFunctionWithName:@"zcopy_fs"];
      zd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
      zcopy_pipeline = [device newRenderPipelineStateWithDescriptor:zd error:&err];
      zd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
      zd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
      zcopy_pipeline_s8 = [device newRenderPipelineStateWithDescriptor:zd error:&err];
      MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
      dd.depthCompareFunction = MTLCompareFunctionAlways;
      dd.depthWriteEnabled = YES;
      zcopy_ds = [device newDepthStencilStateWithDescriptor:dd];
    }
    fill_pipeline = [device
        newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fill32"]
                                      error:&err];
    htile_pipeline = [device
        newComputePipelineStateWithFunction:[lib newFunctionWithName:@"htile_build"]
                                      error:&err];
    edr_peak_pipeline = [device
        newComputePipelineStateWithFunction:[lib newFunctionWithName:@"edr_peak"]
                                      error:&err];
    edr_mask_pipeline = [device
        newComputePipelineStateWithFunction:[lib newFunctionWithName:@"edr_mask"]
                                      error:&err];
    if (!htile_pipeline) {
      DC_ERROR("gpu: htile pipeline: %s", err.localizedDescription.UTF8String);
    }
    scan_val_pipeline = [device
        newComputePipelineStateWithFunction:[lib newFunctionWithName:@"scan_val"]
                                      error:&err];
    scan_loc_pipeline = [device
        newComputePipelineStateWithFunction:[lib newFunctionWithName:@"scan_loc"]
                                      error:&err];
    if (const char *e = std::getenv("DC_ZSTAT")) {
      char *end = nullptr;
      zstat_addr = std::strtoull(e, &end, 16);
      if (*end == ',') {
        zstat_every = std::max<u64>(1, std::strtoull(end + 1, nullptr, 10));
      }
    }
    if (const char *e = std::getenv("DC_SCAN_MAX")) {
      if (!std::strcmp(e, "*")) {
        scan_all = true;
      } else {
        for (const char *q = e; *q;) {
          char *end = nullptr;
          scan_addrs.push_back(std::strtoull(q, &end, 16));
          q = *end ? end + 1 : end;
        }
      }
    }
    if (!copy_pipeline) {
      DC_ERROR("gpu: copy pipeline: %s", err.localizedDescription.UTF8String);
    }
  }

  // Metal compiles lazily and stalls the GPU thread meanwhile; name the slow
  // ones.
  static void ReportSlowCompile(const char *what, u64 a, u64 b,
                                std::chrono::steady_clock::time_point t0) {
    const double s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    if (s > 0.5) {
      DC_WARN("gpu: %s %#llx/%#llx took %.1f s to compile", what, a, b, s);
    }
  }

  void UpdateMemMap() {
    const u64 gen = Core::Memory::DirectMapGeneration();
    if (gen == mem_map_gen && mem_map) {
      return;
    }
    mem_map_gen = gen;
    mem_ranges = Core::Memory::DirectMappings();
    std::sort(mem_ranges.begin(), mem_ranges.end(),
              [](const auto &x, const auto &y) { return x.start < y.start; });
    const auto &ranges = mem_ranges;
    // Largest page size at which no page holds two mappings.
    u32 shift = Shader::GpuMemMap::kMaxShift;
    for (; shift > Shader::GpuMemMap::kMinShift; --shift) {
      const u64 page = 1ull << shift;
      u64 prev_end = 0;
      bool ok = true;
      for (const auto &r : ranges) {
        ok = ok && (r.start & (page - 1)) == 0 &&
             ((prev_end + page - 1) & ~(page - 1)) <= r.start;
        prev_end = r.start + r.size;
      }
      if (ok) {
        break;
      }
    }
    const u64 page = 1ull << shift;
    const u64 lo = ranges.empty() ? 0 : ranges.front().start & ~(page - 1);
    const u64 hi =
        ranges.empty() ? 0 : ranges.back().start + ranges.back().size;
    const u64 pages = (hi - lo + page - 1) >> shift;
    constexpr u64 header = sizeof(Shader::GpuMemMap) / 4;
    std::vector<s32> table(header + pages, Shader::GpuMemMap::kUnmapped);
    auto &map = *reinterpret_cast<Shader::GpuMemMap *>(table.data());
    map = {lo, pages, mem_split, shift, 0, Core::Memory::kDirectSize};
    for (const auto &r : ranges) {
      if (((r.start ^ r.phys) & ((1ull << Shader::GpuMemMap::kMinShift) - 1)) !=
          0) {
        ReportOnce("direct mapping not 16 KB aligned");
      }
      const s32 delta = static_cast<s32>(
          (static_cast<s64>(r.phys) - static_cast<s64>(r.start)) >>
          Shader::GpuMemMap::kMinShift);
      for (u64 va = r.start & ~(page - 1); va < r.start + r.size; va += page) {
        table[header + ((va - lo) >> shift)] = delta;
      }
    }
    // The largest mapping, clipped to one mem buffer, becomes the fast window.
    static const bool fast_map = std::getenv("DC_NO_FAST_MAP") == nullptr;
    for (const auto &r : fast_map ? ranges : decltype(mem_ranges){}) {
      u64 va = r.start, phys = r.phys, size = r.size;
      if (phys < mem_split && phys + size > mem_split) {
        const u64 below = mem_split - phys;
        if (below >= size - below) {
          size = below;
        } else {
          va += below, phys += below, size -= below;
        }
      }
      if (phys + size > Core::Memory::kDirectSize || size <= map.fast_size) {
        continue;
      }
      map.fast_va = va;
      map.fast_size = size;
      map.fast_hi = phys >= mem_split;
      map.fast_off = map.fast_hi ? phys - mem_split : phys;
    }
    // A fresh buffer: work already recorded keeps reading the old table.
    mem_map = [device newBufferWithBytes:table.data()
                                  length:table.size() * 4
                                 options:SharedOpts()];
  }

  // Guest range [va, va + size) as byte `off` of mem buffer `hi` (no ObjC
  // objects: cheap enough per draw). The caller ran UpdateMemMap.
  bool LocatePhys(u64 va, u64 size, u64 *off, bool *hi) const {
    auto it =
        std::upper_bound(mem_ranges.begin(), mem_ranges.end(), va,
                         [](u64 a, const auto &r) { return a < r.start; });
    if (it == mem_ranges.begin()) {
      return false;
    }
    --it;
    if (va + size > it->start + it->size) {
      return false;
    }
    const u64 phys = it->phys + (va - it->start);
    if (phys + size <= mem_split) {
      *off = phys;
      *hi = false;
    } else if (phys >= mem_split) {
      *off = phys - mem_split;
      *hi = true;
    } else {
      return false;
    }
    return true;
  }

  // Guest range [va, va + size) as a range of one of the `mem` buffers.
  bool Locate(u64 va, u64 size, id<MTLBuffer> *buf, u64 *off) {
    UpdateMemMap();
    auto it =
        std::upper_bound(mem_ranges.begin(), mem_ranges.end(), va,
                         [](u64 a, const auto &r) { return a < r.start; });
    if (it == mem_ranges.begin()) {
      return false;
    }
    --it;
    if (va + size > it->start + it->size) {
      return false;
    }
    const u64 phys = it->phys + (va - it->start);
    if (phys + size <= mem_split) {
      *buf = mem[0];
      *off = phys;
    } else if (phys >= mem_split) {
      *buf = mem[1];
      *off = phys - mem_split;
    } else {
      return false;
    }
    return *buf != nil;
  }

  // CP DMA on the host GPU, in order with the work around it.
  bool DmaData(u32 dst_sel, u64 dst, u32 src_sel, u64 src, u32 bytes) {
    if (trace) {
      DC_INFO("gpu: dma q%u %u:%#llx <- %u:%#llx, %u bytes", Gpu::t_queue, dst_sel, dst,
              src_sel, src, bytes);
    }
    if ((bytes & 3) || (dst & 3) || (src_sel != 2 && (src & 3))) {
      return false;
    }
    id<MTLBuffer> dbuf = nil, sbuf = nil;
    u64 doff = 0, soff = 0;
    if (dst_sel == 1) {
      if (dst + bytes > Gpu::kGdsPartition) {
        return false;
      }
      dbuf = gds;
      doff = Gpu::GdsBase() + dst;
    } else if (!Locate(dst, bytes, &dbuf, &doff)) {
      return false;
    }
    if (src_sel == 1) {
      if (src + bytes > Gpu::kGdsPartition) {
        return false;
      }
      sbuf = gds;
      soff = Gpu::GdsBase() + src;
    } else if (src_sel != 2 && !Locate(src, bytes, &sbuf, &soff)) {
      return false;
    }
    const u32 value = static_cast<u32>(src);
    if (src_sel == 2 && dst_sel != 1 && value != ~0u && !expanding_htile &&
        MarkMetaCleared(dst, bytes)) {
      EndPass(); // the next pass over the surface loads its clear value
    }
    EndPass();
    EnsureCommandBuffer();
    if (sbuf) {
      id<MTLBlitCommandEncoder> b = [cmd blitCommandEncoder];
      [b copyFromBuffer:sbuf
               sourceOffset:soff
                   toBuffer:dbuf
          destinationOffset:doff
                       size:bytes];
      [b endEncoding];
      if (trace && gds_readback && src_sel == 1 && dst_sel != 1 && bytes == 4) {
        const u32 q = Gpu::t_queue;
        const u64 addr = dst;
        const u32 off = static_cast<u32>(src);
        [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
          const u32 v = *reinterpret_cast<const u32 *>(addr);
          DC_INFO("gpu: gds-read q%u %#llx <- gds %#x = %u", q, addr, off, v);
        }];
      }
    } else if (value == (value & 0xFF) * 0x01010101u) {
      id<MTLBlitCommandEncoder> b = [cmd blitCommandEncoder];
      [b fillBuffer:dbuf
              range:NSMakeRange(doff, bytes)
              value:static_cast<u8>(value)];
      [b endEncoding];
    } else if (fill_pipeline) {
      const u32 vc[2] = {value, bytes / 4};
      id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
      [e setComputePipelineState:fill_pipeline];
      [e setBuffer:dbuf offset:doff atIndex:0];
      [e setBytes:vc length:sizeof(vc) atIndex:1];
      [e dispatchThreads:MTLSizeMake(bytes / 4, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      [e endEncoding];
    } else {
      return false;
    }
    return true;
  }

  // ================= recording =================

  void EnsureCommandBuffer() {
    if (!cmd) {
      cmd = [queue commandBuffer];
    }
  }

  // Texture uploads: blits from staging memory (Upload) in a command buffer
  // of their own, committed just before the main one so they run first.
  // The GPU lays texels out; the driver's replaceRegion did it on this thread.
  // DC_CPU_UPLOAD=1: the old path (managed textures, replaceRegion) for A/B.
  const bool cpu_upload = std::getenv("DC_CPU_UPLOAD") != nullptr;
  id<MTLCommandBuffer> up_cmd = nil;
  id<MTLBlitCommandEncoder> up_blit = nil;
  id<MTLBlitCommandEncoder> UploadBlit() {
    if (!up_blit) {
      up_cmd = [queue commandBuffer];
      up_blit = [up_cmd blitCommandEncoder];
    }
    return up_blit;
  }
  void FlushUploads() {
    if (up_blit) {
      [up_blit endEncoding];
      up_blit = nil;
      [up_cmd commit];
      up_cmd = nil;
    }
  }

  // DC_GPU_STATS: why passes over the same targets get split.
  // Compute encoder kept open across back-to-back dispatches: an encoder per
  // dispatch costs the host GPU a flush each. Its dispatches stay serial.
  id<MTLComputeCommandEncoder> cenc = nil;
  PassState last_pass;
  void *last_end_ra = nullptr;
  std::unordered_map<void *, u64> split_by;
  u64 stat_passes = 0, stat_split = 0;

  void EndPass() {
    if (cenc) {
      [cenc endEncoding];
      cenc = nil;
    }
    if (enc && stats) {
      last_pass = pass;
      last_end_ra = __builtin_return_address(0);
    }
    if (enc) {
      [enc endEncoding];
      enc = nil;
      if (trace && PeekRt().addr) {
        PeekPass();
      }
      if (dump_passes && trace && dump_frames) {
        DumpPass();
      }
    }
    pass = {};
    if (trace && (scan_all || !scan_addrs.empty())) {
      ScanMax();
    }
  }

  // DC_ZSTAT=<addr>[,<every>]: every <every> flips (300), the depth surface
  // at <addr>: draws that wrote it since the last report and its value range,
  // read back on the GPU timeline (to tell an empty shadow map from a drawn one).
  u64 zstat_addr = 0, zstat_every = 300, zstat_draws = 0;
  void ZStat() {
    auto it = depth_targets.find(zstat_addr);
    const u64 draws = zstat_draws;
    zstat_draws = 0;
    if (it == depth_targets.end() || it->second.empty()) {
      DC_INFO("gpu: zstat %#llx: no depth target (%llu draws)",
              (unsigned long long)zstat_addr, (unsigned long long)draws);
      return;
    }
    RenderTarget *z = it->second.back().get();
    id<MTLTexture> tex = z->tex;
    if (tex && (tex.pixelFormat == MTLPixelFormatDepth32Float ||
                tex.pixelFormat == MTLPixelFormatDepth32Float_Stencil8)) {
      ZStat32(*z, draws);
      return;
    }
    if (!tex || tex.pixelFormat != MTLPixelFormatDepth16Unorm || tex.textureType != MTLTextureType2D) {
      DC_INFO("gpu: zstat %#llx: fmt %lu type %lu not read back (%llu draws)",
              (unsigned long long)zstat_addr, (unsigned long)tex.pixelFormat,
              (unsigned long)tex.textureType, (unsigned long long)draws);
      return;
    }
    const u32 w = z->width, h = z->height;
    id<MTLBuffer> b = [device newBufferWithLength:u64(w) * h * 2
                                          options:MTLResourceStorageModeShared];
    EndPass();
    EnsureCommandBuffer();
    id<MTLBlitCommandEncoder> e = [cmd blitCommandEncoder];
    [e copyFromTexture:tex sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
            sourceSize:MTLSizeMake(w, h, 1) toBuffer:b destinationOffset:0
        destinationBytesPerRow:w * 2 destinationBytesPerImage:u64(w) * h * 2];
    [e endEncoding];
    const u64 addr = zstat_addr, frame = frame_count, written = z->written;
    const bool init = z->initialized, pend = z->clear_pending;
    const double clear = z->clear_depth;
    [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
      const u16 *d = static_cast<const u16 *>(b.contents);
      u32 lo = 0xFFFF, hi = 0;
      u64 n_one = 0, n_zero = 0, sum = 0;
      constexpr u32 kG = 8; // coverage of non-1.0 texels on an 8x8 grid
      u64 cell[kG * kG] = {};
      for (u32 y = 0; y < h; ++y) {
        for (u32 x = 0; x < w; ++x) {
          const u16 v = d[u64(y) * w + x];
          lo = std::min<u32>(lo, v);
          hi = std::max<u32>(hi, v);
          sum += v;
          n_one += v == 0xFFFF;
          n_zero += v == 0;
          if (v != 0xFFFF) {
            ++cell[(y * kG / h) * kG + x * kG / w];
          }
        }
      }
      std::string grid;
      const u64 per = u64(w / kG) * (h / kG);
      for (u32 c = 0; c < kG * kG; ++c) {
        grid += char('0' + std::min<u64>(9, cell[c] * 10 / std::max<u64>(1, per)));
        if (c % kG == kG - 1 && c + 1 < kG * kG) {
          grid += '/';
        }
      }
      const double total = double(w) * h;
      DC_INFO("gpu: zstat %#llx flip %llu %ux%u: %llu draws, written %llu init %d pending %d "
              "clear %g: min %.5f max %.5f mean %.5f, at 1.0 %.1f%%, at 0 %.1f%%, "
              "drawn per 1/64 (0-9) %s",
              (unsigned long long)addr, (unsigned long long)frame, w, h,
              (unsigned long long)draws, (unsigned long long)written, init, pend, clear,
              lo / 65535.0, hi / 65535.0, sum / total / 65535.0, n_one * 100 / total,
              n_zero * 100 / total, grid.c_str());
    }];
  }

  // DC_DUMP_BOUND=<hex addr>[,<slices>]: in a traced flip, the texture bound for
  // that address (render target, storage or upload) is read back once: per-channel
  // mean and max of every slice, and the first slices of level 0 as PPMs
  // (tonemapped x/(1+x)) into DC_DUMP_FRAMES. To see whether lighting inputs
  // such as GT7's light probe atlases hold anything.
  std::set<u64> dumped_bound;
  static float Half(u16 h) {
    const u32 s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    const float v = e == 0 ? std::ldexp(float(m), -24)
                    : e == 31 ? (m ? NAN : INFINITY)
                              : std::ldexp(float(m | 0x400), int(e) - 25);
    return s ? -v : v;
  }
  static float Small(u32 bits, u32 mbits) { // unsigned 11/10-bit float, 5-bit exponent
    const u32 e = bits >> mbits, m = bits & ((1u << mbits) - 1);
    if (e == 31) {
      return m ? NAN : INFINITY;
    }
    return e == 0 ? std::ldexp(float(m), -14 - int(mbits))
                  : std::ldexp(float(m | (1u << mbits)), int(e) - 15 - int(mbits));
  }
  void DumpBound(id<MTLTexture> tex, u64 addr) {
    // <addr>[+<addr>...][,<slices>]
    static const std::pair<std::vector<u64>, u32> want = [] {
      std::pair<std::vector<u64>, u32> p{{}, 6};
      if (const char *e = std::getenv("DC_DUMP_BOUND")) {
        char *q = const_cast<char *>(e);
        do {
          p.first.push_back(std::strtoull(q, &q, 16) & kGpuAddrMask);
        } while (*q == '+' && *++q);
        if (*q == ',') {
          p.second = u32(std::strtoul(q + 1, nullptr, 10));
        }
      }
      return p;
    }();
    // DC_DUMP_BOUND_FMT=<MTLPixelFormat>: any binding of that format (per-frame
    // ring buffers whose address changes every flip).
    static const u64 want_fmt = [] {
      const char *e = std::getenv("DC_DUMP_BOUND_FMT");
      return e ? std::strtoull(e, nullptr, 10) : 0ull;
    }();
    const bool by_fmt = tex && want_fmt && u64(tex.pixelFormat) == want_fmt;
    if (!tex || (!by_fmt && std::find(want.first.begin(), want.first.end(),
                                       addr & kGpuAddrMask) == want.first.end()) ||
        !dumped_bound.insert(Hash(reinterpret_cast<uintptr_t>((__bridge void *)tex),
                                  (addr & kGpuAddrMask) ^ (frame_count << 48)))
             .second) {
      return;
    }
    // Which surface the binding comes from: a render target (and when it was
    // last written) or not, then guest memory.
    std::string src = "not a target (guest memory or cache)";
    {
      id<MTLTexture> root = tex.parentTexture ? tex.parentTexture : tex;
      for (TargetMap *m : {&color_targets, &depth_targets}) {
        const auto it = m->find(addr & kGpuAddrMask);
        if (it == m->end()) {
          continue;
        }
        for (const auto &rt : it->second) {
          id<MTLTexture> rroot = rt->tex.parentTexture ? rt->tex.parentTexture : rt->tex;
          char b[160];
          std::snprintf(b, sizeof(b), "%s%s target %ux%u layers %u levels %u written %llu",
                        rroot == root ? "BOUND " : "other ", m == &depth_targets ? "depth" : "color",
                        rt->width, rt->height, rt->depth_or_layers, rt->levels,
                        (unsigned long long)rt->written);
          src = (src.rfind("not a target", 0) == 0 ? std::string() : src + "; ") + b;
        }
      }
      if (const auto f = memory_fills.find(addr & kGpuAddrMask); f != memory_fills.end()) {
        src += "; memory fill at " + std::to_string(f->second);
      }
    }
    DC_INFO("gpu: dump bound %#llx at draw %llu: type %lu layers %lu levels %lu, %s",
            (unsigned long long)addr, (unsigned long long)draw_count,
            (unsigned long)tex.textureType, (unsigned long)tex.arrayLength,
            (unsigned long)tex.mipmapLevelCount, src.c_str());
    const MTLPixelFormat f = tex.pixelFormat;
    const u32 bpp = f == MTLPixelFormatRGBA16Float || f == MTLPixelFormatRG32Uint ? 8
                    : (f == MTLPixelFormatRG11B10Float || f == MTLPixelFormatRGBA8Unorm ||
                       f == MTLPixelFormatBGRA8Unorm || f == MTLPixelFormatR32Float)
                        ? 4
                        : 0;
    if (!bpp) {
      DC_INFO("gpu: dump bound %#llx: format %lu not handled", (unsigned long long)addr,
              (unsigned long)f);
      return;
    }
    // DC_DUMP_BOUND_LEVEL=<n>: that mip level instead of 0 (filters reading low mips).
    static const u32 level_want = [] {
      const char *e = std::getenv("DC_DUMP_BOUND_LEVEL");
      return e ? u32(std::strtoul(e, nullptr, 10)) : 0u;
    }();
    const u32 level = std::min<u32>(level_want, u32(tex.mipmapLevelCount) - 1);
    const u32 w = std::max(1u, u32(tex.width) >> level), h = std::max(1u, u32(tex.height) >> level);
    const u32 slices = tex.textureType == MTLTextureType2DArray ||
                               tex.textureType == MTLTextureTypeCubeArray ||
                               tex.textureType == MTLTextureTypeCube
                           ? u32(tex.arrayLength) * (tex.textureType == MTLTextureType2DArray ? 1 : 6)
                           : 1;
    const u64 slice_bytes = u64(w) * h * bpp;
    id<MTLBuffer> b = [device newBufferWithLength:slice_bytes * slices
                                          options:MTLResourceStorageModeShared];
    EndPass();
    EnsureCommandBuffer();
    id<MTLBlitCommandEncoder> e = [cmd blitCommandEncoder];
    for (u32 s = 0; s < slices; ++s) {
      [e copyFromTexture:tex sourceSlice:s sourceLevel:level sourceOrigin:MTLOriginMake(0, 0, 0)
              sourceSize:MTLSizeMake(w, h, 1) toBuffer:b destinationOffset:slice_bytes * s
          destinationBytesPerRow:w * bpp destinationBytesPerImage:slice_bytes];
    }
    [e endEncoding];
    const std::string dir = dump_frames ? dump_frames : ".";
    const u64 frame = frame_count;
    const u32 nfiles = want.second;
    [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
      auto texel = [&](const u8 *p, float out[4]) {
        if (f == MTLPixelFormatRGBA16Float) {
          const u16 *q = reinterpret_cast<const u16 *>(p);
          for (int c = 0; c < 4; ++c) out[c] = Half(q[c]);
        } else if (f == MTLPixelFormatRG11B10Float) {
          u32 v;
          std::memcpy(&v, p, 4);
          out[0] = Small(v & 0x7FF, 6);
          out[1] = Small((v >> 11) & 0x7FF, 6);
          out[2] = Small(v >> 22, 5);
          out[3] = 1;
        } else if (f == MTLPixelFormatR32Float) {
          std::memcpy(&out[0], p, 4);
          out[1] = out[2] = out[0];
          out[3] = 1;
        } else if (f == MTLPixelFormatRG32Uint) { // integer lists: values as floats
          u32 q[2];
          std::memcpy(q, p, 8);
          out[0] = float(q[0]);
          out[1] = float(q[1]);
          out[2] = 0;
          out[3] = 1;
        } else {
          const bool bgra = f == MTLPixelFormatBGRA8Unorm;
          out[0] = p[bgra ? 2 : 0] / 255.0f;
          out[1] = p[1] / 255.0f;
          out[2] = p[bgra ? 0 : 2] / 255.0f;
          out[3] = p[3] / 255.0f;
        }
      };
      double all[4] = {}, amax[4] = {};
      u32 nonzero_slices = 0;
      for (u32 s = 0; s < slices; ++s) {
        const u8 *base = static_cast<const u8 *>(b.contents) + slice_bytes * s;
        double sum[4] = {}, mx[4] = {};
        std::vector<u8> img;
        const bool write = s < nfiles;
        if (write) {
          img.resize(u64(w) * h * 3);
        }
        for (u64 i = 0; i < u64(w) * h; ++i) {
          float t[4];
          texel(base + i * bpp, t);
          for (int c = 0; c < 4; ++c) {
            const double v = std::isfinite(t[c]) ? t[c] : 0.0;
            sum[c] += v;
            mx[c] = std::max(mx[c], v);
          }
          if (write) {
            for (int c = 0; c < 3; ++c) {
              const float v = std::isfinite(t[c]) && t[c] > 0 ? t[c] : 0.0f;
              img[i * 3 + c] = u8(std::min(255.0f, 255.0f * v / (1.0f + v)));
            }
          }
        }
        const double n = double(w) * h;
        nonzero_slices += mx[0] + mx[1] + mx[2] > 0;
        for (int c = 0; c < 4; ++c) {
          all[c] += sum[c] / n / slices;
          amax[c] = std::max(amax[c], mx[c]);
        }
        if (write) {
          DC_INFO("gpu: dump bound %#llx flip %llu slice %u: mean %.4g %.4g %.4g %.4g max %.4g %.4g %.4g",
                  (unsigned long long)addr, (unsigned long long)frame, s, sum[0] / n, sum[1] / n,
                  sum[2] / n, sum[3] / n, mx[0], mx[1], mx[2]);
          char path[512];
          std::snprintf(path, sizeof(path), "%s/bound_%llx_f%llu_s%u.ppm", dir.c_str(),
                        (unsigned long long)addr, (unsigned long long)frame, s);
          if (FILE *fp = std::fopen(path, "wb")) {
            std::fprintf(fp, "P6\n%u %u\n255\n", w, h);
            std::fwrite(img.data(), 1, img.size(), fp);
            std::fclose(fp);
          }
        }
      }
      DC_INFO("gpu: dump bound %#llx flip %llu: %ux%u fmt %lu, %u slices, %u not black, "
              "mean %.4g %.4g %.4g max %.4g %.4g %.4g",
              (unsigned long long)addr, (unsigned long long)frame, w, h, (unsigned long)f, slices,
              nonzero_slices, all[0], all[1], all[2], amax[0], amax[1], amax[2]);
    }];
  }

  // ZStat for 32-bit float depth, every slice of an array (shadow cascades).
  // DC_ZSTAT_UV=u,v also prints the texel there in each slice.
  void ZStat32(RenderTarget &z, u64 draws) {
    id<MTLTexture> tex = z.tex;
    const u32 w = z.width, h = z.height;
    const u32 slices = tex.textureType == MTLTextureType2DArray ? u32(tex.arrayLength) : 1;
    const u64 slice_bytes = u64(w) * h * 4;
    id<MTLBuffer> b = [device newBufferWithLength:slice_bytes * slices
                                          options:MTLResourceStorageModeShared];
    EndPass();
    EnsureCommandBuffer();
    id<MTLBlitCommandEncoder> e = [cmd blitCommandEncoder];
    const MTLBlitOption opt = tex.pixelFormat == MTLPixelFormatDepth32Float_Stencil8
                                  ? MTLBlitOptionDepthFromDepthStencil
                                  : MTLBlitOptionNone;
    for (u32 s = 0; s < slices; ++s) {
      [e copyFromTexture:tex sourceSlice:s sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
              sourceSize:MTLSizeMake(w, h, 1) toBuffer:b destinationOffset:slice_bytes * s
          destinationBytesPerRow:w * 4 destinationBytesPerImage:slice_bytes options:opt];
    }
    [e endEncoding];
    static const std::pair<float, float> uv = [] {
      std::pair<float, float> p{-1.0f, -1.0f};
      if (const char *e = std::getenv("DC_ZSTAT_UV")) {
        char *q = nullptr;
        p.first = std::strtof(e, &q);
        p.second = *q == ',' ? std::strtof(q + 1, nullptr) : 0.0f;
      }
      return p;
    }();
    const u64 addr = zstat_addr, frame = frame_count, written = z.written;
    const bool init = z.initialized, pend = z.clear_pending;
    const float clear = z.clear_depth;
    [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
      for (u32 s = 0; s < slices; ++s) {
        const float *d = static_cast<const float *>(b.contents) + u64(w) * h * s;
        float lo = 1e30f, hi = -1e30f;
        double sum = 0;
        u64 n_clear = 0;
        constexpr u32 kG = 8;
        u64 cell[kG * kG] = {};
        for (u32 y = 0; y < h; ++y) {
          for (u32 x = 0; x < w; ++x) {
            const float v = d[u64(y) * w + x];
            lo = std::min(lo, v);
            hi = std::max(hi, v);
            sum += v;
            if (v == clear) {
              ++n_clear;
            } else {
              ++cell[(y * kG / h) * kG + x * kG / w];
            }
          }
        }
        std::string grid;
        const u64 per = u64(w / kG) * (h / kG);
        for (u32 c = 0; c < kG * kG; ++c) {
          grid += char('0' + std::min<u64>(9, cell[c] * 10 / std::max<u64>(1, per)));
          if (c % kG == kG - 1 && c + 1 < kG * kG) {
            grid += '/';
          }
        }
        std::string at;
        if (uv.first >= 0) {
          const u32 x = std::min<u32>(w - 1, u32(uv.first * w)),
                    y = std::min<u32>(h - 1, u32(uv.second * h));
          char buf[96];
          std::snprintf(buf, sizeof(buf), ", at %u,%u: %.5f", x, y, d[u64(y) * w + x]);
          at = buf;
        }
        // DC_ZSTAT_DUMP=<dir>: each slice as an 8-bit PGM, min..max of the drawn texels
        // stretched to 0..255 (clear value white), to see what the light sees.
        if (const char *dir = std::getenv("DC_ZSTAT_DUMP")) {
          float dlo = 1e30f, dhi = -1e30f;
          for (u64 i = 0; i < u64(w) * h; ++i) {
            if (d[i] != clear && std::isfinite(d[i])) {
              dlo = std::min(dlo, d[i]);
              dhi = std::max(dhi, d[i]);
            }
          }
          char path[512];
          std::snprintf(path, sizeof(path), "%s/z_%llx_f%llu_s%u.pgm", dir,
                        (unsigned long long)addr, (unsigned long long)frame, s);
          if (FILE *f = std::fopen(path, "wb")) {
            std::fprintf(f, "P5\n%u %u\n255\n", w, h);
            std::vector<u8> row(w);
            const float span = dhi > dlo ? dhi - dlo : 1.0f;
            for (u32 y = 0; y < h; ++y) {
              for (u32 x = 0; x < w; ++x) {
                const float v = d[u64(y) * w + x];
                row[x] = v == clear || !std::isfinite(v)
                             ? 255
                             : u8(std::clamp((v - dlo) / span, 0.0f, 1.0f) * 254.0f);
              }
              std::fwrite(row.data(), 1, w, f);
            }
            std::fclose(f);
          }
          DC_INFO("gpu: zstat dump %s range %.5f..%.5f", path, dlo, dhi);
        }
        const double total = double(w) * h;
        DC_INFO("gpu: zstat %#llx flip %llu slice %u/%u %ux%u: %llu draws, written %llu init %d "
                "pending %d clear %g: min %.5f max %.5f mean %.5f, at clear %.1f%%%s, drawn %s",
                (unsigned long long)addr, (unsigned long long)frame, s, slices, w, h,
                (unsigned long long)draws, (unsigned long long)written, init, pend, clear, lo, hi,
                sum / total, n_clear * 100 / total, at.c_str(), grid.c_str());
      }
    }];
  }

  void ScanMax() {
    if (!scan_val_pipeline || !scan_loc_pipeline) {
      return;
    }
    auto scan_at = [&](u64 a, const std::vector<std::unique_ptr<RenderTarget>> &list) {
      for (auto &t : list) {
        RenderTarget *rt = t.get();
        const MTLPixelFormat f = rt->tex ? rt->tex.pixelFormat : MTLPixelFormatInvalid;
        if (scan_seen[rt] == rt->written || !rt->tex ||
            f == MTLPixelFormatRGBA8Uint || f == MTLPixelFormatR32Uint ||
            f == MTLPixelFormatRG32Uint || f == MTLPixelFormatRGBA32Uint ||
            f == MTLPixelFormatR16Uint || f == MTLPixelFormatR8Uint ||
            f == MTLPixelFormatRG16Uint || f == MTLPixelFormatRGBA16Uint) {
          continue;
        }
        scan_seen[rt] = rt->written;
        id<MTLTexture> v = rt->tex;
        if (v.textureType == MTLTextureType2D) {
          v = [v newTextureViewWithPixelFormat:v.pixelFormat
                                   textureType:MTLTextureType2DArray
                                        levels:NSMakeRange(0, v.mipmapLevelCount)
                                        slices:NSMakeRange(0, 1)];
        } else if (v.textureType != MTLTextureType2DArray) {
          continue;
        }
        id<MTLBuffer> b = [device newBufferWithLength:32
                                              options:MTLResourceStorageModeShared];
        std::memset(b.contents, 0, 32);
        EnsureCommandBuffer();
        id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
        [e setTexture:v atIndex:0];
        [e setBuffer:b offset:0 atIndex:0];
        for (id<MTLComputePipelineState> ps : {scan_val_pipeline, scan_loc_pipeline}) {
          [e setComputePipelineState:ps];
          for (u32 l = 0; l < v.mipmapLevelCount; ++l) {
            [e setBytes:&l length:4 atIndex:1];
            [e dispatchThreads:MTLSizeMake(std::max<u64>(1, v.width >> l),
                                           std::max<u64>(1, v.height >> l),
                                           v.arrayLength)
                threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
          }
        }
        [e endEncoding];
        __block std::string what = scan_label;
        const u64 addr = a;
        const u32 w = rt->width, h = rt->height;
        const u32 pf = static_cast<u32>(f);
        [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
          const u32 *o = static_cast<const u32 *>(b.contents);
          float c[4];
          std::memcpy(c, o, 4);
          std::memcpy(c + 1, o + 5, 12);
          DC_INFO("gpu: scan %#llx %ux%u fmt %u after %s: max %g at %u,%u slice %u level %u "
                  "rgb %g %g %g",
                  (unsigned long long)addr, w, h, pf, what.c_str(), c[0], o[1], o[2], o[3],
                  o[4], c[1], c[2], c[3]);
        }];
      }
    };
    if (scan_all) {
      for (const auto &[a, list] : color_targets) {
        scan_at(a, list);
      }
    } else {
      for (u64 a : scan_addrs) {
        if (auto it = color_targets.find(a); it != color_targets.end()) {
          scan_at(a, it->second);
        }
      }
    }
  }

  // DC_DUMP_PASSES: small BGRA copies of the targets of the pass just ended.
  // Clears every slice of a depth target now (a clear through memory, which
  // no pass's load action carries).
  void ClearDepthNow(RenderTarget &z, float depth_value) {
    if (!z.tex || !(z.tex.pixelFormat == MTLPixelFormatDepth32Float ||
                    z.tex.pixelFormat == MTLPixelFormatDepth32Float_Stencil8 ||
                    z.tex.pixelFormat == MTLPixelFormatDepth16Unorm)) {
      return;
    }
    if (!std::isfinite(depth_value)) {
      depth_value = 0.0f;
    }
    EndPass();
    EnsureCommandBuffer();
    const u32 slices = z.tex.textureType == MTLTextureType2DArray ? u32(z.tex.arrayLength) : 1;
    for (u32 s = 0; s < slices; ++s) {
      MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
      rp.depthAttachment.texture = z.tex;
      rp.depthAttachment.slice = s;
      rp.depthAttachment.loadAction = MTLLoadActionClear;
      rp.depthAttachment.clearDepth = std::clamp(depth_value, 0.0f, 1.0f);
      rp.depthAttachment.storeAction = MTLStoreActionStore;
      if (HasStencil(z.tex)) {
        rp.stencilAttachment.texture = z.tex;
        rp.stencilAttachment.slice = s;
        rp.stencilAttachment.loadAction = MTLLoadActionLoad;
        rp.stencilAttachment.storeAction = MTLStoreActionStore;
      }
      [[cmd renderCommandEncoderWithDescriptor:rp] endEncoding];
    }
    z.initialized = true;
    z.clear_pending = false;
    z.written = ++write_seq;
  }

  // DC_PEEK_RT=<hex addr>,<x>,<y>: in a traced flip, after every render pass
  // with that colour attachment, texel x,y of level 0 in every slice as floats,
  // with the pass's draw range: which draw wrote (or wiped) a surface.
  struct PeekRtCfg {
    u64 addr = 0;
    u32 x = 0, y = 0;
  };
  static const PeekRtCfg &PeekRt() {
    static const PeekRtCfg c = [] {
      PeekRtCfg p;
      if (const char *e = std::getenv("DC_PEEK_RT")) {
        char *q = nullptr;
        p.addr = std::strtoull(e, &q, 16) & kGpuAddrMask;
        if (*q == ',') p.x = u32(std::strtoul(q + 1, &q, 10));
        if (*q == ',') p.y = u32(std::strtoul(q + 1, &q, 10));
      }
      return p;
    }();
    return c;
  }
  u64 pass_first_draw = 0;
  void PeekPass() {
    const PeekRtCfg &c = PeekRt();
    for (u32 k = 0; k < 8; ++k) {
      id<MTLTexture> t = pass_tex[k];
      if (!t || (pass_addr[k] & kGpuAddrMask) != c.addr) {
        continue;
      }
      id<MTLTexture> root = t.parentTexture ? t.parentTexture : t;
      const MTLPixelFormat f = root.pixelFormat;
      const u32 bpp = f == MTLPixelFormatRGBA16Float ? 8 : 4;
      const u32 slices = root.textureType == MTLTextureType2DArray ? u32(root.arrayLength) : 1;
      if (c.x >= root.width || c.y >= root.height) {
        continue;
      }
      id<MTLBuffer> b = [device newBufferWithLength:u64(bpp) * slices
                                            options:MTLResourceStorageModeShared];
      EnsureCommandBuffer();
      id<MTLBlitCommandEncoder> e = [cmd blitCommandEncoder];
      for (u32 s = 0; s < slices; ++s) {
        [e copyFromTexture:root sourceSlice:s sourceLevel:0 sourceOrigin:MTLOriginMake(c.x, c.y, 0)
                sourceSize:MTLSizeMake(1, 1, 1) toBuffer:b destinationOffset:u64(bpp) * s
            destinationBytesPerRow:bpp destinationBytesPerImage:bpp];
      }
      [e endEncoding];
      const u64 d0 = pass_first_draw, d1 = draw_count, addr = c.addr;
      const u32 x = c.x, y = c.y;
      const bool same = root == t;
      [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
        std::string line;
        for (u32 s = 0; s < slices; ++s) {
          const u8 *p = static_cast<const u8 *>(b.contents) + u64(bpp) * s;
          float v[3] = {};
          if (f == MTLPixelFormatRG11B10Float) {
            u32 w;
            std::memcpy(&w, p, 4);
            v[0] = Small(w & 0x7FF, 6);
            v[1] = Small((w >> 11) & 0x7FF, 6);
            v[2] = Small(w >> 22, 5);
          } else if (f == MTLPixelFormatRGBA16Float) {
            const u16 *q = reinterpret_cast<const u16 *>(p);
            for (int i = 0; i < 3; ++i) v[i] = Half(q[i]);
          } else {
            for (int i = 0; i < 3; ++i) v[i] = p[i] / 255.0f;
          }
          char buf[64];
          std::snprintf(buf, sizeof(buf), " s%u=(%.3g %.3g %.3g)", s, v[0], v[1], v[2]);
          line += buf;
        }
        DC_INFO("gpu: peek rt %#llx %u,%u after draws %llu..%llu%s:%s", (unsigned long long)addr,
                x, y, (unsigned long long)d0, (unsigned long long)d1,
                same ? "" : " (via view)", line.c_str());
      }];
    }
  }

  void DumpPass() {
    for (u32 k = 0; k < 9; ++k) {
      id<MTLTexture> t = pass_tex[k];
      pass_tex[k] = nil;
      if (!t || !copy_pipeline) {
        continue;
      }
      if (t.textureType ==
          MTLTextureType2DArray) { // each slice (cube faces) on its own
        for (NSUInteger sl = 0; sl < std::min<NSUInteger>(t.arrayLength, 12);
             ++sl) {
          DumpTexture([t newTextureViewWithPixelFormat:t.pixelFormat
                                           textureType:MTLTextureType2D
                                                levels:NSMakeRange(0, 1)
                                                slices:NSMakeRange(sl, 1)],
                      k, pass_addr[k]);
        }
      } else if (t.textureType == MTLTextureType2D) {
        DumpTexture(t, k, pass_addr[k]);
      }
    }
  }

  void DumpTexture(id<MTLTexture> t, u32 k, u64 addr) {
    const MTLPixelFormat f = t.pixelFormat;
    if (k < 8 &&
        (f == MTLPixelFormatRGBA8Uint || f == MTLPixelFormatR32Uint ||
         f == MTLPixelFormatRG32Uint || f == MTLPixelFormatRGBA32Uint ||
         f == MTLPixelFormatR16Uint || f == MTLPixelFormatR8Uint ||
         f == MTLPixelFormatRG16Uint || f == MTLPixelFormatRGBA16Uint)) {
      return;
    }
    const u32 w = std::max<u32>(1, static_cast<u32>(t.width) / 4),
              h = std::max<u32>(1, static_cast<u32>(t.height) / 4);
    MTLTextureDescriptor *d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                     width:w
                                    height:h
                                 mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget;
    d.storageMode = MTLStorageModePrivate;
    id<MTLTexture> dst = [device newTextureWithDescriptor:d];
    MTLRenderPassDescriptor *rp =
        [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = dst;
    rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> e = [cmd renderCommandEncoderWithDescriptor:rp];
    const float scale[2] = {1, 1};
    [e setRenderPipelineState:dump_pipeline];
    // DC_DUMP_ALPHA: show alpha; DC_DUMP_GAIN=<x>: scale before the x/(1+x)
    // display curve
    static const float dump_opt[2] = {
        std::getenv("DC_DUMP_ALPHA") ? 1.0f : 0.0f,
        std::getenv("DC_DUMP_GAIN")
            ? std::strtof(std::getenv("DC_DUMP_GAIN"), nullptr)
            : 1.0f};
    [e setFragmentBytes:dump_opt length:sizeof(dump_opt) atIndex:1];
    [e setFragmentTexture:t atIndex:0];
    [e setFragmentBytes:scale length:sizeof(scale) atIndex:0];
    [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [e endEncoding];
    id<MTLBuffer> buf =
        [device newBufferWithLength:u64(w) * h * 4
                            options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> b = [cmd blitCommandEncoder];
    [b copyFromTexture:dst
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(w, h, 1)
                        toBuffer:buf
               destinationOffset:0
          destinationBytesPerRow:w * 4
        destinationBytesPerImage:u64(w) * h * 4];
    [b endEncoding];
    char name[128];
    std::snprintf(name, sizeof(name), "pass_%04zu_%s%u_%llx_d%llu",
                  pass_dumps.size(), k == 8 ? "z" : "c", k, addr, draw_count);
    pass_dumps.push_back({buf, w, h, name});
  }

  void WritePassDumps() {
    for (const auto &pd : pass_dumps) {
      char path[512];
      std::snprintf(path, sizeof(path), "%s/%s.ppm", dump_frames,
                    pd.name.c_str());
      std::ofstream out(path, std::ios::binary);
      out << "P6\n" << pd.w << " " << pd.h << "\n255\n";
      const auto *px = static_cast<const u8 *>(pd.buf.contents);
      for (u64 i = 0; i < u64(pd.w) * pd.h; ++i) {
        const char rgb[3] = {static_cast<char>(px[i * 4 + 2]),
                             static_cast<char>(px[i * 4 + 1]),
                             static_cast<char>(px[i * 4])};
        out.write(rgb, 3);
      }
    }
    pass_dumps.clear();
  }

  void Commit(bool wait) {
    CpuTimer t(cpu_profile ? &cpu_commit_s : nullptr);
    EndPass();
    FlushUploads(); // before any command buffer that samples them
    if (!cmd) {
      if (wait) {
        WaitLast();
      }
      return;
    }
    ++commits;
    FlushSignal();
    if (!chunks.empty() || chunk) {
      // Keep upload memory alive until the GPU is done with it.
      __block std::vector<id<MTLBuffer>> keep = std::move(chunks);
      if (chunk) {
        keep.push_back(chunk);
      }
      [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
        RecycleChunks(keep);
      }];
      chunks.clear();
      chunk = nil;
      chunk_off = 0;
    }
    if (arena_va) {
      const u64 head = arena_commit_head = arena_head;
      [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
        arena_released.store(head, std::memory_order_release);
      }];
    }
    if (pass_timing && !ts_cmd.empty()) {
      __block std::vector<TsPass> passes = std::move(ts_cmd);
      ts_cmd.clear();
      [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
        std::scoped_lock lk(ts_lock);
        for (const TsPass &t : passes) {
          const u32 n = t.compute ? 2 : 4;
          NSData *d = [ts_buf[t.sample / kTsPerBuf]
              resolveCounterRange:NSMakeRange(t.sample % kTsPerBuf, n)];
          if (!d || d.length < n * sizeof(MTLCounterResultTimestamp)) {
            continue;
          }
          const auto *v = static_cast<const MTLCounterResultTimestamp *>(d.bytes);
          // Span of one stage (vertex, fragment or the compute encoder).
          auto span = [&](u32 i) {
            if (v[i].timestamp == MTLCounterErrorValue ||
                v[i + 1].timestamp == MTLCounterErrorValue ||
                v[i + 1].timestamp < v[i].timestamp) {
              return 0.0;
            }
            return double(v[i + 1].timestamp - v[i].timestamp) * ts_ns_per_tick * 1e-9;
          };
          char what[64];
          if (t.compute) {
            std::snprintf(what, sizeof(what), "cs %#llx", (unsigned long long)t.shader);
          } else {
            std::snprintf(what, sizeof(what), "ps %#llx %ux%u", (unsigned long long)t.shader,
                          t.w, t.h);
          }
          auto &e = ts_data[what];
          const double sec = t.compute ? span(0) : span(2);
          e.seconds += sec;
          e.calls += t.compute ? 1 : t.draws;
          e.verts += t.verts;
          (t.compute ? ts_cs_s : ts_fs_s) += sec;
          if (!t.compute) {
            const double vsec = span(0);
            ts_vs_s += vsec;
            auto &ve = ts_data[std::string("vertex ") + what];
            ve.seconds += vsec;
            ve.calls += t.draws;
            ve.verts += t.verts;
          }
        }
      }];
    }
    if (stats) {
      ++stat_commits;
    }
    if (Gpu::LtOn()) {
      const u64 seq = commits;
      DC_INFO("lt %.0f commit cb %llu", Gpu::LtUs(Gpu::LtNow()), (unsigned long long)seq);
      [cmd addCompletedHandler:^(id<MTLCommandBuffer> c) {
        // GPUStartTime/GPUEndTime are host seconds (CACurrentMediaTime).
        const double off = double(Gpu::LtNow()) - CACurrentMediaTime() * 1e9;
        DC_INFO("lt %.0f done cb %llu gpu %.0f..%.0f", Gpu::LtUs(Gpu::LtNow()),
                (unsigned long long)seq, Gpu::LtUs(u64(c.GPUStartTime * 1e9 + off)),
                Gpu::LtUs(u64(c.GPUEndTime * 1e9 + off)));
      }];
    }
    [cmd addCompletedHandler:^(id<MTLCommandBuffer> c) {
      const u64 ns = static_cast<u64>(GpuSeconds(c) * 1e9);
      gpu_ns += ns;
      perf_gpu_ns += ns;
    }];
    // A command buffer that hogs the GPU for seconds starves WindowServer
    // until its watchdog reboots the Mac: bail out before that happens.
    // DC_GPU_WATCHDOG=<seconds>, 0 disables.
    static const double watchdog_s = [] {
      const char *v = std::getenv("DC_GPU_WATCHDOG");
      return v ? std::atof(v) : 1.0;
    }();
    if (watchdog_s > 0) {
      StartWatchdog(watchdog_s);
      const u64 seq = wd_committed.load(std::memory_order_relaxed) + 1;
      wd_commit_ns[seq % kWdRing].store(NowNs(), std::memory_order_relaxed);
      {
        std::scoped_lock lk(wd_lock);
        wd_desc[seq % kWdRing] = std::move(wd_work);
      }
      wd_work.clear();
      wd_committed.store(seq, std::memory_order_release);
      [cmd addCompletedHandler:^(id<MTLCommandBuffer> c) {
        wd_done.store(seq, std::memory_order_release);
        const double s = GpuSeconds(c);
        if (c.status == MTLCommandBufferStatusError || s > watchdog_s) {
          std::string work;
          {
            std::scoped_lock lk(wd_lock);
            work = FormatNotes(wd_desc[seq % kWdRing]);
          }
          DC_ERROR("gpu watchdog: command buffer took %.2f s (%s), exiting; "
                   "it holds:\n%s",
                   s, c.error ? c.error.localizedDescription.UTF8String : "ok",
                   work.c_str());
          std::_Exit(3);
        }
      }];
    }
    [cmd commit];
    last_committed = cmd;
    cmd = nil;
    work_since_commit = 0;
    if (wait) {
      WaitLast();
    }
  }

  void NoteConstDump(u64 addr, u32 bytes) {
    // A new dump replaces whatever older dumps it overlaps. The same range
    // dumped again (the common case) just gets the new generation.
    auto it = const_dumps.lower_bound(addr);
    if (it != const_dumps.end() && it->first == addr && it->second.bytes == bytes &&
        (std::next(it) == const_dumps.end() || std::next(it)->first >= addr + bytes) &&
        (it == const_dumps.begin() ||
         std::prev(it)->first + std::prev(it)->second.bytes <= addr)) {
      it->second.gen = ++dump_gen;
      return;
    }
    if (it != const_dumps.begin() &&
        std::prev(it)->first + std::prev(it)->second.bytes > addr) {
      --it;
    }
    while (it != const_dumps.end() && it->first < addr + bytes) {
      it = const_dumps.erase(it);
    }
    const_dumps[addr] = {bytes, ++dump_gen};
  }

  bool InitArena() {
    u64 phys = 0;
    VAddr va = 0;
    if (Core::Memory::AllocateDirect(0, Core::Memory::kDirectSize, kArenaSize,
                                     64 << 10, 0, &phys) != 0 ||
        Core::Memory::MapDirect(
            &va, kArenaSize,
            Core::Memory::kProtCpuRead | Core::Memory::kProtCpuWrite |
                Core::Memory::kProtGpuRead,
            0, phys, 64 << 10, "gpu constant snapshots") != 0) {
      ReportOnce("cannot allocate the constant snapshot arena");
      return false;
    }
    arena_va = va;
    return true;
  }

  // Guest address of `bytes` of arena space the GPU is not using.
  u64 ArenaAlloc(u64 bytes) {
    bytes = (bytes + 255) & ~255ull;
    if (!arena_va && !InitArena()) {
      return 0;
    }
    u64 off = arena_head % kArenaSize;
    if (off + bytes > kArenaSize) {
      arena_head += kArenaSize - off; // do not split a copy across the wrap
      off = 0;
    }
    if (arena_head + bytes - arena_released.load(std::memory_order_acquire) >
        kArenaSize) {
      // Ring full of in-flight copies (rare): wait for what was committed.
      // A draw may be half recorded here, so the open command buffer stays.
      WaitLast();
      arena_released = arena_commit_head;
    }
    arena_head += bytes;
    return arena_va + off;
  }

  // Translates `p`'s linear bases against the draw's (patched) user data:
  // entry k = dword offset << 1 | upper buffer. Returns the bases that map
  // to one contiguous memory range (MslOptions::linear_mask). DC_NO_LINEAR:
  // none, every access walks the page table. `aligned` gets the returned bases
  // whose stride is a multiple of 4 (MslOptions::linear_aligned); DC_NO_ALIGNED:
  // none, every buffer load keeps its unaligned fallback.
  u32 LinearTable(const Shader::Program &p, const u32 *ud, u32 *out,
                  u32 *aligned) {
    static const bool off = std::getenv("DC_NO_LINEAR") != nullptr;
    static const bool no_aligned = std::getenv("DC_NO_ALIGNED") != nullptr;
    *aligned = 0;
    if (off) {
      return 0;
    }
    CpuTimer t_(cpu_profile ? &cpu_linear_s : nullptr);
    if (cpu_profile) {
      ++cpu_linear_calls;
      cpu_linear_bases += p.linear_bases.size();
    }
    u32 mask = 0;
    Shader::SgprReadCache cache;
    UpdateMemMap();
    for (u32 k = 0; k < p.linear_bases.size(); ++k) {
      out[k] = 0;
      u64 va = 0, bytes = 0, moff = 0;
      bool hi = false, stride4 = false;
      if (!Shader::LinearBaseRange(p.linear_bases[k], ud, &cache, &va, &bytes,
                                   &stride4) ||
          !LocatePhys(va, std::max<u64>(bytes, 4), &moff, &hi) || (moff & 3) ||
          (moff >> 2) >= (1ull << 31)) {
        continue;
      }
      out[k] = (static_cast<u32>(moff >> 2) << 1) | (hi ? 1u : 0u);
      mask |= 1u << k;
      if (stride4 && !no_aligned) {
        *aligned |= 1u << k;
      }
    }
    return mask;
  }

  // Points user-data pointers that land in a CE dump at a snapshot of it.
  void PatchUserData(u32 *ud) {
    std::scoped_lock dl(dump_lock);
    if (const_dumps.empty()) {
      return;
    }
    for (u32 k = 0; k + 1 < 16; ++k) {
      const u64 v =
          ud[k] | (u64(ud[k + 1] & 0xFFF) << 32); // pointer or V# base
      auto it = const_dumps.upper_bound(v);
      if (it == const_dumps.begin()) {
        continue;
      }
      --it;
      const u64 off = v - it->first;
      if (off >= it->second.bytes) {
        continue;
      }
      Snapshot &snap = snapshots[it->first];
      // A copy from an earlier command buffer may already be reused by the
      // arena (only the open command buffer's copies are kept alive), so it
      // is taken again; within one command buffer it is shared.
      if (snap.gen != it->second.gen || !snap.va || snap.start < arena_commit_head) {
        snap.va = ArenaAlloc(it->second.bytes);
        if (!snap.va) {
          return;
        }
        snap.start = arena_head - ((it->second.bytes + 255) & ~255ull);
        std::memcpy(reinterpret_cast<void *>(snap.va),
                    reinterpret_cast<const void *>(it->first),
                    it->second.bytes);
        snap.gen = it->second.gen;
      }
      const u64 nv = snap.va + off;
      ud[k] = static_cast<u32>(nv);
      ud[k + 1] = (ud[k + 1] & ~0xFFFu) | static_cast<u32>(nv >> 32);
      ++k;
    }
  }

  // DC_GPU_PROFILE: puts the work since the last cut in its own command
  // buffer and charges its GPU time to `what`.
  void ProfileCut(std::string what) {
    if (!cmd) {
      return;
    }
    EndPass();
    [cmd addCompletedHandler:^(id<MTLCommandBuffer> c) {
      std::scoped_lock lk(profile_lock);
      auto &e = profile_data[what];
      e.seconds += GpuSeconds(c);
      ++e.calls;
    }];
    Commit(false);
  }

  void PassTimingReport(u64 frames) {
    MTLTimestamp cpu = 0, gpu = 0;
    [device sampleTimestamps:&cpu gpuTimestamp:&gpu];
    if (gpu > ts_gpu0 && cpu > ts_cpu0) {
      ts_ns_per_tick = double(cpu - ts_cpu0) / double(gpu - ts_gpu0);
    }
    std::vector<std::pair<std::string, ProfileEntry>> v;
    double vs = 0, fs = 0, cs = 0;
    {
      std::scoped_lock lk(ts_lock);
      v.assign(ts_data.begin(), ts_data.end());
      ts_data.clear();
      vs = ts_vs_s;
      fs = ts_fs_s;
      cs = ts_cs_s;
      ts_vs_s = ts_fs_s = ts_cs_s = 0;
    }
    std::sort(v.begin(), v.end(), [](const auto &a, const auto &b) {
      return a.second.seconds > b.second.seconds;
    });
    const double f = frames ? double(frames) : 1.0;
    DC_INFO("pass timing: ms/frame summed over passes: fragment %.1f, vertex %.1f, "
            "compute %.1f (%.3f ns/tick); per pass: fragment (or compute) span",
            fs * 1e3 / f, vs * 1e3 / f, cs * 1e3 / f, ts_ns_per_tick);
    for (size_t k = 0; k < v.size() && k < 30; ++k) {
      DC_INFO("pass timing: %6.2f ms/frame %6.1f calls/frame %7.0fk verts/frame  %s",
              v[k].second.seconds * 1e3 / f, v[k].second.calls / f,
              v[k].second.verts / f / 1e3, v[k].first.c_str());
    }
  }

  void ProfileReport() {
    std::vector<std::pair<std::string, ProfileEntry>> v;
    {
      std::scoped_lock lk(profile_lock);
      v.assign(profile_data.begin(), profile_data.end());
      profile_data.clear();
    }
    std::sort(v.begin(), v.end(), [](const auto &a, const auto &b) {
      return a.second.seconds > b.second.seconds;
    });
    double total = 0;
    for (const auto &[k, e] : v) {
      total += e.seconds;
    }
    DC_INFO("gpu profile: %.1f ms GPU total", total * 1e3);
    for (size_t k = 0; k < v.size() && k < 15; ++k) {
      DC_INFO("gpu profile: %7.1f ms %6llu calls  %s",
              v[k].second.seconds * 1e3, (unsigned long long)v[k].second.calls,
              v[k].first.c_str());
    }
  }

  // Command buffers on one queue complete in order, and their completion
  // handlers run in order on Metal's serial completion queue.
  //
  // Callbacks share one event signal, encoded when the command buffer is
  // committed: a signal per callback (async compute releases one after every
  // dispatch) ends the open encoder each time. Nothing on the GPU waits on
  // the event, so the callbacks only run a little later, still in order.
  // DC_NO_SIGNAL_BATCH: one signal per callback, right where it is recorded.
  void Signal(std::function<void()> done) {
    if (!signal_batch) {
      EndPass();
    }
    EnsureCommandBuffer();
    if (!signal_pending) {
      signal_pending = ++signal_value;
    }
    {
      std::scoped_lock lk(signal_lock);
      signal_queue.emplace_back(signal_pending, std::move(done));
    }
    if (!signal_batch) {
      FlushSignal();
    }
  }

  // Encodes the pending signal; no encoder may be open.
  void FlushSignal() {
    if (!signal_pending) {
      return;
    }
    const u64 v = signal_pending;
    signal_pending = 0;
    [cmd encodeSignalEvent:signal_event value:v];
    [signal_event notifyListener:signal_listener
                         atValue:v
                           block:^(id<MTLSharedEvent> e, uint64_t) {
                             RunSignals(e.signaledValue);
                           }];
  }

  // Listener queue: runs, in order, the callbacks of every value reached.
  void RunSignals(u64 reached) {
    for (;;) {
      std::function<void()> fn;
      {
        std::scoped_lock lk(signal_lock);
        if (signal_queue.empty() || signal_queue.front().first > reached) {
          return;
        }
        fn = std::move(signal_queue.front().second);
        signal_queue.pop_front();
      }
      fn();
    }
  }

  // GPU watchdog: a hung command buffer never completes, so a thread also
  // checks how long the oldest pending one has been waiting.
  static constexpr u64 kWdRing = 4096;
  std::atomic<u64> wd_committed{0}, wd_done{0};
  std::atomic<s64> wd_commit_ns[kWdRing] = {};
  // A draw or dispatch, formatted only if the watchdog reports its buffer.
  struct WorkNote {
    void (*format)(const WorkNote &, std::string &);
    const char *fmt;
    u64 args[5];
  };
  template <typename... Args>
  static void FormatNote(const WorkNote &n, std::string &out) {
    char line[96];
    [&]<size_t... I>(std::index_sequence<I...>) {
      std::snprintf(line, sizeof(line), n.fmt, static_cast<Args>(n.args[I])...);
    }(std::index_sequence_for<Args...>{});
    out += line;
  }
  static std::string FormatNotes(const std::vector<WorkNote> &notes) {
    std::string out;
    for (const WorkNote &n : notes) {
      n.format(n, out);
    }
    return out;
  }
  std::vector<WorkNote> wd_work; // draws/dispatches recorded into `cmd`
  std::mutex wd_lock;
  std::vector<WorkNote> wd_desc[kWdRing];

  template <typename... Args> void NoteWork(const char *fmt, Args... args) {
    static_assert(sizeof...(Args) <= 5);
    CpuTimer t(cpu_profile ? &cpu_notework_s : nullptr);
    if (wd_work.size() >= 128) { // the latest ones tell the most
      wd_work.erase(wd_work.begin(), wd_work.begin() + 64);
    }
    wd_work.push_back({&FormatNote<Args...>, fmt, {static_cast<u64>(args)...}});
  }

  static s64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  void StartWatchdog(double limit_s) {
    static std::once_flag once;
    std::call_once(once, [this, limit_s] {
      std::thread([this, limit_s] {
        const s64 limit_ns = static_cast<s64>(limit_s * 3e9);
        for (;;) {
          std::this_thread::sleep_for(std::chrono::milliseconds(250));
          const u64 done = wd_done.load(std::memory_order_acquire);
          if (done >= wd_committed.load(std::memory_order_acquire)) {
            continue;
          }
          const s64 t = wd_commit_ns[(done + 1) % kWdRing].load(
              std::memory_order_relaxed);
          if (NowNs() - t > limit_ns) {
            std::string work;
            {
              std::scoped_lock lk(wd_lock);
              work = FormatNotes(wd_desc[(done + 1) % kWdRing]);
            }
            DC_ERROR("gpu watchdog: command buffer pending for %.2f s, "
                     "exiting; it holds:\n%s",
                     (NowNs() - t) / 1e9, work.c_str());
            std::_Exit(3);
          }
        }
      }).detach();
    });
  }

  // Command buffers on one queue complete in order: the last one is enough.
  void WaitLast() {
    if (last_committed) {
      const auto t0 = std::chrono::steady_clock::now();
      const u64 lt = Gpu::LtOn() ? Gpu::LtNow() : 0;
      [last_committed waitUntilCompleted];
      if (lt) {
        DC_INFO("lt %.0f wait gpu %.0f us", Gpu::LtUs(Gpu::LtNow()), (Gpu::LtNow() - lt) / 1e3);
      }
      if (stats) {
        ++stat_waits;
        stat_wait_s +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                .count();
      }
      last_committed = nil;
    }
  }

  // Upload chunks the GPU is done with, by power-of-two size: fresh shared
  // buffers cost page faults and zeroing per byte (~100 MB a frame while a
  // track streams in its textures).
  std::mutex chunk_pool_lock;
  std::unordered_map<u64, std::vector<id<MTLBuffer>>> chunk_pool;
  id<MTLBuffer> NewChunk(u64 size) {
    const u64 cls = std::bit_ceil(std::max(kUploadChunk, size));
    {
      std::scoped_lock lk(chunk_pool_lock);
      auto &v = chunk_pool[cls];
      if (!v.empty()) {
        id<MTLBuffer> b = v.back();
        v.pop_back();
        return b;
      }
    }
    return [device newBufferWithLength:cls options:MTLResourceStorageModeShared];
  }
  void RecycleChunks(std::vector<id<MTLBuffer>> &bufs) {
    std::scoped_lock lk(chunk_pool_lock);
    for (id<MTLBuffer> b : bufs) {
      auto &v = chunk_pool[b.length];
      if (v.size() < 8) { // keep a bounded amount around
        v.push_back(b);
      }
    }
    bufs.clear();
  }

  // Bump-allocates transient GPU-visible memory for this command buffer.
  std::pair<id<MTLBuffer>, u64> Upload(const void *data, u64 size) {
    size = (size + 15) & ~15ull;
    if (!chunk || chunk_off + size > chunk.length) {
      if (chunk) {
        chunks.push_back(chunk);
      }
      chunk = NewChunk(size);
      chunk_off = 0;
    }
    const u64 off = chunk_off;
    if (data) {
      std::memcpy(static_cast<u8 *>(chunk.contents) + off, data, size);
    }
    chunk_off += size;
    return {chunk, off};
  }

  // ================= shaders =================

  Program *GetProgram(Stage stage, u64 addr, const u32 *ud, u32 num_user_sgprs,
                      u32 vgpr_comp_cnt, u32 compute_rsrc2 = 0) {
    if (!Readable(addr, 16)) {
      return nullptr;
    }
    const u32 *code = reinterpret_cast<const u32 *>(addr);
    auto &list = shaders[addr ^ (static_cast<u64>(stage) << 60)];
    for (auto &e : list) {
      // Code is compared with memory once per frame: shaders are not rewritten
      // while a frame is being drawn.
      if (e.checked_frame == frame_count) {
        const Program &p = *e.prog;
        if (p.fetch_ud < 0 ||
            (u64(ud[p.fetch_ud]) | (u64(ud[p.fetch_ud + 1]) << 32)) == p.fetch_address) {
          return e.prog.get();
        }
      }
      if (!Readable(addr, e.code.size() * 4) ||
          std::memcmp(e.code.data(), code, e.code.size() * 4) != 0) {
        continue;
      }
      const Program &p = *e.prog;
      if (p.fetch_ud >= 0) {
        const u64 f = u64(ud[p.fetch_ud]) | (u64(ud[p.fetch_ud + 1]) << 32);
        if (f != p.fetch_address || !Readable(f, e.fetch.size() * 4) ||
            std::memcmp(e.fetch.data(), reinterpret_cast<const void *>(f),
                        e.fetch.size() * 4) != 0) {
          continue;
        }
      }
      e.checked_frame = frame_count;
      return e.prog.get();
    }
    Shader::BuildInput in{stage, addr, ud, num_user_sgprs, vgpr_comp_cnt};
    in.compute_rsrc2 = compute_rsrc2;
    auto prog = Shader::BuildProgram(in);
    if (!prog) {
      ReportOnce("cannot decode shader at " + std::to_string(addr));
      return nullptr;
    }
    if (dump_dir) {
      char path[512];
      std::snprintf(path, sizeof(path), "%s/%s_%llx_%016llx.bin", dump_dir,
                    stage == Stage::Vertex     ? "vs"
                    : stage == Stage::Fragment ? "ps"
                                               : "cs",
                    addr, prog->hash);
      const u32 size = Shader::Gcn::ProgramSize(code);
      std::ofstream(path, std::ios::binary)
          .write(reinterpret_cast<const char *>(code), size);
      if (prog->fetch_address && prog->fetch_bytes) { // its inlined fetch subroutine
        std::snprintf(path, sizeof(path), "%s/fetch_%llx.bin", dump_dir, (unsigned long long)prog->fetch_address);
        std::ofstream(path, std::ios::binary)
            .write(reinterpret_cast<const char *>(prog->fetch_address), prog->fetch_bytes);
      }
    }
    ShaderEntry e;
    u32 bytes = 16;
    for (const auto &i : prog->code) {
      if (i.pc < Shader::kFetchPcBase) {
        bytes = std::max<u32>(bytes, i.pc + i.size * 4);
      }
    }
    e.code.assign(code, code + bytes / 4);
    if (prog->fetch_address) {
      const auto *f = reinterpret_cast<const u32 *>(prog->fetch_address);
      e.fetch.assign(f, f + prog->fetch_bytes / 4);
    }
    e.prog = std::move(prog);
    list.push_back(std::move(e));
    return list.back().prog.get();
  }

  // MSL for `p`; GPU thread only (reports and dumps).
  std::string MslSource(const Program &p, const Shader::MslOptions &o,
                        u64 key) {
    std::vector<std::string> unsupported;
    std::string src = Shader::GenerateMsl(p, o, &unsupported);
    const char *stage = p.stage == Stage::Vertex     ? "vs"
                        : p.stage == Stage::Fragment ? "ps"
                                                     : "cs";
    for (const auto &u : unsupported) {
      ReportOnce("shader op not translated: " + u);
      // Once per op and shader: which programs lose that op.
      char who[64];
      std::snprintf(who, sizeof(who), " in %s %#llx", stage, (unsigned long long)p.address);
      ReportOnce("shader op not translated: " + u + who);
    }
    if (dump_dir) {
      char path[512];
      std::snprintf(path, sizeof(path), "%s/%s_%016llx_%016llx.metal", dump_dir,
                    stage, p.hash, key);
      std::ofstream(path) << src;
    }
    // DC_MSL_OVERRIDE=<dir>: <dir>/<stage>_<program hash>.metal replaces the
    // generated source (hand-edited experiments)
    static const char *override_dir = std::getenv("DC_MSL_OVERRIDE");
    if (override_dir) {
      char path[512];
      std::snprintf(path, sizeof(path), "%s/%s_%016llx.metal", override_dir,
                    stage, p.hash);
      if (std::ifstream in{path}) {
        src.assign(std::istreambuf_iterator<char>(in), {});
        DC_INFO("gpu: MSL override %s", path);
      }
    }
    return src;
  }

  // Moves finished pipelines into the caches.
  // ================= compute pipeline prewarm =================
  // Every compute pipeline built is recorded (guest code and address, and the
  // inputs of its key) in ~/Library/Caches/DriveclubNative/
  // compute_pipelines.bin. The next launch rebuilds them all on the compiler
  // threads from the start, with the current translator, so the first
  // dispatches after an update (new MSL) or a cleared Metal cache do not wait
  // up to seconds for a compile. DC_NO_PREWARM=1: off.
  struct PrewarmEntry {
    u64 pkey = 0, addr = 0;
    u32 rsrc2 = 0, group = 0, map_hoist = 0, linear_mask = 0, linear_aligned = 0;
    std::array<u32, 16> ud{};
    std::vector<TextureType> types;
    std::vector<u32> code;
  };
  std::mutex prewarm_lock;
  std::unordered_set<u64> prewarm_known;
  static std::string PrewarmPath() {
    const char *home = std::getenv("HOME");
    return home ? std::string(home) + "/Library/Caches/DriveclubNative/compute_pipelines.bin" : "";
  }
  static u64 PrewarmKey(u64 hash, u32 rsrc2, bool map_hoist, u32 linear_mask, u32 linear_aligned,
                        const std::vector<TextureType> &types, u32 group) {
    u64 key = Hash(hash);
    key = Hash(~0u, key); // probe_pc: probed dispatches are not recorded
    key = Hash(rsrc2, key);
    key = Hash(map_hoist, key);
    key = Hash(linear_mask, key);
    key = Hash(linear_aligned, key);
    key = Fnv(types.data(), types.size() * sizeof(TextureType), key);
    return key;
  }
  void PrewarmRecord(const PrewarmEntry &e) {
    const std::string path = PrewarmPath();
    std::scoped_lock lk(prewarm_lock);
    if (path.empty() || !prewarm_known.insert(e.pkey).second) {
      return;
    }
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    std::ofstream f(path, std::ios::binary | std::ios::app);
    auto put = [&](const void *d, size_t n) { f.write(static_cast<const char *>(d), n); };
    const u32 ntex = static_cast<u32>(e.types.size()), ncode = static_cast<u32>(e.code.size());
    put(&e.pkey, 8); put(&e.addr, 8);
    put(&e.rsrc2, 4); put(&e.group, 4); put(&e.map_hoist, 4); put(&e.linear_mask, 4); put(&e.linear_aligned, 4);
    put(e.ud.data(), 64);
    put(&ntex, 4); put(e.types.data(), ntex * sizeof(TextureType));
    put(&ncode, 4); put(e.code.data(), ncode * 4);
  }
  void PrewarmStart() {
    const std::string path = PrewarmPath();
    if (std::getenv("DC_NO_PREWARM") || path.empty()) {
      return;
    }
    std::ifstream f(path, std::ios::binary);
    auto get = [&](void *d, size_t n) { return static_cast<bool>(f.read(static_cast<char *>(d), n)); };
    std::vector<PrewarmEntry> entries;
    for (;;) {
      PrewarmEntry e;
      u32 ntex = 0, ncode = 0;
      if (!get(&e.pkey, 8) || !get(&e.addr, 8) || !get(&e.rsrc2, 4) || !get(&e.group, 4) ||
          !get(&e.map_hoist, 4) || !get(&e.linear_mask, 4) || !get(&e.linear_aligned, 4) ||
          !get(e.ud.data(), 64) || !get(&ntex, 4) || ntex > 64) {
        break;
      }
      e.types.resize(ntex);
      if (!get(e.types.data(), ntex * sizeof(TextureType)) || !get(&ncode, 4) || ncode > (1u << 20)) {
        break;
      }
      e.code.resize(ncode);
      if (!get(e.code.data(), ncode * 4)) {
        break;
      }
      prewarm_known.insert(e.pkey);
      entries.push_back(std::move(e));
    }
    DC_INFO("gpu: prewarming %zu compute pipelines", entries.size());
    for (auto &e : entries) {
      compiler->Submit([this, e = std::move(e)] {
        Shader::BuildInput in{Stage::Compute, e.addr, e.ud.data(), Bits(e.rsrc2, 1, 5),
                              Bits(e.rsrc2, 11, 2), e.code.data()};
        in.compute_rsrc2 = e.rsrc2;
        auto prog = Shader::BuildProgram(in);
        if (!prog) {
          return;
        }
        const u64 key = PrewarmKey(prog->hash, e.rsrc2, e.map_hoist, e.linear_mask,
                                   e.linear_aligned, e.types, e.group);
        if (Hash(e.group, key) != e.pkey) {
          return; // the key's inputs changed meaning: let the dispatch compile it
        }
        Shader::MslOptions o;
        o.textures = e.types;
        o.map_hoist = e.map_hoist;
        o.linear_mask = e.linear_mask;
        o.linear_aligned = e.linear_aligned;
        const std::string src = Shader::GenerateMsl(*prog, o, nullptr);
        id<MTLFunction> fn = compiler->Function(key, src, e.addr);
        if (!fn) {
          return;
        }
        MTLComputePipelineDescriptor *d = [MTLComputePipelineDescriptor new];
        d.computeFunction = fn;
        d.label = [NSString stringWithFormat:@"cs %#llx", e.addr];
        d.maxTotalThreadsPerThreadgroup = e.group;
        NSError *err = nil;
        id<MTLComputePipelineState> pso =
            [device newComputePipelineStateWithDescriptor:d options:MTLPipelineOptionNone
                                               reflection:nil error:&err];
        if (pso && pso.maxTotalThreadsPerThreadgroup >= e.group) { // failures: the dispatch reports
          std::scoped_lock lk(done_lock);
          done_compute.emplace_back(e.pkey, pso);
        }
      }, /*urgent=*/false);
    }
  }

  void TakeCompiled() {
    std::scoped_lock lk(done_lock);
    for (auto &[k, pso] : done_render) {
      pipelines[k] = pso;
      pending.erase(k);
    }
    for (auto &[k, pso] : done_compute) {
      compute_pipelines[k] = pso;
      pending.erase(k);
    }
    done_render.clear();
    done_compute.clear();
  }

  // Blocks until the pipeline `key` has been compiled.
  void WaitCompiled(u64 key) {
    if (!pending.count(key)) {
      return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    while (pending.count(key)) {
      {
        std::unique_lock lk(done_lock);
        done_cv.wait(
            lk, [&] { return !done_render.empty() || !done_compute.empty(); });
      }
      TakeCompiled();
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    if (ms > 100) {
      DC_WARN("gpu: waited %.0f ms for a pipeline (%zu still compiling)", ms,
              pending.size());
    }
  }

  // ================= textures =================

  // Stand-in for textures we cannot provide: white, or zero for what the
  // hardware itself reads as zero (null or INVALID-format descriptors).
  id<MTLTexture> Dummy(TextureType tt, bool zero = false) {
    const u32 key = static_cast<u32>(tt.dim) |
                    (static_cast<u32>(tt.sample) << 4) | (zero ? 0x100u : 0u);
    auto it = dummy_textures.find(key);
    if (it != dummy_textures.end()) {
      return it->second;
    }
    MTLTextureDescriptor *d = [MTLTextureDescriptor new];
    d.textureType = tt.dim == TexDim::Tex2D        ? MTLTextureType2D
                    : tt.dim == TexDim::Tex2DArray ? MTLTextureType2DArray
                                                   : MTLTextureType3D;
    d.pixelFormat = tt.sample == TexSample::Float  ? MTLPixelFormatRGBA8Unorm
                    : tt.sample == TexSample::Uint ? MTLPixelFormatRGBA8Uint
                                                   : MTLPixelFormatRGBA8Sint;
    d.width = d.height = 1;
    if (tt.dim == TexDim::Tex3D) {
      d.depth = 1;
    } else if (tt.dim == TexDim::Tex2DArray) {
      d.arrayLength = 1;
    }
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> t = [device newTextureWithDescriptor:d];
    const u32 white = zero                            ? 0u
                      : tt.sample == TexSample::Float ? 0xFFFFFFFFu
                                                      : 0x01010101u;
    [t replaceRegion:MTLRegionMake3D(0, 0, 0, 1, 1, 1)
          mipmapLevel:0
                slice:0
            withBytes:&white
          bytesPerRow:4
        bytesPerImage:4];
    dummy_textures[key] = t;
    return t;
  }

  // How a shader samples `f`: float, uint or sint.
  static TexSample SampleKind(MTLPixelFormat f) {
    switch (f) {
    case MTLPixelFormatR8Uint:
    case MTLPixelFormatRG8Uint:
    case MTLPixelFormatRGBA8Uint:
    case MTLPixelFormatR16Uint:
    case MTLPixelFormatRG16Uint:
    case MTLPixelFormatRGBA16Uint:
    case MTLPixelFormatR32Uint:
    case MTLPixelFormatRG32Uint:
    case MTLPixelFormatRGBA32Uint:
    case MTLPixelFormatRGB10A2Uint:
      return TexSample::Uint;
    case MTLPixelFormatR8Sint:
    case MTLPixelFormatRG8Sint:
    case MTLPixelFormatRGBA8Sint:
    case MTLPixelFormatR16Sint:
    case MTLPixelFormatRG16Sint:
    case MTLPixelFormatRGBA16Sint:
    case MTLPixelFormatR32Sint:
    case MTLPixelFormatRG32Sint:
    case MTLPixelFormatRGBA32Sint:
      return TexSample::Sint;
    default:
      return TexSample::Float;
    }
  }

  // Change detection over every byte. CRC32C in 8 independent lanes: Rosetta
  // turns the SSE4.2 crc32 into the ARM one, twice the speed of a multiply
  // chain (hot textures are hashed in full every frame, ~25 MB in a race).
  u64 Hash64(const u8 *p, u64 size) {
    stat_hashed += size;
    CpuTimer t(cpu_profile ? &cpu_hash_s : nullptr);
    u64 h[8] = {0x9e3779b9, 0xc2b2ae3d, 0x165667b1, 0x27d4eb2f,
                0x85ebca6b, 0xc2b2ae35, 0x27d4eb2d, 0x165667c5};
    u64 off = 0;
    for (; off + 64 <= size; off += 64) {
      u64 v[8];
      std::memcpy(v, p + off, sizeof(v));
      for (u32 k = 0; k < 8; ++k) {
        h[k] = _mm_crc32_u64(h[k], v[k]);
      }
    }
    for (; off < size; ++off) {
      h[0] = _mm_crc32_u8(static_cast<u32>(h[0]), p[off]);
    }
    return (h[0] | h[1] << 32) ^ ((h[2] | h[3] << 32) * 3) ^
           ((h[4] | h[5] << 32) * 5) ^ ((h[6] | h[7] << 32) * 7);
  }

  u64 ChunkHash(const u8 *base, u64 size, u32 chunk) {
    const u64 off = u64(chunk) * kChunk;
    return Hash64(base + off, std::min(kChunk, size - off));
  }

  // PRT textures with pages not committed (GT7 keeps only the mip tail of
  // some): a copy of the range with the mapped pages and zeros elsewhere, as
  // unmapped PRT tiles read on the GPU. Rebuilt on every call so pages the
  // game commits later show up. nullptr when nothing of it is mapped.
  struct SparseStaging {
    std::vector<u8> bytes;
    std::vector<std::pair<u64, u64>> spans; // mapped [lo, hi) of the last build
  };
  std::unordered_map<u64, SparseStaging> sparse_staging;
  const u8 *SparseSource(u64 addr, u64 size) {
    std::vector<std::pair<u64, u64>> spans;
    for (u64 at = addr, end = addr + size; at < end;) {
      Core::Memory::Vma vma;
      if (!Core::Memory::Query(at, true, &vma) || vma.start >= end) {
        break;
      }
      const u64 lo = std::max(at, vma.start), hi = std::min(end, vma.start + vma.size);
      if (vma.type == Core::Memory::VmaType::Direct || vma.type == Core::Memory::VmaType::Flexible) {
        spans.emplace_back(lo, hi);
      }
      at = vma.start + vma.size;
    }
    if (spans.empty()) {
      sparse_staging.erase(addr);
      return nullptr;
    }
    SparseStaging &st = sparse_staging[addr];
    if (st.bytes.size() != size || st.spans != spans) { // layout changed: holes back to zero
      st.bytes.assign(size, 0);
      st.spans = spans;
    }
    for (const auto &[lo, hi] : spans) { // the committed pages, as they are now
      std::memcpy(st.bytes.data() + (lo - addr), reinterpret_cast<const u8 *>(lo), hi - lo);
    }
    return st.bytes.data();
  }

  // True if the texture memory differs from what was uploaded.
  bool TextureChanged(TextureEntry &e, const u8 *addr, u64 size) {
    const u32 n = static_cast<u32>(e.chunks.size());
    if (e.hot) {
      for (u32 c = 0; c < n; ++c) {
        if (ChunkHash(addr, size, c) != e.chunks[c]) {
          return true;
        }
      }
      return false;
    }
    const u32 c = e.next_chunk++ % n;
    return ChunkHash(addr, size, c) != e.chunks[c];
  }

  // Decodes a T# and returns a Metal texture for it.
  // NS_RETURNS_RETAINED (here and below): hot getters hand back +1, so ARC
  // needs no autorelease per call.
  id<MTLTexture> GetTexture(const u32 t[8], TextureType *type_out,
                            bool storage = false) NS_RETURNS_RETAINED {
    // Writable images mark their target written and are rare: never cached.
    // A texture dump asks for the raw bytes on every upload: also uncached.
    if (storage || dump_texture || !tex_cache) {
      return GetTextureUncached(t, type_out, storage);
    }
    if (cpu_profile) {
      ++cpu_gettex_calls;
    }
    if (texture_cache_frame != frame_count) {
      texture_cache.clear();
      texture_cache_frame = frame_count;
    }
    // The descriptor is the key: t[6] and t[7] are unused by the decoder.
    u64 key = Hash(t[0]);
    for (u32 k = 1; k < 6; ++k) {
      key = Hash(t[k], key);
    }
    if (auto it = texture_cache.find(key); it != texture_cache.end()) {
      if (it->second.epoch == texture_epoch) {
        if (cpu_profile) {
          ++cpu_gettex_hits;
        }
        if (tex_cache_check) {
          // Same descriptor, slow path: the texture object and the type must
          // agree, or the cache is answering for a surface that changed.
          TextureType slow_type;
          id<MTLTexture> slow = GetTextureUncached(t, &slow_type, false);
          ++tex_cache_checks;
          const bool type_same =
              slow_type.dim == it->second.type.dim &&
              slow_type.sample == it->second.type.sample &&
              slow_type.cube == it->second.type.cube &&
              slow_type.compare == it->second.type.compare &&
              slow_type.ufloat == it->second.type.ufloat;
          if (slow != it->second.tex || !type_same) {
            if (++tex_cache_mismatches <= 8) {
              DC_ERROR("gpu: tex cache mismatch %#llx (cached %p slow %p, type "
                       "%d/%d sample %d/%d)",
                       (u64(t[0]) | (u64(t[1] & 0xFF) << 32)) << 8,
                       (__bridge void *)it->second.tex, (__bridge void *)slow,
                       (int)it->second.type.dim, (int)slow_type.dim,
                       (int)it->second.type.sample, (int)slow_type.sample);
            }
          }
        }
        *type_out = it->second.type;
        return it->second.tex;
      }
    }
    CpuTimer t_(cpu_profile ? &cpu_gettex_s : nullptr);
    TextureType tt;
    id<MTLTexture> tex = GetTextureUncached(t, &tt, false);
    texture_cache[key] = TextureCacheEntry{tex, tt, texture_epoch};
    *type_out = tt;
    return tex;
  }

  id<MTLTexture> GetTextureUncached(const u32 t[8], TextureType *type_out,
                                    bool storage) NS_RETURNS_RETAINED {
    const u64 addr =
        ((u64(t[0]) | (u64(t[1] & 0xFF) << 32)) << 8) & kGpuAddrMask;
    const u32 dfmt = (t[1] >> 20) & 0x3F;
    const u32 nfmt = (t[1] >> 26) & 0xF;
    const u32 width = (t[2] & 0x3FFF) + 1;
    const u32 height = ((t[2] >> 14) & 0x3FFF) + 1;
    const u32 swz = t[3] & 0xFFF;
    const u32 tiling = (t[3] >> 20) & 0x1F;
    const u32 ttype = t[3] >> 28;
    const u32 depth = (t[4] & 0x1FFF) + 1;
    const u32 pitch = ((t[4] >> 13) & 0x3FFF) + 1;
    const u32 last_array = (t[5] >> 13) & 0x1FFF;
    const u32 base_array = std::min<u32>(t[5] & 0x1FFF, last_array);
    const u32 base_level = (t[3] >> 12) & 0xF;
    const u32 last_level = std::max(base_level, (t[3] >> 16) & 0xF);

    const TextureFormatInfo fi = TextureFormat(dfmt, nfmt);
    TextureType tt;
    tt.sample = fi.sample;
    if (ttype == 10) {
      tt.dim = TexDim::Tex3D;
    } else if (ttype == 11 || ttype == 12 || ttype == 13 || ttype == 15) {
      tt.dim = TexDim::Tex2DArray;
      tt.cube = ttype == 11 && !storage;
    }
    *type_out = tt;
    if (!addr || ttype < 8 || dfmt == 0) {
      return Dummy(tt, true);
    }

    // Slices a descriptor of an array covers (a cube is 6 faces from its base).
    const u32 slice_count = tt.dim != TexDim::Tex2DArray ? 1
                            : ttype == 11
                                ? std::max<u32>(6, last_array - base_array + 1)
                                : last_array - base_array + 1;

    // A level of a larger image (see MipAlias): that level of it.
    if (auto it = mip_alias.find(addr);
        it != mip_alias.end() && it->second.width == width &&
        it->second.height == height && fi.format != MTLPixelFormatInvalid &&
        PixelBytes(fi.format) == PixelBytes(it->second.fmt)) {
      if (RenderTarget *p = MipParent(addr, it->second.fmt, width, height)) {
        const u32 lv = it->second.level;
        const u32 sl = it->second.slice + (tt.dim == TexDim::Tex2DArray ? base_array : 0);
        const bool as2d = tt.dim == TexDim::Tex2D;
        if (storage) {
          p->initialized = true;
          p->clear_pending = false;
          p->written = ++write_seq;
          NoteTargetWritten(*p);
          type_out->ufloat = p->tex.pixelFormat == MTLPixelFormatRG11B10Float;
          return TargetView(*p, lv + base_level, 1, MTLPixelFormatInvalid,
                            sl, slice_count, kIdentitySwizzle, as2d);
        }
        tt.sample = SampleKind(p->tex.pixelFormat);
        *type_out = tt;
        return TargetView(*p, lv + base_level, last_level - base_level + 1,
                          MTLPixelFormatInvalid, sl, slice_count, swz, as2d);
      }
    }

    // Render-to-texture: sample the render target directly. A block-compressed
    // T# over a target of one texel per block (a shader encoding BC blocks,
    // e.g. the cars' liveries): those blocks as a BC texture; over any other
    // target, the guest memory (the target's texels are not BC data).
    if (!storage && fi.compressed && tt.dim == TexDim::Tex2D) {
      if (RenderTarget *rt = Latest(color_targets, addr, tt.dim)) {
        if (id<MTLTexture> bc = BlocksAsCompressed(*rt, fi, width, height,
                                                   last_level + 1)) {
          if (base_level == 0 && last_level + 1 >= bc.mipmapLevelCount &&
              swz == kIdentitySwizzle) {
            return bc;
          }
          const u32 levels = std::min<u32>(last_level + 1, bc.mipmapLevelCount);
          return [bc newTextureViewWithPixelFormat:bc.pixelFormat
                                       textureType:MTLTextureType2D
                                            levels:NSMakeRange(std::min(base_level, levels - 1),
                                                               levels - std::min(base_level, levels - 1))
                                            slices:NSMakeRange(0, 1)
                                           swizzle:MTLTextureSwizzleChannelsMake(
                                                       Swizzle(swz), Swizzle(swz >> 3),
                                                       Swizzle(swz >> 6), Swizzle(swz >> 9))];
        }
      }
    }
    if (!storage && !fi.compressed) {
      // As shadPS4's image views: the T# format (same texel size) and swizzle.
      // Transient memory reused by a surface of another texel width is not this
      // image, even at the same size: GT7's 720x540 16F over a 512x256 11F
      // scratch, and its 1920x1080 16F half of a 1920x2160 target over an old
      // 1920x1080 R16F, which read as flat red.
      RenderTarget *latest = Latest(color_targets, addr, tt.dim, width, height, fi.bytes_per_block);
      // Cleared through memory after that surface was written: memory is the image.
      if (const auto fill = memory_fills.find(addr);
          latest && fill != memory_fills.end() && fill->second > latest->written) {
        latest = nullptr;
      }
      // A 2D T# over the first slice of an array or cube target (GT7's cs
      // 0x9000e31f00 reads face 0 of the reflection cube 0x900b1b0000 as 2D to
      // build its mip chain): a 2D view of that slice, not the guest memory,
      // which never holds the GPU's picture.
      if (!latest && tt.dim == TexDim::Tex2D) {
        RenderTarget *a = Latest(color_targets, addr, TexDim::Tex2DArray, width, height,
                                 fi.bytes_per_block);
        if (a && a->width == width && a->height == height &&
            a->tex.textureType == MTLTextureType2DArray &&
            !(memory_fills.count(addr) && memory_fills[addr] > a->written)) {
          const u32 lv0 = std::min(base_level, a->levels - 1);
          const u32 nlv = std::clamp<u32>(last_level - base_level + 1, 1, a->levels - lv0);
          MTLPixelFormat vf = a->tex.pixelFormat;
          if (fi.format != MTLPixelFormatInvalid && PixelBytes(fi.format) == PixelBytes(vf) &&
              !(fi.sample == TexSample::Float) == !(SampleKind(vf) == TexSample::Float)) {
            vf = fi.format;
          } else {
            tt.sample = SampleKind(vf);
            *type_out = tt;
          }
          return [a->tex newTextureViewWithPixelFormat:vf
                                           textureType:MTLTextureType2D
                                                levels:NSMakeRange(lv0, nlv)
                                                slices:NSMakeRange(0, 1)
                                               swizzle:MTLTextureSwizzleChannelsMake(
                                                           Swizzle(swz), Swizzle(swz >> 3),
                                                           Swizzle(swz >> 6), Swizzle(swz >> 9))];
        }
      }
      if (RenderTarget *rt = latest) {
        NoteRestoredRead(*rt, addr, false);
        // An array read past the slices rendered so far: grow it (shadPS4 sizes
        // images from the T#), the new slices read as zero until written.
        if (tt.dim == TexDim::Tex2DArray &&
            rt->depth_or_layers < base_array + slice_count &&
            rt->width == width && rt->height == height) {
          rt = &GetColorTarget(addr, rt->tex.pixelFormat, width, height,
                               rt->levels, TexDim::Tex2DArray,
                               base_array + slice_count);
        }
        if (last_level >= rt->levels && rt->width == width &&
            rt->height == height) { // the T# has the levels: so does the image
          rt = &GetColorTarget(addr, rt->tex.pixelFormat, width, height,
                               last_level + 1, rt->dim, rt->depth_or_layers);
        }
        if (!fi.compressed && fi.bytes_per_block) {
          NoteMipLevels(*rt, t, fi.bytes_per_block);
        }
        MTLPixelFormat view_fmt = MTLPixelFormatInvalid;
        if (fi.format != MTLPixelFormatInvalid && !fi.compressed &&
            fi.format != rt->tex.pixelFormat && PixelBytes(fi.format) &&
            PixelBytes(fi.format) == PixelBytes(rt->tex.pixelFormat) &&
            !(fi.sample == TexSample::Float) ==
                !(SampleKind(rt->tex.pixelFormat) == TexSample::Float)) {
          view_fmt = fi.format;
        }
        if (view_fmt == MTLPixelFormatInvalid) {
          tt.sample = SampleKind(
              rt->tex.pixelFormat); // the shader reads what the view holds
          *type_out = tt;
        }
        // COMP_SWAP 2 stores exports as ABGR in memory, which a T# undoes with
        // its selectors; host targets keep export order, so the selectors move.
        u32 rt_swz = swz;
        // A storage write goes through its T# selectors too, so the same T#
        // reads back in place; host storage textures ignore selectors. Constant
        // selectors stay: the channel does not exist in memory.
        if (const auto it = storage_swizzles.find(addr);
            it != storage_swizzles.end() && it->second == swz) {
          rt_swz = 0;
          for (u32 c = 0; c < 4; ++c) {
            const u32 sel = (swz >> (3 * c)) & 7;
            rt_swz |= (sel >= 4 ? 4 + c : sel) << (3 * c);
          }
        } else if (rt->comp_swap == 2) {
          rt_swz = 0;
          for (u32 c = 0; c < 4; ++c) {
            u32 sel = (swz >> (3 * c)) & 7;
            if (rt->three_channels) { // the swap reverses three channels: R <-> B
              rt_swz |= (sel == 4 ? 6 : sel == 6 ? 4 : sel) << (3 * c);
            } else {
              rt_swz |= (sel >= 4 ? 11 - sel : sel) << (3 * c);
            }
          }
        }
        if (RenderTarget *crop = CropOf(*rt, tt, width, height, base_level, last_level)) {
          return TargetView(*crop, 0, 1, view_fmt, 0, 1, rt_swz);
        }
        return TargetView(*rt, base_level, last_level - base_level + 1,
                          view_fmt, base_array, slice_count, rt_swz);
      }
      if (tt.dim == TexDim::Tex2D) {
        // As for color above: scratch memory once used by a depth surface of
        // another size is not this image (GT7's 720x540 over a 1024x512 depth).
        RenderTarget *rt = Latest(depth_targets, addr, tt.dim, width, height);
        if (rt && rt->width == width && rt->height == height) {
          NoteRestoredRead(*rt, addr, true);
          return swz == kIdentitySwizzle
                     ? rt->tex
                     : TargetView(*rt, 0, rt->levels, MTLPixelFormatInvalid, 0,
                                  1, swz);
        }
      }
      if (tt.dim == TexDim::Tex2DArray) {
        // Shadow cascades: a layered depth surface (DB_DEPTH_VIEW) read back
        // as a 2D array, GT7's sun cascades 0x900da10000. Guest memory never
        // holds it, so falling through samples zeros: everything shadowed.
        RenderTarget *rt = Latest(depth_targets, addr, tt.dim, width, height);
        if (rt && rt->width == width && rt->height == height) {
          if (rt->depth_or_layers < base_array + slice_count) { // as for color
            rt = &GetTarget(depth_targets, addr, rt->tex.pixelFormat, width, height,
                            true, rt->levels, TexDim::Tex2DArray,
                            base_array + slice_count);
          }
          return TargetView(*rt, 0, rt->levels, MTLPixelFormatInvalid, base_array,
                            slice_count, swz);
        }
      }
    }

    const u32 layers = tt.dim == TexDim::Tex2DArray ? base_array + slice_count
                       : tt.dim == TexDim::Tex3D    ? depth
                                                    : 1;

    if (storage) {
      // Written by shaders: keep it GPU-side like a render target, so
      // later draws sample what the compute pass produced.
      if (fi.format == MTLPixelFormatInvalid || fi.compressed) {
        ReportOnce("storage image type/format not supported (dim=" +
                   std::to_string(ttype) + " dfmt=" + std::to_string(dfmt) +
                   " nfmt=" + std::to_string(nfmt) +
                   " addr=" + std::to_string(addr) + " " +
                   std::to_string(width) + "x" + std::to_string(height) + ")");
        return Dummy(tt);
      }
      const MTLPixelFormat f = fi.format == MTLPixelFormatRGBA8Unorm_sRGB
                                   ? MTLPixelFormatRGBA8Unorm
                                   : fi.format;
      type_out->ufloat = f == MTLPixelFormatRG11B10Float;
      // A new image starts as its guest memory (shadPS4 uploads an image
      // before a shader writes it): a shader may write only part of it, and
      // the rest is what the game put there (e.g. a color grading LUT).
      id<MTLTexture> initial = nil;
      if (!Latest(color_targets, addr, tt.dim, width, height) &&
          !fi.gpu_only) {
        TextureType mt;
        initial = GetTexture(t, &mt, false);
        if (initial.pixelFormat != f || initial.width != width >> base_level ||
            initial.textureType != (tt.dim == TexDim::Tex3D
                                        ? MTLTextureType3D
                                    : tt.dim == TexDim::Tex2DArray
                                        ? MTLTextureType2DArray
                                        : MTLTextureType2D)) {
          initial = nil;
        }
      }
      RenderTarget &target = GetColorTarget(addr, f, width, height,
                                            last_level + 1, tt.dim, layers);
      if (swz != kIdentitySwizzle) {
        storage_swizzles[addr] = swz;
      } else {
        storage_swizzles.erase(addr);
      }
      if (initial && !target.initialized) {
        EndPass();
        EnsureCommandBuffer();
        id<MTLBlitCommandEncoder> b = [cmd blitCommandEncoder];
        const NSUInteger n =
            std::min<NSUInteger>(initial.mipmapLevelCount,
                                 target.levels - std::min(base_level,
                                                          target.levels - 1));
        const NSUInteger slices =
            tt.dim == TexDim::Tex2DArray
                ? std::min<NSUInteger>(initial.arrayLength,
                                       target.tex.arrayLength)
                : 1;
        [b copyFromTexture:initial
                 sourceSlice:0
                 sourceLevel:0
                   toTexture:target.tex
            destinationSlice:0
            destinationLevel:base_level
                  sliceCount:slices
                  levelCount:n];
        [b endEncoding];
      }
      target.initialized = true;
      target.clear_pending = false;
      target.written = ++write_seq;
      NoteTargetWritten(target);
      return TargetView(target, base_level, last_level + 1 - base_level,
                        MTLPixelFormatInvalid, base_array, slice_count);
    }

    if (fi.format == MTLPixelFormatInvalid || fi.gpu_only) {
      ReportOnce("texture format dfmt=" + std::to_string(dfmt) +
                 " nfmt=" + std::to_string(nfmt) + " not supported");
      return Dummy(tt);
    }
    if (!Tiling::Supported(tiling)) {
      ReportOnce("texture tile mode " + std::to_string(tiling) +
                 " not supported yet");
      return Dummy(tt);
    }
    const u32 width_el = fi.compressed ? (width + 3) / 4 : width;
    const u32 height_el = fi.compressed ? (height + 3) / 4 : height;
    const u32 thick = Tiling::Thickness(tiling);
    // The whole mip chain (3D textures: level 0 only), laid out like shadPS4's
    // ComputeImageSize: levels one after another, each with all its slices.
    const u32 max_levels =
        1 + static_cast<u32>(std::log2(std::max(width, height)));
    const u32 levels =
        tt.dim == TexDim::Tex3D ? 1 : std::min(last_level + 1, max_levels);
    const bool pow2 = (t[3] >> 25) & 1;
    Tiling::MipLevel mips[16];
    const u64 size =
        Tiling::MipLayout(tiling, pitch, height, layers, levels,
                          fi.bytes_per_block, fi.compressed, pow2, mips);
    // Where the texel bytes come from: guest memory, or a staging copy of a
    // partly committed PRT texture (tile modes 16..20 on GCN are PRT).
    const u8 *tex_base = reinterpret_cast<const u8 *>(addr);
    static const bool no_sparse = std::getenv("DC_NO_SPARSE_TEX") != nullptr;
    if (!Readable(addr, size) && !no_sparse && tiling >= 16 && tiling <= 20) {
      if (const u8 *b = SparseSource(addr, size)) {
        tex_base = b;
        ReportOnce("partly committed PRT textures read their holes as zero");
      }
    }
    if (tex_base == reinterpret_cast<const u8 *>(addr) && !Readable(addr, size)) {
      ReportOnce("texture memory not mapped");
      static int logged = 0;
      if (logged < 12) { // which VMAs cover it: several adjacent ones, or holes
        ++logged;
        std::string spans;
        for (u64 at = addr; at < addr + size;) {
          Core::Memory::Vma vma;
          if (!Core::Memory::Query(at, false, &vma) || vma.start + vma.size <= at) {
            spans += " hole@" + std::to_string((at - addr) >> 10) + "K";
            at = (at + 0x10000) & ~0xFFFFull;
            continue;
          }
          spans += " [" + std::to_string((vma.start > addr ? vma.start - addr : 0) >> 10) + "K+" +
                   std::to_string((vma.start + vma.size - std::max(vma.start, at)) >> 10) + "K]";
          at = vma.start + vma.size;
          if (spans.size() > 600) break;
        }
        DC_WARN("gpu: unmapped texture %#llx %u KB tile %u %ux%u:%s", (unsigned long long)addr,
                unsigned(size >> 10), tiling, width, height, spans.c_str());
      }
      return Dummy(tt);
    }

    // GT7's per-frame memory (0x200000000..0x208800000) holds images its async
    // compute fills with buffer stores on the GPU, e.g. the clustered light
    // grid (30x17x16 RG32) the forward shaders read. An upload reads guest
    // memory when the draw is encoded, before the GPU has run those
    // dispatches: the grid read as zeros, no light or probe reached any
    // cluster and the cockpit's indirect light was black. Linear images there
    // are copied from the guest memory buffer on the GPU timeline instead,
    // at the point of the bind, every time (no cache). DC_NO_GPU_LINEAR=1: off.
    static const bool gpu_linear = std::getenv("DC_NO_GPU_LINEAR") == nullptr;
    if (gpu_linear && !storage && Tiling::IsLinear(tiling) && !fi.compressed &&
        tt.dim != TexDim::Tex3D && levels == 1 && addr >= 0x200000000ull &&
        addr + size <= 0x208800000ull && size <= (16ull << 20)) {
      id<MTLBuffer> buf = nil;
      u64 off = 0;
      const u64 row = u64(mips[0].pitch_el) * fi.bytes_per_block;
      if (Locate(addr, size, &buf, &off) && !(off % fi.bytes_per_block) &&
          !(row % fi.bytes_per_block)) {
        u64 gkey = Hash(addr);
        for (u32 k = 1; k < 6; ++k) {
          gkey = Hash(t[k], gkey);
        }
        id<MTLTexture> gt = gpu_linear_tex[gkey];
        if (!gt) {
          MTLTextureDescriptor *d = [MTLTextureDescriptor new];
          d.textureType = tt.dim == TexDim::Tex2DArray ? MTLTextureType2DArray
                                                       : MTLTextureType2D;
          d.pixelFormat = fi.format;
          d.width = width;
          d.height = height;
          if (tt.dim == TexDim::Tex2DArray) {
            d.arrayLength = LayerLimit(layers); // Metal limit; more = a garbage T#
          }
          d.usage = MTLTextureUsageShaderRead;
          d.storageMode = MTLStorageModePrivate;
          d.swizzle = MTLTextureSwizzleChannelsMake(Swizzle(swz), Swizzle(swz >> 3),
                                                    Swizzle(swz >> 6), Swizzle(swz >> 9));
          gt = [device newTextureWithDescriptor:d];
          gpu_linear_tex[gkey] = gt;
        }
        if (gt) {
          auto &last = gpu_linear_copied[gkey];
          if (last.first == dispatch_seq && last.second == frame_count) {
            return gt; // no compute ran since the last copy this frame
          }
          last = {dispatch_seq, frame_count};
          EndPass();
          EnsureCommandBuffer();
          id<MTLBlitCommandEncoder> b = [cmd blitCommandEncoder];
          for (u32 l = 0; l < layers; ++l) {
            [b copyFromBuffer:buf
                       sourceOffset:off + mips[0].offset + mips[0].slice_size * l
                  sourceBytesPerRow:row
                sourceBytesPerImage:mips[0].slice_size
                         sourceSize:MTLSizeMake(width, height, 1)
                          toTexture:gt
                   destinationSlice:tt.dim == TexDim::Tex2DArray ? l : 0
                   destinationLevel:0
                  destinationOrigin:MTLOriginMake(0, 0, 0)];
          }
          [b endEncoding];
          if (trace) { // non-zero dwords of the copied range, as the GPU saw it
            const u8 *src = static_cast<const u8 *>(buf.contents) + off + mips[0].offset;
            const u64 n = mips[0].slice_size * layers / 4, a = addr;
            linear_watch.push_back({src, n, a});
            [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
              u64 nz = 0;
              const u32 *d = reinterpret_cast<const u32 *>(src);
              for (u64 i = 0; i < n; ++i) nz += d[i] != 0;
              DC_INFO("gpu: linear copy %#llx: %llu of %llu dwords non-zero",
                      (unsigned long long)a, (unsigned long long)nz, (unsigned long long)n);
            }];
          }
          return gt;
        }
      }
    }

    u64 key = Hash(addr);
    for (u32 k = 1; k < 6; ++k) {
      key = Hash(t[k], key);
    }
    TextureEntry &e = textures[key];
    // T# BASE_LEVEL..LAST_LEVEL: a view of those levels.
    auto view = [&]() -> id<MTLTexture> {
      if (base_level == 0 || base_level >= levels) {
        return e.tex;
      }
      if (!e.base_view) {
        e.base_view = [e.tex
            newTextureViewWithPixelFormat:e.tex.pixelFormat
                              textureType:e.tex.textureType
                                   levels:NSMakeRange(base_level,
                                                      levels - base_level)
                                   slices:NSMakeRange(
                                              0, e.tex.textureType ==
                                                         MTLTextureType2DArray
                                                     ? e.tex.arrayLength
                                                     : 1)];
      }
      return e.base_view;
    };
    if (e.tex && e.frame == frame_count) {
      if (cpu_profile) {
        ++cpu_gettex_hits;
      }
      return view();
    }
    if (e.tex) {
      if (!TextureChanged(e, tex_base, size)) {
        e.frame = frame_count;
        return view();
      }
      e.hot = true;
    }

    if (!e.tex) {
      MTLTextureDescriptor *d = [MTLTextureDescriptor new];
      d.textureType = tt.dim == TexDim::Tex2D        ? MTLTextureType2D
                      : tt.dim == TexDim::Tex2DArray ? MTLTextureType2DArray
                                                     : MTLTextureType3D;
      d.pixelFormat = fi.format;
      d.width = width;
      d.height = height;
      d.mipmapLevelCount = levels;
      if (tt.dim == TexDim::Tex3D) {
        d.depth = LayerLimit(layers);
      } else if (tt.dim == TexDim::Tex2DArray) {
        d.arrayLength = LayerLimit(layers); // Metal limit; more = a garbage T#
      }
      d.usage = MTLTextureUsageShaderRead;
      d.storageMode = cpu_upload ? MTLStorageModeManaged : MTLStorageModePrivate;
      // One-channel format: W selects its only channel (GT7's glyphs: 1,1,1,W).
      u32 tswz = swz;
      if (fi.format == MTLPixelFormatR8Unorm && ((tswz >> 9) & 7) == 7) {
        tswz = (tswz & 0x1FF) | (4u << 9);
      }
      d.swizzle =
          MTLTextureSwizzleChannelsMake(Swizzle(tswz), Swizzle(tswz >> 3),
                                        Swizzle(tswz >> 6), Swizzle(tswz >> 9));
      e.tex = [device newTextureWithDescriptor:d];
      if (!e.tex) {
        return Dummy(tt);
      }
      texture_bytes += e.tex.allocatedSize;
      texture_bytes_peak = std::max(texture_bytes_peak, texture_bytes);
    }
    const auto *src = tex_base;
    std::vector<u8> linear;
    {
    CpuTimer t(cpu_profile ? &cpu_upload_s : nullptr);
    for (u32 lv = 0; lv < levels; ++lv) {
      const Tiling::MipLevel &m = mips[lv];
      const u32 w = std::max(width >> lv, 1u), h = std::max(height >> lv, 1u);
      const u32 w_el = fi.compressed ? (w + 3) / 4 : w,
                h_el = fi.compressed ? (h + 3) / 4 : h;
      const u64 row = u64(w_el) * fi.bytes_per_block;
      const u64 image = row * h_el;
      if (cpu_upload) {
        linear.resize(image);
      }
      const u32 level_thick = Tiling::Thickness(m.tile_index);
      for (u32 l = 0; l < layers; ++l) {
        const u64 group_offset = m.offset + m.slice_size * (l / level_thick);
        auto [sbuf, soff] = cpu_upload ? std::pair<id<MTLBuffer>, u64>{nil, 0} : Upload(nullptr, image);
        u8 *out = cpu_upload ? linear.data() : static_cast<u8 *>(sbuf.contents) + soff;
        Tiling::DetileSlice(m.tile_index, src + group_offset, out,
                            w_el, h_el, m.pitch_el, fi.bytes_per_block,
                            l % level_thick);
        if (dump_texture == addr && dump_frames && lv == 0 &&
            (trace_frame ? frame_count + 2 >= trace_frame && frame_count <= trace_frame + 2
                         : l == 0)) { // DC_DUMP_TEXTURE=<addr>: raw linear bytes, each
                                      // layer near DC_TRACE_FRAME, else layer 0
          char path[512];
          std::snprintf(path, sizeof(path), "%s/tex_%llx_%ux%u_bpe%u_f%llu_l%u.raw",
                        dump_frames, addr, width_el, height_el,
                        fi.bytes_per_block, frame_count, l);
          DC_INFO("gpu: dump tex %llx layer %u pitch_el %u slice %llu tile %u", addr, l,
                  m.pitch_el, static_cast<unsigned long long>(m.slice_size), m.tile_index);
          std::ofstream(path, std::ios::binary)
              .write(reinterpret_cast<const char *>(out), image);
        }
        if (!cpu_upload) {
          [UploadBlit() copyFromBuffer:sbuf
                          sourceOffset:soff
                     sourceBytesPerRow:row
                   sourceBytesPerImage:image
                            sourceSize:MTLSizeMake(w, h, 1)
                             toTexture:e.tex
                      destinationSlice:tt.dim == TexDim::Tex2DArray ? l : 0
                      destinationLevel:lv
                     destinationOrigin:MTLOriginMake(0, 0, tt.dim == TexDim::Tex3D ? l : 0)];
          continue;
        }
        const MTLRegion region = tt.dim == TexDim::Tex3D
                                     ? MTLRegionMake3D(0, 0, l, w, h, 1)
                                     : MTLRegionMake2D(0, 0, w, h);
        [e.tex replaceRegion:region
                 mipmapLevel:lv
                       slice:tt.dim == TexDim::Tex2DArray ? l : 0
                   withBytes:linear.data()
                 bytesPerRow:row
               bytesPerImage:image];
      }
    }
    (void)thick;
    }
    e.chunks.resize((size + kChunk - 1) / kChunk);
    for (u32 c = 0; c < e.chunks.size(); ++c) {
      e.chunks[c] = ChunkHash(src, size, c);
    }
    e.frame = frame_count;
    e.addr = addr;
    e.size = size;
    if (size >= kTextureHotBytes) {
      e.hot = true;
    }
    TexturePressure();
    return view();
  }

  id<MTLSamplerState> GetSampler(const u32 s[4]) NS_RETURNS_RETAINED {
    const u64 key = Fnv(s, 16);
    auto it = samplers.find(key);
    if (it != samplers.end()) {
      return it->second;
    }
    MTLSamplerDescriptor *d = [MTLSamplerDescriptor new];
    d.sAddressMode = AddressMode(s[0]);
    d.tAddressMode = AddressMode(s[0] >> 3);
    d.rAddressMode = AddressMode(s[0] >> 6);
    const u32 aniso = (s[0] >> 9) & 7;
    const u32 mag = (s[2] >> 20) & 3, min = (s[2] >> 22) & 3,
              mip = (s[2] >> 26) & 3;
    d.magFilter = (mag & 1) ? MTLSamplerMinMagFilterLinear
                            : MTLSamplerMinMagFilterNearest;
    d.minFilter = (min & 1) ? MTLSamplerMinMagFilterLinear
                            : MTLSamplerMinMagFilterNearest;
    d.mipFilter = mip == 0   ? MTLSamplerMipFilterNotMipmapped
                  : mip == 1 ? MTLSamplerMipFilterNearest
                             : MTLSamplerMipFilterLinear;
    if (min >= 2 || mag >= 2) {
      d.maxAnisotropy = std::min<u32>(16, 1u << aniso);
    }
    d.lodMinClamp = static_cast<float>(s[1] & 0xFFF) / 256.0f;
    d.lodMaxClamp = static_cast<float>((s[1] >> 12) & 0xFFF) / 256.0f;
    const u32 border = (s[3] >> 30) & 3;
    d.borderColor = border == 0   ? MTLSamplerBorderColorTransparentBlack
                    : border == 1 ? MTLSamplerBorderColorOpaqueBlack
                                  : MTLSamplerBorderColorOpaqueWhite;
    // FORCE_UNNORMALIZED: coordinates in texels (GT7's glyphs). Metal allows
    // this only without mips or anisotropy, clamped, with one filter. Shaders
    // now normalize such coordinates themselves by the sampled level's size
    // (TextureType::unnorm), which also reads mips at an explicit LOD (GT7's
    // hull shaders sample their Hi-Z pyramid so, ryGT7 FIX-228); the sampler
    // keeps the edge clamps. DC_NO_UNNORM_SHADER=1: the old Metal sampler.
    static const bool unnorm_shader = std::getenv("DC_NO_UNNORM_SHADER") == nullptr;
    if (unnorm_shader && ((s[0] >> 15) & 1)) {
      const auto clamp_mode = [](u32 clamp) {
        return (clamp & 7) >= 4 ? MTLSamplerAddressModeClampToZero
                                : MTLSamplerAddressModeClampToEdge;
      };
      d.sAddressMode = clamp_mode(s[0]);
      d.tAddressMode = clamp_mode(s[0] >> 3);
      d.rAddressMode = clamp_mode(s[0] >> 6);
    } else if ((s[0] >> 15) & 1) {
      d.normalizedCoordinates = NO;
      // Borders stay transparent: smearing the edge texel fills whole quads.
      const auto pixel_clamp = [](u32 clamp) {
        return (clamp & 7) >= 4 ? MTLSamplerAddressModeClampToZero
                                : MTLSamplerAddressModeClampToEdge;
      };
      d.sAddressMode = pixel_clamp(s[0]);
      d.tAddressMode = pixel_clamp(s[0] >> 3);
      d.rAddressMode = pixel_clamp(s[0] >> 6);
      d.minFilter = d.magFilter;
      d.mipFilter = MTLSamplerMipFilterNotMipmapped;
      d.maxAnisotropy = 1;
      d.lodMinClamp = 0.0f;
      d.lodMaxClamp = 0.0f;
    }
    id<MTLSamplerState> st = [device newSamplerStateWithDescriptor:d];
    samplers[key] = st ? st : default_sampler;
    return samplers[key];
  }

  // ================= render targets =================

  // The variant at `addr` written last; with a size, the last one of that size
  // if there is one (aliased memory holds surfaces of several sizes, and a T#
  // means the one matching its own extent, like shadPS4's image lookup). With
  // texel_bytes, variants of another texel width are ignored.
  static RenderTarget *Latest(TargetMap &map, u64 addr,
                              std::optional<TexDim> dim = std::nullopt,
                              u32 width = 0, u32 height = 0, u32 texel_bytes = 0) {
    auto it = map.find(addr);
    if (it == map.end()) {
      return nullptr;
    }
    RenderTarget *best = nullptr;
    RenderTarget *sized = nullptr;
    for (auto &t : it->second) {
      if (dim && t->dim != *dim) {
        continue;
      }
      if (texel_bytes && PixelBytes(t->tex.pixelFormat) != texel_bytes) {
        continue;
      }
      if (!best || t->written > best->written) {
        best = t.get();
      }
      if (width && t->width == width && t->height == height &&
          (!sized || t->written > sized->written)) {
        sized = t.get();
      }
    }
    return sized ? sized : best;
  }

  id<MTLTexture> NewTargetTexture(MTLPixelFormat fmt, u32 width, u32 height,
                                  u32 levels, bool depth,
                                  TexDim dim = TexDim::Tex2D,
                                  u32 depth_or_layers = 1) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor new];
    d.pixelFormat = fmt;
    d.width = width;
    d.height = height;
    d.mipmapLevelCount = levels;
    d.storageMode = MTLStorageModePrivate;
    if (dim == TexDim::Tex3D) {
      d.textureType = MTLTextureType3D;
      d.depth = std::max(depth_or_layers, 1u);
      d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
                MTLTextureUsagePixelFormatView;
    } else if (dim == TexDim::Tex2DArray) {
      d.textureType = MTLTextureType2DArray;
      d.arrayLength = LayerLimit(std::max(depth_or_layers, 1u));
      d.usage = depth
                    ? MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead |
                          MTLTextureUsagePixelFormatView
                    : MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead |
                          MTLTextureUsageShaderWrite |
                          MTLTextureUsagePixelFormatView;
    } else {
      d.textureType = MTLTextureType2D;
      d.usage = depth
                    ? MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead |
                          MTLTextureUsagePixelFormatView
                    : MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead |
                          MTLTextureUsageShaderWrite |
                          MTLTextureUsagePixelFormatView;
    }
    // Integer R32 images may be targets of IMAGE_ATOMIC_ADD.
    if (fmt == MTLPixelFormatR32Uint || fmt == MTLPixelFormatR32Sint) {
      d.usage |= MTLTextureUsageShaderAtomic;
    }
    return [device newTextureWithDescriptor:d];
  }

  // The same memory rendered as RGBA8 and as sRGB RGBA8 is one surface (the
  // cars' liveries are composed as sRGB, their mips built and compressed as
  // linear). Views between the two need no PixelFormatView usage.
  static bool SrgbTwins(MTLPixelFormat a, MTLPixelFormat b) {
    auto linear = [](MTLPixelFormat f) {
      switch (f) {
      case MTLPixelFormatRGBA8Unorm_sRGB: return MTLPixelFormatRGBA8Unorm;
      case MTLPixelFormatBGRA8Unorm_sRGB: return MTLPixelFormatBGRA8Unorm;
      case MTLPixelFormatRG8Unorm_sRGB: return MTLPixelFormatRG8Unorm;
      case MTLPixelFormatR8Unorm_sRGB: return MTLPixelFormatR8Unorm;
      default: return f;
      }
    };
    return a != b && linear(a) == linear(b);
  }

  static bool Holds(const std::vector<std::unique_ptr<RenderTarget>> &list,
                    const RenderTarget *t) {
    return std::any_of(list.begin(), list.end(),
                       [t](const auto &u) { return u.get() == t; });
  }

  RenderTarget &GetTarget(TargetMap &map, u64 addr, MTLPixelFormat fmt,
                          u32 width, u32 height, bool depth, u32 levels = 1,
                          TexDim dim = TexDim::Tex2D, u32 depth_or_layers = 1) {
    auto &list = map[addr];
    const u32 max_levels =
        1 + static_cast<u32>(std::log2(std::max(width, height)));
    levels = std::clamp<u32>(levels, 1, max_levels);
    const bool array = dim == TexDim::Tex2DArray;
    for (auto &t : list) {
      // Arrays are one surface whatever slice range a descriptor covers: they
      // grow.
      if (t->width != width || t->height != height ||
          t->tex.pixelFormat != fmt || t->dim != dim ||
          (!array && t->depth_or_layers != depth_or_layers)) {
        continue;
      }
      const u32 want_layers =
          array ? std::max(t->depth_or_layers, depth_or_layers)
                : t->depth_or_layers;
      if (t->view_of && !Holds(list, t->view_of)) {
        t->view_of = nullptr; // its twin was retired: the view is all that is left
      }
      if (t->view_of && (t->levels < levels || t->depth_or_layers < want_layers)) {
        // A view grows with the texture it views.
        GetTarget(map, addr, t->view_of->tex.pixelFormat, width, height, depth,
                  levels, dim, want_layers);
        return *t;
      }
      if (t->levels < levels ||
          t->depth_or_layers < want_layers) { // grow, keeping what is there
        const u32 new_levels = std::max(t->levels, levels);
        id<MTLTexture> tex = NewTargetTexture(fmt, width, height, new_levels,
                                              depth, dim, want_layers);
        EndPass();
        EnsureCommandBuffer();
        id<MTLBlitCommandEncoder> b = [cmd blitCommandEncoder];
        const u32 slices = array ? t->depth_or_layers : 1;
        for (u32 l = 0; l < t->levels; ++l) {
          [b copyFromTexture:t->tex
                   sourceSlice:0
                   sourceLevel:l
                     toTexture:tex
              destinationSlice:0
              destinationLevel:l
                    sliceCount:slices
                    levelCount:1];
        }
        [b endEncoding];
        t->tex = tex;
        t->levels = new_levels;
        t->depth_or_layers = want_layers;
        t->views.clear();
        for (auto &v : list) { // its twins view the new texture
          if (v->view_of == t.get()) {
            v->tex = [tex newTextureViewWithPixelFormat:v->tex.pixelFormat];
            v->levels = new_levels;
            v->depth_or_layers = want_layers;
            v->views.clear();
          }
        }
        ++texture_epoch; // the old texture object is gone
      }
      return *t;
    }
    RenderTarget *twin = nullptr;
    for (auto &t : list) {
      if (!depth && SrgbTwins(t->tex.pixelFormat, fmt) && t->width == width &&
          t->height == height && t->dim == dim &&
          (array || t->depth_or_layers == depth_or_layers)) {
        twin = t->view_of && Holds(list, t->view_of) ? t->view_of : t.get();
        break;
      }
    }
    if (twin) {
      twin = &GetTarget(map, addr, twin->tex.pixelFormat, width, height, false,
                        levels, dim, depth_or_layers);
    }
    auto rt = std::make_unique<RenderTarget>();
    if (twin) {
      rt->tex = [twin->tex newTextureViewWithPixelFormat:fmt];
      rt->view_of = twin;
      rt->initialized = twin->initialized;
      rt->written = twin->written;
      levels = twin->levels;
      depth_or_layers = twin->depth_or_layers;
    } else {
      rt->tex = NewTargetTexture(fmt, width, height, levels, depth, dim,
                                 depth_or_layers);
    }
    rt->addr = addr;
    rt->width = width;
    rt->height = height;
    rt->levels = levels;
    rt->dim = dim;
    rt->depth_or_layers = depth_or_layers;
    DC_INFO("gpu: %s target %#llx %ux%u (dim=%d d/l=%u) fmt %lu (shader %#llx)%s",
            depth ? "depth" : "render", addr, width, height, (int)dim,
            depth_or_layers, (unsigned long)fmt,
            (unsigned long long)resolving_shader, twin ? ", a view of its sRGB twin" : "");
    list.push_back(std::move(rt));
    ++texture_epoch; // a new surface: descriptors may now resolve elsewhere
    return *list.back();
  }

  // ---- Frame capture (src/gpu/liverpool/capture.cpp) ----
  // Render targets live only on the host GPU, so a capture saves their
  // pixels (state a frame inherits: history, exposure, Hi-Z, cached shadows)
  // and a replay puts them back before each loop.
  struct TargetPlane {
    u32 bpp;
    MTLBlitOption option;
  };
  static std::vector<TargetPlane> TargetPlanes(MTLPixelFormat f) {
    switch (f) {
    case MTLPixelFormatDepth32Float: return {{4, MTLBlitOptionNone}};
    case MTLPixelFormatDepth16Unorm: return {{2, MTLBlitOptionNone}};
    case MTLPixelFormatStencil8: return {{1, MTLBlitOptionNone}};
    case MTLPixelFormatDepth32Float_Stencil8:
      return {{4, MTLBlitOptionDepthFromDepthStencil}, {1, MTLBlitOptionStencilFromDepthStencil}};
    default: break;
    }
    if (const u32 b = PixelBytes(f)) {
      return {{b, MTLBlitOptionNone}};
    }
    return {};
  }
  static u32 TargetSlices(id<MTLTexture> t) {
    switch (t.textureType) {
    case MTLTextureType2DArray: return u32(t.arrayLength);
    case MTLTextureTypeCube: return 6;
    case MTLTextureTypeCubeArray: return 6 * u32(t.arrayLength);
    default: return 1;
    }
  }
  // Calls fn(level, slice, plane, w, h, d, bytes) in the file's order.
  template <typename Fn> static void ForEachTargetImage(id<MTLTexture> t, Fn &&fn) {
    const auto planes = TargetPlanes(t.pixelFormat);
    for (u32 l = 0; l < t.mipmapLevelCount; ++l) {
      const u32 w = std::max<u32>(1, u32(t.width) >> l), h = std::max<u32>(1, u32(t.height) >> l);
      const u32 d = t.textureType == MTLTextureType3D ? std::max<u32>(1, u32(t.depth) >> l) : 1;
      for (u32 sl = 0; sl < TargetSlices(t); ++sl) {
        for (const TargetPlane &p : planes) {
          fn(l, sl, p, w, h, d, u64(w) * p.bpp * h * d);
        }
      }
    }
  }
  static constexpr u32 kTargetsMagic = 0x54524731; // "TRG1"

  void SaveTargets(const char *path) {
    EndPass();
    Commit(true);
    FILE *f = std::fopen(path, "wb");
    if (!f) {
      DC_ERROR("capture: cannot write %s", path);
      return;
    }
    std::fwrite(&kTargetsMagic, 4, 1, f);
    u32 saved = 0, skipped = 0;
    u64 bytes_total = 0;
    for (int depth = 0; depth < 2; ++depth) {
      for (auto &[addr, list] : depth ? depth_targets : color_targets) {
        for (auto &rt : list) {
          id<MTLTexture> t = rt->tex;
          if (rt->view_of || !t || t.sampleCount > 1 || TargetPlanes(t.pixelFormat).empty()) {
            ++skipped;
            continue;
          }
          u64 total = 0;
          ForEachTargetImage(t, [&](u32, u32, const TargetPlane &, u32, u32, u32, u64 b) { total += b; });
          id<MTLBuffer> buf = [device newBufferWithLength:std::max<u64>(total, 16)
                                                  options:MTLResourceStorageModeShared];
          if (!buf) {
            ++skipped;
            continue;
          }
          EnsureCommandBuffer();
          id<MTLBlitCommandEncoder> b = [cmd blitCommandEncoder];
          u64 off = 0;
          ForEachTargetImage(t, [&](u32 l, u32 sl, const TargetPlane &p, u32 w, u32 h, u32 d, u64 n) {
            [b copyFromTexture:t
                         sourceSlice:sl
                         sourceLevel:l
                        sourceOrigin:MTLOriginMake(0, 0, 0)
                          sourceSize:MTLSizeMake(w, h, d)
                            toBuffer:buf
                   destinationOffset:off
              destinationBytesPerRow:u64(w) * p.bpp
            destinationBytesPerImage:u64(w) * p.bpp * h
                             options:p.option];
            off += n;
          });
          [b endEncoding];
          Commit(true);
          const u32 hdr[12] = {u32(depth), u32(t.pixelFormat), rt->width, rt->height, rt->levels,
                               u32(rt->dim), rt->depth_or_layers, rt->initialized ? 1u : 0u,
                               rt->bind_slice, rt->bind_level, rt->comp_swap, rt->three_channels ? 1u : 0u};
          std::fwrite(&addr, 8, 1, f);
          std::fwrite(hdr, sizeof(hdr), 1, f);
          std::fwrite(&total, 8, 1, f);
          std::fwrite(buf.contents, 1, total, f);
          bytes_total += total;
          ++saved;
        }
      }
    }
    std::fclose(f);
    DC_INFO("capture: %u render targets saved (%llu MB), %u skipped (views, MSAA, other formats)", saved,
            (unsigned long long)(bytes_total >> 20), skipped);
  }

  void LoadTargets(const char *path) {
    FILE *f = std::fopen(path, "rb");
    if (!f) {
      return;
    }
    u32 magic = 0;
    if (std::fread(&magic, 4, 1, f) != 1 || magic != kTargetsMagic) {
      std::fclose(f);
      return;
    }
    EndPass();
    u32 loaded = 0;
    restored_lo = write_seq;
    restored_logged.clear();
    for (;;) {
      u64 addr = 0, total = 0;
      u32 hdr[12];
      if (std::fread(&addr, 8, 1, f) != 1 || std::fread(hdr, sizeof(hdr), 1, f) != 1 ||
          std::fread(&total, 8, 1, f) != 1) {
        break;
      }
      id<MTLBuffer> buf = [device newBufferWithLength:std::max<u64>(total, 16)
                                              options:MTLResourceStorageModeShared];
      if (!buf || std::fread(buf.contents, 1, total, f) != total) {
        break;
      }
      const bool depth = hdr[0] != 0;
      // DC_REPLAY_TARGETS_ONLY=<hex>,...: restore only these addresses (state
      // a frame inherits, to tell it from what the replay must draw itself).
      static const std::vector<u64> only = [] {
        std::vector<u64> v;
        for (const char *q = std::getenv("DC_REPLAY_TARGETS_ONLY"); q && *q;) {
          char *e = nullptr;
          v.push_back(std::strtoull(q, &e, 16));
          q = (e && *e == ',') ? e + 1 : nullptr;
        }
        return v;
      }();
      if (!only.empty() && std::find(only.begin(), only.end(), addr) == only.end()) {
        continue;
      }
      RenderTarget &rt = GetTarget(depth ? depth_targets : color_targets, addr, MTLPixelFormat(hdr[1]), hdr[2],
                                   hdr[3], depth, hdr[4], TexDim(hdr[5]), hdr[6]);
      id<MTLTexture> t = rt.tex;
      u64 need = 0;
      ForEachTargetImage(t, [&](u32, u32, const TargetPlane &, u32, u32, u32, u64 b) { need += b; });
      if (need != total) {
        DC_WARN("capture: target %#llx came back with another layout; left as is", (unsigned long long)addr);
        continue;
      }
      EnsureCommandBuffer();
      id<MTLBlitCommandEncoder> b = [cmd blitCommandEncoder];
      u64 off = 0;
      ForEachTargetImage(t, [&](u32 l, u32 sl, const TargetPlane &p, u32 w, u32 h, u32 d, u64 n) {
        [b copyFromBuffer:buf
                   sourceOffset:off
              sourceBytesPerRow:u64(w) * p.bpp
            sourceBytesPerImage:u64(w) * p.bpp * h
                     sourceSize:MTLSizeMake(w, h, d)
                      toTexture:t
               destinationSlice:sl
               destinationLevel:l
              destinationOrigin:MTLOriginMake(0, 0, 0)
                        options:p.option];
        off += n;
      });
      [b endEncoding];
      rt.initialized = hdr[7] != 0;
      rt.bind_slice = hdr[8];
      rt.bind_level = hdr[9];
      rt.comp_swap = hdr[10];
      rt.three_channels = hdr[11] != 0;
      rt.clear_pending = false;
      rt.written = ++write_seq;
      ++loaded;
    }
    std::fclose(f);
    Commit(true);
    texture_cache.clear();
    ++texture_epoch;
    restored_hi = write_seq;
    DC_INFO("capture: %u render targets restored", loaded);
  }

  // `rt` holds BC blocks, one per texel (a shader compressed a texture into
  // it): a `fi.format` texture of width x height with those blocks, copied
  // again whenever `rt` is written. nil when the shapes do not match.
  struct BcCopy {
    id<MTLTexture> tex = nil;
    u64 written = ~0ull;
  };
  std::unordered_map<const RenderTarget *, BcCopy> bc_copies;

  id<MTLTexture> BlocksAsCompressed(RenderTarget &rt, const TextureFormatInfo &fi,
                                    u32 width, u32 height, u32 levels) {
    if (!rt.tex || rt.tex.textureType != MTLTextureType2D ||
        PixelBytes(rt.tex.pixelFormat) != fi.bytes_per_block ||
        rt.width != (width + 3) / 4 || rt.height != (height + 3) / 4) {
      return nil;
    }
    levels = std::clamp<u32>(std::min<u32>(levels, rt.levels), 1,
                             1 + static_cast<u32>(std::log2(std::max(width, height))));
    BcCopy &c = bc_copies[&rt];
    if (c.tex && c.written == rt.written && c.tex.mipmapLevelCount == levels) {
      return c.tex;
    }
    if (!c.tex || c.tex.mipmapLevelCount != levels || c.tex.pixelFormat != fi.format) {
      MTLTextureDescriptor *d =
          [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fi.format
                                                             width:width
                                                            height:height
                                                         mipmapped:levels > 1];
      d.mipmapLevelCount = levels;
      d.storageMode = MTLStorageModePrivate;
      d.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
      c.tex = [device newTextureWithDescriptor:d];
      if (!c.tex) {
        return nil;
      }
    }
    c.written = rt.written;
    u64 total = 0;
    for (u32 l = 0; l < levels; ++l) {
      total += u64(std::max<u32>(1, rt.width >> l)) * std::max<u32>(1, rt.height >> l) *
               fi.bytes_per_block;
    }
    id<MTLBuffer> buf = [device newBufferWithLength:total options:MTLResourceStorageModePrivate];
    EndPass();
    EnsureCommandBuffer();
    id<MTLBlitCommandEncoder> b = [cmd blitCommandEncoder];
    u64 off = 0;
    for (u32 l = 0; l < levels; ++l) {
      const u32 bw = std::max<u32>(1, rt.width >> l), bh = std::max<u32>(1, rt.height >> l);
      const u64 row = u64(bw) * fi.bytes_per_block;
      [b copyFromTexture:rt.tex
                   sourceSlice:0
                   sourceLevel:l
                  sourceOrigin:MTLOriginMake(0, 0, 0)
                    sourceSize:MTLSizeMake(bw, bh, 1)
                      toBuffer:buf
             destinationOffset:off
        destinationBytesPerRow:row
      destinationBytesPerImage:row * bh];
      [b copyFromBuffer:buf
                 sourceOffset:off
            sourceBytesPerRow:row
          sourceBytesPerImage:row * bh
                   sourceSize:MTLSizeMake(std::max<u32>(1, width >> l),
                                          std::max<u32>(1, height >> l), 1)
                    toTexture:c.tex
             destinationSlice:0
             destinationLevel:l
            destinationOrigin:MTLOriginMake(0, 0, 0)];
      off += row * bh;
    }
    [b endEncoding];
    return c.tex;
  }

  // Levels [base, base + count) and, for arrays, slices [sbase, sbase + scount)
  // of `rt` (the texture itself when that is all of it).
  static constexpr u32 kIdentitySwizzle = 0xFAC; // T# DST_SEL x=R y=G z=B w=A
  // Vertex buffers of Metal-fetched attributes: buffer(16 + k).
  static constexpr u32 kVertexBufferBase = 16;
  static constexpr u32 kMaxVertexBuffers = 14;

  // A T# shorter than the target at its address (GT7's blur keeps its passes
  // in two halves of one surface and reads each by its own extent). Metal views
  // cannot crop and normalized coordinates would span both halves: the T# gets
  // a copy of its rows, taken again after each new pass on the target.
  RenderTarget *CropOf(RenderTarget &rt, const TextureType &tt, u32 width, u32 height,
                       u32 base_level, u32 last_level) {
    if (tt.dim != TexDim::Tex2D || rt.dim != TexDim::Tex2D || base_level || last_level ||
        rt.levels != 1 || rt.width != width || rt.height <= height ||
        rt.tex.sampleCount != 1) {
      return nullptr;
    }
    auto &crop = target_crops[Hash(rt.addr, u64(height) << 32 | width)];
    if (!crop || crop->tex.pixelFormat != rt.tex.pixelFormat) {
      MTLTextureDescriptor *d = [MTLTextureDescriptor new];
      d.textureType = MTLTextureType2D;
      d.pixelFormat = rt.tex.pixelFormat;
      d.width = width;
      d.height = height;
      d.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
      d.storageMode = MTLStorageModePrivate;
      crop = std::make_unique<RenderTarget>();
      crop->tex = [device newTextureWithDescriptor:d];
      crop->addr = rt.addr;
      crop->width = width;
      crop->height = height;
      if (!crop->tex) {
        crop.reset();
        return nullptr;
      }
    }
    rt.has_crops = true;
    EndPass();
    EnsureCommandBuffer();
    id<MTLBlitCommandEncoder> b = [cmd blitCommandEncoder];
    [b copyFromTexture:rt.tex
           sourceSlice:0
           sourceLevel:0
          sourceOrigin:MTLOriginMake(0, 0, 0)
            sourceSize:MTLSizeMake(width, height, 1)
             toTexture:crop->tex
      destinationSlice:0
      destinationLevel:0
     destinationOrigin:MTLOriginMake(0, 0, 0)];
    [b endEncoding];
    return crop.get();
  }

  id<MTLTexture> TargetView(RenderTarget &rt, u32 base, u32 count,
                            MTLPixelFormat fmt = MTLPixelFormatInvalid,
                            u32 sbase = 0, u32 scount = ~0u,
                            u32 swz = kIdentitySwizzle, bool as2d = false) {
    if (fmt == MTLPixelFormatInvalid) {
      fmt = rt.tex.pixelFormat;
    }
    // One slice of an array seen through a 2D descriptor (a face or mip
    // level of a cube written or read as a plain image): a 2D view.
    as2d = as2d && rt.tex.textureType == MTLTextureType2DArray;
    if (as2d) {
      scount = 1;
    }
    base = std::min(base, rt.levels - 1);
    count = std::clamp<u32>(count, 1, rt.levels - base);
    const u32 total = rt.tex.textureType == MTLTextureType2DArray
                          ? static_cast<u32>(rt.tex.arrayLength)
                          : 1;
    sbase = std::min(sbase, total - 1);
    scount = std::clamp<u32>(scount, 1, total - sbase);
    swz &= 0xFFF;
    if (!as2d && base == 0 && count == rt.levels && sbase == 0 && scount == total &&
        fmt == rt.tex.pixelFormat && swz == kIdentitySwizzle) {
      return rt.tex;
    }
    u64 key = Hash(base | (u64(count) << 8) | (u64(fmt) << 16) |
                   (u64(sbase) << 32) | (u64(scount) << 46));
    key = Hash(swz | (u64(as2d) << 32), key);
    auto it = rt.views.find(key);
    if (it != rt.views.end()) {
      return it->second;
    }
    id<MTLTexture> v = [rt.tex
        newTextureViewWithPixelFormat:fmt
                          textureType:as2d ? MTLTextureType2D : rt.tex.textureType
                               levels:NSMakeRange(base, count)
                               slices:NSMakeRange(sbase, scount)
                              swizzle:MTLTextureSwizzleChannelsMake(
                                          Swizzle(swz), Swizzle(swz >> 3),
                                          Swizzle(swz >> 6),
                                          Swizzle(swz >> 9))];
    rt.views[key] = v;
    return v;
  }

  RenderTarget &GetColorTarget(u64 addr, MTLPixelFormat fmt, u32 width,
                               u32 height, u32 levels = 1,
                               TexDim dim = TexDim::Tex2D,
                               u32 depth_or_layers = 1) {
    return GetTarget(color_targets, addr, fmt, width, height, false, levels,
                     dim, depth_or_layers);
  }

  RenderTarget &GetDepthTarget(u64 addr, MTLPixelFormat fmt, u32 width,
                               u32 height) {
    return GetTarget(depth_targets, addr, fmt, width, height, true);
  }

  void NoteMeta(u64 meta, u64 target, bool depth, u32 width = 0, u32 height = 0) {
    if (meta) {
      metas[meta] = {target, depth, width, height};
    }
  }

  // A fill or a clear shader rewrote metadata in [addr, addr + bytes):
  // the surfaces it describes now read as their clear value.
  bool expanding_htile = false;

  // HTILE at guest [base, base + bytes) from the depth surface `m` describes
  // (see htile_build). DC_HTILE_FULL_RANGE keeps the old full-range fill.
  bool BuildHtile(u64 base, u64 bytes, const Meta &m) {
    static const bool full = std::getenv("DC_HTILE_FULL_RANGE") != nullptr;
    RenderTarget *z = full || !htile_pipeline
                          ? nullptr
                          : Latest(depth_targets, m.target, std::nullopt, m.width, m.height);
    id<MTLBuffer> buf = nil;
    u64 off = 0;
    if (!z || !z->tex || z->width != m.width || z->height != m.height ||
        z->tex.textureType != MTLTextureType2D ||
        z->tex.sampleCount != 1 || !Locate(base, bytes, &buf, &off)) {
      return false;
    }
    const float clear = static_cast<float>(z->clear_depth);
    u32 p[4] = {(z->width + 7) / 8, (z->height + 7) / 8, static_cast<u32>(bytes / 4), ~0u};
    if (!z->initialized || z->clear_pending) {
      std::memcpy(&p[3], &clear, 4);
    }
    EndPass();
    EnsureCommandBuffer();
    id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
    [e setComputePipelineState:htile_pipeline];
    [e setTexture:z->tex atIndex:0];
    [e setBuffer:buf offset:off atIndex:0];
    [e setBytes:p length:sizeof(p) atIndex:1];
    [e dispatchThreads:MTLSizeMake(p[0], p[1], 1)
        threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    [e endEncoding];
    return true;
  }

  bool MarkMetaCleared(u64 addr, u64 bytes) {
    bool any = false;
    for (const auto &[meta, m] : metas) {
      if (meta < addr || meta >= addr + bytes) {
        continue;
      }
      auto it = (m.depth ? depth_targets : color_targets).find(m.target);
      if (it != (m.depth ? depth_targets : color_targets).end()) {
        for (auto &t : it->second) {
          t->clear_pending = true;
        }
      }
      any = true;
      if (trace) {
        DC_INFO("gpu:   fast clear of %s %#llx (meta %#llx)",
                m.depth ? "depth" : "color", m.target, meta);
      }
    }
    return any;
  }

  void OnContextRegs(const u32 *body, u32 n) {
    if (n < 2) {
      return;
    }
    const u32 reg = kContextRegBase + body[0];
    const u32 count = n - 1;
    const u32 base0 = Reg::CB_COLOR0_BASE, stride = Reg::kColorTargetStride;
    if (reg >= base0 && reg < base0 + 8 * stride) {
      const u32 cb = (reg - base0) / stride, field = (reg - base0) % stride;
      if (field == Reg::CB_BASE) {
        cb_extent[cb] = (count == 0x0E || count == 0x0D || count == 0x0B) &&
                                body[n] == kNopSizeHint
                            ? body[n + 1]
                            : 0;
      } else if (field == Reg::CB_CMASK && count == 4 &&
                 body[n] == kNopSizeHint) {
        cb_extent[cb] = body[n + 1];
      }
    } else if (reg == Reg::DB_Z_INFO) {
      db_extent = count == 8 && body[21] == kNopSizeHint ? body[22] : 0;
    }
  }

  static bool HasStencil(id<MTLTexture> t) {
    return t.pixelFormat == MTLPixelFormatDepth32Float_Stencil8;
  }

  void BeginPass(RenderTarget *colors[8], RenderTarget *depth, bool clear_depth,
                 float depth_value, bool clear_stencil = false,
                 u32 stencil_value = 0) {
    EndPass();
    if (work_since_commit >= kCommitEvery) {
      Commit(false); // keep the host GPU busy while the rest is recorded
    }
    EnsureCommandBuffer();
    MTLRenderPassDescriptor *rp =
        [MTLRenderPassDescriptor renderPassDescriptor];
    PassState ps;
    u32 w = ~0u, h = ~0u;
    const u64 seq = ++write_seq;
    for (u32 k = 0; k < 8; ++k) {
      if (!colors[k]) {
        continue;
      }
      RenderTarget &rt = *colors[k];
      if (rt.has_crops) {
        ++texture_epoch; // its crops go stale: the next bind copies them again
      }
      rp.colorAttachments[k].texture = rt.tex;
      rp.colorAttachments[k].slice = draw_layers ? 0 : rt.bind_slice;
      rp.colorAttachments[k].level = rt.bind_level;
      rp.colorAttachments[k].loadAction = rt.initialized && !rt.clear_pending
                                              ? MTLLoadActionLoad
                                              : MTLLoadActionClear;
      rp.colorAttachments[k].clearColor =
          rt.clear_pending ? rt.clear_color
          : debug_clear    ? MTLClearColorMake(0.1, 0.1, 0.3, 1)
                           : MTLClearColorMake(0, 0, 0, 0);
      rp.colorAttachments[k].storeAction = MTLStoreActionStore;
      rt.initialized = true;
      rt.clear_pending = false;
      rt.written = seq;
      NoteTargetWritten(rt);
      ps.color[k] = reinterpret_cast<u64>(&rt);
      ps.slice[k] = rt.bind_slice | (rt.bind_level << 24);
      w = std::min(w, std::max(rt.width >> rt.bind_level, 1u));
      h = std::min(h, std::max(rt.height >> rt.bind_level, 1u));
    }
    if (depth) {
      const bool clear =
          clear_depth || depth->clear_pending || !depth->initialized;
      const u32 depth_slice = draw_layers ? 0 : depth->bind_slice;
      rp.depthAttachment.texture = depth->tex;
      rp.depthAttachment.slice = depth_slice;
      rp.depthAttachment.loadAction =
          clear ? MTLLoadActionClear : MTLLoadActionLoad;
      rp.depthAttachment.clearDepth = clear_depth ? depth_value
                                      : depth->clear_pending
                                          ? depth->clear_depth
                                          : 1.0;
      rp.depthAttachment.storeAction = MTLStoreActionStore;
      if (HasStencil(depth->tex)) {
        const bool sclear =
            clear_stencil || depth->clear_pending || !depth->initialized;
        rp.stencilAttachment.texture = depth->tex;
        rp.stencilAttachment.slice = depth_slice;
        rp.stencilAttachment.loadAction =
            sclear ? MTLLoadActionClear : MTLLoadActionLoad;
        rp.stencilAttachment.clearStencil = clear_stencil ? stencil_value
                                            : depth->clear_pending
                                                ? depth->clear_stencil
                                                : 0;
        rp.stencilAttachment.storeAction = MTLStoreActionStore;
      }
      depth->initialized = true;
      depth->clear_pending = false;
      depth->written = seq;
      ps.depth = reinterpret_cast<u64>(depth);
      ps.depth_slice = depth->bind_slice;
      w = std::min(w, depth->width);
      h = std::min(h, depth->height);
    }
    if (draw_layers) {
      rp.renderTargetArrayLength = draw_layers;
      ps.layers = draw_layers;
    }
    pass_width = w;
    pass_height = h;
    rp.renderTargetWidth = w;
    rp.renderTargetHeight = h;
    if (pass_timing) {
      const u32 sample = TsBegin(0, w, h, false);
      MTLRenderPassSampleBufferAttachmentDescriptor *a =
          rp.sampleBufferAttachments[0];
      a.sampleBuffer = ts_buf[sample / kTsPerBuf];
      a.startOfVertexSampleIndex = sample % kTsPerBuf;
      a.endOfVertexSampleIndex = sample % kTsPerBuf + 1;
      a.startOfFragmentSampleIndex = sample % kTsPerBuf + 2;
      a.endOfFragmentSampleIndex = sample % kTsPerBuf + 3;
    }
    enc = [cmd renderCommandEncoderWithDescriptor:rp];
    bound = {};
    if (stats) {
      ++stat_passes;
      if (ps == last_pass && last_end_ra) {
        ++stat_split;
        ++split_by[last_end_ra];
      }
      last_end_ra = nullptr;
    }
    pass = ps;
    pass_draws = 0;
    pass_first_draw = draw_count;
    if ((dump_passes || PeekRt().addr) && trace) {
      for (u32 k = 0; k < 8; ++k) {
        pass_tex[k] = colors[k] ? colors[k]->tex : nil;
        pass_addr[k] = colors[k] ? colors[k]->addr : 0;
      }
      pass_tex[8] = depth ? depth->tex : nil;
      pass_addr[8] = depth ? depth->addr : 0;
    }
  }

  // ================= draws =================

  // DB_DEPTH_CONTROL + DB_STENCIL_CONTROL + DB_STENCILREFMASK(_BF), the way
  // shadPS4's UpdateDepthStencilState reads them.
  id<MTLDepthStencilState> GetDepthState(u32 db_control, bool depth_clear,
                                         bool stencil, u32 st_control,
                                         u32 ref_front, u32 ref_back,
                                         bool stencil_clear) {
    const bool backface = (db_control >> 7) & 1;
    if (!backface) {
      ref_back = ref_front;
    }
    u64 key = Hash(db_control & 0xFFF00FFFu);
    key = Hash(depth_clear, key);
    if (stencil) {
      key = Hash(st_control, key);
      key = Hash(ref_front & 0x00FFFF00u,
                 key); // masks; the reference is set per draw
      key = Hash(ref_back & 0x00FFFF00u, key);
      key = Hash(stencil_clear, key);
    }
    auto it = depth_states.find(key);
    if (it != depth_states.end()) {
      return it->second;
    }
    MTLDepthStencilDescriptor *d = [MTLDepthStencilDescriptor new];
    const bool z_enable = (db_control >> 1) & 1;
    d.depthCompareFunction =
        z_enable ? CompareFunc(db_control >> 4) : MTLCompareFunctionAlways;
    d.depthWriteEnabled = z_enable && ((db_control >> 2) & 1) && !depth_clear;
    if (stencil && (db_control & 1)) {
      auto face = [&](u32 fail, u32 zpass, u32 zfail, u32 func, u32 ref) {
        MTLStencilDescriptor *f = [MTLStencilDescriptor new];
        f.stencilFailureOperation = StencilOp(fail);
        f.depthStencilPassOperation = StencilOp(zpass);
        f.depthFailureOperation = StencilOp(zfail);
        f.stencilCompareFunction = CompareFunc(func);
        f.readMask = (ref >> 8) & 0xFF;
        f.writeMask = stencil_clear ? 0 : (ref >> 16) & 0xFF;
        return f;
      };
      d.frontFaceStencil = face(st_control, st_control >> 4, st_control >> 8,
                                db_control >> 8, ref_front);
      d.backFaceStencil =
          backface ? face(st_control >> 12, st_control >> 16, st_control >> 20,
                          db_control >> 20, ref_back)
                   : face(st_control, st_control >> 4, st_control >> 8,
                          db_control >> 8, ref_front);
    }
    id<MTLDepthStencilState> st = [device newDepthStencilStateWithDescriptor:d];
    depth_states[key] = st;
    return st;
  }

  // GCN REPLACE_OP writes STENCILOPVAL, REPLACE_TEST the test value (shadPS4: a
  // face with a ReplaceOp takes its reference from op_val).
  static u32 StencilRef(u32 st_ops, u32 refmask) {
    const bool op = (st_ops & 0xF) == 4 || ((st_ops >> 4) & 0xF) == 4 ||
                    ((st_ops >> 8) & 0xF) == 4;
    return op ? refmask >> 24 : refmask & 0xFF;
  }

  struct Resources {
    std::vector<id<MTLTexture>> tex;
    std::vector<id<MTLSamplerState>> smp;
  };
  // Whether a draw samples a color or depth attachment of the open pass. On
  // the host's tile-based GPU such a read sees memory as of the pass's last
  // store, not what the pass drew since: the fog draw (ps 0x9000f9d700)
  // samples the scene and depth it draws into. Ending the pass stores the
  // tiles first.
  static bool SamplesAttachment(const Resources &res, RenderTarget *const colors[8],
                                const RenderTarget *depth) {
    for (id<MTLTexture> t : res.tex) {
      id<MTLTexture> base = t.parentTexture ? t.parentTexture : t;
      if (depth && base == depth->tex) {
        return true;
      }
      for (u32 k = 0; k < 8; ++k) {
        if (colors[k] && base == colors[k]->tex) {
          return true;
        }
      }
    }
    return false;
  }
  // Draw-time scratch, kept across draws so a draw does not allocate.
  std::vector<TextureType> draw_vs_types, draw_ps_types;
  Resources draw_vs_res, draw_ps_res;
  // Evaluates each image instruction's descriptors against the user data.
  // `ssharps_`/`tsharps_` are members so a draw does not allocate them again.
  std::vector<std::array<u32, 4>> ssharps_;
  std::vector<std::array<u32, 8>> tsharps_;
  struct ResolveEntry {
    u64 epoch;
    std::vector<TextureType> types;
    Resources res;
  };
  std::unordered_map<u64, ResolveEntry> resolve_cache; // this frame's Resolve results
  u64 resolve_cache_frame = 0;
  void Resolve(const Program &p, const u32 *ud, std::vector<TextureType> &types,
               Resources &out) {
    types.clear();
    out.tex.clear();
    out.smp.clear();
    ssharps_.clear();
    tsharps_.clear();
    Shader::SgprReadCache rc;
    for (const auto &ss : p.samplers) {
      std::array<u32, 4> s{};
      for (u32 k = 0; k < 4; ++k) {
        s[k] = Shader::ResolveSgpr(ss[k], ud, &rc);
      }
      ssharps_.push_back(s);
    }
    bool cacheable = tex_cache && !tex_cache_check && !dump_texture && !trace;
    for (const auto &img : p.images) {
      std::array<u32, 8> t{};
      for (u32 k = 0; k < 8; ++k) {
        t[k] = Shader::ResolveSgpr(img.tsharp[k], ud, &rc);
      }
      tsharps_.push_back(t);
      cacheable &= !img.storage;
    }
    // Draws in a frame often repeat a program with the same descriptors:
    // the whole result is reused while no texture changed (texture_epoch).
    u64 key = 0;
    if (cacheable) {
      if (resolve_cache_frame != frame_count) {
        resolve_cache.clear();
        resolve_cache_frame = frame_count;
      }
      key = Hash(&p);
      key = Fnv(ssharps_.data(), ssharps_.size() * sizeof(ssharps_[0]), key);
      key = Fnv(tsharps_.data(), tsharps_.size() * sizeof(tsharps_[0]), key);
      if (cpu_profile) {
        ++cpu_resolve_calls;
      }
      if (auto it = resolve_cache.find(key);
          it != resolve_cache.end() && it->second.epoch == texture_epoch) {
        if (cpu_profile) {
          ++cpu_resolve_hits;
        }
        types = it->second.types;
        out.tex = it->second.res.tex;
        out.smp = it->second.res.smp;
        return;
      }
    }
    for (size_t i = 0; i < p.images.size(); ++i) {
      const auto &img = p.images[i];
      const u32 *t = tsharps_[i].data();
      TextureType tt;
      id<MTLTexture> tex = GetTexture(t, &tt, img.storage);
      if (trace && (t[0] == 0 || std::getenv("DC_TRACE_TSHARP"))) {
        DC_INFO("gpu:   T# %08x %08x %08x %08x %08x %08x %08x %08x src kind %u "
                "ud %u off %u",
                t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7],
                img.tsharp[0].kind, img.tsharp[0].ud, img.tsharp[0].offset);
      }
      if (trace) {
        DC_INFO("gpu:   tex%zu addr=%#llx dfmt=%u nfmt=%u %ux%u tile=%u "
                "type=%u swz=%#x%s",
                out.tex.size(), (u64(t[0]) | (u64(t[1] & 0xFF) << 32)) << 8,
                (t[1] >> 20) & 0x3F, (t[1] >> 26) & 0xF, (t[2] & 0x3FFF) + 1,
                ((t[2] >> 14) & 0x3FFF) + 1, (t[3] >> 20) & 0x1F, t[3] >> 28,
                t[3] & 0xFFF, img.storage ? " storage" : "");
        DC_INFO("gpu:     bound: type %lu %lux%lu layers %lu levels %lu fmt "
                "%lu base_array %u last_array %u",
                (unsigned long)tex.textureType, (unsigned long)tex.width,
                (unsigned long)tex.height, (unsigned long)tex.arrayLength,
                (unsigned long)tex.mipmapLevelCount,
                (unsigned long)tex.pixelFormat, t[5] & 0x1FFF,
                (t[5] >> 13) & 0x1FFF);
        DumpBound(tex, (u64(t[0]) | (u64(t[1] & 0xFF) << 32)) << 8);
      }
      // S# FORCE_UNNORMALIZED: the shader turns texel coordinates of the level
      // it samples into normalized ones (its sampler stays normalized, with mips).
      static const bool unnorm_shader = std::getenv("DC_NO_UNNORM_SHADER") == nullptr;
      if (unnorm_shader && img.has_sampler && img.sampler_index < ssharps_.size()) {
        tt.unnorm = (ssharps_[img.sampler_index][0] >> 15) & 1;
      }
      if (img.compare && img.has_sampler &&
          img.sampler_index < ssharps_.size()) {
        tt.compare = static_cast<u8>((ssharps_[img.sampler_index][0] >> 12) & 7);
        static const bool always =
            std::getenv("DC_CMP_ALWAYS") != nullptr; // debug experiment
        if (always) {
          tt.compare = 7;
        }
        static const bool flip =
            std::getenv("DC_CMP_FLIP") != nullptr; // debug experiment
        if (flip && tt.compare >= 1 && tt.compare <= 6 && tt.compare != 2 &&
            tt.compare != 5) {
          tt.compare = tt.compare == 1   ? 6
                       : tt.compare == 3 ? 4
                       : tt.compare == 4 ? 3
                                         : 1;
        }
        if (trace) {
          const auto &ss = ssharps_[img.sampler_index];
          DC_INFO("gpu:   compare sampler S# %08x %08x %08x %08x func %u",
                  ss[0], ss[1], ss[2], ss[3], tt.compare);
        }
      }
      types.push_back(tt);
      out.tex.push_back(tex);
    }
    for (const auto &s : ssharps_) {
      CpuTimer t(cpu_profile ? &cpu_sampler_s : nullptr);
      out.smp.push_back(GetSampler(s.data()));
    }
    if (cacheable) {
      resolve_cache[key] = ResolveEntry{texture_epoch, types, out};
    }
  }

  // The target whose mip level lives at `addr` with this size and format.
  RenderTarget *MipParent(u64 addr, MTLPixelFormat fmt, u32 width, u32 height) {
    auto it = mip_alias.find(addr);
    if (it == mip_alias.end() || it->second.fmt != fmt ||
        it->second.width != width || it->second.height != height) {
      return nullptr;
    }
    RenderTarget *p = Latest(color_targets, it->second.parent, std::nullopt,
                             width << it->second.level,
                             height << it->second.level);
    return p && p->levels > it->second.level && p->tex.pixelFormat == fmt
               ? p
               : nullptr;
  }

  // shadPS4 MipOf for a sampled render target: each level past the first of
  // the T#'s memory becomes an alias; a target already rendered there is
  // copied into that level and retired.
  void NoteMipLevels(RenderTarget &rt, const u32 t[8], u32 bytes_per_el) {
    const u32 tiling = (t[3] >> 20) & 0x1F;
    const u32 last_level = (t[3] >> 16) & 0xF;
    if (!last_level || rt.levels <= 1 || !Tiling::Supported(tiling)) {
      return;
    }
    // ImageInfo NumLayers: depth + 1 slices (cubes: full cubes), pow2 padded.
    const u32 ttype = t[3] >> 28;
    u32 layers = rt.dim == TexDim::Tex2DArray ? (t[4] & 0x1FFF) + 1 : 1;
    if (ttype == 11) {
      layers *= 6;
    }
    const bool pow2 = (t[3] >> 25) & 1;
    if (pow2) {
      layers = std::bit_ceil(layers);
    }
    const u32 pitch = ((t[4] >> 13) & 0x3FFF) + 1;
    const u32 levels = std::min(last_level + 1, rt.levels);
    Tiling::MipLevel mips[16];
    Tiling::MipLayout(tiling, pitch, rt.height, layers, levels, bytes_per_el,
                      false, pow2, mips);
    // Every (level, slice) of the image starts at its own address: GT7 writes
    // the faces of its reflection cube's mips one by one as 2D storage images
    // (cs 0x9000e31f00 at 0x900b3b0000, 0x900b3c0000, ...) and reads faces of
    // level 0 as 2D. Each is an alias of that slice of that level.
    const u32 alias_slices =
        rt.dim == TexDim::Tex2DArray ? std::min<u32>(layers, rt.depth_or_layers) : 1;
    for (u32 m = 0; m < levels; ++m) {
     for (u32 k = 0; k < alias_slices; ++k) {
      if (m == 0 && k == 0) {
        continue; // the image itself
      }
      if (k && !mips[m].slice_size) {
        break;
      }
      const u64 a = rt.addr + mips[m].offset + u64(k) * mips[m].slice_size;
      MipAlias alias{rt.addr, m, k, std::max(rt.width >> m, 1u),
                     std::max(rt.height >> m, 1u), rt.tex.pixelFormat};
      auto [it, added] = mip_alias.try_emplace(a, alias);
      if (!added) {
        continue;
      }
      ++texture_epoch; // a new level alias: descriptors at `a` resolve to it
      // Already rendered as a target of its own: move its slices in.
      auto list = color_targets.find(a);
      if (list == color_targets.end()) {
        continue;
      }
      auto &v = list->second;
      for (size_t i = 0; i < v.size();) {
        RenderTarget &c = *v[i];
        if (c.width != alias.width || c.height != alias.height ||
            c.tex.pixelFormat != alias.fmt || c.levels < 1) {
          ++i;
          continue;
        }
        const u32 dst_slices = rt.dim == TexDim::Tex2DArray ? rt.depth_or_layers : 1;
        if (c.initialized && k < dst_slices) {
          const u32 n = std::min<u32>(
              c.dim == TexDim::Tex2DArray ? c.depth_or_layers : 1, dst_slices - k);
          EndPass();
          EnsureCommandBuffer();
          id<MTLBlitCommandEncoder> b = [cmd blitCommandEncoder];
          [b copyFromTexture:c.tex
                   sourceSlice:0
                   sourceLevel:0
                     toTexture:rt.tex
              destinationSlice:k
              destinationLevel:m
                    sliceCount:n
                    levelCount:1];
          [b endEncoding];
          rt.written = std::max(rt.written, c.written);
        }
        DC_INFO("gpu: target %#llx %ux%u is level %u slice %u of %#llx", a, c.width,
                c.height, m, k, rt.addr);
        retired.push_back(std::move(v[i]));
        v.erase(v.begin() + i);
        ++texture_epoch; // surfaces went away: descriptors at `a` change
      }
      if (v.empty()) {
        color_targets.erase(list);
      }
     }
    }
  }

  // Color target `k` as the CB registers describe it; notes its CMASK and
  // clear color for fast clears.
  RenderTarget &ColorTargetFor(const RegFile &r, u32 k) {
    const u32 info = r.ColorTarget(k, Reg::CB_INFO);
    const u64 base = u64(r.ColorTarget(k, Reg::CB_BASE)) << 8;
    u32 width = ((r.ColorTarget(k, Reg::CB_PITCH) & 0x7FF) + 1) * 8;
    const u32 slice = ((r.ColorTarget(k, Reg::CB_SLICE) & 0x3FFFFF) + 1) * 64;
    u32 height = slice / width;
    if (cb_extent[k]) { // the real size; pitch and slice are padded
      width = cb_extent[k] & 0xFFFF;
      height = cb_extent[k] >> 16;
    }
    const ColorFormatInfo cf =
        ColorTargetFormat(ColorInfo::Format(info), ColorInfo::NumberType(info));
    width = std::max(width, 1u);
    height = std::max(height, 1u);
    // Rendering into one slice of an array (cube faces):
    // CB_COLOR_VIEW.SLICE_START.
    const u32 view = r.ColorTarget(k, Reg::CB_VIEW);
    const u32 slice_start = view & 0x7FF, slice_max = (view >> 13) & 0x7FF;
    if (RenderTarget *p = MipParent(base, cf.format, width, height)) {
      p->bind_slice = slice_start + mip_alias[base].slice;
      p->bind_level = mip_alias[base].level;
      p->clear_color = ClearColor(info, r.ColorTarget(k, Reg::CB_CLEAR_WORD0),
                                  r.ColorTarget(k, Reg::CB_CLEAR_WORD1));
      return *p;
    }
    bool array = slice_start || slice_max;
    if (!array) {
      if (RenderTarget *a = Latest(color_targets, base, TexDim::Tex2DArray)) {
        array = a->width == width && a->height == height &&
                a->tex.pixelFormat == cf.format;
      }
    }
    RenderTarget &rt = array
                           ? GetColorTarget(base, cf.format, width, height, 1,
                                            TexDim::Tex2DArray,
                                            std::max(slice_start, slice_max) + 1)
                           : GetColorTarget(base, cf.format, width, height);
    rt.bind_slice = array ? slice_start : 0;
    rt.bind_level = 0;
    rt.clear_color = ClearColor(info, r.ColorTarget(k, Reg::CB_CLEAR_WORD0),
                                r.ColorTarget(k, Reg::CB_CLEAR_WORD1));
    if (ColorInfo::FastClear(info)) {
      NoteMeta(u64(r.ColorTarget(k, Reg::CB_CMASK)) << 8, base, false);
    }
    return rt;
  }

  // CB_COLOR_CONTROL.MODE = eliminate: tiles still fast-cleared get the
  // clear color. Here: a target with a pending clear is cleared now.
  void EliminateFastClear(const RegFile &r) {
    const u32 info = r.ColorTarget(0, Reg::CB_INFO);
    if (!ColorInfo::Format(info) || !r.ColorTarget(0, Reg::CB_BASE)) {
      return;
    }
    RenderTarget &rt = ColorTargetFor(r, 0);
    if (!rt.clear_pending) {
      return;
    }
    RenderTarget *colors[8] = {&rt};
    BeginPass(colors, nullptr, false, 1.0f);
    EndPass();
  }

  // Compute shaders may write a depth surface's memory as a color image
  // (Driveclub's half-resolution depth for rain and particles): when that
  // image is newer than the depth texture, the depth takes it over.
  // Right after the tonemap draw, before the HUD: keeps the tonemapped image
  // and the unclamped peak per pixel. Flip masks out what the HUD covered.
  void EdrCapture() {
    edr_armed = false;
    RenderTarget *fin = Latest(color_targets, edr_final);
    RenderTarget *scene = Latest(color_targets, edr_scene);
    RenderTarget *bloom = Latest(color_targets, edr_bloom);
    if (!fin || !fin->tex || !scene || !scene->tex || !bloom || !bloom->tex ||
        !edr_peak_pipeline) {
      return;
    }
    EndPass();
    EnsureCommandBuffer();
    id<MTLTexture> f = fin->tex;
    if (!edr_base || edr_base.width != f.width || edr_base.height != f.height) {
      MTLTextureDescriptor *d = [MTLTextureDescriptor
          texture2DDescriptorWithPixelFormat:f.pixelFormat
                                       width:f.width
                                      height:f.height
                                   mipmapped:NO];
      d.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
      d.storageMode = MTLStorageModePrivate;
      edr_base = [device newTextureWithDescriptor:d];
      edr_base_view = [edr_base newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm];
      d = [MTLTextureDescriptor
          texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                       width:f.width
                                      height:f.height
                                   mipmapped:NO];
      d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
      d.storageMode = MTLStorageModePrivate;
      edr_peak = [device newTextureWithDescriptor:d];
      DC_INFO("edr: capture %lux%lu fmt %d exposure %g", (unsigned long)f.width,
              (unsigned long)f.height, (int)f.pixelFormat, edr_exposure);
    }
    id<MTLBlitCommandEncoder> b = [cmd blitCommandEncoder];
    [b copyFromTexture:f
           sourceSlice:0
           sourceLevel:0
          sourceOrigin:MTLOriginMake(0, 0, 0)
            sourceSize:MTLSizeMake(f.width, f.height, 1)
             toTexture:edr_base
      destinationSlice:0
      destinationLevel:0
     destinationOrigin:MTLOriginMake(0, 0, 0)];
    [b endEncoding];
    id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
    [e setComputePipelineState:edr_peak_pipeline];
    [e setTexture:scene->tex atIndex:0];
    [e setTexture:bloom->tex atIndex:1];
    [e setTexture:edr_peak atIndex:2];
    [e setBytes:&edr_exposure length:sizeof(edr_exposure) atIndex:0];
    [e dispatchThreads:MTLSizeMake(f.width, f.height, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [e endEncoding];
    edr_have = true;
  }

  void DepthFromColor(RenderTarget &z) {
    static const bool off = std::getenv("DC_NO_DEPTH_FROM_COLOR") != nullptr; // A/B
    RenderTarget *c = off ? nullptr : Latest(color_targets, z.addr, TexDim::Tex2D, z.width, z.height);
    if (!c || !c->tex || c->written <= z.written || c->width != z.width ||
        c->height != z.height || c->tex.textureType != MTLTextureType2D ||
        z.tex.textureType != MTLTextureType2D) {
      return;
    }
    const MTLPixelFormat zf = z.tex.pixelFormat;
    id<MTLRenderPipelineState> pso = zf == MTLPixelFormatDepth32Float ? zcopy_pipeline
                                     : zf == MTLPixelFormatDepth32Float_Stencil8 ? zcopy_pipeline_s8
                                                                                 : nil;
    if (!pso) {
      ReportOnce("depth from a color image: depth format " + std::to_string(zf));
      return;
    }
    EndPass();
    EnsureCommandBuffer();
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.depthAttachment.texture = z.tex;
    rp.depthAttachment.loadAction = MTLLoadActionDontCare;
    rp.depthAttachment.storeAction = MTLStoreActionStore;
    if (zf == MTLPixelFormatDepth32Float_Stencil8) {
      rp.stencilAttachment.texture = z.tex;
      rp.stencilAttachment.loadAction = z.initialized ? MTLLoadActionLoad : MTLLoadActionClear;
      rp.stencilAttachment.storeAction = MTLStoreActionStore;
    }
    id<MTLRenderCommandEncoder> e = [cmd renderCommandEncoderWithDescriptor:rp];
    [e setRenderPipelineState:pso];
    [e setDepthStencilState:zcopy_ds];
    [e setFragmentTexture:c->tex atIndex:0];
    [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [e endEncoding];
    z.initialized = true;
    z.clear_pending = false;
    z.written = ++write_seq;
    if (trace) {
      DC_INFO("gpu:   depth %#llx from its color image (%ux%u fmt %lu)",
              (unsigned long long)z.addr, z.width, z.height,
              (unsigned long)c->tex.pixelFormat);
    }
  }

  // CB_RESOLVE: the multisampled CB0 into CB1 (Driveclub's rain is drawn
  // with MSAA at half resolution and composited from the resolved copy).
  // Targets are single sampled here, so the resolve is a copy.
  void ResolveColor(const RegFile &r) {
    for (u32 k = 0; k < 2; ++k) {
      if (!ColorInfo::Format(r.ColorTarget(k, Reg::CB_INFO)) ||
          !r.ColorTarget(k, Reg::CB_BASE)) {
        return;
      }
    }
    RenderTarget &src = ColorTargetFor(r, 0);
    RenderTarget &dst = ColorTargetFor(r, 1);
    if (&src == &dst) {
      return;
    }
    if (src.clear_pending) {
      RenderTarget *colors[8] = {&src};
      BeginPass(colors, nullptr, false, 1.0f);
      EndPass();
    }
    if (src.tex.pixelFormat != dst.tex.pixelFormat) {
      ReportOnce("resolve between different formats");
      return;
    }
    const u32 sw = std::max(1u, src.width >> src.bind_level);
    const u32 sh = std::max(1u, src.height >> src.bind_level);
    const u32 dw = std::max(1u, dst.width >> dst.bind_level);
    const u32 dh = std::max(1u, dst.height >> dst.bind_level);
    EndPass();
    EnsureCommandBuffer();
    id<MTLBlitCommandEncoder> b = [cmd blitCommandEncoder];
    [b copyFromTexture:src.tex
               sourceSlice:src.bind_slice
               sourceLevel:src.bind_level
              sourceOrigin:MTLOriginMake(0, 0, 0)
                sourceSize:MTLSizeMake(std::min(sw, dw), std::min(sh, dh), 1)
                 toTexture:dst.tex
          destinationSlice:dst.bind_slice
          destinationLevel:dst.bind_level
         destinationOrigin:MTLOriginMake(0, 0, 0)];
    [b endEncoding];
    dst.initialized = true;
    dst.clear_pending = false;
    dst.written = ++write_seq;
    NoteTargetWritten(dst);
    if (trace) {
      DC_INFO("gpu:   resolved %#llx into %#llx", (unsigned long long)src.addr,
              (unsigned long long)dst.addr);
    }
  }

  // The dword a fill shader stores, when it loads it with s_buffer_load_dword
  // from offset 0 of a buffer in user data (the CMASK/HTILE fill shader).
  static std::optional<u32> FillValue(const Program &p, const u32 *ud) {
    for (const auto &i : p.code) {
      if (i.enc != Shader::Gcn::Enc::SMRD ||
          static_cast<Shader::Gcn::Smrd>(i.op) != Shader::Gcn::Smrd::S_BUFFER_LOAD_DWORD ||
          !i.imm || i.smrd_offset != 0 || i.sbase + 3u >= 16u) {
        continue;
      }
      const u64 addr = (u64(ud[i.sbase]) | (u64(ud[i.sbase + 1] & 0xFFF) << 32)) & kGpuAddrMask;
      if (!addr || !Readable(addr, 4)) {
        return std::nullopt;
      }
      u32 v;
      std::memcpy(&v, reinterpret_cast<const void *>(addr), 4);
      return v;
    }
    return std::nullopt;
  }

  // True if `p` stores through the V# in user SGPRs ud..ud+3 and never loads
  // from it.
  static bool OnlyStoresTo(const Program &p, u32 ud) {
    bool store = false;
    for (const auto &i : p.code) {
      if ((i.enc != Shader::Gcn::Enc::MUBUF &&
           i.enc != Shader::Gcn::Enc::MTBUF) ||
          i.srsrc != ud) {
        continue;
      }
      const bool is_store =
          (i.op >= 4 && i.op <= 7) ||
          (i.enc == Shader::Gcn::Enc::MUBUF && i.op >= 24 && i.op <= 31);
      if (!is_store) {
        return false;
      }
      store = true;
    }
    return store;
  }

  // Probe slot for this draw/dispatch of the probed shader, or -1.
  int ProbeSlot(u64 shader, const std::string &what) {
    if (!probe_buf || !trace ||
        (shader != probe.shader && probe.shader != ~0ull) ||
        probe_slots.size() >= kProbeSlots) {
      return -1;
    }
    probe_slots.push_back(what);
    return static_cast<int>(probe_slots.size() - 1);
  }

  void PrintProbes() {
    const u32 *d = static_cast<const u32 *>(probe_buf.contents);
    for (size_t k = 0; k < probe_slots.size(); ++k) {
      const u32 *w = d + k * Shader::kProbeWords;
      if (!w[0]) {
        if (probe.shader != ~0ull) {
          DC_INFO("probe %zu (%s): not reached", k, probe_slots[k].c_str());
        }
        continue;
      }
      DC_INFO("probe %zu (%s) at %u,%u: exec=%#x vcc=%#x m0=%#x", k,
              probe_slots[k].c_str(), w[4], w[5], w[1], w[2], w[3]);
      std::string line;
      for (u32 n = 0; n < 96; ++n) {
        float f;
        std::memcpy(&f, &w[8 + n], 4);
        char b[64];
        std::snprintf(b, sizeof(b), " v%u=%08x(%g)", n, w[8 + n], f);
        line += b;
        if (n % 6 == 5) {
          DC_INFO("probe  %s", line.c_str());
          line.clear();
        }
      }
      for (u32 n = 0; n < 104; ++n) {
        char b[32];
        std::snprintf(b, sizeof(b), " s%u=%08x", n, w[104 + n]);
        line += b;
        if (n % 8 == 7) {
          DC_INFO("probe  %s", line.c_str());
          line.clear();
        }
      }
    }
    probe_slots.clear();
    std::memset(probe_buf.contents, 0, probe_buf.length);
  }

  void Draw(const RegFile &r, const DrawCall &dc);
  void Dispatch(const RegFile &r, u32 x, u32 y, u32 z);
  u64 indirect_args = 0; // guest address of the group counts (DispatchIndirect)

  // An ES or GS run inside the copy VS's render pass as a vertex function
  // (MslOptions::ring_vertex): queued by DrawGs, run by Draw once the pass is
  // open, each followed by a vertex->vertex buffer barrier. Running them as
  // compute kernels split the pass per GS draw (144 times a GT7 race frame,
  // each a store and reload of the 1080p targets: 10 fps, 30 without them).
  struct RingReq {
    Shader::RingInfo ring;
    u64 addr = 0;
    u32 ud[16] = {};
    u32 num_user_sgprs = 0;
    u32 threads = 0;
    u64 index_va = 0;
  };
  std::vector<RingReq> ring_pending;

  // The ES and GS fused into the copy VS (MslOptions::FusedRing): no
  // barriers, which cost ~0.7 ms per GS draw as separate vertex functions
  // (35 ms of a race frame). Set by DrawGs for the copy draw that follows.
  struct FusedPending {
    bool on = false;
    Shader::MslOptions::FusedRing fused;
    u64 key = 0; // of the ES, GS and ring layout: part of the VS pipeline key
    u64 es_addr = 0, gs_addr = 0;
    u32 es_ud[16] = {}, gs_ud[16] = {};
    u64 es_index_va = 0; // ES vertex indices (base vertex folded in), or 0
    // The guest's ring bases and the offsets to this draw's arena copies.
    u64 esgs_base = 0, esgs_bias = 0, gsvs_base = 0, gsvs_bias = 0;
  } fused_pending;
  std::unordered_map<u64, std::string> fused_src; // dc_es / dc_gs sources by their keys

  // Whether `p` can run as a vertex function: no LDS, barriers or cross-lane
  // ops (a vertex function has no threadgroup). DC_GS_COMPUTE=1 keeps the
  // compute path for every ring program.
  bool RingAsVertex(const Program *p) const {
    static const bool compute = std::getenv("DC_GS_COMPUTE") != nullptr;
    return p && !compute && p->images.empty() && !p->uses_lds && !p->uses_barrier && !p->cross_lane;
  }
  // Fused into the copy VS: as above, and nothing that must run once per
  // thread (every vertex of a primitive runs its ES and GS again).
  bool RingAsFused(const Program *p) const {
    static const bool vertex = std::getenv("DC_GS_VERTEX") != nullptr; // the separate-draw path instead
    return RingAsVertex(p) && !vertex && !p->uses_atomic;
  }

  // Draws the queued ring programs (RingReq) in the open pass, with the
  // pass's attachment formats in their pipelines. False: a pipeline failed,
  // the copy VS must not draw.
  bool RunRingPending(RenderTarget *const colors[8], const MTLPixelFormat color_fmt[8],
                      MTLPixelFormat depth_fmt) {
    UpdateMemMap();
    for (const RingReq &q : ring_pending) {
      Program *p = GetProgram(Stage::Compute, q.addr, q.ud, q.num_user_sgprs, 0);
      if (!p) {
        return false;
      }
      Shader::MslOptions o;
      o.map_hoist = map_hoist;
      o.ring = q.ring;
      o.ring_vertex = true;
      u64 key = Hash(p->hash);
      key = Hash(o.map_hoist, key);
      key = Hash(u64(q.ring.stage) | u64(q.ring.esgs_itemsize) << 8 |
                     u64(q.ring.vert_itemsize) << 24 | u64(q.ring.slots) << 40 |
                     u64(q.ring.input_verts) << 48 | u64(q.ring.index_size) << 52 | 1ull << 56,
                 key);
      u64 pkey = key;
      for (u32 k = 0; k < 8; ++k) {
        pkey = Hash(colors[k] ? u64(color_fmt[k]) + 1 : 0, pkey);
      }
      pkey = Hash(u64(depth_fmt), pkey);
      TakeCompiled();
      auto it = pipelines.find(pkey);
      if (it == pipelines.end()) {
        if (!pending.count(pkey)) {
          pending.insert(pkey);
          MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
          d.rasterizationEnabled = NO;
          // A layered pass (renderTargetArrayLength) needs the topology set.
          d.inputPrimitiveTopology = MTLPrimitiveTopologyClassPoint;
          for (u32 k = 0; k < 8; ++k) {
            if (colors[k]) {
              d.colorAttachments[k].pixelFormat = color_fmt[k];
            }
          }
          d.depthAttachmentPixelFormat = depth_fmt;
          if (depth_fmt == MTLPixelFormatDepth32Float_Stencil8) {
            d.stencilAttachmentPixelFormat = depth_fmt;
          }
          compiler->Submit([this, d, src = MslSource(*p, o, key), key, pkey, addr = q.addr] {
            id<MTLRenderPipelineState> pso = nil;
            if (id<MTLFunction> fn = compiler->Function(key, src, addr)) {
              d.vertexFunction = fn;
              NSError *err = nil;
              pso = [device newRenderPipelineStateWithDescriptor:d error:&err];
              if (!pso) {
                DC_ERROR("gpu: ring vertex pipeline %#llx: %s", addr,
                         err ? err.localizedDescription.UTF8String : "?");
              }
            }
            {
              std::scoped_lock lk(done_lock);
              done_render.emplace_back(pkey, pso);
            }
            done_cv.notify_all();
          }, /*urgent=*/true);
        }
        WaitCompiled(pkey);
        it = pipelines.find(pkey);
      }
      id<MTLRenderPipelineState> pso = it == pipelines.end() ? nil : it->second;
      if (!pso) {
        ReportOnce("gs emulation: no vertex pipeline for ring program " + Hex(q.addr));
        return false;
      }
      u32 cud[16];
      std::memcpy(cud, q.ud, sizeof(cud));
      PatchUserData(cud);
      [enc setRenderPipelineState:pso];
      [enc setVertexBytes:cud length:64 atIndex:Shader::Binding::UserData];
      [enc setVertexBuffer:mem[0] offset:0 atIndex:Shader::Binding::Mem0];
      [enc setVertexBuffer:mem[1] offset:0 atIndex:Shader::Binding::Mem1];
      [enc setVertexBuffer:mem_map offset:0 atIndex:Shader::Binding::MemMap];
      [enc setVertexBuffer:gds offset:Gpu::GdsBase() atIndex:Shader::Binding::Gds];
      if (q.index_va) {
        id<MTLBuffer> ib = nil;
        u64 ioff = 0;
        if (!Locate(q.index_va, 4, &ib, &ioff)) {
          return false;
        }
        [enc setVertexBuffer:ib offset:ioff atIndex:Shader::Binding::GsIndex];
      }
      [enc drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:q.threads];
      // The next stage (GS, or the copy VS) reads what this one wrote.
      // DC_RING_NOBARRIER=1: measure the barriers' cost (wrong output).
      static const bool no_barrier = std::getenv("DC_RING_NOBARRIER") != nullptr;
      if (!no_barrier) {
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers
                        afterStages:MTLRenderStageVertex
                       beforeStages:MTLRenderStageVertex];
      }
      if (trace) {
        DC_INFO("gpu:   ring vertex %s %#llx: %u threads, index buffer %#llx",
                q.ring.stage == Shader::RingStage::Export ? "es" : "gs", (unsigned long long)q.addr,
                q.threads, (unsigned long long)q.index_va);
        if (q.ring.stage == Shader::RingStage::Geometry && q.index_va) {
          // Once the GPU ran it: did the GS write its indices?
          const u64 va = q.index_va, n = u64(q.threads) * q.ring.IndicesPerPrim();
          [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
            const u32 *d = reinterpret_cast<const u32 *>(va);
            u64 nz = 0;
            for (u64 i = 0; i < n; ++i) {
              nz += d[i] != 0;
            }
            DC_INFO("gpu: ring vertex gs indices at %#llx: %llu of %llu non-zero", (unsigned long long)va,
                    (unsigned long long)nz, (unsigned long long)n);
          }];
        }
      }
    }
    bound = {}; // the copy VS rebinds everything
    return true;
  }

  // Runs the ES or GS of a GS pipeline as a compute kernel, one thread per
  // vertex / primitive (Shader::RingInfo). Ring programs only touch memory
  // (no textures). `index_va`: where a GS writes the copy-VS indices.
  bool RunRing(const Shader::RingInfo &ring, u64 addr, const u32 *ud,
               u32 num_user_sgprs, u32 threads, u32 group, u64 index_va) {
    Program *p = GetProgram(Stage::Compute, addr, ud, num_user_sgprs, 0);
    if (!p || !p->images.empty()) {
      ReportOnce("gs emulation: ring program " + Hex(addr) +
                 (p ? " samples textures" : " cannot be decoded"));
      return false;
    }
    Shader::MslOptions o;
    o.map_hoist = map_hoist;
    o.ring = ring;
    u64 key = Hash(p->hash);
    key = Hash(o.map_hoist, key);
    key = Hash(u64(ring.stage) | u64(ring.esgs_itemsize) << 8 |
                   u64(ring.vert_itemsize) << 24 | u64(ring.slots) << 40 |
                   u64(ring.input_verts) << 48 | u64(ring.index_size) << 52,
               key);
    TakeCompiled();
    auto it = compute_pipelines.find(key);
    if (it == compute_pipelines.end()) {
      if (!pending.count(key)) {
        pending.insert(key);
        compiler->Submit([this, src = MslSource(*p, o, key), key, addr] {
          id<MTLComputePipelineState> pso = nil;
          if (id<MTLFunction> fn = compiler->Function(key, src, addr)) {
            NSError *err = nil;
            pso = [device newComputePipelineStateWithFunction:fn error:&err];
            if (!pso) {
              DC_ERROR("gpu: ring pipeline %#llx: %s", addr,
                       err ? err.localizedDescription.UTF8String : "?");
            }
          }
          {
            std::scoped_lock lk(done_lock);
            done_compute.emplace_back(key, pso);
          }
          done_cv.notify_all();
        }, /*urgent=*/true);
      }
      WaitCompiled(key);
      it = compute_pipelines.find(key);
    }
    id<MTLComputePipelineState> pso = it->second;
    if (!pso) {
      return false;
    }
    EndPass();
    EnsureCommandBuffer();
    UpdateMemMap();
    u32 cud[16];
    std::memcpy(cud, ud, sizeof(cud));
    PatchUserData(cud);
    id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
    [e setComputePipelineState:pso];
    [e setBytes:cud length:64 atIndex:Shader::Binding::UserData];
    [e setBuffer:mem[0] offset:0 atIndex:Shader::Binding::Mem0];
    [e setBuffer:mem[1] offset:0 atIndex:Shader::Binding::Mem1];
    [e setBuffer:mem_map offset:0 atIndex:Shader::Binding::MemMap];
    [e setBuffer:gds offset:Gpu::GdsBase() atIndex:Shader::Binding::Gds];
    if (index_va) {
      id<MTLBuffer> ib = nil;
      u64 ioff = 0;
      if (!Locate(index_va, 4, &ib, &ioff)) {
        [e endEncoding];
        return false;
      }
      [e setBuffer:ib offset:ioff atIndex:Shader::Binding::GsIndex];
    }
    [e setThreadgroupMemoryLength:16 atIndex:0];
    // Exact thread count; the last group may be partial (lanes stay 0..n).
    [e dispatchThreads:MTLSizeMake(threads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
    [e endEncoding];
    return true;
  }

  // Geometry-shader pipeline (VGT_SHADER_STAGES_EN.GS_EN). Metal has no GS:
  // the ES and GS run as compute kernels filling the guest's ESGS / GSVS
  // rings, then the copy VS draws a triangle list from the GSVS ring through
  // the indices the GS wrote. Handles non-indexed point input and line input,
  // indexed or not, with triangle-strip output; otherwise returns false and the
  // draw is skipped. DC_NO_GS=1: skip them all; DC_GS_ONLY=<gs addr>: only that GS.
  bool DrawGs(const RegFile &r, const DrawCall &dc) {
    static const bool off = std::getenv("DC_NO_GS") != nullptr;
    static const u64 only = std::getenv("DC_GS_ONLY") ? std::strtoull(std::getenv("DC_GS_ONLY"), nullptr, 16) : 0;
    if (off || !dc.count || (only && r.ProgramAddr(Reg::SPI_SHADER_PGM_LO_GS) != only)) {
      return false;
    }
    // An indirect draw's count is the CP's read of its args, as for any draw.
    const bool lines = r.Primitive() == PrimType::LineList;
    if ((!lines && (r.Primitive() != PrimType::PointList || dc.indexed)) || dc.instances != 1 ||
        (r[Reg::VGT_GS_OUT_PRIM_TYPE] & 0x3F) != 2 || (r[Reg::VGT_GS_INSTANCE_CNT] & 1)) {
      ReportOnce("gs emulation: unsupported pipeline (prim " +
                 std::to_string(static_cast<u32>(r.Primitive())) + ", out " +
                 std::to_string(r[Reg::VGT_GS_OUT_PRIM_TYPE] & 0x3F) + ", indexed " +
                 std::to_string(dc.indexed) + ", instances " + std::to_string(dc.instances) +
                 ", indirect " + std::to_string(dc.indirect_args != 0) + ", gs instances " +
                 std::to_string(r[Reg::VGT_GS_INSTANCE_CNT]) + ")");
      return false;
    }
    // Emitted vertices per output-dword plane: the copy VS reads output
    // dword k at k * plane bytes, a plane holding `slots` vertices of 16 lanes.
    const u64 vs_addr = r.ProgramAddr(Reg::SPI_SHADER_PGM_LO_VS);
    Program *vs = GetProgram(Stage::Vertex, vs_addr, r.UserData(Reg::SPI_SHADER_USER_DATA_VS_0),
                             Bits(r[Reg::SPI_SHADER_PGM_RSRC2_VS], 1, 5),
                             Bits(r[Reg::SPI_SHADER_PGM_RSRC1_VS], 24, 2));
    std::vector<u32> offsets;
    for (const auto &i : vs ? vs->code : std::vector<Shader::Gcn::Inst>{}) {
      if (i.enc == Shader::Gcn::Enc::MUBUF) {
        offsets.push_back(i.offset);
      }
    }
    std::sort(offsets.begin(), offsets.end());
    u32 plane = 0;
    for (size_t k = 1; k < offsets.size(); ++k) {
      if (offsets[k] != offsets[k - 1]) {
        plane = plane ? std::min(plane, offsets[k] - offsets[k - 1]) : offsets[k] - offsets[k - 1];
      }
    }
    Shader::RingInfo ring;
    ring.esgs_itemsize = r[Reg::VGT_ESGS_RING_ITEMSIZE] & 0x7FFF;
    ring.vert_itemsize = r[Reg::VGT_GS_VERT_ITEMSIZE] & 0x7FFF;
    ring.slots = plane / 64;
    // Planes past the 12-bit immediate leave every offset in soffset, out of
    // sight here; a plane holds VGT_GS_MAX_VERT_OUT vertices.
    if (!plane) {
      ring.slots = r[Reg::VGT_GS_MAX_VERT_OUT] & 0x7FF;
    }
    if (plane % 64 || ring.slots < 3 || ring.slots > 64 || !ring.esgs_itemsize ||
        !ring.vert_itemsize) {
      std::string list;
      for (const u32 o : offsets) {
        list += " " + std::to_string(o);
      }
      ReportOnce("gs emulation: copy shader layout not understood (plane " + std::to_string(plane) +
                 ", esgs item " + std::to_string(ring.esgs_itemsize) + ", vert item " +
                 std::to_string(ring.vert_itemsize) + ", vs " + std::to_string(vs_addr) +
                 ", max verts " + std::to_string(r[Reg::VGT_GS_MAX_VERT_OUT]) + ", offsets" + list + ")");
      return false;
    }
    // Never write past the rings, whatever count the guest gave: primitives
    // that do not fit are dropped. The guest does not set VGT_*_RING_SIZE in its
    // command buffers, so the sizes come from the ring V#s: the one the GS's
    // first buffer access with a known base reads (ESGS) and the copy VS's (GSVS).
    // `base_out`: the ring V#'s 48-bit base as the shader sees it (dc_vbase).
    auto ring_bytes = [&](Program *p, const u32 *ud, u64 *base_out) -> u64 {
      for (const auto &i : p ? p->code : std::vector<Shader::Gcn::Inst>{}) {
        if (i.enc != Shader::Gcn::Enc::MUBUF) {
          continue;
        }
        const u32 k = p->LinearBaseOf(i.pc);
        if (k >= p->linear_bases.size()) {
          continue; // an SGPR soffset has no base; the next access has the same V#
        }
        u64 va = 0, bytes = 0;
        Shader::SgprReadCache cache;
        const Shader::LinearBase &b = p->linear_bases[k];
        const u32 w0 = Shader::ResolveSgpr(b.words[0], ud, &cache), w1 = Shader::ResolveSgpr(b.words[1], ud, &cache);
        *base_out = u64(w0) | (u64(w1 & 0xFFF) << 32);
        return Shader::LinearBaseRange(b, ud, &cache, &va, &bytes) ? bytes : 0;
      }
      return 0;
    };
    const u32 *gs_ud = r.UserData(Reg::SPI_SHADER_USER_DATA_GS_0);
    Program *gsp = GetProgram(Stage::Compute, r.ProgramAddr(Reg::SPI_SHADER_PGM_LO_GS), gs_ud,
                              Bits(r[Reg::SPI_SHADER_PGM_RSRC2_GS], 1, 5), 0);
    u64 esgs_bytes = u64(r[Reg::VGT_ESGS_RING_SIZE]) << 8;
    u64 gsvs_bytes = u64(r[Reg::VGT_GSVS_RING_SIZE]) << 8;
    u64 esgs_va = 0, gsvs_va = 0;
    const u64 esgs_probe = ring_bytes(gsp, gs_ud, &esgs_va);
    const u64 gsvs_probe = ring_bytes(vs, r.UserData(Reg::SPI_SHADER_USER_DATA_VS_0), &gsvs_va);
    esgs_bytes = esgs_bytes ? esgs_bytes : esgs_probe;
    gsvs_bytes = gsvs_bytes ? gsvs_bytes : gsvs_probe;
    if (!esgs_bytes || !gsvs_bytes) {
      ReportOnce("gs emulation: ring sizes not set");
      return false;
    }
    ring.input_verts = lines ? 2 : 1;
    const u32 n = static_cast<u32>(std::min<u64>({dc.count / ring.input_verts,
                                                  esgs_bytes / ring.EsWaveBytes() * 64 / ring.input_verts,
                                                  gsvs_bytes / ring.GsGroupBytes() * 16}));
    const u64 index_bytes = u64(n) * ring.IndicesPerPrim() * 4;
    if (!n || index_bytes > kArenaSize / 8) {
      ReportOnce("gs emulation: " + std::to_string(dc.count) + " vertices do not fit");
      return false;
    }
    const u64 index_va = ArenaAlloc(index_bytes);
    if (!index_va) {
      return false;
    }
    ring.stage = Shader::RingStage::Export;
    ring.index_size = dc.indexed ? dc.index_size : 0;
    u64 es_index_va = dc.indexed ? dc.index_base + u64(dc.index_offset) * dc.index_size : 0;
    // A base vertex reaches the ES as explicit indices: one pipeline serves
    // every base. GT7 draws each text glyph as a line at its own base.
    const u32 es_verts = n * ring.input_verts;
    const u32 base_vertex = r[Reg::VGT_INDX_OFFSET] + (dc.indexed ? 0 : dc.first_vertex);
    if (base_vertex) {
      if (dc.indexed && !Readable(es_index_va, u64(es_verts) * dc.index_size)) {
        ReportOnce("index buffer not mapped");
        return false;
      }
      const u64 va = ArenaAlloc(u64(es_verts) * 4);
      if (!va) {
        return false;
      }
      auto *out = reinterpret_cast<u32 *>(va);
      for (u32 k = 0; k < es_verts; ++k) {
        u32 v = k;
        if (dc.indexed) {
          v = dc.index_size == 4 ? reinterpret_cast<const u32 *>(es_index_va)[k]
                                 : reinterpret_cast<const u16 *>(es_index_va)[k];
        }
        out[k] = v + base_vertex;
      }
      es_index_va = va;
      ring.index_size = 4;
    }
    const u64 es_addr = r.ProgramAddr(Reg::SPI_SHADER_PGM_LO_ES);
    const u32 *es_ud = r.UserData(Reg::SPI_SHADER_USER_DATA_ES_0);
    const u32 es_nus = Bits(r[Reg::SPI_SHADER_PGM_RSRC2_ES], 1, 5);
    const u64 gs_addr = r.ProgramAddr(Reg::SPI_SHADER_PGM_LO_GS);
    const u32 gs_nus = Bits(r[Reg::SPI_SHADER_PGM_RSRC2_GS], 1, 5);
    // Inside the pass as vertex functions when both programs allow it (see
    // RingReq); the copy VS's Draw runs them once its pass is open.
    ring_pending.clear();
    fused_pending = {};
    Program *esp = GetProgram(Stage::Compute, es_addr, es_ud, es_nus, 0);
    Program *gsp2 = GetProgram(Stage::Compute, gs_addr, gs_ud, gs_nus, 0);
    auto ring_key = [](const Shader::RingInfo &q, u64 key) {
      return Hash(u64(q.stage) | u64(q.esgs_itemsize) << 8 | u64(q.vert_itemsize) << 24 | u64(q.slots) << 40 |
                      u64(q.input_verts) << 48 | u64(q.index_size) << 52 | 2ull << 56,
                  key);
    };
    bool fused = false;
    // This draw's own rings in the arena: draws in flight at once must not
    // share the guest's (the fused VS of a later draw would overwrite them
    // while an earlier draw's vertices still read).
    const u64 esgs_used = u64((es_verts + 63) / 64) * ring.EsWaveBytes();
    const u64 gsvs_used = u64((n + 15) / 16) * ring.GsGroupBytes();
    u64 esgs_region = 0, gsvs_region = 0;
    if (RingAsFused(esp) && RingAsFused(gsp2) && esgs_va && gsvs_va && (esgs_region = ArenaAlloc(esgs_used)) &&
        (gsvs_region = ArenaAlloc(gsvs_used))) {
      // Both fused into the copy VS (FusedPending): sources of dc_es / dc_gs.
      Shader::MslOptions oe;
      oe.map_hoist = map_hoist;
      oe.ring = ring;
      oe.ring_fused = true;
      const u64 ke = ring_key(ring, Hash(oe.map_hoist, Hash(esp->hash)));
      Shader::RingInfo gring = ring;
      gring.stage = Shader::RingStage::Geometry;
      gring.index_size = 0;
      Shader::MslOptions og;
      og.map_hoist = map_hoist;
      og.ring = gring;
      og.ring_fused = true;
      const u64 kg = ring_key(gring, Hash(og.map_hoist, Hash(gsp2->hash)));
      FusedPending &f = fused_pending;
      f.on = true;
      f.fused.on = true;
      f.fused.input_verts = ring.input_verts;
      f.fused.ipp = ring.IndicesPerPrim();
      f.fused.es_index_size = static_cast<u8>(ring.index_size);
      auto cached = [&](u64 key, const Program &p, const Shader::MslOptions &opt) -> const std::string & {
        auto it = fused_src.find(key);
        if (it == fused_src.end()) {
          it = fused_src.emplace(key, MslSource(p, opt, key)).first;
        }
        return it->second;
      };
      f.fused.es_src = cached(ke, *esp, oe);
      f.fused.gs_src = cached(kg, *gsp2, og);
      f.key = Hash(ke, kg);
      f.es_addr = es_addr;
      f.gs_addr = gs_addr;
      std::memcpy(f.es_ud, es_ud, sizeof(f.es_ud));
      std::memcpy(f.gs_ud, gs_ud, sizeof(f.gs_ud));
      f.es_index_va = es_index_va;
      f.esgs_base = esgs_va;
      f.esgs_bias = esgs_region - esgs_va;
      f.gsvs_base = gsvs_va;
      f.gsvs_bias = gsvs_region - gsvs_va;
      fused = true;
    } else if (RingAsVertex(esp) && RingAsVertex(gsp2)) {
      RingReq es;
      es.ring = ring;
      es.addr = es_addr;
      std::memcpy(es.ud, es_ud, sizeof(es.ud));
      es.num_user_sgprs = es_nus;
      es.threads = es_verts;
      es.index_va = es_index_va;
      ring_pending.push_back(es);
      RingReq gs;
      ring.stage = Shader::RingStage::Geometry;
      ring.index_size = 0;
      gs.ring = ring;
      gs.addr = gs_addr;
      std::memcpy(gs.ud, gs_ud, sizeof(gs.ud));
      gs.num_user_sgprs = gs_nus;
      gs.threads = n;
      gs.index_va = index_va;
      ring_pending.push_back(gs);
    } else {
      if (!RunRing(ring, es_addr, es_ud, es_nus, es_verts, 64, es_index_va)) {
        return false;
      }
      ring.stage = Shader::RingStage::Geometry;
      ring.index_size = 0;
      if (!RunRing(ring, gs_addr, gs_ud, gs_nus, n, 16, index_va)) {
        return false;
      }
    }
    if (trace) {
      DC_INFO("gpu:   gs emulated: %u of %u points, %u slots, esgs %u of %llu B, gsvs %u of %llu B, "
              "indices at %#llx",
              n, dc.count, ring.slots, (n * ring.input_verts + 63) / 64 * ring.EsWaveBytes(),
              (unsigned long long)esgs_bytes,
              (n + 15) / 16 * ring.GsGroupBytes(), (unsigned long long)gsvs_bytes,
              (unsigned long long)index_va);
    }
    // The copy VS as an ordinary indexed triangle list (GS stage off).
    auto copy = std::make_unique<RegFile>(r);
    (*copy)[Reg::VGT_SHADER_STAGES_EN] &= ~0x20u;
    (*copy)[Reg::VGT_PRIMITIVE_TYPE] = static_cast<u32>(PrimType::TriList);
    (*copy)[Reg::VGT_INDX_OFFSET] = 0; // the ES took the base; these index the ring
    DrawCall d = dc;
    d.indexed = true;
    d.count = n * ring.IndicesPerPrim();
    d.index_base = index_va;
    d.index_offset = 0;
    d.index_size = 4;
    d.indirect_args = 0; // the count is the GS output's, not the guest's args
    if (fused) {
      // No index buffer: the fused VS takes its ring index from the GS it runs.
      d.indexed = false;
      d.index_base = 0;
      d.index_offset = 0;
      d.first_vertex = 0;
    }
    Draw(*copy, d);
    ring_pending.clear(); // a draw that returned early leaves them unrun
    fused_pending = {};
    return true;
  }

  // DC_FIND_ADDR=<addr>: in the traced frame, the draws and dispatches whose
  // user data, or tables it points to (two levels deep), hold a V# covering
  // that address (finds the shaders reading or writing a buffer).
  void FindRefs(const u32 *d, u32 n, int depth, const std::string &path,
                const char *who, u64 shader) {
    // DC_FIND_SIZE=<min>-<max>: V#s of that byte size instead (buffers in
    // GT7's per-frame ring move every run, so no fixed address finds them).
    const u64 find = find_addr;
    if (!find && !find_size_min) {
      return;
    }
    for (u32 k = 0; k + 4 <= n; ++k) {
      const u64 base = (u64(d[k]) | (u64(d[k + 1] & 0xFFF) << 32)) & kGpuAddrMask;
      const u32 stride = (d[k + 1] >> 16) & 0x3FFF;
      const u64 size = stride ? u64(d[k + 2]) * stride : d[k + 2];
      const bool hit = find ? find >= base && find < base + std::max<u64>(size, 16)
                            : size >= find_size_min && size <= find_size_max &&
                                  (d[k + 3] & 0xFFF) != 0; // dst_sel set: a V#
      if (base && hit) {
        DC_INFO("gpu: find %#llx: %s %#llx %s[%u] = %08x %08x %08x %08x",
                (unsigned long long)find, who, (unsigned long long)shader,
                path.c_str(), k, d[k], d[k + 1], d[k + 2], d[k + 3]);
      }
    }
    if (depth >= 2) {
      return;
    }
    for (u32 k = 0; k + 2 <= n; ++k) {
      const u64 p = (u64(d[k]) | (u64(d[k + 1]) << 32)) & kGpuAddrMask;
      id<MTLBuffer> b = nil;
      u64 off = 0;
      const bool ring = p >= 0x200000000ull && p < 0x208800000ull; // GT7 per-frame
      if ((p > 0x7000000000ull || ring) && !(p & 3) && Locate(p, 256, &b, &off)) {
        FindRefs(reinterpret_cast<const u32 *>(p), 64, depth + 1,
                 path + "[" + std::to_string(k) + "]->", who, shader);
      }
    }
  }
  void StartCapture();
  bool BeginCapture(const std::string &path);
  void Flip(const FlipInfo &f);
  std::unordered_map<u64, id<MTLComputePipelineState>> compute_pipelines;

  // DC_DEBUG_MEMTEST: reads `addr` through the shader memory path and
  // compares with what the CPU sees.
  void MemTest(u64 addr) {
    static bool done = false;
    if (done || !Readable(addr, 16)) {
      return;
    }
    done = true;
    std::string src = Shader::kMslPrelude;
    src += "kernel void memtest(device uint* gm0 [[buffer(1)]], device uint* "
           "gm1 [[buffer(2)]], "
           "device const DcMap& gmap [[buffer(3)]], constant ulong& addr "
           "[[buffer(4)]], device uint* out [[buffer(5)]]) {"
           " DcMem mem = {gm0, gm1, &gmap}; for (uint i = 0; i < 4; ++i) "
           "out[i] = dc_ld(mem, addr + i * 4);"
           " out[4] = uint(gmap.pages); out[5] = uint(gmap.split >> 20); }";
    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:@(src.c_str())
                                              options:nil
                                                error:&err];
    id<MTLComputePipelineState> cp = [device
        newComputePipelineStateWithFunction:[lib newFunctionWithName:@"memtest"]
                                      error:&err];
    if (!cp) {
      DC_ERROR("memtest: %s", err.localizedDescription.UTF8String);
      return;
    }
    UpdateMemMap();
    id<MTLBuffer> out =
        [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> c = [queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [c computeCommandEncoder];
    [e setComputePipelineState:cp];
    [e setBuffer:mem[0] offset:0 atIndex:1];
    [e setBuffer:mem[1] offset:0 atIndex:2];
    [e setBuffer:mem_map offset:0 atIndex:3];
    [e setBytes:&addr length:8 atIndex:4];
    [e setBuffer:out offset:0 atIndex:5];
    [e dispatchThreads:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [e endEncoding];
    [c commit];
    [c waitUntilCompleted];
    const u32 *g = static_cast<const u32 *>(out.contents);
    const u32 *cpu = reinterpret_cast<const u32 *>(addr);
    DC_INFO("memtest %#llx: gpu %08x %08x %08x %08x | cpu %08x %08x %08x %08x "
            "| ranges %u split %u MB (%s)",
            addr, g[0], g[1], g[2], g[3], cpu[0], cpu[1], cpu[2], cpu[3], g[4],
            g[5], c.error ? c.error.localizedDescription.UTF8String : "ok");
  }

  // Reads a presented frame back and writes it as a PPM (debug only).
  void DumpFrame(id<MTLTexture> tex) {
    const u32 w = static_cast<u32>(tex.width), h = static_cast<u32>(tex.height);
    id<MTLBuffer> buf =
        [device newBufferWithLength:u64(w) * h * 4
                            options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> b = [cmd blitCommandEncoder];
    [b copyFromTexture:tex
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(w, h, 1)
                        toBuffer:buf
               destinationOffset:0
          destinationBytesPerRow:w * 4
        destinationBytesPerImage:u64(w) * h * 4];
    [b endEncoding];
    Commit(true);
    EnsureCommandBuffer();
    char path[512];
    std::snprintf(path, sizeof(path), "%s/frame_%04llu.ppm", dump_frames,
                  frame_count);
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << w << " " << h << "\n255\n";
    const auto *px = static_cast<const u8 *>(buf.contents);
    std::vector<u8> rgb(u64(w) * h * 3);
    for (u64 i = 0; i < u64(w) * h; ++i) { // BGRA -> RGB
      rgb[i * 3] = px[i * 4 + 2];
      rgb[i * 3 + 1] = px[i * 4 + 1];
      rgb[i * 3 + 2] = px[i * 4];
    }
    out.write(reinterpret_cast<const char *>(rgb.data()),
              static_cast<std::streamsize>(rgb.size()));
  }

#include "gpu/metal/mtl_tess.inc"
};

void MetalRenderer::Impl::Draw(const RegFile &r, const DrawCall &dc) {
  // Without ES and GS the copy VS reads whatever earlier GS draws left in the
  // GSVS ring: valid clip positions, so stray primitives on screen. A draw that
  // was not emulated is skipped.
  // DC_TRACE_RT=<hex target>,<draws>: trace the flip after the first one that
  // draws that often into the target, for runs whose flip numbers vary.
  static const u64 trace_rt =
      std::getenv("DC_TRACE_RT") ? std::strtoull(std::getenv("DC_TRACE_RT"), nullptr, 16) : 0;
  if (trace_rt && !trace_frame && (u64(r.ColorTarget(0, Reg::CB_BASE)) << 8) == trace_rt) {
    static const u32 want = [] {
      const char *c = std::strchr(std::getenv("DC_TRACE_RT"), ',');
      return c ? u32(std::atoi(c + 1)) : 1u;
    }();
    static u64 seen_frame = 0;
    static u32 seen = 0;
    seen = seen_frame == frame_count ? seen + 1 : 1;
    seen_frame = frame_count;
    if (seen >= want) {
      trace_frame = frame_count + 1;
      DC_INFO("gpu: DC_TRACE_RT: tracing flip %llu", (unsigned long long)trace_frame);
    }
  }
  if (trace) { // Which hardware stages this draw asks for, tessellation included.
    DC_INFO("gpu: stages 0x%x prim %u count %u cb_view %#x db_view %#x pos_format %#x vs_out %#x",
            r[Reg::VGT_SHADER_STAGES_EN], static_cast<u32>(r.Primitive()), dc.count,
            r.ColorTarget(0, Reg::CB_VIEW), r[Reg::DB_DEPTH_VIEW],
            r[Reg::SPI_SHADER_POS_FORMAT], r[Reg::PA_CL_VS_OUT_CNTL]);
  }
  if (r[Reg::VGT_SHADER_STAGES_EN] & 0x20) {
    const bool drawn = DrawGs(r, dc);
    if (trace && !drawn)
      DC_INFO("gpu:   skipped: gs not emulated");
    return;
  }
  if ((r[Reg::VGT_SHADER_STAGES_EN] & 0x4) && DrawTess(r, dc)) {
    return;
  }
  CpuTimer draw_timer(cpu_profile ? &cpu_draw_s : nullptr);
  DrawLap(0);
  // DC_PEEK_RT_EACH=1: with DC_PEEK_RT, every draw into the peeked target
  // ends its pass, so the peek names the single draw that wrote the texel.
  static const bool peek_each = std::getenv("DC_PEEK_RT_EACH") != nullptr;
  if (peek_each && trace && enc && PeekRt().addr) {
    for (u32 k = 0; k < 8; ++k) {
      if (pass_tex[k] && (pass_addr[k] & kGpuAddrMask) == PeekRt().addr) {
        EndPass();
        break;
      }
    }
  }
  ++draw_count;
  ++stat_draws;
  ++work_since_commit;
  const u64 vs_addr = r.ProgramAddr(Reg::SPI_SHADER_PGM_LO_VS);
  const u64 ps_addr = r.ProgramAddr(Reg::SPI_SHADER_PGM_LO_PS);
  dbg_shader = ps_addr;
  // DC_TRACE_PS=<hex ps>,<frames>: trace the flip after the n-th one that runs
  // that pixel shader, for scenes whose flip numbers vary between replays.
  static const u64 trace_ps =
      std::getenv("DC_TRACE_PS") ? std::strtoull(std::getenv("DC_TRACE_PS"), nullptr, 16) : 0;
  if (trace_ps && ps_addr == trace_ps && !trace_frame) {
    static const u32 want = [] {
      const char *c = std::strchr(std::getenv("DC_TRACE_PS"), ',');
      return c ? u32(std::atoi(c + 1)) : 1u;
    }();
    static u64 last_frame = ~0ull;
    static u32 frames = 0;
    if (last_frame != frame_count) {
      last_frame = frame_count;
      if (++frames >= want) {
        trace_frame = frame_count + 1;
        DC_INFO("gpu: DC_TRACE_PS: tracing flip %llu", (unsigned long long)trace_frame);
      }
    }
  }
  if (edr_armed) {
    EdrCapture();
  }
  if (edr_on && ps_addr == 0x7003d7ef00) {
    const u32 *ud = r.UserData(Reg::SPI_SHADER_USER_DATA_PS_0);
    const u64 table = (u64(ud[12]) | (u64(ud[13] & 0xFFFF) << 32)) & kGpuAddrMask;
    if (Readable(table, 160)) {
      const u32 *t = reinterpret_cast<const u32 *>(table);
      const u64 cb = (u64(t[36]) | (u64(t[37] & 0xFFF) << 32)) & kGpuAddrMask;
      if (Readable(cb, 600)) {
        edr_exposure = reinterpret_cast<const float *>(cb)[146];
        edr_scene = (u64(ud[0]) | (u64(ud[1] & 0xFF) << 32)) << 8;
        edr_bloom = (u64(t[0]) | (u64(t[1] & 0xFF) << 32)) << 8;
        edr_final = u64(r.ColorTarget(0, Reg::CB_BASE)) << 8;
        edr_armed = true;
      }
    }
  }
  if (trace && (scan_all || !scan_addrs.empty())) {
    scan_label = "pass with draw #" + std::to_string(draw_count) + " ps " + Hex(ps_addr);
  }
  const u32 cb_mode = Bits(r[Reg::CB_COLOR_CONTROL], 4, 3);
  const u32 db_render = r[Reg::DB_RENDER_CONTROL];
  const u32 db_control = r[Reg::DB_DEPTH_CONTROL];
  const PrimType prim = r.Primitive();

  if (trace) {
    DC_INFO("gpu: draw #%llu prim=%#x count=%u idx=%d vs=%#llx ps=%#llx "
            "cb_mode=%u db_render=%#x "
            "db_ctl=%#x cb0=%#llx info=%#x mask=%#x z=%#llx/%#llx zinfo=%#x "
            "blend0=%#x colfmt=%#x a2m=%#x",
            draw_count, static_cast<u32>(prim), dc.count, dc.indexed, vs_addr,
            ps_addr, cb_mode, db_render, db_control,
            u64(r.ColorTarget(0, Reg::CB_BASE)) << 8,
            r.ColorTarget(0, Reg::CB_INFO), r[Reg::CB_TARGET_MASK],
            u64(r[Reg::DB_Z_READ_BASE]) << 8, u64(r[Reg::DB_Z_WRITE_BASE]) << 8,
            r[Reg::DB_Z_INFO], r[Reg::CB_BLEND0_CONTROL],
            r[Reg::SPI_SHADER_COL_FORMAT], r[Reg::DB_ALPHA_TO_MASK]);
    DC_INFO("gpu:   index base %#llx offset %u size %u instances %u first_vertex %u indirect %#llx",
            (unsigned long long)dc.index_base, dc.index_offset, dc.index_size, dc.instances,
            dc.first_vertex, (unsigned long long)dc.indirect_args);
    // Geometry-shader pipeline (VGT_SHADER_STAGES_EN.GS_EN): ES and GS are not
    // run, only the copy VS. Log what emulating them needs and dump both
    // programs next to the other shaders.
    if (r[Reg::VGT_SHADER_STAGES_EN] & 0x20) {
      const u64 es = r.ProgramAddr(Reg::SPI_SHADER_PGM_LO_ES);
      const u64 gs = r.ProgramAddr(Reg::SPI_SHADER_PGM_LO_GS);
      DC_INFO("gpu:   gs pipeline: stages_en=%#x gs_mode=%#x es=%#llx rsrc %#x %#x gs=%#llx rsrc %#x %#x "
              "max_vert_out=%u vert_itemsize=%u,%u,%u,%u esgs_itemsize=%u gsvs_itemsize=%u "
              "gsvs_offset=%u,%u,%u out_prim=%#x instance_cnt=%#x",
              r[Reg::VGT_SHADER_STAGES_EN], r[Reg::VGT_GS_MODE], es,
              r[Reg::SPI_SHADER_PGM_RSRC1_ES], r[Reg::SPI_SHADER_PGM_RSRC2_ES], gs,
              r[Reg::SPI_SHADER_PGM_RSRC1_GS], r[Reg::SPI_SHADER_PGM_RSRC2_GS],
              r[Reg::VGT_GS_MAX_VERT_OUT], r[Reg::VGT_GS_VERT_ITEMSIZE],
              r[Reg::VGT_GS_VERT_ITEMSIZE + 1], r[Reg::VGT_GS_VERT_ITEMSIZE + 2],
              r[Reg::VGT_GS_VERT_ITEMSIZE + 3], r[Reg::VGT_ESGS_RING_ITEMSIZE],
              r[Reg::VGT_GSVS_RING_ITEMSIZE], r[Reg::VGT_GSVS_RING_OFFSET_1],
              r[Reg::VGT_GSVS_RING_OFFSET_1 + 1], r[Reg::VGT_GSVS_RING_OFFSET_1 + 2],
              r[Reg::VGT_GS_OUT_PRIM_TYPE], r[Reg::VGT_GS_INSTANCE_CNT]);
      for (const auto &[name, ud0] : {std::pair{"es", Reg::SPI_SHADER_USER_DATA_ES_0},
                                      std::pair{"gs", Reg::SPI_SHADER_USER_DATA_GS_0}}) {
        std::string w;
        for (u32 k = 0; k < 16; ++k) {
          char b[12];
          std::snprintf(b, sizeof(b), " %08x", r[ud0 + k]);
          w += b;
        }
        DC_INFO("gpu:   gs pipeline: %s user data%s", name, w.c_str());
      }
      for (const auto &[name, addr] : {std::pair{"es", es}, std::pair{"gs", gs}}) {
        const auto *code = reinterpret_cast<const u32 *>(addr);
        if (dump_dir && addr && Readable(addr, 4)) {
          char path[512];
          std::snprintf(path, sizeof(path), "%s/%s_%llx.bin", dump_dir, name, addr);
          std::ofstream(path, std::ios::binary)
              .write(reinterpret_cast<const char *>(code), Shader::Gcn::ProgramSize(code));
        }
      }
    }
    // Tessellation pipeline (HS_EN): not drawn yet. Log what emulating it needs.
    if (r[Reg::VGT_SHADER_STAGES_EN] & 0x4) {
      const u64 ls = r.ProgramAddr(Reg::SPI_SHADER_PGM_LO_LS);
      const u64 hs = r.ProgramAddr(Reg::SPI_SHADER_PGM_LO_HS);
      const u32 config = r[Reg::VGT_LS_HS_CONFIG];
      DC_INFO("gpu:   tess pipeline: ls=%#llx rsrc %#x %#x hs=%#llx rsrc %#x %#x vs rsrc2 %#x "
              "config=%#x (patches %u cp in %u out %u) tf_param=%#x hos min %#x max %#x",
              ls, r[Reg::SPI_SHADER_PGM_RSRC1_LS], r[Reg::SPI_SHADER_PGM_RSRC2_LS], hs,
              r[Reg::SPI_SHADER_PGM_RSRC1_HS], r[Reg::SPI_SHADER_PGM_RSRC2_HS],
              r[Reg::SPI_SHADER_PGM_RSRC2_VS], config, config & 0xFF, (config >> 8) & 0x3F,
              (config >> 14) & 0x3F, r[Reg::VGT_TF_PARAM], r[Reg::VGT_HOS_MIN_TESS_LEVEL],
              r[Reg::VGT_HOS_MAX_TESS_LEVEL]);
      DC_INFO("gpu:   tess pipeline: tf_ring_size %#x hs_offchip %#x tf_mem_base %#x "
              "vs rsrc1 %#x vgt_hos_cntl %#x",
              r[0xC24E], r[0xC24F], r[0xC250], r[Reg::SPI_SHADER_PGM_RSRC1_VS], r[0xA285]);
      for (const auto &[name, ud0] : {std::pair{"ls", Reg::SPI_SHADER_USER_DATA_LS_0},
                                      std::pair{"hs", Reg::SPI_SHADER_USER_DATA_HS_0},
                                      std::pair{"vs", Reg::SPI_SHADER_USER_DATA_VS_0}}) {
        std::string w;
        for (u32 k = 0; k < 16; ++k) {
          char b[12];
          std::snprintf(b, sizeof(b), " %08x", r[ud0 + k]);
          w += b;
        }
        DC_INFO("gpu:   tess pipeline: %s user data%s", name, w.c_str());
      }
      for (const auto &[name, addr] : {std::pair{"ls", ls}, std::pair{"hs", hs}}) {
        const auto *code = reinterpret_cast<const u32 *>(addr);
        if (dump_dir && addr && Readable(addr, 4)) {
          char path[512];
          std::snprintf(path, sizeof(path), "%s/%s_%llx.bin", dump_dir, name, addr);
          std::ofstream(path, std::ios::binary)
              .write(reinterpret_cast<const char *>(code), Shader::Gcn::ProgramSize(code));
        }
      }
    }
    if (r.ColorTarget(0, Reg::CB_VIEW)) {
      DC_INFO("gpu:   cb0 view=%#x attrib=%#x pitch=%#x slice=%#x",
              r.ColorTarget(0, Reg::CB_VIEW), r.ColorTarget(0, Reg::CB_ATTRIB),
              r.ColorTarget(0, Reg::CB_PITCH), r.ColorTarget(0, Reg::CB_SLICE));
    }
  }
  if (!vs_addr) {
    return;
  }
  if ((r[Reg::PA_CL_CLIP_CNTL] >> 22) &
      1) { // DX_RASTERIZATION_KILL: nothing reaches the rasterizer
    return;
  }
  static const std::vector<u64> skip_ps = [] { // DC_SKIP_PS=<hex>,<hex>,...
    std::vector<u64> v;
    for (const char *p = std::getenv("DC_SKIP_PS"); p && *p;) {
      char *q = nullptr;
      v.push_back(std::strtoull(p, &q, 16));
      p = *q == ',' ? q + 1 : q;
      if (q == p && *p) break;
    }
    return v;
  }();
  if (!skip_ps.empty() && std::find(skip_ps.begin(), skip_ps.end(), ps_addr) !=
                              skip_ps.end()) { // debug: drop these pixel shaders' draws
    static std::atomic<u32> skipped{0};
    if (skipped.fetch_add(1) == 0) {
      DC_INFO("gpu: DC_SKIP_PS: first draw skipped (ps %#llx, %zu shaders listed)", (unsigned long long)ps_addr,
              skip_ps.size());
    }
    return;
  }
  // DC_SKIP_DRAW=<hex ps>:<count>: drop only that shader's draws of that index count.
  static const std::pair<u64, u32> skip_draw = [] {
    std::pair<u64, u32> p{0, 0};
    if (const char *e = std::getenv("DC_SKIP_DRAW")) {
      char *q = nullptr;
      p.first = std::strtoull(e, &q, 16);
      p.second = *q == ':' ? u32(std::strtoul(q + 1, nullptr, 10)) : 0;
    }
    return p;
  }();
  if (skip_draw.first && ps_addr == skip_draw.first && dc.count == skip_draw.second) {
    return;
  }

  // ---- attachments ----
  DrawLap(1);
  if (cb_mode == 2 || cb_mode == 5) { // fast-clear eliminate / FMASK decompress
    EliminateFastClear(r);
    return;
  }
  if (cb_mode == 3) { // CB_RESOLVE
    ResolveColor(r);
    return;
  }
  RenderTarget *colors[8] = {};
  MTLPixelFormat color_fmt[8] = {};
  OutType out_type[8] = {};
  u32 col_format = ps_addr ? r[Reg::SPI_SHADER_COL_FORMAT] : 0;
  const u32 target_mask = r[Reg::CB_TARGET_MASK];
  bool any_color = false;
  if (cb_mode == 1 && ps_addr) {
    for (u32 k = 0; k < kNumColorTargets; ++k) {
      const u32 info = r.ColorTarget(k, Reg::CB_INFO);
      const u64 base = u64(r.ColorTarget(k, Reg::CB_BASE)) << 8;
      const u32 fmt = ColorInfo::Format(info);
      if (!fmt || !base || !((target_mask >> (4 * k)) & 0xF) ||
          !((col_format >> (4 * k)) & 0xF)) {
        col_format &= ~(0xFu << (4 * k));
        continue;
      }
      const ColorFormatInfo cf =
          ColorTargetFormat(fmt, ColorInfo::NumberType(info));
      colors[k] = &ColorTargetFor(r, k);
      if (colors[k]->comp_swap != ColorInfo::CompSwap(info)) {
        colors[k]->comp_swap = ColorInfo::CompSwap(info);
        colors[k]->three_channels = fmt == 6 || fmt == 7;
        ++texture_epoch; // T# selectors over it change meaning
      }
      color_fmt[k] = cf.format;
      out_type[k] = cf.out;
      any_color = true;
    }
  } else {
    col_format = 0;
  }
  // Dual-source blending: CB blends MRT0 with the MRT1 export as Source1.
  // Metal takes that second source as color(0) index(1), with no attachment 1.
  const u32 blend0 = colors[0] ? r[Reg::CB_BLEND0_CONTROL] : 0;
  auto source1 = [](u32 f) { return f >= 15 && f <= 18; };
  const bool dual_source =
      ((blend0 >> 30) & 1) &&
      (source1(blend0 & 0x1F) || source1((blend0 >> 8) & 0x1F) ||
       (((blend0 >> 29) & 1) &&
        (source1((blend0 >> 16) & 0x1F) || source1((blend0 >> 24) & 0x1F))));
  if (dual_source) {
    u32 f1 = (r[Reg::SPI_SHADER_COL_FORMAT] >> 4) & 0xF;
    if (!f1) {
      f1 = col_format & 0xF;
    }
    col_format = (col_format & ~0xF0u) | (f1 << 4);
    colors[1] = nullptr;
    color_fmt[1] = MTLPixelFormatInvalid;
    out_type[1] = out_type[0];
  }

  RenderTarget *depth = nullptr;
  const u32 z_info = r[Reg::DB_Z_INFO];
  const u32 zfmt = z_info & 3;
  const bool depth_clear = db_render & 1;
  const bool stencil_clear = (db_render >> 1) & 1;
  // shadPS4 StencilValid(): a stencil format and a stencil address.
  const bool stencil_valid =
      (r[Reg::DB_STENCIL_INFO] & 1) && r[Reg::DB_STENCIL_WRITE_BASE];
  const bool z_used = ((db_control >> 1) & 1) || depth_clear ||
                      (stencil_valid && ((db_control & 1) || stencil_clear));
  MTLPixelFormat depth_fmt = MTLPixelFormatInvalid;
  if (zfmt && z_used) {
    const u64 zbase = u64(r[Reg::DB_Z_WRITE_BASE]) << 8;
    const u32 size = r[Reg::DB_DEPTH_SIZE];
    u32 w = ((size & 0x7FF) + 1) * 8, h = (((size >> 11) & 0x7FF) + 1) * 8;
    if (db_extent) {
      w = db_extent & 0xFFFF;
      h = db_extent >> 16;
    }
    depth_fmt = stencil_valid ? MTLPixelFormatDepth32Float_Stencil8
                : zfmt == 1   ? MTLPixelFormatDepth16Unorm
                              : MTLPixelFormatDepth32Float;
    if (zbase && w && h) {
      // DB_DEPTH_VIEW: like CB_COLOR_VIEW, a cube or a set of shadow cascades.
      const u32 first = r[Reg::DB_DEPTH_VIEW] & 0x7FF,
                last = (r[Reg::DB_DEPTH_VIEW] >> 13) & 0x7FF;
      bool array = first || last;
      if (!array) {
        if (RenderTarget *a = Latest(depth_targets, zbase, TexDim::Tex2DArray)) {
          array = a->width == w && a->height == h && a->tex.pixelFormat == depth_fmt;
        }
      }
      depth = array ? &GetTarget(depth_targets, zbase, depth_fmt, w, h, true, 1,
                                 TexDim::Tex2DArray, std::max(first, last) + 1)
                    : &GetDepthTarget(zbase, depth_fmt, w, h);
      depth->bind_slice = array ? first : 0;
      zstat_draws += zbase == zstat_addr;
      DepthFromColor(*depth);
      depth->clear_depth = r.F(Reg::DB_DEPTH_CLEAR);
      depth->clear_stencil = r[Reg::DB_STENCIL_CLEAR] & 0xFF;
      if (ZInfo::TileSurface(z_info)) {
        NoteMeta(u64(r[Reg::DB_HTILE_DATA_BASE]) << 8, zbase, true, w, h);
      }
    } else {
      depth_fmt = MTLPixelFormatInvalid;
    }
  }
  if (!any_color && !depth) {
    if (trace)
      DC_INFO("gpu:   skipped: no attachments");
    return;
  }

  // PA_CL_VS_OUT_CNTL.USE_VTX_RENDER_TARGET_INDX: the VS picks the slice of the
  // view, a cube face for instance. Metal needs every attachment layered.
  struct LayerScope {
    u32 &v;
    ~LayerScope() { v = 0; }
  } layer_scope{draw_layers};
  if ((r[Reg::PA_CL_VS_OUT_CNTL] >> 18) & 1) {
    u32 layers = ~0u;
    for (u32 k = 0; k < 8; ++k) {
      if (!colors[k]) {
        continue;
      }
      const u32 view = r.ColorTarget(k, Reg::CB_VIEW);
      const u32 first = view & 0x7FF, last = (view >> 13) & 0x7FF;
      layers = colors[k]->tex.textureType == MTLTextureType2DArray && !first
                   ? std::min<u32>({layers, last + 1,
                                    u32(colors[k]->tex.arrayLength)})
                   : 0;
    }
    if (depth) {
      const u32 first = r[Reg::DB_DEPTH_VIEW] & 0x7FF,
                last = (r[Reg::DB_DEPTH_VIEW] >> 13) & 0x7FF;
      layers = depth->tex.textureType == MTLTextureType2DArray && !first
                   ? std::min<u32>({layers, last + 1, u32(depth->tex.arrayLength)})
                   : 0;
    }
    draw_layers = layers != ~0u && layers > 1 ? layers : 0;
  }
  PassState want;
  want.layers = draw_layers;
  for (u32 k = 0; k < 8; ++k) {
    want.color[k] = reinterpret_cast<u64>(colors[k]);
    want.slice[k] =
        colors[k] ? colors[k]->bind_slice | (colors[k]->bind_level << 24) : 0;
  }
  want.depth = reinterpret_cast<u64>(depth);
  want.depth_slice = depth ? depth->bind_slice : 0;
  if ((depth_clear || (stencil_clear && stencil_valid)) && depth) {
    // The draw only exists to clear depth and/or stencil.
    depth->bounds_seq = 0;
    if (!depth_clear) {
      // Stencil only: keep depth, so end any pass and reload it.
      BeginPass(colors, depth, false, 1.0f, true,
                r[Reg::DB_STENCIL_CLEAR] & 0xFF);
    } else {
      BeginPass(colors, depth, true, r.F(Reg::DB_DEPTH_CLEAR), stencil_clear,
                r[Reg::DB_STENCIL_CLEAR] & 0xFF);
    }
    return;
  }
  // ---- shaders ----
  DrawLap(2);
  const u32 *vs_ud = r.UserData(Reg::SPI_SHADER_USER_DATA_VS_0);
  const u32 *ps_ud = r.UserData(Reg::SPI_SHADER_USER_DATA_PS_0);
  // DC_PEEK_VS=<vs>: in the traced frame, the V#s of that vertex shader's
  // resource table (user data 12-13) and how many of each buffer's records
  // have a FLT_MAX or non-finite dword among their first three.
  if (trace) {
    FindRefs(vs_ud, 16, 0, "ud", "vs", vs_addr);
    FindRefs(ps_ud, 16, 0, "ud", "ps", ps_addr);
  }
  if (trace && peek_vs && vs_addr == peek_vs) {
    const u64 table = (u64(vs_ud[12]) | (u64(vs_ud[13]) << 32)) & kGpuAddrMask;
    for (u32 e = 0; e < 8 && table; ++e) {
      const u32 *v = reinterpret_cast<const u32 *>(table + e * 16);
      const u64 base = (u64(v[0]) | (u64(v[1] & 0xFFF) << 32)) & kGpuAddrMask;
      const u32 stride = (v[1] >> 16) & 0x3FFF, num = v[2];
      u32 big = 0, first = ~0u;
      if (base && stride >= 12 && stride <= 256 && num < (1u << 20)) {
        for (u32 n = 0; n < num; ++n) {
          const float *f = reinterpret_cast<const float *>(base + u64(n) * stride);
          if (!(std::fabs(f[0]) < 1e30f) || !(std::fabs(f[1]) < 1e30f) ||
              !(std::fabs(f[2]) < 1e30f)) {
            big += 1;
            first = std::min(first, n);
          }
        }
      }
      DC_INFO("gpu: peek vs %#llx table %#llx [%u] = %08x %08x %08x %08x: base %#llx "
              "stride %u records %u, %u big (first %d)",
              (unsigned long long)vs_addr, (unsigned long long)table, e, v[0], v[1], v[2], v[3],
              (unsigned long long)base, stride, num, big, first == ~0u ? -1 : int(first));
      if (rain_diag && !find_addr && num && big == num) {
        find_addr = base;
        trace_frame = frame_count + 1;
        DC_INFO("gpu: rain diag: %#llx is all FLT_MAX, finding its users in flip %llu",
                (unsigned long long)base, (unsigned long long)trace_frame);
      }
    }
  }
  Program *vs = GetProgram(Stage::Vertex, vs_addr, vs_ud,
                           Bits(r[Reg::SPI_SHADER_PGM_RSRC2_VS], 1, 5),
                           Bits(r[Reg::SPI_SHADER_PGM_RSRC1_VS], 24, 2));
  Program *ps = ps_addr && any_color
                    ? GetProgram(Stage::Fragment, ps_addr, ps_ud,
                                 Bits(r[Reg::SPI_SHADER_PGM_RSRC2_PS], 1, 5), 0)
                    : nullptr;
  if (!vs || (ps_addr && any_color && !ps)) {
    if (trace)
      DC_INFO("gpu:   skipped: shader decode failed");
    return;
  }

  if (trace) { // DC_PEEK_PS=<ps>: its first two V# buffers, printed after the
               // traced frame
    static const u64 peek_ps =
        std::getenv("DC_PEEK_PS")
            ? std::strtoull(std::getenv("DC_PEEK_PS"), nullptr, 16)
            : 0;
    if (ps_addr == peek_ps) {
      for (u32 k = 0; k < 8; k += 4) {
        peek_addrs.push_back(
            (u64(ps_ud[k]) | (u64(ps_ud[k + 1] & 0xFFF) << 32)) & kGpuAddrMask);
        peek_addrs.push_back(peek_addrs.back() + 64);
      }
      // DC_PEEK_PS_AT=<ud>:<byte offset>,...: 8 floats at V#(ud)+off, read when
      // the draw is recorded
      if (const char *at = std::getenv("DC_PEEK_PS_AT")) {
        for (const char *q = at; *q;) {
          char *e = nullptr;
          const u32 n = std::strtoul(q, &e, 10);
          const u32 off = *e == ':' ? std::strtoul(e + 1, &e, 16) : 0;
          const u64 a =
              ((u64(ps_ud[n & 15]) | (u64(ps_ud[(n + 1) & 15] & 0xFFF) << 32)) &
               kGpuAddrMask) +
              off;
          if (Readable(a, 32)) {
            const float *v = reinterpret_cast<const float *>(a);
            DC_INFO("gpu:   peek ud%u+%#x: %g %g %g %g %g %g %g %g", n, off,
                    v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
          }
          q = *e ? e + 1 : e;
        }
      }
    }
  }
  // ---- resources ----
  DrawLap(3);
  std::vector<TextureType> &vs_types = draw_vs_types;
  std::vector<TextureType> &ps_types = draw_ps_types;
  Resources &vs_res = draw_vs_res, &ps_res = draw_ps_res;
  resolving_shader = ps_addr;
  {
  CpuTimer t(cpu_profile ? &cpu_resolve_s : nullptr);
  Resolve(*vs, vs_ud, vs_types, vs_res);
  if (ps) {
    Resolve(*ps, ps_ud, ps_types, ps_res);
  }
  }

  bool pending_clear = depth && depth->clear_pending;
  for (u32 k = 0; k < 8; ++k) {
    pending_clear = pending_clear || (colors[k] && colors[k]->clear_pending);
  }
  // Depth bounds test (Metal on AMD has none): the pixel shader compares a copy
  // of the depth buffer, taken when the depth may have changed since the last
  // copy.
  const bool depth_bounds = depth && ps && ((db_control >> 3) & 1);
  if (depth && ((db_control >> 2) & 1)) {
    depth->bounds_seq = 0; // this draw writes depth: an older copy goes stale
  }
  if (depth_bounds &&
      (!depth->bounds_copy || depth->bounds_seq != depth->written ||
       depth->clear_pending)) {
    if (depth->clear_pending ||
        !depth->initialized) { // apply the pending clear before copying
      BeginPass(colors, depth, false, 1.0f);
    }
    EndPass();
    EnsureCommandBuffer();
    if (!depth->bounds_copy) {
      MTLTextureDescriptor *d = [MTLTextureDescriptor
          texture2DDescriptorWithPixelFormat:depth->tex.pixelFormat
                                       width:depth->width
                                      height:depth->height
                                   mipmapped:NO];
      d.usage = MTLTextureUsageShaderRead;
      d.storageMode = MTLStorageModePrivate;
      depth->bounds_copy = [device newTextureWithDescriptor:d];
    }
    id<MTLBlitCommandEncoder> b = [cmd blitCommandEncoder];
    [b copyFromTexture:depth->tex
             sourceSlice:depth->bind_slice
             sourceLevel:0
               toTexture:depth->bounds_copy
        destinationSlice:0
        destinationLevel:0
              sliceCount:1
              levelCount:1];
    [b endEncoding];
    BeginPass(colors, depth, false, 1.0f);
    depth->bounds_seq = depth->written;
  } else if (!enc || !(want == pass) || pending_clear ||
             (pass_draws && (SamplesAttachment(ps_res, colors, depth) ||
                             SamplesAttachment(vs_res, colors, depth)))) {
    BeginPass(colors, depth, false, 1.0f);
    if (depth && depth->bounds_seq && depth_bounds) {
      depth->bounds_seq = depth->written; // a new pass over unchanged depth
    }
  }

  ++pass_draws;

  // ---- functions ----
  DrawLap(4);
  // DC_PROBE_CB=<addr>: with DC_PROBE=*:nan, only draws into that color target.
  static const u64 probe_cb =
      std::getenv("DC_PROBE_CB")
          ? std::strtoull(std::getenv("DC_PROBE_CB"), nullptr, 16)
          : 0;
  const bool probe_target =
      !probe_cb || (u64(r.ColorTarget(0, Reg::CB_BASE)) << 8) == probe_cb;
  const int probe_slot =
      ps && probe_target && probe_buf && trace
          ? ProbeSlot(ps_addr,
                      "draw " + std::to_string(draw_count) + " ps " +
                          Hex(ps_addr) + " cb0 " +
                          Hex(u64(r.ColorTarget(0, Reg::CB_BASE)) << 8))
          : -1;
  const bool rect = prim == PrimType::RectList;
  Shader::MslOptions vo;
  vo.rect_list = rect;
  vo.points = prim == PrimType::PointList;
  vo.vtx_point_size = vo.points && (r[Reg::PA_CL_VS_OUT_CNTL] & 1);
  vo.vtx_layer = draw_layers != 0;
  // DC_DEBUG_VSOUT=1: the first tessellated (domain) VS of the traced frame.
  const bool capture = debug_vsout && !debug_vs_buf &&
                       (debug_vsout == 1 ? tess_draw && trace : vs_addr == debug_vsout && (!tess_draw || trace));
  vo.debug_capture = capture;
  vo.textures = vs_types; // borrowed: lives until the end of this draw
  vo.map_hoist = map_hoist;
  if (tess_draw) {
    vo.tess = tess_draw->info;
  }
  // User data as bound (pointers into CE dumps moved to their snapshots),
  // and the linear bases translated from it once. A commit before binding
  // retires the snapshots, so binding patches again if one happened.
  u32 vud[16], pud[16];
  std::memcpy(vud, vs_ud, sizeof(vud));
  std::memcpy(pud, ps_ud, sizeof(pud));
  PatchUserData(vud);
  if (ps) {
    PatchUserData(pud);
  }
  const u64 patched_at = commits;
  u32 vlin[Shader::kMaxLinearBases], plin[Shader::kMaxLinearBases];
  // A fused copy VS reads its draw's arena copy of the GSVS ring through the
  // biased address: no host-translated bases for it.
  vo.linear_mask = fused_pending.on ? 0 : LinearTable(*vs, vud, vlin, &vo.linear_aligned);
  if (fused_pending.on) {
    vo.linear_aligned = 0;
  }
  // Vertex fetch by Metal for the fetch loads it can do (all or none): the
  // V# words resolved here give each one a vertex buffer and a format.
  // DC_SW_FETCH keeps every load in the shader.
  static const bool sw_fetch = std::getenv("DC_SW_FETCH") != nullptr;
  struct VertexBuffer {
    id<MTLBuffer> buf;
    u64 off;
    MTLVertexFormat format;
    u32 stride, bytes;
  };
  VertexBuffer vbufs[kMaxVertexBuffers];
  u32 nvbufs = 0;
  u64 vlayout = 0;
  if (!sw_fetch && !rect && !tess_draw && !fused_pending.on && !vs->vertex_inputs.empty() &&
      vs->vertex_inputs.size() <= kMaxVertexBuffers) {
    Shader::SgprReadCache rc;
    bool ok = true;
    draw_vattrs.clear();
    for (u32 k = 0; k < vs->vertex_inputs.size(); ++k) {
      const Shader::VertexInput &in = vs->vertex_inputs[k];
      u32 v[4];
      for (u32 j = 0; j < 4; ++j) {
        v[j] = Shader::ResolveSgpr(in.vsharp[j], vs_ud, &rc);
      }
      const u64 base = (u64(v[0]) | (u64(v[1] & 0xFFF) << 32)) + in.offset;
      const u32 stride = (v[1] >> 16) & 0x3FFF;
      const u32 dfmt = in.typed ? in.dfmt : (v[3] >> 15) & 15;
      const u32 nfmt = in.typed ? in.nfmt : (v[3] >> 12) & 7;
      Shader::MslOptions::VertexAttr a;
      a.pc = in.pc;
      a.swizzle = in.typed || (v[3] & 0xFFF) == kIdentitySwizzle ? 0 : (v[3] & 0x7FFF);
      u32 bytes = 0;
      const MTLVertexFormat f = VertexFormat(dfmt, nfmt, &a.kind, &bytes);
      id<MTLBuffer> buf = nil;
      u64 off = 0;
      if (f == MTLVertexFormatInvalid || (base & 3) || (stride & 3) || !v[2] ||
          !Locate(base, stride ? u64(stride) * v[2] : bytes, &buf, &off) || (off & 3)) {
        ok = false;
        break;
      }
      draw_vattrs.push_back(a);
      vbufs[nvbufs++] = {buf, off, f, stride, bytes};
      vlayout = Hash(u64(f) | (u64(stride) << 32), vlayout + 1);
    }
    if (!ok) {
      draw_vattrs.clear();
      nvbufs = 0;
      vlayout = 0;
    }
    vo.vertex_attrs = draw_vattrs;
  }
  u64 vs_key = Hash(vs->hash);
  if (fused_pending.on) { // the copy VS of a GS pipeline, ES and GS fused in
    vo.fused = fused_pending.fused;
    vs_key = Hash(fused_pending.key, vs_key);
  }
  vs_key = Hash(vo.map_hoist, vs_key);
  vs_key = Hash(vo.linear_mask, vs_key);
  vs_key = Hash(vo.linear_aligned, vs_key);
  for (const auto &a : vo.vertex_attrs) {
    vs_key = Hash(u64(a.pc) | (u64(a.kind) << 32) | (u64(a.swizzle) << 40), vs_key);
  }
  vs_key = Hash(rect, vs_key);
  if (tess_draw) {
    vs_key = Fnv(&vo.tess, sizeof(vo.tess), vs_key);
    vs_key = Hash(u64(tess_draw->partition) | u64(tess_draw->winding) << 8, vs_key);
  }
  // A layered pipeline states its topology class, so that goes in the key too.
  const MTLPrimitiveTopologyClass layer_topology =
      prim == PrimType::PointList ? MTLPrimitiveTopologyClassPoint
      : prim == PrimType::LineList || prim == PrimType::LineStrip
          ? MTLPrimitiveTopologyClassLine
          : MTLPrimitiveTopologyClassTriangle;
  vs_key = Hash(u32(vo.points) | (u32(vo.vtx_point_size) << 1) |
                    (u32(vo.vtx_layer) << 2) |
                    (vo.vtx_layer ? u32(layer_topology) << 3 : 0),
                vs_key);
  vs_key = Hash(capture, vs_key);
  vs_key = Fnv(vs_types.data(), vs_types.size() * sizeof(TextureType), vs_key);
  u64 ps_key = 0;
  Shader::MslOptions fo;
  if (ps) {
    for (u32 k = 0; k < 32; ++k) {
      fo.ps_input_cntl[k] = r[Reg::SPI_PS_INPUT_CNTL_0 + k];
    }
    fo.vs_param_mask = vs->param_mask;
    fo.ps_input_addr = r[Reg::SPI_PS_INPUT_ADDR];
    fo.col_format = col_format;
    fo.dual_source = dual_source;
    fo.map_hoist = map_hoist;
    fo.linear_mask = LinearTable(*ps, pud, plin, &fo.linear_aligned);
    for (u32 k = 0; k < 8; ++k) {
      fo.mrt_type[k] = out_type[k];
      if (color_fmt[k] == MTLPixelFormatRG11B10Float) {
        fo.mrt_ufloat |= 1u << k;
      }
    }
    // Only what the shader reads goes into its key: inputs and MRTs it does
    // not use keep values from earlier draws, which made new keys (and
    // pipeline compiles, during which draws are skipped) for the same shader.
    for (u32 k = 0; k < 32; ++k) {
      if (!((ps->attr_mask >> k) & 1)) {
        fo.ps_input_cntl[k] = 0;
      }
    }
    const u32 used_mrt = ps->mrt_mask | (dual_source ? 2u : 0u);
    for (u32 k = 0; k < 8; ++k) {
      if (!((used_mrt >> k) & 1)) {
        fo.col_format &= ~(0xFu << (4 * k));
        fo.mrt_type[k] = Shader::OutType::Float;
        fo.mrt_ufloat &= ~(1u << k);
      }
    }
    fo.textures = ps_types; // borrowed, like vo.textures
    fo.depth_target = depth != nullptr;
    fo.depth_bounds = depth_bounds;
    // DC_NO_A2M=1: ignore DB_ALPHA_TO_MASK (A/B).
    static const bool no_a2m = std::getenv("DC_NO_A2M") != nullptr;
    fo.alpha_to_mask = !no_a2m && (r[Reg::DB_ALPHA_TO_MASK] & 1);
    if (probe_slot >= 0) {
      fo.probe_pc = probe.pc;
      fo.probe_x = probe.x;
      fo.probe_y = probe.y;
    }
    ps_key = Hash(ps->hash);
    ps_key = Hash(fo.probe_pc, ps_key);
    ps_key = Hash(fo.depth_target, ps_key);
    ps_key = Hash(fo.depth_bounds, ps_key);
    ps_key = Hash(fo.alpha_to_mask, ps_key);
    ps_key = Fnv(fo.ps_input_cntl.data(), sizeof(fo.ps_input_cntl), ps_key);
    ps_key = Hash(fo.vs_param_mask, ps_key);
    ps_key = Hash(fo.ps_input_addr, ps_key);
    ps_key = Hash(fo.col_format, ps_key);
    ps_key = Hash(fo.dual_source, ps_key);
    ps_key = Hash(fo.map_hoist, ps_key);
    ps_key = Hash(fo.linear_mask, ps_key);
    ps_key = Hash(fo.linear_aligned, ps_key);
    ps_key = Fnv(fo.mrt_type.data(), sizeof(fo.mrt_type), ps_key);
    ps_key = Hash(fo.mrt_ufloat, ps_key);
    ps_key =
        Fnv(ps_types.data(), ps_types.size() * sizeof(TextureType), ps_key);
  }

  // ---- pipeline ----
  DrawLap(5);
  PipelineKey pk;
  pk.vs = vlayout ? Hash(vlayout, vs_key) : vs_key; // formats and strides live in the pipeline
  pk.ps = ps_key;
  for (u32 k = 0; k < 8; ++k) {
    pk.color_fmt[k] = static_cast<u32>(color_fmt[k]);
    pk.blend[k] = colors[k] ? r[Reg::CB_BLEND0_CONTROL + k] : 0;
  }
  pk.write_mask = target_mask;
  pk.shader_mask = r[Reg::CB_SHADER_MASK];
  pk.depth_fmt = static_cast<u32>(depth_fmt);
  // DC_DEBUG_FLAT_RT=<addr>: magenta PS only for draws into that color target.
  static const u64 flat_rt =
      std::getenv("DC_DEBUG_FLAT_RT") ? std::strtoull(std::getenv("DC_DEBUG_FLAT_RT"), nullptr, 16) : 0;
  const bool flat_draw = debug_fs && (tess_flat ? tess_draw != nullptr
                                      : flat_rt ? (u64(r.ColorTarget(0, Reg::CB_BASE)) << 8) == flat_rt
                                                : true);
  const u64 pkey = flat_draw ? Hash(u64(0xF1A7), Hash(pk)) : Hash(pk);
  TakeCompiled();
  auto pit = pipelines.find(pkey);
  if (pit == pipelines.end()) {
    if (!pending.count(pkey)) {
      MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
      if (vo.vtx_layer) {
        d.inputPrimitiveTopology = layer_topology;
      }
      if (tess_draw) {
        d.tessellationPartitionMode = tess_draw->partition;
        d.maxTessellationFactor = 64;
        d.tessellationFactorFormat = MTLTessellationFactorFormatHalf;
        d.tessellationControlPointIndexType = MTLTessellationControlPointIndexTypeNone;
        d.tessellationFactorStepFunction = MTLTessellationFactorStepFunctionPerPatch;
        d.tessellationOutputWindingOrder = tess_draw->winding;
        d.tessellationFactorScaleEnabled = NO;
      }
      if (nvbufs) {
        MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
        for (u32 k = 0; k < nvbufs; ++k) {
          const VertexBuffer &vb = vbufs[k];
          vd.attributes[k].format = vb.format;
          vd.attributes[k].offset = 0;
          vd.attributes[k].bufferIndex = kVertexBufferBase + k;
          MTLVertexBufferLayoutDescriptor *l = vd.layouts[kVertexBufferBase + k];
          l.stride = vb.stride ? vb.stride : (vb.bytes + 3) & ~3u;
          l.stepFunction = vb.stride ? MTLVertexStepFunctionPerVertex
                                     : MTLVertexStepFunctionConstant;
          l.stepRate = vb.stride ? 1 : 0;
        }
        d.vertexDescriptor = vd;
      }
      for (u32 k = 0; k < 8; ++k) {
        if (!colors[k]) {
          continue;
        }
        MTLRenderPipelineColorAttachmentDescriptor *ca = d.colorAttachments[k];
        ca.pixelFormat = color_fmt[k];
        const u32 b = pk.blend[k];
        const u32 wm = (target_mask >> (4 * k)) & 0xF;
        ca.writeMask = ((wm & 1) ? MTLColorWriteMaskRed : 0) |
                       ((wm & 2) ? MTLColorWriteMaskGreen : 0) |
                       ((wm & 4) ? MTLColorWriteMaskBlue : 0) |
                       ((wm & 8) ? MTLColorWriteMaskAlpha : 0);
        const bool blendable = out_type[k] == OutType::Float &&
                               color_fmt[k] != MTLPixelFormatRGBA32Float;
        // CB_SHADER_MASK without alpha: the hardware blends with source alpha 1
        // (shadPS4 swaps SrcAlpha factors for One/Zero).
        const bool alpha_masked =
            !((r[Reg::CB_SHADER_MASK] >> (4 * k + 3)) & 1);
        if ((b >> 30) & 1 && blendable && !flat_draw) {
          ca.blendingEnabled = YES;
          auto fixa = [&](u32 f) {
            return !alpha_masked ? BlendFactor(f)
                   : f == 4      ? MTLBlendFactorOne
                   : f == 5      ? MTLBlendFactorZero
                                 : BlendFactor(f);
          };
          ca.sourceRGBBlendFactor = fixa(b & 0x1F);
          ca.rgbBlendOperation = BlendOp((b >> 5) & 7);
          ca.destinationRGBBlendFactor = fixa((b >> 8) & 0x1F);
          if ((b >> 29) & 1) {
            ca.sourceAlphaBlendFactor = BlendFactor((b >> 16) & 0x1F);
            ca.alphaBlendOperation = BlendOp((b >> 21) & 7);
            ca.destinationAlphaBlendFactor = BlendFactor((b >> 24) & 0x1F);
          } else {
            ca.sourceAlphaBlendFactor = ca.sourceRGBBlendFactor;
            ca.alphaBlendOperation = ca.rgbBlendOperation;
            ca.destinationAlphaBlendFactor = ca.destinationRGBBlendFactor;
          }
        }
      }
      d.depthAttachmentPixelFormat = depth_fmt;
      if (depth_fmt == MTLPixelFormatDepth32Float_Stencil8) {
        d.stencilAttachmentPixelFormat = depth_fmt;
      }
      pending.insert(pkey);
      std::string vsrc = MslSource(*vs, vo, vs_key);
      std::string fsrc = ps ? MslSource(*ps, fo, ps_key) : std::string();
      const bool flat = flat_draw && ps && out_type[0] == OutType::Float;
      const bool will_wait = sync_shaders || probe_slot >= 0 || capture;
      compiler->Submit([this, d, vsrc = std::move(vsrc), fsrc = std::move(fsrc),
                        vs_key, ps_key, pkey, flat, vs_addr, ps_addr] {
        id<MTLRenderPipelineState> pso = nil;
        id<MTLFunction> vfn = compiler->Function(vs_key, vsrc, vs_addr);
        id<MTLFunction> ffn =
            fsrc.empty() ? nil : compiler->Function(ps_key, fsrc, ps_addr);
        if (vfn && (fsrc.empty() || ffn)) {
          d.vertexFunction = vfn;
          d.fragmentFunction = flat ? debug_fs : ffn;
          NSError *err = nil;
          const auto t0 = std::chrono::steady_clock::now();
          pso = [device newRenderPipelineStateWithDescriptor:d error:&err];
          ReportSlowCompile("pipeline", vs_addr, ps_addr, t0);
          if (!pso) {
            DC_ERROR("gpu: pipeline (vs %#llx ps %#llx): %s", vs_addr, ps_addr,
                     err ? err.localizedDescription.UTF8String : "?");
          }
        }
        {
          std::scoped_lock lk(done_lock);
          done_render.emplace_back(pkey, pso);
        }
        done_cv.notify_all();
      }, will_wait);
    }
    // Probed draws must run, and so must fused GS draws (text glyphs: each
    // ES+GS+VS combination is its own pipeline, skipping it loses letters).
    // A DC_DEBUG_VSOUT capture is a one-off pipeline: wait for it too.
    if (!sync_shaders && probe_slot < 0 && !fused_pending.on && !capture) {
      if (trace)
        DC_INFO("gpu:   skipped: pipeline compiling");
      return;
    }
    WaitCompiled(pkey);
    pit = pipelines.find(pkey);
  }
  id<MTLRenderPipelineState> pso = pit->second;
  if (!pso) {
    if (trace)
      DC_INFO("gpu:   skipped: no pipeline");
    return;
  }

  // ---- fixed function ----
  DrawLap(6);
  Rect sc{0, 0, static_cast<s32>(pass_width), static_cast<s32>(pass_height)};
  sc = Intersect(sc, ScissorReg(r[Reg::PA_SC_SCREEN_SCISSOR_TL],
                                r[Reg::PA_SC_SCREEN_SCISSOR_BR]));
  sc = Intersect(sc, ScissorReg(r[Reg::PA_SC_WINDOW_SCISSOR_TL],
                                r[Reg::PA_SC_WINDOW_SCISSOR_BR]));
  sc = Intersect(sc, ScissorReg(r[Reg::PA_SC_GENERIC_SCISSOR_TL],
                                r[Reg::PA_SC_GENERIC_SCISSOR_BR]));
  sc = Intersect(sc, ScissorReg(r[Reg::PA_SC_VPORT_SCISSOR_0_TL],
                                r[Reg::PA_SC_VPORT_SCISSOR_0_TL + 1]));
  if (trace) {
    DC_INFO("gpu:   vte=%#x vp=[%g %g | %g %g | %g %g] scissor=(%d,%d)-(%d,%d) "
            "[scr %#x %#x win %#x %#x gen %#x %#x vp %#x %#x mode0 %#x] "
            "pass=%ux%u clip=%#x su=%#x",
            r[Reg::PA_CL_VTE_CNTL], r.F(Reg::PA_CL_VPORT_XSCALE),
            r.F(Reg::PA_CL_VPORT_XSCALE + 1), r.F(Reg::PA_CL_VPORT_XSCALE + 2),
            r.F(Reg::PA_CL_VPORT_XSCALE + 3), r.F(Reg::PA_CL_VPORT_XSCALE + 4),
            r.F(Reg::PA_CL_VPORT_XSCALE + 5), sc.x0, sc.y0, sc.x1, sc.y1,
            r[Reg::PA_SC_SCREEN_SCISSOR_TL], r[Reg::PA_SC_SCREEN_SCISSOR_BR],
            r[Reg::PA_SC_WINDOW_SCISSOR_TL], r[Reg::PA_SC_WINDOW_SCISSOR_BR],
            r[Reg::PA_SC_GENERIC_SCISSOR_TL], r[Reg::PA_SC_GENERIC_SCISSOR_BR],
            r[Reg::PA_SC_VPORT_SCISSOR_0_TL],
            r[Reg::PA_SC_VPORT_SCISSOR_0_TL + 1], r[0xA292], pass_width,
            pass_height, r[Reg::PA_CL_CLIP_CNTL], r[Reg::PA_SU_SC_MODE_CNTL]);
  }
  if (sc.x1 <= sc.x0 || sc.y1 <= sc.y0) {
    return;
  }
  // The ES and GS of this copy-VS draw, in the pass just opened (RingReq).
  if (!ring_pending.empty()) {
    const bool ok = RunRingPending(colors, color_fmt, depth_fmt);
    ring_pending.clear();
    if (!ok) {
      if (trace)
        DC_INFO("gpu:   skipped: ring vertex pipeline");
      return;
    }
  }

  // Each set below is skipped when `enc` already has that value (see
  // BoundState): ~10 Objective-C calls per draw otherwise.
  const bool cache = state_cache;
  if (!cache || bound.pso != pso) {
    bound.pso = pso;
    [enc setRenderPipelineState:pso];
  }
  if (!cache || !bound.has_viewport || bound.viewport[0] != pass_width ||
      bound.viewport[1] != pass_height) {
    bound.has_viewport = true;
    bound.viewport[0] = pass_width;
    bound.viewport[1] = pass_height;
    [enc setViewport:MTLViewport{0, 0, static_cast<double>(pass_width),
                                 static_cast<double>(pass_height), 0, 1}];
  }
  const MTLScissorRect scissor{static_cast<NSUInteger>(sc.x0),
                               static_cast<NSUInteger>(sc.y0),
                               static_cast<NSUInteger>(sc.x1 - sc.x0),
                               static_cast<NSUInteger>(sc.y1 - sc.y0)};
  if (!cache || !bound.has_scissor ||
      std::memcmp(&bound.scissor, &scissor, sizeof(scissor))) {
    bound.has_scissor = true;
    bound.scissor = scissor;
    [enc setScissorRect:scissor];
  }
  if (depth) {
    const bool st = HasStencil(depth->tex);
    const u32 st_ops = r[Reg::DB_STENCIL_CONTROL];
    const u32 ref_f = r[Reg::DB_STENCILREFMASK],
              ref_b = r[Reg::DB_STENCILREFMASK_BF];
    // DC_NO_STENCIL=1: stencil test off for every draw (debugging).
    static const bool no_stencil = std::getenv("DC_NO_STENCIL") != nullptr;
    id<MTLDepthStencilState> ds = GetDepthState(
        no_stencil ? db_control & ~1u : db_control, depth_clear, st, st_ops, ref_f, ref_b, stencil_clear);
    if (!cache || bound.depth_state != ds) {
      bound.depth_state = ds;
      [enc setDepthStencilState:ds];
    }
    if (st && (db_control & 1)) {
      const bool backface = (db_control >> 7) & 1;
      const u32 front_ref = StencilRef(st_ops, ref_f);
      const u32 back_ref =
          backface ? StencilRef(st_ops >> 12, ref_b) : front_ref;
      if (!cache || !bound.has_stencil_ref ||
          bound.stencil_ref[0] != front_ref ||
          bound.stencil_ref[1] != back_ref) {
        bound.has_stencil_ref = true;
        bound.stencil_ref[0] = front_ref;
        bound.stencil_ref[1] = back_ref;
        [enc setStencilFrontReferenceValue:front_ref
                        backReferenceValue:back_ref];
      }
    }
  }
  // PA_CL_CLIP_CNTL.ZCLIP_NEAR/FAR_DISABLE: depth is clamped instead of clipped
  // (shadPS4 ZclipEnable(): clipping only when both are enabled).
  const u32 clip_cntl = r[Reg::PA_CL_CLIP_CNTL];
  const bool zclip = !((clip_cntl >> 26) & 1) && !((clip_cntl >> 27) & 1);
  const MTLDepthClipMode clip_mode =
      zclip ? MTLDepthClipModeClip : MTLDepthClipModeClamp;
  if (!cache || bound.clip != int(clip_mode)) {
    bound.clip = int(clip_mode);
    [enc setDepthClipMode:clip_mode];
  }
  const u32 su = r[Reg::PA_SU_SC_MODE_CNTL];
  static const bool no_cull = std::getenv("DC_NO_CULL") != nullptr;
  // shadPS4 IsPrimitiveCulled: only triangle-type primitives are culled (not
  // rect lists, points or lines).
  const bool cullable = prim == PrimType::TriList ||
                        prim == PrimType::TriStrip ||
                        prim == PrimType::TriFan ||
                        prim == PrimType::QuadList ||
                        prim == PrimType::Polygon;
  const bool cull_front = (su & 1) && !no_cull && cullable,
             cull_back = ((su >> 1) & 1) && !no_cull && cullable;
  const MTLCullMode cull_mode = (cull_front && cull_back) ? MTLCullModeNone
                                : cull_front              ? MTLCullModeFront
                                : cull_back               ? MTLCullModeBack
                                                          : MTLCullModeNone;
  if (!cache || bound.cull != int(cull_mode)) {
    bound.cull = int(cull_mode);
    [enc setCullMode:cull_mode];
  }
  const MTLWinding winding = ((su >> 2) & 1) ? MTLWindingClockwise
                                             : MTLWindingCounterClockwise;
  if (!cache || bound.winding != int(winding)) {
    bound.winding = int(winding);
    [enc setFrontFacingWinding:winding];
  }
  if (depth) {
    // Polygon offset (shadow maps rely on it): POLY_OFFSET_FRONT/BACK_ENABLE.
    const bool front = (su >> 11) & 1, back = (su >> 12) & 1;
    float bias[3] = {0, 0, 0}; // depth bias, slope scale, clamp
    if (front || back) {
      const u32 base = Reg::PA_SU_POLY_OFFSET_FRONT_SCALE + (front ? 0 : 2);
      bias[0] = r.F(base + 1);
      bias[1] = r.F(base) / 16.0f;
      bias[2] = r.F(Reg::PA_SU_POLY_OFFSET_CLAMP);
    }
    if (!cache || !bound.has_depth_bias ||
        std::memcmp(bound.depth_bias, bias, sizeof(bias))) {
      bound.has_depth_bias = true;
      std::memcpy(bound.depth_bias, bias, sizeof(bias));
      [enc setDepthBias:bias[0] slopeScale:bias[1] clamp:bias[2]];
    }
  }
  const float blend[4] = {
      r.F(Reg::CB_BLEND_RED), r.F(Reg::CB_BLEND_RED + 1),
      r.F(Reg::CB_BLEND_RED + 2), r.F(Reg::CB_BLEND_RED + 3)};
  if (!cache || !bound.has_blend ||
      std::memcmp(bound.blend, blend, sizeof(blend))) {
    bound.has_blend = true;
    std::memcpy(bound.blend, blend, sizeof(blend));
    [enc setBlendColorRed:blend[0] green:blend[1] blue:blend[2] alpha:blend[3]];
  }

  Shader::GpuDrawConsts consts{};
  const u32 vte = r[Reg::PA_CL_VTE_CNTL];
  const float xs = r.F(Reg::PA_CL_VPORT_XSCALE),
              xo = r.F(Reg::PA_CL_VPORT_XSCALE + 1);
  const float ys = r.F(Reg::PA_CL_VPORT_XSCALE + 2),
              yo = r.F(Reg::PA_CL_VPORT_XSCALE + 3);
  const float zs = r.F(Reg::PA_CL_VPORT_XSCALE + 4),
              zo = r.F(Reg::PA_CL_VPORT_XSCALE + 5);
  consts.vp_scale[0] = (vte & 1) ? xs : 1.0f;
  consts.vp_offset[0] = (vte & 2) ? xo : 0.0f;
  consts.vp_scale[1] = (vte & 4) ? ys : 1.0f;
  consts.vp_offset[1] = (vte & 8) ? yo : 0.0f;
  consts.vp_scale[2] = (vte & 16) ? zs : 1.0f;
  consts.vp_offset[2] = (vte & 32) ? zo : 0.0f;
  consts.rt_scale[0] = 2.0f / static_cast<float>(pass_width);
  consts.rt_scale[1] = 2.0f / static_cast<float>(pass_height);
  consts.vte = vte;
  if (fused_pending.on) {
    consts.esgs_base = fused_pending.esgs_base;
    consts.esgs_bias = fused_pending.esgs_bias;
    consts.gsvs_base = fused_pending.gsvs_base;
    consts.gsvs_bias = fused_pending.gsvs_bias;
  }

  // ---- bindings ----
  DrawLap(7);
  static const char *memtest = std::getenv("DC_DEBUG_MEMTEST");
  if (memtest) {
    const u64 a = std::strtoull(memtest, nullptr, 16);
    MemTest(a ? a : vs_addr);
  }
  if (commits != patched_at) {
    std::memcpy(vud, vs_ud, sizeof(vud));
    std::memcpy(pud, ps_ud, sizeof(pud));
    PatchUserData(vud);
    if (ps) {
      PatchUserData(pud);
    }
    // The pipeline assumed these bases translated (and aligned): they still must be.
    auto moved = [&](const Shader::Program &p, const u32 *ud, u32 *lin,
                     const Shader::MslOptions &o) {
      u32 aligned = 0;
      const u32 mask = LinearTable(p, ud, lin, &aligned);
      return (mask & o.linear_mask) != o.linear_mask ||
             (aligned & o.linear_aligned) != o.linear_aligned;
    };
    if ((vo.linear_mask && moved(*vs, vud, vlin, vo)) ||
        (ps && fo.linear_mask && moved(*ps, pud, plin, fo))) {
      ReportOnce("draw skipped: a linear base moved between key and bind");
      return;
    }
  }
  UpdateMemMap();
  static_assert(sizeof(consts) <= 96);
  BindBytes(0, 0, vud, 64, Shader::Binding::UserData);
  BindBuffer(0, mem[0], 0, Shader::Binding::Mem0);
  BindBuffer(0, mem[1], 0, Shader::Binding::Mem1);
  BindBuffer(0, mem_map, 0, Shader::Binding::MemMap);
  if (vo.linear_mask) {
    BindBytes(0, 2, vlin, 4 * (32 - std::countl_zero(vo.linear_mask)), Shader::Binding::Linear);
  }
  for (u32 k = 0; k < nvbufs; ++k) {
    BindBuffer(0, vbufs[k].buf, vbufs[k].off, kVertexBufferBase + k);
  }
  BindBytes(0, 1, &consts, sizeof(consts), Shader::Binding::Draw);
  if (fused_pending.on) {
    u32 eud[16], gud[16];
    std::memcpy(eud, fused_pending.es_ud, sizeof(eud));
    std::memcpy(gud, fused_pending.gs_ud, sizeof(gud));
    PatchUserData(eud);
    PatchUserData(gud);
    [enc setVertexBytes:eud length:64 atIndex:Shader::Binding::RingEsUd];
    [enc setVertexBytes:gud length:64 atIndex:Shader::Binding::RingGsUd];
    bound.buf[0][Shader::Binding::RingEsUd] = bound.buf[0][Shader::Binding::RingGsUd] = nil;
    BindBuffer(0, gds, Gpu::GdsBase(), Shader::Binding::Gds);
    if (fused_pending.es_index_va) {
      id<MTLBuffer> ib = nil;
      u64 ioff = 0;
      if (!Locate(fused_pending.es_index_va, 4, &ib, &ioff)) {
        if (trace)
          DC_INFO("gpu:   skipped: fused ES index buffer not mapped");
        return;
      }
      BindBuffer(0, ib, ioff, Shader::Binding::GsIndex);
    }
    if (trace) {
      DC_INFO("gpu:   fused ES %#llx + GS %#llx into the copy VS, %u vertices",
              (unsigned long long)fused_pending.es_addr, (unsigned long long)fused_pending.gs_addr, dc.count);
    }
  }
  for (u32 k = 0; k < vs_res.tex.size(); ++k) {
    BindTexture(0, vs_res.tex[k], k);
  }
  for (u32 u = 0; u < vs_res.smp.size() && u < 16; ++u) {
    BindSampler(0, vs_res.smp[u], u);
  }
  if (ps) {
    BindBytes(1, 0, pud, 64, Shader::Binding::UserData);
    BindBuffer(1, mem[0], 0, Shader::Binding::Mem0);
    BindBuffer(1, mem[1], 0, Shader::Binding::Mem1);
    BindBuffer(1, mem_map, 0, Shader::Binding::MemMap);
    if (fo.linear_mask) {
      BindBytes(1, 2, plin, 4 * (32 - std::countl_zero(fo.linear_mask)), Shader::Binding::Linear);
    }
    for (u32 k = 0; k < ps_res.tex.size(); ++k) {
      BindTexture(1, ps_res.tex[k], k);
    }
    for (u32 u = 0; u < ps_res.smp.size() && u < 16; ++u) {
      BindSampler(1, ps_res.smp[u], u);
    }
    if (depth_bounds) {
      const float zb[2] = {r.F(Reg::DB_DEPTH_BOUNDS_MIN),
                           r.F(Reg::DB_DEPTH_BOUNDS_MAX)};
      if (trace) {
        DC_INFO("gpu:   depth bounds [%g, %g] depth fmt %lu", zb[0], zb[1],
                (unsigned long)depth->tex.pixelFormat);
      }
      [enc setFragmentBytes:zb
                     length:sizeof(zb)
                    atIndex:Shader::kDepthBoundsBuffer];
      bound.buf[1][Shader::kDepthBoundsBuffer] = nil;
      BindTexture(1, depth->bounds_copy, Shader::kDepthBoundsTexture);
    }
    if (probe_slot >= 0) {
      BindBuffer(1, probe_buf, u64(probe_slot) * Shader::kProbeWords * 4, Shader::kProbeBinding);
    }
  }

  if (capture) {
    debug_vs_domain = tess_draw != nullptr;
    debug_vs_buf = [device newBufferWithLength:debug_vs_domain ? (6145 + 240000) * 16 : 64 * 16
                                       options:MTLResourceStorageModeShared];
    if (debug_vs_domain) {
      std::memset(debug_vs_buf.contents, 0xFF, debug_vs_buf.length); // NaN = corner not run
      static_cast<u32 *>(debug_vs_buf.contents)[6144 * 4] = 0;     // vertex counter
    }
    BindBuffer(0, debug_vs_buf, 0, 6);
    DC_INFO("gpu: capturing VS %#llx ud: %08x %08x %08x %08x %08x %08x %08x "
            "%08x %08x %08x %08x %08x %08x %08x %08x %08x",
            vs_addr, vs_ud[0], vs_ud[1], vs_ud[2], vs_ud[3], vs_ud[4], vs_ud[5],
            vs_ud[6], vs_ud[7], vs_ud[8], vs_ud[9], vs_ud[10], vs_ud[11],
            vs_ud[12], vs_ud[13], vs_ud[14], vs_ud[15]);
  }
  // ---- primitive assembly ----
  const u32 base_vertex = r[Reg::VGT_INDX_OFFSET];
  const NSUInteger instances = std::max<u32>(dc.instances, 1);
  // DRAW_(INDEX_)INDIRECT: the guest args (Metal's layout) give the instance
  // count when the GPU runs the draw; paths that cannot use them need the
  // CP's early read.
  // Off by default: Driveclub's particle counts go negative on the GPU too
  // (instances 0xFFFFFFFF hung the GPU), so the count bug is upstream.
  static const bool use_gpu_args = std::getenv("DC_GPU_DRAW_ARGS") != nullptr;
  id<MTLBuffer> abuf = nil;
  u64 aoff = 0;
  const bool gpu_args = dc.indirect_args && use_gpu_args && !(dc.indirect_args & 3) &&
                        Locate(dc.indirect_args, dc.indexed ? 20 : 16, &abuf, &aoff);
  const bool cpu_count_unusable = dc.indirect_args && dc.instances == 0;
  MTLPrimitiveType mprim = MTLPrimitiveTypeTriangle;
  bool to_list = false; // fans and quads are rebuilt as triangle lists
  switch (prim) {
  case PrimType::PointList:
    mprim = MTLPrimitiveTypePoint;
    break;
  case PrimType::LineList:
    mprim = MTLPrimitiveTypeLine;
    break;
  case PrimType::LineStrip:
    mprim = MTLPrimitiveTypeLineStrip;
    break;
  case PrimType::TriList:
    mprim = MTLPrimitiveTypeTriangle;
    break;
  case PrimType::TriStrip:
    mprim = MTLPrimitiveTypeTriangleStrip;
    break;
  case PrimType::TriFan:
  case PrimType::Polygon:
  case PrimType::QuadList:
    to_list = true;
    break;
  case PrimType::RectList:
    if (cpu_count_unusable) {
      return;
    }
    if (dc.indexed) {
      ReportOnce("indexed RECTLIST draws are not supported");
      return;
    }
    [enc drawPrimitives:MTLPrimitiveTypeTriangle
            vertexStart:0
            vertexCount:(dc.count / 3) * 6
          instanceCount:instances];
    return;
  case static_cast<PrimType>(9): // DI_PT_PATCH: the domain pass of DrawTess
    if (!tess_draw) {
      ReportOnce("patch draw skipped (tessellation off: DC_NO_TESS / DC_TESS=0, or not emulated)");
      return;
    }
    if (static const bool nodepth = std::getenv("DC_TESS_NODEPTH") != nullptr; nodepth) {
      static id<MTLDepthStencilState> always = [&] {
        MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
        dd.depthCompareFunction = MTLCompareFunctionAlways;
        dd.depthWriteEnabled = NO;
        return [device newDepthStencilStateWithDescriptor:dd];
      }();
      [enc setDepthStencilState:always];
      bound.depth_state = nil;
    }
    if (static const bool nocull = std::getenv("DC_TESS_NOCULL") != nullptr; nocull) {
      [enc setCullMode:MTLCullModeNone];
      bound.cull = int(MTLCullModeNone);
    }
    [enc setVertexBuffer:tess_lds offset:0 atIndex:Shader::Binding::TessLds];
    bound.buf[0][Shader::Binding::TessLds] = nil;
    [enc setTessellationFactorBuffer:tess_draw->factors offset:0 instanceStride:0];
    [enc drawPatches:0
              patchStart:0
              patchCount:tess_draw->patches
        patchIndexBuffer:nil
  patchIndexBufferOffset:0
           instanceCount:1
            baseInstance:0];
    return;
  default:
    ReportOnce("primitive type " + std::to_string(static_cast<u32>(prim)) +
               " not supported");
    return;
  }

  if (!dc.indexed && !to_list && gpu_args) {
    [enc drawPrimitives:mprim indirectBuffer:abuf indirectBufferOffset:aoff];
    return;
  }
  if (cpu_count_unusable) {
    return;
  }
  if (!dc.indexed && !to_list) {
    [enc drawPrimitives:mprim
            vertexStart:base_vertex
            vertexCount:dc.count
          instanceCount:instances];
    return;
  }
  if (!dc.indexed && prim == PrimType::QuadList) {
    const u32 quads = std::min<u32>(dc.count / 4, 0x10000);
    [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                    indexCount:quads * 6
                     indexType:MTLIndexTypeUInt32
                   indexBuffer:quad_indices
             indexBufferOffset:0
                 instanceCount:instances
                    baseVertex:base_vertex
                  baseInstance:0];
    return;
  }

  // Indexed lists and strips read the guest index buffer in place (Metal
  // restarts strips at 0xFFFF / 0xFFFFFFFF like VGT_MULTI_PRIM_IB_RESET_INDX =
  // -1).
  if (dc.indexed && !to_list && (dc.index_size == 2 || dc.index_size == 4) &&
      gpu_args) {
    // The first index is in the args, relative to INDEX_BASE.
    id<MTLBuffer> ibuf = nil;
    u64 ioff = 0;
    if ((dc.index_base % dc.index_size) == 0 &&
        Locate(dc.index_base, (u64(dc.index_offset) + dc.count) * dc.index_size, &ibuf, &ioff)) {
      [enc drawIndexedPrimitives:mprim
                       indexType:dc.index_size == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                     indexBuffer:ibuf
               indexBufferOffset:ioff
                  indirectBuffer:abuf
            indirectBufferOffset:aoff];
      return;
    }
  }
  if (cpu_count_unusable) {
    return;
  }
  if (dc.indexed && !to_list && (dc.index_size == 2 || dc.index_size == 4)) {
    const u64 src = dc.index_base + u64(dc.index_offset) * dc.index_size;
    id<MTLBuffer> ibuf = nil;
    u64 ioff = 0;
    if ((src % dc.index_size) == 0 &&
        Locate(src, u64(dc.count) * dc.index_size, &ibuf, &ioff)) {
      [enc drawIndexedPrimitives:mprim
                      indexCount:dc.count
                       indexType:dc.index_size == 2 ? MTLIndexTypeUInt16
                                                    : MTLIndexTypeUInt32
                     indexBuffer:ibuf
               indexBufferOffset:ioff
                   instanceCount:instances
                      baseVertex:base_vertex
                    baseInstance:0];
      return;
    }
  }

  // Gather this draw's indices (guest buffer or implicit 0..n-1).
  std::vector<u32> idx(dc.count);
  if (dc.indexed) {
    const u64 src = dc.index_base + u64(dc.index_offset) * dc.index_size;
    if (!Readable(src, u64(dc.count) * dc.index_size)) {
      ReportOnce("index buffer not mapped");
      return;
    }
    if (dc.index_size == 4) {
      std::memcpy(idx.data(), reinterpret_cast<const void *>(src),
                  dc.count * 4ull);
    } else {
      const auto *s16 = reinterpret_cast<const u16 *>(src);
      for (u32 k = 0; k < dc.count; ++k) {
        idx[k] =
            s16[k] == 0xFFFF ? 0xFFFFFFFFu : s16[k]; // keep the restart index
      }
    }
  } else {
    for (u32 k = 0; k < dc.count; ++k) {
      idx[k] = k;
    }
  }
  if (to_list) {
    std::vector<u32> list;
    if (prim == PrimType::QuadList) {
      for (u32 q = 0; q + 3 < dc.count; q += 4) {
        list.insert(list.end(), {idx[q], idx[q + 1], idx[q + 2], idx[q],
                                 idx[q + 2], idx[q + 3]});
      }
    } else {
      for (u32 k = 1; k + 1 < dc.count; ++k) {
        list.insert(list.end(), {idx[0], idx[k], idx[k + 1]});
      }
    }
    idx.swap(list);
    mprim = MTLPrimitiveTypeTriangle;
  }
  if (idx.empty()) {
    return;
  }
  auto [buf, off] = Upload(idx.data(), idx.size() * 4);
  [enc drawIndexedPrimitives:mprim
                  indexCount:idx.size()
                   indexType:MTLIndexTypeUInt32
                 indexBuffer:buf
           indexBufferOffset:off
               instanceCount:instances
                  baseVertex:base_vertex
                baseInstance:0];
}

void MetalRenderer::Impl::Dispatch(const RegFile &r, u32 gx, u32 gy, u32 gz) {
  const u64 addr = r.ProgramAddr(Reg::COMPUTE_PGM_LO);
  dbg_shader = addr;
  const u32 rsrc2 = r[Reg::COMPUTE_PGM_RSRC2];
  const u32 *ud = r.UserData(Reg::COMPUTE_USER_DATA_0);
  const u32 tx = std::max<u32>(1, r[Reg::COMPUTE_NUM_THREAD_X] & 0xFFFF);
  const u32 ty = std::max<u32>(1, r[Reg::COMPUTE_NUM_THREAD_Y] & 0xFFFF);
  const u32 tz = std::max<u32>(1, r[Reg::COMPUTE_NUM_THREAD_Z] & 0xFFFF);
  // DC_LOG_CS=<hex addr>[,<hex addr>...]: user data of those dispatches and the
  // first dwords their user-data pointer (ud[0..1]) names, now and once the GPU
  // has run the command buffer (a count a previous dispatch writes).
  static const std::vector<u64> log_cs = [] {
    std::vector<u64> v;
    if (const char *e = std::getenv("DC_LOG_CS")) {
      for (char *q = const_cast<char *>(e); *q;) {
        v.push_back(std::strtoull(q, &q, 16));
        if (*q) ++q;
      }
    }
    return v;
  }();
  if (!log_cs.empty() && std::find(log_cs.begin(), log_cs.end(), addr) != log_cs.end()) {
    const u64 ptr = (u64(ud[0]) | (u64(ud[1] & 0xFFFF) << 32)) & kGpuAddrMask;
    u32 now[4] = {};
    if (Readable(ptr, 16)) {
      std::memcpy(now, reinterpret_cast<const void *>(ptr), 16);
    }
    DC_INFO("gpu: cs %#llx %ux%ux%u q%u flip %llu ud %08x %08x %08x %08x ptr %#llx: %08x %08x %08x %08x",
            (unsigned long long)addr, gx, gy, gz, Gpu::t_queue, (unsigned long long)frame_count, ud[0], ud[1],
            ud[2], ud[3], (unsigned long long)ptr, now[0], now[1], now[2], now[3]);
    if (Readable(ptr, 96)) {
      EnsureCommandBuffer();
      const u64 a = addr;
      [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
        const u32 *d = reinterpret_cast<const u32 *>(ptr);
        char line[400];
        int n = std::snprintf(line, sizeof(line), "gpu: cs %#llx ptr %#llx after:", (unsigned long long)a,
                              (unsigned long long)ptr);
        for (int k = 0; k < 24 && n < int(sizeof(line)) - 10; ++k) {
          n += std::snprintf(line + n, sizeof(line) - n, " %08x", d[k]);
        }
        DC_INFO("%s", line);
      }];
    }
  }
  // Memset shape: 256 threads in one line, destination V# of dwords in user data.
  if (tx == 256 && ty == 1 && tz == 1 && gy == 1 && gz == 1 && ((ud[1] >> 16) & 0x3FFF) == 4) {
    const u64 dst = ud[0] | (u64(ud[1] & 0xFFF) << 32);
    if (color_targets.count(dst)) {
      memory_fills[dst] = ++write_seq;
      ++texture_epoch;
    }
    // The same memset over a depth surface clears it (GT7 clears its reflection
    // cube depth 0x900b460000 to 0, reverse Z, this way each frame). The value
    // is dword 0 of the constant buffer V# in ud[4..7] (dword 1 for odd
    // dwords, the stencil half of a pair). Without it the cube's depth kept the
    // 1.0 of its creation, every GEQUAL test failed and the reflection cube,
    // its filtered copies and the cockpit's indirect light stayed black.
    if (const auto it = depth_targets.find(dst); it != depth_targets.end()) {
      const u64 cb = (u64(ud[4]) | (u64(ud[5] & 0xFFF) << 32)) & kGpuAddrMask;
      u32 value = 0;
      if (cb && Readable(cb, 4)) {
        std::memcpy(&value, reinterpret_cast<const void *>(cb), 4);
      }
      float depth_value;
      std::memcpy(&depth_value, &value, 4);
      for (auto &z : it->second) {
        ClearDepthNow(*z, depth_value);
      }
      if (trace) {
        DC_INFO("gpu:   memset clears depth %#llx to %g", (unsigned long long)dst, depth_value);
      }
    }
  }
  if (trace) {
    DC_INFO("gpu: dispatch q%u cs=%#llx groups %ux%ux%u%s threads %ux%ux%u rsrc2=%#x "
            "ud=%08x %08x %08x %08x %08x %08x %08x "
            "%08x %08x %08x %08x %08x %08x %08x %08x %08x",
            Gpu::t_queue, addr, gx, gy, gz, indirect_args ? " (indirect)" : "", tx, ty, tz,
            rsrc2, ud[0], ud[1], ud[2], ud[3],
            ud[4], ud[5], ud[6], ud[7], ud[8], ud[9], ud[10], ud[11], ud[12],
            ud[13], ud[14], ud[15]);
    FindRefs(ud, 16, 0, "ud", "cs", addr);
    // DC_RAIN_DIAG: what the drop lighting shader leaves in the drop colors
    // once the GPU has run it (a peek at record time only sees what the CPU
    // wrote), and the light count it loops over.
    if (rain_diag && addr == kRainDropLightCs && cmd) {
      const u64 table = (u64(ud[12]) | (u64(ud[13]) << 32)) & kGpuAddrMask;
      if (Readable(table, 40 * 4)) {
        const u32 *t = reinterpret_cast<const u32 *>(table);
        const u64 colors = (u64(t[20]) | (u64(t[21] & 0xFFF) << 32)) & kGpuAddrMask;
        const u64 lights = (u64(t[36]) | (u64(t[37] & 0xFFF) << 32)) & kGpuAddrMask;
        const u32 n = std::min<u32>(t[22], gx * tx);
        const u32 light_count = Readable(lights, 4) ? *reinterpret_cast<const u32 *>(lights) : ~0u;
        if (Readable(colors, u64(n) * 16)) {
          [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
            const float *c = reinterpret_cast<const float *>(colors);
            u32 big = 0;
            for (u32 i = 0; i < n; ++i) {
              big += !(std::fabs(c[i * 4]) < 1e30f) || !(std::fabs(c[i * 4 + 1]) < 1e30f) ||
                     !(std::fabs(c[i * 4 + 2]) < 1e30f);
            }
            DC_INFO("gpu: rain diag: after cs %#llx: colors %#llx %u/%u big, [0] = %g %g %g %g, "
                    "lights %#llx count %u",
                    (unsigned long long)kRainDropLightCs, (unsigned long long)colors, big, n, c[0], c[1],
                    c[2], c[3], (unsigned long long)lights, light_count);
          }];
        }
      }
    }
  }
  if (!addr || !gx || !gy || !gz) {
    return;
  }
  static const u64 skip_cs =
      std::getenv("DC_SKIP_CS")
          ? std::strtoull(std::getenv("DC_SKIP_CS"), nullptr, 16)
          : 0;
  if (skip_cs && addr == skip_cs) { // debug: drop one compute shader's dispatches
    return;
  }
  ++stat_dispatches;
  ++dispatch_seq;
  if (trace && (scan_all || !scan_addrs.empty())) {
    scan_label = "dispatch cs " + Hex(addr);
  }
  // DC_PEEK_CS_TABLE=<hex cs>,...: in the traced flip, the V#s in the table
  // at that dispatch's ud[0..1] (64 dwords), each with its non-zero dword
  // count once the GPU has run the dispatch (where its outputs go).
  if (trace && std::getenv("DC_PEEK_CS_TABLE")) {
    static const std::vector<u64> want = [] {
      std::vector<u64> v;
      char *q = const_cast<char *>(std::getenv("DC_PEEK_CS_TABLE"));
      do v.push_back(std::strtoull(q, &q, 16)); while (*q == ',' && *++q);
      return v;
    }();
    const u64 table = (u64(ud[0]) | (u64(ud[1] & 0xFFFF) << 32)) & kGpuAddrMask;
    if (std::find(want.begin(), want.end(), addr) != want.end() && Readable(table, 256)) {
      const u32 *t = reinterpret_cast<const u32 *>(table);
      std::vector<std::tuple<u32, u64, u64>> vs;
      for (u32 k = 0; k + 4 <= 64; k += 4) {
        const u64 base = (u64(t[k]) | (u64(t[k + 1] & 0xFFF) << 32)) & kGpuAddrMask;
        const u32 stride = (t[k + 1] >> 16) & 0x3FFF;
        const u64 size = stride ? u64(t[k + 2]) * stride : t[k + 2];
        if (base && (t[k + 3] & 0xFFF) && size && size <= (64ull << 20) && Readable(base, size)) {
          vs.emplace_back(k, base, size);
          DC_INFO("gpu: cs %#llx table %#llx [%u] V# %08x %08x %08x %08x base %#llx size %llu",
                  (unsigned long long)addr, (unsigned long long)table, k, t[k], t[k + 1],
                  t[k + 2], t[k + 3], (unsigned long long)base, (unsigned long long)size);
        }
      }
      peek_cs_table_pending.push_back({addr, std::move(vs)});
    }
  }
  if (trace) {
    static const u64 peek_cs =
        std::getenv("DC_PEEK_CS")
            ? std::strtoull(std::getenv("DC_PEEK_CS"), nullptr, 16)
            : 0;
    if (addr == peek_cs) {
      for (u32 k = 0; k + 1 < 12; k += 4) {
        peek_addrs.push_back((u64(ud[k]) | (u64(ud[k + 1] & 0xFFF) << 32)) &
                             kGpuAddrMask);
      }
      // DC_PEEK_TABLE=<ud>,<dword>: also the V# found at pointer ud[n..n+1] +
      // dword*4
      if (const char *pt = std::getenv("DC_PEEK_TABLE")) {
        char *e = nullptr;
        const u32 n = std::strtoul(pt, &e, 10);
        const u32 off = *e ? std::strtoul(e + 1, nullptr, 10) : 0;
        const u64 table = u64(ud[n]) | (u64(ud[n + 1] & 0xFFFF) << 32);
        if (n + 1 < 16 && Readable(table + off * 4, 16)) {
          const u32 *v = reinterpret_cast<const u32 *>(table + off * 4);
          DC_INFO("gpu: peek table %#llx+%u: V# %08x %08x %08x %08x", table,
                  off, v[0], v[1], v[2], v[3]);
          peek_addrs.push_back((u64(v[0]) | (u64(v[1] & 0xFFF) << 32)) &
                               kGpuAddrMask);
        }
      }
    }
  }
  // The Gnm driver copies surfaces with a buffer-copy shader (V# source in
  // user data 0-3, destination in 4-7). Surfaces we render live only in
  // Metal textures, so copy the texture and give the destination one too.
  if (tx == 64 && ty == 1 && tz == 1) {
    const u64 src = (u64(ud[0]) | (u64(ud[1] & 0xFFF) << 32)) & kGpuAddrMask;
    const u64 dst = (u64(ud[4]) | (u64(ud[5] & 0xFFF) << 32)) & kGpuAddrMask;
    const bool from_depth = Latest(depth_targets, src) != nullptr;
    RenderTarget *from =
        from_depth ? Latest(depth_targets, src) : Latest(color_targets, src);
    if (from && dst && dst != src) {
      RenderTarget &to =
          GetTarget(from_depth ? depth_targets : color_targets, dst,
                    from->tex.pixelFormat, from->width, from->height,
                    from_depth, from->levels, from->dim, from->depth_or_layers);
      EndPass();
      EnsureCommandBuffer();
      id<MTLBlitCommandEncoder> b = [cmd blitCommandEncoder];
      [b copyFromTexture:from->tex toTexture:to.tex];
      [b endEncoding];
      to.initialized = true;
      to.clear_pending = from->clear_pending;
      to.clear_color = from->clear_color;
      to.clear_depth = from->clear_depth;
      to.written = ++write_seq;
      NoteTargetWritten(to);
      if (trace || from->dim != TexDim::Tex2D) {
        DC_INFO("gpu:   surface copy %#llx -> %#llx (%ux%u dim %d layers %u fmt %lu)",
                src, dst, from->width, from->height, (int)from->dim,
                from->depth_or_layers, (unsigned long)from->tex.pixelFormat);
      }
      return;
    }
  }
  Program *cs = GetProgram(Stage::Compute, addr, ud, Bits(rsrc2, 1, 5),
                           Bits(rsrc2, 11, 2), rsrc2);
  if (!cs) {
    return;
  }
  // A shader that only stores to a CMASK/HTILE surface it gets straight
  // from user data is a fast clear (others read metadata, e.g. HTILE for
  // tile depth bounds, and run normally).
  for (u32 k = 0; k + 3 < 16 && !metas.empty(); ++k) {
    const u64 base =
        (u64(ud[k]) | (u64(ud[k + 1] & 0xFFF) << 32)) & kGpuAddrMask;
    auto m = metas.find(base);
    if (m == metas.end() || !OnlyStoresTo(*cs, k)) {
      continue;
    }
    // HTILE filled with expanded tiles (ZMASK 0xF) after the game wrote the
    // depth itself (a compute downsample): the data stands, it is no clear.
    const std::optional<u32> fill = FillValue(*cs, ud);
    if (m->second.depth && fill && (*fill & 0xF) == 0xF) {
      if (trace) {
        DC_INFO("gpu:   HTILE of %#llx expanded (fill %#x): not a clear",
                (unsigned long long)m->second.target, *fill);
      }
      EndPass();
      return;
    }
    if (MarkMetaCleared(base, 1)) {
      EndPass();
      return;
    }
  }
  // A shader reading HTILE (per-tile depth bounds for light culling) would
  // see stale memory: depth lives in Metal textures. Build it from the depth
  // texture, so light culling gets real per-tile depth bounds; failing that,
  // expanded tiles with the full Z range (nothing culled).
  for (u32 k = 0; k + 3 < 16 && !metas.empty(); ++k) {
    const u64 base =
        (u64(ud[k]) | (u64(ud[k + 1] & 0xFFF) << 32)) & kGpuAddrMask;
    auto m = metas.find(base);
    if (m == metas.end() || !m->second.depth) {
      continue;
    }
    const u32 stride = (ud[k + 1] >> 16) & 0x3FFF;
    const u64 bytes = u64(ud[k + 2]) * (stride ? stride : 1);
    if (bytes && bytes < (64ull << 20) && !BuildHtile(base, bytes, m->second)) {
      expanding_htile = true; // our own fill, not the game clearing depth
      DmaData(0, base, 2, 0xFFFC000Fu, static_cast<u32>(bytes & ~3ull));
      expanding_htile = false;
    }
    k += 3;
  }
  std::vector<TextureType> types;
  Resources res;
  resolving_shader = addr;
  Resolve(*cs, ud, types, res);

  // One encoder per dispatch while timing them or dumping a traced frame.
  const bool batch = !pass_timing && !trace;
  if (!batch || enc) {
    EndPass();
  }
  if (++work_since_commit >= kCommitEvery) {
    Commit(false);
  }
  // DC_CAPTURE_CS=<cs>[:<flip>]: a command buffer holding only that dispatch
  // goes into ../bench/capture_cs_<cs>.gputrace for Xcode's shader profiler
  // (needs MTL_CAPTURE_ENABLED=1).
  static const char *capture_cs_env = std::getenv("DC_CAPTURE_CS");
  static const u64 capture_cs =
      capture_cs_env ? std::strtoull(capture_cs_env, nullptr, 16) : 0;
  static const u64 capture_cs_flip =
      capture_cs_env && std::strchr(capture_cs_env, ':')
          ? std::strtoull(std::strchr(capture_cs_env, ':') + 1, nullptr, 10)
          : 0;
  static bool capture_cs_done = false;
  bool capturing = false;
  if (capture_cs == addr && !capture_cs_done && frame_count >= capture_cs_flip) {
    capture_cs_done = true;
    Commit(false);
    capturing = BeginCapture("../bench/capture_cs_" + Hex(addr) + ".gputrace");
  }
  EnsureCommandBuffer();
  u32 cud[16];
  std::memcpy(cud, ud, sizeof(cud));
  PatchUserData(cud);
  UpdateMemMap();
  Shader::MslOptions o;
  o.textures = types; // borrowed: `types` lives until the key is hashed
  o.map_hoist = map_hoist;
  u32 lin[Shader::kMaxLinearBases];
  o.linear_mask = LinearTable(*cs, cud, lin, &o.linear_aligned);
  const int probe_slot =
      probe_buf && trace
          ? ProbeSlot(addr, "dispatch " + std::to_string(stat_dispatches) +
                                " cs " + Hex(addr))
          : -1;
  if (probe_slot >= 0) {
    o.probe_pc = probe.pc;
    o.probe_x = probe.x;
    o.probe_y = probe.y;
  }
  u64 key = Hash(cs->hash);
  key = Hash(o.probe_pc, key);
  key = Hash(rsrc2, key);
  key = Hash(o.map_hoist, key);
  key = Hash(o.linear_mask, key);
  key = Hash(o.linear_aligned, key);
  key = Fnv(types.data(), types.size() * sizeof(TextureType), key);
  const u32 group = tx * ty * tz;
  const u64 pkey = Hash(group, key);
  TakeCompiled();
  auto it = compute_pipelines.find(pkey);
  if (it == compute_pipelines.end()) {
    if (!pending.count(pkey)) {
      pending.insert(pkey);
      PrewarmEntry rec; // for the next launch's prewarm (see PrewarmStart)
      if (probe_slot < 0) {
        const auto *code = reinterpret_cast<const u32 *>(addr);
        rec.pkey = pkey;
        rec.addr = addr;
        rec.rsrc2 = rsrc2;
        rec.group = group;
        rec.map_hoist = o.map_hoist;
        rec.linear_mask = o.linear_mask;
        rec.linear_aligned = o.linear_aligned;
        std::copy(ud, ud + 16, rec.ud.begin());
        rec.types.assign(types.begin(), types.end());
        rec.code.assign(code, code + Shader::Gcn::ProgramSize(code) / 4);
      }
      compiler->Submit([this, src = MslSource(*cs, o, key), key, pkey, group,
                        addr, rec = std::move(rec)] {
        id<MTLComputePipelineState> pso = nil;
        if (id<MTLFunction> fn = compiler->Function(key, src, addr)) {
          MTLComputePipelineDescriptor *d = [MTLComputePipelineDescriptor new];
          d.computeFunction = fn;
          d.label = [NSString stringWithFormat:@"cs %#llx", addr]; // GPU captures
          d.maxTotalThreadsPerThreadgroup =
              group; // the guest's group size must fit as is
          NSError *err = nil;
          const auto t0 = std::chrono::steady_clock::now();
          pso = [device
              newComputePipelineStateWithDescriptor:d
                                            options:MTLPipelineOptionNone
                                         reflection:nil
                                              error:&err];
          ReportSlowCompile("compute pipeline", addr, 0, t0);
          if (!pso) {
            DC_ERROR("gpu: compute pipeline %#llx: %s", addr,
                     err ? err.localizedDescription.UTF8String : "?");
          } else if (pso.maxTotalThreadsPerThreadgroup < group) {
            DC_ERROR("gpu: compute shader %#llx needs %u threads per group, "
                     "Metal allows %lu",
                     addr, group,
                     (unsigned long)pso.maxTotalThreadsPerThreadgroup);
            pso = nil;
          }
        }
        if (pso && rec.pkey) {
          PrewarmRecord(rec);
        }
        {
          std::scoped_lock lk(done_lock);
          done_compute.emplace_back(pkey, pso);
        }
        done_cv.notify_all();
      }, /*urgent=*/true); // waited for right below
    }
    // Dispatches often produce data later work depends on: wait.
    WaitCompiled(pkey);
    it = compute_pipelines.find(pkey);
  }
  id<MTLComputePipelineState> pso = it->second;
  if (!pso) {
    if (capturing) {
      [[MTLCaptureManager sharedCaptureManager] stopCapture];
    }
    return;
  }

  id<MTLComputeCommandEncoder> e = nil;
  if (pass_timing) {
    const u32 sample = TsBegin(addr, gx, gy, true);
    MTLComputePassDescriptor *cp = [MTLComputePassDescriptor computePassDescriptor];
    cp.sampleBufferAttachments[0].sampleBuffer = ts_buf[sample / kTsPerBuf];
    cp.sampleBufferAttachments[0].startOfEncoderSampleIndex = sample % kTsPerBuf;
    cp.sampleBufferAttachments[0].endOfEncoderSampleIndex = sample % kTsPerBuf + 1;
    e = [cmd computeCommandEncoderWithDescriptor:cp];
  } else if (batch) {
    if (!cenc) {
      cenc = [cmd computeCommandEncoder];
    }
    e = cenc;
  } else {
    e = [cmd computeCommandEncoder];
  }
  [e setComputePipelineState:pso];
  [e setBytes:cud length:64 atIndex:Shader::Binding::UserData];
  [e setBuffer:mem[0] offset:0 atIndex:Shader::Binding::Mem0];
  [e setBuffer:mem[1] offset:0 atIndex:Shader::Binding::Mem1];
  [e setBuffer:mem_map offset:0 atIndex:Shader::Binding::MemMap];
  [e setBuffer:gds offset:Gpu::GdsBase() atIndex:Shader::Binding::Gds];
  if (o.linear_mask) {
    [e setBytes:lin
         length:4 * (32 - std::countl_zero(o.linear_mask))
        atIndex:Shader::Binding::Linear];
  }
  for (u32 k = 0; k < res.tex.size(); ++k) {
    [e setTexture:res.tex[k] atIndex:k];
  }
  for (u32 u = 0; u < res.smp.size() && u < 16; ++u) {
    [e setSamplerState:res.smp[u] atIndex:u];
  }
  if (probe_slot >= 0) {
    [e setBuffer:probe_buf
          offset:u64(probe_slot) * Shader::kProbeWords * 4
         atIndex:Shader::kProbeBinding];
  }
  // LDS_SIZE is in 128-dword units on CI; Metal wants a multiple of 16 bytes.
  const u32 lds = std::max<u32>(16, Bits(rsrc2, 15, 9) * 512);
  [e setThreadgroupMemoryLength:std::min<u32>(lds, 32768) atIndex:0];
  id<MTLBuffer> ibuf = nil;
  u64 ioff = 0;
  if (indirect_args && Locate(indirect_args, 12, &ibuf, &ioff)) {
    [e dispatchThreadgroupsWithIndirectBuffer:ibuf
                         indirectBufferOffset:ioff
                        threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
  } else {
    [e dispatchThreadgroups:MTLSizeMake(gx, gy, gz)
        threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
  }
  if (!batch) {
    [e endEncoding];
  }
  if (capturing) {
    Commit(false);
    [[MTLCaptureManager sharedCaptureManager] stopCapture];
    DC_INFO("gpu: captured dispatch cs %#llx", addr);
  }
  // GT7 reads back what the GPU writes into its CPU pools (from 0xe000000000,
  // ryGT7 FIX-213): a linear storage image there must reach guest memory.
  static const bool cpu_readback = std::getenv("DC_NO_CPU_READBACK") == nullptr;
  for (u32 k = 0; cpu_readback && k < res.tex.size() && k < tsharps_.size(); ++k) {
    const u32 *t = tsharps_[k].data();
    const u64 base = ((u64(t[0]) | (u64(t[1] & 0xFF) << 32)) << 8) & kGpuAddrMask;
    id<MTLTexture> tex = res.tex[k];
    if (!cs->images[k].storage || base < 0xe000000000ull || ((t[3] >> 20) & 0x1F) != 8 ||
        tex.textureType != MTLTextureType2D) {
      continue;
    }
    const u32 row = (((t[4] >> 13) & 0x3FFF) + 1) * PixelBytes(tex.pixelFormat);
    const u32 size = row * u32(tex.height);
    if (!row || !Readable(base, size)) {
      continue;
    }
    EndPass();
    EnsureCommandBuffer();
    id<MTLBuffer> out = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> b = [cmd blitCommandEncoder];
    [b copyFromTexture:tex
                 sourceSlice:0
                 sourceLevel:0
                sourceOrigin:MTLOriginMake(0, 0, 0)
                  sourceSize:MTLSizeMake(tex.width, tex.height, 1)
                    toBuffer:out
           destinationOffset:0
      destinationBytesPerRow:row
    destinationBytesPerImage:size];
    [b endEncoding];
    [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
      std::memcpy(reinterpret_cast<void *>(base), out.contents, size);
    }];
    if (trace) {
      DC_INFO("gpu:   readback %ux%u to guest %#llx", u32(tex.width), u32(tex.height),
              (unsigned long long)base);
    }
  }
  if (dump_passes && trace && dump_frames) {
    u32 n = 0;
    for (u32 k = 0; k < res.tex.size() && n < 8; ++k) {
      if (cs->images[k].storage &&
          (res.tex[k].textureType == MTLTextureType2D ||
           res.tex[k].textureType == MTLTextureType2DArray)) {
        pass_tex[n] = res.tex[k];
        pass_addr[n++] = addr;
      }
    }
    DumpPass();
  }
  if (trace && (scan_all || !scan_addrs.empty())) {
    ScanMax();
  }
  // DC_DUMP_CS=<cs>: with DC_TRACE_FRAME, the textures that shader samples.
  static const u64 dump_cs = std::getenv("DC_DUMP_CS")
                                 ? std::strtoull(std::getenv("DC_DUMP_CS"),
                                                 nullptr, 16)
                                 : 0;
  if (dump_cs == addr && trace && dump_frames) {
    for (u32 k = 0; k < res.tex.size(); ++k) {
      id<MTLTexture> t = res.tex[k];
      if (cs->images[k].storage || !copy_pipeline) {
        continue;
      }
      const NSUInteger n =
          t.textureType == MTLTextureType2DArray ? t.arrayLength : 1;
      for (NSUInteger l = 0; l < std::min<NSUInteger>(t.mipmapLevelCount, 3);
           ++l) {
        for (NSUInteger sl = 0; sl < std::min<NSUInteger>(n, 12); ++sl) {
          if (t.textureType != MTLTextureType2D &&
              t.textureType != MTLTextureType2DArray) {
            continue;
          }
          DumpTexture([t newTextureViewWithPixelFormat:t.pixelFormat
                                           textureType:MTLTextureType2D
                                                levels:NSMakeRange(l, 1)
                                                slices:NSMakeRange(sl, 1)],
                      k, (u64(k) << 8) | (l << 4) | sl);
        }
      }
    }
  }
}

// Where F11/F12 write captures and frame dumps: the session log folder when
// launched from Finder (whose working directory is "/"), otherwise ../bench.
std::filesystem::path TraceDir() {
  std::filesystem::path base = Launcher::LogDir();
  if (base.empty()) {
    base = "../bench";
  }
  std::error_code ec;
  std::filesystem::create_directories(base, ec);
  if (ec) {
    const char *home = std::getenv("HOME");
    base = std::filesystem::path(home ? home : "/tmp") / "Desktop/Driveclub Logs";
    std::filesystem::create_directories(base, ec);
  }
  return base;
}

// F11: records the next two frames into capture_<flip>.gputrace for
// Xcode (needs MTL_CAPTURE_ENABLED=1 in the environment at launch).
void MetalRenderer::Impl::StartCapture() {
  Commit(false); // the capture starts at a command buffer boundary
  if (BeginCapture((TraceDir() / ("capture_" + std::to_string(frame_count) + ".gputrace")).string())) {
    capture_end = frame_count + 2;
  }
}

// Starts a GPU capture of everything committed from now on into `path`.
bool MetalRenderer::Impl::BeginCapture(const std::string &path) {
  MTLCaptureManager *m = [MTLCaptureManager sharedCaptureManager];
  if (![m supportsDestination:MTLCaptureDestinationGPUTraceDocument]) {
    DC_ERROR("gpu: GPU capture needs MTL_CAPTURE_ENABLED=1 at launch");
    return false;
  }
  MTLCaptureDescriptor *d = [MTLCaptureDescriptor new];
  d.captureObject = device;
  d.destination = MTLCaptureDestinationGPUTraceDocument;
  d.outputURL = [NSURL fileURLWithPath:@(path.c_str())];
  NSError *err = nil;
  if (![m startCaptureWithDescriptor:d error:&err]) {
    DC_ERROR("gpu: capture failed: %s",
             err ? err.localizedDescription.UTF8String : "?");
    return false;
  }
  DC_INFO("gpu: capturing into %s", path.c_str());
  return true;
}

void MetalRenderer::Impl::Flip(const FlipInfo &f) {
  CpuTimer t(cpu_profile ? &cpu_flip_s : nullptr);
  ++frame_count;
  if (!linear_watch.empty() && trace_frame && frame_count > trace_frame + 1) {
    const u64 age = frame_count - trace_frame;
    for (const auto &w : linear_watch) {
      u64 nz = 0;
      const u32 *d = reinterpret_cast<const u32 *>(w.src);
      for (u64 i = 0; i < w.n; ++i) nz += d[i] != 0;
      DC_INFO("gpu: linear watch %#llx +%llu flips: %llu of %llu non-zero",
              (unsigned long long)w.addr, (unsigned long long)age,
              (unsigned long long)nz, (unsigned long long)w.n);
    }
    if (age >= 8) linear_watch.clear();
  }
  if (!peek_cs_table_pending.empty() && trace_frame && frame_count > trace_frame + 1) {
    Commit(true); // everything encoded so far has run
    for (const auto &[cs, vs] : peek_cs_table_pending) {
      for (const auto &[k, base, size] : vs) {
        u64 nz = 0;
        const u32 *d = reinterpret_cast<const u32 *>(base);
        for (u64 i = 0; i < size / 4; ++i) nz += d[i] != 0;
        DC_INFO("gpu: cs %#llx table [%u] %#llx: %llu of %llu dwords non-zero",
                (unsigned long long)cs, k, (unsigned long long)base,
                (unsigned long long)nz, (unsigned long long)(size / 4));
      }
    }
    peek_cs_table_pending.clear();
  }
  if (frame_count % kTextureSweepEvery == 0) {
    EvictTextures(kTextureMaxAge);
  }
  if (capture_end && frame_count >= capture_end) {
    Commit(false);
    [[MTLCaptureManager sharedCaptureManager] stopCapture];
    capture_end = 0;
    DC_INFO("gpu: capture done");
  }
  static const u64 capture_at = // DC_CAPTURE_AT=<flip>: F11 at that flip
      std::getenv("DC_CAPTURE_AT") ? std::strtoull(std::getenv("DC_CAPTURE_AT"), nullptr, 10) : 0;
  if (capture_at && frame_count == capture_at) {
    g_capture_request.store(true);
  }
  if (g_capture_request.exchange(false) && !capture_end) {
    StartCapture();
  }
  if (g_trace_request.exchange(false)) {
    trace_frame = frame_count + 1;
    dump_passes = true;
    scan_all = true;
    if (!dump_frames || key_trace_only) {
      std::error_code ec;
      const auto dir = TraceDir() / ("f12_" + std::to_string(trace_frame));
      std::filesystem::create_directories(dir, ec);
      key_trace_dir = dir.string();
      dump_frames = key_trace_dir.c_str();
      key_trace_only = true;
    }
    DC_INFO("gpu: F12: tracing flip %llu into %s", (unsigned long long)trace_frame, dump_frames);
  }
  if (trace_frame) {
    trace = frame_count == trace_frame;
  }
  // DC_PEEK_MEM=<hex addr>,<bytes>: right after a traced flip, once the GPU
  // has run everything encoded so far, how many dwords there are non-zero
  // (guest memory the GPU fills with buffer stores, e.g. GT7's light grid).
  if (trace_frame && frame_count == trace_frame + 1) {
    if (const char *e = std::getenv("DC_PEEK_MEM")) {
      char *q = nullptr;
      const u64 a = std::strtoull(e, &q, 16) & kGpuAddrMask;
      const u64 n = *q == ',' ? std::strtoull(q + 1, nullptr, 0) : 4096;
      if (Readable(a, n)) {
        EndPass();
        EnsureCommandBuffer();
        const u64 frame = trace_frame;
        [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
          const u32 *d = reinterpret_cast<const u32 *>(a);
          u64 nz = 0, first = ~0ull;
          for (u64 i = 0; i < n / 4; ++i) {
            if (d[i]) {
              ++nz;
              first = std::min(first, i);
            }
          }
          DC_INFO("gpu: peek mem %#llx after flip %llu: %llu of %llu dwords non-zero, first at +%#llx = %08x",
                  (unsigned long long)a, (unsigned long long)frame, (unsigned long long)nz,
                  (unsigned long long)(n / 4), (unsigned long long)(first == ~0ull ? 0 : first * 4),
                  first == ~0ull ? 0u : d[first]);
        }];
      }
    }
  }
  // Also right after a traced flip: the surface its probes sampled.
  if (zstat_addr && (frame_count % zstat_every == 0 ||
                     (trace_frame && frame_count == trace_frame + 1))) {
    ZStat();
  }
  {
    const auto now = std::chrono::steady_clock::now();
    if (perf_frames++ == 0) {
      perf_t0 = now;
      perf_gpu_ns = 0;
    } else if (const double dt = std::chrono::duration<double>(now - perf_t0).count(); dt >= 10.0) {
      const u64 frames = perf_frames - 1;
      const double gpu_s = perf_gpu_ns.exchange(0) / 1e9;
      DC_INFO("perf: %.1f fps, gpu %.1f ms/frame (busy %.0f%%), %zu pipelines compiling",
              frames / dt, gpu_s * 1e3 / frames, gpu_s / dt * 100, pending.size());
      LogMemory();
      perf_frames = 1;
      perf_t0 = now;
    }
  }
  if (stats) {
    const auto now = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(now - stat_t0).count();
    static u64 frames = 0;
    ++frames;
    if (dt >= 1.0) {
      // CPU time of the thread that records the draws (the graphics CP).
      static double cp_cpu_last = 0;
      double cp_cpu = 0;
      thread_basic_info_data_t info{};
      mach_msg_type_number_t n = THREAD_BASIC_INFO_COUNT;
      if (draw_thread && thread_info(draw_thread, THREAD_BASIC_INFO,
                                     reinterpret_cast<thread_info_t>(&info), &n) == KERN_SUCCESS) {
        cp_cpu = info.user_time.seconds + info.user_time.microseconds * 1e-6 +
                 info.system_time.seconds + info.system_time.microseconds * 1e-6;
      }
      DC_INFO("gpu stats: CP thread %.1f ms CPU per frame",
              (cp_cpu - cp_cpu_last) * 1e3 / frames);
      cp_cpu_last = cp_cpu;
      DC_INFO(
          "gpu stats: %.1f fps, gpu busy %.0f%%, wait %.0f%%, per frame: %.0f "
          "draws %.0f dispatches %.1f commits %.1f waits %.1f MB hashed, "
          "%zu pipelines compiling",
          frames / dt, gpu_ns.exchange(0) / 1e9 / dt * 100,
          stat_wait_s / dt * 100, stat_draws / double(frames),
          stat_dispatches / double(frames), stat_commits / double(frames),
          stat_waits / double(frames),
          stat_hashed / double(frames) / (1 << 20), pending.size());
      {
        std::vector<std::pair<u64, void *>> top;
        for (auto &[ra, n] : split_by) {
          top.push_back({n, ra});
        }
        std::sort(top.rbegin(), top.rend());
        std::string by;
        for (size_t k = 0; k < top.size() && k < 6; ++k) {
          char buf[48];
          std::snprintf(buf, sizeof(buf), " %p:%llu", top[k].second,
                        (unsigned long long)top[k].first);
          by += buf;
        }
        DC_INFO("gpu stats: %.0f passes/frame, %.0f split from the same targets by%s",
                stat_passes / double(frames), stat_split / double(frames), by.c_str());
        split_by.clear();
        stat_passes = stat_split = 0;
      }
      if (cpu_profile) {
        // Wall clock per frame vs the part of it spent inside draws.
        const double wall = dt * 1e3 / frames;
        const double draw = cpu_draw_s * 1e3 / frames;
        DC_INFO("cpu profile: %.1f ms/frame wall, draw %.1f (resolve %.1f, "
                "texture upload %.1f, hash %.1f), outside draws %.1f",
                wall, draw, cpu_resolve_s * 1e3 / frames,
                cpu_upload_s * 1e3 / frames, cpu_hash_s * 1e3 / frames,
                wall - draw);
        DC_INFO("cpu profile:   GetTexture %.1f ms/frame, %.0f calls/frame, "
                "%.0f same-frame hits, %.0f hashed chunks",
                cpu_gettex_s * 1e3 / frames, cpu_gettex_calls / double(frames),
                cpu_gettex_hits / double(frames), stat_hashed / 65536.0 / frames);
        DC_INFO("cpu profile:   resolve: GetTexture %.1f, GetSampler %.1f, "
                "rest (sgpr+alloc) %.1f",
                cpu_gettex_s * 1e3 / frames, cpu_sampler_s * 1e3 / frames,
                (cpu_resolve_s - cpu_gettex_s - cpu_sampler_s) * 1e3 / frames);
        DC_INFO("cpu profile:   LinearTable %.2f ms/frame, %.0f calls/frame, %.0f bases/frame",
                cpu_linear_s * 1e3 / frames, cpu_linear_calls / double(frames),
                cpu_linear_bases / double(frames));
        cpu_linear_s = 0;
        cpu_linear_calls = cpu_linear_bases = 0;
        DC_INFO("cpu profile:   Resolve cache: %.0f lookups/frame, %.0f%% hits",
                cpu_resolve_calls / double(frames),
                cpu_resolve_calls ? 100.0 * cpu_resolve_hits / cpu_resolve_calls : 0.0);
        cpu_resolve_calls = cpu_resolve_hits = 0;
        DC_INFO("cpu profile:   outside draws: flip %.1f, commit %.1f, "
                "NoteWork %.1f, rest %.1f",
                cpu_flip_s * 1e3 / frames, cpu_commit_s * 1e3 / frames,
                cpu_notework_s * 1e3 / frames,
                (wall - draw - cpu_flip_s * 1e3 / frames -
                 cpu_commit_s * 1e3 / frames - cpu_notework_s * 1e3 / frames));
        std::string sec;
        for (u32 k = 0; k < 8; ++k) {
          char b[48];
          std::snprintf(b, sizeof(b), " %s %.1f", kDrawSections[k], cpu_section_s[k] * 1e3 / frames);
          sec += b;
          cpu_section_s[k] = 0;
        }
        DC_INFO("cpu profile:   draw sections (ms/frame, from each marker to the next):%s", sec.c_str());
        cpu_draw_s = cpu_resolve_s = cpu_hash_s = cpu_upload_s = 0;
        cpu_gettex_s = cpu_notework_s = cpu_flip_s = cpu_commit_s = 0;
        cpu_sampler_s = cpu_sgpr_s = 0;
        cpu_gettex_calls = cpu_gettex_hits = 0;
      }
      if (tex_cache_check) {
        DC_INFO("tex cache check: %llu hits verified, %llu mismatches",
                (unsigned long long)tex_cache_checks,
                (unsigned long long)tex_cache_mismatches);
      }
      stat_hashed = 0;
      if (pass_timing) {
        PassTimingReport(frames);
      }
      frames = stat_commits = stat_waits = stat_draws = stat_dispatches = 0;
      if (profile) {
        ProfileReport();
      }
      stat_wait_s = 0;
      stat_t0 = now;
    }
  }
  if (edr_armed) {
    EdrCapture();
  }
  EndPass();
  EnsureCommandBuffer();

  __strong id<MTLTexture> &dst = frames[frame_index];
  if (!dst || dst.width != f.width || dst.height != f.height) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                     width:f.width
                                    height:f.height
                                 mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    d.storageMode = MTLStorageModePrivate;
    dst = [device newTextureWithDescriptor:d];
  }

  id<MTLTexture> src = nil;
  float scale[2] = {1.0f, 1.0f};
  RenderTarget *rt = Latest(color_targets, f.address);
  if (rt && rt->tex) {
    id<MTLTexture> t = rt->tex;
    // Scan-out shows the stored bytes: sample sRGB targets without decoding.
    if (t.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB) {
      t = [t newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm];
    }
    src = t;
    scale[0] = static_cast<float>(f.width) / static_cast<float>(rt->width);
    scale[1] = static_cast<float>(f.height) / static_cast<float>(rt->height);
  } else if (Readable(f.address, u64(f.pitch) * f.height * 4)) {
    // Nothing rendered there on the GPU: the CPU wrote the image.
    MTLTextureDescriptor *d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                     width:f.width
                                    height:f.height
                                 mipmapped:NO];
    d.storageMode = MTLStorageModeManaged;
    d.usage = MTLTextureUsageShaderRead;
    src = [device newTextureWithDescriptor:d];
    [src replaceRegion:MTLRegionMake2D(0, 0, f.width, f.height)
           mipmapLevel:0
             withBytes:reinterpret_cast<const void *>(f.address)
           bytesPerRow:f.pitch * 4];
  }
  if (src && copy_pipeline) {
    MTLRenderPassDescriptor *rp =
        [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = dst;
    rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> e = [cmd renderCommandEncoderWithDescriptor:rp];
    [e setRenderPipelineState:copy_pipeline];
    [e setFragmentTexture:src atIndex:0];
    [e setFragmentBytes:scale length:sizeof(scale) atIndex:0];
    [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [e endEncoding];
  }
  id<MTLTexture> edr_frame = nil;
  if (edr_have && src && edr_peak.width == dst.width && edr_peak.height == dst.height) {
    __strong id<MTLTexture> &o = edr_out[frame_index];
    if (!o) {
      MTLTextureDescriptor *d = [MTLTextureDescriptor
          texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                       width:dst.width
                                      height:dst.height
                                   mipmapped:NO];
      d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
      d.storageMode = MTLStorageModePrivate;
      o = [device newTextureWithDescriptor:d];
    }
    id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
    [e setComputePipelineState:edr_mask_pipeline];
    [e setTexture:edr_base_view atIndex:0];
    [e setTexture:dst atIndex:1];
    [e setTexture:edr_peak atIndex:2];
    [e setTexture:o atIndex:3];
    [e dispatchThreads:MTLSizeMake(dst.width, dst.height, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [e endEncoding];
    edr_frame = o;
  }
  edr_have = false;
  if (debug_vs_buf) {
    Commit(true);
    EnsureCommandBuffer();
    const float *d = static_cast<const float *>(debug_vs_buf.contents);
    if (debug_vs_domain) { // per-patch corner clip positions -> file, summary in the log
      if (dump_frames) {
        std::string file = std::string(dump_frames) + "/domain_corners.bin";
        std::ofstream(file, std::ios::binary).write(static_cast<const char *>(debug_vs_buf.contents),
                                                     static_cast<std::streamsize>(debug_vs_buf.length));
      }
      u32 run = 0, bad = 0;
      for (u32 c = 0; c < 4096; ++c) {
        const float *q = d + c * 4;
        u32 bits;
        std::memcpy(&bits, q, 4);
        if (bits == 0xFFFFFFFFu) continue;
        ++run;
        bad += !std::isfinite(q[0]) || !std::isfinite(q[1]) || !std::isfinite(q[3]) || std::fabs(q[3]) < 1e-4f;
      }
      DC_INFO("gpu: domain vertices: %u", static_cast<const u32 *>(debug_vs_buf.contents)[6144 * 4]);
      u32 out = 0;
      for (u32 v = 0; v < 1024; ++v) {
        const float *q = d + (4096 + v * 2) * 4;
        if (q[6] == 1.0f) {
          if (out++ < 16) {
            DC_INFO("gpu: domain outlier patch %u uv (%g %g): clip (%g %g %g %g)", v, q[4], q[5], q[0], q[1], q[2], q[3]);
          }
        }
      }
      DC_INFO("gpu: domain corners: %u written, %u non-finite or w~0; %u patches with outliers", run, bad, out);
      debug_vs_domain = false;
      debug_vs_buf = nil;
    }
    for (u32 v = 0; v < 8 && !debug_vs_domain; ++v) {
      DC_INFO("gpu: vs out %u: raw (%g %g %g %g) -> clip (%g %g %g %g)", v,
              d[v * 8], d[v * 8 + 1], d[v * 8 + 2], d[v * 8 + 3], d[v * 8 + 4],
              d[v * 8 + 5], d[v * 8 + 6], d[v * 8 + 7]);
    }
    debug_vsout = 0;
  }
  if (src && dump_frames && (key_trace_only ? trace : frame_count % dump_every == 0)) {
    DumpFrame(dst);
  }
  if (!pass_dumps.empty()) {
    Commit(true);
    EnsureCommandBuffer();
    WritePassDumps();
  }
  if (probe_buf && !probe_slots.empty()) {
    Commit(true);
    EnsureCommandBuffer();
    PrintProbes();
  }
  if (!peek_addrs.empty()) {
    Commit(true);
    EnsureCommandBuffer();
    for (u64 a : peek_addrs) {
      if (Readable(a, 32)) {
        const float *v = reinterpret_cast<const float *>(a);
        DC_INFO("gpu: peek %#llx: %g %g %g %g %g %g %g %g", a, v[0], v[1], v[2],
                v[3], v[4], v[5], v[6], v[7]);
      }
    }
    peek_addrs.clear();
  }
  if (const char *peek = std::getenv("DC_PEEK");
      peek && trace) { // debug: DC_PEEK=addr,addr: guest floats
    Commit(true);
    EnsureCommandBuffer();
    for (const char *q = peek; *q;) {
      char *end = nullptr;
      const u64 a = std::strtoull(q, &end, 16);
      if (Readable(a, 32)) {
        const float *v = reinterpret_cast<const float *>(a);
        DC_INFO("gpu: peek %#llx: %g %g %g %g %g %g %g %g", a, v[0], v[1], v[2],
                v[3], v[4], v[5], v[6], v[7]);
      }
      q = *end ? end + 1 : end;
    }
  }
  Commit(false);
  if (src) {
    latest_boost.store((__bridge void *)edr_frame, std::memory_order_release);
    latest_frame.store((__bridge void *)dst, std::memory_order_release);
    frame_index = (frame_index + 1) % 3;
  }
  if (frame_count <= 3 || frame_count % 300 == 0) {
    DC_INFO("gpu: flip #%llu %#llx %ux%u (%s), %llu draws so far", frame_count,
            f.address, f.width, f.height, rt ? "gpu target" : "cpu memory",
            draw_count);
  }
}

void RequestFrameTrace() {
  g_trace_request.store(true, std::memory_order_release);
}

void RequestGpuCapture() {
  g_capture_request.store(true, std::memory_order_release);
}

MetalRenderer::MetalRenderer(Device &device) : impl(std::make_unique<Impl>()) {
  impl->device = (__bridge id<MTLDevice>)device.Handle();
  impl->queue = (__bridge id<MTLCommandQueue>)device.Queue();
  impl->Init();
}

MetalRenderer::~MetalRenderer() {
  Gpu::LtLock lk(impl->lock);
  impl->Commit(true);
}

void MetalRenderer::Draw(const RegFile &regs, const DrawCall &draw) {
  Gpu::LtLock lk(impl->lock);
  if (!impl->draw_thread) {
    impl->draw_thread = pthread_mach_thread_np(pthread_self());
  }
  @autoreleasepool {
    impl->NoteWork("  draw vs %#llx ps %#llx count %u x%u (idx %u)\n",
                   regs.ProgramAddr(Liverpool::Reg::SPI_SHADER_PGM_LO_VS),
                   regs.ProgramAddr(Liverpool::Reg::SPI_SHADER_PGM_LO_PS),
                   draw.count, draw.instances, draw.indexed);
    impl->Draw(regs, draw);
    if (impl->trace && !impl->scan_addrs.empty()) {
      impl->EndPass(); // DC_SCAN_MAX=<addrs>: scan after every draw
    }
    if (impl->pass_timing && impl->enc && !impl->ts_cmd.empty() &&
        !impl->ts_cmd.back().compute) {
      Impl::TsPass &t = impl->ts_cmd.back();
      t.verts += u64(draw.count) * draw.instances;
      if (!t.draws++) {
        t.shader = regs.ProgramAddr(Liverpool::Reg::SPI_SHADER_PGM_LO_PS);
      }
    }
    if (impl->profile) {
      char what[64];
      std::snprintf(what, sizeof(what), "vs %#llx ps %#llx",
                    regs.ProgramAddr(Liverpool::Reg::SPI_SHADER_PGM_LO_VS),
                    regs.ProgramAddr(Liverpool::Reg::SPI_SHADER_PGM_LO_PS));
      impl->ProfileCut(what);
    }
  }
}

void MetalRenderer::Dispatch(const RegFile &regs, std::uint32_t x,
                             std::uint32_t y, std::uint32_t z) {
  Gpu::LtLock lk(impl->lock);
  @autoreleasepool {
    impl->NoteWork("  dispatch cs %#llx %ux%ux%u\n",
                   regs.ProgramAddr(Liverpool::Reg::COMPUTE_PGM_LO), x, y, z);
    impl->Dispatch(regs, x, y, z);
    if (impl->profile) {
      char what[64];
      std::snprintf(what, sizeof(what), "cs %#llx %ux%ux%u",
                    regs.ProgramAddr(Liverpool::Reg::COMPUTE_PGM_LO), x, y, z);
      impl->ProfileCut(what);
    }
  }
}

bool MetalRenderer::DispatchIndirect(const RegFile &regs, std::uint64_t args) {
  Gpu::LtLock lk(impl->lock);
  id<MTLBuffer> buf = nil;
  u64 off = 0;
  if ((args & 3) || !impl->Locate(args, 12, &buf, &off)) {
    return false;
  }
  @autoreleasepool {
    impl->NoteWork("  dispatch cs %#llx indirect %#llx\n",
                   regs.ProgramAddr(Liverpool::Reg::COMPUTE_PGM_LO), args);
    impl->indirect_args = args;
    impl->Dispatch(regs, 1, 1, 1);
    impl->indirect_args = 0;
    if (impl->profile) {
      char what[64];
      std::snprintf(what, sizeof(what), "cs %#llx indirect",
                    regs.ProgramAddr(Liverpool::Reg::COMPUTE_PGM_LO));
      impl->ProfileCut(what);
    }
  }
  return true;
}

void MetalRenderer::OnSubmit() {
  Gpu::LtLock lk(impl->lock);
  // DC_SERIAL: wait for the GPU after every submit (debug: rules out CPU/GPU
  // races on guest memory).
  static const bool serial = std::getenv("DC_SERIAL") != nullptr;
  @autoreleasepool {
    impl->Commit(serial);
  }
}

void MetalRenderer::WaitIdle() {
  Gpu::LtLock lk(impl->lock);
  @autoreleasepool {
    impl->Commit(true);
  }
}

void MetalRenderer::Signal(std::function<void()> done) {
  Gpu::LtLock lk(impl->lock);
  @autoreleasepool {
    impl->Signal(std::move(done));
  }
}

void MetalRenderer::Flip(const FlipInfo &flip) {
  flips.fetch_add(1, std::memory_order_relaxed);
  Gpu::LtLock lk(impl->lock);
  @autoreleasepool {
    impl->Flip(flip);
  }
}

bool MetalRenderer::DmaData(std::uint32_t dst_sel, std::uint64_t dst,
                            std::uint32_t src_sel, std::uint64_t src,
                            std::uint32_t bytes) {
  Gpu::LtLock lk(impl->lock);
  @autoreleasepool {
    return impl->DmaData(dst_sel, dst, src_sel, src, bytes);
  }
}

void MetalRenderer::OnContextRegs(const std::uint32_t *body,
                                  std::uint32_t body_dwords) {
  // Only color target and depth info writes matter: skip the lock (which the
  // compute queues contend for) for every other register.
  if (body_dwords < 2) {
    return;
  }
  const u32 reg = kContextRegBase + body[0];
  const u32 base0 = Reg::CB_COLOR0_BASE;
  if ((reg < base0 || reg >= base0 + 8 * Reg::kColorTargetStride) &&
      reg != Reg::DB_Z_INFO) {
    return;
  }
  Gpu::LtLock lk(impl->lock);
  impl->OnContextRegs(body, body_dwords);
}

void MetalRenderer::NoteConstDump(std::uint64_t addr, std::uint32_t bytes) {
  std::scoped_lock lk(impl->dump_lock);
  impl->NoteConstDump(addr, bytes);
}

void MetalRenderer::SaveTargets(const char *path) {
  Gpu::LtLock lk(impl->lock);
  @autoreleasepool {
    impl->SaveTargets(path);
  }
}

void MetalRenderer::LoadTargets(const char *path) {
  Gpu::LtLock lk(impl->lock);
  @autoreleasepool {
    impl->LoadTargets(path);
  }
}

std::uint8_t *MetalRenderer::Gds() {
  return static_cast<std::uint8_t *>(impl->gds.contents) + Gpu::GdsBase();
}

void *MetalRenderer::LatestFrame() const {
  return impl->latest_frame.load(std::memory_order_acquire);
}

void MetalRenderer::SetEdr(bool enabled) { impl->edr_on = enabled; }

void *MetalRenderer::LatestBoost() const {
  return impl->latest_boost.load(std::memory_order_acquire);
}

} // namespace Gpu::Metal
