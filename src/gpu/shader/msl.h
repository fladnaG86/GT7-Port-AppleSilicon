#pragma once

// GCN -> Metal Shading Language translation.
//
// Every GCN wave lane becomes one Metal invocation. Scalar and vector
// registers turn into local variables; 64-bit lane masks (EXEC, VCC, v_cmp
// results) are kept per lane as all-ones / all-zero words, so the bitwise
// scalar ops that combine masks work unchanged. Guest memory is reached
// through two Metal buffers that alias the whole direct-memory object, with a
// page table translating guest virtual addresses to offsets.

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "common/types.h"
#include "gpu/shader/program.h"

namespace Gpu::Shader {

// Fixed resource bindings shared by every generated shader.
namespace Binding {
constexpr u32 UserData = 0; // constant uint[16]
constexpr u32 Mem0 = 1;     // device uint*: direct memory [0, split)
constexpr u32 Mem1 = 2;     // device uint*: direct memory [split, end)
constexpr u32 MemMap = 3;   // constant DcMap
constexpr u32 Draw = 4;     // constant DcDraw (vertex stage)
constexpr u32 Gds = 5;      // device uint[16K]: the queue's 64 KB GDS partition (compute stage)
constexpr u32 Linear = 8;   // constant uint[]: translated linear bases (MslOptions::linear_mask)
constexpr u32 GsIndex = 9;  // device uint*: copy-VS indices a GS kernel writes (MslOptions::ring)
constexpr u32 TessLds = 10; // device uint*: LS/HS LDS slices (MslOptions::tess)
constexpr u32 TessFactors = 11; // device uint*: tess factors the HS stores (MslOptions::tess)
constexpr u32 RingEsUd = 12;    // constant uint[16]: the ES's user data (MslOptions::fused)
constexpr u32 RingGsUd = 13;    // constant uint[16]: the GS's user data (MslOptions::fused)
constexpr u32 TessParams = 14;  // constant uint4: patches per group, LDS dwords per group (MslOptions::tess)
} // namespace Binding

// Geometry-shader emulation. Metal has no GS stage: the ES and the GS of a GS
// pipeline run as compute kernels that write the ESGS / GSVS rings in guest
// memory as the hardware would, and the copy VS then draws from the GSVS
// ring. The ring offsets the hardware would hand out (es2gs, gs vertex
// offsets, gs2vs, the copy VS's vertex) are chosen here, consistently with
// the swizzled-buffer address math: ES waves are 64 threads, GS groups 16
// (the GSVS ring's index stride), each with its own slice of the ring.
enum class RingStage : u8 { None, Export, Geometry };
struct RingInfo {
    RingStage stage = RingStage::None;
    u32 esgs_itemsize = 0; // dwords per ES vertex (VGT_ESGS_RING_ITEMSIZE)
    u32 vert_itemsize = 0; // GS: dwords per emitted vertex (VGT_GS_VERT_ITEMSIZE)
    u32 slots = 0;         // GS: emitted vertices per output-dword plane of a 16-lane group
    u32 input_verts = 1;   // vertices per input primitive: 1 point, 2 line
    u32 index_size = 0;    // ES: bytes per index read from Binding::GsIndex; 0 = not indexed
    bool operator==(const RingInfo&) const = default;

    // Byte size of one ES wave's / one GS group's slice of its ring.
    u32 EsWaveBytes() const { return esgs_itemsize * 64 * 4; }
    u32 GsGroupBytes() const { return vert_itemsize * slots * 16 * 4; }
    // Copy-VS indices a GS thread writes: its strip as a triangle list.
    u32 IndicesPerPrim() const { return slots > 2 ? (slots - 2) * 3 : 0; }
};

// Tessellation emulation (VGT_SHADER_STAGES_EN.HS_EN). Metal's tessellator
// has no LS/HS stages: the LS and HS run as compute kernels with LDS kept in
// a device buffer (one slice per HS threadgroup, so the HS kernel finds what
// the LS wrote), the HS writes tess factors and off-chip outputs to guest
// memory through its V#s, and the VS runs as Metal's post-tessellation vertex
// function. The system SGPRs / VGPRs the hardware would hand each stage are
// built here from the threadgroup and patch ids, consistently between stages.
enum class TessStage : u8 { None, Local, Hull, Domain };
struct TessInfo {
    TessStage stage = TessStage::None;
    u32 patches = 0;     // patches per HS threadgroup (VGT_LS_HS_CONFIG.NUM_PATCHES)
    u32 cp_in = 0;       // LS vertices per patch
    u32 cp_out = 0;      // HS threads per patch
    u32 lds_dwords = 0;  // LS/HS: LDS slice per threadgroup (Binding::TessLds)
    u32 oc_bytes = 0;    // off-chip bytes per threadgroup (the oc offset SGPR)
    u32 tf_bytes = 0;    // HS: tess-factor bytes per patch (the tf offset SGPR)
    bool oc_sgpr = false; // RSRC2.OC_LDS_EN: the off-chip offset SGPR follows user data
    u8 tf_sgpr = 0xFF;   // HS: the tf offset SGPR; dword stores with it as soffset go to Binding::TessFactors
    u8 index_size = 0;   // LS: bytes per index read from Binding::GsIndex; 0 = not indexed
    u8 domain = 2;       // VGT_TF_PARAM.TYPE: 1 tri, 2 quad
    bool operator==(const TessInfo&) const = default;
};

// Host-side mirrors of the MSL structs in the prelude.
// Header of the page table bound at Binding::MemMap; `pages` s32 entries
// follow it, each the page's (phys - va) in 16 KB units, or kUnmapped.
struct GpuMemMap {
    static constexpr u32 kMinShift = 14;
    static constexpr u32 kMaxShift = 21;
    static constexpr s32 kUnmapped = INT32_MIN;
    u64 base;
    u64 pages;
    u64 split;
    u32 shift;
    u32 pad;
    u64 limit; // physical size
    // Fast window: guest [fast_va, fast_va + fast_size) maps linearly to byte
    // fast_off of mem buffer fast_hi, so shaders skip the table there.
    u64 fast_va;
    u64 fast_size;
    u64 fast_off;
    u32 fast_hi;
    u32 pad2;
};
struct alignas(16) GpuDrawConsts {
    float vp_scale[4];  // viewport scale (1 where the transform is disabled)
    float vp_offset[4]; // viewport offset (0 where disabled)
    float rt_scale[2];  // 2 / render target size
    u32 vte;            // PA_CL_VTE_CNTL
    u32 pad;
    // Fused GS pipeline (MslOptions::FusedRing): the guest's ESGS / GSVS ring
    // bases and the offsets to this draw's own copies of them, so draws in
    // flight at once do not share the rings (u64 wrap-around arithmetic).
    u64 esgs_base, esgs_bias, gsvs_base, gsvs_bias;
};

enum class TexDim : u8 { Tex2D, Tex2DArray, Tex3D };
// Uscaled8/16: USCALED unsigned normalized storage the shader reads as the
// integer value in a float (0..255 / 0..65535). Bound as the UNORM format, so
// it filters, and scaled after the read. Appended so older keys stay valid.
enum class TexSample : u8 { Float, Uint, Sint, Uscaled8, Uscaled16 };
inline bool FloatSample(TexSample s) {
    return s == TexSample::Float || s == TexSample::Uscaled8 || s == TexSample::Uscaled16;
}

struct TextureType {
    TexDim dim = TexDim::Tex2D;
    TexSample sample = TexSample::Float;
    u8 compare = 3; // depth-compare sampling: S# DEPTH_COMPARE_FUNC (ref <op> texel), default LEQUAL
    bool cube = false; // a cube (Tex2DArray of faces) sampled with V_CUBE* coordinates
    bool ufloat = false; // unsigned small-float storage (R11G11B10): writes clamp like the CB does
    // S# FORCE_UNNORMALIZED: coordinates in texels of the level sampled (an explicit
    // LOD addresses that mip, GT7's hull shaders read their Hi-Z pyramid so).
    bool unnorm = false;
    bool operator==(const TextureType&) const = default;
};

enum class OutType : u8 { Float, Uint, Sint };

constexpr u32 kProbeWords = 256;
constexpr u32 kProbeExportNan = 0xFFFFFFFEu; // probe_pc: at color exports that are NaN/Inf
constexpr u32 kProbeExportAny = 0xFFFFFFFDu; // probe_pc: at every color export
constexpr u32 kProbeBinding = 7;
constexpr u32 kDepthBoundsBuffer = 6;
// DC_GUARD_REPORT=1: pixel shaders count loop-budget exhaustions here (one
// atomic_uint slot per shader, bound with an offset by the renderer).
constexpr u32 kGuardReportBinding = 15;
bool GuardReportEnabled();
constexpr u32 kDepthBoundsTexture = 30;

struct MslOptions {
    // Vertex.
    bool rect_list = false; // expand RECTLIST: 3 vertices -> 2 triangles
    bool debug_capture = false; // write positions to buffer(6) (debugging)
    bool points = false;          // point list: VsOut carries [[point_size]]
    bool vtx_point_size = false;  // PA_CL_VS_OUT_CNTL.USE_VTX_POINT_SIZE: POS1.x is the size
    bool vtx_layer = false;       // USE_VTX_RENDER_TARGET_INDX on a layered target: POS1.z is the slice
    // Fetch loads the host does with a vertex descriptor: attribute k is the
    // load at vertex_attrs[k].pc (Program::vertex_inputs), read from
    // [[attribute(k)]] instead of guest memory.
    struct VertexAttr {
        enum Kind : u8 { Float, Uint, Sint, Uscaled, Sscaled };
        u32 pc = 0;
        Kind kind = Float;
        u32 swizzle = 0; // V# word 3 for dst_sel (MUBUF), 0 = none (MTBUF)
    };
    std::span<const VertexAttr> vertex_attrs; // borrowed, like `textures`
    // Fragment.
    std::array<u32, 32> ps_input_cntl{}; // SPI_PS_INPUT_CNTL_n
    u32 vs_param_mask = 0;               // params the paired VS exports
    u32 ps_input_addr = 0;               // SPI_PS_INPUT_ADDR
    u32 col_format = 0;                  // SPI_SHADER_COL_FORMAT
    std::array<OutType, 8> mrt_type{};
    u32 mrt_ufloat = 0; // MRTs stored as unsigned small floats (R11G11B10)
    bool depth_target = true; // false drops depth exports (no depth attachment)
    // DB_ALPHA_TO_MASK on a single-sample target: MRT0 alpha below 0.5
    // discards the fragment (GT7's foliage cut-outs).
    bool alpha_to_mask = false;
    // Dual-source blending: the MRT1 export is the second blend source,
    // [[color(0), index(1)]], declared even when the shader never writes it.
    bool dual_source = false;
    // Guest memory helpers read the page-table header from registers filled
    // once at entry instead of from the buffer on every access.
    bool map_hoist = false;
    // Program::linear_bases the host translated for this draw (bit k = base k):
    // their accesses index memory directly. Entry k of the buffer at
    // Binding::Linear is the base's dword offset << 1 | 1 for the upper buffer
    // (entry 2k).
    u32 linear_mask = 0;
    // Bases of linear_mask whose V# stride is a multiple of 4: buffer accesses on
    // them with constant, dword-aligned offsets drop the unaligned fallback.
    u32 linear_aligned = 0;
    // Binding::Linear holds two entries per base: 2k as above, 2k + 1 the bytes
    // mapped contiguously from base k (0: none): accesses on bases outside
    // linear_mask take entry k directly when they fall inside, else walk.
    // Set by the paths that fill and bind the whole table.
    bool linear_window = false;
    // Compute only: this program is the ES or GS of a GS pipeline (RingInfo).
    RingInfo ring;
    // The ES / GS as a vertex function with rasterization off (one vertex per
    // thread) instead of a compute kernel: it runs inside the render pass
    // that draws the copy VS, so no pass split. Needs no LDS, barriers or
    // cross-lane ops in the program.
    bool ring_vertex = false;
    // The ES / GS as `static void dc_es(...)` / `dc_gs(...)` without prelude
    // or entry point, for a fused copy VS (FusedRing).
    bool ring_fused = false;
    // Copy VS of a GS pipeline with the ES and GS fused in: each copy-VS
    // vertex runs the ES for its primitive's input vertices, the GS for the
    // primitive (both write the guest rings as the compute path would, every
    // vertex of a primitive writing the same values), and takes its ring
    // index from the GS's strip. No barriers, no index buffer, no pass split.
    struct FusedRing {
        bool on = false;
        u32 input_verts = 1;
        u32 ipp = 0;          // copy-VS indices per primitive (RingInfo::IndicesPerPrim)
        u8 es_index_size = 0; // ES reads its vertex index from Binding::GsIndex (0: thread id)
        // Borrowed MSL of dc_es / dc_gs (the renderer's cache owns them and
        // never drops entries): copying them per draw cost the GPU thread
        // more than the draw itself.
        const std::string* es_src = nullptr;
        const std::string* gs_src = nullptr;
    } fused;
    // LS / HS (compute) or domain VS (vertex) of a tessellation pipeline.
    TessInfo tess;
    // DB_DEPTH_CONTROL.DEPTH_BOUNDS_ENABLE: discard where the stored depth (a copy at
    // texture(kDepthBoundsTexture)) is outside [min, max] (buffer(kDepthBoundsBuffer)).
    bool depth_bounds = false;
    // Register probe (debugging): before the instruction at `probe_pc`, the
    // fragment at pixel (probe_x, probe_y) -- for compute, the thread at that
    // global grid position -- writes its
    // registers to buffer(7): [0] hit, [1] exec, [2] vcc, [3] m0, [8 + n] vN
    // (n < 96), [104 + n] sN (n < 104). kProbeWords per slot.
    u32 probe_pc = ~0u;
    u32 probe_x = 0, probe_y = 0;
    // Both. Borrowed: the caller keeps the storage alive for the call.
    std::span<const TextureType> textures; // per Program::images slot
};

// Returns the MSL source; the entry point is "main0". Opcodes that could not
// be translated are appended to `unsupported`.
std::string GenerateMsl(const Program& p, const MslOptions& o, std::vector<std::string>* unsupported);

// TessInfo::patches and ::lds_dwords reach the LS / HS / domain shaders as
// Binding::TessParams, so they are not part of the pipeline key: the game
// changes both from draw to draw (NUM_PATCHES and the LS's LDS_SIZE follow
// the patch count) and every pair was a new compile. Opt-in for now:
// DC_TESS_UNIFORMS=1 (default: baked into the MSL as constants, as before).
bool TessBakedConsts();

} // namespace Gpu::Shader
