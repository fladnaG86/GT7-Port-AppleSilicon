#pragma once

// MSL helpers shared by every translated shader.

namespace Gpu::Shader {

inline constexpr const char* kMslPrelude = R"MSL(#include <metal_stdlib>
using namespace metal;

// Guest VA -> physical page table: a header followed by one entry per page,
// the page's (phys - va) in 16 KB units (INT_MIN = unmapped).
struct DcMap { ulong base; ulong pages; ulong split; uint shift; uint pad; ulong limit;
               ulong fva; ulong fsize; ulong foff; uint fhi; uint pad2; };
struct DcDraw { float4 vp_scale; float4 vp_offset; float2 rt_scale; uint vte; uint pad;
                ulong esgs_base; ulong esgs_bias; ulong gsvs_base; ulong gsvs_bias; };
// DcMap's header copied into registers once per invocation (MslOptions::map_hoist).
struct DcMapL { ulong base; ulong pages; ulong split; ulong limit; uint shift; device const int* entries;
                ulong fva; ulong fsize; ulong foff; uint fhi; };
struct DcMem { device uint* lo; device uint* hi; device const DcMap* map; DcMapL mapl; constant uint* lin; };
// A base the host translated (MslOptions::linear_mask): guest `va` maps to `p`
// and the access stays inside the range the host checked.
struct DcLin { device uint* p; ulong va; };

// ---- guest memory ----
static inline device uint* dc_ptr(thread const DcMem& m, ulong va) {
    ulong fo = va - m.map->fva;
    if (fo < m.map->fsize) return (m.map->fhi ? m.hi : m.lo) + ((fo + m.map->foff) >> 2);
    ulong i = (va - m.map->base) >> m.map->shift;
    if (i >= m.map->pages) return nullptr;
    int e = ((device const int*)(m.map + 1))[i];
    if (e == int(0x80000000u)) return nullptr;
    ulong p = va + ulong(long(e) << 14);
    if (p >= m.map->limit) return nullptr;
    return p < m.map->split ? m.lo + (p >> 2) : m.hi + ((p - m.map->split) >> 2);
}
// `bytes` contiguous bytes at dword-aligned `va` (one translation for a
// multi-dword load), or nullptr.
static inline device uint* dc_ptr_n(thread const DcMem& m, ulong va, uint bytes) {
    device uint* p = dc_ptr(m, va);
    ulong last = va + ulong(bytes - 4u);
    if (((va ^ last) >> m.map->shift) != 0ul && dc_ptr(m, last) != p + ((bytes - 4u) >> 2)) return nullptr;
    return p;
}
static inline device uint* dc_ptr(thread const DcLin& m, ulong va) { return m.p + ((va - m.va) >> 2); }
static inline device uint* dc_ptr_n(thread const DcLin& m, ulong va, uint bytes) { return dc_ptr(m, va); }
// Linear base `k`: entry 2k = dword offset << 1 | upper memory buffer,
// entry 2k + 1 = its window (DcLinW).
static inline DcLin dc_lin(thread const DcMem& m, uint k, ulong va) {
    uint e = m.lin[2u * k];
    return DcLin{((e & 1u) ? m.hi : m.lo) + (e >> 1), va};
}
// The page-table walk of dc_ptr / dc_ptr_n as one out-of-line copy, for the
// rare accesses outside a window: inlined at every access it doubled the
// shaders and tripled their compile time.
__attribute__((noinline)) static device uint* dc_walk(device uint* lo, device uint* hi, device const DcMap* map,
                                                      ulong va) {
    ulong fo = va - map->fva;
    if (fo < map->fsize) return (map->fhi ? hi : lo) + ((fo + map->foff) >> 2);
    ulong i = (va - map->base) >> map->shift;
    if (i >= map->pages) return nullptr;
    int e = ((device const int*)(map + 1))[i];
    if (e == int(0x80000000u)) return nullptr;
    ulong p = va + ulong(long(e) << 14);
    if (p >= map->limit) return nullptr;
    return p < map->split ? lo + (p >> 2) : hi + ((p - map->split) >> 2);
}
__attribute__((noinline)) static device uint* dc_walk_n(device uint* lo, device uint* hi, device const DcMap* map,
                                                        ulong va, uint bytes) {
    device uint* p = dc_walk(lo, hi, map, va);
    ulong last = va + ulong(bytes - 4u);
    if (((va ^ last) >> map->shift) != 0ul && dc_walk(lo, hi, map, last) != p + ((bytes - 4u) >> 2)) return nullptr;
    return p;
}
// A base outside linear_mask with a window (MslOptions::linear_window): `win`
// bytes from `va` are mapped contiguously at `p` (a multiple of 4, 0 = none).
// Inside, no page-table walk; outside (or below va), the walk as before.
struct DcLinW { device uint* p; ulong va; ulong win; device uint* lo; device uint* hi; device const DcMap* map; };
static inline device uint* dc_ptr(thread const DcLinW& m, ulong va) {
    ulong o = va - m.va;
    return o < m.win ? m.p + (o >> 2) : dc_walk(m.lo, m.hi, m.map, va);
}
static inline device uint* dc_ptr_n(thread const DcLinW& m, ulong va, uint bytes) {
    ulong o = va - m.va;
    return o < m.win && ulong(bytes) <= m.win - o ? m.p + (o >> 2) : dc_walk_n(m.lo, m.hi, m.map, va, bytes);
}
static inline DcLinW dc_linw(thread const DcMem& m, uint k, ulong va) {
    uint e = m.lin[2u * k];
    return DcLinW{((e & 1u) ? m.hi : m.lo) + (e >> 1), va, ulong(m.lin[2u * k + 1u]), m.lo, m.hi, m.map};
}
// A linear base whose every access is known dword aligned
// (MslOptions::linear_aligned): loads need no unaligned fallback.
struct DcLinA { device uint* p; ulong va; };
static inline device uint* dc_ptr(thread const DcLinA& m, ulong va) { return m.p + ((va - m.va) >> 2); }
static inline device uint* dc_ptr_n(thread const DcLinA& m, ulong va, uint bytes) { return dc_ptr(m, va); }
static inline DcLinA dc_lina(thread const DcMem& m, uint k, ulong va) {
    uint e = m.lin[2u * k];
    return DcLinA{((e & 1u) ? m.hi : m.lo) + (e >> 1), va};
}
static inline uint dc_ld(thread const DcLinA& m, ulong va) { return *dc_ptr(m, va); }
template <typename M>
static inline bool dc_aligned(thread const M& m, ulong va) { return (va & 3ul) == 0ul; }
static inline bool dc_aligned(thread const DcLinA& m, ulong va) { return true; }
template <typename M>
static inline uint dc_ld(thread const M& m, ulong va) {
    device uint* p = dc_ptr(m, va & ~3ul);
    uint lo = p ? *p : 0u;
    uint sh = uint(va & 3ul) * 8u;
    if (sh == 0u) return lo;
    device uint* q = dc_ptr(m, (va & ~3ul) + 4ul);
    uint hi = q ? *q : 0u;
    return (lo >> sh) | (hi << (32u - sh));
}
template <typename M>
static inline void dc_st(thread const M& m, ulong va, uint v) {
    device uint* p = dc_ptr(m, va & ~3ul);
    if (p) *p = v;
}
// Byte / short stores: atomic and+or, since neighbouring lanes may write the
// other bytes of the same dword (a plain read-modify-write also trips the
// AMD Metal compiler).
template <typename M>
static inline void dc_st_part(thread const M& m, ulong va, uint v, uint bytes) {
    device uint* p = dc_ptr(m, va & ~3ul);
    if (!p) return;
    uint sh = uint(va & 3ul) * 8u;
    uint mask = (bytes == 1u ? 0xffu : 0xffffu) << sh;
    device atomic_uint* a = (device atomic_uint*)p;
    atomic_fetch_and_explicit(a, ~mask, memory_order_relaxed);
    atomic_fetch_or_explicit(a, (v << sh) & mask, memory_order_relaxed);
}
// Swizzled buffer (V# SWIZZLE_EN / ADD_TID_ENABLE): index_stride records
// interleave at element_size granularity (word 3 bits 21-22, 19-20).
static inline ulong dc_swizzled(uint idx, uint off, uint stride, uint w3) {
    uint is = 8u << ((w3 >> 21) & 3u);
    uint es = 2u << ((w3 >> 19) & 3u);
    return ulong(idx / is) * stride * is + ulong(off / es) * es * is + ulong(idx % is) * es + (off % es);
}
static inline ulong dc_vbase(uint a, uint b) { return ulong(a) | (ulong(b & 0xfffu) << 32); }
static inline uint dc_vstride(uint b) { return (b >> 16) & 0x3fffu; }

// ---- typed buffer fetch (BUF_DATA_FORMAT / BUF_NUM_FORMAT) ----
static inline float dc_ufloat(uint v, uint mbits) {
    uint e = (v >> mbits) & 31u;
    uint mt = v & ((1u << mbits) - 1u);
    float fm = float(mt) / float(1u << mbits);
    if (e == 0u) return ldexp(fm, -14);
    if (e == 31u) return mt == 0u ? INFINITY : NAN;
    return ldexp(1.0f + fm, int(e) - 15);
}
static inline uint4 dc_fmt_convert(uint4 r, uint4 bits, uint nfmt, uint n) {
    bool is_int = nfmt == 4u || nfmt == 5u;
    uint4 o = uint4(0u, 0u, 0u, is_int ? 1u : 0x3f800000u);
    for (uint k = 0u; k < n; ++k) {
        uint v = r[k];
        uint b = bits[k];
        int sv = b >= 32u ? int(v) : (int(v << (32u - b)) >> (32u - b));
        float maxu = b >= 32u ? 4294967295.0f : float((1u << b) - 1u);
        uint res;
        switch (nfmt) {
        case 0u: res = as_type<uint>(float(v) / maxu); break;
        case 1u:
        case 6u: res = as_type<uint>(max(float(sv) / float((1u << (b - 1u)) - 1u), -1.0f)); break;
        case 2u: res = as_type<uint>(float(v)); break;
        case 3u: res = as_type<uint>(float(sv)); break;
        case 4u: res = v; break;
        case 5u: res = uint(sv); break;
        default:
            res = b == 32u ? v
                : b == 16u ? as_type<uint>(float(as_type<half>(ushort(v))))
                           : as_type<uint>(dc_ufloat(v, b - 5u));
            break;
        }
        o[k] = res;
    }
    return o;
}
template <typename M>
static inline uint4 dc_load_fmt(thread const M& m, ulong a, uint dfmt, uint nfmt) {
    uint words = dfmt == 14u ? 4u : dfmt == 13u ? 3u : (dfmt == 11u || dfmt == 12u) ? 2u : 1u;
    uint4 w = uint4(0u);
    if (dc_aligned(m, a)) {
        device uint* p = dc_ptr_n(m, a, words * 4u);
        if (p) for (uint k = 0u; k < words; ++k) w[k] = p[k];
    } else {
        for (uint k = 0u; k < words; ++k) w[k] = dc_ld(m, a + ulong(k) * 4ul);
    }
    uint w0 = w.x;
    uint4 r = uint4(0u);
    uint4 bits = uint4(32u);
    uint n = 0u;
    switch (dfmt) {
    case 1u: r.x = w0 & 0xffu; bits = uint4(8u); n = 1u; break;
    case 2u: r.x = w0 & 0xffffu; bits = uint4(16u); n = 1u; break;
    case 3u: r.xy = uint2(w0 & 0xffu, (w0 >> 8) & 0xffu); bits = uint4(8u); n = 2u; break;
    case 4u: r.x = w0; n = 1u; break;
    case 5u: r.xy = uint2(w0 & 0xffffu, w0 >> 16); bits = uint4(16u); n = 2u; break;
    case 6u: r.xyz = uint3(w0 & 0x7ffu, (w0 >> 11) & 0x7ffu, w0 >> 22); bits = uint4(11u, 11u, 10u, 0u); n = 3u; break;
    case 7u: r.xyz = uint3(w0 & 0x3ffu, (w0 >> 10) & 0x7ffu, w0 >> 21); bits = uint4(10u, 11u, 11u, 0u); n = 3u; break;
    case 8u: r = uint4(w0 & 3u, (w0 >> 2) & 0x3ffu, (w0 >> 12) & 0x3ffu, w0 >> 22); bits = uint4(2u, 10u, 10u, 10u); n = 4u; break;
    case 9u: r = uint4(w0 & 0x3ffu, (w0 >> 10) & 0x3ffu, (w0 >> 20) & 0x3ffu, w0 >> 30); bits = uint4(10u, 10u, 10u, 2u); n = 4u; break;
    case 10u: r = uint4(w0 & 0xffu, (w0 >> 8) & 0xffu, (w0 >> 16) & 0xffu, w0 >> 24); bits = uint4(8u); n = 4u; break;
    case 11u: r.xy = w.xy; n = 2u; break;
    case 12u: r = uint4(w.x & 0xffffu, w.x >> 16, w.y & 0xffffu, w.y >> 16); bits = uint4(16u); n = 4u; break;
    case 13u: r.xyz = w.xyz; n = 3u; break;
    case 14u: r = w; n = 4u; break;
    default: break;
    }
    return dc_fmt_convert(r, bits, nfmt, n);
}
// Typed buffer store: packs v (float bits or integers) into dfmt/nfmt.
static inline uint dc_pack(uint v, uint bits, uint nfmt) {
    uint maxu = bits >= 32u ? 0xffffffffu : ((1u << bits) - 1u);
    float f = as_type<float>(v);
    switch (nfmt) {
    case 0u: return uint(rint(saturate(f) * float(maxu)));
    case 1u: return uint(int(rint(clamp(f, -1.0f, 1.0f) * float(maxu >> 1)))) & maxu;
    case 2u: return uint(clamp(f, 0.0f, float(maxu))) & maxu;
    case 3u: return uint(int(f)) & maxu;
    case 4u: return v & maxu;
    case 5u: return v & maxu;
    default: return bits == 32u ? v : bits == 16u ? uint(as_type<ushort>(half(f))) : 0u;
    }
}
template <typename M>
static inline void dc_store_fmt(thread const M& m, ulong a, uint4 v, uint dfmt, uint nfmt) {
    switch (dfmt) {
    case 1u: dc_st_part(m, a, dc_pack(v.x, 8u, nfmt), 1u); break;
    case 2u: dc_st_part(m, a, dc_pack(v.x, 16u, nfmt), 2u); break;
    case 3u: dc_st_part(m, a, dc_pack(v.x, 8u, nfmt) | (dc_pack(v.y, 8u, nfmt) << 8), 2u); break;
    case 4u: dc_st(m, a, v.x); break;
    case 5u: dc_st(m, a, dc_pack(v.x, 16u, nfmt) | (dc_pack(v.y, 16u, nfmt) << 16)); break;
    case 10u:
        dc_st(m, a, dc_pack(v.x, 8u, nfmt) | (dc_pack(v.y, 8u, nfmt) << 8) | (dc_pack(v.z, 8u, nfmt) << 16) |
                        (dc_pack(v.w, 8u, nfmt) << 24));
        break;
    case 11u: dc_st(m, a, v.x); dc_st(m, a + 4ul, v.y); break;
    case 12u:
        dc_st(m, a, dc_pack(v.x, 16u, nfmt) | (dc_pack(v.y, 16u, nfmt) << 16));
        dc_st(m, a + 4ul, dc_pack(v.z, 16u, nfmt) | (dc_pack(v.w, 16u, nfmt) << 16));
        break;
    case 13u: dc_st(m, a, v.x); dc_st(m, a + 4ul, v.y); dc_st(m, a + 8ul, v.z); break;
    case 14u: dc_st(m, a, v.x); dc_st(m, a + 4ul, v.y); dc_st(m, a + 8ul, v.z); dc_st(m, a + 12ul, v.w); break;
    default: break;
    }
}
static inline uint dc_sel(uint4 v, uint sel, bool is_int) {
    switch (sel & 7u) {
    case 1u: return is_int ? 1u : 0x3f800000u;
    case 4u: return v.x;
    case 5u: return v.y;
    case 6u: return v.z;
    case 7u: return v.w;
    default: return 0u;
    }
}
static inline uint4 dc_swizzle(uint4 v, uint w3) {
    uint nf = (w3 >> 12) & 7u;
    bool is_int = nf == 4u || nf == 5u;
    return uint4(dc_sel(v, w3, is_int), dc_sel(v, w3 >> 3, is_int), dc_sel(v, w3 >> 6, is_int), dc_sel(v, w3 >> 9, is_int));
}

// ---- ALU helpers ----
static inline float dc_mul_legacy(float a, float b) { return (a == 0.0f || b == 0.0f) ? 0.0f : a * b; }
static inline uint dc_bfe_u(uint v, uint off, uint w) {
    off &= 31u; w &= 31u;
    if (w == 0u) return 0u;
    return uint((ulong(v) >> off) & ((1ul << w) - 1ul));
}
static inline uint dc_bfe_i(uint v, uint off, uint w) {
    off &= 31u; w &= 31u;
    if (w == 0u) return 0u;
    uint x = dc_bfe_u(v, off, w);
    return uint(int(x << (32u - w)) >> (32u - w));
}
// Scalar BFE takes a 7-bit width.
static inline uint dc_sbfe_u(uint v, uint off, uint w) {
    off &= 31u;
    if (w == 0u) return 0u;
    if (w >= 32u) return v >> off;
    return uint((ulong(v) >> off) & ((1ul << w) - 1ul));
}
static inline uint dc_sbfe_i(uint v, uint off, uint w) {
    if (w == 0u) return 0u;
    if (w >= 32u) return uint(int(v) >> (off & 31u));
    uint x = dc_sbfe_u(v, off, w);
    return uint(int(x << (32u - w)) >> (32u - w));
}
static inline uint dc_ffbh_u(uint v) { return v == 0u ? 0xffffffffu : clz(v); }
static inline uint dc_ffbl(uint v) { return v == 0u ? 0xffffffffu : ctz(v); }

// ---- wave-wide operations on lane masks (compute) ----
// A guest lane mask lives here as this lane's bit, all ones or zero in both
// halves. SALU ops that read a mask as a number (count, first set bit) need
// the whole wave's bits; the SIMD group stands in for the wave. Every lane
// calls these unconditionally (SIMD group functions need every thread), then
// a value that is not a lane mask keeps its plain meaning.
static inline bool dc_is_mask(ulong x) { return x == 0ul || x == ~0ul; }
static inline ulong dc_ballot(bool b) {
    return ulong(simd_vote::vote_t(simd_ballot(b)));
}
static inline uint dc_wave_bcnt1(ulong x) {
    const uint n = simd_sum(x != 0ul ? 1u : 0u);
    return dc_is_mask(x) ? n : uint(popcount(x));
}
static inline uint dc_wave_ff1(ulong x) {
    const ulong b = dc_ballot(x != 0ul);
    const ulong v = dc_is_mask(x) ? b : x;
    return v == 0ul ? 0xffffffffu : uint(ctz(v));
}
// S_CBRANCH_EXECZ / VCCZ: the wave jumps only when no lane of the 64 has the
// bit (GT7's Hi-Z occlusion test sums per-cell votes with ds_swizzle and
// readlane 31/63 under such a branch: per-lane skipping lost the sum, ryGT7
// FIX-190). This SIMD group votes with simd_all; in a 64-thread group the two
// halves exchange their votes through threadgroup memory tagged with a
// generation, spinning (capped) instead of a barrier so a half that already
// left cannot hang the other. A mask used as a plain number is uniform, so
// the answer is the same.
static inline bool dc_wave_zero(uint v, threadgroup atomic_uint* wz, uint lidx, uint tg, thread uint& gen) {
    const bool zero_ = simd_all(v == 0u);
    if (tg != 64u) return zero_;
    const uint g = ++gen;
    const uint h = (lidx >> 5) & 1u;
    uint other = 0u;
    if (simd_is_first()) {
        atomic_store_explicit(&wz[h], (g << 1) | (zero_ ? 0u : 1u), memory_order_relaxed);
        for (uint spin = 0u; spin < 8192u; ++spin) {
            other = atomic_load_explicit(&wz[h ^ 1u], memory_order_relaxed);
            if ((other >> 1) >= g) break;
        }
    }
    other = simd_broadcast_first(other);
    if ((other >> 1) != g) return zero_; // the other half left or diverged: this half alone
    return zero_ && (other & 1u) == 0u;
}
// V_READFIRSTLANE: the first lane whose EXEC bit is set.
static inline uint dc_readfirstlane(uint v, bool active) {
    const ulong b = dc_ballot(active);
    return simd_shuffle(v, ushort(b == 0ul ? 0u : uint(ctz(b))));
}
static inline uint dc_ffbh_i(uint v) { uint t = int(v) < 0 ? ~v : v; return t == 0u ? 0xffffffffu : clz(t); }
static inline uint dc_cvt_u32(float f) {
    if (isnan(f) || f <= 0.0f) return 0u;
    return f >= 4294967040.0f ? 0xffffffffu : uint(f);
}
static inline uint dc_cvt_i32(float f) {
    if (isnan(f)) return 0u;
    if (f <= -2147483648.0f) return 0x80000000u;
    if (f >= 2147483520.0f) return 0x7fffffffu;
    return uint(int(f));
}
static inline uint dc_f16(float f) { return uint(as_type<ushort>(half(f))); }
// V_CVT_PKRTZ: round toward zero, so finite values never become infinity.
static inline uint dc_f16_rtz(float f) {
    if (isnan(f)) return 0x7e00u;
    if (isinf(f)) return f > 0.0f ? 0x7c00u : 0xfc00u;
    half h = half(clamp(f, -65504.0f, 65504.0f));
    if (abs(float(h)) > abs(f)) {
        ushort b = as_type<ushort>(h);
        h = as_type<half>(ushort(b - 1u)); // one ulp toward zero
    }
    return uint(as_type<ushort>(h));
}
static inline float4 dc_probe(float4 v) {
    if (any(isnan(v))) return float4(1000.0f, 0.0f, 1000.0f, 1.0f);
    if (any(isinf(v))) return float4(1000.0f, 1000.0f, 0.0f, 1.0f);
    return v;
}
// The VALU clamp modifier: NaN becomes 0 (Metal's saturate keeps NaN).
static inline float dc_sat(float f) { return f > 0.0f ? min(f, 1.0f) : 0.0f; }

// ---- f64 (Metal has no double): IEEE binary64 bits in a ulong, integer
// soft-float. Round to nearest, denormals flushed to zero, NaN payloads not
// kept. Exact integer math, so fast-math cannot fold it away.
constant ulong DC_D_SIGN = 0x8000000000000000ul;
constant ulong DC_D_M52 = 0x000ffffffffffffful;
constant ulong DC_D_INF = 0x7ff0000000000000ul;
// m holds the significand with its leading bit at 61 (9 guard bits), e biased.
static inline ulong dc_d_pack(ulong sign, int e, ulong m) {
    ulong r = m + 0x100ul;
    if ((r >> 62) != 0ul) { r >>= 1; e += 1; }
    if (e >= 0x7ff) return sign | DC_D_INF;
    if (e <= 0) return sign;
    return sign | (ulong(e) << 52) | ((r >> 9) & DC_D_M52);
}
static inline ulong dc_d_add(ulong a, ulong b) {
    int ea = int((a >> 52) & 0x7fful), eb = int((b >> 52) & 0x7fful);
    if (ea == 0x7ff) return a;
    if (eb == 0x7ff) return b;
    if (eb == 0) return ea == 0 ? (a & b & DC_D_SIGN) : a;
    if (ea == 0) return b;
    if ((a & ~DC_D_SIGN) < (b & ~DC_D_SIGN)) { ulong t = a; a = b; b = t; int te = ea; ea = eb; eb = te; }
    ulong ma = ((a & DC_D_M52) | (1ul << 52)) << 9;
    ulong mb = ((b & DC_D_M52) | (1ul << 52)) << 9;
    int d = ea - eb;
    if (d > 62) mb = 1ul;
    else if (d > 0) mb = (mb >> d) | (((mb << (64 - d)) != 0ul) ? 1ul : 0ul);
    ulong m = ((a ^ b) & DC_D_SIGN) != 0ul ? ma - mb : ma + mb;
    if (m == 0ul) return 0ul;
    int e = ea;
    int lz = int(clz(m));
    if (lz < 2) { m = (m >> 1) | (m & 1ul); e += 1; }
    else { m <<= (lz - 2); e -= (lz - 2); }
    return dc_d_pack(a & DC_D_SIGN, e, m);
}
static inline ulong dc_d_mul(ulong a, ulong b) {
    ulong sign = (a ^ b) & DC_D_SIGN;
    int ea = int((a >> 52) & 0x7fful), eb = int((b >> 52) & 0x7fful);
    if (ea == 0x7ff || eb == 0x7ff) return sign | DC_D_INF;
    if (ea == 0 || eb == 0) return sign;
    ulong ma = (a & DC_D_M52) | (1ul << 52), mb = (b & DC_D_M52) | (1ul << 52);
    ulong hi = mulhi(ma, mb), lo = ma * mb;
    int e = ea + eb - 1023;
    ulong m;
    if (((hi >> 41) & 1ul) != 0ul) { m = (hi << 20) | (lo >> 44); m |= (lo << 20) != 0ul ? 1ul : 0ul; e += 1; }
    else { m = (hi << 21) | (lo >> 43); m |= (lo << 21) != 0ul ? 1ul : 0ul; }
    return dc_d_pack(sign, e, m);
}
static inline ulong dc_d_fma(ulong a, ulong b, ulong c) { return dc_d_add(dc_d_mul(a, b), c); }
static inline ulong dc_f2d(float f) {
    uint x = as_type<uint>(f);
    ulong sign = ulong(x & 0x80000000u) << 32;
    uint e = (x >> 23) & 0xffu;
    if (e == 0u) return sign;
    if (e == 0xffu) return sign | DC_D_INF | (ulong(x & 0x7fffffu) << 29);
    return sign | (ulong(e - 127u + 1023u) << 52) | (ulong(x & 0x7fffffu) << 29);
}
static inline float dc_d2f(ulong d) {
    uint sign = uint(d >> 32) & 0x80000000u;
    int e = int((d >> 52) & 0x7fful);
    if (e == 0x7ff) return as_type<float>(sign | 0x7f800000u | ((d & DC_D_M52) != 0ul ? 0x400000u : 0u));
    e = e - 1023 + 127;
    if (e <= 0) return as_type<float>(sign);
    ulong m = ((d & DC_D_M52) | (1ul << 52)) + (1ul << 28);
    if ((m >> 53) != 0ul) { m >>= 1; e += 1; }
    if (e >= 0xff) return as_type<float>(sign | 0x7f800000u);
    return as_type<float>(sign | (uint(e) << 23) | (uint(m >> 29) & 0x7fffffu));
}
static inline ulong dc_d_rcp(ulong x) {
    int e = int((x >> 52) & 0x7fful);
    if (e == 0) return (x & DC_D_SIGN) | DC_D_INF;
    if (e == 0x7ff) return (x & DC_D_M52) != 0ul ? x : (x & DC_D_SIGN);
    // Estimate in float on the significand (exponent handled apart), then Newton.
    ulong xm = (x & (DC_D_SIGN | DC_D_M52)) | (1023ul << 52);  // x scaled to [1, 2)
    ulong y = dc_f2d(1.0f / dc_d2f(xm));
    const ulong two = 0x4000000000000000ul;
    y = dc_d_mul(y, dc_d_add(two, dc_d_mul(xm, y) ^ DC_D_SIGN));
    y = dc_d_mul(y, dc_d_add(two, dc_d_mul(xm, y) ^ DC_D_SIGN));
    int ey = int((y >> 52) & 0x7fful) - (e - 1023);
    if (ey >= 0x7ff) return (y & DC_D_SIGN) | DC_D_INF;
    if (ey <= 0) return y & DC_D_SIGN;
    return (y & ~(0x7fful << 52)) | (ulong(ey) << 52);
}
static inline ulong dc_d_trunc(ulong x) {
    int e = int((x >> 52) & 0x7fful) - 1023;
    if (e < 0) return x & DC_D_SIGN;
    if (e >= 52) return x;
    return x & ~(DC_D_M52 >> e);
}
static inline ulong dc_d_key(ulong x) { return (x & DC_D_SIGN) != 0ul ? ~x : (x | DC_D_SIGN); }  // unsigned order
static inline ulong dc_d_min(ulong a, ulong b) { return dc_d_key(a) <= dc_d_key(b) ? a : b; }
static inline ulong dc_d_max(ulong a, ulong b) { return dc_d_key(a) >= dc_d_key(b) ? a : b; }
static inline bool dc_d_lt(ulong a, ulong b) { return dc_d_key(a) < dc_d_key(b); }
static inline ulong dc_d_sat(ulong x) {
    if ((x & DC_D_SIGN) != 0ul || x == 0ul) return 0ul;
    return x > 0x3ff0000000000000ul ? 0x3ff0000000000000ul : x;
}
// What the CB stores for R11G11B10: negatives and NaN become 0, overflow the largest
// finite value (65024).
static inline float4 dc_ufloat(float4 v) {
    return float4(select(float3(0.0f), min(v.xyz, float3(65024.0f)), v.xyz > 0.0f), v.w);
}
static inline float dc_f32(uint v) { return float(as_type<half>(ushort(v & 0xffffu))); }
// V_MED3_F32: with a NaN input the result is MIN3 of the inputs (min ignoring the
// NaN), so med3(hi, NaN, 0), a shader's saturate, gives 0 like the hardware. The
// plain median formula gave hi: NaN lighting on GT7's driver glove became a 65000
// pixel that the bloom spread into a blue veil over half the screen.
static inline float dc_med3(float a, float b, float c) {
    if (isnan(a) || isnan(b) || isnan(c)) return fmin(fmin(a, b), c);
    return max(min(a, b), min(max(a, b), c));
}
static inline int dc_med3i(int a, int b, int c) { return max(min(a, b), min(max(a, b), c)); }
static inline uint dc_med3u(uint a, uint b, uint c) { return max(min(a, b), min(max(a, b), c)); }
// The three vertex values of an interpolated param, from its value and screen
// derivatives against the barycentrics b it is linear in (see msl.cpp PerVertex). Values within 1/256 of an
// integer snap to it: shaders add P0 + P10 and truncate probe indices.
static inline float4 dc_snap(float4 v) {
    const float4 r = rint(v);
    return select(v, r, abs(v - r) < 0.00390625f);
}
static inline void dc_vertex_values(float4 v, float3 b, thread float4* out) {
    const float4 vx = dfdx(v), vy = dfdy(v);
    const float b1x = dfdx(b.y), b1y = dfdy(b.y), b2x = dfdx(b.z), b2y = dfdy(b.z);
    const float det = b1x * b2y - b2x * b1y;
    const float id = det != 0.0f ? 1.0f / det : 0.0f;
    const float4 d1 = (vx * b2y - vy * b2x) * id, d2 = (vy * b1x - vx * b1y) * id;
    const float4 v0 = v - b.y * d1 - b.z * d2;
    out[0] = dc_snap(v0);
    out[1] = dc_snap(v0 + d1);
    out[2] = dc_snap(v0 + d2);
}
static inline float dc_cubeid(float x, float y, float z) {
    if (abs(z) >= abs(x) && abs(z) >= abs(y)) return z < 0.0f ? 5.0f : 4.0f;
    if (abs(y) >= abs(x)) return y < 0.0f ? 3.0f : 2.0f;
    return x < 0.0f ? 1.0f : 0.0f;
}
static inline float dc_cubesc(float x, float y, float z) {
    if (abs(z) >= abs(x) && abs(z) >= abs(y)) return z < 0.0f ? -x : x;
    if (abs(y) >= abs(x)) return x;
    return x < 0.0f ? z : -z;
}
static inline float dc_cubetc(float x, float y, float z) {
    if (abs(z) >= abs(x) && abs(z) >= abs(y)) return -y;
    if (abs(y) >= abs(x)) return y < 0.0f ? -z : z;
    return -y;
}
static inline float dc_cubema(float x, float y, float z) {
    if (abs(z) >= abs(x) && abs(z) >= abs(y)) return 2.0f * z;
    if (abs(y) >= abs(x)) return 2.0f * y;
    return 2.0f * x;
}
static inline bool dc_class(float f, uint mask) {
    uint b = as_type<uint>(f);
    bool neg = (b >> 31) != 0u;
    uint e = (b >> 23) & 0xffu;
    uint mt = b & 0x7fffffu;
    uint bit;
    if (e == 0xffu) bit = mt != 0u ? ((mt & 0x400000u) != 0u ? 1u : 0u) : (neg ? 2u : 9u);
    else if (e == 0u) bit = mt == 0u ? (neg ? 5u : 6u) : (neg ? 4u : 7u);
    else bit = neg ? 3u : 8u;
    return ((mask >> bit) & 1u) != 0u;
}

// ---- position: guest viewport transform -> Metal clip space ----
static inline float4 dc_viewport(float4 p, constant DcDraw& d) {
    float w = p.w;
    float3 q = p.xyz;
    if ((d.vte & 0x100u) == 0u) q.xy /= w;
    if ((d.vte & 0x200u) == 0u) q.z /= w;
    float3 s = q * d.vp_scale.xyz + d.vp_offset.xyz;
    float2 n = float2(s.x * d.rt_scale.x - 1.0f, 1.0f - s.y * d.rt_scale.y);
    return float4(n * w, s.z * w, w);
}
)MSL";

} // namespace Gpu::Shader
