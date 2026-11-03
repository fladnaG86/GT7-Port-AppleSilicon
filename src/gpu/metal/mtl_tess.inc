// Tessellation emulation, included inside MetalRenderer::Impl (mtl_renderer.mm).
//
// A draw with VGT_SHADER_STAGES_EN.HS_EN and DI_PT_PATCH runs as:
//   1. the LS as a compute kernel, one thread per control point, threadgroups
//      of NUM_PATCHES * cp_in threads; its LDS is a device buffer slice per
//      threadgroup (Shader::TessInfo);
//   2. the HS as a compute kernel, NUM_PATCHES * cp_out threads per group, on
//      the same LDS slices; it writes tess factors and off-chip outputs to the
//      guest buffers its V#s name, with system SGPRs chosen so threadgroup g
//      writes patches g * NUM_PATCHES.. contiguously;
//   3. a kernel turning the guest tess factors (floats) into Metal's half
//      factors buffer;
//   4. the VS as Metal's post-tessellation vertex function (drawPatches),
//      given u, v, the relative patch id and the patch id as the hardware
//      would.
// DC_TESS=1 turns it on (default off: the draws stay skipped).

struct TessDraw {
  Shader::TessInfo info;
  id<MTLBuffer> factors = nil;
  u32 patches = 0;
  MTLTessellationPartitionMode partition = MTLTessellationPartitionModeInteger;
  MTLWinding winding = MTLWindingClockwise;
};
const TessDraw *tess_draw = nullptr; // set while Draw() records the domain pass
id<MTLBuffer> tess_lds = nil;
id<MTLComputePipelineState> tess_tf_pso = nil;

// Compiles (or finds) the compute pipeline of an LS or HS.
id<MTLComputePipelineState> TessKernel(Program *p, const Shader::MslOptions &o, u64 addr,
                                      const std::vector<TextureType> &types) {
  u64 key = Hash(p->hash);
  key = Hash(o.map_hoist, key);
  key = Fnv(&o.tess, sizeof(o.tess), key ^ 0x7e55);
  key = Fnv(types.data(), types.size() * sizeof(TextureType), key);
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
            DC_ERROR("gpu: tess pipeline %#llx: %s", addr,
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
  return it->second;
}

bool RunTessKernel(const Shader::TessInfo &info, u64 addr, const u32 *ud, u32 rsrc2,
                   u32 threads, u32 group, u64 index_va, id<MTLBuffer> tf = nil) {
  Program *p = GetProgram(Stage::Compute, addr, ud, Bits(rsrc2, 1, 5), 2);
  if (!p) {
    ReportOnce("tess: program " + Hex(addr) + " cannot be decoded");
    return false;
  }
  Shader::MslOptions o;
  o.map_hoist = map_hoist;
  o.tess = info;
  std::vector<TextureType> types;
  Resources res;
  resolving_shader = addr;
  Resolve(*p, ud, types, res); // GT7's terrain HS samples a texture to cull patches
  o.textures = types;
  id<MTLComputePipelineState> pso = TessKernel(p, o, addr, types);
  if (!pso || pso.maxTotalThreadsPerThreadgroup < group) {
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
  [e setBuffer:tess_lds offset:0 atIndex:Shader::Binding::TessLds];
  if (tf) {
    [e setBuffer:tf offset:0 atIndex:Shader::Binding::TessFactors];
  }
  for (u32 k = 0; k < res.tex.size(); ++k) {
    [e setTexture:res.tex[k] atIndex:k];
  }
  for (u32 u = 0; u < res.smp.size() && u < 16; ++u) {
    [e setSamplerState:res.smp[u] atIndex:u];
  }
  if (index_va) {
    id<MTLBuffer> ib = nil;
    u64 ioff = 0;
    if (!Locate(index_va, 4, &ib, &ioff)) {
      [e endEncoding];
      return false;
    }
    [e setBuffer:ib offset:ioff atIndex:Shader::Binding::GsIndex];
  }
  [e dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
  [e endEncoding];
  return true;
}

// The HS's float factors (Binding::TessFactors) as Metal's half factors.
bool TessFactors(id<MTLBuffer> in, u32 floats, u32 patches, float max_factor, id<MTLBuffer> out) {
  if (!tess_tf_pso) {
    std::string src = Shader::kMslPrelude;
    src += "kernel void tess_tf(device const float* in [[buffer(1)]], constant uint4& a [[buffer(4)]], "
           "device half* out [[buffer(5)]], uint i [[thread_position_in_grid]]) {\n"
           "  if (i >= a.z) return;\n"
           "  float mx = as_type<float>(a.w);\n"
           "  for (uint k = 0u; k < a.y; ++k) {\n"
           "    float f = in[i * a.y + k];\n"
           "    out[i * uint(a.y) + k] = half(isnan(f) ? 0.0f : min(f, mx));\n"
           "  }\n}\n";
    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:@(src.c_str()) options:nil error:&err];
    id<MTLFunction> fn = lib ? [lib newFunctionWithName:@"tess_tf"] : nil;
    tess_tf_pso = fn ? [device newComputePipelineStateWithFunction:fn error:&err] : nil;
    if (!tess_tf_pso) {
      DC_ERROR("gpu: tess factor kernel: %s", err ? err.localizedDescription.UTF8String : "?");
      return false;
    }
  }
  float mf = max_factor;
  u32 mfb;
  std::memcpy(&mfb, &mf, 4);
  const u32 a[4] = {0, floats, patches, mfb};
  id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
  [e setComputePipelineState:tess_tf_pso];
  [e setBuffer:in offset:0 atIndex:1];
  [e setBytes:a length:sizeof(a) atIndex:4];
  [e setBuffer:out offset:0 atIndex:5];
  [e dispatchThreads:MTLSizeMake(patches, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  [e endEncoding];
  return true;
}

// Returns false when the draw is not emulated (it is then skipped as before).
bool DrawTess(const RegFile &r, const DrawCall &dc) {
  static const bool on = std::getenv("DC_TESS") != nullptr;
  // DC_TESS_OC=<bytes>: off-chip bytes per threadgroup (default below).
  static const u32 oc_env = std::getenv("DC_TESS_OC") ? u32(std::strtoul(std::getenv("DC_TESS_OC"), nullptr, 0)) : 0;
  if (!on || tess_draw || !dc.count || r.Primitive() != PrimType(9)) {
    return false;
  }
  const u32 config = r[Reg::VGT_LS_HS_CONFIG];
  const u32 tf = r[Reg::VGT_TF_PARAM];
  Shader::TessInfo info;
  info.patches = config & 0xFF;
  info.cp_in = (config >> 8) & 0x3F;
  info.cp_out = (config >> 14) & 0x3F;
  info.domain = u8(tf & 3);
  const u32 rsrc2_ls = r[Reg::SPI_SHADER_PGM_RSRC2_LS];
  const u32 rsrc2_hs = r[Reg::SPI_SHADER_PGM_RSRC2_HS];
  const u32 rsrc2_vs = r[Reg::SPI_SHADER_PGM_RSRC2_VS];
  // LDS_SIZE is in 128-dword units; twice that per slice in case it is 256 here.
  info.lds_dwords = Bits(rsrc2_ls, 7, 9) * 256;
  info.oc_sgpr = (rsrc2_hs >> 7) & 1;
  info.tf_bytes = info.domain == 2 ? 24 : 16;
  info.oc_bytes = oc_env ? oc_env : 0x4000;
  const u32 n = dc.count / std::max(info.cp_in, 1u);
  if ((info.domain != 1 && info.domain != 2) || !info.patches || !info.cp_in || !info.cp_out ||
      dc.instances > 1 || info.patches * info.cp_in > 1024 || !info.lds_dwords || !n ||
      ((rsrc2_vs >> 7) & 1) != info.oc_sgpr) {
    ReportOnce("tess: unsupported pipeline (config " + Hex(config) + ", tf " + Hex(tf) +
               ", instances " + std::to_string(dc.instances) + ", ls rsrc2 " + Hex(rsrc2_ls) +
               ", hs rsrc2 " + Hex(rsrc2_hs) + ", vs rsrc2 " + Hex(rsrc2_vs) + ")");
    return false;
  }
  const u32 groups = (n + info.patches - 1) / info.patches;
  const u64 lds_bytes = u64(groups) * info.lds_dwords * 4;
  if (!tess_lds || tess_lds.length < lds_bytes) {
    tess_lds = [device newBufferWithLength:std::max<u64>(lds_bytes, 4u << 20)
                                   options:MTLResourceStorageModePrivate];
  }
  // LS vertex indices: the index buffer (or a base vertex) as u32s.
  u64 index_va = 0;
  info.index_size = 0;
  const u32 verts = n * info.cp_in;
  const u32 base_vertex = r[Reg::VGT_INDX_OFFSET] + (dc.indexed ? 0 : dc.first_vertex);
  if (dc.indexed || base_vertex) {
    const u64 src = dc.indexed ? dc.index_base + u64(dc.index_offset) * dc.index_size : 0;
    if (dc.indexed && !Readable(src, u64(verts) * dc.index_size)) {
      ReportOnce("tess: index buffer not mapped");
      return false;
    }
    index_va = ArenaAlloc(u64(verts) * 4);
    if (!index_va) {
      return false;
    }
    auto *out = reinterpret_cast<u32 *>(index_va);
    for (u32 k = 0; k < verts; ++k) {
      u32 v = k;
      if (dc.indexed) {
        v = dc.index_size == 4 ? reinterpret_cast<const u32 *>(src)[k]
                               : reinterpret_cast<const u16 *>(src)[k];
      }
      out[k] = v + base_vertex;
    }
    info.index_size = 4;
  }
  info.stage = Shader::TessStage::Local;
  if (!RunTessKernel(info, r.ProgramAddr(Reg::SPI_SHADER_PGM_LO_LS),
                     r.UserData(Reg::SPI_SHADER_USER_DATA_LS_0), rsrc2_ls, verts,
                     info.patches * info.cp_in, index_va)) {
    return false;
  }
  info.index_size = 0;
  info.stage = Shader::TessStage::Hull;
  info.tf_sgpr = u8(Bits(rsrc2_hs, 1, 5) + (info.oc_sgpr ? 1 : 0));
  const u64 hs_addr = r.ProgramAddr(Reg::SPI_SHADER_PGM_LO_HS);
  const u32 *hs_ud = r.UserData(Reg::SPI_SHADER_USER_DATA_HS_0);
  // Factors of culled patches stay 0 (Metal then drops the patch, as GCN does).
  const u64 tf_bytes = std::max<u64>(u64(groups) * info.patches * info.tf_bytes, 16);
  id<MTLBuffer> tf_buf = [device newBufferWithLength:tf_bytes options:MTLResourceStorageModePrivate];
  {
    id<MTLBlitCommandEncoder> b = (EndPass(), EnsureCommandBuffer(), [cmd blitCommandEncoder]);
    [b fillBuffer:tf_buf range:NSMakeRange(0, tf_bytes) value:0];
    [b endEncoding];
  }
  if (!RunTessKernel(info, hs_addr, hs_ud, rsrc2_hs, n * info.cp_out,
                     info.patches * info.cp_out, 0, tf_buf)) {
    return false;
  }
  TessDraw t;
  t.patches = n;
  t.factors = [device newBufferWithLength:u64(n) * info.tf_bytes / 2
                                  options:MTLResourceStorageModePrivate];
  float max_factor = r.F(Reg::VGT_HOS_MAX_TESS_LEVEL);
  max_factor = max_factor >= 1.0f && max_factor <= 64.0f ? max_factor : 64.0f;
  if (!TessFactors(tf_buf, info.tf_bytes / 4, n, max_factor, t.factors)) {
    return false;
  }
  static const MTLTessellationPartitionMode parts[4] = {
      MTLTessellationPartitionModeInteger, MTLTessellationPartitionModePow2,
      MTLTessellationPartitionModeFractionalOdd, MTLTessellationPartitionModeFractionalEven};
  t.partition = parts[std::min<u32>((tf >> 2) & 7, 3)];
  t.winding = ((tf >> 5) & 7) == 3 ? MTLWindingCounterClockwise : MTLWindingClockwise;
  info.stage = Shader::TessStage::Domain;
  t.info = info;
  if (trace) {
    DC_INFO("gpu:   tess emulated: %u patches in %u groups of %u, cp %u->%u, lds %u dw, oc %u B/group "
            "(sgpr %d), tf sgpr %u, domain %u part %u",
            n, groups, info.patches, info.cp_in, info.cp_out, info.lds_dwords, info.oc_bytes,
            int(info.oc_sgpr), info.tf_sgpr, info.domain, (tf >> 2) & 7);
  }
  auto copy = std::make_unique<RegFile>(r);
  (*copy)[Reg::VGT_SHADER_STAGES_EN] &= ~0x7u; // LS and HS done
  tess_draw = &t;
  Draw(*copy, dc);
  tess_draw = nullptr;
  return true;
}
