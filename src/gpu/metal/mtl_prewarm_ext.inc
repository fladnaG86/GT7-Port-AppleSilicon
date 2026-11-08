// Extended pipeline prewarm, included inside MetalRenderer::Impl (mtl_renderer.mm)
// right after the render prewarm's records (RenderRecord, ParseRenderRecord).
//
// The render prewarm (RenderPrewarmStart) covers ordinary VS+PS pipelines.
// This adds, in ~/Library/Caches/DriveclubNative/render_pipelines_ext.bin
// (its own file: older builds stop reading render_pipelines.bin at the first
// record they do not know):
//   - kRecTess: the domain-pass render pipeline of a tessellated draw
//     (DrawTess), a RenderRecord plus the domain TessInfo and the tessellator
//     partition / winding;
//   - kRecFused: the copy-VS render pipeline of a GS draw with the ES and GS
//     fused in (DrawGs), a RenderRecord plus both ring programs (guest code,
//     user data) and the ES's RingInfo. GT7 draws its text glyphs so: 600+ KB
//     of MSL each, 6-8 s to compile cold, and every such draw waits;
//   - kRecKernel: the LS and HS compute kernels of a tessellated draw
//     (TessKernel), which every tessellated draw waits for.
// All are rebuilt at launch on the compiler workers like the render prewarm,
// under their pipeline key (pending), with the current translator; a record
// whose recomputed key differs is dropped and compiles the old way.
// DC_NO_EXT_PREWARM=1: neither recorded nor prewarmed (DC_NO_PREWARM /
// DC_NO_RENDER_PREWARM switch these off too).

const bool ext_prewarm_on = std::getenv("DC_NO_EXT_PREWARM") == nullptr;
static constexpr u32 kExtRecordMagic = 0x31585052; // "RPX1"; change when the records change
enum : u32 { kRecKernel = 3 };
std::unordered_set<u64> kernel_known;

static std::string ExtPrewarmPath() {
  const char *home = std::getenv("HOME");
  return home ? std::string(home) + "/Library/Caches/DriveclubNative/render_pipelines_ext.bin" : "";
}

// An LS or HS kernel of a tessellated draw.
struct KernelRecord {
  u64 pkey = 0, addr = 0;
  u32 nus = 0, map_hoist = 0;
  std::array<u32, 16> ud{};
  std::vector<u32> code, fetch;
  Shader::TessInfo tess;
  std::vector<TextureType> types;
};

// Serialization helpers shared by the records below.
struct ExtWriter {
  std::string b;
  void put(const void *d, size_t n) { b.append(static_cast<const char *>(d), n); }
  void put32(u32 v) { put(&v, 4); }
  void put64(u64 v) { put(&v, 8); }
  template <typename V> void putv(const V &v) {
    put32(static_cast<u32>(v.size()));
    put(v.data(), v.size() * sizeof(v[0]));
  }
  void tess(const Shader::TessInfo &t) {
    for (u32 v : {u32(t.stage), t.patches, t.cp_in, t.cp_out, t.lds_dwords, t.oc_bytes, t.tf_bytes,
                  u32(t.oc_sgpr), u32(t.tf_sgpr), u32(t.index_size), u32(t.domain)})
      put32(v);
  }
  void ring(const Shader::RingInfo &q) {
    for (u32 v : {u32(q.stage), q.esgs_itemsize, q.vert_itemsize, q.slots, q.input_verts, q.index_size}) put32(v);
  }
  void prog(const RingProgRecord &q) {
    put64(q.addr);
    put32(q.nus);
    put(q.ud.data(), 64);
    putv(q.code);
    putv(q.fetch);
  }
};
struct ExtReader {
  const char *p;
  size_t n, at = 0;
  bool ok = true;
  void get(void *d, size_t k) {
    if (!ok || at + k > n) {
      ok = false;
      return;
    }
    std::memcpy(d, p + at, k);
    at += k;
  }
  u32 get32() {
    u32 v = 0;
    get(&v, 4);
    return v;
  }
  u64 get64() {
    u64 v = 0;
    get(&v, 8);
    return v;
  }
  template <typename V> void getv(V &v, u32 max) {
    const u32 c = get32();
    if (!ok || c > max) {
      ok = false;
      return;
    }
    v.resize(c);
    get(v.data(), c * sizeof(v[0]));
  }
  void tess(Shader::TessInfo &t) {
    t.stage = static_cast<Shader::TessStage>(get32());
    t.patches = get32();
    t.cp_in = get32();
    t.cp_out = get32();
    t.lds_dwords = get32();
    t.oc_bytes = get32();
    t.tf_bytes = get32();
    t.oc_sgpr = get32() != 0;
    t.tf_sgpr = static_cast<u8>(get32());
    t.index_size = static_cast<u8>(get32());
    t.domain = static_cast<u8>(get32());
  }
  void ring(Shader::RingInfo &q) {
    q.stage = static_cast<Shader::RingStage>(get32());
    q.esgs_itemsize = get32();
    q.vert_itemsize = get32();
    q.slots = get32();
    q.input_verts = get32();
    q.index_size = get32();
  }
  void prog(RingProgRecord &q) {
    q.addr = get64();
    q.nus = get32();
    get(q.ud.data(), 64);
    getv(q.code, 1u << 18);
    getv(q.fetch, 1u << 12);
  }
};

void ExtAppend(u32 kind, const std::string &payload) {
  const std::string path = ExtPrewarmPath();
  if (path.empty()) {
    return;
  }
  std::error_code ec;
  std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
  std::ofstream f(path, std::ios::binary | std::ios::app);
  const u32 head[3] = {kExtRecordMagic, static_cast<u32>(payload.size() + 4), kind};
  f.write(reinterpret_cast<const char *>(head), 12);
  f.write(payload.data(), payload.size());
}

// Draw (pipeline miss): what a tessellated or fused GS draw adds to its record.
void ExtFillRecord(RenderRecord &rec, u64 tess_pw) {
  if (tess_draw) {
    rec.kind = kRecTess;
    rec.ext.tess = tess_draw->info;
    rec.ext.tess_pw = tess_pw;
  } else if (fused_pending.on) {
    const FusedPending &f = fused_pending;
    rec.kind = kRecFused;
    auto prog = [&](RingProgRecord &q, u64 addr, u32 nus, const u32 *ud, const Program *p) {
      q.addr = addr;
      q.nus = nus;
      std::copy(ud, ud + 16, q.ud.begin());
      q.code = GuestCode(addr);
      if (p && p->fetch_address && p->fetch_bytes) {
        q.fetch = GuestCode(p->fetch_address, p->fetch_bytes);
      }
    };
    prog(rec.ext.es, f.es_addr, f.es_nus, f.es_ud, f.esp);
    prog(rec.ext.gs, f.gs_addr, f.gs_nus, f.gs_ud, f.gsp);
    rec.ext.ring = f.ring;
  }
}

// RenderPrewarmRecord of a kRecTess / kRecFused record: `base` is its RPW1 body.
void ExtPrewarmRecord(const RenderRecord &e, const std::string &base) {
  ExtWriter w;
  w.putv(base);
  if (e.kind == kRecTess) {
    w.tess(e.ext.tess);
    w.put64(e.ext.tess_pw);
  } else {
    w.prog(e.ext.es);
    w.prog(e.ext.gs);
    w.ring(e.ext.ring);
  }
  std::scoped_lock lk(prewarm_lock);
  if (!render_known.insert(e.pkey).second) {
    return;
  }
  ExtAppend(e.kind, w.b);
}

KernelRecord KernelRecordOf(const Program &p, const Shader::MslOptions &o, u64 addr,
                            const std::vector<TextureType> &types, const u32 *ud, u32 nus, u64 key) {
  KernelRecord k;
  if (!ext_prewarm_on || !ud) {
    return k; // pkey 0: not recorded
  }
  k.pkey = key;
  k.addr = addr;
  k.nus = nus;
  k.map_hoist = o.map_hoist;
  std::copy(ud, ud + 16, k.ud.begin());
  k.code = GuestCode(addr);
  if (p.fetch_address && p.fetch_bytes) {
    k.fetch = GuestCode(p.fetch_address, p.fetch_bytes);
  }
  k.tess = o.tess;
  k.types = types;
  return k;
}

void KernelPrewarmRecord(const KernelRecord &k) {
  ExtWriter w;
  w.put64(k.pkey);
  w.put64(k.addr);
  w.put32(k.nus);
  w.put32(k.map_hoist);
  w.put(k.ud.data(), 64);
  w.putv(k.code);
  w.putv(k.fetch);
  w.tess(k.tess);
  w.putv(k.types);
  std::scoped_lock lk(prewarm_lock);
  if (!kernel_known.insert(k.pkey).second) {
    return;
  }
  ExtAppend(kRecKernel, w.b);
}

// Reads render_pipelines_ext.bin: render records are appended to `entries`
// (kind set), kernels to `kernels`. Returns the bytes read.
u64 ExtPrewarmLoad(std::vector<RenderRecord> &entries, std::vector<KernelRecord> &kernels) {
  const std::string path = ExtPrewarmPath();
  if (!ext_prewarm_on || path.empty()) {
    return 0;
  }
  std::ifstream f(path, std::ios::binary);
  std::vector<char> buf;
  u64 bytes = 0;
  for (;;) {
    u32 head[3];
    if (!f.read(reinterpret_cast<char *>(head), 12) || head[0] != kExtRecordMagic || head[1] < 4 ||
        head[1] > (8u << 20)) {
      break;
    }
    buf.resize(head[1] - 4);
    if (!f.read(buf.data(), buf.size())) {
      break;
    }
    bytes += 8 + head[1];
    ExtReader rd{buf.data(), buf.size()};
    const u32 kind = head[2];
    if (kind == kRecTess || kind == kRecFused) {
      std::vector<char> base;
      rd.getv(base, 8u << 20);
      RenderRecord e;
      if (!rd.ok || !ParseRenderRecord(base.data(), base.size(), e)) {
        break;
      }
      e.kind = kind;
      if (kind == kRecTess) {
        rd.tess(e.ext.tess);
        e.ext.tess_pw = rd.get64();
      } else {
        rd.prog(e.ext.es);
        rd.prog(e.ext.gs);
        rd.ring(e.ext.ring);
      }
      if (!rd.ok || rd.at != rd.n) {
        break;
      }
      if (render_known.insert(e.pkey).second) {
        entries.push_back(std::move(e));
      }
    } else if (kind == kRecKernel) {
      KernelRecord k;
      k.pkey = rd.get64();
      k.addr = rd.get64();
      k.nus = rd.get32();
      k.map_hoist = rd.get32();
      rd.get(k.ud.data(), 64);
      rd.getv(k.code, 1u << 18);
      rd.getv(k.fetch, 1u << 12);
      rd.tess(k.tess);
      rd.getv(k.types, 128);
      if (!rd.ok || rd.at != rd.n || k.code.empty()) {
        break;
      }
      if (kernel_known.insert(k.pkey).second) {
        kernels.push_back(std::move(k));
      }
    } // other kinds (newer builds): skipped
  }
  return bytes;
}

// RenderPrewarmJob: the vertex options a kRecTess / kRecFused record adds.
// A fused copy VS borrows `es_src` / `gs_src`; `fused_key` goes into VsKey.
bool ExtPrewarmOptions(const RenderRecord &e, Shader::MslOptions &vo, std::string &es_src, std::string &gs_src,
                       u64 &fused_key) {
  if (e.kind == kRecTess) {
    vo.tess = e.ext.tess;
    return true;
  }
  if (e.kind != kRecFused) {
    return false;
  }
  auto build = [](const RingProgRecord &q) {
    Shader::BuildInput in{Stage::Compute, q.addr, q.ud.data(), q.nus, 0, q.code.data()};
    in.fetch_code = q.fetch.empty() ? nullptr : q.fetch.data();
    return Shader::BuildProgram(in);
  };
  auto esp = build(e.ext.es);
  auto gsp = build(e.ext.gs);
  if (!esp || !gsp) {
    return false;
  }
  // As DrawGs builds them.
  const Shader::RingInfo &ring = e.ext.ring;
  Shader::MslOptions oe;
  oe.map_hoist = e.vmap_hoist;
  oe.ring = ring;
  oe.ring_fused = true;
  const u64 ke = FusedRingKey(ring, Hash(oe.map_hoist, Hash(esp->hash)));
  Shader::RingInfo gring = ring;
  gring.stage = Shader::RingStage::Geometry;
  gring.index_size = 0;
  Shader::MslOptions og;
  og.map_hoist = e.vmap_hoist;
  og.ring = gring;
  og.ring_fused = true;
  const u64 kg = FusedRingKey(gring, Hash(og.map_hoist, Hash(gsp->hash)));
  es_src = Shader::GenerateMsl(*esp, oe, nullptr);
  gs_src = Shader::GenerateMsl(*gsp, og, nullptr);
  vo.fused.on = true;
  vo.fused.input_verts = ring.input_verts;
  vo.fused.ipp = ring.IndicesPerPrim();
  vo.fused.es_index_size = static_cast<u8>(ring.index_size);
  vo.fused.es_src = &es_src;
  vo.fused.gs_src = &gs_src;
  fused_key = Hash(ke, kg);
  return true;
}

void KernelPrewarmJob(const KernelRecord &k) {
  id<MTLComputePipelineState> pso = nil;
  auto finish = [&] {
    {
      std::scoped_lock lk(done_lock);
      if (pso) {
        done_compute.emplace_back(k.pkey, pso);
      } else {
        done_failed.push_back(k.pkey); // TessKernel compiles it the old way
      }
    }
    done_cv.notify_all();
    if (!pso) {
      ++prewarm_render_dropped;
    }
    if (++prewarm_render_done == prewarm_render_total) {
      DC_INFO("gpu: render prewarm done: %u pipelines in %.1f s, %u dropped", prewarm_render_total.load(),
              std::chrono::duration<double>(std::chrono::steady_clock::now() - prewarm_t0).count(),
              prewarm_render_dropped.load());
    }
  };
  Shader::BuildInput in{Stage::Compute, k.addr, k.ud.data(), k.nus, 2, k.code.data()};
  in.fetch_code = k.fetch.empty() ? nullptr : k.fetch.data();
  auto p = Shader::BuildProgram(in);
  if (!p) {
    finish();
    return;
  }
  Shader::MslOptions o;
  o.map_hoist = k.map_hoist != 0;
  o.tess = k.tess;
  o.textures = k.types;
  const u64 key = TessKernelKey(p->hash, o.map_hoist, o.tess, k.types);
  if (key != k.pkey) {
    finish(); // the key's inputs changed meaning
    return;
  }
  id<MTLFunction> fn =
      compiler->Function(key, compiler->Has(key) ? std::string() : Shader::GenerateMsl(*p, o, nullptr), k.addr, true);
  if (fn) {
    NSError *err = nil;
    pso = [device newComputePipelineStateWithFunction:fn error:&err];
  }
  finish();
}
