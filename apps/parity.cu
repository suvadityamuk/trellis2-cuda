// Parity harness: runs CUDA stages on reference inputs dumped by tools/ref_dump.py and reports bitwise agreement.
#include "models.h"
#include "spconv.h"
#include <cmath>
#include <fstream>

static TensorMap R;
static int g_fail = 0;

static void cmp(const std::string& name, const Tensor& a, const Tensor& ref) {
  if (a.numel() != ref.numel()) { printf("%-48s numel %lld vs ref %lld  MISMATCH\n", name.c_str(), (long long)a.numel(), (long long)ref.numel()); g_fail++; return; }
  auto x = to_host_vec<float>(cast(a, F32)), y = to_host_vec<float>(cast(ref, F32));
  auto hb = to_host(a), rb = to_host(ref);
  bool same_dt = a.dt == ref.dt; i64 nbad = 0; double mx = 0, ms = 0, sa = 0;
  size_t es = dsize(a.dt);
  for (i64 i = 0; i < a.numel(); i++) {
    bool eq = same_dt ? !memcmp(hb.data() + i * es, rb.data() + i * es, es) : x[i] == y[i];
    if (!eq && nbad++ < 4 && env("T2_IDX", "") == "1") printf("    [%lld] %.9g vs %.9g\n", (long long)i, x[i], y[i]);
    double d = std::fabs(x[i] - y[i]); mx = std::max(mx, d); sa += d; ms = std::max(ms, (double)std::fabs(y[i]));
  }
  printf("%-48s %10lld elems  mismatch %10lld  maxabs %.3g  meanabs %.3g (ref max %.3g)%s\n", name.c_str(), (long long)a.numel(), (long long)nbad, mx,
         sa / std::max<i64>(1, a.numel()), ms, same_dt ? "" : "  [dtype differs]");
  if (nbad) g_fail++;
}
static TensorMap Rd;
static Tensor rf(const std::string& k) {
  REQ(R.count(k), "ref missing %s", k.c_str());
  auto it = Rd.find(k); if (it != Rd.end()) return it->second;
  return Rd[k] = from_host(R[k].p, R[k].sh, R[k].dt);
}
static Tensor flat(const Tensor& t, i64 cols) { return t.view({t.numel() / cols, cols}); }

int main(int argc, char** argv) {
  REQ(argc >= 2, "usage: parity <cmd> ...");
  setvbuf(stdout, nullptr, _IOLBF, 0);
  std::string cmd = argv[1], refdir = env("T2_REF", "/workspace/ref"), ck = env("T2_CKPT", "/workspace/ckpts");
  R = load_safetensors_host(refdir + "/pipeline.safetensors");
  if (env("T2_DEEP", "") == "1") for (auto& [k, v] : load_safetensors_host(refdir + "/call.safetensors")) R[k] = v;
  struct M { const char* name; const char* file; bool dense; int sample; const char* cond; SamplerCfg s; };
  std::vector<M> ms = {
    {"sparse_structure_flow_model", "ss_flow_img_dit_1_3B_64_bf16", true, 0, "cond512/cond", {7.5, 0.7, 0.6, 1.0, 5.0}},
    {"shape_slat_flow_model_512", "slat_flow_img2shape_dit_1_3B_512_bf16", false, 1, "cond512/cond", {7.5, 0.5, 0.6, 1.0, 3.0}},
    {"shape_slat_flow_model_1024", "slat_flow_img2shape_dit_1_3B_1024_bf16", false, 2, "cond1024/cond", {7.5, 0.5, 0.6, 1.0, 3.0}},
    {"tex_slat_flow_model_1024", "slat_flow_imgshape2tex_dit_1_3B_1024_bf16", false, 3, "cond1024/cond", {1.0, 0.0, 0.6, 0.9, 3.0}},
  };
  if (cmd == "consts") {
    DiT m; m.load(ck + "/" + ms[0].file + ".safetensors", true);
    cmp("const/ss_rope_phases", m.phases, rf("const/ss_rope_phases"));
    extern std::vector<float> rope_freqs();
    auto f = rope_freqs(); cmp("const/rope_freqs", from_host(f.data(), {21}, F32), rf("const/rope_freqs"));
    extern std::vector<float> t_freqs();
    auto tf = t_freqs(); cmp("const/t_freqs", from_host(tf.data(), {128}, F32), rf("const/t_freqs"));
    std::vector<float> inv(16); for (int i = 0; i < 16; i++) inv[i] = 1.0f / (float)std::pow(100.0, (double)(i * (4.0f / 64)));
    cmp("const/dino_inv_freq", from_host(inv.data(), {16}, F32), rf("const/dino_inv_freq"));
    return g_fail;
  }
  if (cmd == "pre" || cmd == "dino") {  // image -> preprocess -> resize/normalize -> DINOv3 at 512 and 1024
    Image8 pre = preprocess_image(load_image(env("T2_IMAGE", "/workspace/TRELLIS.2/assets/example_image/T.png")));
    cmp("preprocess/out", from_host(pre.d.data(), {(i64)pre.d.size()}, U8), rf("preprocess/out"));
    Dino d; if (cmd == "dino") d.load(ck + "/dinov3_vitl16.safetensors");
    for (int r : {512, 1024}) {
      std::string p = "dino" + std::to_string(r) + "/";
      Tensor in = image_to_input(resize_lanczos(pre, r, r)); cmp(p + "in", in, rf(p + "in"));
      if (cmd != "dino") continue;
      d.dbg = [&](const std::string& n, const Tensor& v) { if (R.count(p + n)) cmp(p + n, v, rf(p + n)); };
      Timer tm; tm.start(); Tensor out = d.forward(rf(p + "in")); float t = tm.ms();
      cmp(p + "out", out, rf(p + "out")); printf("  dino %d: %.2f ms\n", r, t);
    }
    printf(g_fail ? "PARITY FAIL (%d)\n" : "PARITY OK\n", g_fail);
    return g_fail;
  }
  if (cmd == "dec" || cmd == "mesh") {  // shape decoder (with subdivisions) then texture decoder guided by our subdivisions
    std::vector<Tensor> subs; Tensor V, F;
    for (int t = 0; t < 2; t++) {
      std::string nm = t ? "tex_slat_decoder" : "shape_slat_decoder", in = t ? "tex/slat" : "shape/slat";
      SDecoder d; d.load(ck + (t ? "/tex_dec_next_dc_f16c32_fp16.safetensors" : "/shape_dec_next_dc_f16c32_fp16.safetensors"), !t);
      d.dbg = [&](const std::string& n, const Tensor& v) { auto k = nm + "/deep/" + n; if (R.count(k)) cmp(k, v, rf(k)); };
      Timer tm; tm.start();
      Tensor co, out = d.run(rf(in + ".feats"), rf(in + ".coords"), t ? &subs : nullptr, t ? nullptr : &subs, -1, &co);
      printf("  %s %.1f ms\n", nm.c_str(), tm.ms());
      if (!t) for (size_t i = 0; i < subs.size(); i++) cmp("dec_shape/subs." + std::to_string(i) + ".feats", subs[i], rf("dec_shape/subs." + std::to_string(i) + ".feats"));
      if (cmd != "mesh") continue;
      if (!t) {
        tm.start(); fdg_to_mesh(out, co, 1024, V, F); printf("  fdg_to_mesh %.1f ms\n", tm.ms());
        cmp("dec_shape/v", V, rf("dec_shape/v")); cmp("dec_shape/f", F, rf("dec_shape/f"));
        tm.start(); fill_holes(V, F, 3e-2f); printf("  fill_holes %.1f ms\n", tm.ms());
        cmp("final/v", V, rf("final/v")); cmp("final/f", F, rf("final/f"));
      } else {
        scale_(out, 0.5f, 0.5f); cmp("final/attrs", out, rf("final/attrs"));
      }
    }
    printf(g_fail ? "PARITY FAIL (%d)\n" : "PARITY OK\n", g_fail);
    return g_fail;
  }
  if (cmd == "randn" || cmd == "pipe") {  // CPU RNG stream vs dumped noise, then the whole pipeline vs every stage dump
    std::mt19937 g(42);
    for (int i = 0; i < 4; i++) {
      Tensor r = rf("noise/" + std::to_string(i)); auto h = torch_randn(g, r.numel());
      cmp("noise/" + std::to_string(i), from_host(h.data(), r.sh, F32), r);
    }
    if (cmd == "pipe") {
      Pipeline P; P.load(ck);
      P.dbg = [&](const std::string& n, const Tensor& v) { if (R.count(n)) cmp(n, v, rf(n)); };
      Timer tm; tm.start();
      P.times = [&](const char* n) { printf("  %-18s %9.1f ms\n", n, tm.ms()); tm.start(); };
      P.run(load_image(env("T2_IMAGE", "/workspace/TRELLIS.2/assets/example_image/T.png")));
    }
    printf(g_fail ? "PARITY FAIL (%d)\n" : "PARITY OK\n", g_fail);
    return g_fail;
  }
  if (cmd == "glb") {  // to_glb geometry from the dumped to_glb input mesh
    for (auto& [k, v] : load_safetensors_host(refdir + "/glb.safetensors")) R[k] = v;
    Glb g; Timer tm;  // simplify is nondeterministic in the reference too: only its sizes are reported
    g.dbg = [&](const std::string& n, const Tensor& v) { printf("  [%.1f ms]\n", tm.ms()); if (R.count(n) && n.find("simplify") == std::string::npos) cmp(n, v, rf(n)); tm.start(); };
    tm.start(); Tensor a, b; g.remesh(rf("glb/in_v"), rf("glb/in_f"), a, b);
    g.simplify(a, b, 1000000);
    printf("  simplify: %lld verts %lld faces (ref %lld %lld)\n", (long long)a.size(0), (long long)b.size(0),
           (long long)R["glb/simplify0.0"].sh[0], (long long)R["glb/simplify0.1"].sh[0]);
    g.unwrap(rf("glb/simplify0.0"), rf("glb/simplify0.1"));
    printf(g_fail ? "PARITY FAIL (%d)\n" : "PARITY OK\n", g_fail);
    return g_fail;
  }
  if (cmd == "bake") {  // texture bake from the dumped unwrapped mesh, hole-filled mesh and decoded attribute volume
    for (auto& [k, v] : load_safetensors_host(refdir + "/glb.safetensors")) R[k] = v;
    Glb g; Timer tm;
    g.dbg = [&](const std::string& n, const Tensor& v) { printf("  [%.1f ms]\n", tm.ms()); if (R.count(n)) cmp(n, v, rf(n)); tm.start(); };
    g.V0 = rf("glb/fill_holes0.0"); g.F0 = rf("glb/fill_holes0.1"); g.bvh = std::make_unique<BVH>(g.V0, g.F0);
    tm.start(); g.unwrap(rf("glb/simplify0.0"), rf("glb/simplify0.1"));
    g.bake(rf("final/attrs"), rf("final/coords"));
    std::string mine = g.glb_bytes(); printf("  [%.1f ms] glb export\n", tm.ms());
    std::ifstream fi(refdir + "/ref.glb", std::ios::binary); std::string ref((std::istreambuf_iterator<char>(fi)), {});
    // vertex normals are the trailing buffer view; CuMesh sums each vertex's faces in atomic (run-dependent) order, so
    // near-cancelling sums flip between official runs too (~20 of 474k verts > 1e-5 run to run): gate at 0.05% of normals
    size_t nb = (size_t)g.N.numel() * 4, d = 0, nbad = 0, nbig = 0; float mx = 0;
    while (d < std::min(mine.size(), ref.size()) && mine[d] == ref[d]) d++;
    bool same = mine.size() == ref.size() && d >= ref.size() - nb;
    for (size_t o = ref.size() - nb; same && o < ref.size(); o += 12) {
      float a[3], b[3], m = 0; memcpy(a, &mine[o], 12); memcpy(b, &ref[o], 12);
      for (int c = 0; c < 3; c++) m = std::max(m, std::fabs(a[c] - b[c]));
      mx = std::max(mx, m); nbad += m > 0; nbig += m > 1e-5f;
    }
    printf("%-48s %zu bytes vs ref %zu  first diff @ %zu  %s (normals: %zu differ, %zu > 1e-5, maxabs %.3g)\n", "ref.glb", mine.size(), ref.size(),
           d, mine == ref ? "IDENTICAL" : same ? "identical except vertex normals" : "DIFFERS", nbad, nbig, mx);
    if (!same || nbig * 2000 > (size_t)g.N.size(0)) { g_fail++; std::ofstream("/tmp/mine.glb", std::ios::binary) << mine; }
    printf(g_fail ? "PARITY FAIL (%d)\n" : "PARITY OK\n", g_fail);
    return g_fail;
  }
  if (cmd == "spconv") {  // single sparse convs of the shape decoder vs FlexGEMM (tools/ref_spconv.py)
    auto S = load_safetensors_host(refdir + "/spconv.safetensors");
    auto W = load_safetensors(ck + "/shape_dec_next_dc_f16c32_fp16.safetensors");
    for (int j = 0; S.count("conv" + std::to_string(j) + "/in"); j++) {
      std::string p = "conv" + std::to_string(j) + "/";
      auto up = [&](const std::string& k) { return from_host(S[k].p, S[k].sh, S[k].dt); };
      auto nm = to_host(up(p + "name")); std::string name(nm.begin(), nm.end());
      Tensor x = up(p + "in"), w = W[name + ".weight"], b = W[name + ".bias"];
      w = w.view({w.size(0), 27, w.size(4)});
      Timer tm; tm.start();
      SpConvCache c = build_neighbors(up(p + "coords"), 4096, 4096, 4096);
      Tensor y = subm_conv(x, w, &b, c);
      float t = tm.ms();
      auto cfg = spconv_cfg((int)x.size(0), (int)x.size(1), (int)w.size(0));
      cmp(p + name, y, up(p + "out"));
      printf("  N=%lld Ci=%lld Co=%lld B1=%d BK=%d S=%d  %.2f ms\n", (long long)x.size(0), (long long)x.size(1), (long long)w.size(0), cfg.B1, cfg.BK, cfg.S, t);
    }
    printf(g_fail ? "PARITY FAIL (%d)\n" : "PARITY OK\n", g_fail);
    return g_fail;
  }
  for (auto& mm : ms) {
    if (argc > 2 && std::string(argv[2]) != mm.name) continue;
    DiT m; m.load(ck + "/" + mm.file + ".safetensors", mm.dense);
    std::string p = std::string(mm.name) + "/call" + env("T2_CALL", "0") + "/";
    Tensor cond = flat(rf(mm.cond), 1024), neg = zeros(cond.sh, F32);
    if (!mm.dense) m.set_coords(rf(p + "x.coords"));
    Tensor concat = mm.sample == 3 ? rf(std::string(mm.name) + "/call0/concat_cond.feats") : Tensor();
    if (cmd == "flow") {  // single model call vs reference call0 (+ per-block taps)
      Tensor x = mm.dense ? rf(p + "x") : rf(p + "x.feats");
      float t = to_host_vec<float>(rf(p + "t"))[0];
      m.dbg = [&](const std::string& n, const Tensor& v) {
        auto k = std::string(mm.name) + "/deep/" + n; if (R.count(k)) cmp(n, v, rf(k));
      };
      Tensor xin = x;
      if (mm.dense) { auto h = to_host_vec<float>(x); std::vector<float> tr(h.size()); int C = m.cin, N = 4096;
        for (int c = 0; c < C; c++) for (int n = 0; n < N; n++) tr[n * C + c] = h[c * N + n]; xin = from_host(tr.data(), {N, C}, F32); }
      if (concat.p) {
        auto a = to_host_vec<float>(x), b = to_host_vec<float>(concat); i64 N = x.size(0); int ca = (int)x.size(1), cb = (int)concat.size(1);
        std::vector<float> c(N * (ca + cb));
        for (i64 r = 0; r < N; r++) { memcpy(&c[r * (ca + cb)], &a[r * ca], ca * 4); memcpy(&c[r * (ca + cb) + ca], &b[r * cb], cb * 4); }
        xin = from_host(c.data(), {N, ca + cb}, F32);
      }
      Timer tm; tm.start();
      Tensor out = m.forward(xin, t, cond);
      float ms_ = tm.ms();
      if (mm.dense) { auto h = to_host_vec<float>(out); std::vector<float> tr(h.size()); int C = m.cout, N = 4096;
        for (int c = 0; c < C; c++) for (int n = 0; n < N; n++) tr[c * N + n] = h[n * C + c]; out = from_host(tr.data(), {C, N}, F32); }
      cmp(p + "out", out, mm.dense ? rf(p + "out") : rf(p + "out.feats"));
      printf("  forward %.2f ms (N=%lld)\n", ms_, (long long)xin.size(0));
    } else if (cmd == "sample") {  // whole 12-step sampling from reference noise
      std::string s = "sample" + std::to_string(mm.sample) + "/";
      Tensor noise = mm.dense ? flat(rf(s + "noise"), 4096) : rf(s + "noise.feats");
      int nshow = 0;
      m.dbg = [&](const std::string& n, const Tensor& v) {
        auto k = std::string(mm.name) + "/" + n;
        if (R.count(k) && nshow < 6) { int f = g_fail; cmp(k, v, rf(k)); if (g_fail > f) nshow++; }
      };
      Timer tm; tm.start();
      Tensor out = flow_sample(m, noise, cond, neg, mm.s, concat.p ? &concat : nullptr);
      float ms_ = tm.ms();
      cmp(s + "out", out, mm.dense ? rf(s + "out") : rf(s + "out.feats"));
      printf("  sampling %.1f ms\n", ms_);
    }
  }
  printf(g_fail ? "PARITY FAIL (%d)\n" : "PARITY OK\n", g_fail);
  return g_fail;
}
