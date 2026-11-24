#include "dit.h"
#include <cmath>

std::vector<float> rope_freqs();
std::vector<float> t_freqs();

__global__ void k_add_cast(const float* a, const bf16* b, bf16* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = (bf16)(a[i] + (float)b[i]);
}
__global__ void k_transpose(const float* x, float* y, int R, int C) {  // y[c, r] = x[r, c]
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= (i64)R * C) return;
  int r = i / C, c = i % C; y[(i64)c * R + r] = x[i];
}
__global__ void k_cat2(const float* a, int ca, const float* b, int cb, float* o, i64 n) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; int w = ca + cb; if (i >= n * w) return;
  i64 r = i / w; int c = i % w; o[i] = c < ca ? a[r * ca + c] : b[r * cb + c - ca];
}
static Tensor transpose2(const Tensor& x, int R, int C) {
  auto y = empty({C, R}, F32); k_transpose<<<cdiv((i64)R * C, 256), 256, 0, stream()>>>(x.ptr<float>(), y.ptr<float>(), R, C); return y;
}

void DiT::load(const std::string& path, bool dense_) {
  dense = dense_;
  auto W = load_safetensors(path);
  auto g = [&](const std::string& k, DT dt) { REQ(W.count(k), "missing %s", k.c_str()); auto& t = W[k]; return t.dt == dt ? t : cast(t, dt); };
  t0W = g("t_embedder.mlp.0.weight", F32); t0b = g("t_embedder.mlp.0.bias", F32);
  t2W = g("t_embedder.mlp.2.weight", F32); t2b = g("t_embedder.mlp.2.bias", F32);
  aW = g("adaLN_modulation.1.weight", F32); ab = g("adaLN_modulation.1.bias", F32);
  inW = g("input_layer.weight", F32); inb = g("input_layer.bias", F32);
  outW = g("out_layer.weight", F32); outb = g("out_layer.bias", F32);
  cin = (int)inW.size(1); cout = (int)outW.size(0);
  blk.resize(NB);
  for (int i = 0; i < NB; i++) {
    auto p = "blocks." + std::to_string(i) + ".";
    auto b = [&](const std::string& k) { return g(p + k, BF16); };
    auto f = [&](const std::string& k) { return g(p + k, F32); };
    blk[i] = {f("modulation"), b("self_attn.to_qkv.weight"), b("self_attn.to_qkv.bias"), f("self_attn.q_rms_norm.gamma"),
              f("self_attn.k_rms_norm.gamma"), b("self_attn.to_out.weight"), b("self_attn.to_out.bias"), f("norm2.weight"),
              f("norm2.bias"), b("cross_attn.to_q.weight"), b("cross_attn.to_q.bias"), b("cross_attn.to_kv.weight"),
              b("cross_attn.to_kv.bias"), f("cross_attn.q_rms_norm.gamma"), f("cross_attn.k_rms_norm.gamma"),
              b("cross_attn.to_out.weight"), b("cross_attn.to_out.bias"), b("mlp.mlp.0.weight"), b("mlp.mlp.0.bias"),
              b("mlp.mlp.2.weight"), b("mlp.mlp.2.bias")};
  }
  if (dense) {  // RotaryPositionEmbedder buffer, computed on CPU at init (scalar polar => libm cosf/sinf)
    auto fr = rope_freqs();
    std::vector<float> ph(4096 * 64 * 2);
    for (int n = 0; n < 4096; n++)
      for (int j = 0; j < 64; j++) {
        int crd[3] = {n / 256, n / 16 % 16, n % 16};
        float a = j < 63 ? (float)crd[j / 21] * fr[j % 21] : 0.f;
        ph[(n * 64 + j) * 2] = 1.f * std::cos(a); ph[(n * 64 + j) * 2 + 1] = 1.f * std::sin(a);
      }
    phases = from_host(ph.data(), {4096, 64, 2}, F32);
  }
}

// torch CPU: 1.0 / 10000.0 ** (arange(21)/21)  and  exp(-log(10000) * arange(128) / 128)
std::vector<float> rope_freqs() {
  std::vector<float> f(21);
  for (int i = 0; i < 21; i++) f[i] = 1.f / std::pow(10000.f, (float)i / 21.f);
  return f;
}
// torch CPU exp goes through MKL VML (AVX512 path): correctly rounded except these entries, which are +1 ulp.
std::vector<float> t_freqs() {
  std::vector<float> f(128); const float c = (float)-std::log(10000.0);
  for (int i = 0; i < 128; i++) f[i] = (float)std::exp((double)((c * (float)i) / 128.f));
  for (int i : {13, 44, 119}) f[i] = std::nextafter(f[i], INFINITY);
  return f;
}
void DiT::set_coords(const Tensor& coords4) {
  static Tensor fr = [] { auto v = rope_freqs(); return from_host(v.data(), {21}, F32); }();
  phases = rope_phases(coords4, fr, D);
}


// ------------------------------------------------------------------------------------------------ fast build (T2_FAST=1)
// Fused bf16 kernels: same math as the parity path but without the fp32 round trips and the intermediate bf16
// roundings between consecutive elementwise ops. One block (128 threads) per token row, C = 1536.
// x += h * g (h, g optional; x keeps its bf16 rounding since it is the residual state), then
// y = LayerNorm(x) * A + B with (A, B) = (1 + sc, sh) [adaLN modulate] or (w, b) [affine] or (1, 0).
__global__ void k_res_ln(bf16* __restrict__ X, const bf16* __restrict__ Hh, const bf16* g, const bf16* sh, const bf16* sc, const float* w,
                         const float* b, bf16* __restrict__ Y, float eps) {
  constexpr int C = DiT::C, NT = 128, E = C / NT;
  __shared__ float s1[4], s2[4];
  bf16* x = X + (i64)blockIdx.x * C; int t = threadIdx.x;
  float v[E], sum = 0.f;
#pragma unroll
  for (int j = 0; j < E; j++) {
    int i = t + j * NT; float a = (float)x[i];
    if (Hh) { float hv = (float)Hh[(i64)blockIdx.x * C + i]; if (g) hv *= (float)g[i]; bf16 r = (bf16)(a + hv); x[i] = r; a = (float)r; }
    v[j] = a; sum += a;
  }
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(~0u, sum, o);
  if ((t & 31) == 0) s1[t >> 5] = sum;
  __syncthreads();
  float mean = (s1[0] + s1[1] + s1[2] + s1[3]) * (1.f / C), ss = 0.f;
#pragma unroll
  for (int j = 0; j < E; j++) { v[j] -= mean; ss += v[j] * v[j]; }
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(~0u, ss, o);
  if ((t & 31) == 0) s2[t >> 5] = ss;
  __syncthreads();
  float rstd = rsqrtf((s2[0] + s2[1] + s2[2] + s2[3]) * (1.f / C) + eps);
  bf16* y = Y + (i64)blockIdx.x * C;
#pragma unroll
  for (int j = 0; j < E; j++) {
    int i = t + j * NT; float o = v[j] * rstd;
    if (sc) o = o * (1.f + (float)sc[i]) + (float)sh[i];
    else if (w) o = o * w[i] + b[i];
    y[i] = (bf16)o;
  }
}
// Multi-head QK RMSNorm (F.normalize * gamma * sqrt(D)) + RoPE in one pass. One warp per (row, head), 4 elems per lane.
// src: [rows, >= ns*H*D] with row stride rs; stream j (j < ns) is read at offset j*H*D and written to dst[j] ([rows, H, D]).
__global__ void k_qk_rope(const bf16* __restrict__ src, i64 rs, i64 rows, const float* g0, const float* g1, float scale, const float2* ph,
                          bf16* __restrict__ d0, bf16* __restrict__ d1) {
  constexpr int H = DiT::H, D = DiT::D;
  i64 r = blockIdx.x * (i64)blockDim.y + threadIdx.y; if (r >= rows * H) return;
  i64 row = r / H; int h = r % H, l = threadIdx.x;
#pragma unroll
  for (int j = 0; j < 2; j++) {
    bf16* dst = j ? d1 : d0; if (!dst) continue;
    const float* gm = (j ? g1 : g0) + h * D + l * 4;
    const uint2 raw = *(const uint2*)(src + row * rs + (i64)j * H * D + h * D + l * 4);
    const bf16* e = (const bf16*)&raw;
    float v[4] = {(float)e[0], (float)e[1], (float)e[2], (float)e[3]};
    float s = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
    float k = scale / fmaxf(sqrtf(s), 1e-12f);
#pragma unroll
    for (int i = 0; i < 4; i++) v[i] = v[i] * k * gm[i];
    if (ph) {
      float2 p0 = ph[row * (D / 2) + l * 2], p1 = ph[row * (D / 2) + l * 2 + 1];
      float a = v[0], b = v[1]; v[0] = a * p0.x - b * p0.y; v[1] = a * p0.y + b * p0.x;
      a = v[2]; b = v[3]; v[2] = a * p1.x - b * p1.y; v[3] = a * p1.y + b * p1.x;
    }
    uint2 out; bf16* oe = (bf16*)&out;
#pragma unroll
    for (int i = 0; i < 4; i++) oe[i] = (bf16)v[i];
    *(uint2*)(dst + r * D + l * 4) = out;
  }
}
static void res_ln(Tensor& x, const Tensor* h, const bf16* g, const bf16* sh, const bf16* sc, const Tensor* w, const Tensor* b, Tensor& y) {
  k_res_ln<<<(unsigned)x.size(0), 128, 0, stream()>>>(x.ptr<bf16>(), h ? h->ptr<bf16>() : nullptr, g, sh, sc, w ? w->ptr<float>() : nullptr,
                                                      b ? b->ptr<float>() : nullptr, y.ptr<bf16>(), 1e-6f);
}
static void qk_rope(const void* src, i64 rs, i64 rows, const Tensor& g0, const Tensor* g1, const Tensor* ph, Tensor& d0, Tensor* d1) {
  k_qk_rope<<<cdiv(rows * DiT::H, 8), dim3(32, 8), 0, stream()>>>((const bf16*)src, rs, rows, g0.ptr<float>(), g1 ? g1->ptr<float>() : nullptr,
      (float)std::sqrt(128.0), ph ? ph->ptr<float2>() : nullptr, d0.ptr<bf16>(), d1 ? d1->ptr<bf16>() : nullptr);
}
// T2_DITPROF=1: per-op wall time (device-synchronised), printed per sampler run.
static struct DProf {
  bool on = env("T2_DITPROF") == "1"; double t = 0; std::map<std::string, double> acc;
  void start() { if (on) { dsync(); t = now_ms(); } }
  void tick(const char* n) { if (on) { dsync(); double x = now_ms(); acc[n] += x - t; t = x; } }
  void dump(const char* tag) {
    if (!on) return;
    double tot = 0; for (auto& [k, v] : acc) tot += v;
    printf("    dit[%s] %.1f ms:", tag, tot); for (auto& [k, v] : acc) printf(" %s %.1f", k.c_str(), v); printf("\n"); acc.clear();
  }
} dprof;
static bool fuse_mode() { static bool f = fast_mode() && env("T2_DIT_FUSE", "1") == "1"; return f; }
static bool gelu_epi() { static bool f = fast_mode() && env("T2_DIT_GELU_EPI", "1") == "1"; return f; }

#define TAP(n, t) do { if (dbg) dbg(n, t); } while (0)
static const float kRmsScale = (float)std::sqrt(128.0);

Tensor DiT::forward(const Tensor& x, float t, const Tensor& cond_f) {
  const i64 N = x.size(0);
  static Tensor tf = [] { auto v = t_freqs(); return from_host(v.data(), {128}, F32); }();
  Tensor tfe = timestep_freq_embed(t, tf);
  TAP("t_embedder.mlp.0.in", tfe);
  Tensor te = linear(tfe, t0W, &t0b);
  TAP("t_embedder.mlp.0.out", te);
  act_(te, SILU);
  te = linear(te, t2W, &t2b);
  TAP("t_embedder.out", te);
  act_(te, SILU);
  Tensor modf = linear(te, aW, &ab);
  TAP("adaLN_modulation.out", modf);
  Tensor mod = cast(modf, BF16);  // [1, 6C]
  Tensor h0 = linear(x, inW, &inb);
  TAP("input_layer.out", h0);
  Tensor xs = cast(h0, BF16);
  Tensor cond = cast(cond_f, BF16);
  const i64 L = cond.size(0);

  auto& cache = kv[cond_f.p];
  if (cache.empty()) {
    for (auto& b : blk) {
      Tensor kvt = linear(cond, b.kvW, &b.kvb);  // [L, 2, H, D]
      Tensor kn = empty({L, H, D}, BF16);
      mh_rmsnorm(kvt.p, 2 * C, H, D, L, b.ckg, kRmsScale, kn.p, BF16);
      cache.push_back({kn, kvt});
    }
  }
  int cuq[2] = {0, (int)N}, cuk[2] = {0, (int)L};
  static Tensor cu_q, cu_k; static int cached_n = -1, cached_l = -1;
  if (!dense && (cached_n != N || cached_l != L)) { cu_q = from_host(cuq, {2}, I32); cu_k = from_host(cuk, {2}, I32); cached_n = N; cached_l = L; }
  auto attn = [&](const void* q, i64 qrs, const void* k, i64 krs, const void* v, i64 vrs, void* o, i64 Tk) {
#ifndef T2_NO_FA3
    if (fast_mode()) {
      static int nsm = [] { int n; CK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0)); return n; }();
      static Tensor lse; if (lse.numel() < H * N) lse = empty({H * N}, F32);
      return fa3_fwd(q, qrs, k, krs, v, vrs, o, H, (int)N, (int)Tk, lse.ptr<float>(), nsm, stream());
    }
#endif
    if (dense) flash_attn(q, qrs, k, krs, v, vrs, o, H, (int)N, (int)Tk, nullptr, nullptr, 1, 0, 0);
    else flash_attn(q, qrs, k, krs, v, vrs, o, H, (int)N, (int)Tk, cu_q.ptr<int>(), (Tk == N ? cu_q : cu_k).ptr<int>(), 1, (int)N, (int)Tk);
  };

  Tensor q = empty({N, H, D}, BF16), k = empty({N, H, D}, BF16), o = empty({N, H, D}, BF16);

  if (fuse_mode()) {
    dprof.start();
    Tensor mall = empty({NB, 6 * C}, BF16);
    for (int i = 0; i < NB; i++)
      k_add_cast<<<cdiv(6 * C, 256), 256, 0, stream()>>>(blk[i].mod.ptr<float>(), mod.ptr<bf16>(), mall.ptr<bf16>() + (i64)i * 6 * C, 6 * C);
    auto ch = [&](int i, int j) { return mall.ptr<bf16>() + ((i64)i * 6 + j) * C; };
    Tensor h = empty({N, C}, BF16), h2 = empty({N, C}, BF16), qkv = empty({N, 3 * C}, BF16), f = empty({N, blk[0].f1W.size(0)}, BF16), ov = o.view({N, C});
    res_ln(xs, nullptr, nullptr, ch(0, 0), ch(0, 1), nullptr, nullptr, h); dprof.tick("norm");
    for (int i = 0; i < NB; i++) {
      auto& b = blk[i];
      linear(h, b.qkvW, &b.qkvb, qkv); dprof.tick("gemm_qkv");
      qk_rope(qkv.p, 3 * C, N, b.qg, &b.kg, &phases, q, &k); dprof.tick("qknorm");
      dprof.tick("qknorm"); attn(q.p, C, k.p, C, qkv.ptr<bf16>() + 2 * C, 3 * C, o.p, N); dprof.tick("self_attn"); dprof.tick("self_attn");
      linear(ov, b.o1W, &b.o1b, h2); dprof.tick("gemm_o");
      res_ln(xs, &h2, ch(i, 2), nullptr, nullptr, &b.n2w, &b.n2b, h); dprof.tick("norm");
      linear(h, b.qW, &b.qb, h2); dprof.tick("gemm_o");
      qk_rope(h2.p, C, N, b.cqg, nullptr, nullptr, q, nullptr); dprof.tick("qknorm");
      auto& [kn, kvt] = cache[i];
      dprof.tick("qknorm"); attn(q.p, C, kn.p, C, kvt.ptr<bf16>() + C, 2 * C, o.p, L); dprof.tick("cross_attn"); dprof.tick("cross_attn");
      linear(ov, b.o2W, &b.o2b, h2); dprof.tick("gemm_o");
      res_ln(xs, &h2, nullptr, ch(i, 3), ch(i, 4), nullptr, nullptr, h); dprof.tick("norm");
      linear(h, b.f1W, &b.f1b, f, gelu_epi()); dprof.tick("gemm_f1");
      if (!gelu_epi()) { act_(f, GELU_TANH); dprof.tick("gelu"); }
      linear(f, b.f2W, &b.f2b, h2); dprof.tick("gemm_f2");
      if (i + 1 < NB) res_ln(xs, &h2, ch(i, 5), ch(i + 1, 0), ch(i + 1, 1), nullptr, nullptr, h);
      else { Tensor g = h2; g.p = ch(i, 5); g.sh = {C}; g.own = nullptr; gated_residual_(xs, h2, &g); }
      dprof.tick("norm");
    }
    Tensor y = linear(layer_norm_f32(cast(xs, F32), nullptr, nullptr, 1e-5f), outW, &outb); dprof.tick("out");
    return y;
  }
  static const int tb = atoi(env("T2_TAPBLK", "0").c_str());
  dprof.start();
  for (int i = 0; i < NB; i++) {
    auto& b = blk[i];
    const std::string bn = "blocks." + std::to_string(i) + ".";
    Tensor m = empty({6 * C}, BF16);
    k_add_cast<<<cdiv(6 * C, 256), 256, 0, stream()>>>(b.mod.ptr<float>(), mod.ptr<bf16>(), m.ptr<bf16>(), 6 * C);
    auto ch = [&](int j) { Tensor t = m; t.p = m.ptr<bf16>() + j * C; t.sh = {C}; return t; };
    // self-attention
    Tensor h = modulate(layer_norm(xs, nullptr, nullptr, 1e-6f), ch(0), ch(1));
    if (i == tb) TAP(bn + "self_attn.to_qkv.in", h);
    dprof.tick("norm"); Tensor qkv = linear(h, b.qkvW, &b.qkvb); dprof.tick("gemm_qkv");  // [N, 3, H, D]
    if (i == tb) TAP(bn + "self_attn.to_qkv.out", qkv);
    mh_rmsnorm(qkv.p, 3 * C, H, D, N, b.qg, kRmsScale, q.p, BF16);
    mh_rmsnorm(qkv.ptr<bf16>() + C, 3 * C, H, D, N, b.kg, kRmsScale, k.p, BF16);
    if (i == tb) { TAP(bn + "self_attn.q_rms_norm.out", q); TAP(bn + "self_attn.k_rms_norm.out", k); }
    rope_(q.p, N, H, D, phases, BF16);
    rope_(k.p, N, H, D, phases, BF16);
    dprof.tick("qknorm"); attn(q.p, C, k.p, C, qkv.ptr<bf16>() + 2 * C, 3 * C, o.p, N); dprof.tick("self_attn");
    if (i == tb) TAP(bn + "self_attn.to_out.in", o);
    h = linear(o.view({N, C}), b.o1W, &b.o1b); dprof.tick("gemm_o");
    if (i == tb) TAP(bn + "self_attn.out", h);
    Tensor g = ch(2);
    gated_residual_(xs, h, &g);
    // cross-attention
    h = layer_norm(xs, &b.n2w, &b.n2b, 1e-6f);
    dprof.tick("norm"); Tensor cq = linear(h, b.qW, &b.qb); dprof.tick("gemm_o");
    mh_rmsnorm(cq.p, C, H, D, N, b.cqg, kRmsScale, q.p, BF16);
    auto& [kn, kvt] = cache[i];
    dprof.tick("qknorm"); attn(q.p, C, kn.p, C, kvt.ptr<bf16>() + C, 2 * C, o.p, L); dprof.tick("cross_attn");
    if (i == tb) TAP(bn + "cross_attn.to_out.in", o);
    h = linear(o.view({N, C}), b.o2W, &b.o2b); dprof.tick("gemm_o");
    if (i == tb) TAP(bn + "cross_attn.out", h);
    add_(xs, h);
    // MLP
    if (i == tb) TAP(bn + "norm3.in", xs);
    h = layer_norm(xs, nullptr, nullptr, 1e-6f);
    if (i == tb) TAP(bn + "norm3.out", h);
    h = modulate(h, ch(3), ch(4));
    if (i == tb) TAP(bn + "mlp.mlp.0.in", h);
    dprof.tick("norm"); h = linear(h, b.f1W, &b.f1b); dprof.tick("gemm_f1");
    if (i == tb) TAP(bn + "mlp.mlp.0.out", h);
    act_(h, GELU_TANH); dprof.tick("gelu");
    if (i == tb) TAP(bn + "mlp.mlp.1.out", h);
    h = linear(h, b.f2W, &b.f2b); dprof.tick("gemm_f2");
    if (i == tb) TAP(bn + "mlp.out", h);
    g = ch(5);
    gated_residual_(xs, h, &g); dprof.tick("norm");
    TAP("blocks." + std::to_string(i) + ".out", xs);
  }
  Tensor y = layer_norm_f32(cast(xs, F32), nullptr, nullptr, 1e-5f);
  return linear(y, outW, &outb);
}

// ------------------------------------------------------------------------------------------------ sampler
static float sparse_std(const Tensor& x) {  // VarLenTensor.std(dim=1): sqrt(mean(x^2) - mean(x)^2), single batch
  int C = (int)x.size(1);
  float m = segment_mean(row_reduce(x, C, 1));
  float m2 = segment_mean(row_reduce(mul_t(x, x), C, 1));
  return sqrtf(m2 - m * m);
}

Tensor flow_sample(DiT& m, const Tensor& noise, const Tensor& cond, const Tensor& neg, const SamplerCfg& c, const Tensor* concat) {
  const double r = c.rescale_t, sm = c.sigma_min, step = -1.0 / c.steps;
  std::vector<double> ts(c.steps + 1);
  for (int i = 0; i <= c.steps; i++) {
    double t = i == c.steps ? 0.0 : i * step + 1.0;  // np.linspace(1, 0, steps+1)
    ts[i] = r * t / (1 + (r - 1) * t);
  }
  int ncall = 0;
  auto call = [&](const Tensor& x, double t, const Tensor& cnd) {
    float tt = (float)(1000 * t);
    std::string p = "call" + std::to_string(ncall++) + "/";
    if (m.dbg) { m.dbg(p + (m.dense ? "x" : "x.feats"), x); m.dbg(p + "t", from_host(&tt, {1}, F32)); }
    Tensor o;
    if (m.dense) {
      int Cc = (int)x.size(0), Nn = (int)x.size(1);
      o = transpose2(m.forward(transpose2(x, Cc, Nn), tt, cnd), Nn, m.cout);
    } else if (concat) {
      auto xc = empty({x.size(0), x.size(1) + concat->size(1)}, F32);
      k_cat2<<<cdiv(xc.numel(), 256), 256, 0, stream()>>>(x.ptr<float>(), (int)x.size(1), concat->ptr<float>(), (int)concat->size(1), xc.ptr<float>(), x.size(0));
      o = m.forward(xc, tt, cnd);
    } else o = m.forward(x, tt, cnd);
    if (m.dbg) m.dbg(p + (m.dense ? "out" : "out.feats"), o);
    return o;
  };
  auto stdf = [&](const Tensor& x) { return m.dense ? std_all(x) : sparse_std(x); };
  Tensor x = clone(noise);
  struct D_ { ~D_() { dprof.dump("sampler"); } } d_;
  for (int i = 0; i < c.steps; i++) {
    double t = ts[i], tp = ts[i + 1];
    Tensor pred;
    if (c.gs == 1 || !(c.lo <= t && t <= c.hi)) pred = call(x, t, cond);
    else {
      Tensor pos = call(x, t, cond), ng = call(x, t, neg);
      pred = axpby(pos, (float)c.gs, ng, (float)(1 - c.gs), 0);
      if (c.rescale > 0) {
        float a = (float)(1 - sm), bb = (float)(sm + (1 - sm) * t);
        Tensor x0p = axpby(x, a, pos, bb, 1), x0c = axpby(x, a, pred, bb, 1);
        float ratio = stdf(x0p) / stdf(x0c);
        Tensor x0r = clone(x0c);
        scale_(x0r, ratio);
        Tensor x0 = axpby(x0r, (float)c.rescale, x0c, (float)(1 - c.rescale), 0);
        pred = axpby(x, a, x0, 1.f, 1);
        scale_(pred, 1.f / bb);
      }
    }
    x = axpby(x, 1.f, pred, (float)(t - tp), 1);
  }
  return x;
}
