// Elementwise / norm / reduction kernels replicating ATen 2.6 CUDA arithmetic exactly (same op order, rounding
// points and reduction trees). Compiled without fast-math, like ATen.
#include "ops.h"
#include <cmath>

template <class T> __device__ __forceinline__ float ld(const T* p, i64 i) { return (float)p[i]; }
template <> __device__ __forceinline__ float ld<float>(const float* p, i64 i) { return p[i]; }
template <class T> __device__ __forceinline__ T st(float v) { return (T)v; }
template <> __device__ __forceinline__ float st<float>(float v) { return v; }

#define DISPATCH_F(dt, T, ...) \
  switch (dt) { case F32: { using T = float; __VA_ARGS__; } break; case BF16: { using T = bf16; __VA_ARGS__; } break; \
                case F16: { using T = f16; __VA_ARGS__; } break; default: REQ(false, "dtype %s", dname(dt)); }

// ------------------------------------------------------------------------------------------------ LayerNorm
// Port of at::native vectorized_layer_norm_kernel (float input, vec_size 4, block 32x4).
struct WD { float mean, sigma2, count; };
__device__ __forceinline__ WD wd_online(float v, WD c) {
  float delta = v - c.mean, nc = c.count + 1.f, nm = c.mean + delta * (1.f / nc);
  return {nm, c.sigma2 + delta * (v - nm), nc};
}
__device__ __forceinline__ WD wd_combine(WD b, WD a) {
  float delta = b.mean - a.mean, count = a.count + b.count, mean, sigma2;
  if (count > 0.f) {
    float coef = 1.f / count, nA = a.count * coef, nB = b.count * coef;
    mean = nA * a.mean + nB * b.mean; sigma2 = a.sigma2 + b.sigma2 + delta * delta * a.count * nB;
  } else { mean = 0.f; sigma2 = 0.f; }
  return {mean, sigma2, count};
}
__global__ void k_layer_norm(int N, float eps, const float* __restrict__ X, const float* g, const float* b, float* Y) {
  __shared__ float buf[6];
  const float4* xv = (const float4*)(X + (i64)blockIdx.x * N);
  int numx = blockDim.x * blockDim.y, thrx = threadIdx.x + threadIdx.y * blockDim.x, nv = N / 4;
  WD wd{0.f, 0.f, 0.f};
  for (int i = thrx; i < nv; i += numx) {
    float4 d = xv[i];
    wd = wd_online(d.x, wd); wd = wd_online(d.y, wd); wd = wd_online(d.z, wd); wd = wd_online(d.w, wd);
  }
  for (int o = 16; o > 0; o >>= 1) {
    WD w2{__shfl_down_sync(~0u, wd.mean, o), __shfl_down_sync(~0u, wd.sigma2, o), __shfl_down_sync(~0u, wd.count, o)};
    wd = wd_combine(wd, w2);
  }
  float* msb = buf; float* cb = buf + blockDim.y;
  for (int o = blockDim.y / 2; o > 0; o /= 2) {
    if (threadIdx.x == 0 && threadIdx.y >= o && threadIdx.y < 2 * o) { int w = threadIdx.y - o; msb[2 * w] = wd.mean; msb[2 * w + 1] = wd.sigma2; cb[w] = wd.count; }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y < o) wd = wd_combine(wd, WD{msb[2 * threadIdx.y], msb[2 * threadIdx.y + 1], cb[threadIdx.y]});
    __syncthreads();
  }
  if (threadIdx.x == 0 && threadIdx.y == 0) { msb[0] = wd.mean; msb[1] = wd.sigma2 / float(N); }
  __syncthreads();
  float mean = msb[0], rstd = rsqrtf(msb[1] + eps);
  float4* yv = (float4*)(Y + (i64)blockIdx.x * N);
  for (int i = thrx; i < nv; i += numx) {
    float4 d = xv[i], o; float* dd = (float*)&d; float* oo = (float*)&o;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      if (g && b) oo[j] = g[i * 4 + j] * (rstd * (dd[j] - mean)) + b[i * 4 + j];
      else if (g) oo[j] = g[i * 4 + j] * (rstd * (dd[j] - mean));
      else if (b) oo[j] = (rstd * (dd[j] - mean)) + b[i * 4 + j];
      else oo[j] = rstd * (dd[j] - mean);
    }
    yv[i] = o;
  }
}
Tensor layer_norm_f32(const Tensor& x, const Tensor* w, const Tensor* b, float eps) {
  REQ(x.dt == F32, "ln dtype"); int N = (int)x.size(-1); i64 M = x.numel() / N;
  REQ(N % 4 == 0 && (uintptr_t)x.p % 16 == 0, "ln vectorized path only");
  auto y = empty(x.sh, F32);
  if (M) k_layer_norm<<<M, dim3(32, 4), 0, stream()>>>(N, eps, x.ptr<float>(), w ? w->ptr<float>() : nullptr, b ? b->ptr<float>() : nullptr, y.ptr<float>());
  return y;
}
Tensor layer_norm(const Tensor& x, const Tensor* w, const Tensor* b, float eps) {
  if (x.dt == F32) return layer_norm_f32(x, w, b, eps);
  return cast(layer_norm_f32(cast(x, F32), w, b, eps), x.dt);
}

// ------------------------------------------------------------------------------------------------ activations
template <class T> __global__ void k_act(const T* x, T* y, i64 n, int a) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= n) return;
  float v = ld(x, i);
  if (a == SILU) { y[i] = st<T>(v / (1.f + expf(-v))); return; }
  if (a == GELU_ERF) { constexpr float kAlpha = M_SQRT1_2; y[i] = st<T>(v * float(0.5) * (float(1) + ::erff(v * kAlpha))); return; }
  constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * float(0.5);
  constexpr float kKappa = 0.044715;
  float x3 = v * v * v, inner = kBeta * (v + kKappa * x3);
  y[i] = st<T>(float(0.5) * v * (float(1) + tanhf(inner)));
}
void act_(Tensor& x, Act a) { DISPATCH_F(x.dt, T, k_act<T><<<cdiv(x.numel(), 256), 256, 0, stream()>>>(x.ptr<T>(), x.ptr<T>(), x.numel(), a)); }
Tensor act(const Tensor& x, Act a) { auto y = empty(x.sh, x.dt); DISPATCH_F(x.dt, T, k_act<T><<<cdiv(x.numel(), 256), 256, 0, stream()>>>(x.ptr<T>(), y.ptr<T>(), x.numel(), a)); return y; }

// ------------------------------------------------------------------------------------------------ modulation / residual
template <class T> __global__ void k_mod(const T* x, const T* sh, const T* sc, T* y, i64 n, int C) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= n) return;
  int c = i % C;
  T s1 = st<T>(1.f + ld(sc, c));
  T h = st<T>(__fmul_rn(ld(x, i), (float)s1));
  y[i] = st<T>(__fadd_rn((float)h, ld(sh, c)));
}
Tensor modulate(const Tensor& x, const Tensor& shift, const Tensor& scale) {
  auto y = empty(x.sh, x.dt); int C = (int)x.size(-1);
  DISPATCH_F(x.dt, T, k_mod<T><<<cdiv(x.numel(), 256), 256, 0, stream()>>>(x.ptr<T>(), shift.ptr<T>(), scale.ptr<T>(), y.ptr<T>(), x.numel(), C));
  return y;
}
template <class T> __global__ void k_gres(T* x, const T* h, const T* g, i64 n, int C) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= n) return;
  float hv = g ? (float)st<T>(__fmul_rn(ld(h, i), ld(g, i % C))) : ld(h, i);
  x[i] = st<T>(__fadd_rn(ld(x, i), hv));
}
void gated_residual_(Tensor& x, const Tensor& h, const Tensor* gate) {
  DISPATCH_F(x.dt, T, k_gres<T><<<cdiv(x.numel(), 256), 256, 0, stream()>>>(x.ptr<T>(), h.ptr<T>(), gate ? gate->ptr<T>() : nullptr, x.numel(), (int)x.size(-1)));
}
void add_(Tensor& x, const Tensor& y) { gated_residual_(x, y, nullptr); }
template <class T> __global__ void k_bias(T* y, const T* b, i64 n, i64 S, int C) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i < n) y[i] = st<T>(ld(y, i) + ld(b, i / S % C));
}
void bias_add_ncdhw_(Tensor& y, const Tensor& b) {
  int C = (int)y.size(1); i64 S = y.numel() / y.size(0) / C;
  Tensor bb = b.dt == y.dt ? b : cast(b, y.dt);
  DISPATCH_F(y.dt, T, k_bias<T><<<cdiv(y.numel(), 256), 256, 0, stream()>>>(y.ptr<T>(), bb.ptr<T>(), y.numel(), S, C));
}

// ------------------------------------------------------------------------------------------------ multi-head RMSNorm
// F.normalize: x / max(||x||, 1e-12); ||x|| via ATen norm reduce (one warp per row, vt0=4 strided accumulators,
// ascending shfl_down tree). Then *gamma (fp32 [H,D]) * scale (python float) and cast.
template <class T> __global__ void k_mhrms(const T* x, i64 rs, int H, int D, i64 rows, const float* gamma, float scale, T* y) {
  i64 r = blockIdx.x * (i64)blockDim.y + threadIdx.y; if (r >= rows * H) return;
  const T* xr = x + (r / H) * rs + (r % H) * D; int l = threadIdx.x;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  int idx = l;
  while (idx + 3 * 32 < D) {
#pragma unroll
    for (int i = 0; i < 4; i++) { float v = ld(xr, idx + i * 32); acc[i] = acc[i] + v * v; }
    idx += 128;
  }
#pragma unroll
  for (int i = 0; i < 4; i++) { if (idx >= D) break; float v = ld(xr, idx); acc[i] = acc[i] + v * v; idx += 32; }
  float s = acc[0]; s = s + acc[1]; s = s + acc[2]; s = s + acc[3];
  for (int o = 1; o < 32; o <<= 1) s = s + __shfl_down_sync(~0u, s, o);
  float nrm = fmaxf(sqrtf(__shfl_sync(~0u, s, 0)), 1e-12f);
  const float* gr = gamma + (r % H) * D; T* yr = y + r * D;
  for (int i = l; i < D; i += 32) { float t = ld(xr, i) / nrm; t = t * gr[i]; t = t * scale; yr[i] = st<T>(t); }
}
void mh_rmsnorm(const void* x, i64 rs, int H, int D, i64 rows, const Tensor& gamma, float scale, void* y, DT dt) {
  REQ(D == 128 && gamma.dt == F32, "rmsnorm cfg");
  int g = cdiv(rows * H, 16);
  DISPATCH_F(dt, T, k_mhrms<T><<<g, dim3(32, 16), 0, stream()>>>((const T*)x, rs, H, D, rows, gamma.ptr<float>(), scale, (T*)y));
}

// ------------------------------------------------------------------------------------------------ RoPE
template <class T> __global__ void k_rope(T* xy, i64 rows, int H, int D, const float2* ph) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; int half = D / 2; if (i >= rows * H * half) return;
  int j = i % half; i64 r = i / half / H;
  float a = ld(xy, 2 * i), b = ld(xy, 2 * i + 1); float2 p = ph[r * half + j];
  float c = p.x, d = p.y;
  float re = a * c - b * d, im = a * d + b * c;
  xy[2 * i] = st<T>(re); xy[2 * i + 1] = st<T>(im);
}
void rope_(void* xy, i64 rows, int H, int D, const Tensor& ph, DT dt) {
  i64 n = rows * H * (D / 2);
  DISPATCH_F(dt, T, k_rope<T><<<cdiv(n, 256), 256, 0, stream()>>>((T*)xy, rows, H, D, ph.ptr<float2>()));
}
// DINOv3 apply_rotary_pos_emb + transpose: y[h, t, :] (contiguous [H, T, D]) from x[t, h*D:] ([T, H*D]); the first
// P tokens are prefix (unrotated). Each torch op (mul, cat(-x2,x1), mul, add) rounds separately.
__global__ void k_rope_half(const float* x, i64 T, int H, int D, int P, const float* cs, const float* sn, float* y) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= T * H * D) return;
  int d = i % D, h = i / D % H; i64 t = i / D / H;
  const float* xr = x + t * H * D + h * D; float v = xr[d];
  if (t >= P) {
    int hd = D / 2; float r = d < hd ? -xr[d + hd] : xr[d - hd]; i64 o = (t - P) * D + d;
    v = __fadd_rn(__fmul_rn(v, cs[o]), __fmul_rn(r, sn[o]));
  }
  y[((i64)h * T + t) * D + d] = v;
}
Tensor rope_half_bhtd(const Tensor& x, int H, int P, const Tensor& cs, const Tensor& sn) {
  i64 T = x.size(0); int D = (int)x.size(1) / H; auto y = empty({H, T, D}, F32);
  k_rope_half<<<cdiv(x.numel(), 256), 256, 0, stream()>>>(x.ptr<float>(), T, H, D, P, cs.ptr<float>(), sn.ptr<float>(), y.ptr<float>());
  return y;
}
__global__ void k_phases(const int* c, i64 N, const float* f, int nf, int half, float2* ph) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= N * half) return;
  i64 n = i / half; int j = i % half;
  if (j >= 3 * nf) { ph[i] = make_float2(1.f * cosf(0.f), 1.f * sinf(0.f)); return; }
  float ang = (float)c[n * 4 + 1 + j / nf] * f[j % nf];
  ph[i] = make_float2(1.f * cosf(ang), 1.f * sinf(ang));
}
Tensor rope_phases(const Tensor& coords4, const Tensor& freqs, int D) {
  i64 N = coords4.size(0); int half = D / 2; auto ph = empty({N, half, 2}, F32);
  k_phases<<<cdiv(N * half, 256), 256, 0, stream()>>>(coords4.ptr<int>(), N, freqs.ptr<float>(), (int)freqs.numel(), half, ph.ptr<float2>());
  return ph;
}
__global__ void k_tfreq(float t, const float* f, int half, float* o) {
  int i = threadIdx.x + blockIdx.x * blockDim.x; if (i >= half) return;
  float a = t * f[i]; o[i] = cosf(a); o[half + i] = sinf(a);
}
Tensor timestep_freq_embed(float t, const Tensor& freqs) {
  int half = (int)freqs.numel(); auto o = empty({1, 2 * half}, F32);
  k_tfreq<<<cdiv(half, 128), 128, 0, stream()>>>(t, freqs.ptr<float>(), half, o.ptr<float>()); return o;
}

// ------------------------------------------------------------------------------------------------ reductions
// Inner-dim reduce [R, L] for L <= 128 not vectorized: block (32, 16), one warp per output (ATen config).
__global__ void k_row_reduce(const float* x, i64 R, int L, int op, float factor, float* out) {
  i64 r = blockIdx.x * (i64)blockDim.y + threadIdx.y; if (r >= R) return;
  const float* xr = x + r * L; int l = threadIdx.x;
  float acc[4] = {0.f, 0.f, 0.f, 0.f}; int idx = l;
  auto red = [&](float a, float v) { return op == 0 ? a + v * v : a + v; };
  while (idx + 3 * 32 < L) {
#pragma unroll
    for (int i = 0; i < 4; i++) acc[i] = red(acc[i], xr[idx + i * 32]);
    idx += 128;
  }
#pragma unroll
  for (int i = 0; i < 4; i++) { if (idx >= L) break; acc[i] = red(acc[i], xr[idx]); idx += 32; }
  float s = acc[0]; s = s + acc[1]; s = s + acc[2]; s = s + acc[3];
  for (int o = 1; o < 32; o <<= 1) s = s + __shfl_down_sync(~0u, s, o);
  if (l == 0) out[r] = op == 0 ? sqrtf(s) : s * factor;
}
Tensor row_reduce(const Tensor& x, int L, int op) {
  REQ(L <= 128 && x.dt == F32, "row_reduce cfg"); i64 R = x.numel() / L; auto o = empty({R}, F32);
  k_row_reduce<<<cdiv(R, 16), dim3(32, 16), 0, stream()>>>(x.ptr<float>(), R, L, op, float(1) / L, o.ptr<float>());
  return o;
}
// Welford full reduction, vt0=2, block 512x1, shared tree then ascending warp shuffles (ATen std_var_kernel).
struct WF { float mean, m2; int n; float nf; };
__device__ __forceinline__ WF wf_red(WF a, float d) {
  int nn = a.n + 1; float nnf = (float)nn, delta = d - a.mean, nm = a.mean + delta / nnf, nd = d - nm;
  return {nm, a.m2 + delta * nd, nn, nnf};
}
__device__ __forceinline__ WF wf_comb(WF a, WF b) {
  if (a.nf == 0) return b; if (b.nf == 0) return a;
  float delta = b.mean - a.mean, nc = a.nf + b.nf, nb = b.nf / nc;
  return {a.mean + delta * nb, a.m2 + b.m2 + delta * delta * a.nf * nb, -1, nc};
}
__global__ void k_std_all(const float* x, int n, float* out) {
  __shared__ WF sh[512];
  int t = threadIdx.x; WF v[2] = {{0, 0, 0, 0}, {0, 0, 0, 0}}; int idx = t;
  while (idx + 512 < n) { v[0] = wf_red(v[0], x[idx]); v[1] = wf_red(v[1], x[idx + 512]); idx += 1024; }
  for (int i = 0; i < 2; i++) { if (idx >= n) break; v[i] = wf_red(v[i], x[idx]); idx += 512; }
  WF a = wf_comb(v[0], v[1]);
  sh[t] = a;
  for (int o = 256; o >= 32; o >>= 1) { __syncthreads(); if (t < o) { a = wf_comb(a, sh[t + o]); sh[t] = a; } }
  __syncthreads();
  for (int o = 1; o < 32; o <<= 1) {
    WF b{__shfl_down_sync(~0u, a.mean, o), __shfl_down_sync(~0u, a.m2, o), __shfl_down_sync(~0u, a.n, o), __shfl_down_sync(~0u, a.nf, o)};
    a = wf_comb(a, b);
  }
  if (t == 0) { float div = a.nf > 1 ? a.nf - 1 : 0; out[0] = sqrtf(a.m2 / div); }
}
float std_all(const Tensor& x) {
  REQ(x.dt == F32 && x.numel() >= 32768, "std cfg");  // config valid for 512*16 <= n < 512*256
  auto o = empty({1}, F32); k_std_all<<<1, 512, 0, stream()>>>(x.ptr<float>(), (int)x.numel(), o.ptr<float>());
  return to_host_vec<float>(o)[0];
}
__global__ void k_seg_mean(const float* x, i64 n, float* out) { float s = 0.f; for (i64 i = 0; i < n; i++) s = s + x[i]; out[0] = s / (float)n; }
float segment_mean(const Tensor& x) { auto o = empty({1}, F32); k_seg_mean<<<1, 1, 0, stream()>>>(x.ptr<float>(), x.numel(), o.ptr<float>()); return to_host_vec<float>(o)[0]; }

// ------------------------------------------------------------------------------------------------ fp32 scalar ops
__global__ void k_axpby(const float* x, float a, const float* y, float b, float* o, i64 n, int mode) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= n) return;
  float u = __fmul_rn(x[i], a), v = __fmul_rn(y[i], b); o[i] = mode == 0 ? __fadd_rn(u, v) : __fsub_rn(u, v);
}
Tensor axpby(const Tensor& x, float a, const Tensor& y, float b, int mode) {
  auto o = empty(x.sh, F32); k_axpby<<<cdiv(x.numel(), 256), 256, 0, stream()>>>(x.ptr<float>(), a, y.ptr<float>(), b, o.ptr<float>(), x.numel(), mode); return o;
}
__global__ void k_scale(float* x, float a, float b, i64 n) { i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i < n) x[i] = __fadd_rn(__fmul_rn(x[i], a), b); }
void scale_(Tensor& x, float a, float b) { k_scale<<<cdiv(x.numel(), 256), 256, 0, stream()>>>(x.ptr<float>(), a, b, x.numel()); }
__global__ void k_mul(const float* x, const float* y, float* o, i64 n) { i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i < n) o[i] = x[i] * y[i]; }
Tensor mul_t(const Tensor& x, const Tensor& y) { auto o = empty(x.sh, F32); k_mul<<<cdiv(x.numel(), 256), 256, 0, stream()>>>(x.ptr<float>(), y.ptr<float>(), o.ptr<float>(), x.numel()); return o; }
