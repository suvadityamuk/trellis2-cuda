// Trellis2ImageTo3DPipeline.run (1024_cascade) up to decode_latent: image -> mesh (V, F) + PBR voxel attributes.
#include "models.h"
#include <cub/cub.cuh>

static const float kShapeMean[32] = {0.781296, 0.018091, -0.495192, -0.558457, 1.06053, 0.093252, 1.518149, -0.933218, -0.732996, 2.604095, -0.118341, -2.143904, 0.495076, -2.179512, -2.130751, -0.996944, 0.261421, -2.217463, 1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667, -0.059621, 2.22078, 1.621212, 0.87723, 0.567247, -3.175944, -3.186688, 1.578665};
static const float kShapeStd[32] = {5.972266, 4.706852, 5.44501, 5.209927, 5.32022, 4.547237, 5.020802, 5.444004, 5.226681, 5.683095, 4.831436, 5.286469, 5.652043, 5.367606, 5.525084, 4.730578, 4.805265, 5.124013, 5.530808, 5.619001, 5.10393, 5.41767, 5.269677, 5.547194, 5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207, 5.347008, 5.405691};
static const float kTexMean[32] = {3.501659, 2.212398, 2.226094, 0.251093, -0.026248, -0.687364, 0.439898, -0.928075, 0.029398, -0.339596, -0.869527, 1.038479, -0.972385, 0.126042, -1.129303, 0.455149, -1.209521, 2.069067, 0.544735, 2.569128, -0.323407, 2.293, -1.925608, -1.217717, 1.213905, 0.971588, -0.023631, 0.10675, 2.021786, 0.250524, -0.662387, -0.768862};
static const float kTexStd[32] = {2.665652, 2.743913, 2.765121, 2.595319, 3.037293, 2.291316, 2.144656, 2.911822, 2.969419, 2.501689, 2.154811, 3.163343, 2.621215, 2.381943, 3.186697, 3.021588, 2.295916, 3.234985, 3.233086, 2.26014, 2.874801, 2.810596, 3.29272, 2.674999, 2.680878, 2.372054, 2.451546, 2.353556, 2.995195, 2.379849, 2.786195, 2.77519};

// mode 0: x * s + m ; mode 1: (x - m) / s   (per column, each op rounded like separate ATen kernels)
__global__ void k_colnorm(const float* x, float* y, i64 n, int C, const float* s, const float* m, int mode) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= n) return;
  int c = (int)(i % C);
  y[i] = mode ? __fdiv_rn(__fsub_rn(x[i], m[c]), s[c]) : __fadd_rn(__fmul_rn(x[i], s[c]), m[c]);
}
static Tensor colnorm(const Tensor& x, const float* s, const float* m, int mode) {
  int C = (int)x.size(1); Tensor S = from_host(s, {C}, F32), M = from_host(m, {C}, F32), y = empty(x.sh, F32);
  k_colnorm<<<cdiv(x.numel(), 256), 256, 0, stream()>>>(x.ptr<float>(), y.ptr<float>(), x.numel(), C, S.ptr<float>(), M.ptr<float>(), mode);
  return y;
}

// cat([c[:, :1], ((c[:, 1:] + 0.5) / lr * (hr // 16)).int()]).unique(dim=0) for a single batch; keys sort lexicographically.
__global__ void k_quant(const int4* c, int n, float inv_lr, float g, int G, int* key) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  int4 q = c[i]; auto f = [&](int v) { return (int)__fmul_rn(__fmul_rn(__fadd_rn((float)v, 0.5f), inv_lr), g); };
  key[i] = (f(q.y) * G + f(q.z)) * G + f(q.w);
}
__global__ void k_unkey(const int* key, int n, int G, int4* c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  int k = key[i]; c[i] = make_int4(0, k / (G * G), k / G % G, k % G);
}
static Tensor quantize_unique(const Tensor& hr, int lr_res, int hr_res) {
  int n = (int)hr.size(0), G = hr_res / 16;
  Tensor k = empty({n}, I32), ks = empty({n}, I32), ku = empty({n}, I32), cnt = empty({1}, I32);
  k_quant<<<cdiv(n, 256), 256, 0, stream()>>>(hr.ptr<int4>(), n, 1.f / lr_res, (float)G, G, k.ptr<int>());
  size_t a = 0, b = 0;
  cub::DeviceRadixSort::SortKeys(nullptr, a, k.ptr<int>(), ks.ptr<int>(), n, 0, 32, stream());
  cub::DeviceSelect::Unique(nullptr, b, ks.ptr<int>(), ku.ptr<int>(), cnt.ptr<int>(), n, stream());
  Tensor tmp = empty({(i64)std::max(a, b)}, U8);
  cub::DeviceRadixSort::SortKeys(tmp.p, a, k.ptr<int>(), ks.ptr<int>(), n, 0, 32, stream());
  cub::DeviceSelect::Unique(tmp.p, b, ks.ptr<int>(), ku.ptr<int>(), cnt.ptr<int>(), n, stream());
  int m = to_host_vec<int>(cnt)[0];
  Tensor out = empty({m, 4}, I32);
  k_unkey<<<cdiv(m, 256), 256, 0, stream()>>>(ku.ptr<int>(), m, G, out.ptr<int4>());
  return out;
}

void Pipeline::load(const std::string& ck) {
  dino.load(ck + "/dinov3_vitl16.safetensors");
  ss.load(ck + "/ss_flow_img_dit_1_3B_64_bf16.safetensors", true);
  sh512.load(ck + "/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors", false);
  sh1024.load(ck + "/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors", false);
  tex.load(ck + "/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors", false);
  ssd.load(ck + "/ss_dec_conv3d_16l8_fp16.safetensors");
  shd.load(ck + "/shape_dec_next_dc_f16c32_fp16.safetensors", true);
  txd.load(ck + "/tex_dec_next_dc_f16c32_fp16.safetensors", false);
}

Pipeline::Out Pipeline::run(const Image8& rgba, uint64_t seed) {
  auto tap = [&](const std::string& n, const Tensor& v) { if (dbg) dbg(n, v); };
  auto tm = [&](const char* n) { if (times) { dsync(); times(n); } };
  std::mt19937 g((uint32_t)seed);
  for (DiT* m : {&ss, &sh512, &sh1024, &tex}) m->kv.clear();  // keyed by cond pointer, which the allocator may reuse
  Image8 pre = preprocess_image(rgba);
  Tensor c512 = dino.forward(image_to_input(resize_lanczos(pre, 512, 512)));
  Tensor c1024 = dino.forward(image_to_input(resize_lanczos(pre, 1024, 1024)));
  Tensor n512 = zeros(c512.sh, F32), n1024 = zeros(c1024.sh, F32);
  tap("cond512/cond", c512); tap("cond1024/cond", c1024); tm("cond");

  auto noise = [&](i64 r, i64 c) { auto h = torch_randn(g, r * c); return from_host(h.data(), {r, c}, F32); };
  Tensor z = flow_sample(ss, noise(8, 4096), c512, n512, {7.5, 0.7, 0.6, 1.0, 5.0});
  Tensor coords = occupancy_coords(ssd.forward(z.view({1, 8, 16, 16, 16})), 32);
  tap("ss/coords", coords); tm("sparse_structure");

  sh512.set_coords(coords);
  Tensor s = colnorm(flow_sample(sh512, noise(coords.size(0), 32), c512, n512, {7.5, 0.5, 0.6, 1.0, 3.0}), kShapeStd, kShapeMean, 0);
  Tensor hr; shd.run(s, coords, nullptr, nullptr, 4, &hr);
  tap("shape/hr_coords", hr);
  coords = quantize_unique(hr, 512, 1024);
  sh1024.set_coords(coords);
  s = colnorm(flow_sample(sh1024, noise(coords.size(0), 32), c1024, n1024, {7.5, 0.5, 0.6, 1.0, 3.0}), kShapeStd, kShapeMean, 0);
  tap("shape/slat.coords", coords); tap("shape/slat.feats", s); tm("shape_slat");

  tex.set_coords(coords);
  Tensor sn = colnorm(s, kShapeStd, kShapeMean, 1);
  Tensor t = colnorm(flow_sample(tex, noise(coords.size(0), 32), c1024, n1024, {1.0, 0.0, 0.6, 0.9, 3.0}, &sn), kTexStd, kTexMean, 0);
  tap("tex/slat.feats", t); tm("tex_slat");

  Out o; std::vector<Tensor> subs; Tensor co;
  Tensor h = shd.run(s, coords, nullptr, &subs, -1, &co);
  fdg_to_mesh(h, co, 1024, o.V, o.F);
  tap("dec_shape/v", o.V); tap("dec_shape/f", o.F);
  o.attrs = txd.run(t, coords, &subs, nullptr, -1, &o.coords);
  scale_(o.attrs, 0.5f, 0.5f);
  fill_holes(o.V, o.F, 3e-2f);
  tap("final/v", o.V); tap("final/f", o.F); tap("final/attrs", o.attrs); tm("decode");
  return o;
}
