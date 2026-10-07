// SparseUnetVaeDecoder / FlexiDualGridVaeDecoder (ConvNeXt + C2S res blocks, fp16 torso), as in
// trellis2/models/sc_vaes/sparse_unet_vae.py with FlexGEMM submanifold convs.
#include "models.h"
#include <cub/cub.cuh>

void SDecoder::load(const std::string& path, bool subdiv) {
  W = load_safetensors(path); pred_subdiv = subdiv;
  for (auto& [k, v] : W) {
    bool torso = k.rfind("blocks.", 0) == 0, ln = k.find("norm") != std::string::npos;
    v = cast(v, torso && !ln ? F16 : F32);  // convert_to_fp16 touches Linear/Conv in blocks only; LayerNorm32 stays fp32
  }
}
const Tensor& SDecoder::w(const std::string& k) { REQ(W.count(k), "sdec missing %s", k.c_str()); return W[k]; }

Tensor SDecoder::conv(const std::string& p, const Tensor& x, SpConvCache& c) {
  const Tensor& wt = w(p + ".weight");
  return subm_conv(x, wt.view({wt.size(0), 27, wt.size(4)}), &w(p + ".bias"), c);
}

// subdivision mask (logits > 0) -> per-parent child counts
__global__ void k_c2s_count(const f16* sub, int n, int* cnt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  int c = 0; for (int s = 0; s < 8; s++) c += __half2float(sub[i * 8 + s]) > 0.f;
  cnt[i] = c;
}
// children in (parent, s) ascending order: coords (b, 2x + s%2, 2y + s/2%2, 2z + s/4%2), src = parent*8 + s
__global__ void k_c2s_emit(const f16* sub, const int4* co, const int* off, int n, int4* nco, int* src) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  int r = off[i]; int4 q = co[i];
  for (int s = 0; s < 8; s++) if (__half2float(sub[i * 8 + s]) > 0.f) {
    nco[r] = make_int4(q.x, q.y * 2 + (s & 1), q.z * 2 + (s >> 1 & 1), q.w * 2 + (s >> 2 & 1)); src[r++] = i * 8 + s;
  }
}
__global__ void k_gather_rows(const f16* x, const int* src, i64 m, int C, f16* y) {  // y[r] = x.view(-1, C)[src[r]]
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= m * C) return;
  i64 r = i / C; int c = i % C; y[i] = x[(i64)src[r] * C + c];
}
// h += repeat_interleave(C2S(x), rep, dim=1): child channel c <- x[parent, s*Cs + c/rep]
__global__ void k_add_skip(f16* h, const f16* x, const int* src, i64 m, int Co, int Cin, int rep) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= m * Co) return;
  i64 r = i / Co; int c = i % Co, p = src[r] >> 3, s = src[r] & 7, Cs = Cin / 8;
  h[i] = __float2half_rn(__half2float(h[i]) + __half2float(x[(i64)p * Cin + s * Cs + c / rep]));
}

SDecoder::Level SDecoder::c2s(const Level& in, const Tensor& sub) {
  int n = (int)in.coords.size(0);
  Tensor cnt = empty({n}, I32), off = empty({n + 1}, I32);
  k_c2s_count<<<cdiv(n, 256), 256, 0, stream()>>>(sub.ptr<f16>(), n, cnt.ptr<int>());
  CK(cudaMemsetAsync(off.p, 0, 4, stream()));
  size_t tb = 0; cub::DeviceScan::InclusiveSum(nullptr, tb, cnt.ptr<int>(), off.ptr<int>() + 1, n, stream());
  Tensor tmp = empty({(i64)tb}, U8); cub::DeviceScan::InclusiveSum(tmp.p, tb, cnt.ptr<int>(), off.ptr<int>() + 1, n, stream());
  int m = to_host_vec<int>(off.slice0(n, n + 1))[0];
  Level o; o.coords = empty({m, 4}, I32); o.src = empty({m}, I32);
  k_c2s_emit<<<cdiv(n, 256), 256, 0, stream()>>>(sub.ptr<f16>(), in.coords.ptr<int4>(), off.ptr<int>(), n, o.coords.ptr<int4>(), o.src.ptr<int>());
  o.nb = std::make_shared<SpConvCache>(build_neighbors(o.coords, 1 << 12, 1 << 12, 1 << 12));
  return o;
}

Tensor SDecoder::convnext(const std::string& p, const Tensor& x, SpConvCache& c) {
  Tensor h = conv(p + ".conv", x, c);
  h = layer_norm(h, &w(p + ".norm.weight"), &w(p + ".norm.bias"), 1e-6f);
  h = linear(h, w(p + ".mlp.0.weight"), &w(p + ".mlp.0.bias"));
  act_(h, SILU);
  h = linear(h, w(p + ".mlp.2.weight"), &w(p + ".mlp.2.bias"));
  add_(h, x);
  return h;
}

// SparseResBlockC2S3d: returns (h at child level, subdiv logits used)
Tensor SDecoder::up(const std::string& p, const Tensor& x, Level& lv, const Tensor* guide, Tensor* sub_out) {
  Tensor sub = guide ? *guide : linear(x, w(p + ".to_subdiv.weight"), &w(p + ".to_subdiv.bias"));
  if (sub_out) *sub_out = sub;
  Tensor h = layer_norm(x, &w(p + ".norm1.weight"), &w(p + ".norm1.bias"), 1e-6f);
  act_(h, SILU);
  h = conv(p + ".conv1", h, *lv.nb);
  int Cin = (int)x.size(1), Co = (int)h.size(1) / 8;
  Level ch = c2s(lv, sub);
  i64 m = ch.coords.size(0);
  Tensor hc = empty({m, Co}, F16);
  k_gather_rows<<<cdiv(m * Co, 256), 256, 0, stream()>>>(h.ptr<f16>(), ch.src.ptr<int>(), m, Co, hc.ptr<f16>());
  h = layer_norm(hc, nullptr, nullptr, 1e-6f);
  act_(h, SILU);
  h = conv(p + ".conv2", h, *ch.nb);
  k_add_skip<<<cdiv(m * Co, 256), 256, 0, stream()>>>(h.ptr<f16>(), x.ptr<f16>(), ch.src.ptr<int>(), m, Co, Cin, Co / (Cin / 8));
  lv = ch;
  return h;
}

Tensor SDecoder::run(const Tensor& feats, const Tensor& coords, const std::vector<Tensor>* guide, std::vector<Tensor>* subs,
                     int stop_level, Tensor* coords_out) {
  auto tap = [&](const std::string& n, const Tensor& v) { if (dbg) dbg(n, v); };
  Level lv; lv.coords = coords; lv.nb = std::make_shared<SpConvCache>(build_neighbors(coords, 1 << 12, 1 << 12, 1 << 12));
  Tensor h = linear(feats, w("from_latent.weight"), &w("from_latent.bias"));
  tap("from_latent.out.feats", h);
  h = cast(h, F16);
  const int nblk[5] = {4, 16, 8, 4, 0};
  for (int i = 0; i < 5; i++) {
    if (i == stop_level) { if (coords_out) *coords_out = lv.coords; return Tensor(); }
    for (int j = 0; j <= nblk[i] - (i == 4); j++) {
      std::string p = "blocks." + std::to_string(i) + "." + std::to_string(j);
      if (j < nblk[i]) h = convnext(p, h, *lv.nb);
      else {
        Tensor s;
        h = up(p, h, lv, guide ? &(*guide)[i] : nullptr, &s);
        if (subs) subs->push_back(s);
        tap(p + ".out.coords", lv.coords);
      }
      tap(p + ".out.feats", h);
    }
  }
  if (coords_out) *coords_out = lv.coords;
  h = layer_norm_f32(cast(h, F32), nullptr, nullptr, 1e-5f);
  h = linear(h, w("output_layer.weight"), &w("output_layer.bias"));
  tap("output_layer.out.feats", h);
  return h;
}
