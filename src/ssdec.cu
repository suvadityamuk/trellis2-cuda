// SparseStructureDecoder (TRELLIS-image-large ss_dec_conv3d_16l8_fp16) + occupancy -> coords, as in
// Trellis2ImageTo3DPipeline.sample_sparse_structure.
#include "models.h"
#include <cub/cub.cuh>

template <class Ti, class To> __global__ void k_tr(const Ti* x, To* y, int C, i64 S) {  // y[s, c] = x[c, s]
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= C * S) return;
  i64 s = i / C; int c = i % C; y[i] = (To)(float)x[c * S + s];
}
template <class Ti, class To> __global__ void k_trb(const Ti* x, To* y, int C, i64 S) {  // y[c, s] = x[s, c]
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= C * S) return;
  int c = i / S; i64 s = i % S; y[i] = (To)(float)x[s * C + c];
}
// ChannelLayerNorm32: permute to channels-last, LayerNorm in fp32, cast back, permute back (pure data movement + LN).
static Tensor channel_ln(const Tensor& x, const Tensor& g, const Tensor& b) {
  int C = (int)x.size(1); i64 S = x.numel() / C;
  Tensor t = empty({S, C}, F32), y = empty(x.sh, x.dt);
  if (x.dt == F16) k_tr<f16, float><<<cdiv(C * S, 256), 256, 0, stream()>>>(x.ptr<f16>(), t.ptr<float>(), C, S);
  else k_tr<float, float><<<cdiv(C * S, 256), 256, 0, stream()>>>(x.ptr<float>(), t.ptr<float>(), C, S);
  t = layer_norm_f32(t, &g, &b, 1e-5f);
  if (x.dt == F16) k_trb<float, f16><<<cdiv(C * S, 256), 256, 0, stream()>>>(t.ptr<float>(), y.ptr<f16>(), C, S);
  else k_trb<float, float><<<cdiv(C * S, 256), 256, 0, stream()>>>(t.ptr<float>(), y.ptr<float>(), C, S);
  return y;
}
template <class T> __global__ void k_ps3(const T* x, T* y, int C, int R) {  // pixel_shuffle_3d(x, 2)
  i64 R2 = 2 * R, i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= (i64)C * R2 * R2 * R2) return;
  int d = i % R2, w = i / R2 % R2, h = i / (R2 * R2) % R2, c = i / (R2 * R2 * R2);
  y[i] = x[((((i64)c * 8 + (h & 1) * 4 + (w & 1) * 2 + (d & 1)) * R + h / 2) * R + w / 2) * R + d / 2];
}

void SSDecoder::load(const std::string& path) {
  W = load_safetensors(path);
  for (auto& [k, v] : W) {
    bool fp16 = k.rfind("blocks.", 0) == 0 || k.rfind("middle_block.", 0) == 0;
    bool conv = k.find("conv") != std::string::npos;
    v = cast(v, fp16 && conv ? F16 : F32);  // convert_to_fp16 only touches Conv3d params
  }
}
Tensor SSDecoder::res(const std::string& p, const Tensor& x) {
  auto w = [&](const std::string& k) { REQ(W.count(p + k), "ssdec missing %s", (p + k).c_str()); return W[p + k]; };
  Tensor h = channel_ln(x, w("norm1.weight"), w("norm1.bias")); act_(h, SILU);
  h = conv(h, w("conv1.weight"), &W[p + "conv1.bias"], 1, 1);
  h = channel_ln(h, w("norm2.weight"), w("norm2.bias")); act_(h, SILU);
  h = conv(h, w("conv2.weight"), &W[p + "conv2.bias"], 1, 1);
  add_(h, x);
  return h;
}
Tensor SSDecoder::forward(const Tensor& z) {  // z: fp32 [1, 8, 16, 16, 16] -> fp32 [1, 1, 64, 64, 64]
  Tensor h = cast(conv(z, W["input_layer.weight"], &W["input_layer.bias"], 1, 1), F16);
  for (int i = 0; i < 2; i++) h = res("middle_block." + std::to_string(i) + ".", h);
  int ch[3] = {512, 128, 32}, bi = 0;
  for (int l = 0; l < 3; l++) {
    for (int j = 0; j < 2; j++) h = res("blocks." + std::to_string(bi++) + ".", h);
    if (l < 2) {
      auto p = "blocks." + std::to_string(bi++) + ".";
      Tensor u = conv(h, W[p + "conv.weight"], &W[p + "conv.bias"], 1, 1);
      int R = (int)u.size(2), C = (int)u.size(1) / 8;
      Tensor y = empty({1, C, 2 * R, 2 * R, 2 * R}, F16);
      k_ps3<f16><<<cdiv(y.numel(), 256), 256, 0, stream()>>>(u.ptr<f16>(), y.ptr<f16>(), C, R);
      h = y; (void)ch;
    }
  }
  h = channel_ln(cast(h, F32), W["out_layer.0.weight"], W["out_layer.0.bias"]); act_(h, SILU);
  return conv(h, W["out_layer.2.weight"], &W["out_layer.2.bias"], 1, 1);
}

__global__ void k_occ(const float* v, int R, int r, int* flag) {  // max_pool3d(v > 0, R/r) > 0.5
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= r * r * r) return;
  int f = R / r, x = i / (r * r), y = i / r % r, z = i % r, o = 0;
  for (int a = 0; a < f; a++) for (int b = 0; b < f; b++) for (int c = 0; c < f; c++)
    o |= v[((i64)(x * f + a) * R + (y * f + b)) * R + z * f + c] > 0.f;
  flag[i] = o;
}
__global__ void k_coords(const int* idx, int n, int r, int* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  int v = idx[i]; out[i * 4] = 0; out[i * 4 + 1] = v / (r * r); out[i * 4 + 2] = v / r % r; out[i * 4 + 3] = v % r;
}
Tensor occupancy_coords(const Tensor& logits, int r) {
  int R = (int)logits.size(-1), n = r * r * r;
  Tensor flag = empty({n}, I32), idx = empty({n}, I32), cnt = empty({1}, I32);
  k_occ<<<cdiv(n, 256), 256, 0, stream()>>>(logits.ptr<float>(), R, r, flag.ptr<int>());
  cub::CountingInputIterator<int> it(0); size_t tb = 0;
  cub::DeviceSelect::Flagged(nullptr, tb, it, flag.ptr<int>(), idx.ptr<int>(), cnt.ptr<int>(), n, stream());
  Tensor tmp = empty({(i64)tb}, U8);
  cub::DeviceSelect::Flagged(tmp.p, tb, it, flag.ptr<int>(), idx.ptr<int>(), cnt.ptr<int>(), n, stream());
  int m = to_host_vec<int>(cnt)[0];
  Tensor out = empty({m, 4}, I32);
  if (m) k_coords<<<cdiv(m, 256), 256, 0, stream()>>>(idx.ptr<int>(), m, r, out.ptr<int>());
  return out;
}
