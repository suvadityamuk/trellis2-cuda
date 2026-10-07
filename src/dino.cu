// DINOv3 ViT-L/16 (facebook/dinov3-vitl16-pretrain-lvd1689m, transformers 4.56 DINOv3ViTModel, sdpa) in fp32, as used by
// DinoV3FeatureExtractor.extract_features: embeddings -> 24 layers -> F.layer_norm (no affine, eps 1e-5).
#include "models.h"
#include <cmath>

static constexpr int C = 1024, H = 16, D = 64, L = 24, NREG = 4, PS = 16;

void Dino::load(const std::string& path) { W = load_safetensors(path); for (auto& [k, v] : W) v = cast(v, F32); }

// tokens [1 + NREG + P, C]: cls, registers, patch_embeddings.flatten(2).transpose(1, 2)
__global__ void k_tokens(const float* cls, const float* reg, const float* pe, int P, float* y) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= (i64)(1 + NREG + P) * C) return;
  int t = i / C, c = i % C;
  y[i] = t == 0 ? cls[c] : t <= NREG ? reg[(t - 1) * C + c] : pe[(i64)c * P + t - 1 - NREG];
}
// DINOv3ViTRopePositionEmbedding (eval): angles = 2*pi*coords[:, :, None] * inv_freq, flatten, tile(2), cos/sin
__global__ void k_dino_rope(int n, const float* inv, float* cs, float* sn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n * n * D) return;
  int d = i % D, p = i / D, a = (d % (D / 2)) / (D / 4), f = d % (D / 4);
  float c = ((a == 0 ? p / n : p % n) + 0.5f) * (1.0f / n);
  c = 2.0f * c - 1.0f;
  float ang = __fmul_rn(__fmul_rn((float)(2 * M_PI), c), inv[f]);
  cs[i] = cosf(ang); sn[i] = sinf(ang);
}

Tensor Dino::forward(const Tensor& img) {
  auto w = [&](const std::string& k) -> Tensor& { REQ(W.count(k), "dino missing %s", k.c_str()); return W[k]; };
  auto tap = [&](const std::string& n, const Tensor& v) { if (dbg) dbg(n, v); };
  int n = (int)img.size(2) / PS, P = n * n, T = 1 + NREG + P;
  Tensor pe = conv(img, w("embeddings.patch_embeddings.weight"), &w("embeddings.patch_embeddings.bias"), 0, PS);
  Tensor h = empty({T, C}, F32);
  k_tokens<<<cdiv((i64)T * C, 256), 256, 0, stream()>>>(w("embeddings.cls_token").ptr<float>(), w("embeddings.register_tokens").ptr<float>(),
                                                       pe.ptr<float>(), P, h.ptr<float>());
  tap("embeddings", h);
  std::vector<float> inv(D / 4);  // 1 / 100 ** arange(0, 1, 4 / D)
  for (int i = 0; i < D / 4; i++) inv[i] = 1.0f / (float)std::pow(100.0, (double)(i * (4.0f / D)));
  Tensor invd = from_host(inv.data(), {D / 4}, F32), cs = empty({P, D}, F32), sn = empty({P, D}, F32);
  k_dino_rope<<<cdiv(P * D, 256), 256, 0, stream()>>>(n, invd.ptr<float>(), cs.ptr<float>(), sn.ptr<float>());
  tap("rope.cos", cs); tap("rope.sin", sn);
  for (int l = 0; l < L; l++) {
    auto p = "layer." + std::to_string(l) + ".";
    auto lt = [&](const std::string& s, const Tensor& v) { if (l == 0) tap(p + s, v); };
    Tensor x = layer_norm_f32(h, &w(p + "norm1.weight"), &w(p + "norm1.bias"), 1e-5f); lt("norm1", x);
    Tensor q = linear(x, w(p + "attention.q_proj.weight"), &w(p + "attention.q_proj.bias"));
    Tensor k = linear(x, w(p + "attention.k_proj.weight"), nullptr);
    Tensor v = linear(x, w(p + "attention.v_proj.weight"), &w(p + "attention.v_proj.bias"));
    lt("attention.q_proj", q);
    Tensor qr = rope_half_bhtd(q, H, 1 + NREG, cs, sn), kr = rope_half_bhtd(k, H, 1 + NREG, cs, sn), o = empty({T, C}, F32);
    i64 bs = (i64)T * C;
    mem_eff_attn({qr.ptr<float>(), bs, D, (i64)T * D}, {kr.ptr<float>(), bs, D, (i64)T * D}, {v.ptr<float>(), bs, C, D},
                 o.ptr<float>(), 1, T, T, H, D, 0.125f);
    Tensor a = linear(o, w(p + "attention.o_proj.weight"), &w(p + "attention.o_proj.bias")); lt("attention", a);
    gated_residual_(h, a, &w(p + "layer_scale1.lambda1"));
    x = layer_norm_f32(h, &w(p + "norm2.weight"), &w(p + "norm2.bias"), 1e-5f);
    Tensor u = linear(x, w(p + "mlp.up_proj.weight"), &w(p + "mlp.up_proj.bias")); act_(u, GELU_ERF);
    Tensor m = linear(u, w(p + "mlp.down_proj.weight"), &w(p + "mlp.down_proj.bias")); lt("mlp", m);
    gated_residual_(h, m, &w(p + "layer_scale2.lambda1"));
    tap(p.substr(0, p.size() - 1), h);
  }
  return layer_norm_f32(h, nullptr, nullptr, 1e-5f);
}
