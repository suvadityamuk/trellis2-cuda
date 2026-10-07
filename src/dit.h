#pragma once
#include "ops.h"
#include <functional>

void flash_attn(const void* q, i64 qrs, const void* k, i64 krs, const void* v, i64 vrs, void* o, int H, int Tq, int Tk,
                const int* cu_q, const int* cu_k, int B, int max_q, int max_k);
void fa3_fwd(const void* q, i64 qrs, const void* k, i64 krs, const void* v, i64 vrs, void* o, int H, int Tq, int Tk, float* lse, int num_sm,
             cudaStream_t s);

// TRELLIS.2 flow DiT (SparseStructureFlowModel when dense, SLatFlowModel otherwise). B=1 per call, like the official
// sampler (CFG runs cond and neg_cond as two separate calls).
struct DiT {
  static constexpr int C = 1536, H = 12, D = 128, NB = 30, F = 8192;
  bool dense = false;
  int cin = 0, cout = 0;
  struct Blk { Tensor mod, qkvW, qkvb, qg, kg, o1W, o1b, n2w, n2b, qW, qb, kvW, kvb, cqg, ckg, o2W, o2b, f1W, f1b, f2W, f2b; };
  std::vector<Blk> blk;
  Tensor t0W, t0b, t2W, t2b, aW, ab, inW, inb, outW, outb;
  Tensor phases;  // [N, 64] complex64: dense = fixed 16^3 grid, sparse = set per coords via set_coords
  std::map<const void*, std::vector<std::pair<Tensor, Tensor>>> kv;  // cond -> per-block (k_normed, kv)
  std::function<void(const std::string&, const Tensor&)> dbg;        // parity tap, names match ref_dump hook keys

  void load(const std::string& path, bool dense);
  void set_coords(const Tensor& coords4);  // int32 [N,4]
  // x: fp32 [N, cin] (token-major) ; t: python float (already *1000 in sampler) ; cond: fp32 [L, 1024]
  Tensor forward(const Tensor& x, float t, const Tensor& cond);
};

struct SamplerCfg { double gs, rescale, lo, hi, rescale_t; int steps = 12; double sigma_min = 1e-5; };  // python floats
// noise: dense [cin, 4096] channel-major (torch NCDHW) or sparse [N, cin]. concat (sparse only): [N, cc] appended to x.
Tensor flow_sample(DiT& m, const Tensor& noise, const Tensor& cond, const Tensor& neg, const SamplerCfg& c, const Tensor* concat = nullptr);
