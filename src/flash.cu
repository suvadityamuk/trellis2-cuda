// flash-attn 2.7.3 forward (vendored kernels, hdim128 bf16, non-causal), params set exactly as flash_api.cpp does.
#include "ops.h"
#include "flash.h"
#include <cutlass/numeric_types.h>
#include <cmath>

template <typename T, int D, bool C> void run_mha_fwd_(Flash_fwd_params& p, cudaStream_t s);

// q/k/v: row-major [tokens, H, 128] views with given row strides (elements); o contiguous [Tq, H, 128].
// cu_q/cu_k null => dense batch-1 call (flash_attn_func); else varlen with B sequences.
void flash_attn(const void* q, i64 qrs, const void* k, i64 krs, const void* v, i64 vrs, void* o, int H, int Tq, int Tk,
                const int* cu_q, const int* cu_k, int B, int max_q, int max_k) {
  const int d = 128, dr = 128;
  bool varlen = cu_q != nullptr;
  int sq = varlen ? max_q : Tq, sk = varlen ? max_k : Tk;
  auto rm = [](int x, int m) { return (x + m - 1) / m * m; };
  Tensor lse = varlen ? empty({H, Tq}, F32) : empty({1, H, Tq}, F32);
  Flash_fwd_params p = {};
  p.is_bf16 = true;
  p.q_ptr = (void*)q; p.k_ptr = (void*)k; p.v_ptr = (void*)v; p.o_ptr = o;
  p.q_row_stride = qrs; p.k_row_stride = krs; p.v_row_stride = vrs; p.o_row_stride = (i64)H * d;
  p.q_head_stride = d; p.k_head_stride = d; p.v_head_stride = d; p.o_head_stride = d;
  if (!varlen) { p.q_batch_stride = qrs * Tq; p.k_batch_stride = krs * Tk; p.v_batch_stride = vrs * Tk; p.o_batch_stride = (i64)H * d * Tq; }
  p.cu_seqlens_q = (int*)cu_q; p.cu_seqlens_k = (int*)cu_k; p.seqused_k = nullptr;
  p.p_ptr = nullptr; p.softmax_lse_ptr = lse.p;
  p.b = varlen ? B : 1; p.h = H; p.h_k = H; p.h_h_k_ratio = 1;
  p.seqlen_q = sq; p.seqlen_k = sk; p.seqlen_q_rounded = rm(sq, 128); p.seqlen_k_rounded = rm(sk, 128);
  p.d = d; p.d_rounded = dr;
  float softmax_scale = (float)std::pow(128.0, -0.5);
  p.softcap = 0.0; p.scale_softmax = softmax_scale; p.scale_softmax_log2 = softmax_scale * M_LOG2E;
  p.p_dropout = 1.f - 0.f;
  p.p_dropout_in_uint8_t = uint8_t(std::floor(p.p_dropout * 255.0));
  p.rp_dropout = 1.f / p.p_dropout; p.scale_softmax_rp_dropout = p.rp_dropout * p.scale_softmax;
  p.is_causal = false; p.window_size_left = -1; p.window_size_right = -1;
  p.is_seqlens_k_cumulative = true;
  p.unpadded_lse = varlen; p.seqlenq_ngroups_swapped = false;
  if (varlen) p.total_q = Tq;
  p.page_block_size = 1;
  p.alibi_slopes_ptr = nullptr;
  p.num_splits = 0;
  if (!varlen) {  // set_params_splitkv heuristic: our shapes always fill the GPU => 1 split
    static int nsm = [] { int n; CK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0)); return n; }();
    REQ(H * cdiv(sq, 64) >= 0.8f * nsm * 2, "flash split-kv path not ported (H=%d sq=%d)", H, sq);
    p.num_splits = 1;
  }
  static Tensor rng = empty({2}, I64);
  p.rng_state = (uint64_t*)rng.p;
  run_mha_fwd_<cutlass::bfloat16_t, 128, false>(p, stream());
}
