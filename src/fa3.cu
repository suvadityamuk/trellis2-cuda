// FlashAttention-3 (flash-attention v2.7.3 hopper/) forward for the fast build: bf16 hdim128, non-causal, batch 1, no split.
// Params as hopper/flash_api.cpp mha_fwd sets them. Built with flash/Flash_*_params renamed to avoid clashing with FA2.
#include "flash.h"
#include <cutlass/numeric_types.h>
#include <cmath>
#include <cstdint>

template <int Arch, typename T, int D, bool Split, bool PagedKV, bool Softcap, bool PackGQA>
void run_mha_fwd_(Flash_fwd_params& p, cudaStream_t s);

void fa3_fwd(const void* q, int64_t qrs, const void* k, int64_t krs, const void* v, int64_t vrs, void* o, int H, int Tq, int Tk,
             float* lse, int num_sm, cudaStream_t s) {
  auto rm = [](int x, int m) { return (x + m - 1) / m * m; };
  Flash_fwd_params p = {};
  p.is_bf16 = true;
  p.q_ptr = (void*)q; p.k_ptr = (void*)k; p.v_ptr = (void*)v; p.o_ptr = o;
  p.q_row_stride = qrs; p.k_row_stride = krs; p.v_row_stride = vrs; p.o_row_stride = (int64_t)H * 128;
  p.q_head_stride = p.k_head_stride = p.v_head_stride = p.o_head_stride = 128; p.v_dim_stride = 1;
  p.q_batch_stride = qrs * Tq; p.k_batch_stride = krs * Tk; p.v_batch_stride = vrs * Tk; p.o_batch_stride = (int64_t)H * 128 * Tq;
  p.softmax_lse_ptr = lse;
  p.b = p.b_k = 1; p.h = p.h_k = H; p.seqlen_q = Tq; p.seqlen_k = Tk; p.seqlen_q_rounded = rm(Tq, 128); p.seqlen_k_rounded = rm(Tk, 128);
  p.d = p.d_rounded = 128; p.total_q = Tq; p.total_k = Tk;
  p.scale_softmax = (float)std::pow(128.0, -0.5); p.softcap = 0.f;
  p.p_dropout = 1.f; p.p_dropout_in_uint8_t = 255; p.rp_dropout = 1.f;
  p.window_size_left = p.window_size_right = -1;
  p.arch = 90; p.num_sm = num_sm; p.page_size = 1; p.num_splits = 1;
  run_mha_fwd_<90, cutlass::bfloat16_t, 128, false, false, false, false>(p, s);
}
