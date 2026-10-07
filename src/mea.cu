// fp32 SDPA through ATen's vendored cutlass mem-efficient kernel (torch 2.6 picks AttentionKernel<float, Sm80, aligned,
// 64, 64, 64> first for fp32 on sm80+), launched with exactly the params _efficient_attention_forward fills.
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>
#include "models.h"
using namespace PyTorchMemEffAttention;
using AK = AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>;
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm) k_fmha_f32(typename AK::Params p) {
  if (!p.advance_to_block()) return;
  AK::attention_kernel(p);
}

// q,k,v: BMHK views with element strides; o: contiguous [B, M, H, D]
void mem_eff_attn(BMHK q, BMHK k, BMHK v, float* o, int B, int M, int N, int H, int D, float scale) {
  REQ(D <= AK::kMaxK, "mea head dim");
  typename AK::Params p;
  p.query_ptr = q.p; p.key_ptr = k.p; p.value_ptr = v.p; p.output_ptr = o; p.output_accum_ptr = nullptr; p.logsumexp_ptr = nullptr;
  p.num_heads = H; p.head_dim = D; p.head_dim_value = D; p.num_queries = M; p.num_keys = N; p.num_batches = B;
  p.custom_mask_type = 0; p.seqlen_k_ptr = nullptr; p.scale = scale;
  p.q_strideB = q.sB; p.k_strideB = k.sB; p.v_strideB = v.sB; p.q_strideM = (int)q.sM; p.k_strideM = (int)k.sM; p.v_strideM = (int)v.sM;
  p.q_strideH = (int)q.sH; p.k_strideH = (int)k.sH; p.v_strideH = (int)v.sH; p.o_strideM = H * D; p.use_dropout = false;
  size_t smem = sizeof(typename AK::SharedStorage);
  if (smem > 0xc000) CK(cudaFuncSetAttribute(k_fmha_f32, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  AK::check_supported(p);
  k_fmha_f32<<<p.getBlocksGrid(), p.getThreadsGrid(), smem, stream()>>>(p);
  CK(cudaGetLastError());
}
