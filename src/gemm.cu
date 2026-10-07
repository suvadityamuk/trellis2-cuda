// Linear layers routed exactly like ATen 2.6 addmm/mm (cuBLASLt gemm_and_bias with bias, cublasGemmEx without),
// so heuristics pick the same algorithms and results are bitwise identical to torch.
#include "ops.h"
#include <cublasLt.h>
#include <cublas_v2.h>

#define CB(x) do { auto s_ = (x); REQ(s_ == CUBLAS_STATUS_SUCCESS, "cublas %d", (int)s_); } while (0)

static cublasLtHandle_t lt() { static cublasLtHandle_t h = [] { cublasLtHandle_t h; CB(cublasLtCreate(&h)); return h; }(); return h; }
static cublasHandle_t cb() {
  static cublasHandle_t h = [] {
    cublasHandle_t h; CB(cublasCreate(&h)); CB(cublasSetStream(h, stream()));
    static void* ws; size_t n = 32u << 20; CK(cudaMalloc(&ws, n)); CB(cublasSetWorkspace(h, ws, n));
    return h;
  }();
  return h;
}
static void* lt_ws() { static void* w = [] { void* w; CK(cudaMalloc(&w, 1 << 20)); return w; }(); return w; }
static uint32_t align_of(const void* p) { uint32_t a = 256; while ((uintptr_t)p % a) a /= 2; return a; }
static cudaDataType_t cdt(DT d) { return d == BF16 ? CUDA_R_16BF : d == F16 ? CUDA_R_16F : CUDA_R_32F; }

struct LtKey { int m, n, k, dt, aa, ab, ac, ad; bool operator<(const LtKey& o) const { return memcmp(this, &o, sizeof(*this)) < 0; } };

// y[M,N] = x[M,K] W[N,K]^T + b : cublas column-major gemm(transa=T, transb=N, m=N, n=M, k=K).
void linear(const Tensor& x, const Tensor& W, const Tensor* b, Tensor& y) {
  int K = (int)W.size(1), N = (int)W.size(0), M = (int)(x.numel() / K);
  REQ(x.dt == W.dt && y.dt == W.dt && x.size(-1) == K && y.numel() == (i64)M * N, "linear shapes/dtypes");
  if (M == 0) return;
  float alpha = 1.f, beta = 0.f;
  if (b) {
    static std::map<LtKey, cublasLtMatmulHeuristicResult_t> cache;
    LtKey key{N, M, K, (int)W.dt, (int)align_of(W.p), (int)align_of(x.p), (int)align_of(y.p), (int)align_of(b->p)};
    cudaDataType_t t = cdt(W.dt);
    cublasLtMatmulDesc_t d; CB(cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N; cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_BIAS;
    CB(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
    CB(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
    CB(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof ep));
    CB(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &b->p, sizeof(void*)));
    cublasLtMatrixLayout_t A, B, C;
    CB(cublasLtMatrixLayoutCreate(&A, t, K, N, K)); CB(cublasLtMatrixLayoutCreate(&B, t, K, M, K)); CB(cublasLtMatrixLayoutCreate(&C, t, N, M, N));
    auto it = cache.find(key);
    if (it == cache.end()) {
      cublasLtMatmulPreference_t pr; CB(cublasLtMatmulPreferenceCreate(&pr));
      uint64_t ws = 1 << 20; CB(cublasLtMatmulPreferenceSetAttribute(pr, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
      uint32_t al[4] = {(uint32_t)key.aa, (uint32_t)key.ab, (uint32_t)key.ac, (uint32_t)key.ad};
      CB(cublasLtMatmulPreferenceSetAttribute(pr, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &al[0], 4));
      CB(cublasLtMatmulPreferenceSetAttribute(pr, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &al[1], 4));
      CB(cublasLtMatmulPreferenceSetAttribute(pr, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &al[2], 4));
      CB(cublasLtMatmulPreferenceSetAttribute(pr, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &al[3], 4));
      cublasLtMatmulHeuristicResult_t r{}; int nr = 0;
      CB(cublasLtMatmulAlgoGetHeuristic(lt(), d, A, B, C, C, pr, 1, &r, &nr)); REQ(nr, "no lt algo");
      cublasLtMatmulPreferenceDestroy(pr);
      it = cache.emplace(key, r).first;
    }
    CB(cublasLtMatmul(lt(), d, &alpha, W.p, A, x.p, B, &beta, y.p, C, y.p, C, &it->second.algo, lt_ws(), 1 << 20, stream()));
    cublasLtMatrixLayoutDestroy(A); cublasLtMatrixLayoutDestroy(B); cublasLtMatrixLayoutDestroy(C); cublasLtMatmulDescDestroy(d);
  } else if (W.dt == F32) {
    CB(cublasSgemm(cb(), CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W.ptr<float>(), K, x.ptr<float>(), K, &beta, y.ptr<float>(), N));
  } else {
    CB(cublasGemmEx(cb(), CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W.p, cdt(W.dt), K, x.p, cdt(W.dt), K, &beta, y.p, cdt(W.dt), N,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
}
Tensor linear(const Tensor& x, const Tensor& W, const Tensor* b) {
  auto s = x.sh; s.back() = W.size(0); auto y = empty(s, W.dt); linear(x, W, b, y); return y;
}
