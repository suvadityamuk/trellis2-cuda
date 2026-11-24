// Force-included into CuMesh / cuBVH CUDA sources: route cudaMalloc/cudaFree (their own and thrust's temporaries) through
// the stream-ordered pool (cudaMallocAsync on the legacy default stream those libraries run on). CuMesh allocates and
// frees dozens of scratch buffers per simplify / chart-clustering step; a driver allocation each time dominated them.
// Allocation only: no arithmetic changes, so this is used by the parity build too.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
template <class T> static inline cudaError_t t2_cm_malloc(T** p, size_t n) {
  if (n == 0) { *p = nullptr; return cudaSuccess; }
  return cudaMallocAsync((void**)p, n, (cudaStream_t)0);
}
static inline cudaError_t t2_cm_free(void* p) { return p ? cudaFreeAsync(p, (cudaStream_t)0) : cudaSuccess; }
#define cudaMalloc(p, n) t2_cm_malloc(p, n)
#define cudaFree(p) t2_cm_free(p)
