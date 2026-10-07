#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define C10_CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s: %s\n", #x, cudaGetErrorString(e_)); abort(); } } while (0)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
