#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#ifdef __cplusplus
#include <algorithm>
#include <cstring>
#include <iostream>
#define LOG(x) if (0) std::cerr
#endif
#define NVDR_CHECK(COND, ERR) do { if (!(COND)) { fprintf(stderr, "nvdiffrast: %s\n", ERR); abort(); } } while (0)
#define NVDR_CHECK_CUDA_ERROR(CALL) do { cudaError_t e_ = CALL; if (e_) { fprintf(stderr, "nvdiffrast: %s [%s]\n", cudaGetErrorString(e_), #CALL); abort(); } } while (0)
