#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s @ %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); abort(); } } while (0)
#define REQ(c, ...) do { if (!(c)) { fprintf(stderr, "check failed %s @ %s:%d: ", #c, __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); abort(); } } while (0)

using bf16 = __nv_bfloat16;
using f16 = __half;
using i64 = int64_t;
enum DT : uint8_t { F32, F16, BF16, I32, I64, U8, F64, I8 };
size_t dsize(DT d);
const char* dname(DT d);

cudaStream_t stream();

// Device tensor (row-major, contiguous). Storage is refcounted and stream-ordered (cudaMallocAsync).
struct Tensor {
  void* p = nullptr;
  DT dt = F32;
  std::vector<i64> sh;
  std::shared_ptr<void> own;
  i64 numel() const { i64 n = 1; for (auto s : sh) n *= s; return n; }
  size_t bytes() const { return numel() * dsize(dt); }
  i64 size(int i) const { return sh[i < 0 ? i + (int)sh.size() : i]; }
  int dim() const { return (int)sh.size(); }
  bool defined() const { return p != nullptr || numel() == 0 && !sh.empty(); }
  template <class T> T* ptr() const { return (T*)p; }
  Tensor view(std::vector<i64> s) const;
  Tensor slice0(i64 a, i64 b) const;  // rows [a,b) along dim 0 (shares storage)
};

Tensor empty(std::vector<i64> sh, DT dt);
Tensor zeros(std::vector<i64> sh, DT dt);
Tensor from_host(const void* src, std::vector<i64> sh, DT dt);
std::vector<char> to_host(const Tensor& t);
template <class T> std::vector<T> to_host_vec(const Tensor& t) { auto b = to_host(t); std::vector<T> v(b.size() / sizeof(T)); memcpy(v.data(), b.data(), b.size()); return v; }
Tensor clone(const Tensor& t);
Tensor cast(const Tensor& t, DT dt);  // ATen-exact .to(dtype)
void dsync();

using TensorMap = std::map<std::string, Tensor>;
// safetensors: load into 256B-aligned device allocations (matches torch param alignment for cuBLAS heuristics)
TensorMap load_safetensors(const std::string& path, const std::string& prefix = "");
TensorMap load_safetensors_host(const std::string& path);  // tensors with p = host memory
void save_safetensors(const std::string& path, const TensorMap& m);

struct Timer { cudaEvent_t a, b; Timer(); void start(); float ms(); };
double now_ms();
std::string env(const char* k, const char* d = "");
inline bool fast_mode() { static bool f = env("T2_FAST") == "1"; return f; }  // tolerance-bounded fast build paths

__host__ __device__ inline int cdiv(i64 a, i64 b) { return (int)((a + b - 1) / b); }
