// Minimal torch::Tensor surface used by CuMesh / cubvh sources, backed by the t2 device Tensor (no libtorch).
#pragma once
#include "core.h"
#include "c10/util/Exception.h"
#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace c10 { enum class ScalarType { Float, Int, Long, UInt8, Int8, UInt32, UInt64, Bool }; }
namespace torch {
using ScalarType = c10::ScalarType;
constexpr ScalarType kFloat32 = ScalarType::Float, kFloat = ScalarType::Float, kInt32 = ScalarType::Int, kInt = ScalarType::Int,
                     kInt64 = ScalarType::Long, kLong = ScalarType::Long, kUInt8 = ScalarType::UInt8, kInt8 = ScalarType::Int8,
                     kUInt32 = ScalarType::UInt32, kUInt64 = ScalarType::UInt64, kBool = ScalarType::Bool;
enum DeviceType { kCUDA, kCPU };
struct TensorOptions {
  ScalarType st = ScalarType::Float;
  TensorOptions device(DeviceType) const { return *this; }
  TensorOptions dtype(ScalarType s) const { auto o = *this; o.st = s; return o; }
};
inline TensorOptions dtype(ScalarType s) { return TensorOptions{s}; }
inline DT to_dt(ScalarType s) {
  switch (s) {
    case ScalarType::Float: return F32;
    case ScalarType::Int: case ScalarType::UInt32: return I32;
    case ScalarType::Long: case ScalarType::UInt64: return I64;
    case ScalarType::Int8: return I8;
    default: return U8;
  }
}
struct Tensor {
  ::Tensor t;
  ScalarType st = ScalarType::Float;
  Tensor() = default;
  Tensor(const ::Tensor& x, ScalarType s) : t(x), st(s) {}
  int64_t size(int i) const { return t.size(i); }
  int64_t numel() const { return t.numel(); }
  int dim() const { return t.dim(); }
  bool is_empty() const { return t.numel() == 0; }
  bool is_cuda() const { return true; }
  bool is_contiguous() const { return true; }
  ScalarType scalar_type() const { return st; }
  ScalarType dtype() const { return st; }
  TensorOptions options() const { return TensorOptions{st}; }
  DeviceType device() const { return kCUDA; }
  void* data_ptr() const { return t.p; }
  template <class T> T* data_ptr() const { return (T*)t.p; }
};
inline Tensor empty(std::vector<int64_t> s, TensorOptions o) { Tensor r(::empty(s, to_dt(o.st)), o.st); dsync(); return r; }
inline Tensor zeros(std::vector<int64_t> s, TensorOptions o) { Tensor r(::zeros(s, to_dt(o.st)), o.st); dsync(); return r; }
template <class V> inline Tensor full(std::vector<int64_t> s, V v, TensorOptions o) {
  int64_t n = 1; for (auto x : s) n *= x;
  if (o.st == ScalarType::Float) { std::vector<float> h(n, (float)v); return Tensor(from_host(h.data(), s, F32), o.st); }
  std::vector<uint32_t> h(n, (uint32_t)v); return Tensor(from_host(h.data(), s, I32), o.st);
}
inline Tensor wrap(const ::Tensor& x, ScalarType s) { return Tensor(x, s); }
}  // namespace torch
namespace at { using Tensor = torch::Tensor; namespace cuda {
struct CUDAStream { cudaStream_t stream() const { return 0; } operator cudaStream_t() const { return 0; } };
inline CUDAStream getCurrentCUDAStream() { return {}; }
}  }
#define TORCH_EXTENSION_NAME t2_cumesh
