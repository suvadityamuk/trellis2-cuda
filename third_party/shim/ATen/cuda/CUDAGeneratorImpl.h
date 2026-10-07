#pragma once
// Layout-identical stand-in for at::PhiloxCudaState (torch 2.6) so flash-attn compiles without libtorch.
#include <cstdint>
namespace at {
struct PhiloxCudaState {
  PhiloxCudaState() = default;
  PhiloxCudaState(uint64_t seed, uint64_t offset) { seed_.val = seed; offset_.val = offset; }
  union Payload { uint64_t val; int64_t* ptr; };
  Payload seed_{};
  Payload offset_{};
  uint32_t offset_intragraph_ = 0;
  bool captured_ = false;
};
}  // namespace at
