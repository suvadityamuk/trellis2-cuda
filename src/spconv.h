#pragma once
#include "ops.h"
#include <map>

// Submanifold 3x3x3 neighbor cache for one coordinate set (FlexGEMM SubMConv3dNeighborCache, masked variant)
struct SpConvCache {
  int N = 0;
  Tensor nbr, gray, sorted;                          // [N,27] u32 (0xffffffff = none), [N] mask, [N] gray-code order
  std::map<int, std::pair<Tensor, Tensor>> vk;      // B1 -> (valid kernel list, segment offsets [N/B1 + 1])
  const std::pair<Tensor, Tensor>& valid(int B1);
};
SpConvCache build_neighbors(const Tensor& coords, int W, int H, int D);  // coords int32 [N,4] (b,x,y,z)
struct SpConvCfg { int B1, BK, S; };
SpConvCfg spconv_cfg(int N, int Ci, int Co);  // FlexGEMM autotune choice
// y = subm_conv3d(x, w, b): x fp16 [N,Ci], w fp16 [Co,27,Ci] (= weight.reshape(Co, V, Ci)), b fp16/fp32 [Co]
Tensor subm_conv(const Tensor& x, const Tensor& w, const Tensor* b, SpConvCache& c);
