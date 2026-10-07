// Wavefront TELEA inpainting on the GPU. OpenCV's INPAINT_TELEA marches pixels one at a time in fast-marching order;
// here every unknown pixel gets an integer 4-connected distance level L from the known region, and all pixels of level
// L are filled in parallel from pixels of lower levels with TELEA's weights (direction x distance x level).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

namespace {

constexpr int MAXC = 4;

__global__ void k_init(const uint8_t* mask, int64_t P, int* lv) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; if (i < P) lv[i] = mask[i] ? -1 : 0;
}
__global__ void k_mark(int* lv, int H, int W, int L, int* cnt) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; if (i >= (int64_t)H * W || lv[i] != -1) return;
  int y = (int)(i / W), x = (int)(i % W), p = L - 1;
  if ((x > 0 && lv[i - 1] == p) || (x < W - 1 && lv[i + 1] == p) || (y > 0 && lv[i - W] == p) || (y < H - 1 && lv[i + W] == p)) {
    lv[i] = L; atomicAdd(cnt, 1);
  }
}
// img: planar [C, H, W] uint8, filled in place.
__global__ void k_fill(const int* lv, int H, int W, int L, int R, int C, uint8_t* img) {
  int64_t P = (int64_t)H * W, i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; if (i >= P || lv[i] != L) return;
  int y = (int)(i / W), x = (int)(i % W);
  auto kn = [&](int r, int c) { if (r < 0 || c < 0 || r >= H || c >= W) return false; int v = lv[(int64_t)r * W + c]; return v >= 0 && v < L; };
  auto tv = [&](int r, int c) { return (float)lv[(int64_t)r * W + c]; };
  auto I = [&](int ch, int r, int c) { return (float)img[ch * P + (int64_t)r * W + c]; };
  float gx = kn(y, x + 1) ? (kn(y, x - 1) ? (tv(y, x + 1) - tv(y, x - 1)) * 0.5f : tv(y, x + 1) - L) : (kn(y, x - 1) ? L - tv(y, x - 1) : 0.f);
  float gy = kn(y + 1, x) ? (kn(y - 1, x) ? (tv(y + 1, x) - tv(y - 1, x)) * 0.5f : tv(y + 1, x) - L) : (kn(y - 1, x) ? L - tv(y - 1, x) : 0.f);
  float Ia[MAXC] = {}, Jx[MAXC] = {}, Jy[MAXC] = {}, s = 1e-20f;
  for (int k = y - R; k <= y + R; k++)
    for (int l = x - R; l <= x + R; l++) {
      if ((k - y) * (k - y) + (l - x) * (l - x) > R * R || !kn(k, l)) continue;
      float ry = (float)(y - k), rx = (float)(x - l), vl = rx * rx + ry * ry;
      float dir = rx * gx + ry * gy; if (fabsf(dir) <= 0.01f) dir = 1e-6f;
      float w = fabsf(dir / (vl * sqrtf(vl) * (1 + fabsf(tv(k, l) - L))));
      bool r1 = kn(k, l + 1), l1 = kn(k, l - 1), d1 = kn(k + 1, l), u1 = kn(k - 1, l);
      for (int c = 0; c < C; c++) {
        float v = I(c, k, l);
        float ix = r1 ? (l1 ? (I(c, k, l + 1) - I(c, k, l - 1)) * 2.f : I(c, k, l + 1) - v) : (l1 ? v - I(c, k, l - 1) : 0.f);
        float iy = d1 ? (u1 ? (I(c, k + 1, l) - I(c, k - 1, l)) * 2.f : I(c, k + 1, l) - v) : (u1 ? v - I(c, k - 1, l) : 0.f);
        Ia[c] += w * v; Jx[c] -= w * ix * rx; Jy[c] -= w * iy * ry;
      }
      s += w;
    }
  for (int c = 0; c < C; c++) {
    float sat = Ia[c] / s + (Jx[c] + Jy[c]) / (sqrtf(Jx[c] * Jx[c] + Jy[c] * Jy[c]) + 1e-20f);
    img[c * P + i] = (uint8_t)fminf(fmaxf(rintf(sat + 0.5f), 0.f), 255.f);
  }
}

int blocks(int64_t n, int b) { return (int)((n + b - 1) / b); }

}  // namespace

torch::Tensor telea_inpaint(torch::Tensor const& image, torch::Tensor const& mask, int64_t radius) {
  TORCH_CHECK(image.is_cuda() && mask.is_cuda() && image.device() == mask.device(), "telea_inpaint: inputs must be CUDA tensors on one device");
  TORCH_CHECK(image.scalar_type() == at::kByte, "telea_inpaint: image must be uint8");
  TORCH_CHECK(image.dim() == 2 || (image.dim() == 3 && image.size(2) >= 1 && image.size(2) <= MAXC), "telea_inpaint: image must be [H, W] or [H, W, C] with C <= 4");
  TORCH_CHECK(mask.dim() == 2 && mask.size(0) == image.size(0) && mask.size(1) == image.size(1), "telea_inpaint: mask must be [H, W]");
  TORCH_CHECK(radius >= 1 && radius <= 64, "telea_inpaint: radius must be in [1, 64]");
  const at::cuda::OptionalCUDAGuard guard(device_of(image));
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  int H = (int)image.size(0), W = (int)image.size(1), C = image.dim() == 3 ? (int)image.size(2) : 1;
  int64_t P = (int64_t)H * W;
  auto img = (image.dim() == 3 ? image.permute({2, 0, 1}) : image.unsqueeze(0)).contiguous().clone();
  auto m = (mask.scalar_type() == at::kBool ? mask : mask != 0).to(at::kByte).contiguous();
  auto opt = image.options().dtype(at::kInt);
  auto lv = torch::empty({P}, opt), cnt = torch::zeros({1}, opt);
  if (P > 0) {
    k_init<<<blocks(P, 256), 256, 0, s>>>(m.data_ptr<uint8_t>(), P, lv.data_ptr<int>());
    for (int L = 1;; L++) {
      cnt.zero_();
      k_mark<<<blocks(P, 256), 256, 0, s>>>(lv.data_ptr<int>(), H, W, L, cnt.data_ptr<int>());
      k_fill<<<blocks(P, 256), 256, 0, s>>>(lv.data_ptr<int>(), H, W, L, (int)radius, C, img.data_ptr<uint8_t>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      if (cnt.item<int>() == 0) break;
    }
  }
  return image.dim() == 3 ? img.permute({1, 2, 0}).contiguous() : img.squeeze(0);
}
