#pragma once
#include "core.h"

// ---- GEMM (gemm.cu): torch F.linear semantics, y = x W^T + b
void linear(const Tensor& x, const Tensor& W, const Tensor* b, Tensor& y);
Tensor linear(const Tensor& x, const Tensor& W, const Tensor* b);

// ---- ATen-exact elementwise / norm kernels (ops.cu). All match torch 2.6 CUDA bit-for-bit.
Tensor layer_norm(const Tensor& x, const Tensor* w, const Tensor* b, float eps);  // any float dtype in/out (LayerNorm32 semantics)
Tensor layer_norm_f32(const Tensor& x, const Tensor* w, const Tensor* b, float eps);  // x fp32 -> fp32 (F.layer_norm)
enum Act { SILU, GELU_TANH, GELU_ERF };
Tensor act(const Tensor& x, Act a);
void act_(Tensor& x, Act a);
// out = x*(1+scale[c]) + shift[c]   (three rounded ops, broadcast over rows; scale/shift are [C] views)
Tensor modulate(const Tensor& x, const Tensor& shift, const Tensor& scale);
// x += h*gate[c] (two rounded ops) ; gate may be null => x += h
void gated_residual_(Tensor& x, const Tensor& h, const Tensor* gate);
void add_(Tensor& x, const Tensor& y);
void bias_add_ncdhw_(Tensor& y, const Tensor& b);  // y[:, c, ...] += b[c]  (b cast to y dtype first, like ATen add_)
// F.normalize(x.float(),-1)*gamma*scale -> dtype(x). x: [rows, H, D] strided view (row stride, head stride)
void mh_rmsnorm(const void* x, i64 row_stride, int H, int D, i64 rows, const Tensor& gamma, float scale, void* y, DT dt);
// RoPE: y = complex(x.float()) * phases[row, d/2] -> dtype. x,y contiguous [rows, H, D]; phases [rows, D/2] complex64
void rope_(void* xy, i64 rows, int H, int D, const Tensor& phases, DT dt);
// phases from int coords [N,4] (b,x,y,z): outer(coord, freqs) -> polar, padded to D/2
// DINOv3 rotate_half RoPE on [T, H*D] rows >= P, output transposed to contiguous [H, T, D]; cs/sn [T-P, D]
Tensor rope_half_bhtd(const Tensor& x, int H, int P, const Tensor& cs, const Tensor& sn);
Tensor rope_phases(const Tensor& coords4, const Tensor& freqs, int D);
Tensor timestep_freq_embed(float t, const Tensor& freqs);  // [1, 2*half] = cat(cos(t*f), sin(t*f))

// ---- reductions (ATen Reduce.cuh emulation)
Tensor row_reduce(const Tensor& x, int L, int op);  // op 0 = sum-of-squares sqrt (norm2), 1 = mean ; x fp32 [R, L] -> [R]
float std_all(const Tensor& x);                       // torch.std over all elems of a B=1 tensor (Welford, unbiased)
float segment_mean(const Tensor& x);                  // torch.segment_reduce(mean) single segment, fp32 [N] -> scalar

// ---- fp32 scalar ops (torch Tensor <op> python-float semantics)
Tensor axpby(const Tensor& x, float a, const Tensor& y, float b, int mode);  // mode0: x*a + y*b (rounded each), 1: x*a - y*b
void scale_(Tensor& x, float a, float b = 0.f);  // x = x*a + b (each op rounded)
Tensor mul_t(const Tensor& x, const Tensor& y);
