// Point-to-triangle-mesh unsigned distance on the GPU: linear BVH (Karras radix tree over 63-bit Morton codes of
// triangle centroids) built per call, then a best-first stack traversal per query point. Exact nearest triangle (ties:
// lowest face index), closest-point barycentrics as in Ericson, Real-Time Collision Detection, 5.1.5.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <torch/all.h>

#include <climits>

namespace {

struct Tri { float3 a, b, c; };

__device__ __forceinline__ float3 sub(float3 x, float3 y) { return make_float3(x.x - y.x, x.y - y.y, x.z - y.z); }
__device__ __forceinline__ float dot(float3 x, float3 y) { return x.x * y.x + x.y * y.y + x.z * y.z; }
__device__ __forceinline__ float3 mad(float3 x, float3 d, float t) { return make_float3(x.x + d.x * t, x.y + d.y * t, x.z + d.z * t); }

// Closest point on segment x->x+d to p as parameter t in [0, 1].
__device__ __forceinline__ float seg_t(float3 x, float3 d, float3 p) {
  float dd = dot(d, d);
  return dd > 0.f ? fminf(fmaxf(dot(sub(p, x), d) / dd, 0.f), 1.f) : 0.f;
}

__device__ __forceinline__ float3 cross(float3 x, float3 y) {
  return make_float3(x.y * y.z - x.z * y.y, x.z * y.x - x.x * y.z, x.x * y.y - x.y * y.x);
}
__device__ __forceinline__ float safe_div(float x, float y) { return y > 0.f ? x / y : 0.f; }

// Squared distance from p to triangle t; bary receives the closest point's barycentric coordinates (u, v, w).
__device__ float closest(const Tri& t, float3 p, float3& bary) {
  float3 ab = sub(t.b, t.a), ac = sub(t.c, t.a), ap = sub(p, t.a);
  float u, v, w;
  auto nearest_edge = [&] {
    float3 bc = sub(t.c, t.b);
    float s0 = seg_t(t.a, ab, p), s1 = seg_t(t.b, bc, p), s2 = seg_t(t.a, ac, p);
    float3 q0 = sub(p, mad(t.a, ab, s0)), q1 = sub(p, mad(t.b, bc, s1)), q2 = sub(p, mad(t.a, ac, s2));
    float e0 = dot(q0, q0), e1 = dot(q1, q1), e2 = dot(q2, q2);
    if (e0 <= e1 && e0 <= e2) { u = 1.f - s0; v = s0; w = 0.f; }
    else if (e1 <= e2) { u = 0.f; v = 1.f - s1; w = s1; }
    else { u = 1.f - s2; v = 0.f; w = s2; }
  };
  // Ericson's region tests assume nonzero area; on (near-)degenerate triangles va, vb, vc are rounding noise.
  float3 n = cross(ab, ac);
  if (!(dot(n, n) > 1e-10f * dot(ab, ab) * dot(ac, ac))) nearest_edge();
  else {
    float d1 = dot(ab, ap), d2 = dot(ac, ap);
    float3 bp = sub(p, t.b); float d3 = dot(ab, bp), d4 = dot(ac, bp);
    float3 cp = sub(p, t.c); float d5 = dot(ab, cp), d6 = dot(ac, cp);
    float va = d3 * d6 - d5 * d4, vb = d5 * d2 - d1 * d6, vc = d1 * d4 - d3 * d2;
    if (d1 <= 0.f && d2 <= 0.f) { u = 1.f; v = 0.f; w = 0.f; }
    else if (d3 >= 0.f && d4 <= d3) { u = 0.f; v = 1.f; w = 0.f; }
    else if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) { v = safe_div(d1, d1 - d3); u = 1.f - v; w = 0.f; }
    else if (d6 >= 0.f && d5 <= d6) { u = 0.f; v = 0.f; w = 1.f; }
    else if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) { w = safe_div(d2, d2 - d6); u = 1.f - w; v = 0.f; }
    else if (va <= 0.f && d4 - d3 >= 0.f && d5 - d6 >= 0.f) { w = safe_div(d4 - d3, (d4 - d3) + (d5 - d6)); v = 1.f - w; u = 0.f; }
    else {
      float s = safe_div(1.f, va + vb + vc); v = vb * s; w = vc * s; u = 1.f - v - w;
      if (!(s > 0.f && u >= -1e-4f && v >= -1e-4f && w >= -1e-4f)) nearest_edge();
    }
  }
  bary = make_float3(u, v, w);
  float3 q = make_float3(u * t.a.x + v * t.b.x + w * t.c.x, u * t.a.y + v * t.b.y + w * t.c.y, u * t.a.z + v * t.b.z + w * t.c.z);
  float3 e = sub(p, q);
  return dot(e, e);
}

template <class I>
__global__ void k_tri(const float3* v, const I* f, int64_t n, Tri* t, float3* cen) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; if (i >= n) return;
  float3 a = v[f[3 * i]], b = v[f[3 * i + 1]], c = v[f[3 * i + 2]];
  t[i] = {a, b, c};
  cen[i] = make_float3((a.x + b.x + c.x) / 3, (a.y + b.y + c.y) / 3, (a.z + b.z + c.z) / 3);
}
__device__ __forceinline__ uint64_t spread21(uint64_t x) {
  x &= 0x1fffff; x = (x | x << 32) & 0x1f00000000ffffull; x = (x | x << 16) & 0x1f0000ff0000ffull;
  x = (x | x << 8) & 0x100f00f00f00f00full; x = (x | x << 4) & 0x10c30c30c30c30c3ull; return (x | x << 2) & 0x1249249249249249ull;
}
__global__ void k_morton(const float3* c, int64_t n, const float3* mm, uint64_t* key, int* idx) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; if (i >= n) return;
  float3 lo = mm[0], hi = mm[1];
  auto q = [](float x, float l, float h) { float e = h - l; return (uint64_t)(e > 0.f ? fminf(fmaxf((x - l) / e * 2097151.f, 0.f), 2097151.f) : 0.f); };
  key[i] = spread21(q(c[i].x, lo.x, hi.x)) << 2 | spread21(q(c[i].y, lo.y, hi.y)) << 1 | spread21(q(c[i].z, lo.z, hi.z));
  idx[i] = (int)i;
}
__global__ void k_gather(const Tri* t, const int* idx, int64_t n, Tri* o) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; if (i < n) o[i] = t[idx[i]];
}
// Common-prefix length of sorted keys i and j; equal keys fall back to their indices so the tree stays well formed.
__device__ __forceinline__ int delta(const uint64_t* k, int64_t n, int64_t i, int64_t j) {
  if (j < 0 || j >= n) return -1;
  return k[i] == k[j] ? 64 + __clzll((long long)(i ^ j)) : __clzll((long long)(k[i] ^ k[j]));
}
// Internal node i gets children (c >= 0: internal node c, c < 0: leaf ~c). par spans 2n-1 slots: internal i, leaf n-1+j.
__global__ void k_tree(const uint64_t* k, int64_t n, int2* ch, int* par) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; if (i >= n - 1) return;
  int d = delta(k, n, i, i + 1) > delta(k, n, i, i - 1) ? 1 : -1, dmin = delta(k, n, i, i - d);
  int64_t lmax = 2; while (delta(k, n, i, i + lmax * d) > dmin) lmax *= 2;
  int64_t l = 0; for (int64_t t = lmax / 2; t >= 1; t /= 2) if (delta(k, n, i, i + (l + t) * d) > dmin) l += t;
  int64_t j = i + l * d; int dn = delta(k, n, i, j); int64_t s = 0;
  for (int64_t t = (l + 1) / 2;; t = (t + 1) / 2) { if (delta(k, n, i, i + (s + t) * d) > dn) s += t; if (t == 1) break; }
  int64_t g = i + s * d + (d < 0 ? -1 : 0), lo = i < j ? i : j, hi = i < j ? j : i;
  int a = lo == g ? ~(int)g : (int)g, b = hi == g + 1 ? ~(int)(g + 1) : (int)(g + 1);
  ch[i] = make_int2(a, b);
  par[a < 0 ? n - 1 + ~a : a] = (int)i; par[b < 0 ? n - 1 + ~b : b] = (int)i;
}
// Leaf boxes, then bottom-up refit: the second child to arrive at a node computes its box.
__global__ void k_box(const Tri* t, const int2* ch, const int* par, int64_t n, int* flag, float* box) {
  int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; if (j >= n) return;
  float* b = box + (n - 1 + j) * 6; const Tri& q = t[j];
  b[0] = fminf(fminf(q.a.x, q.b.x), q.c.x); b[1] = fminf(fminf(q.a.y, q.b.y), q.c.y); b[2] = fminf(fminf(q.a.z, q.b.z), q.c.z);
  b[3] = fmaxf(fmaxf(q.a.x, q.b.x), q.c.x); b[4] = fmaxf(fmaxf(q.a.y, q.b.y), q.c.y); b[5] = fmaxf(fmaxf(q.a.z, q.b.z), q.c.z);
  if (n == 1) return;
  int p = par[n - 1 + j];
  while (true) {
    __threadfence();
    if (atomicAdd(&flag[p], 1) == 0) return;
    int2 c = ch[p]; const float* x = box + (c.x < 0 ? n - 1 + ~c.x : c.x) * 6; const float* y = box + (c.y < 0 ? n - 1 + ~c.y : c.y) * 6;
    float* o = box + (int64_t)p * 6;
    for (int e = 0; e < 3; e++) { o[e] = fminf(__ldcg(x + e), __ldcg(y + e)); o[3 + e] = fmaxf(__ldcg(x + 3 + e), __ldcg(y + 3 + e)); }
    if (p == 0) return;
    p = par[p];
  }
}
__device__ __forceinline__ float box_d2(const float* b, float3 p) {
  float dx = fmaxf(fmaxf(__ldg(b) - p.x, p.x - __ldg(b + 3)), 0.f), dy = fmaxf(fmaxf(__ldg(b + 1) - p.y, p.y - __ldg(b + 4)), 0.f),
        dz = fmaxf(fmaxf(__ldg(b + 2) - p.z, p.z - __ldg(b + 5)), 0.f);
  return dx * dx + dy * dy + dz * dz;
}
__global__ void k_query(const float3* pts, int64_t m, const Tri* t, const int* idx, const int2* ch, const float* box, int64_t n,
                        float* dist, int64_t* fid, float3* uvw) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; if (i >= m) return;
  float3 p = pts[i], bb, bary = make_float3(1.f, 0.f, 0.f);
  float best = INFINITY; int bl = -1, bf = INT_MAX;
  int st[128], sp = 0; st[sp++] = n == 1 ? ~0 : 0;
  while (sp) {
    int c = st[--sp];
    if (c < 0) {
      float d = closest(t[~c], p, bb); int f = idx[~c];
      if (d < best || (d == best && f < bf)) { best = d; bl = ~c; bf = f; bary = bb; }
      continue;
    }
    int2 k = __ldg(&ch[c]);
    float dl = box_d2(box + (k.x < 0 ? n - 1 + ~k.x : k.x) * 6, p), dr = box_d2(box + (k.y < 0 ? n - 1 + ~k.y : k.y) * 6, p);
    if (dl > dr) { int s = k.x; k.x = k.y; k.y = s; float f = dl; dl = dr; dr = f; }
    if (dr <= best) st[sp++] = k.y;
    if (dl <= best) st[sp++] = k.x;
  }
  dist[i] = sqrtf(best); fid[i] = bl < 0 ? -1 : bf; uvw[i] = bary;
}

int blocks(int64_t n, int b) { return (int)((n + b - 1) / b); }

}  // namespace

std::vector<torch::Tensor> mesh_udf(torch::Tensor const& vertices, torch::Tensor const& faces, torch::Tensor const& points) {
  TORCH_CHECK(vertices.is_cuda() && faces.is_cuda() && points.is_cuda(), "mesh_udf: all inputs must be CUDA tensors");
  TORCH_CHECK(vertices.device() == faces.device() && vertices.device() == points.device(), "mesh_udf: inputs must be on one device");
  TORCH_CHECK(vertices.dim() == 2 && vertices.size(1) == 3 && vertices.scalar_type() == at::kFloat, "mesh_udf: vertices must be float32 [V, 3]");
  TORCH_CHECK(points.dim() == 2 && points.size(1) == 3 && points.scalar_type() == at::kFloat, "mesh_udf: points must be float32 [M, 3]");
  TORCH_CHECK(faces.dim() == 2 && faces.size(1) == 3 && (faces.scalar_type() == at::kInt || faces.scalar_type() == at::kLong),
              "mesh_udf: faces must be int32 or int64 [F, 3]");
  TORCH_CHECK(faces.size(0) > 0 && faces.size(0) < INT_MAX, "mesh_udf: need 1 <= F < 2^31 faces");
  const at::cuda::OptionalCUDAGuard guard(device_of(vertices));
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  auto V = vertices.contiguous(), F = faces.contiguous(), P = points.contiguous();
  int64_t n = F.size(0), m = P.size(0);
  auto opt = V.options();
  auto u8 = opt.dtype(at::kByte), i32 = opt.dtype(at::kInt), i64 = opt.dtype(at::kLong);
  auto dist = torch::empty({m}, opt), fid = torch::empty({m}, i64), uvw = torch::empty({m, 3}, opt);
  if (m == 0) return {dist, fid, uvw};

  auto t0 = torch::empty({n * (int64_t)sizeof(Tri)}, u8), cen = torch::empty({n, 3}, opt), mm = torch::empty({2, 3}, opt);
  if (F.scalar_type() == at::kInt)
    k_tri<<<blocks(n, 256), 256, 0, s>>>((const float3*)V.data_ptr<float>(), F.data_ptr<int>(), n, (Tri*)t0.data_ptr(), (float3*)cen.data_ptr<float>());
  else
    k_tri<<<blocks(n, 256), 256, 0, s>>>((const float3*)V.data_ptr<float>(), F.data_ptr<int64_t>(), n, (Tri*)t0.data_ptr(), (float3*)cen.data_ptr<float>());
  auto lo = torch::amin(cen, 0), hi = torch::amax(cen, 0);
  mm.select(0, 0).copy_(lo); mm.select(0, 1).copy_(hi);

  auto key = torch::empty({n}, i64), ks = torch::empty({n}, i64), idx = torch::empty({n}, i32), is = torch::empty({n}, i32);
  k_morton<<<blocks(n, 256), 256, 0, s>>>((const float3*)cen.data_ptr<float>(), n, (const float3*)mm.data_ptr<float>(),
                                         (uint64_t*)key.data_ptr<int64_t>(), idx.data_ptr<int>());
  size_t tb = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, tb, (uint64_t*)key.data_ptr<int64_t>(), (uint64_t*)ks.data_ptr<int64_t>(), idx.data_ptr<int>(),
                                  is.data_ptr<int>(), (int)n, 0, 63, s);
  auto tmp = torch::empty({(int64_t)tb}, u8);
  cub::DeviceRadixSort::SortPairs(tmp.data_ptr(), tb, (uint64_t*)key.data_ptr<int64_t>(), (uint64_t*)ks.data_ptr<int64_t>(), idx.data_ptr<int>(),
                                  is.data_ptr<int>(), (int)n, 0, 63, s);
  auto tri = torch::empty({n * (int64_t)sizeof(Tri)}, u8);
  k_gather<<<blocks(n, 256), 256, 0, s>>>((const Tri*)t0.data_ptr(), is.data_ptr<int>(), n, (Tri*)tri.data_ptr());
  auto ch = torch::empty({std::max<int64_t>(n - 1, 1), 2}, i32), box = torch::empty({2 * n - 1, 6}, opt);
  auto par = torch::empty({2 * n}, i32), flag = torch::zeros({std::max<int64_t>(n - 1, 1)}, i32);
  if (n > 1) k_tree<<<blocks(n - 1, 256), 256, 0, s>>>((const uint64_t*)ks.data_ptr<int64_t>(), n, (int2*)ch.data_ptr<int>(), par.data_ptr<int>());
  k_box<<<blocks(n, 256), 256, 0, s>>>((const Tri*)tri.data_ptr(), (const int2*)ch.data_ptr<int>(), par.data_ptr<int>(), n, flag.data_ptr<int>(),
                                      box.data_ptr<float>());
  k_query<<<blocks(m, 128), 128, 0, s>>>((const float3*)P.data_ptr<float>(), m, (const Tri*)tri.data_ptr(), is.data_ptr<int>(),
                                        (const int2*)ch.data_ptr<int>(), box.data_ptr<float>(), n, dist.data_ptr<float>(),
                                        fid.data_ptr<int64_t>(), (float3*)uvw.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dist, fid, uvw};
}
