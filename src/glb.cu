// o_voxel.postprocess.to_glb geometry (remesh=True branch): CuMesh fill_holes -> cuBVH -> cumesh.remeshing.remesh_narrow_band_dc
// -> CuMesh.simplify loop -> CuMesh.uv_unwrap (fast clustering + xatlas) -> vertex normals.
#include "models.h"
#include "cumesh.h"
#include "hash/api.h"
#include "remesh/api.h"
#include "xatlas.h"
#include <gpu/bvh.cuh>
#include <cub/cub.cuh>
#include <climits>
#include <cstring>
#include <fstream>
#include <functional>
#include <thread>
#include <unordered_map>

struct NvdrCtx;
std::shared_ptr<NvdrCtx> nvdr_ctx();
void nvdr_rasterize(NvdrCtx& c, const float* pos, int nv, const int* tri, int nt, int h, int w, float* out, float* out_db, cudaStream_t s);
void nvdr_interpolate(const float* attr, int nv, int na, const float* rast, int h, int w, const int* tri, int nt, float* out, cudaStream_t s);
void inpaint_telea(uint8_t* img, int H, int W, bool rgb, const uint8_t* mask, double radius);
namespace flex_gemm::grid_sample {
std::tuple<torch::Tensor, torch::Tensor> hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight(
  torch::Tensor& keys, torch::Tensor& vals, const torch::Tensor& coords, const torch::Tensor& grid, const int W, const int H, const int D);
}

namespace cb = cumesh::cubvh;
using TT = torch::Tensor;
static TT tu32(const Tensor& t) { return torch::wrap(t, torch::kUInt32); }
static TT ti32(const Tensor& t) { return torch::wrap(t, torch::kInt32); }
static TT tf32(const Tensor& t) { return torch::wrap(t, torch::kFloat32); }
static void fence() { CK(cudaDeviceSynchronize()); }  // CuMesh/cuBVH run on the legacy default stream
static void prof(const char* n) {
  static bool on = env("T2_PROF") == "1"; static double t = now_ms();
  if (on) { fence(); double x = now_ms(); printf("    %-24s %8.1f ms\n", n, x - t); t = x; }
}

// ---- GPU LBVH (fast build): Karras radix tree over 63-bit Morton codes; leaf tests use cuBVH's Triangle math, so
// distances equal cuBVH's (min over triangles); only exact-tie face choices may differ.
struct LBVH { Tensor tri, ch, box; i64 n = 0; };  // tri: cb::Triangle [n] (Morton order); ch: int2 [n-1]; box: float [2n-1, 6]
__global__ void k_lb_tri(const float3* v, const int3* f, i64 n, cb::Triangle* t, float3* cen) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= n) return;
  int3 q = f[i]; float3 a = v[q.x], b = v[q.y], c = v[q.z];
  t[i] = {Eigen::Vector3f(a.x, a.y, a.z), Eigen::Vector3f(b.x, b.y, b.z), Eigen::Vector3f(c.x, c.y, c.z), (int64_t)i};
  cen[i] = make_float3((a.x + b.x + c.x) / 3, (a.y + b.y + c.y) / 3, (a.z + b.z + c.z) / 3);
}
__device__ __forceinline__ uint64_t spread21(uint64_t x) {
  x &= 0x1fffff; x = (x | x << 32) & 0x1f00000000ffffull; x = (x | x << 16) & 0x1f0000ff0000ffull;
  x = (x | x << 8) & 0x100f00f00f00f00full; x = (x | x << 4) & 0x10c30c30c30c30c3ull; return (x | x << 2) & 0x1249249249249249ull;
}
__global__ void k_lb_morton(const float3* c, i64 n, float3 lo, float3 sc, uint64_t* key, int* idx) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= n) return;
  auto q = [](float x, float l, float s) { return (uint64_t)fminf(fmaxf((x - l) * s, 0.f), 2097151.f); };
  key[i] = spread21(q(c[i].x, lo.x, sc.x)) << 2 | spread21(q(c[i].y, lo.y, sc.y)) << 1 | spread21(q(c[i].z, lo.z, sc.z)); idx[i] = (int)i;
}
__global__ void k_lb_gather(const cb::Triangle* t, const int* idx, i64 n, cb::Triangle* o) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i < n) o[i] = t[idx[i]];
}
__device__ __forceinline__ int lb_delta(const uint64_t* k, i64 n, i64 i, i64 j) {
  if (j < 0 || j >= n) return -1;
  return k[i] == k[j] ? 64 + __clzll((long long)(i ^ j)) : __clzll((long long)(k[i] ^ k[j]));
}
// internal node i: children encoded c >= 0 internal, c < 0 leaf ~c; parent[] over 2n-1 slots (internal i, leaf n-1+j)
__global__ void k_lb_tree(const uint64_t* k, i64 n, int2* ch, int* par) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= n - 1) return;
  int d = lb_delta(k, n, i, i + 1) > lb_delta(k, n, i, i - 1) ? 1 : -1, dmin = lb_delta(k, n, i, i - d);
  i64 lmax = 2; while (lb_delta(k, n, i, i + lmax * d) > dmin) lmax *= 2;
  i64 l = 0; for (i64 t = lmax / 2; t >= 1; t /= 2) if (lb_delta(k, n, i, i + (l + t) * d) > dmin) l += t;
  i64 j = i + l * d; int dn = lb_delta(k, n, i, j); i64 s = 0;
  for (i64 t = (l + 1) / 2;; t = (t + 1) / 2) { if (lb_delta(k, n, i, i + (s + t) * d) > dn) s += t; if (t == 1) break; }
  i64 g = i + s * d + (d < 0 ? -1 : 0), lo = i < j ? i : j, hi = i < j ? j : i;
  int a = lo == g ? ~(int)g : (int)g, b = hi == g + 1 ? ~(int)(g + 1) : (int)(g + 1);
  ch[i] = make_int2(a, b);
  par[a < 0 ? n - 1 + ~a : a] = (int)i; par[b < 0 ? n - 1 + ~b : b] = (int)i;
}
__global__ void k_lb_box(const cb::Triangle* t, const int2* ch, const int* par, i64 n, int* flag, float* box) {
  i64 j = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (j >= n) return;
  float* b = box + (n - 1 + j) * 6; const cb::Triangle& q = t[j];
  for (int c = 0; c < 3; c++) { b[c] = fminf(fminf(q.a[c], q.b[c]), q.c[c]); b[3 + c] = fmaxf(fmaxf(q.a[c], q.b[c]), q.c[c]); }
  if (n == 1) return;
  int p = par[n - 1 + j];
  while (true) {
    __threadfence();
    if (atomicAdd(&flag[p], 1) == 0) return;
    int2 c = ch[p]; const float* x = box + (c.x < 0 ? n - 1 + ~c.x : c.x) * 6; const float* y = box + (c.y < 0 ? n - 1 + ~c.y : c.y) * 6;
    float* o = box + (i64)p * 6;
    for (int e = 0; e < 3; e++) { o[e] = fminf(__ldcg(x + e), __ldcg(y + e)); o[3 + e] = fmaxf(__ldcg(x + 3 + e), __ldcg(y + 3 + e)); }
    if (p == 0) return;
    p = par[p];
  }
}
__device__ __forceinline__ float lb_bdist(const float* b, const Eigen::Vector3f& p) {
  Eigen::Vector3f mn(__ldg(b), __ldg(b + 1), __ldg(b + 2)), mx(__ldg(b + 3), __ldg(b + 4), __ldg(b + 5));
  return (mn - p).cwiseMax(p - mx).cwiseMax(0.0f).squaredNorm();
}
__global__ void k_lb_udf(const float3* pts, i64 m, const cb::Triangle* t, const int2* ch, const float* box, i64 n, float* dist, int64_t* fid, float3* uvw) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= m) return;
  Eigen::Vector3f p(pts[i].x, pts[i].y, pts[i].z);
  float best = 1000.f * 1000.f; int bi = -1;
  int st[96], sp = 0; st[sp++] = n == 1 ? ~0 : 0;
  while (sp) {
    int c = st[--sp];
    if (c < 0) { float d = t[~c].distance_sq(p); if (d < best || (d == best && ~c < bi)) { best = d; bi = ~c; } continue; }
    int2 k = __ldg(&ch[c]);
    float dl = lb_bdist(box + (k.x < 0 ? n - 1 + ~k.x : k.x) * 6, p), dr = lb_bdist(box + (k.y < 0 ? n - 1 + ~k.y : k.y) * 6, p);
    if (dl > dr) { int s = k.x; k.x = k.y; k.y = s; float f = dl; dl = dr; dr = f; }
    if (dr <= best) st[sp++] = k.y;
    if (dl <= best) st[sp++] = k.x;
  }
  if (bi < 0) { bi = 0; best = 0.f; }
  dist[i] = sqrtf(best); fid[i] = t[bi].id;
  if (uvw) { Eigen::Vector3f b = t[bi].barycentric(t[bi].closest_point(p)); uvw[i] = make_float3(b.x(), b.y(), b.z()); }
}
struct MinF3 { __device__ float3 operator()(float3 a, float3 b) const { return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)); } };
struct MaxF3 { __device__ float3 operator()(float3 a, float3 b) const { return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); } };
static LBVH lbvh_build(const Tensor& V, const Tensor& F) {
  LBVH b; i64 n = b.n = F.size(0); size_t ts = sizeof(cb::Triangle);
  Tensor t0 = empty({n * (i64)ts}, U8), cen = empty({n, 3}, F32), mm = empty({2, 3}, F32);
  k_lb_tri<<<cdiv(n, 256), 256, 0, stream()>>>(V.ptr<float3>(), F.ptr<int3>(), n, (cb::Triangle*)t0.p, cen.ptr<float3>());
  size_t a = 0; float big = 3.4e38f;
  cub::DeviceReduce::Reduce(nullptr, a, cen.ptr<float3>(), mm.ptr<float3>(), (int)n, MinF3(), make_float3(big, big, big), stream());
  Tensor tmp = empty({(i64)a}, U8);
  cub::DeviceReduce::Reduce(tmp.p, a, cen.ptr<float3>(), mm.ptr<float3>(), (int)n, MinF3(), make_float3(big, big, big), stream());
  cub::DeviceReduce::Reduce(tmp.p, a, cen.ptr<float3>(), mm.ptr<float3>() + 1, (int)n, MaxF3(), make_float3(-big, -big, -big), stream());
  auto h = to_host_vec<float>(mm);
  auto sc = [&](int c) { float e = h[3 + c] - h[c]; return e > 0 ? 2097151.f / e : 0.f; };
  Tensor key = empty({n}, I64), ks = empty({n}, I64), idx = empty({n}, I32), is = empty({n}, I32);
  k_lb_morton<<<cdiv(n, 256), 256, 0, stream()>>>(cen.ptr<float3>(), n, make_float3(h[0], h[1], h[2]), make_float3(sc(0), sc(1), sc(2)), key.ptr<uint64_t>(), idx.ptr<int>());
  size_t sb = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, sb, key.ptr<uint64_t>(), ks.ptr<uint64_t>(), idx.ptr<int>(), is.ptr<int>(), (int)n, 0, 63, stream());
  Tensor st = empty({(i64)sb}, U8);
  cub::DeviceRadixSort::SortPairs(st.p, sb, key.ptr<uint64_t>(), ks.ptr<uint64_t>(), idx.ptr<int>(), is.ptr<int>(), (int)n, 0, 63, stream());
  b.tri = empty({n * (i64)ts}, U8);
  k_lb_gather<<<cdiv(n, 256), 256, 0, stream()>>>((const cb::Triangle*)t0.p, is.ptr<int>(), n, (cb::Triangle*)b.tri.p);
  b.ch = empty({std::max<i64>(n - 1, 1), 2}, I32); b.box = empty({2 * n - 1, 6}, F32);
  Tensor par = empty({2 * n}, I32), flag = zeros({std::max<i64>(n - 1, 1)}, I32);
  if (n > 1) k_lb_tree<<<cdiv(n - 1, 256), 256, 0, stream()>>>(ks.ptr<uint64_t>(), n, b.ch.ptr<int2>(), par.ptr<int>());
  k_lb_box<<<cdiv(n, 256), 256, 0, stream()>>>((const cb::Triangle*)b.tri.p, b.ch.ptr<int2>(), par.ptr<int>(), n, flag.ptr<int>(), b.box.ptr<float>());
  return b;
}

struct BVH::Impl { std::vector<cb::Triangle> tri; cb::GPUMemory<cb::Triangle> gpu; std::unique_ptr<cb::TriangleBvh> bvh; LBVH lb; };
BVH::BVH(const Tensor& V, const Tensor& F) : p(std::make_shared<Impl>()) {
  if (fast_mode()) { p->lb = lbvh_build(V, F); prof("lbvh build"); return; }
  auto v = to_host_vec<float>(V); auto f = to_host_vec<int>(F); size_t n = F.size(0);
  p->tri.resize(n);
  auto row = [&](int i) { return Eigen::Vector3f(v[3 * (size_t)i], v[3 * (size_t)i + 1], v[3 * (size_t)i + 2]); };
  for (size_t i = 0; i < n; i++) p->tri[i] = {row(f[3 * i]), row(f[3 * i + 1]), row(f[3 * i + 2]), (int64_t)i};
  prof("bvh host prep");
  p->bvh = cb::TriangleBvh::make(); p->bvh->build(p->tri, 8);
  p->gpu.resize_and_copy_from_host(p->tri); prof("bvh build");
}
Tensor BVH::udf(const Tensor& pts, Tensor* face_id, Tensor* uvw) const {
  i64 n = pts.size(0); Tensor d = empty({n}, F32), fid = empty({n}, I64);
  if (uvw) *uvw = empty({n, 3}, F32);
  if (fast_mode()) {
    auto& b = p->lb;
    k_lb_udf<<<cdiv(n, 128), 128, 0, stream()>>>(pts.ptr<float3>(), n, (const cb::Triangle*)b.tri.p, b.ch.ptr<int2>(), b.box.ptr<float>(), b.n,
                                                d.ptr<float>(), fid.ptr<int64_t>(), uvw ? uvw->ptr<float3>() : nullptr);
    if (face_id) *face_id = fid;
    return d;
  }
  fence();
  p->bvh->unsigned_distance_gpu((uint32_t)n, pts.ptr<float>(), d.ptr<float>(), fid.ptr<int64_t>(), uvw ? uvw->ptr<float>() : nullptr, p->gpu.data(), 0);
  fence();
  if (face_id) *face_id = fid;
  return d;
}

static Tensor hash_keys(i64 cap) { Tensor k = empty({cap}, I32); CK(cudaMemsetAsync(k.p, 0xff, k.bytes(), stream())); return k; }
template <class T, class It> static Tensor select_flagged(It in, const Tensor& flag, i64 n, DT dt, std::vector<i64> tail) {
  Tensor out = empty([&] { std::vector<i64> s{n}; s.insert(s.end(), tail.begin(), tail.end()); return s; }(), dt), cnt = empty({1}, I32);
  size_t tb = 0;
  cub::DeviceSelect::Flagged(nullptr, tb, in, flag.ptr<int>(), (T*)out.p, cnt.ptr<int>(), (int)n, stream());
  Tensor tmp = empty({(i64)tb}, U8);
  cub::DeviceSelect::Flagged(tmp.p, tb, in, flag.ptr<int>(), (T*)out.p, cnt.ptr<int>(), (int)n, stream());
  return out.slice0(0, to_host_vec<int>(cnt)[0]);
}

// ((c + 0.5) / base - 0.5) * scale + center   |   (g / res - 0.5) * scale + center   (ATen scalar div == mul by reciprocal)
__global__ void k_grid_pts(const int* c, i64 n, float half, float inv, float scale, float3 cen, float* p) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= n * 3) return;
  float ce = i % 3 == 0 ? cen.x : i % 3 == 1 ? cen.y : cen.z;
  p[i] = __fadd_rn(__fmul_rn(__fsub_rn(__fmul_rn(__fadd_rn((float)c[i], half), inv), 0.5f), scale), ce);
}
__global__ void k_grid_pts_f(const float* c, i64 n, float inv, float scale, float3 cen, float* p) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= n * 3) return;
  float ce = i % 3 == 0 ? cen.x : i % 3 == 1 ? cen.y : cen.z;
  p[i] = __fadd_rn(__fmul_rn(__fsub_rn(__fmul_rn(c[i], inv), 0.5f), scale), ce);
}
__global__ void k_band(const float* d, i64 n, float eps, float thr, int* flag) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i < n) flag[i] = fabsf(__fsub_rn(d[i], eps)) < thr;
}
__global__ void k_sub_(float* d, i64 n, float eps) { i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i < n) d[i] = __fsub_rn(d[i], eps); }
__global__ void k_expand8(const int3* c, i64 n, int3* o) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= n * 8) return;
  int3 q = c[i / 8]; int s = (int)(i % 8);
  o[i] = make_int3(q.x * 2 + (s & 1), q.y * 2 + (s >> 1 & 1), q.z * 2 + (s >> 2 & 1));
}
__global__ void k_to4(const int3* c, i64 n, int4* o) { i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i < n) o[i] = make_int4(0, c[i].x, c[i].y, c[i].z); }
__constant__ int c_dc_off[3][4][3] = {
  {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}},
  {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}},
  {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}},
};
__global__ void k_nz(const int* x, i64 n, int* f) { i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i < n) f[i] = x[i] != 0; }
__global__ void k_edge_keys(const int* sel, i64 m, const int3* c, int4* keys) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= m * 4) return;
  int e = sel[i / 4], k = (int)(i % 4), a = e % 3; int3 q = c[e / 3];
  keys[i] = make_int4(0, q.x + c_dc_off[a][k][0], q.y + c_dc_off[a][k][1], q.z + c_dc_off[a][k][2]);
}
__global__ void k_quad_valid(const uint32_t* idx, i64 m, int* f) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= m) return;
  f[i] = idx[4 * i] != 0xffffffffu && idx[4 * i + 1] != 0xffffffffu && idx[4 * i + 2] != 0xffffffffu && idx[4 * i + 3] != 0xffffffffu;
}
__global__ void k_gather_dir(const int* sel, i64 m, const int* inter, int* dir) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i < m) dir[i] = inter[sel[i]];
}
__global__ void k_scatter_iota(const int* u, i64 k, int* map) { i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i < k) map[u[i]] = (int)i; }
__global__ void k_remap(int* q, i64 n, const int* map) { i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i < n) q[i] = map[q[i]]; }
__global__ void k_gather_rows3(const float* x, const int* idx, i64 n, float* y) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= n * 3) return; y[i] = x[(i64)idx[i / 3] * 3 + i % 3];
}
// torch.cross(a, b, dim=1) on CUDA compiles to fma(a1, b2, -(a2 * b1)); (x * y).sum(dim=1) over 3 reduces as (x0 + x2) + x1.
__device__ __forceinline__ float3 cross_at(float3 a, float3 b) {
  return make_float3(__fmaf_rn(a.y, b.z, -__fmul_rn(a.z, b.y)), __fmaf_rn(a.z, b.x, -__fmul_rn(a.x, b.z)), __fmaf_rn(a.x, b.y, -__fmul_rn(a.y, b.x)));
}
__device__ __forceinline__ float3 sub3(float3 a, float3 b) { return make_float3(__fsub_rn(a.x, b.x), __fsub_rn(a.y, b.y), __fsub_rn(a.z, b.z)); }
__device__ __forceinline__ float align_at(const float3* v, const int* t) {
  float3 n0 = cross_at(sub3(v[t[1]], v[t[0]]), sub3(v[t[2]], v[t[0]])), n1 = cross_at(sub3(v[t[2]], v[t[1]]), sub3(v[t[3]], v[t[1]]));
  return fabsf(__fadd_rn(__fadd_rn(__fmul_rn(n0.x, n1.x), __fmul_rn(n0.z, n1.z)), __fmul_rn(n0.y, n1.y)));
}
__global__ void k_dc_tris(const int4* quad, const int* dir, i64 L, const float3* v, int* f) {
  i64 l = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (l >= L) return;
  int4 q4 = quad[l]; int q[4] = {q4.x, q4.y, q4.z, q4.w}; bool p = dir[l] == 1;
  const int s1n[6] = {0, 1, 2, 0, 2, 3}, s1p[6] = {0, 2, 1, 0, 3, 2}, s2n[6] = {0, 1, 3, 3, 1, 2}, s2p[6] = {0, 3, 1, 3, 2, 1};
  int t0[6], t1[6];
  for (int k = 0; k < 6; k++) { t0[k] = q[p ? s1p[k] : s1n[k]]; t1[k] = q[p ? s2p[k] : s2n[k]]; }
  const int* t = align_at(v, t0) > align_at(v, t1) ? t0 : t1;
  for (int k = 0; k < 6; k++) f[l * 6 + k] = t[k];
}

static void remesh_dc(const BVH& bvh, float3 cen, double scale, int res, double band, Tensor& V, Tensor& F) {
  double eps = band * scale / res; float fs = (float)scale;
  int base = res; while (base > 32) base /= 2;
  std::vector<int> g0((size_t)base * base * base * 3);
  for (int x = 0, i = 0; x < base; x++) for (int y = 0; y < base; y++) for (int z = 0; z < base; z++, i++) { g0[3 * i] = x; g0[3 * i + 1] = y; g0[3 * i + 2] = z; }
  Tensor c = from_host(g0.data(), {(i64)base * base * base, 3}, I32);
  while (true) {
    i64 n = c.size(0); Tensor pts = empty({n, 3}, F32), flag = empty({n}, I32);
    k_grid_pts<<<cdiv(n * 3, 256), 256, 0, stream()>>>(c.ptr<int>(), n, 0.5f, 1.f / base, fs, cen, pts.ptr<float>());
    Tensor d = bvh.udf(pts);
    k_band<<<cdiv(n, 256), 256, 0, stream()>>>(d.ptr<float>(), n, (float)eps, (float)(0.87 * (scale / base)), flag.ptr<int>());
    c = select_flagged<int3>(c.ptr<int3>(), flag, n, I32, {3});
    if (base >= res) break;
    base *= 2; Tensor e = empty({c.size(0) * 8, 3}, I32);
    k_expand8<<<cdiv(c.size(0) * 8, 256), 256, 0, stream()>>>(c.ptr<int3>(), c.size(0), e.ptr<int3>()); c = e;
  }
  i64 nv = c.size(0); Tensor c4 = empty({nv, 4}, I32);
  k_to4<<<cdiv(nv, 256), 256, 0, stream()>>>(c.ptr<int3>(), nv, c4.ptr<int4>());
  Tensor hk = hash_keys(2 * nv), hv = empty({2 * nv}, I32);
  TT thk = tu32(hk), thv = tu32(hv);
  fence(); cumesh::hashmap_insert_3d_idx_as_val_cuda(thk, thv, ti32(c4), res, res, res);
  Tensor gv = cumesh::get_sparse_voxel_grid_active_vertices(thk, thv, ti32(c), res, res, res).t; fence();
  i64 ng = gv.size(0); Tensor pv = empty({ng, 3}, F32);
  k_grid_pts<<<cdiv(ng * 3, 256), 256, 0, stream()>>>(gv.ptr<int>(), ng, 0.f, 1.f / res, fs, cen, pv.ptr<float>());
  Tensor dv = bvh.udf(pv);
  k_sub_<<<cdiv(ng, 256), 256, 0, stream()>>>(dv.ptr<float>(), ng, (float)eps);
  Tensor g4 = empty({ng, 4}, I32);
  k_to4<<<cdiv(ng, 256), 256, 0, stream()>>>(gv.ptr<int3>(), ng, g4.ptr<int4>());
  Tensor vk = hash_keys(2 * ng), vvl = empty({2 * ng}, I32); TT tvk = tu32(vk), tvv = tu32(vvl);
  fence(); cumesh::hashmap_insert_3d_idx_as_val_cuda(tvk, tvv, ti32(g4), res + 1, res + 1, res + 1);
  auto [tdual, tinter] = cumesh::simple_dual_contour(tvk, tvv, ti32(c), tf32(dv), res + 1, res + 1, res + 1); fence();
  Tensor dual = tdual.t, inter = tinter.t;

  Tensor nz = empty({nv * 3}, I32);
  k_nz<<<cdiv(nv * 3, 256), 256, 0, stream()>>>(inter.ptr<int>(), nv * 3, nz.ptr<int>());
  cub::CountingInputIterator<int> it(0);
  Tensor sel = select_flagged<int>(it, nz, nv * 3, I32, {});
  i64 M = sel.size(0); Tensor keys = empty({M * 4, 4}, I32);
  k_edge_keys<<<cdiv(M * 4, 256), 256, 0, stream()>>>(sel.ptr<int>(), M, c.ptr<int3>(), keys.ptr<int4>());
  fence(); Tensor idx = cumesh::hashmap_lookup_3d_cuda(thk, thv, ti32(keys), res, res, res).t; fence();
  Tensor ok = empty({M}, I32), dirM = empty({M}, I32);
  k_quad_valid<<<cdiv(M, 256), 256, 0, stream()>>>(idx.ptr<uint32_t>(), M, ok.ptr<int>());
  k_gather_dir<<<cdiv(M, 256), 256, 0, stream()>>>(sel.ptr<int>(), M, inter.ptr<int>(), dirM.ptr<int>());
  Tensor quad = select_flagged<int4>(idx.ptr<int4>(), ok, M, I32, {4}), dir = select_flagged<int>(dirM.ptr<int>(), ok, M, I32, {});
  i64 L = quad.size(0);

  Tensor qs = empty({L * 4}, I32), qu = empty({L * 4}, I32), cnt = empty({1}, I32); size_t a = 0, b = 0;
  cub::DeviceRadixSort::SortKeys(nullptr, a, quad.ptr<int>(), qs.ptr<int>(), (int)(L * 4), 0, 32, stream());
  cub::DeviceSelect::Unique(nullptr, b, qs.ptr<int>(), qu.ptr<int>(), cnt.ptr<int>(), (int)(L * 4), stream());
  Tensor tmp = empty({(i64)std::max(a, b)}, U8);
  cub::DeviceRadixSort::SortKeys(tmp.p, a, quad.ptr<int>(), qs.ptr<int>(), (int)(L * 4), 0, 32, stream());
  cub::DeviceSelect::Unique(tmp.p, b, qs.ptr<int>(), qu.ptr<int>(), cnt.ptr<int>(), (int)(L * 4), stream());
  i64 K = to_host_vec<int>(cnt)[0];
  Tensor vmap = zeros({nv}, I32), dv3 = empty({K, 3}, F32);
  k_scatter_iota<<<cdiv(K, 256), 256, 0, stream()>>>(qu.ptr<int>(), K, vmap.ptr<int>());
  k_remap<<<cdiv(L * 4, 256), 256, 0, stream()>>>(quad.ptr<int>(), L * 4, vmap.ptr<int>());
  k_gather_rows3<<<cdiv(K * 3, 256), 256, 0, stream()>>>(dual.ptr<float>(), qu.ptr<int>(), K, dv3.ptr<float>());
  V = empty({K, 3}, F32);
  k_grid_pts_f<<<cdiv(K * 3, 256), 256, 0, stream()>>>(dv3.ptr<float>(), K, 1.f / res, fs, cen, V.ptr<float>());
  F = empty({L * 2, 3}, I32);
  k_dc_tris<<<cdiv(L, 256), 256, 0, stream()>>>(quad.ptr<int4>(), dir.ptr<int>(), L, V.ptr<float3>(), F.ptr<int>());
}

static void cm_read(cumesh::CuMesh& m, Tensor& V, Tensor& F) { auto [v, f] = m.read(); fence(); V = v.t; F = f.t; }

// CuMesh.simplify (python driver around simplify_step)
static void simplify(cumesh::CuMesh& m, int target, double thresh = 1e-8, float l_edge = 1e-2f, float l_skinny = 1e-3f) {
  int nf = m.num_faces(); if (nf <= target) return;
  while (true) {
    auto [nv2, nf2] = m.simplify_step(l_edge, l_skinny, (float)thresh, false); fence();
    if (nf2 <= target) break;
    if ((double)(nf - nf2) / nf < 1e-2) thresh *= 10;
    nf = nf2;
  }
}

__global__ void k_gather_rows(const float* x, const int* idx, i64 n, int C, float* y) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= n * C) return; y[i] = x[(i64)idx[i / C] * C + i % C];
}
static Tensor gather_rows(const Tensor& x, const Tensor& idx) {
  int C = (int)x.size(1); i64 n = idx.numel(); Tensor y = empty({n, C}, F32);
  k_gather_rows<<<cdiv(n * C, 256), 256, 0, stream()>>>(x.ptr<float>(), idx.ptr<int>(), n, C, y.ptr<float>()); return y;
}

// CuMesh.uv_unwrap(compute_charts_kwargs=to_glb's, return_vmaps=True): vertices, faces, uvs, vmaps
static void uv_unwrap(cumesh::CuMesh& m, Tensor& V, Tensor& F, Tensor& UV, Tensor& vmaps) {
  m.remove_degenerate_faces(1e-24f, 1e-12f);
  prof("unwrap start");
  m.compute_charts((float)1.5707963267948966, 0, 1, 1.f, 0.1f, 1e-4f); fence(); prof("compute_charts");
  Tensor nv, nf; cm_read(m, nv, nf);
  auto [nc, cid, cvm, cf, cvo, cfo] = m.read_atlas_charts(); fence();
  auto hv = to_host_vec<float>(gather_rows(nv, cvm.t)); auto hf = to_host_vec<int>(cf.t);
  auto hvo = to_host_vec<int>(cvo.t), hfo = to_host_vec<int>(cfo.t), hvm = to_host_vec<int>(cvm.t);
  // One xatlas mesh per chart. Fast build: xatlas chart growth is quadratic in mesh size and serial per mesh, so charts
  // above `split` faces are cut by recursive centroid-median splits into independent meshes (extra seams only).
  struct XM { std::vector<float> pos; std::vector<uint32_t> idx; std::vector<int> gid; };
  std::vector<XM> xm;
  const int split = fast_mode() ? std::stoi(env("T2_XA_SPLIT", "1024")) : INT_MAX;
  auto emit = [&](int i, const uint32_t* tri, size_t nt) {  // tri: chart-local vertex ids
    XM o; std::unordered_map<uint32_t, uint32_t> remap;
    for (size_t k = 0; k < nt * 3; k++) {
      auto [it, fresh] = remap.try_emplace(tri[k], (uint32_t)o.gid.size());
      if (fresh) { o.gid.push_back(hvm[hvo[i] + tri[k]]); for (int c = 0; c < 3; c++) o.pos.push_back(hv[((size_t)hvo[i] + tri[k]) * 3 + c]); }
      o.idx.push_back(it->second);
    }
    xm.push_back(std::move(o));
  };
  std::function<void(int, std::vector<uint32_t>&, size_t, size_t)> cut = [&](int i, std::vector<uint32_t>& t, size_t a, size_t b) {
    if (b - a <= (size_t)split) { emit(i, &t[a * 3], b - a); return; }
    auto cen = [&](size_t f, int c) { float s = 0; for (int k = 0; k < 3; k++) s += hv[((size_t)hvo[i] + t[f * 3 + k]) * 3 + c]; return s; };
    float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (size_t f = a; f < b; f++) for (int c = 0; c < 3; c++) { float v = cen(f, c); lo[c] = std::min(lo[c], v); hi[c] = std::max(hi[c], v); }
    int ax = 0; for (int c = 1; c < 3; c++) if (hi[c] - lo[c] > hi[ax] - lo[ax]) ax = c;
    std::vector<std::pair<float, uint32_t>> key(b - a);
    for (size_t f = a; f < b; f++) key[f - a] = {cen(f, ax), (uint32_t)f};
    size_t mid = (b - a) / 2;
    std::nth_element(key.begin(), key.begin() + mid, key.end());
    std::vector<uint32_t> tmp((b - a) * 3);
    for (size_t r = 0; r < key.size(); r++) memcpy(&tmp[r * 3], &t[key[r].second * 3], 12);
    memcpy(&t[a * 3], tmp.data(), tmp.size() * 4);
    cut(i, t, a, a + mid); cut(i, t, a + mid, b);
  };
  for (int i = 0; i < nc; i++) {
    int nt = hfo[i + 1] - hfo[i];
    if (nt <= split) {
      XM o; o.pos.assign(hv.begin() + (size_t)hvo[i] * 3, hv.begin() + (size_t)hvo[i + 1] * 3);
      o.gid.assign(hvm.begin() + hvo[i], hvm.begin() + hvo[i + 1]);
      for (int k = hfo[i] * 3; k < hfo[i + 1] * 3; k++) o.idx.push_back((uint32_t)(hf[k] - hvo[i]));
      xm.push_back(std::move(o));
    } else {
      std::vector<uint32_t> t(hf.begin() + (size_t)hfo[i] * 3, hf.begin() + (size_t)hfo[i + 1] * 3);
      for (auto& x : t) x -= hvo[i];
      cut(i, t, 0, nt);
    }
  }
  if (env("T2_XA_VERBOSE") == "1") cumesh_xatlas::SetPrint(printf, true);
  cumesh_xatlas::Atlas* A = cumesh_xatlas::Create();
  for (auto& o : xm) {
    cumesh_xatlas::MeshDecl d;
    d.vertexCount = (uint32_t)o.gid.size(); d.vertexPositionData = o.pos.data(); d.vertexPositionStride = 12;
    d.indexCount = (uint32_t)o.idx.size(); d.indexData = o.idx.data(); d.indexFormat = cumesh_xatlas::IndexFormat::UInt32;
    REQ(cumesh_xatlas::AddMesh(A, d) == cumesh_xatlas::AddMeshError::Success, "xatlas AddMesh failed");
  }
  cumesh_xatlas::ChartOptions co;
  co.maxChartArea = 0; co.maxBoundaryLength = 0; co.normalDeviationWeight = 2; co.roundnessWeight = 0.01f; co.straightnessWeight = 6;
  co.normalSeamWeight = 4; co.textureSeamWeight = 0.5f; co.maxCost = 2; co.maxIterations = 1; co.useInputMeshUvs = false; co.fixWinding = false;
  prof("xatlas add");
  cumesh_xatlas::ComputeCharts(A, co); prof("xatlas ComputeCharts");
  cumesh_xatlas::PackOptions po;
  po.maxChartSize = 0; po.padding = 0; po.texelsPerUnit = 0; po.resolution = 0; po.bilinear = true; po.blockAlign = false;
  po.bruteForce = false; po.rotateCharts = true; po.rotateChartsToAxis = true;
  cumesh_xatlas::PackCharts(A, po); prof("xatlas PackCharts");
  if (env("T2_PROF") == "1") {
    std::vector<int> sz(nc); for (int i = 0; i < nc; i++) sz[i] = hfo[i + 1] - hfo[i];
    std::sort(sz.rbegin(), sz.rend());
    printf("    charts %d -> %zu meshes (largest %d %d %d of %d faces), atlas %ux%u, %u xatlas charts\n", nc, xm.size(), sz[0], sz[1], sz[2], hfo[nc], A->width, A->height, A->chartCount);
    if (auto p = env("T2_CHARTS"); !p.empty()) { std::ofstream o(p); for (int s : sz) o << s << '\n'; }
  }
  std::vector<int> vm, ff; std::vector<float> uv; float W = (float)A->width, H = (float)A->height;
  for (size_t i = 0, cnt = 0; i < xm.size(); i++) {
    const auto& me = A->meshes[i];
    for (uint32_t k = 0; k < me.vertexCount; k++) {
      const auto& v = me.vertexArray[k]; vm.push_back(xm[i].gid[v.xref]);
      uv.push_back(W > 0 && H > 0 ? v.uv[0] / W : 0.f); uv.push_back(W > 0 && H > 0 ? v.uv[1] / H : 0.f);
    }
    for (uint32_t k = 0; k < me.indexCount; k++) ff.push_back((int)me.indexArray[k] + cnt);
    cnt += me.vertexCount;
  }
  cumesh_xatlas::Destroy(A);
  vmaps = from_host(vm.data(), {(i64)vm.size()}, I32); F = from_host(ff.data(), {(i64)ff.size() / 3, 3}, I32);
  UV = from_host(uv.data(), {(i64)vm.size(), 2}, F32); V = gather_rows(nv, vmaps);
}

void Glb::remesh(const Tensor& V, const Tensor& F, Tensor& Vr, Tensor& Fr) {
  cumesh::CuMesh m; fence(); m.init(tf32(V), ti32(F));
  m.fill_holes(3e-2f); fence(); cm_read(m, V0, F0); tap("glb/fill_holes0.0", V0); tap("glb/fill_holes0.1", F0);
  bvh = std::make_unique<BVH>(V0, F0);
  remesh_dc(*bvh, make_float3(0, 0, 0), (1024 + 3 * 1.0) / 1024 * 1.0, 1024, 1.0, Vr, Fr); prof("remesh_dc");
  tap("glb/init1.0", Vr); tap("glb/init1.1", Fr);
}
void Glb::simplify(Tensor& V, Tensor& F, int target) {
  cumesh::CuMesh m; fence(); m.init(tf32(V), ti32(F)); fence();
  ::simplify(m, target); cm_read(m, V, F); tap("glb/simplify0.0", V); tap("glb/simplify0.1", F);
}
void Glb::unwrap(const Tensor& Vs, const Tensor& Fs) {
  cumesh::CuMesh m; fence(); m.init(tf32(Vs), ti32(Fs)); fence();
  Tensor vmaps; uv_unwrap(m, V, F, UV, vmaps);
  tap("glb/uv_unwrap.0", V); tap("glb/uv_unwrap.1", F); tap("glb/uv_unwrap.2", UV); tap("glb/uv_unwrap.3", vmaps);
  m.compute_vertex_normals(); fence();
  N = gather_rows(m.read_vertex_normals().t, vmaps); fence();
}
void Glb::geometry(const Tensor& V, const Tensor& F, int target) {
  Tensor a, b; remesh(V, F, a, b); simplify(a, b, target); unwrap(a, b);
}

__global__ void k_uv_pos(const float2* uv, i64 n, float4* p) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x;
  if (i < n) p[i] = make_float4(__fsub_rn(__fmul_rn(uv[i].x, 2.f), 1.f), __fsub_rn(__fmul_rn(uv[i].y, 2.f), 1.f), 0.f, 1.f);
}
__global__ void k_rast_merge(const float4* c, i64 n, float off, float4* r) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= n) return;
  float4 v = c[i]; if (v.w > 0) r[i] = make_float4(v.x, v.y, v.z, __fadd_rn(v.w, off));
}
__global__ void k_wpos(const float4* r, i64 n, int* f) { i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i < n) f[i] = r[i].w > 0; }
// (V0[F0[fid]] * uvw[..., None]).sum(1) -> (p - aabb0) / voxel_size
__global__ void k_reproj(const float3* v, const int3* f, const int64_t* fid, const float3* uvw, i64 n, float lo, float vs, float3* g) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= n) return;
  int3 t = f[fid[i]]; float3 w = uvw[i], a = v[t.x], b = v[t.y], c = v[t.z];
  auto s = [](float a, float b, float c, float u, float v, float w) { return __fadd_rn(__fadd_rn(__fmul_rn(a, u), __fmul_rn(b, v)), __fmul_rn(c, w)); };
  g[i] = make_float3(__fdiv_rn(__fsub_rn(s(a.x, b.x, c.x, w.x, w.y, w.z), lo), vs), __fdiv_rn(__fsub_rn(s(a.y, b.y, c.y, w.x, w.y, w.z), lo), vs),
                     __fdiv_rn(__fsub_rn(s(a.z, b.z, c.z, w.x, w.y, w.z), lo), vs));
}
__global__ void k_coords4(const int3* c, i64 n, int4* o) { i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i < n) o[i] = make_int4(0, c[i].x, c[i].y, c[i].z); }
// flex_gemm indice_weighed_sum_fwd (Triton: acc += x * w contracted to fma, v = 0..7)
template <int C> __global__ void k_wsum(const float* x, const uint32_t* idx, const float* w, i64 n, float* y) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= n) return;
  float acc[C] = {};
  for (int v = 0; v < 8; v++) {
    uint32_t j = idx[i * 8 + v]; float wv = w[i * 8 + v];
    for (int c = 0; c < C; c++) acc[c] = __fmaf_rn(j != 0xffffffffu ? x[(i64)j * C + c] : 0.f, wv, acc[c]);
  }
  for (int c = 0; c < C; c++) y[i * C + c] = acc[c];
}
// np.clip(attrs * 255, 0, 255).astype(uint8) into 6 planes [P]: base r, g, b, metallic, roughness, alpha
__global__ void k_to_u8(const float* a, const int* pix, i64 n, i64 P, uint8_t* o) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= n) return;
  for (int c = 0; c < 6; c++) o[c * P + pix[i]] = (uint8_t)fminf(fmaxf(__fmul_rn(a[i * 6 + c], 255.f), 0.f), 255.f);
}

// Fast build TELEA: fill by 4-connected distance level L (T := L); each level in parallel from levels < L.
__global__ void k_lv_init(const int* flag, i64 P, int* lv) { i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i < P) lv[i] = flag[i] ? 0 : -1; }
__global__ void k_lv_mark(int* lv, int T, int L, int* cnt) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= (i64)T * T || lv[i] != -1) return;
  int y = i / T, x = i % T; int p = L - 1;
  if ((x > 0 && lv[i - 1] == p) || (x < T - 1 && lv[i + 1] == p) || (y > 0 && lv[i - T] == p) || (y < T - 1 && lv[i + T] == p)) { lv[i] = L; atomicAdd(cnt, 1); }
}
template <int R> __global__ void k_telea_lv(const int* lv, int T, int L, uint8_t* img, i64 P, int c0) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= (i64)T * T || lv[i] != L) return;
  int y = i / T, x = i % T;
  auto kn = [&](int r, int c) { if (r < 0 || c < 0 || r >= T || c >= T) return false; int v = lv[(i64)r * T + c]; return v >= 0 && v < L; };
  auto tv = [&](int r, int c) { return (float)lv[(i64)r * T + c]; };
  auto I = [&](int ch, int r, int c) { return (float)img[ch * P + (i64)r * T + c]; };
  float gx = kn(y, x + 1) ? (kn(y, x - 1) ? (tv(y, x + 1) - tv(y, x - 1)) * 0.5f : tv(y, x + 1) - L) : (kn(y, x - 1) ? L - tv(y, x - 1) : 0.f);
  float gy = kn(y + 1, x) ? (kn(y - 1, x) ? (tv(y + 1, x) - tv(y - 1, x)) * 0.5f : tv(y + 1, x) - L) : (kn(y - 1, x) ? L - tv(y - 1, x) : 0.f);
  float Ia[3] = {}, Jx[3] = {}, Jy[3] = {}, s = 1e-20f;
  for (int k = y - R; k <= y + R; k++)
    for (int l = x - R; l <= x + R; l++) {
      if ((k - y) * (k - y) + (l - x) * (l - x) > R * R || !kn(k, l)) continue;
      float ry = (float)(y - k), rx = (float)(x - l), vl = rx * rx + ry * ry;
      float dir = rx * gx + ry * gy; if (fabsf(dir) <= 0.01f) dir = 1e-6f;
      float w = fabsf(dir / (vl * sqrtf(vl) * (1 + fabsf(tv(k, l) - L))));
      bool r1 = kn(k, l + 1), l1 = kn(k, l - 1), d1 = kn(k + 1, l), u1 = kn(k - 1, l);
      for (int c = 0; c < 3; c++) {
        int ch = c0 + c; float v = I(ch, k, l);
        float ix = r1 ? (l1 ? (I(ch, k, l + 1) - I(ch, k, l - 1)) * 2.f : I(ch, k, l + 1) - v) : (l1 ? v - I(ch, k, l - 1) : 0.f);
        float iy = d1 ? (u1 ? (I(ch, k + 1, l) - I(ch, k - 1, l)) * 2.f : I(ch, k + 1, l) - v) : (u1 ? v - I(ch, k - 1, l) : 0.f);
        Ia[c] += w * v; Jx[c] -= w * ix * rx; Jy[c] -= w * iy * ry;
      }
      s += w;
    }
  for (int c = 0; c < 3; c++) {
    float sat = Ia[c] / s + (Jx[c] + Jy[c]) / (sqrtf(Jx[c] * Jx[c] + Jy[c] * Jy[c]) + 1e-20f);
    img[(c0 + c) * P + i] = (uint8_t)fminf(fmaxf(rintf(sat + 0.5f), 0.f), 255.f);
  }
}
static void telea_gpu(const Tensor& flag, int T, Tensor& u8) {
  i64 P = (i64)T * T; Tensor lv = empty({P}, I32), cnt = zeros({1}, I32);
  k_lv_init<<<cdiv(P, 256), 256, 0, stream()>>>(flag.ptr<int>(), P, lv.ptr<int>());
  std::string dump = env("T2_TELEA"); std::ofstream o; if (!dump.empty()) o.open(dump);
  for (int L = 1;; L++) {
    CK(cudaMemsetAsync(cnt.p, 0, 4, stream()));
    k_lv_mark<<<cdiv(P, 256), 256, 0, stream()>>>(lv.ptr<int>(), T, L, cnt.ptr<int>());
    k_telea_lv<3><<<cdiv(P, 256), 256, 0, stream()>>>(lv.ptr<int>(), T, L, u8.ptr<uint8_t>(), P, 0);
    k_telea_lv<1><<<cdiv(P, 256), 256, 0, stream()>>>(lv.ptr<int>(), T, L, u8.ptr<uint8_t>(), P, 3);
    int n = to_host_vec<int>(cnt)[0]; if (n == 0) break;
    if (!dump.empty()) o << n << '\n';
  }
}

void Glb::bake(const Tensor& attrs, const Tensor& coords, int T) {
  tex = T; i64 P = (i64)T * T, nv = V.size(0), nf = F.size(0);
  Tensor pos = empty({nv, 4}, F32), rast = zeros({P, 4}, F32), chunk = empty({P, 4}, F32), db = empty({P, 4}, F32);
  k_uv_pos<<<cdiv(nv, 256), 256, 0, stream()>>>(UV.ptr<float2>(), nv, pos.ptr<float4>());
  auto ctx = nvdr_ctx();
  for (i64 i = 0; i < nf; i += 100000) {
    nvdr_rasterize(*ctx, pos.ptr<float>(), (int)nv, F.ptr<int>() + i * 3, (int)std::min<i64>(100000, nf - i), T, T, chunk.ptr<float>(), db.ptr<float>(), stream());
    k_rast_merge<<<cdiv(P, 256), 256, 0, stream()>>>(chunk.ptr<float4>(), P, (float)i, rast.ptr<float4>());
  }
  chunk = db = Tensor();
  Tensor flag = empty({P}, I32), ipos = empty({P, 3}, F32);
  k_wpos<<<cdiv(P, 256), 256, 0, stream()>>>(rast.ptr<float4>(), P, flag.ptr<int>());
  nvdr_interpolate(V.ptr<float>(), (int)nv, 3, rast.ptr<float>(), T, T, F.ptr<int>(), (int)nf, ipos.ptr<float>(), stream());
  rast = Tensor();
  Tensor vpos = select_flagged<float3>(ipos.ptr<float3>(), flag, P, F32, {3}); ipos = Tensor();
  cub::CountingInputIterator<int> it(0);
  Tensor pix = select_flagged<int>(it, flag, P, I32, {});
  i64 L = vpos.size(0);
  prof("bake raster");
  Tensor fid, uvw; bvh->udf(vpos, &fid, &uvw); prof("bake udf");
  Tensor grid = empty({1, L, 3}, F32);
  k_reproj<<<cdiv(L, 256), 256, 0, stream()>>>(V0.ptr<float3>(), F0.ptr<int3>(), fid.ptr<int64_t>(), uvw.ptr<float3>(), L, -0.5f, 1.f / 1024, grid.ptr<float3>());
  tap("glb/grid", grid);
  i64 M = coords.size(0); Tensor c4 = coords;
  if (coords.size(1) == 3) { c4 = empty({M, 4}, I32); k_coords4<<<cdiv(M, 256), 256, 0, stream()>>>(coords.ptr<int3>(), M, c4.ptr<int4>()); }
  Tensor hk = hash_keys((i64)(2.0 * M)), hv = empty({(i64)(2.0 * M)}, I32); TT thk = tu32(hk), thv = tu32(hv);
  fence();
  auto [nb, wt] = flex_gemm::grid_sample::hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight(thk, thv, ti32(c4), tf32(grid), 1024, 1024, 1024);
  fence();
  Tensor samp = empty({1, L, 6}, F32);
  k_wsum<6><<<cdiv(L, 256), 256, 0, stream()>>>(attrs.ptr<float>(), nb.t.ptr<uint32_t>(), wt.t.ptr<float>(), L, samp.ptr<float>());
  tap("glb/attrs", samp);
  Tensor u8 = zeros({P * 6}, U8);
  k_to_u8<<<cdiv(L, 256), 256, 0, stream()>>>(samp.ptr<float>(), pix.ptr<int>(), L, P, u8.ptr<uint8_t>());
  std::vector<uint8_t> h;
  auto rgb = [&] { std::vector<uint8_t> o(P * 3); for (i64 i = 0; i < P * 3; i++) o[i] = h[(i % 3) * P + i / 3]; return from_host(o.data(), {T, T, 3}, U8); };
  if (fast_mode()) {
    prof("bake gpu");
    telea_gpu(flag, T, u8); h = to_host_vec<uint8_t>(u8);
  } else {
    h = to_host_vec<uint8_t>(u8); auto hf = to_host_vec<int>(flag);
    std::vector<uint8_t> inv(P); for (i64 i = 0; i < P; i++) inv[i] = !hf[i];
    if (dbg) {
      tap("glb/inpaint_in0", rgb());
      for (int k = 1; k < 4; k++) tap("glb/inpaint_in" + std::to_string(k), from_host(h.data() + P * (2 + k), {T, T, 1}, U8));
    }
    prof("bake gpu");
    std::vector<std::thread> th;
    for (int k = 0; k < 6; k++) th.emplace_back([&, k] { inpaint_telea(h.data() + P * k, T, T, k < 3, inv.data(), k < 3 ? 3 : 1); });
    for (auto& t : th) t.join();
  }
  prof("telea");
  if (dbg) {
    tap("glb/inpaint0", rgb());
    for (int k = 1; k < 4; k++) tap("glb/inpaint" + std::to_string(k), from_host(h.data() + P * (2 + k), {T, T}, U8));
  }
  base_rgba.resize(P * 4); mr_rgb.resize(P * 3);
  for (i64 i = 0; i < P; i++) {
    for (int c = 0; c < 3; c++) base_rgba[i * 4 + c] = h[c * P + i];
    base_rgba[i * 4 + 3] = h[P * 5 + i]; mr_rgb[i * 3] = 0; mr_rgb[i * 3 + 1] = h[P * 4 + i]; mr_rgb[i * 3 + 2] = h[P * 3 + i];
  }
}
