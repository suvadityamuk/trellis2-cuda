// FlexiDualGrid -> triangle mesh (o_voxel.convert.flexible_dual_grid_to_mesh, inference branch) and CuMesh-based
// mesh post-processing (trellis2 Mesh.fill_holes call sequence).
#include "models.h"
#include "cumesh.h"
#include <cub/cub.cuh>

__device__ __forceinline__ uint64_t mix64(uint64_t k) { k ^= k >> 33; k *= 0xff51afd7ed558ccdull; k ^= k >> 33; return k; }
__global__ void k_vhash_insert(const int4* c, int n, uint64_t* keys, int* vals, uint64_t mask) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  int4 q = c[i]; uint64_t k = ((uint64_t)q.y << 42) | ((uint64_t)q.z << 21) | (uint64_t)q.w;
  for (uint64_t s = mix64(k) & mask;; s = (s + 1) & mask)
    if (atomicCAS((unsigned long long*)&keys[s], ~0ull, k) == ~0ull) { vals[s] = i; return; }
}
__device__ __forceinline__ int vhash_find(const uint64_t* keys, const int* vals, uint64_t mask, int x, int y, int z, int R) {
  if (x < 0 || y < 0 || z < 0 || x >= R || y >= R || z >= R) return -1;
  uint64_t k = ((uint64_t)x << 42) | ((uint64_t)y << 21) | (uint64_t)z;
  for (uint64_t s = mix64(k) & mask; keys[s] != ~0ull; s = (s + 1) & mask) if (keys[s] == k) return vals[s];
  return -1;
}
__constant__ int c_edge_off[3][4][3] = {
  {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}},
  {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}},
  {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}},
};
// vertices = (coords + (2*sigmoid(h[:, :3]) - 0.5)) * voxel_size + aabb_min ; w = softplus(h[:, 6])
__global__ void k_fdg_verts(const float* h, const int4* c, int n, float vs, float3* v, float* w) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  int4 q = c[i]; int qc[3] = {q.y, q.z, q.w}; float o[3];
  for (int a = 0; a < 3; a++) {
    float s = 1.f / (1.f + expf(-h[i * 7 + a]));
    float d = __fsub_rn(__fmul_rn(2.f, s), 0.5f);
    o[a] = __fadd_rn(__fmul_rn(__fadd_rn((float)qc[a], d), vs), -0.5f);
  }
  v[i] = make_float3(o[0], o[1], o[2]);
  float x = h[i * 7 + 6];
  w[i] = x > 20.f ? x : log1pf(expf(x));
}
__global__ void k_fdg_quads(const float* h, const int4* c, int n, int R, const uint64_t* keys, const int* vals, uint64_t mask, int4* quad, int* ok) {
  int e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= n * 3) return;
  int i = e / 3, a = e % 3; int4 q = c[i]; int r[4]; bool g = h[i * 7 + 3 + a] > 0.f;
  for (int k = 0; k < 4 && g; k++) {
    r[k] = vhash_find(keys, vals, mask, q.y + c_edge_off[a][k][0], q.z + c_edge_off[a][k][1], q.w + c_edge_off[a][k][2], R);
    g = r[k] >= 0;
  }
  ok[e] = g; if (g) quad[e] = make_int4(r[0], r[1], r[2], r[3]);
}
__global__ void k_fdg_tris(const int4* quad, const int* sel, int L, const float* w, int3* f) {
  int l = blockIdx.x * blockDim.x + threadIdx.x; if (l >= L) return;
  int4 q = quad[sel[l]];
  if (w[q.x] * w[q.z] > w[q.y] * w[q.w]) { f[2 * l] = make_int3(q.x, q.y, q.z); f[2 * l + 1] = make_int3(q.x, q.z, q.w); }
  else { f[2 * l] = make_int3(q.x, q.y, q.w); f[2 * l + 1] = make_int3(q.w, q.y, q.z); }
}

void fdg_to_mesh(const Tensor& h, const Tensor& coords, int R, Tensor& V, Tensor& F) {
  int n = (int)h.size(0);
  uint64_t cap = 1; while (cap < 2ull * n) cap <<= 1;
  Tensor keys = empty({(i64)cap}, I64), vals = empty({(i64)cap}, I32), w = empty({n}, F32);
  CK(cudaMemsetAsync(keys.p, 0xff, keys.bytes(), stream()));
  k_vhash_insert<<<cdiv(n, 256), 256, 0, stream()>>>(coords.ptr<int4>(), n, keys.ptr<uint64_t>(), vals.ptr<int>(), cap - 1);
  V = empty({n, 3}, F32);
  k_fdg_verts<<<cdiv(n, 256), 256, 0, stream()>>>(h.ptr<float>(), coords.ptr<int4>(), n, 1.f / R, V.ptr<float3>(), w.ptr<float>());
  Tensor quad = empty({(i64)n * 3, 4}, I32), ok = empty({(i64)n * 3}, I32), sel = empty({(i64)n * 3}, I32), cnt = empty({1}, I32);
  k_fdg_quads<<<cdiv(n * 3, 256), 256, 0, stream()>>>(h.ptr<float>(), coords.ptr<int4>(), n, R, keys.ptr<uint64_t>(), vals.ptr<int>(), cap - 1,
                                                    quad.ptr<int4>(), ok.ptr<int>());
  cub::CountingInputIterator<int> it(0); size_t tb = 0;
  cub::DeviceSelect::Flagged(nullptr, tb, it, ok.ptr<int>(), sel.ptr<int>(), cnt.ptr<int>(), n * 3, stream());
  Tensor tmp = empty({(i64)tb}, U8);
  cub::DeviceSelect::Flagged(tmp.p, tb, it, ok.ptr<int>(), sel.ptr<int>(), cnt.ptr<int>(), n * 3, stream());
  int L = to_host_vec<int>(cnt)[0];
  F = empty({2 * (i64)L, 3}, I32);
  if (L) k_fdg_tris<<<cdiv(L, 256), 256, 0, stream()>>>(quad.ptr<int4>(), sel.ptr<int>(), L, w.ptr<float>(), F.ptr<int3>());
}

// CuMesh runs on the legacy default stream: fence our stream before, the device after.
static void cm_init(cumesh::CuMesh& m, const Tensor& V, const Tensor& F) {
  dsync(); m.init(torch::wrap(V, torch::kFloat32), torch::wrap(F, torch::kInt32));
}
static void cm_read(cumesh::CuMesh& m, Tensor& V, Tensor& F) {
  auto [v, f] = m.read(); CK(cudaDeviceSynchronize()); V = v.t; F = f.t;
}

void fill_holes(Tensor& V, Tensor& F, float max_hole_perimeter) {
  cumesh::CuMesh m; cm_init(m, V, F);
  m.get_edges(); m.get_boundary_info();
  if (m.num_boundaries() == 0) return;
  m.get_vertex_edge_adjacency(); m.get_vertex_boundary_adjacency(); m.get_manifold_boundary_adjacency();
  m.read_manifold_boundary_adjacency();
  m.get_boundary_connected_components(); m.get_boundary_loops();
  if (m.num_boundary_loops() == 0) return;
  m.fill_holes(max_hole_perimeter);
  cm_read(m, V, F);
}
