// FlexGEMM submanifold conv3d (masked implicit GEMM split-K) re-implemented with mma.sync. Per output element the
// fp32 accumulation is the same chain of m16n8k16 fp16 MMAs (valid kernel offsets ascending, Ci ascending) over the
// same split-K ranges as the Triton kernel, so results match for a given (B1, BK, SPLITK) choice.
#include "spconv.h"
#include <cub/cub.cuh>
#include <fstream>
#include <regex>
#include <sstream>

static constexpr uint32_t EMPTY = 0xffffffffu;
static constexpr uint64_t HEMPTY = ~0ull;

__device__ __forceinline__ uint32_t hsh(uint64_t k) { k ^= k >> 33; k *= 0xff51afd7ed558ccdull; k ^= k >> 33; return (uint32_t)k; }
__global__ void k_hash_insert(const int4* c, int n, int W, int H, int D, uint64_t* keys, uint32_t* vals, uint32_t mask) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  int4 q = c[i]; uint64_t k = (((uint64_t)q.x * W + q.y) * H + q.z) * D + q.w;
  for (uint32_t s = hsh(k) & mask;; s = (s + 1) & mask) {
    uint64_t prev = atomicCAS((unsigned long long*)&keys[s], HEMPTY, k);
    if (prev == HEMPTY || prev == k) { vals[s] = i; return; }
  }
}
__global__ void k_neighbors(const int4* c, int n, int W, int H, int D, const uint64_t* keys, const uint32_t* vals, uint32_t mask,
                            uint32_t* nbr, int* gray, int* bin) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  int4 q = c[i]; uint32_t g = 0;
  for (int v = 0; v < 27; v++) {
    int x = q.y + v / 9 - 1, y = q.z + v / 3 % 3 - 1, z = q.w + v % 3 - 1; uint32_t r = EMPTY;
    if (x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D) {
      uint64_t k = (((uint64_t)q.x * W + x) * H + y) * D + z;
      for (uint32_t s = hsh(k) & mask; keys[s] != HEMPTY; s = (s + 1) & mask) if (keys[s] == k) { r = vals[s]; break; }
    }
    nbr[(i64)i * 27 + v] = r; if (r != EMPTY) g |= 1u << v;
  }
  uint32_t b = g; for (int v = 1; v < 27; v++) b ^= g >> v;
  gray[i] = (int)g; bin[i] = (int)b;
}

SpConvCache build_neighbors(const Tensor& coords, int W, int H, int D) {
  SpConvCache c; int n = (int)coords.size(0); c.N = n;
  uint32_t cap = 1; while (cap < 2u * n) cap <<= 1;
  Tensor keys = empty({(i64)cap}, I64), vals = empty({(i64)cap}, I32);
  CK(cudaMemsetAsync(keys.p, 0xff, keys.bytes(), stream()));
  k_hash_insert<<<cdiv(n, 256), 256, 0, stream()>>>(coords.ptr<int4>(), n, W, H, D, keys.ptr<uint64_t>(), vals.ptr<uint32_t>(), cap - 1);
  c.nbr = empty({n, 27}, I32); c.gray = empty({n}, I32); Tensor bin = empty({n}, I32), bs = empty({n}, I32), idx = empty({n}, I32);
  k_neighbors<<<cdiv(n, 256), 256, 0, stream()>>>(coords.ptr<int4>(), n, W, H, D, keys.ptr<uint64_t>(), vals.ptr<uint32_t>(), cap - 1,
                                                  c.nbr.ptr<uint32_t>(), c.gray.ptr<int>(), bin.ptr<int>());
  std::vector<int> ar(n); for (int i = 0; i < n; i++) ar[i] = i;
  Tensor iota = from_host(ar.data(), {n}, I32); c.sorted = empty({n}, I32);
  size_t tb = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, tb, bin.ptr<int>(), bs.ptr<int>(), iota.ptr<int>(), c.sorted.ptr<int>(), n, 0, 32, stream());
  Tensor tmp = empty({(i64)tb}, U8);
  cub::DeviceRadixSort::SortPairs(tmp.p, tb, bin.ptr<int>(), bs.ptr<int>(), iota.ptr<int>(), c.sorted.ptr<int>(), n, 0, 32, stream());
  return c;
}

__global__ void k_block_or(const int* gray, const int* sorted, int n, int B1, int* code, int* cnt) {
  int b = blockIdx.x * blockDim.x + threadIdx.x; if (b >= cdiv(n, B1)) return;
  int o = 0; for (int i = b * B1; i < min(n, (b + 1) * B1); i++) o |= gray[sorted[i]];
  code[b] = o; cnt[b] = __popc(o);
}
__global__ void k_block_list(const int* code, const int* seg, int nb, int* list) {
  int b = blockIdx.x * blockDim.x + threadIdx.x; if (b >= nb) return;
  int c = code[b], s = seg[b]; while (c) { int p = __ffs(c) - 1; list[s++] = p; c &= c - 1; }
}
const std::pair<Tensor, Tensor>& SpConvCache::valid(int B1) {
  auto it = vk.find(B1); if (it != vk.end()) return it->second;
  int nb = cdiv(N, B1); Tensor code = empty({nb}, I32), seg = zeros({nb + 1}, I32);
  k_block_or<<<cdiv(nb, 256), 256, 0, stream()>>>(gray.ptr<int>(), sorted.ptr<int>(), N, B1, code.ptr<int>(), seg.ptr<int>() + 1);
  size_t tb = 0; cub::DeviceScan::InclusiveSum(nullptr, tb, seg.ptr<int>() + 1, seg.ptr<int>() + 1, nb, stream());
  Tensor tmp = empty({(i64)tb}, U8); cub::DeviceScan::InclusiveSum(tmp.p, tb, seg.ptr<int>() + 1, seg.ptr<int>() + 1, nb, stream());
  int L = to_host_vec<int>(seg.slice0(nb, nb + 1))[0];
  Tensor list = empty({std::max(L, 1)}, I32);
  k_block_list<<<cdiv(nb, 256), 256, 0, stream()>>>(code.ptr<int>(), seg.ptr<int>(), nb, list.ptr<int>());
  return vk.emplace(B1, std::make_pair(list, seg)).first->second;
}

// ---------------------------------------------------------------------------------------------- GEMM
// CTA: BM = 32*WM rows (sorted order, inside one B1 block) x 64 out channels, 4 warps, K in 32-channel chunks.
namespace {
constexpr int BN = 64, KC = 32, LDS = KC + 8;
__device__ __forceinline__ void ldsm4(uint32_t* r, const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma16816(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
}  // namespace

template <int WM> __global__ void __launch_bounds__(128) k_spconv(const f16* __restrict__ x, const f16* __restrict__ w, const f16* __restrict__ bias,
    const uint32_t* __restrict__ nbr, const int* __restrict__ sorted, const int* __restrict__ list, const int* __restrict__ seg,
    void* out, int N, int Ci, int Co, int B1, int BK, int S) {
  constexpr int BM = 32 * WM, WN = 4 / WM, NI = BN / WN / 8, RA = BM / 32;
  __shared__ __align__(16) f16 As[2][BM][LDS], Bs[2][BN][LDS];
  int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, wm = wid % WM, wn = wid / WM;
  int row0 = blockIdx.x * BM, co0 = blockIdx.y * BN, s = blockIdx.z, blk = row0 / B1;
  int vs = seg[blk], vl = seg[blk + 1] - vs, nk = Ci / BK;
  int k0 = cdiv((i64)nk * vl * s, S) * (BK / KC), k1 = cdiv((i64)nk * vl * (s + 1), S) * (BK / KC), cpv = Ci / KC;
  int rows[RA]; for (int j = 0; j < RA; j++) { int r = row0 + (tid + j * 128) / 4; rows[j] = r < N ? sorted[r] : 0; }
  float acc[2][NI][4] = {};
  uint4 ra[RA], rb[2];
  auto gload = [&](int kc) {
    int v = list[vs + kc / cpv], ci = (kc % cpv) * KC;
    for (int j = 0; j < RA; j++) {
      int c8 = (tid % 4) * 8; uint32_t src = nbr[(i64)rows[j] * 27 + v];
      ra[j] = src == EMPTY ? make_uint4(0, 0, 0, 0) : *(const uint4*)(x + (i64)src * Ci + ci + c8);
    }
    for (int j = 0; j < 2; j++) {
      int q = tid + j * 128, r = q / 4, c8 = (q % 4) * 8, co = (co0 + r) % Co;
      rb[j] = *(const uint4*)(w + ((i64)co * 27 + v) * Ci + ci + c8);
    }
  };
  auto sstore = [&](int buf) {
    for (int j = 0; j < RA; j++) { int q = tid + j * 128; *(uint4*)&As[buf][q / 4][(q % 4) * 8] = ra[j]; }
    for (int j = 0; j < 2; j++) { int q = tid + j * 128; *(uint4*)&Bs[buf][q / 4][(q % 4) * 8] = rb[j]; }
  };
  if (k0 < k1) { gload(k0); sstore(0); }
  __syncthreads();
  for (int kc = k0, buf = 0; kc < k1; kc++, buf ^= 1) {
    if (kc + 1 < k1) gload(kc + 1);
#pragma unroll
    for (int kk = 0; kk < KC; kk += 16) {
      uint32_t a[2][4], b[NI][2];
      for (int mi = 0; mi < 2; mi++) ldsm4(a[mi], &As[buf][wm * 32 + mi * 16 + (lane & 15)][kk + (lane >> 4) * 8]);
      for (int nj = 0; nj < NI / 2; nj++) {
        uint32_t t[4]; ldsm4(t, &Bs[buf][wn * NI * 8 + nj * 16 + (lane & 7) + (lane >> 4) * 8][kk + ((lane >> 3) & 1) * 8]);
        b[nj * 2][0] = t[0]; b[nj * 2][1] = t[1]; b[nj * 2 + 1][0] = t[2]; b[nj * 2 + 1][1] = t[3];
      }
      for (int mi = 0; mi < 2; mi++) for (int ni = 0; ni < NI; ni++) mma16816(acc[mi][ni], a[mi], b[ni]);
    }
    if (kc + 1 < k1) sstore(buf ^ 1);
    __syncthreads();
  }
  for (int mi = 0; mi < 2; mi++) for (int ni = 0; ni < NI; ni++) for (int h = 0; h < 2; h++) {
    int r = row0 + wm * 32 + mi * 16 + (lane >> 2) + h * 8; if (r >= N) continue;
    int orow = sorted[r];
    for (int e = 0; e < 2; e++) {
      int co = co0 + wn * NI * 8 + ni * 8 + (lane & 3) * 2 + e; if (co >= Co) continue;
      float v = acc[mi][ni][h * 2 + e];
      if (S == 1) {  // non-split Triton kernel: round to fp16, then fp16 bias add
        f16 c = __float2half_rn(v); if (bias) c = __hadd(c, bias[co]);
        ((f16*)out)[(i64)orow * Co + co] = c;
      } else {       // split kernel: fp32 bias add in split 0
        if (bias && s == 0) v = v + __half2float(bias[co]);
        ((float*)out)[((i64)s * N + orow) * Co + co] = v;
      }
    }
  }
}

// ATen sum(dim=0) of contiguous [S, M] fp32 (outer reduction: 4 interleaved accumulators, combined left to right) -> fp16
__global__ void k_splitk_sum(const float* p, int S, i64 M, f16* y) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= M) return;
  float a[4] = {0.f, 0.f, 0.f, 0.f}; for (int s = 0; s < S; s++) a[s & 3] = a[s & 3] + p[s * M + i];
  y[i] = __float2half_rn(((a[0] + a[1]) + a[2]) + a[3]);
}

SpConvCfg spconv_cfg(int N, int Ci, int Co);
Tensor subm_conv(const Tensor& x, const Tensor& w, const Tensor* b, SpConvCache& c) {
  int N = (int)x.size(0), Ci = (int)x.size(1), Co = (int)w.size(0);
  REQ(x.dt == F16 && w.dt == F16 && w.size(1) == 27 && w.size(2) == Ci && Ci % KC == 0 && N == c.N, "spconv shapes");
  SpConvCfg cfg = spconv_cfg(N, Ci, Co);
  int WM = cfg.B1 >= 64 ? 2 : 1;
  REQ(cfg.B1 % (32 * WM) == 0 && cfg.BK % KC == 0 && Ci % cfg.BK == 0, "spconv cfg B1=%d BK=%d Ci=%d", cfg.B1, cfg.BK, Ci);
  auto& [list, seg] = c.valid(cfg.B1);
  Tensor bb; if (b) bb = b->dt == F16 ? *b : cast(*b, F16);
  Tensor y = empty({N, Co}, F16), part = cfg.S > 1 ? empty({cfg.S, N, Co}, F32) : Tensor();
  dim3 grid(cdiv(N, 32 * WM), cdiv(Co, BN), cfg.S);
  auto k = WM == 2 ? k_spconv<2> : k_spconv<1>;
  k<<<grid, 128, 0, stream()>>>(x.ptr<f16>(), w.ptr<f16>(), b ? bb.ptr<f16>() : nullptr, c.nbr.ptr<uint32_t>(), c.sorted.ptr<int>(),
                                list.ptr<int>(), seg.ptr<int>(), cfg.S > 1 ? part.p : y.p, N, Ci, Co, cfg.B1, cfg.BK, cfg.S);
  if (cfg.S > 1) k_splitk_sum<<<cdiv((i64)N * Co, 256), 256, 0, stream()>>>(part.ptr<float>(), cfg.S, (i64)N * Co, y.ptr<f16>());
  return y;
}

// ---------------------------------------------------------------------------------------------- autotune table
// Replays FlexGEMM's persisted choices (assets/flexgemm_autotune.json, or T2_FLEXGEMM_CACHE). The reference must run
// with the same file (FLEX_GEMM_AUTOTUNE_CACHE_PATH) for bitwise parity; missing entries fall back to B1=128, BK=64.
static std::string section(const std::string& s, const std::string& name) {
  auto p = s.find("\"" + name + "\""); if (p == std::string::npos) return "";
  p = s.find('{', p); int d = 0; size_t q = p;
  for (; q < s.size(); q++) { if (s[q] == '{') d++; else if (s[q] == '}' && --d == 0) break; }
  return s.substr(p, q - p + 1);
}
SpConvCfg spconv_cfg(int N, int Ci, int Co) {
  static std::map<std::tuple<int, int, int>, int> split;              // (LOGN, Ci, Co) -> SPLITK
  static std::map<std::tuple<int, int, int, int>, std::pair<int, int>> tri;  // (LOGN, Ci, Co, S) -> (B1, BK)
  static bool loaded = false;
  if (!loaded) {
    loaded = true;
    std::string path = env("T2_FLEXGEMM_CACHE", T2_FLEXGEMM_CACHE_DEFAULT);
    std::ifstream f(path); std::stringstream ss; ss << f.rdbuf();
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    std::string s = section(ss.str(), prop.name);
    const std::string M = "flex_gemm.kernels.triton.spconv.sparse_submanifold_conv_fwd_masked_implicit_gemm";
    std::smatch m;
    std::string a = section(s, M + "_splitk.sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk");
    std::regex rs(R"re("\(2\^(\d+), (\d+), (\d+), 27\)":\s*\{\s*"SPLITK":\s*(\d+))re");
    for (auto it = std::sregex_iterator(a.begin(), a.end(), rs); it != std::sregex_iterator(); ++it)
      split[{std::stoi((*it)[1]), std::stoi((*it)[2]), std::stoi((*it)[3])}] = std::stoi((*it)[4]);
    std::regex rk(R"re("\((\d+), (\d+), (\d+), 27, (?:(\d+), )?True[^"]*":\s*\{\s*"kwargs":\s*\{\s*"B1":\s*(\d+),\s*"B2":\s*\d+,\s*"BK":\s*(\d+))re");
    for (auto nm : {M + "_splitk.sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk_kernel",
                    M + ".sparse_submanifold_conv_fwd_masked_implicit_gemm_kernel"}) {
      std::string b = section(s, nm);
      for (auto it = std::sregex_iterator(b.begin(), b.end(), rk); it != std::sregex_iterator(); ++it) {
        int S = (*it)[4].matched ? std::stoi((*it)[4]) : 1;
        tri[{std::stoi((*it)[1]), std::stoi((*it)[2]), std::stoi((*it)[3]), S}] = {std::stoi((*it)[5]), std::stoi((*it)[6])};
      }
    }
    if (env("T2_VERBOSE") == "1") fprintf(stderr, "[spconv] %zu split entries, %zu kernel entries from %s\n", split.size(), tri.size(), path.c_str());
  }
  int L = 0; while ((2 << L) <= N) L++;
  auto si = split.find({L, Ci, Co});
  int S = si == split.end() ? 1 : si->second;
  auto ti = tri.find({L, Ci, Co, S});
  SpConvCfg c{128, 64, S};
  if (ti != tri.end()) { c.B1 = ti->second.first; c.BK = ti->second.second; }
  else if (env("T2_STRICT") == "1") REQ(false, "no autotune entry for N=2^%d Ci=%d Co=%d S=%d", L, Ci, Co, S);
  return c;
}
