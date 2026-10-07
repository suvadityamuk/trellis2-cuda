#include "core.h"
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

size_t dsize(DT d) { static const size_t s[] = {4, 2, 2, 4, 8, 1, 8, 1}; return s[d]; }
const char* dname(DT d) { static const char* s[] = {"F32", "F16", "BF16", "I32", "I64", "U8", "F64", "I8"}; return s[d]; }

cudaStream_t stream() {
  static cudaStream_t s = [] {
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaMemPool_t pool; CK(cudaDeviceGetDefaultMemPool(&pool, 0));
    uint64_t thr = UINT64_MAX; CK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr));
    return s;
  }();
  return s;
}
void dsync() { CK(cudaStreamSynchronize(stream())); }

// 512B-aligned like the torch caching allocator; slack keeps vector loads in-bounds.
Tensor empty(std::vector<i64> sh, DT dt) {
  Tensor t; t.sh = std::move(sh); t.dt = dt;
  size_t n = (t.bytes() + 511) & ~size_t(511);
  if (n) {
    void* p; CK(cudaMallocAsync(&p, n, stream()));
    t.p = p; t.own = std::shared_ptr<void>(p, [](void* q) { cudaFreeAsync(q, stream()); });
  }
  return t;
}
Tensor zeros(std::vector<i64> sh, DT dt) { auto t = empty(sh, dt); if (t.bytes()) CK(cudaMemsetAsync(t.p, 0, t.bytes(), stream())); return t; }
Tensor from_host(const void* src, std::vector<i64> sh, DT dt) {
  auto t = empty(sh, dt); if (t.bytes()) CK(cudaMemcpyAsync(t.p, src, t.bytes(), cudaMemcpyHostToDevice, stream())); dsync(); return t;
}
std::vector<char> to_host(const Tensor& t) {
  std::vector<char> v(t.bytes()); if (t.bytes()) CK(cudaMemcpyAsync(v.data(), t.p, t.bytes(), cudaMemcpyDeviceToHost, stream())); dsync(); return v;
}
Tensor clone(const Tensor& t) { auto o = empty(t.sh, t.dt); if (t.bytes()) CK(cudaMemcpyAsync(o.p, t.p, t.bytes(), cudaMemcpyDeviceToDevice, stream())); return o; }
Tensor Tensor::view(std::vector<i64> s) const {
  i64 n = 1, neg = -1; for (size_t i = 0; i < s.size(); i++) { if (s[i] < 0) neg = i; else n *= s[i]; }
  if (neg >= 0) s[neg] = numel() / n;
  Tensor t = *this; t.sh = s; REQ(t.numel() == numel(), "bad view"); return t;
}
Tensor Tensor::slice0(i64 a, i64 b) const {
  Tensor t = *this; t.sh[0] = b - a; t.p = (char*)p + a * (numel() / sh[0]) * dsize(dt); return t;
}

template <class A, class B> __device__ __forceinline__ B cvt(A a) { return (B)(float)a; }
template <> __device__ __forceinline__ float cvt<float, float>(float a) { return a; }
template <> __device__ __forceinline__ int cvt<float, int>(float a) { return (int)a; }
template <> __device__ __forceinline__ float cvt<int, float>(int a) { return (float)a; }
template <> __device__ __forceinline__ float cvt<i64, float>(i64 a) { return (float)a; }
template <> __device__ __forceinline__ int cvt<i64, int>(i64 a) { return (int)a; }
template <> __device__ __forceinline__ i64 cvt<int, i64>(int a) { return a; }
template <> __device__ __forceinline__ uint8_t cvt<float, uint8_t>(float a) { return (uint8_t)a; }
template <class A, class B> __global__ void k_cast(const A* a, B* b, i64 n) {
  i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i < n) b[i] = cvt<A, B>(a[i]);
}
template <class A> static void cast_from(const Tensor& t, Tensor& o) {
  i64 n = t.numel(); int g = cdiv(n, 256);
  switch (o.dt) {
    case F32: k_cast<A, float><<<g, 256, 0, stream()>>>(t.ptr<A>(), o.ptr<float>(), n); break;
    case BF16: k_cast<A, bf16><<<g, 256, 0, stream()>>>(t.ptr<A>(), o.ptr<bf16>(), n); break;
    case F16: k_cast<A, f16><<<g, 256, 0, stream()>>>(t.ptr<A>(), o.ptr<f16>(), n); break;
    default: REQ(false, "cast to %s", dname(o.dt));
  }
}
Tensor cast(const Tensor& t, DT dt) {
  if (t.dt == dt) return t;
  auto o = empty(t.sh, dt); if (!t.numel()) return o;
  switch (t.dt) {
    case F32: cast_from<float>(t, o); break;
    case BF16: cast_from<bf16>(t, o); break;
    case F16: cast_from<f16>(t, o); break;
    case I32: REQ(dt == F32, "i32 cast"); k_cast<int, float><<<cdiv(t.numel(), 256), 256, 0, stream()>>>(t.ptr<int>(), o.ptr<float>(), t.numel()); break;
    case U8: REQ(dt == F32, "u8 cast"); k_cast<uint8_t, float><<<cdiv(t.numel(), 256), 256, 0, stream()>>>(t.ptr<uint8_t>(), o.ptr<float>(), t.numel()); break;
    case I64: REQ(dt == F32, "i64 cast"); k_cast<i64, float><<<cdiv(t.numel(), 256), 256, 0, stream()>>>(t.ptr<i64>(), o.ptr<float>(), t.numel()); break;
    default: REQ(false, "cast from %s", dname(t.dt));
  }
  return o;
}

// ---------------- safetensors ----------------
static DT parse_dt(const std::string& s) {
  if (s == "F32") return F32; if (s == "F16") return F16; if (s == "BF16") return BF16; if (s == "I32") return I32;
  if (s == "I64") return I64; if (s == "U8" || s == "BOOL") return U8; if (s == "F64") return F64; if (s == "I8") return I8;
  REQ(false, "dtype %s", s.c_str()); return F32;
}
struct STEntry { std::string name; DT dt; std::vector<i64> sh; size_t a, b; };
// Minimal parser for the fixed safetensors header grammar.
static std::vector<STEntry> parse_header(const char* h, size_t n) {
  std::vector<STEntry> out; size_t i = 0;
  auto ws = [&] { while (i < n && (h[i] == ' ' || h[i] == '\n' || h[i] == '\t' || h[i] == '\r' || h[i] == ',' || h[i] == ':')) i++; };
  auto str = [&] { ws(); REQ(h[i] == '"', "json"); size_t s = ++i; while (h[i] != '"') i += h[i] == '\\' ? 2 : 1; return std::string(h + s, h + i++); };
  auto num = [&] { ws(); char* e; i64 v = strtoll(h + i, &e, 10); i = e - h; return v; };
  auto skip = [&] { ws(); int d = 0; do { if (h[i] == '{') d++; else if (h[i] == '}') d--; else if (h[i] == '"') { str(); continue; } i++; } while (d > 0); };
  ws(); i++;  // {
  while (true) {
    ws(); if (h[i] == '}') break;
    STEntry e; e.name = str(); ws();
    if (e.name == "__metadata__") { skip(); continue; }
    i++;  // {
    while (true) {
      ws(); if (h[i] == '}') { i++; break; }
      auto k = str(); ws();
      if (k == "dtype") e.dt = parse_dt(str());
      else if (k == "shape") { i++; while (ws(), h[i] != ']') e.sh.push_back(num()); i++; }
      else if (k == "data_offsets") { i++; e.a = num(); e.b = num(); ws(); i++; }
    }
    out.push_back(e);
  }
  return out;
}
struct Mapped { void* p; size_t n; ~Mapped() { munmap(p, n); } };
static std::shared_ptr<Mapped> map_file(const std::string& path) {
  int fd = open(path.c_str(), O_RDONLY); REQ(fd >= 0, "open %s", path.c_str());
  struct stat st; fstat(fd, &st); void* p = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
  REQ(p != MAP_FAILED, "mmap"); return std::shared_ptr<Mapped>(new Mapped{p, (size_t)st.st_size});
}
TensorMap load_safetensors_host(const std::string& path) {
  auto m = map_file(path); const char* base = (const char*)m->p;
  uint64_t hn; memcpy(&hn, base, 8); auto es = parse_header(base + 8, hn);
  TensorMap out;
  for (auto& e : es) { Tensor t; t.dt = e.dt; t.sh = e.sh; t.p = (void*)(base + 8 + hn + e.a); t.own = m; out[e.name] = t; }
  return out;
}
TensorMap load_safetensors(const std::string& path, const std::string& prefix) {
  auto m = map_file(path); const char* base = (const char*)m->p;
  uint64_t hn; memcpy(&hn, base, 8); auto es = parse_header(base + 8, hn);
  size_t tot = 0; for (auto& e : es) tot += (e.b - e.a + 511) & ~size_t(511);
  void* dev; CK(cudaMalloc(&dev, tot + 512)); std::shared_ptr<void> own(dev, [](void* q) { cudaFree(q); });
  TensorMap out; size_t off = 0;
  for (auto& e : es) {
    if (!prefix.empty() && e.name.rfind(prefix, 0) != 0) continue;
    Tensor t; t.dt = e.dt; t.sh = e.sh; t.p = (char*)dev + off; t.own = own;
    CK(cudaMemcpy(t.p, base + 8 + hn + e.a, e.b - e.a, cudaMemcpyHostToDevice));
    off += (e.b - e.a + 511) & ~size_t(511); out[e.name.substr(prefix.size())] = t;
  }
  return out;
}
void save_safetensors(const std::string& path, const TensorMap& m) {
  std::string h = "{"; size_t off = 0; std::vector<std::vector<char>> data;
  for (auto& [k, t] : m) {
    data.push_back(to_host(t));
    h += (h.size() > 1 ? "," : "") + std::string("\"") + k + "\":{\"dtype\":\"" + dname(t.dt) + "\",\"shape\":[";
    for (size_t i = 0; i < t.sh.size(); i++) h += (i ? "," : "") + std::to_string(t.sh[i]);
    h += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + data.back().size()) + "]}";
    off += data.back().size();
  }
  h += "}"; while (h.size() % 8) h += ' ';
  FILE* f = fopen(path.c_str(), "wb"); REQ(f, "open %s", path.c_str());
  uint64_t n = h.size(); fwrite(&n, 8, 1, f); fwrite(h.data(), 1, n, f);
  for (auto& d : data) fwrite(d.data(), 1, d.size(), f);
  fclose(f);
}

Timer::Timer() { cudaEventCreate(&a); cudaEventCreate(&b); }
void Timer::start() { cudaEventRecord(a, stream()); }
float Timer::ms() { cudaEventRecord(b, stream()); cudaEventSynchronize(b); float m; cudaEventElapsedTime(&m, a, b); return m; }
double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
std::string env(const char* k, const char* d) { const char* v = getenv(k); return v ? v : d; }
