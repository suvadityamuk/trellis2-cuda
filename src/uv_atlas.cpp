// Fast-build UV atlas: CuMesh charts -> xatlas parameterization + packing, run as many independent group atlases in
// parallel native threads, then tiled into one atlas.
//
// The official uv_unwrap adds every CuMesh chart to a single xatlas Atlas and packs them serially (4096 random placement
// attempts per chart into one growing bitmap). Here charts above `max_faces` faces are cut by recursive centroid-median
// splits, the pieces are dealt round-robin by area into groups, and every group runs AddMesh/ComputeCharts/PackCharts on
// its own thread with one shared texelsPerUnit (xatlas's own 1024^2-at-75%-utilization estimate over the whole mesh), so
// texel density is uniform. The group atlases are then shelf-packed into one atlas and UVs are normalized by its size.
#include "xatlas.h"
#include "xa_hw.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace xa = cumesh_xatlas;
namespace {
struct Piece { std::vector<float> P; std::vector<uint32_t> I; std::vector<int32_t> gmap; double area = 0; };
struct GroupOut { std::vector<int32_t> vmap, faces; std::vector<float> uv; int w = 0, h = 0; std::string err; };
template <class Fn> void par_for(int64_t n, int threads, Fn fn) {  // fn(i), dynamic schedule
  std::atomic<int64_t> next{0}; std::vector<std::thread> pool;
  int T = (int)std::max<int64_t>(1, std::min<int64_t>(threads, n));
  auto work = [&] { for (int64_t i; (i = next.fetch_add(1)) < n;) fn(i); };
  for (int t = 1; t < T; t++) pool.emplace_back(work);
  work();
  for (auto& th : pool) th.join();
}
}  // namespace

// cv: [V, 3] chart vertices (concatenated), cf: [F, 3] faces indexing cv rows (as CuMesh read_atlas_charts returns them),
// vo/fo: [nc + 1] chart vertex/face offsets. -> vmap [V'] into cv rows, faces [F', 3], uvs [V', 2] in [0, 1].
// info: pieces, groups, atlas width, atlas height, xatlas charts.
void uv_pack_groups(const std::vector<float>& cv, const std::vector<int>& cf, const std::vector<int>& vo, const std::vector<int>& fo,
                    int ngroups, int nthreads, int max_faces, std::vector<int>& vmap, std::vector<int>& faces, std::vector<float>& uvs, int info[5]) {
  const int64_t nc = (int64_t)vo.size() - 1;
  const float* P = cv.data(); const int32_t* Fi = cf.data();
  if (nthreads <= 0) nthreads = (int)xa_hw_real();
  const char* pe = std::getenv("T2_PROF"); const bool pon = pe && pe[0] == '1';
  auto now = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
  double t0 = now();
  auto lap = [&](const char* n) { if (pon) { double x = now(); std::printf("      %-22s %8.1f ms\n", n, x - t0); t0 = x; } };

  std::vector<std::vector<Piece>> per_chart(nc);
  const int64_t CH = 16, nchunk = (nc + CH - 1) / CH;
  par_for(nchunk, nthreads, [&](int64_t ci) {
    std::vector<int32_t> loc;
    for (int64_t c = ci * CH; c < std::min(nc, (ci + 1) * CH); c++) {
      const int64_t nf = fo[c + 1] - fo[c];
      std::vector<std::vector<int64_t>> leaves;
      std::vector<int64_t> all(nf); std::iota(all.begin(), all.end(), (int64_t)fo[c]);
      if (max_faces <= 0 || nf <= max_faces) leaves.push_back(std::move(all));
      else {
        std::vector<std::vector<int64_t>> stack; stack.push_back(std::move(all));
        auto cen = [&](int64_t f, int k) { return P[3 * (int64_t)Fi[3 * f] + k] + P[3 * (int64_t)Fi[3 * f + 1] + k] + P[3 * (int64_t)Fi[3 * f + 2] + k]; };
        while (!stack.empty()) {
          auto cur = std::move(stack.back()); stack.pop_back();
          if ((int64_t)cur.size() <= max_faces) { leaves.push_back(std::move(cur)); continue; }
          float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
          for (int64_t f : cur) for (int k = 0; k < 3; k++) { float x = cen(f, k); lo[k] = std::min(lo[k], x); hi[k] = std::max(hi[k], x); }
          int ax = 0; for (int k = 1; k < 3; k++) if (hi[k] - lo[k] > hi[ax] - lo[ax]) ax = k;
          auto mid = cur.begin() + cur.size() / 2;
          std::nth_element(cur.begin(), mid, cur.end(), [&](int64_t x, int64_t y) { float a = cen(x, ax), b = cen(y, ax); return a < b || (a == b && x < y); });
          stack.emplace_back(cur.begin(), mid); stack.emplace_back(mid, cur.end());
        }
      }
      loc.assign(vo[c + 1] - vo[c], -1);
      for (auto& lf : leaves) {
        Piece pc;
        for (int64_t f : lf) {
          for (int k = 0; k < 3; k++) {
            int32_t gv = Fi[3 * f + k]; int32_t& l = loc[gv - vo[c]];
            if (l < 0) { l = (int32_t)pc.gmap.size(); pc.gmap.push_back(gv); pc.P.insert(pc.P.end(), P + 3 * (int64_t)gv, P + 3 * (int64_t)gv + 3); }
            pc.I.push_back((uint32_t)l);
          }
          const float* p0 = P + 3 * (int64_t)Fi[3 * f]; const float* p1 = P + 3 * (int64_t)Fi[3 * f + 1]; const float* p2 = P + 3 * (int64_t)Fi[3 * f + 2];
          double ux = p1[0] - p0[0], uy = p1[1] - p0[1], uz = p1[2] - p0[2], vx = p2[0] - p0[0], vy = p2[1] - p0[1], vz = p2[2] - p0[2];
          double cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
          pc.area += 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
        }
        for (int32_t gv : pc.gmap) loc[gv - vo[c]] = -1;
        per_chart[c].push_back(std::move(pc));
      }
    }
  });
  std::vector<Piece> pieces;
  for (auto& v : per_chart) for (auto& pc : v) pieces.push_back(std::move(pc));
  per_chart.clear();
  const int64_t np = (int64_t)pieces.size();
  lap("xa split");

  // Shared texelsPerUnit: xatlas's packCharts estimate (resolution 1024 at 75% utilization) over the whole mesh.
  double total = 0; for (auto& pc : pieces) total += pc.area;
  const float tpu = (float)std::sqrt((1024.0 * 1024.0) / std::max(1.0, total / 0.75));

  // Deal pieces to groups round-robin in decreasing area order (balances both area and count).
  int G = (int)std::max<int64_t>(1, std::min<int64_t>(ngroups, np));
  std::vector<int64_t> order(np); std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return pieces[a].area > pieces[b].area; });
  std::vector<std::vector<int64_t>> groups(G);
  for (int64_t i = 0; i < np; i++) groups[i % G].push_back(order[i]);

  std::vector<GroupOut> out(G); std::atomic<unsigned> ncharts{0};
  std::vector<double> tg(G * 4, 0.0);
  xa_hw_override.store(2);  // one worker per Atlas (xatlas needs >= 2: its single worker uses thread index 1)
  par_for(G, nthreads, [&](int64_t g) {
    auto& o = out[g]; double ta = now();
    xa::Atlas* atlas = xa::Create();
    for (int64_t pi : groups[g]) {
      const Piece& pc = pieces[pi];
      xa::MeshDecl d;
      d.vertexCount = (uint32_t)pc.gmap.size(); d.vertexPositionData = pc.P.data(); d.vertexPositionStride = sizeof(float) * 3;
      d.indexCount = (uint32_t)pc.I.size(); d.indexData = pc.I.data(); d.indexFormat = xa::IndexFormat::UInt32;
      auto r = xa::AddMesh(atlas, d, (uint32_t)groups[g].size());
      if (r != xa::AddMeshError::Success) { o.err = xa::StringForEnum(r); break; }
    }
    if (o.err.empty()) {
      xa::ChartOptions co;  // CuMesh Atlas.compute_charts defaults == xatlas defaults
      double tb = now(); tg[g * 4] = tb - ta;
      xa::ComputeCharts(atlas, co);
      double tc = now(); tg[g * 4 + 1] = tc - tb;
      xa::PackOptions po;   // CuMesh Atlas.pack_charts defaults, with the shared texel scale
      po.texelsPerUnit = tpu;
      xa::PackCharts(atlas, po);
      tg[g * 4 + 2] = now() - tc;
      o.w = (int)atlas->width; o.h = (int)atlas->height; ncharts += atlas->chartCount;
      for (uint32_t m = 0; m < atlas->meshCount; m++) {
        const auto& mesh = atlas->meshes[m];
        int32_t base = (int32_t)o.vmap.size();
        const Piece& pc = pieces[groups[g][m]];
        for (uint32_t i = 0; i < mesh.vertexCount; i++) {
          const auto& v = mesh.vertexArray[i];
          o.vmap.push_back(pc.gmap[v.xref]); o.uv.push_back(v.uv[0]); o.uv.push_back(v.uv[1]);
        }
        for (uint32_t i = 0; i < mesh.indexCount; i++) o.faces.push_back(base + (int32_t)mesh.indexArray[i]);
      }
    }
    double td = now(); xa::Destroy(atlas); tg[g * 4 + 3] = now() - td;
  });
  xa_hw_override.store(0);
  if (pon) {
    double sm[4] = {}, mx[4] = {}; for (int g = 0; g < G; g++) for (int k = 0; k < 4; k++) { sm[k] += tg[g * 4 + k]; mx[k] = std::max(mx[k], tg[g * 4 + k]); }
    std::printf("      per-group sum/max ms: add %.0f/%.0f charts %.0f/%.0f pack %.0f/%.0f destroy %.0f/%.0f (%d threads)\n",
                sm[0], mx[0], sm[1], mx[1], sm[2], mx[2], sm[3], mx[3], nthreads);
  }
  lap("xa groups");
  for (auto& o : out) if (!o.err.empty()) throw std::runtime_error("xatlas AddMesh failed: " + o.err);

  // Shelf-pack the group atlases (tallest first), 1 texel apart. UVs are normalized by the final W x H, so texel density
  // scales with 1/sqrt(W*H): try a range of shelf widths and keep the smallest area.
  std::vector<int> gord(G); std::iota(gord.begin(), gord.end(), 0);
  std::stable_sort(gord.begin(), gord.end(), [&](int a, int b) { return out[a].h != out[b].h ? out[a].h > out[b].h : out[a].w > out[b].w; });
  double sumA = 0; int maxW = 0;
  for (auto& o : out) { sumA += (double)(o.w + 1) * (o.h + 1); maxW = std::max(maxW, o.w + 1); }
  std::vector<int> ox(G), oy(G), bx(G), by(G); int W = 0, H = 0; double bestA = 1e300;
  for (int k = 0; k <= 64; k++) {
    int rowW = std::max(maxW, (int)std::ceil(std::sqrt(sumA) * (0.7 + 0.02 * k)));
    int x = 0, y = 0, rowH = 0, w = 0;
    for (int g : gord) {
      if (x > 0 && x + out[g].w + 1 > rowW) { y += rowH; x = 0; rowH = 0; }
      bx[g] = x; by[g] = y; x += out[g].w + 1; rowH = std::max(rowH, out[g].h + 1); w = std::max(w, x);
    }
    int h = y + rowH;
    if ((double)w * h < bestA) { bestA = (double)w * h; W = w; H = h; ox = bx; oy = by; }
  }

  int64_t NV = 0, NF = 0; std::vector<int64_t> vbase(G), fbase(G);
  for (int g = 0; g < G; g++) { vbase[g] = NV; fbase[g] = NF; NV += (int64_t)out[g].vmap.size(); NF += (int64_t)out[g].faces.size() / 3; }
  vmap.resize(NV); faces.resize(NF * 3); uvs.resize(NV * 2);
  par_for(G, nthreads, [&](int64_t g) {
    const auto& o = out[g];
    for (size_t i = 0; i < o.vmap.size(); i++) {
      vmap[vbase[g] + i] = o.vmap[i];
      uvs[2 * (vbase[g] + i)] = (o.uv[2 * i] + ox[g]) / (float)W;
      uvs[2 * (vbase[g] + i) + 1] = (o.uv[2 * i + 1] + oy[g]) / (float)H;
    }
    for (size_t i = 0; i < o.faces.size(); i++) faces[3 * fbase[g] + i] = (int32_t)(o.faces[i] + vbase[g]);
  });
  lap("xa tile");
  if (info) { info[0] = (int)np; info[1] = G; info[2] = W; info[3] = H; info[4] = (int)ncharts.load(); }
}
