// cv::inpaint(src, mask, dst, radius, INPAINT_TELEA) for 8-bit 1/3-channel images (OpenCV 5.0.0 photo/inpaint.cpp), per channel.
// Unqualified sqrt(float) resolves to std::sqrt(float) (cvstd.hpp `using std::sqrt` + `using namespace cv`), fabs to ::fabs(double).
// Build without FP contraction.
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <queue>
#include <vector>

namespace {
enum : uint8_t { KNOWN = 0, BAND = 1, INSIDE = 2, CHANGE = 3 };
struct HE { float T; int i, j, order; bool operator>(const HE& r) const { return T > r.T || (T == r.T && order > r.order); } };
struct Heap {
  std::priority_queue<HE, std::vector<HE>, std::greater<HE>> q; int next = 0;
  void push(int i, int j, float T) { q.push({T, i, j, next++}); }
  bool pop(int& i, int& j) { if (q.empty()) return false; i = q.top().i; j = q.top().j; q.pop(); return true; }
};
struct M8 { int R, C; std::vector<uint8_t> d; uint8_t& operator()(int i, int j) { return d[(size_t)i * C + j]; } };


float solve(int i1, int j1, int i2, int j2, M8& f, const std::vector<float>& t, int C) {
  double a11 = t[(size_t)i1 * C + j1], a22 = t[(size_t)i2 * C + j2], m12 = a11 < a22 ? a11 : a22, sol;
  if (f(i1, j1) != INSIDE)
    if (f(i2, j2) != INSIDE) sol = std::fabs(a11 - a22) >= 1.0 ? 1 + m12 : (a11 + a22 + std::sqrt((double)(2 - (a11 - a22) * (a11 - a22)))) * 0.5;
    else sol = 1 + a11;
  else if (f(i2, j2) != INSIDE) sol = 1 + a22;
  else sol = 1 + m12;
  return (float)sol;
}
inline float min4(float a, float b, float c, float d) { a = b < a ? b : a; c = d < c ? d : c; return c < a ? c : a; }
inline float dist4(int i, int j, M8& f, const std::vector<float>& t, int C) {
  return min4(solve(i - 1, j, i, j - 1, f, t, C), solve(i + 1, j, i, j - 1, f, t, C), solve(i - 1, j, i, j + 1, f, t, C), solve(i + 1, j, i, j + 1, f, t, C));
}
const int DI[4] = {-1, 0, 1, 0}, DJ[4] = {0, -1, 0, 1};

void calc_fmm(M8& f, std::vector<float>& t, Heap& h) {
  int ii, jj;
  while (h.pop(ii, jj)) {
    f(ii, jj) = CHANGE;
    for (int q = 0; q < 4; q++) {
      int i = ii + DI[q], j = jj + DJ[q];
      if (i <= 0 || j <= 0 || i > f.R || j > f.C) continue;
      if (f(i, j) == INSIDE) { float d = dist4(i, j, f, t, f.C); t[(size_t)i * f.C + j] = d; f(i, j) = BAND; h.push(i, j, d); }
    }
  }
  for (size_t k = 0; k < f.d.size(); k++) if (f.d[k] == CHANGE) { f.d[k] = KNOWN; t[k] = -t[k]; }
}

// One channel; D3: the 3-channel variant's weight (per-color math is otherwise identical, so channels run independently)
template <bool D3> void telea(M8& f, std::vector<float>& t, uint8_t* out, int range, Heap& h) {
  const int R = f.R, C = f.C, W = C - 2; int ii, jj;
  auto T = [&](int i, int j) { return t[(size_t)i * C + j]; };
  auto O = [&](int i, int j) -> uint8_t& { return out[(size_t)i * W + j]; };
  while (h.pop(ii, jj)) {
    f(ii, jj) = KNOWN;
    for (int q = 0; q < 4; q++) {
      int i = ii + DI[q], j = jj + DJ[q];
      if (i <= 0 || j <= 0 || i > R - 1 || j > C - 1) continue;
      if (f(i, j) != INSIDE) continue;
      float dist = dist4(i, j, f, t, C); t[(size_t)i * C + j] = dist;
      float gx, gy;
      if (f(i, j + 1) != INSIDE) gx = f(i, j - 1) != INSIDE ? (float)(T(i, j + 1) - T(i, j - 1)) * 0.5f : (float)(T(i, j + 1) - T(i, j));
      else gx = f(i, j - 1) != INSIDE ? (float)(T(i, j) - T(i, j - 1)) : 0.f;
      if (f(i + 1, j) != INSIDE) gy = f(i - 1, j) != INSIDE ? (float)(T(i + 1, j) - T(i - 1, j)) * 0.5f : (float)(T(i + 1, j) - T(i, j));
      else gy = f(i - 1, j) != INSIDE ? (float)(T(i, j) - T(i - 1, j)) : 0.f;
      float Jx = 0, Jy = 0, Ia = 0, s = 1.0e-20f;
      for (int k = i - range; k <= i + range; k++) {
        int km = k - 1 + (k == 1), kp = k - 1 - (k == R - 2);
        for (int l = j - range; l <= j + range; l++) {
          int lm = l - 1 + (l == 1), lp = l - 1 - (l == C - 2);
          if (!(k > 0 && l > 0 && k < R - 1 && l < C - 1)) continue;
          if (f(k, l) == INSIDE || (l - j) * (l - j) + (k - i) * (k - i) > range * range) continue;
          {
            float ry = (float)(i - k), rx = (float)(j - l), vl = rx * rx + ry * ry;
            float dst = D3 ? (float)(1. / (vl * std::sqrt((double)vl))) : (float)(1. / (vl * std::sqrt(vl)));
            float lev = (float)(1. / (1 + std::fabs((double)(T(k, l) - T(i, j)))));
            float dir = rx * gx + ry * gy;
            if (std::fabs(dir) <= 0.01) dir = 0.000001f;
            float w = (float)std::fabs(dst * lev * dir), ix, iy;
            if (f(k, l + 1) != INSIDE) ix = f(k, l - 1) != INSIDE ? (float)(O(km, lp + 1) - O(km, lm - 1)) * 2.0f : (float)(O(km, lp + 1) - O(km, lm));
            else ix = f(k, l - 1) != INSIDE ? (float)(O(km, lp) - O(km, lm - 1)) : 0.f;
            if (f(k + 1, l) != INSIDE) iy = f(k - 1, l) != INSIDE ? (float)(O(kp + 1, lm) - O(km - 1, lm)) * 2.0f : (float)(O(kp + 1, lm) - O(km, lm));
            else iy = f(k - 1, l) != INSIDE ? (float)(O(kp, lm) - O(km - 1, lm)) : 0.f;
            Ia += w * (float)O(k - 1, l - 1);
            Jx -= w * (ix * rx);
            Jy -= w * (iy * ry);
            s += w;
          }
        }
      }
      float sat = (float)(Ia / s + (Jx + Jy) / (std::sqrt(Jx * Jx + Jy * Jy) + 1.0e-20f));
      double v = std::nearbyint((double)sat + 0.5);
      O(i - 1, j - 1) = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
      f(i, j) = BAND; h.push(i, j, dist);
    }
  }
}

M8 dilate(M8& m, int r) {  // MORPH_RECT (2r+1)^2 (r=1 with cross when `cross`), constant border = no contribution
  M8 o{m.R, m.C, std::vector<uint8_t>(m.d.size())}; std::vector<uint8_t> tmp(m.d.size());
  for (int i = 0; i < m.R; i++) for (int j = 0; j < m.C; j++) {
    uint8_t v = 0; for (int l = std::max(0, j - r); l <= std::min(m.C - 1, j + r); l++) v = std::max(v, m(i, l)); tmp[(size_t)i * m.C + j] = v;
  }
  for (int i = 0; i < m.R; i++) for (int j = 0; j < m.C; j++) {
    uint8_t v = 0; for (int k = std::max(0, i - r); k <= std::min(m.R - 1, i + r); k++) v = std::max(v, tmp[(size_t)k * m.C + j]); o(i, j) = v;
  }
  return o;
}
M8 dilate_cross(M8& m) {
  M8 o{m.R, m.C, m.d};
  for (int i = 0; i < m.R; i++) for (int j = 0; j < m.C; j++) {
    uint8_t v = m(i, j);
    if (i > 0) v = std::max(v, m(i - 1, j)); if (i < m.R - 1) v = std::max(v, m(i + 1, j));
    if (j > 0) v = std::max(v, m(i, j - 1)); if (j < m.C - 1) v = std::max(v, m(i, j + 1));
    o(i, j) = v;
  }
  return o;
}
void sub_(M8& a, const M8& b) { for (size_t k = 0; k < a.d.size(); k++) a.d[k] = a.d[k] > b.d[k] ? a.d[k] - b.d[k] : 0; }
void border0(M8& a) {
  for (int j = 0; j < a.C; j++) a(0, j) = a(a.R - 1, j) = 0;
  for (int i = 0; i < a.R; i++) a(i, 0) = a(i, a.C - 1) = 0;
}
}  // namespace

// In place on one channel plane [H, W]; inpaint where mask != 0. rgb: plane belongs to a 3-channel image.
void inpaint_telea(uint8_t* img, int H, int W, bool rgb, const uint8_t* mask_in, double radius) {
  int range = (int)std::nearbyint(radius); range = std::min(std::max(range, 1), 100);
  int R = H + 2, C = W + 2;
  M8 mask{R, C, std::vector<uint8_t>((size_t)R * C, KNOWN)};
  for (int i = 0; i < H; i++) for (int j = 0; j < W; j++) if (mask_in[(size_t)i * W + j]) mask(i + 1, j + 1) = INSIDE;
  border0(mask);
  std::vector<float> t((size_t)R * C, 1.0e6f);
  M8 band = dilate_cross(mask); sub_(band, mask); border0(band);
  Heap heap, outh;
  for (int i = 0; i < R; i++) for (int j = 0; j < C; j++) if (band(i, j)) { heap.push(i, j, 0); t[(size_t)i * C + j] = 0; }
  M8 out = dilate(mask, range); sub_(out, mask);
  for (int i = 0; i < R; i++) for (int j = 0; j < C; j++) if (band(i, j)) outh.push(i, j, 0);
  sub_(out, band); border0(out);
  calc_fmm(out, t, outh);
  rgb ? telea<true>(mask, t, img, range, heap) : telea<false>(mask, t, img, range, heap);
}
