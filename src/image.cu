// Image preprocessing replicating Trellis2ImageTo3DPipeline.preprocess_image / DinoV3FeatureExtractor input handling:
// numpy float32 premultiply, PIL crop (round-half-even box), PIL 8bpc LANCZOS resample (Resample.c), torchvision Normalize.
#include "models.h"
#include <cmath>
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include <stb_image.h>
#ifdef T2_WEBP
#include <webp/decode.h>
#endif

Image8 load_image(const std::string& path) {
  Image8 im; im.c = 4;
  if (path.size() > 5 && path.substr(path.size() - 5) == ".webp") {
#ifdef T2_WEBP
    FILE* f = fopen(path.c_str(), "rb"); REQ(f, "open %s", path.c_str());
    std::vector<uint8_t> buf; uint8_t tmp[1 << 16]; size_t n;
    while ((n = fread(tmp, 1, sizeof(tmp), f)) > 0) buf.insert(buf.end(), tmp, tmp + n);
    fclose(f);
    uint8_t* p = WebPDecodeRGBA(buf.data(), buf.size(), &im.w, &im.h); REQ(p, "webp decode %s", path.c_str());
    im.d.assign(p, p + (size_t)im.w * im.h * 4); WebPFree(p);
    return im;
#else
    REQ(false, "built without libwebp");
#endif
  }
  int c; uint8_t* p = stbi_load(path.c_str(), &im.w, &im.h, &c, 4); REQ(p, "load %s", path.c_str());
  im.d.assign(p, p + (size_t)im.w * im.h * 4); stbi_image_free(p);
  return im;
}

Image8 resize_lanczos(const Image8& im, int W, int H);

// PIL Image.resize on RGBA: convert to premultiplied RGBa (Convert.c rgba2rgbA), resample, convert back (rgbA2rgba)
static Image8 resize_rgba(const Image8& im, int W, int H) {
  Image8 p = im;
  for (size_t i = 0; i < p.d.size(); i += 4)
    for (int c = 0; c < 3; c++) { unsigned t = p.d[i + c] * p.d[i + 3] + 128; p.d[i + c] = (uint8_t)(((t >> 8) + t) >> 8); }
  Image8 o = resize_lanczos(p, W, H);
  for (size_t i = 0; i < o.d.size(); i += 4) {
    unsigned a = o.d[i + 3];
    if (a != 255 && a != 0) for (int c = 0; c < 3; c++) o.d[i + c] = (uint8_t)std::min(255u, 255u * o.d[i + c] / a);
  }
  return o;
}

Image8 preprocess_image(const Image8& in0) {
  REQ(in0.c == 4, "preprocess: RGBA only");
  bool has_alpha = false;
  for (int i = 0; i < in0.w * in0.h; i++) has_alpha |= in0.d[i * 4 + 3] != 255;
  REQ(has_alpha, "opaque input needs background removal (BiRefNet), not supported");
  double scale = std::min(1.0, 1024.0 / std::max(in0.w, in0.h));
  const Image8 in = scale < 1 ? resize_rgba(in0, (int)(in0.w * scale), (int)(in0.h * scale)) : in0;
  int x0 = in.w, y0 = in.h, x1 = -1, y1 = -1;
  for (int y = 0; y < in.h; y++) for (int x = 0; x < in.w; x++)
    if (in.d[((size_t)y * in.w + x) * 4 + 3] > 0.8 * 255) { x0 = std::min(x0, x); x1 = std::max(x1, x); y0 = std::min(y0, y); y1 = std::max(y1, y); }
  double cx = (x0 + x1) / 2.0, cy = (y0 + y1) / 2.0; int size = std::max(x1 - x0, y1 - y0), hs = size / 2;
  int bx0 = (int)std::nearbyint(cx - hs), by0 = (int)std::nearbyint(cy - hs), bx1 = (int)std::nearbyint(cx + hs), by1 = (int)std::nearbyint(cy + hs);
  Image8 o; o.w = bx1 - bx0; o.h = by1 - by0; o.c = 3; o.d.assign((size_t)o.w * o.h * 3, 0);
  for (int y = 0; y < o.h; y++) for (int x = 0; x < o.w; x++) {
    int sx = x + bx0, sy = y + by0; if (sx < 0 || sy < 0 || sx >= in.w || sy >= in.h) continue;
    const uint8_t* s = &in.d[((size_t)sy * in.w + sx) * 4]; float a = (float)s[3] / 255.f;
    for (int c = 0; c < 3; c++) o.d[((size_t)y * o.w + x) * 3 + c] = (uint8_t)(((float)s[c] / 255.f * a) * 255.f);
  }
  return o;
}

// ---- PIL Resample.c (8bpc, LANCZOS)
static constexpr int PB = 32 - 8 - 2;
static double lanczos(double x) {
  auto sinc = [](double x) { if (x == 0.0) return 1.0; x = x * M_PI; return sin(x) / x; };
  return (-3.0 <= x && x < 3.0) ? sinc(x) * sinc(x / 3) : 0.0;
}
static int coeffs(int inSize, float in0, float in1, int outSize, std::vector<int>& bounds, std::vector<int>& kk) {
  double scale = (double)(in1 - in0) / outSize, fs = scale < 1.0 ? 1.0 : scale, support = 3.0 * fs;
  int ksize = (int)ceil(support) * 2 + 1;
  std::vector<double> k(outSize * ksize); bounds.resize(outSize * 2);
  for (int xx = 0; xx < outSize; xx++) {
    double center = in0 + (xx + 0.5) * scale, ww = 0.0, ss = 1.0 / fs;
    int xmin = (int)(center - support + 0.5); if (xmin < 0) xmin = 0;
    int xmax = (int)(center + support + 0.5); if (xmax > inSize) xmax = inSize;
    xmax -= xmin; double* kp = &k[xx * ksize]; int x;
    for (x = 0; x < xmax; x++) { double w = lanczos((x + xmin - center + 0.5) * ss); kp[x] = w; ww += w; }
    for (x = 0; x < xmax; x++) if (ww != 0.0) kp[x] /= ww;
    for (; x < ksize; x++) kp[x] = 0;
    bounds[xx * 2] = xmin; bounds[xx * 2 + 1] = xmax;
  }
  kk.resize(k.size());
  for (size_t i = 0; i < k.size(); i++) kk[i] = k[i] < 0 ? (int)(-0.5 + k[i] * (1 << PB)) : (int)(0.5 + k[i] * (1 << PB));
  return ksize;
}
static inline uint8_t clip8(int v) { v >>= PB; return v < 0 ? 0 : v > 255 ? 255 : v; }
Image8 resize_lanczos(const Image8& im, int W, int H) {
  const int C = im.c;
  if (im.w == W && im.h == H) return im;
  std::vector<int> bh, kh, bv, kv;
  int ksh = coeffs(im.w, 0, im.w, W, bh, kh), ksv = coeffs(im.h, 0, im.h, H, bv, kv);
  bool nh = W != im.w, nv = H != im.h;
  int yf = bv[0], yl = bv[H * 2 - 2] + bv[H * 2 - 1];
  Image8 cur = im;
  if (nh) {
    for (int i = 0; i < H; i++) bv[i * 2] -= yf;
    Image8 t; t.w = W; t.h = yl - yf; t.c = C; t.d.resize((size_t)t.w * t.h * C);
    for (int y = 0; y < t.h; y++) for (int x = 0; x < W; x++) {
      int ss[4] = {1 << (PB - 1), 1 << (PB - 1), 1 << (PB - 1), 1 << (PB - 1)}; const int* k = &kh[x * ksh];
      for (int j = 0; j < bh[x * 2 + 1]; j++) for (int c = 0; c < C; c++) ss[c] += cur.d[((size_t)(y + yf) * cur.w + j + bh[x * 2]) * C + c] * k[j];
      for (int c = 0; c < C; c++) t.d[((size_t)y * W + x) * C + c] = clip8(ss[c]);
    }
    cur = std::move(t);
  }
  if (nv) {
    Image8 t; t.w = W; t.h = H; t.c = C; t.d.resize((size_t)W * H * C);
    for (int y = 0; y < H; y++) { const int* k = &kv[y * ksv];
      for (int x = 0; x < W; x++) {
        int ss[4] = {1 << (PB - 1), 1 << (PB - 1), 1 << (PB - 1), 1 << (PB - 1)};
        for (int j = 0; j < bv[y * 2 + 1]; j++) for (int c = 0; c < C; c++) ss[c] += cur.d[((size_t)(j + bv[y * 2]) * W + x) * C + c] * k[j];
        for (int c = 0; c < C; c++) t.d[((size_t)y * W + x) * C + c] = clip8(ss[c]);
      }
    }
    cur = std::move(t);
  }
  return cur;
}

__global__ void k_norm(const uint8_t* x, int n, float* y) {  // (u8/255 [numpy]) .sub_(mean).div_(std) [torch, cuda]
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= 3 * n) return;
  const float mean[3] = {0.485f, 0.456f, 0.406f}, sd[3] = {0.229f, 0.224f, 0.225f};
  int c = i / n, p = i % n;
  y[i] = __fdiv_rn(__fsub_rn(__fdiv_rn((float)x[p * 3 + c], 255.f), mean[c]), sd[c]);
}
Tensor image_to_input(const Image8& im) {
  int n = im.w * im.h; Tensor u = from_host(im.d.data(), {(i64)n * 3}, U8), y = empty({1, 3, im.h, im.w}, F32);
  k_norm<<<cdiv(3 * n, 256), 256, 0, stream()>>>(u.ptr<uint8_t>(), n, y.ptr<float>());
  return y;
}
