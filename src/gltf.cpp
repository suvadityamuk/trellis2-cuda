// trimesh.Trimesh(...).export(extension_webp=True) for the to_glb output: GLB 2.0 with PIL-encoded WebP textures.
#include "models.h"
#include <webp/encode.h>
#include <webp/mux.h>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <thread>

// Python repr(float(np.float32 x))
static std::string pyrepr(float f) {
  double x = f;
  if (x == 0) return std::signbit(x) ? "-0.0" : "0.0";
  char b[64]; auto r = std::to_chars(b, b + 64, x, std::chars_format::scientific); std::string s(b, r.ptr);
  std::string neg = s[0] == '-' ? "-" : ""; if (!neg.empty()) s = s.substr(1);
  size_t e = s.find('e'); int ex = std::stoi(s.substr(e + 1)); std::string d = s.substr(0, e); d.erase(std::remove(d.begin(), d.end(), '.'), d.end());
  int n = (int)d.size(), dp = ex + 1;
  if (dp > -4 && dp <= 16) {
    if (dp <= 0) return neg + "0." + std::string(-dp, '0') + d;
    if (dp >= n) return neg + d + std::string(dp - n, '0') + ".0";
    return neg + d.substr(0, dp) + "." + d.substr(dp);
  }
  char eb[16]; snprintf(eb, 16, "e%c%02d", ex < 0 ? '-' : '+', std::abs(ex));
  return neg + d.substr(0, 1) + (n > 1 ? "." + d.substr(1) : "") + eb;
}

// PIL Image.save(format="WEBP") defaults (Pillow 11 _webp.WebPEncode): lossy q80, alpha_quality 100, method 4, ARGB picture
static std::string pil_webp(const uint8_t* px, int w, int h, int nc) {
  WebPConfig cfg; WebPConfigInit(&cfg);
  cfg.lossless = 0; cfg.quality = 80.f; cfg.alpha_quality = 100.f; cfg.method = fast_mode() ? 0 : 4; cfg.exact = 0; cfg.thread_level = 1;
  REQ(WebPValidateConfig(&cfg), "webp config");
  WebPPicture pic; WebPPictureInit(&pic); pic.width = w; pic.height = h; pic.use_argb = 1;
  REQ(WebPPictureAlloc(&pic), "webp alloc");
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      const uint8_t* s = px + ((size_t)y * w + x) * nc;
      pic.argb[(size_t)y * pic.argb_stride + x] = (uint32_t)s[2] | (uint32_t)s[1] << 8 | (uint32_t)s[0] << 16 | (uint32_t)(nc == 4 ? s[3] : 0xff) << 24;
    }
  WebPMemoryWriter wr; WebPMemoryWriterInit(&wr); pic.writer = WebPMemoryWrite; pic.custom_ptr = &wr;
  REQ(WebPEncode(&cfg, &pic), "webp encode %d", pic.error_code);
  WebPPictureFree(&pic);
  WebPData img = {wr.mem, wr.size}, out = {};
  WebPMux* mux = WebPMuxNew(); WebPMuxSetImage(mux, &img, 0); WebPMuxAssemble(mux, &out); WebPMuxDelete(mux);
  std::string r((const char*)out.bytes, out.size); WebPDataClear(&out); WebPMemoryWriterClear(&wr);
  return r;
}

template <int C> static std::string minmax(const std::vector<float>& v, bool mx) {
  float m[C]; for (int c = 0; c < C; c++) m[c] = v[c];
  for (size_t i = C; i < v.size(); i++) { int c = i % C; m[c] = mx ? std::max(m[c], v[i]) : std::min(m[c], v[i]); }
  std::string s = "["; for (int c = 0; c < C; c++) s += (c ? "," : "") + pyrepr(m[c]); return s + "]";
}

std::string Glb::glb_bytes() const {
  auto v = to_host_vec<float>(V), n = to_host_vec<float>(N), uv = to_host_vec<float>(UV); auto f = to_host_vec<uint32_t>(F);
  for (size_t i = 0; i < v.size(); i += 3) { float y = v[i + 1]; v[i + 1] = v[i + 2]; v[i + 2] = -y; y = n[i + 1]; n[i + 1] = n[i + 2]; n[i + 2] = -y; }
  // to_glb flips v in float32; trimesh flips it back in float64. Normals: util.unitize in float64 (OpenBLAS dot, then 1/norm)
  for (size_t i = 1; i < uv.size(); i += 2) uv[i] = (float)(1.0 - (double)(1.f - uv[i]));
  for (size_t i = 0; i < n.size(); i += 3) {
    double a = n[i], b = n[i + 1], c = n[i + 2], nr = std::sqrt(a * a + b * b + c * c);
    if (nr > 1e-13) nr = 1.0 / nr;
    n[i] = (float)(a * nr); n[i + 1] = (float)(b * nr); n[i + 2] = (float)(c * nr);
  }
  std::string img[2];
  double t = now_ms();
  std::thread t0([&] { img[0] = pil_webp(base_rgba.data(), tex, tex, 4); }), t1([&] { img[1] = pil_webp(mr_rgb.data(), tex, tex, 3); });
  t0.join(); t1.join();
  if (env("T2_PROF") == "1") printf("    %-24s %8.1f ms\n", "webp", now_ms() - t);
  auto raw = [](const void* p, size_t b) { return std::string((const char*)p, b); };
  std::string blobs[6] = {raw(f.data(), f.size() * 4), raw(v.data(), v.size() * 4), img[0], img[1], raw(uv.data(), uv.size() * 4), raw(n.data(), n.size() * 4)};
  std::string bin, views;
  for (int i = 0; i < 6; i++) {
    blobs[i].resize((blobs[i].size() + 3) / 4 * 4, '\0');
    views += (i ? "," : "") + std::string("{\"buffer\":0,\"byteOffset\":") + std::to_string(bin.size()) + ",\"byteLength\":" + std::to_string(blobs[i].size()) + "}";
    bin += blobs[i];
  }
  uint32_t fmax = 0, fmin = UINT32_MAX; for (auto x : f) { fmax = std::max(fmax, x); fmin = std::min(fmin, x); }
  std::string nv = std::to_string(V.size(0));
  std::string js = "{\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"asset\":{\"version\":\"2.0\",\"generator\":\"https://github.com/mikedh/trimesh\"},"
    "\"accessors\":[{\"componentType\":5125,\"type\":\"SCALAR\",\"bufferView\":0,\"count\":" + std::to_string(f.size()) + ",\"max\":[" + std::to_string(fmax) +
    "],\"min\":[" + std::to_string(fmin) + "]},{\"componentType\":5126,\"type\":\"VEC3\",\"byteOffset\":0,\"bufferView\":1,\"count\":" + nv +
    ",\"max\":" + minmax<3>(v, true) + ",\"min\":" + minmax<3>(v, false) + "},{\"componentType\":5126,\"type\":\"VEC2\",\"byteOffset\":0,\"bufferView\":4,\"count\":" + nv +
    ",\"max\":" + minmax<2>(uv, true) + ",\"min\":" + minmax<2>(uv, false) + "},{\"componentType\":5126,\"count\":" + nv + ",\"type\":\"VEC3\",\"byteOffset\":0,\"bufferView\":5" +
    ",\"max\":" + minmax<3>(n, true) + ",\"min\":" + minmax<3>(n, false) + "}],"
    "\"meshes\":[{\"name\":\"geometry_0\",\"extras\":{},\"primitives\":[{\"attributes\":{\"POSITION\":1,\"TEXCOORD_0\":2,\"NORMAL\":3},\"indices\":0,\"mode\":4,\"material\":0}]}],"
    "\"images\":[{\"bufferView\":2,\"mimeType\":\"image/webp\"},{\"bufferView\":3,\"mimeType\":\"image/webp\"}],"
    "\"textures\":[{\"extensions\":{\"EXT_texture_webp\":{\"source\":0}}},{\"extensions\":{\"EXT_texture_webp\":{\"source\":1}}}],"
    "\"materials\":[{\"pbrMetallicRoughness\":{\"baseColorTexture\":{\"index\":0},\"baseColorFactor\":[1.0,1.0,1.0,1.0],\"roughnessFactor\":1.0,\"metallicFactor\":1.0,"
    "\"metallicRoughnessTexture\":{\"index\":1}},\"alphaMode\":\"OPAQUE\",\"doubleSided\":false}],\"nodes\":[{\"name\":\"geometry_0\",\"mesh\":0}],"
    "\"extensionsUsed\":[\"EXT_texture_webp\"],\"extensionsRequired\":[\"EXT_texture_webp\"],\"buffers\":[{\"byteLength\":" + std::to_string(bin.size()) +
    "}],\"bufferViews\":[" + views + "]}";
  js += std::string(4 - (js.size() + 20) % 4, ' ');
  uint32_t hdr[5] = {0x46546C67u, 2, (uint32_t)(js.size() + bin.size() + 28), (uint32_t)js.size(), 0x4E4F534Au}, bh[2] = {(uint32_t)bin.size(), 0x004E4942u};
  return raw(hdr, 20) + js + raw(bh, 8) + bin;
}
void Glb::save(const std::string& path) const { std::ofstream(path, std::ios::binary) << glb_bytes(); }
