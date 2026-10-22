// C API over the pipeline for embedding (ctypes). Same stages and settings as apps/t2.cu.
// All entry points return 0 on success, or -1 with t2_last_error() set. Calls are serialized.
#include "models.h"
#include <exception>
#include <mutex>

#define T2_API extern "C" __attribute__((visibility("default")))

namespace {
std::mutex mu;
std::unique_ptr<Pipeline> pipe;
std::string last_error;

template <class F> int guarded(F&& f) {
  std::lock_guard<std::mutex> lk(mu);
  try { f(); return 0; }
  catch (const std::exception& e) { last_error = e.what(); }
  catch (...) { last_error = "unknown error"; }
  return -1;
}
}  // namespace

T2_API const char* t2_last_error() { return last_error.c_str(); }

// ckpt_dir: the safetensors listed in Pipeline::load.
T2_API int t2_load(const char* ckpt_dir) {
  return guarded([&] {
    if (pipe) return;
    auto p = std::make_unique<Pipeline>();
    p->load(ckpt_dir); dsync();
    pipe = std::move(p);
  });
}

// rgba: h*w*4 bytes with a real alpha channel (no background removal). times (may be null) receives seconds for
// pipeline, remesh, simplify, uv_unwrap, bake, export. nv/nf (may be null) receive the output vertex/face counts.
T2_API int t2_image_to_glb(const uint8_t* rgba, int w, int h, uint64_t seed, int faces, int texture, const char* out_glb,
                           double* times, int64_t* nv, int64_t* nf) {
  return guarded([&] {
    REQ(pipe, "t2_load has not succeeded");
    REQ(w > 0 && h > 0 && faces > 0 && texture >= 256 && texture <= 8192, "bad arguments");
    Image8 im; im.w = w; im.h = h; im.c = 4; im.d.assign(rgba, rgba + (size_t)w * h * 4);
    double t = now_ms(); int k = 0;
    auto lap = [&](const char*) { dsync(); double n = now_ms(); if (times) times[k] = (n - t) / 1e3; k++; t = n; };
    pipe->times = nullptr;
    auto o = pipe->run(im, seed); lap("pipeline");
    Glb g; Tensor V, F;
    if (o.F.size(0) > 16777216) g.simplify(o.V, o.F, 16777216);
    g.remesh(o.V, o.F, V, F); lap("remesh");
    g.simplify(V, F, faces); lap("simplify");
    g.unwrap(V, F); lap("uv_unwrap");
    g.bake(o.attrs, o.coords, texture); lap("bake");
    g.save(out_glb); lap("export");
    if (nv) *nv = g.V.size(0);
    if (nf) *nf = g.F.size(0);
  });
}
