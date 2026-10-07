// TRELLIS.2 image -> textured GLB (pipeline_type='1024_cascade', example.py to_glb settings).
// Usage: t2 <image> <out.glb> [seed]
//        t2 --bench <out.json> <out_dir> <images...>   (models loaded once, 1 untimed warmup on images[0], like tools/bench_ref.py)
#include "models.h"
#include <fstream>

struct Times { double pipeline, to_glb, exp, total; };

static Times run(Pipeline& P, const std::string& img, const std::string& out, uint64_t seed, bool verbose) {
  double ts = now_ms(), t0 = ts;
  auto lap = [&](const char* n) { dsync(); double t = now_ms(); if (verbose) printf("  %-12s %9.1f ms\n", n, t - ts); ts = t; };
  Image8 im = load_image(img);
  P.times = lap;
  auto o = P.run(im, seed);
  dsync(); double t1 = now_ms();
  Glb g; Tensor V, F;
  if (o.F.size(0) > 16777216) g.simplify(o.V, o.F, 16777216);  // example.py: mesh.simplify(16777216), nvdiffrast limit
  g.remesh(o.V, o.F, V, F); lap("remesh");
  g.simplify(V, F, 1000000); lap("simplify");
  g.unwrap(V, F); lap("uv_unwrap");
  g.bake(o.attrs, o.coords); lap("bake");
  dsync(); double t2 = now_ms();
  g.save(out); lap("export");
  double t3 = now_ms();
  if (verbose) printf("  %-12s %9.1f ms  (%lld verts, %lld faces)\n", "total", t3 - t0, (long long)g.V.size(0), (long long)g.F.size(0));
  return {(t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3, (t3 - t0) / 1e3};
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  bool bench = argc > 1 && std::string(argv[1]) == "--bench";
  REQ(bench ? argc >= 5 : argc >= 3, "usage: t2 <image> <out.glb> [seed] | t2 --bench <out.json> <out_dir> <images...>");
  double t0 = now_ms();
  Pipeline P; P.load(env("T2_CKPT", "/workspace/ckpts")); dsync();
  printf("  %-12s %9.1f ms\n", "load", now_ms() - t0);
  if (!bench) { run(P, argv[1], argv[2], argc > 3 ? std::stoull(argv[3]) : 42, true); return 0; }
  std::string dir = argv[3];
  auto base = [](std::string p) { return p.substr(p.find_last_of('/') + 1); };
  auto fmt = [](const Times& t) {
    char b[160]; snprintf(b, 160, "{\"pipeline\": %.4f, \"to_glb\": %.4f, \"export\": %.4f, \"total\": %.4f}", t.pipeline, t.to_glb, t.exp, t.total);
    return std::string(b);
  };
  printf("warmup %s\n", fmt(run(P, argv[4], dir + "/warmup.glb", 42, false)).c_str());
  std::string js = "{";
  for (int i = 4; i < argc; i++) {
    Times t = run(P, argv[i], dir + "/" + base(argv[i]) + ".glb", 42, env("T2_PROF") == "1");
    printf("%s %s\n", base(argv[i]).c_str(), fmt(t).c_str());
    js += (i > 4 ? ",\n \"" : "\n \"") + base(argv[i]) + "\": " + fmt(t);
  }
  std::ofstream(argv[2]) << js << "\n}\n";
}
