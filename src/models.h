#pragma once
#include "dit.h"
#include "spconv.h"
#include <functional>
#include <random>

// ---- dense conv2d/conv3d (conv.cu): torch ConvNd forward via the cuDNN v8 path, NC[D]HW, cubic kernel
void conv(const Tensor& x, const Tensor& w, const Tensor* b, int pad, int stride, Tensor& y);
Tensor conv(const Tensor& x, const Tensor& w, const Tensor* b, int pad, int stride);

// ---- sparse-structure VAE decoder (ssdec.cu)
struct SSDecoder {
  TensorMap W;
  void load(const std::string& path);
  Tensor forward(const Tensor& z);  // fp32 [1,8,16,16,16] -> fp32 logits [1,1,64,64,64]
  Tensor res(const std::string& p, const Tensor& x);
};
Tensor occupancy_coords(const Tensor& logits, int r);  // (logits>0) max-pooled to r^3 -> int32 [M,4] (0,x,y,z)

// ---- sparse SLat VAE decoders (sdec.cu): shape (FlexiDualGrid, predicts subdivisions) and texture (guided by them)
struct SDecoder {
  struct Level { Tensor coords, src; std::shared_ptr<SpConvCache> nb; };  // src: parent*8 + child slot (C2S output rows)
  TensorMap W;
  bool pred_subdiv = true;
  std::function<void(const std::string&, const Tensor&)> dbg;
  void load(const std::string& path, bool pred_subdiv);
  // fp32 latent feats [N,32] + coords -> fp32 [M, out] at the finest level (coords_out); subs: per-level fp16 logits [n,8].
  // stop_level=k returns after k upsamples with only coords_out set (SparseUnetVaeDecoder.upsample).
  Tensor run(const Tensor& feats, const Tensor& coords, const std::vector<Tensor>* guide, std::vector<Tensor>* subs,
             int stop_level = -1, Tensor* coords_out = nullptr);
 private:
  const Tensor& w(const std::string& k);
  Tensor conv(const std::string& p, const Tensor& x, SpConvCache& c);
  Tensor convnext(const std::string& p, const Tensor& x, SpConvCache& c);
  Tensor up(const std::string& p, const Tensor& x, Level& lv, const Tensor* guide, Tensor* sub_out);
  Level c2s(const Level& in, const Tensor& sub);
};

// ---- fp32 SDPA (mea.cu): ATen mem-efficient attention kernel
struct BMHK { const float* p; i64 sB, sM, sH; };
void mem_eff_attn(BMHK q, BMHK k, BMHK v, float* o, int B, int M, int N, int H, int D, float scale);

// ---- DINOv3 ViT-L/16 feature extractor (dino.cu): DinoV3FeatureExtractor.extract_features + final layer_norm
struct Dino {
  TensorMap W;
  std::function<void(const std::string&, const Tensor&)> dbg;
  void load(const std::string& path);
  Tensor forward(const Tensor& img);  // normalized fp32 [1,3,S,S] -> [1+4+(S/16)^2, 1024]
};

// ---- image preprocessing (image.cpp): PIL/numpy-exact host code
struct Image8 { int w = 0, h = 0, c = 0; std::vector<uint8_t> d; };
Image8 load_image(const std::string& path);                // RGBA (PNG via stb_image, WebP via libwebp)
Image8 preprocess_image(const Image8& rgba);                // Trellis2ImageTo3DPipeline.preprocess_image (alpha input)
Image8 resize_lanczos(const Image8& im, int W, int H);      // PIL Image.resize(LANCZOS), 8bpc
Tensor image_to_input(const Image8& rgb);                   // /255, ImageNet normalize -> fp32 [1,3,H,W] (device)

void fdg_to_mesh(const Tensor& h, const Tensor& coords, int R, Tensor& V, Tensor& F);  // o_voxel flexible_dual_grid_to_mesh
void fill_holes(Tensor& V, Tensor& F, float max_hole_perimeter);                     // trellis2 Mesh.fill_holes (CuMesh)

// ---- full pipeline (pipeline.cu): Trellis2ImageTo3DPipeline.run(pipeline_type='1024_cascade') through decode_latent
std::vector<float> torch_randn(std::mt19937& g, i64 n);  // torch.manual_seed + torch.randn (CPU) stream
struct Pipeline {
  Dino dino; DiT ss, sh512, sh1024, tex; SSDecoder ssd; SDecoder shd, txd;
  std::function<void(const std::string&, const Tensor&)> dbg;
  std::function<void(const char*)> times;  // called (after a device sync) at each stage end
  struct Out { Tensor V, F, attrs, coords; };  // attrs: fp32 [M, 6] PBR voxels at coords (int32 [M, 4])
  void load(const std::string& ckpt_dir);
  Out run(const Image8& rgba, uint64_t seed = 42);
};

// ---- GLB export (glb.cu): o_voxel.postprocess.to_glb(remesh=True, remesh_band=1, remesh_project=0)
struct BVH {  // cumesh.cuBVH
  struct Impl; std::shared_ptr<Impl> p;
  BVH(const Tensor& V, const Tensor& F);
  Tensor udf(const Tensor& pts, Tensor* face_id = nullptr, Tensor* uvw = nullptr) const;  // face_id int64, uvw fp32 [n,3]
};
struct Glb {
  std::function<void(const std::string&, const Tensor&)> dbg;
  Tensor V0, F0; std::unique_ptr<BVH> bvh;  // hole-filled input mesh + its BVH (texture lookup)
  Tensor V, F, UV, N;                       // output mesh: positions, faces, uvs, vertex normals
  void geometry(const Tensor& V, const Tensor& F, int decimation_target = 1000000);  // = remesh -> simplify -> unwrap
  void remesh(const Tensor& V, const Tensor& F, Tensor& Vr, Tensor& Fr);  // fill_holes, cuBVH, narrow-band DC
  void simplify(Tensor& V, Tensor& F, int target);                        // nondeterministic (CuMesh simplify_step races)
  void unwrap(const Tensor& Vs, const Tensor& Fs);                         // uv_unwrap + vertex normals
  // UV raster (nvdiffrast) -> reproject onto V0/F0 -> trilinear grid_sample_3d (FlexGEMM) -> uint8 -> cv2 TELEA inpaint
  void bake(const Tensor& attrs, const Tensor& coords, int tex = 4096);  // attrs fp32 [M,6], coords int32 [M,3] or [M,4] (0,x,y,z)
  int tex = 0; std::vector<uint8_t> base_rgba, mr_rgb;                   // baseColor RGBA, metallicRoughness (0, R, M)
  std::string glb_bytes() const;                                          // trimesh export(extension_webp=True)
  void save(const std::string& path) const;
 private:
  void tap(const std::string& n, const Tensor& v) { if (dbg) dbg(n, v); }
};
