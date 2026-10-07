// nvdiffrast v0.4.0 rasterize_fwd_cuda / interpolate_fwd (torch glue, instance mode, single image, no peeling/attr diffs).
#include "common/common.h"
#include "common/rasterize.h"
#include "common/interpolate.h"
#include "common/cudaraster/CudaRaster.hpp"
#include "common/cudaraster/impl/Constants.hpp"
#include <memory>

void RasterizeCudaFwdShaderKernel(const RasterizeCudaFwdShaderParams p);
void InterpolateFwdKernel(const InterpolateKernelParams p);

struct NvdrCtx { CR::CudaRaster cr; };
std::shared_ptr<NvdrCtx> nvdr_ctx() { return std::make_shared<NvdrCtx>(); }

// pos: [V,4] fp32 device, tri: [T,3] int32 device; out/out_db: [h,w,4] fp32 device
void nvdr_rasterize(NvdrCtx& c, const float* pos, int nv, const int* tri, int nt, int h, int w, float* out, float* out_db, cudaStream_t s) {
  CR::CudaRaster* cr = &c.cr;
  int height = (h + CR_TILE_SIZE - 1) & (-CR_TILE_SIZE), width = (w + CR_TILE_SIZE - 1) & (-CR_TILE_SIZE);
  cr->setVertexBuffer((void*)pos, nv);
  cr->setIndexBuffer((void*)tri, nt);
  cr->setBufferSize(w, h, 1);
  cr->setRenderModeFlags(0);
  int tcx = (width + CR_MAXVIEWPORT_SIZE - 1) / CR_MAXVIEWPORT_SIZE, tcy = (height + CR_MAXVIEWPORT_SIZE - 1) / CR_MAXVIEWPORT_SIZE;
  int tsx = ((width + tcx - 1) / tcx + CR_TILE_SIZE - 1) & (-CR_TILE_SIZE), tsy = ((height + tcy - 1) / tcy + CR_TILE_SIZE - 1) & (-CR_TILE_SIZE);
  for (int ty = 0; ty < tcy; ty++)
    for (int tx = 0; tx < tcx; tx++) {
      int ox = tx * tsx, oy = ty * tsy;
      cr->setViewport(w - ox < tsx ? w - ox : tsx, h - oy < tsy ? h - oy : tsy, ox, oy);
      cr->deferredClear(0u);
      NVDR_CHECK(cr->drawTriangles(nullptr, false, s), "subtriangle count overflow");
    }
  RasterizeCudaFwdShaderParams p{};
  p.pos = pos; p.tri = tri; p.in_idx = (const int*)cr->getColorBuffer(); p.out = out; p.out_db = out_db;
  p.numTriangles = nt; p.numVertices = nv; p.width_in = width; p.height_in = height; p.width_out = w; p.height_out = h;
  p.depth = 1; p.instance_mode = 1;
  p.xs = 2.f / (float)w; p.xo = 1.f / (float)w - 1.f; p.ys = 2.f / (float)h; p.yo = 1.f / (float)h - 1.f;
  dim3 bs = getLaunchBlockSize(RAST_CUDA_FWD_SHADER_KERNEL_BLOCK_WIDTH, RAST_CUDA_FWD_SHADER_KERNEL_BLOCK_HEIGHT, w, h);
  dim3 gs = getLaunchGridSize(bs, w, h, 1);
  void* args[] = {&p};
  NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((void*)RasterizeCudaFwdShaderKernel, gs, bs, args, 0, s));
}

// attr: [1,V,A] broadcast (instance mode), rast: [h,w,4]; out: [h,w,A]
void nvdr_interpolate(const float* attr, int nv, int na, const float* rast, int h, int w, const int* tri, int nt, float* out, cudaStream_t s) {
  InterpolateKernelParams p = {};
  p.instance_mode = 1; p.numVertices = nv; p.numAttr = na; p.numTriangles = nt; p.height = h; p.width = w; p.depth = 1;
  p.numDiffAttr = 0; p.attr = attr; p.rast = rast; p.tri = tri; p.rastDB = nullptr; p.attrBC = 1; p.out = out; p.outDA = nullptr;
  dim3 bs = getLaunchBlockSize(IP_FWD_MAX_KERNEL_BLOCK_WIDTH, IP_FWD_MAX_KERNEL_BLOCK_HEIGHT, w, h);
  dim3 gs = getLaunchGridSize(bs, w, h, 1);
  void* args[] = {&p};
  NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((void*)InterpolateFwdKernel, gs, bs, args, 0, s));
}
