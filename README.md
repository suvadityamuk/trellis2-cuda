# TRELLIS.2 in CUDA C++

Image → textured GLB for [TRELLIS.2-4B](https://github.com/microsoft/TRELLIS.2) (`1024_cascade`, `example.py` `to_glb` settings),
with no Python at runtime. Two builds from one binary:

- **parity** (default): bitwise identical to the official PyTorch pipeline from the input image to the final mesh and attributes,
  and through the GLB writer (texture bake, TELEA inpaint, WebP, glTF bytes). The one exception is the vertex-normal buffer:
  CuMesh accumulates normals with atomics, so the official code is nondeterministic there too.
- **fast** (`T2_FAST=1`): tolerance-bounded; see below.

The full worklog (method, bugs, figures) is in [docs/blog.md](docs/blog.md).

## Results (H200, 5 example images, warm process, 1 warmup, seconds)

| Image | Official eager | Official torch.compile | CUDA parity | CUDA fast |
|---|---|---|---|---|
| 0a34 | 89.0 | 89.1 | 41.3 | **16.3** |
| 0e49 | 154.2 | 131.2 | 65.8 | **34.4** |
| 0f16 | 86.9 | 89.6 | 54.2 | **22.4** |
| 130c | 143.6 | 81.1 | 29.6 | **13.4** |
| T.png | 92.8 | 86.8 | 70.0 | **20.2** |
| **Sum** | 566.5 | 477.8 | 260.9 (2.2×) | **106.6 (5.3×)** |

Times are pipeline + `to_glb` + export, with model loading excluded. Raw numbers are in `bench/*.json`. Speedups are relative to eager.

![total time](docs/speedup.png)

![per-image stage breakdown](docs/timing.png)

Most of the official time is in `to_glb`: a serial CPU cuBVH build, quadratic xatlas chart growth, and single-threaded TELEA.
The large eager `pipeline` times on 0e49 and 130c are FlexGEMM's Triton autotuning on first-seen sparse-conv shapes.
The torch.compile run came afterwards and reused that saved cache (`~/.flex_gemm/autotune_cache.json`). The CUDA build
has no per-shape tuning step.

## Outputs

Rendered with nvdiffrast from the exported GLBs (base color with simple shading, same camera; `tools/render_glb.py`):

![renders](docs/renders.png)

Turntable of the CUDA fast output for `T.png`:

![turntable](docs/T_fast.gif)

## Fast build

| Change | Effect (T.png) |
|---|---|
| GPU LBVH (Morton + radix sort + Karras tree) instead of cuBVH's CPU build | 7.1 s → 5 ms, remesh still bitwise |
| Split CuMesh charts > 1024 faces before xatlas (its chart growth is quadratic and serial per mesh) | ComputeCharts 36 s → 1.3 s, ~1% more charts |
| GPU wavefront TELEA (per distance level, all 6 channels) | 8.2 s → 0.15 s |
| FlashAttention-3 (hopper, bf16 hdim128) for all DiT attention | shape 7.7 → 5.6 s, tex 3.7 → 2.6 s |
| WebP method 0 | export 6.0 → 1.4 s |

These also help parity mode without changing its output: xatlas sized to the cgroup CPU quota (`third_party/shim/xa_hw.h`).

**Tolerance gates.**
1. DiT sampling error (teacher-forced, `parity sample`) matches the official pipeline's own error when its attention
   backend is switched from flash_attn to xformers. Sparse-structure stage: mean abs 0.0145 / max 1.35 for both, and both
   give 3534 tokens.
2. End-to-end GLB vs official eager (`tools/compare_glb.py`: symmetric Chamfer distance over the bbox diagonal, and
   base-color L1 at matched surface samples). The fast build falls in the band of official torch.compile vs eager:

| Image | compile vs eager (Chamfer / color) | parity vs eager | fast vs eager |
|---|---|---|---|
| 0a34 | 2.4e-3 / 16.2 | 1.3e-3 / 3.3 | 2.4e-3 / 20.5 |
| 0e49 | 4.0e-3 / 23.3 | 1.7e-3 / 7.9 | 3.8e-3 / 23.6 |
| 0f16 | 2.9e-3 / 26.9 | 1.7e-3 / 5.6 | 3.0e-3 / 24.1 |
| 130c | 4.5e-3 / 25.8 | 0.9e-3 / 4.0 | 4.6e-3 / 26.6 |
| T.png | 3.0e-3 / 25.2 | 1.6e-3 / 6.1 | 2.8e-3 / 28.7 |

(Parity vs eager is nonzero only because CuMesh simplify is nondeterministic run to run.)

## Cost

HF Jobs compute, from job running time × listed price:

| Job | Flavor | Price | Running | Cost |
|---|---|---|---|---|
| `6ac54e95…` (all builds, parity runs, benchmarks) | h200 (1× H200, 23 vCPU) | $5.00/h | 8.4 h | ~$42 |
| `6ac55477…` (CPU box started while the H200 was queued 43 min) | cpu-performance | $1.90/h | 8.7 h | ~$17 |
| `6ac5e12c…` + `6ac5e099…` (fresh-container parity reproduction) | h200 | $5.00/h | 58 min | ~$5 |
| **Total** | | | | **~$64** |

The benchmarks themselves are a small part of this. One full 4-way pass over the 5 images is about 24 GPU-minutes (~$2).
Most of the cost is development time on an idle-billing `sleep infinity` job.

Cursor model/API usage is not included. It is not visible from inside the agent; see the Cursor dashboard
([cursor.com/dashboard](https://cursor.com/dashboard) → Usage) for this session.

## Build and run

Requires CUDA 12.4+, cuDNN 9, CMake ≥ 3.24, Ninja, and an sm_90 GPU (FA3 is sm_90a).

```bash
tools/setup_ref.sh                         # official PyTorch reference (parity checks and baselines only)
tools/build.sh build                       # fetches pinned third-party sources, builds build/t2 and build/parity
tools/repro_parity.sh build                # official dumps + bitwise checks on the 5 example images
./build/t2 image.png out.glb [seed]        # parity build
T2_FAST=1 ./build/t2 image.png out.glb     # fast build
./build/t2 --bench out.json out_dir img... # timing as in tools/bench_ref.py
```

Checkpoints are read from `$T2_CKPT` (default `/workspace/ckpts`), as safetensors from `microsoft/TRELLIS.2-4B` and DINOv3.
Set `T2_PROF=1` for per-stage timing.

**Parity reproducibility.** FlexGEMM autotunes sparse-conv tiles by timing them, so its choices (and hence the fp32
accumulation order) can differ between machines. `assets/flexgemm_autotune.json` holds the H200 choices for the five
example images. The CUDA build reads it by default (`$T2_FLEXGEMM_CACHE` overrides), and `tools/repro_parity.sh` points
the official pipeline at the same file through `FLEX_GEMM_AUTOTUNE_CACHE_PATH`. On a fresh H200 container, that script
passes all 15 checks: 5 images × `pipe`, `glb` and `bake`. `pipe` and `glb` are bitwise; `bake` is identical except
the vertex normals. A new image or GPU needs one official run first, to add its tiles to the cache.

## Layout

- `src/`: models (DINOv3, flow DiTs, sparse-structure, shape and texture decoders), GEMM/attention/sparse-conv ops, the
  `to_glb` port (`glb.cu`), OpenCV TELEA (`inpaint.cpp`), and the trimesh/PIL GLB writer (`gltf.cpp`).
- `apps/parity.cu`: per-stage bitwise checks against dumps from `tools/ref_dump.py`.
- `tools/`: offline-only Python (reference dumps, official timing, GLB comparison, plots, and renders).
- `bench/`: raw timings, plus the T.png chart-size and TELEA-level dumps. `docs/`: blog, plots and renders.
- `assets/flexgemm_autotune.json`: the sparse-conv tile choices shared by both sides for bitwise parity.

Third-party CUDA code is reused unmodified: flash-attention 2.7.3 (FA2 + FA3), CuMesh, cuBVH, xatlas, FlexGEMM,
nvdiffrast 0.4.0, cudnn-frontend, cutlass, and libwebp 1.4.0.
