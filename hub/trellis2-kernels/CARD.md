---
library_name: kernels
license: apache-2.0
tags:
  - kernels
  - cuda
  - 3d
  - mesh
  - inpainting
---

# trellis2-kernels

Two standalone CUDA kernels from [TRELLIS.2 in CUDA C++](https://github.com/suvadityamuk/trellis2-cuda), a port of
Microsoft's [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) image-to-3D pipeline. In the port, they replace serial CPU steps in mesh post-processing:

| Function | What it does | In the port (H200) |
|---|---|---|
| `mesh_udf` | Distance from query points to a triangle mesh, plus the nearest face and barycentric coordinates. Builds a GPU linear BVH on each call. | BVH build 7.1 s → 5 ms (vs. cuBVH's CPU build) |
| `telea_inpaint` | TELEA inpainting on the GPU, an alternative to `cv2.inpaint(..., cv2.INPAINT_TELEA)`. | 4096² texture atlas: 8.2 s → 0.15 s |

## Usage

```python
# pip install -U kernels
import torch
from kernels import get_kernel

k = get_kernel("{{ repo_id }}", version={{ version }}, trust_remote_code=True)

# Point-to-mesh distance
vertices = torch.rand(1000, 3, device="cuda")                    # float32 [V, 3]
faces = torch.randint(0, 1000, (500, 3), device="cuda")          # int32/int64 [F, 3]
points = torch.randn(4096, 3, device="cuda")                     # float32 [M, 3]
dist, face_id, bary = k.mesh_udf(vertices, faces, points)        # [M], [M] int64, [M, 3]

# Inpainting
image = torch.randint(0, 256, (512, 512, 3), dtype=torch.uint8, device="cuda")
mask = torch.zeros(512, 512, dtype=torch.bool, device="cuda")
mask[200:300, 200:300] = True                                    # nonzero = fill
filled = k.telea_inpaint(image, mask, radius=3)                  # uint8, same shape as image
```

## API

`mesh_udf(vertices, faces, points) -> (distance, face_id, barycentric)`

- `vertices`: float32 `[V, 3]`, CUDA. `faces`: int32 or int64 `[F, 3]`. `points`: float32 `[M, 3]`.
- Returns the unsigned distance (float32 `[M]`), the index of the nearest face (int64 `[M]`), and the barycentric coordinates of the closest point on that face (float32 `[M, 3]`).
- When several faces are equally close, the lowest face index wins, so results don't depend on the BVH's shape. Degenerate (zero-area) triangles are handled.

`telea_inpaint(image, mask, radius=3) -> image`

- `image`: uint8 `[H, W]` or `[H, W, C]` with `C <= 4`, CUDA. `mask`: `[H, W]` bool or uint8; nonzero pixels are filled. `radius`: neighborhood radius in pixels.
- Pixels outside the mask are returned unchanged.

## Accuracy

- `mesh_udf` matches a float64 brute-force reference to within 1e-5, for both the distance and the chosen face.
- `telea_inpaint` is close to OpenCV's output but not identical. OpenCV fills pixels one at a time in fast-marching order; this kernel fills every pixel at the same integer distance from known pixels in parallel, using TELEA's weights. On the test images, the mean absolute difference from `cv2.inpaint` over inpainted pixels is under 6 (on a 0–255 scale).

## Compatibility

- Linux x86_64 with CUDA. There are 17 builds covering torch 2.10 through 2.15 with CUDA 12.6, 12.8, 13.0, 13.2 and 13.4, compiled for sm_75 through sm_120; `get_kernel` picks the one matching your environment.
- Known issue: `telea_inpaint` faults on Blackwell (sm_120, e.g. RTX PRO 6000). Use `cv2.inpaint` on those GPUs for now.

### Test results

The full test suite (4 `mesh_udf` tests, 7 `telea_inpaint` tests including the comparison with `cv2.inpaint`) on an NVIDIA H200 (sm_90, driver 595.112.07), one run per torch release and every CUDA version:

| torch | CUDA | Result |
|---|---|---|
| 2.15.0 | 13.4 | 11/11 passed |
| 2.15.0 | 13.2 | 11/11 passed |
| 2.14.1 | 12.6 | 11/11 passed |
| 2.13.0 | 13.0 | 11/11 passed |
| 2.12.1 | 13.2 | 11/11 passed |
| 2.11.0 | 12.8 | 11/11 passed |
| 2.10.0 | 12.6 | 11/11 passed |

The other 10 builds passed kernel-builder's checks (ABI compatibility and loading with `get_kernel`) but were not run on a GPU.

## Source and license

The source, tests and build config are in [`hub/trellis2-kernels`](https://github.com/suvadityamuk/trellis2-cuda/tree/main/hub/trellis2-kernels). Licensed under Apache-2.0.
