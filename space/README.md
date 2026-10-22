---
title: TRELLIS.2 in CUDA C++
emoji: 🧊
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
python_version: "3.12"
short_description: TRELLIS.2 image-to-3D run by a pure CUDA C++ port
startup_duration_timeout: 1h
license: other
models:
  - microsoft/TRELLIS.2-4B
  - microsoft/TRELLIS-image-large
  - facebook/dinov3-vitl16-pretrain-lvd1689m
---

Image → textured GLB with [TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B), executed by a C++/CUDA port of
the whole pipeline (`1024_cascade`, `to_glb` with remeshing, 1M-face simplification and 4096² texture baking).
PyTorch is installed only for the CUDA runtime libraries it bundles and for ZeroGPU itself; inference goes through
`lib/libt2.so`, loaded with `ctypes` inside the `@spaces.GPU` worker (`t2.py`).

- `lib/libt2.so` is the port's fast build compiled for sm_120 (the RTX PRO 6000 Blackwell behind ZeroGPU), with
  FlashAttention-2 in place of the Hopper-only FlashAttention-3, and the exact CPU TELEA texture inpaint (the fast
  build's GPU inpaint faults on sm_120). On a full RTX PRO 6000 its GLBs stay within the official torch.compile-vs-eager
  spread of the exact build (Chamfer ≤ 1.4e-3 of the bbox diagonal, base-color L1 ≤ 5.9/255 on the five examples).
- Inputs need a foreground mask: a transparent background is used as is, otherwise rembg (u2net, CPU) removes it.
- Example images are from the [TRELLIS.2 repository](https://github.com/microsoft/TRELLIS.2) (MIT).

`libt2.so` contains nvdiffrast and cuBVH code under NVIDIA's non-commercial source code license, so this build is for
non-commercial use; see `LICENSES/`.
