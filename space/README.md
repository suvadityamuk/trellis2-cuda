---
title: TRELLIS.2 in CUDA C++
emoji: 🧊
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.10.0
app_file: app.py
python_version: "3.12"
short_description: TRELLIS.2 image-to-3D, CUDA C++ port vs PyTorch, timed
startup_duration_timeout: 1h
license: other
models:
  - microsoft/TRELLIS.2-4B
  - microsoft/TRELLIS-image-large
  - facebook/dinov3-vitl16-pretrain-lvd1689m
---

Image → textured GLB with [TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B), run twice on the same GPU and
timed side by side: once by a C++/CUDA port of the whole pipeline (`1024_cascade`, `to_glb` with remeshing, face
simplification and texture baking), once by the official PyTorch code.

- The port is `lib/libt2.so`, called through `ctypes` by `t2_run.py`, a subprocess of the `@spaces.GPU` worker. It
  links the CUDA 12 runtime, cuBLAS and cuDNN, installed at startup into `/tmp/t2-cu12`; the app's torch is CUDA 13,
  and both cuDNN builds use the soname `libcudnn.so.9`, so they cannot share one process.
- The PyTorch side is the `trellis2` package and the requirements of Microsoft's
  [TRELLIS.2 Space](https://huggingface.co/spaces/microsoft/TRELLIS.2) (torch 2.11, FlashAttention-2, prebuilt sm_120
  FlexGEMM, CuMesh, nvdiffrast and o_voxel wheels, their FlexGEMM autotune cache), run eagerly as in their `example.py`.
  It stays in CPU memory between runs because both pipelines together exceed the 48 GB ZeroGPU slice; moving it to
  the GPU, like the port's weight load, is reported but not counted.

- `lib/libt2.so` is the port's fast build compiled for sm_120 (the RTX PRO 6000 Blackwell behind ZeroGPU), with
  FlashAttention-2 in place of the Hopper-only FlashAttention-3, and the exact CPU TELEA texture inpaint (the fast
  build's GPU inpaint faults on sm_120). On a full RTX PRO 6000 its GLBs stay within the official torch.compile-vs-eager
  spread of the exact build (Chamfer ≤ 1.4e-3 of the bbox diagonal, base-color L1 ≤ 5.9/255 on the five examples).
- Inputs need a foreground mask: a transparent background is used as is, otherwise rembg (u2net, CPU) removes it.
- Example images are from the [TRELLIS.2 repository](https://github.com/microsoft/TRELLIS.2) (MIT).

`libt2.so` contains nvdiffrast and cuBVH code under NVIDIA's non-commercial source code license, so this build is for
non-commercial use; see `LICENSES/`.
