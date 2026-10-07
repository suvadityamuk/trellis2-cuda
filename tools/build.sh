#!/usr/bin/env bash
# Fetch pinned third-party CUDA sources and build. Usage: tools/build.sh [build_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
FA=third_party/flash-attention
if [ ! -d $FA/csrc/cutlass/include ]; then
  git clone -q --depth 1 -b v2.7.3 https://github.com/Dao-AILab/flash-attention.git $FA
  git -C $FA submodule update --init --depth 1 csrc/cutlass
fi
CF=third_party/cudnn-frontend  # torch v2.6.0 submodule pin
if [ ! -d $CF/include ]; then
  git clone -q https://github.com/NVIDIA/cudnn-frontend.git $CF && git -C $CF checkout -q 936021bfed8c91dc416af1588b2c4eca631a9e45
fi
PT=third_party/pytorch  # torch v2.6.0: mem_eff_attention sources + its cutlass pin
if [ ! -d $PT/aten ]; then
  git clone -q --depth 1 --filter=blob:none --sparse -b v2.6.0 https://github.com/pytorch/pytorch $PT
  git -C $PT sparse-checkout set aten/src/ATen/native/transformers/cuda/mem_eff_attention
fi
CP=third_party/cutlass-pt
if [ ! -d $CP/include ]; then
  git clone -q --filter=blob:none https://github.com/NVIDIA/cutlass.git $CP && git -C $CP checkout -q bbe579a9e3beb6ea6626d9227ec32d0dae119a49
fi
CM=third_party/CuMesh  # pinned by trellis2 setup.sh
if [ ! -d $CM/src ]; then
  git clone -q https://github.com/JeffreyXiang/CuMesh.git $CM && git -C $CM checkout -q 12289e1062f0603f2f0d0771b02e1395d247f26f
  git -C $CM submodule update --init --recursive -q
fi
XA=$CM/third_party/xatlas/xatlas.cpp
grep -q xa_hw.h $XA || sed -i 's/std::thread::hardware_concurrency()/xa_hw()/g; s|^#include <thread>$|#include <thread>\n#include "xa_hw.h"|' $XA
FG=third_party/FlexGEMM  # pinned by trellis2 setup.sh (grid_sample_3d CUDA kernels)
if [ ! -d $FG/flex_gemm ]; then
  git clone -q https://github.com/JeffreyXiang/FlexGEMM.git $FG && git -C $FG checkout -q 6dd94a859c26ee8246888502eada3dd8ad85532e
fi
NV=third_party/nvdiffrast  # pinned by trellis2 setup.sh (v0.4.0)
if [ ! -d $NV/csrc ]; then
  git clone -q https://github.com/NVlabs/nvdiffrast.git $NV && git -C $NV checkout -q 253ac4fcea7de5f396371124af597e6cc957bfae
fi
WP=third_party/libwebp  # Pillow 11.0.0 wheel's bundled libwebp
[ -d $WP/src ] || git clone -q --depth 1 -b v1.4.0 https://github.com/webmproject/libwebp.git $WP
TC=third_party/torch-cpu  # torch v2.6.0 avx_mathfun.h (CPU randn path)
mkdir -p $TC && [ -s $TC/avx_mathfun.h ] || \
  curl -sL https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/aten/src/ATen/native/cpu/avx_mathfun.h | \
  sed 's|#include <ATen/native/cpu/Intrinsics.h>|#include <immintrin.h>|' > $TC/avx_mathfun.h
mkdir -p third_party/stb && [ -f third_party/stb/stb_image.h ] || \
  python3 -c "import urllib.request as u; u.urlretrieve('https://raw.githubusercontent.com/nothings/stb/f58f558c120e9b32c217290b80bad1a0729fbb2c/stb_image.h', 'third_party/stb/stb_image.h')"
B=${1:-build}
cmake -S . -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build $B -j
