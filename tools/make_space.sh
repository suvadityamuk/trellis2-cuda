#!/usr/bin/env bash
# Assemble the Hugging Face Space in space/ from an sm_120 build (T2_CMAKE_ARGS=-DT2_CUDA_ARCH=120 tools/build.sh build120).
# Usage: tools/make_space.sh <build_dir> <TRELLIS.2 checkout>; then: hf upload <ns>/<space> space . --repo-type space
set -euo pipefail
cd "$(dirname "$0")/.."
B=$1 TR=$2 S=space TP=third_party
mkdir -p $S/lib $S/assets $S/examples $S/LICENSES
cp $B/libt2.so $S/lib/
cp assets/flexgemm_autotune.json $S/assets/
E=$TR/assets/example_image
cp $E/0a34*.webp $E/0e49*.webp $E/0f16*.webp $E/130c*.webp $E/T.png $S/examples/
cp LICENSE $S/LICENSES/trellis2-cuda-LICENSE; cp NOTICE $S/LICENSES/trellis2-cuda-NOTICE
cp $TR/LICENSE $S/LICENSES/TRELLIS.2-LICENSE
for p in nvdiffrast/LICENSE.txt CuMesh/LICENSE CuMesh/third_party/cubvh/LICENSE CuMesh/third_party/xatlas/LICENSE \
         FlexGEMM/LICENSE flash-attention/LICENSE flash-attention/csrc/cutlass/LICENSE.txt cutlass-pt/LICENSE.txt \
         cudnn-frontend/LICENSE.txt libwebp/COPYING; do
  [ -f $TP/$p ] && cp $TP/$p "$S/LICENSES/$(echo ${p%.txt} | tr / -)" || echo "missing license: $TP/$p"
done
sed -n '2,/^\*\//p' $TP/torch-cpu/avx_mathfun.h > $S/LICENSES/avx_mathfun
sed -n '/^This software is available under 2 licenses/,$p' $TP/stb/stb_image.h | sed -n '1,/^\*\//p' > $S/LICENSES/stb_image
ls $S/LICENSES
