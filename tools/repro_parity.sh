#!/usr/bin/env bash
# Bitwise parity on the repo's example images, after tools/setup_ref.sh and tools/build.sh.
# The official pipeline and the CUDA build share one FlexGEMM autotune cache (assets/flexgemm_autotune.json): the official
# run fills in any missing entries, then both sides use the same sparse-conv tiles. Usage: tools/repro_parity.sh [build_dir]
set -uo pipefail
cd "$(dirname "$0")/.."
B=${1:-build} E=/workspace/TRELLIS.2/assets/example_image
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=$PWD/assets/flexgemm_autotune.json T2_FLEXGEMM_CACHE=$PWD/assets/flexgemm_autotune.json
mkdir -p assets
fail=0
for p in $E/0a34*.webp $E/0e49*.webp $E/0f16*.webp $E/130c*.webp $E/T.png; do
  n=$(basename $p | cut -c1-4) R=/workspace/ref_$n
  [ -f $R/ref.glb ] || python tools/ref_dump.py --image $p --out $R --deep 0 > $R.log 2>&1 || { echo "$n: ref_dump failed"; fail=1; continue; }
  for c in pipe glb bake; do
    T2_REF=$R T2_IMAGE=$p T2_STRICT=1 $B/parity $c > $R.$c.txt 2>&1
    echo "$n $c: $(tail -1 $R.$c.txt)"; grep -q "^PARITY OK" $R.$c.txt || fail=1
  done
done
exit $fail
