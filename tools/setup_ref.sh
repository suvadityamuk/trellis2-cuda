#!/usr/bin/env bash
# Reference (official PyTorch) environment on the H200 job. Used only for parity dumps and baselines.
set -euo pipefail
export MAX_JOBS=${MAX_JOBS:-20} TORCH_CUDA_ARCH_LIST="9.0" FLASH_ATTN_CUDA_ARCHS=90
W=/workspace; mkdir -p $W/ext && cd $W
apt-get update -qq && apt-get install -y -qq git libjpeg-dev libgl1 libglib2.0-0 libwebp-dev >/dev/null
pip install -q imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers==4.56.2 \
  safetensors pandas lpips zstandard kornia timm "huggingface_hub[cli]" pillow
pip install -q git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
pip install -q flash-attn==2.7.3 --no-build-isolation
[ -d TRELLIS.2 ] || git clone -q --recursive https://github.com/microsoft/TRELLIS.2 && (cd TRELLIS.2 && git checkout -q 75fbf0183001ed9876c8dbb35de6b68552ee08bd)
for r in NVlabs/nvdiffrast@v0.4.0 JeffreyXiang/CuMesh JeffreyXiang/FlexGEMM; do
  n=${r%@*}; n=${n#*/}; b=${r#*@}; [ "$b" = "$r" ] && b=""
  [ -d ext/$n ] || git clone -q --recursive ${b:+-b $b} https://github.com/${r%@*} ext/$n
  pip install -q ext/$n --no-build-isolation
done
pip install -q TRELLIS.2/o-voxel --no-build-isolation
hf download microsoft/TRELLIS.2-4B --quiet >/dev/null
hf download microsoft/TRELLIS-image-large ckpts/ss_dec_conv3d_16l8_fp16.json ckpts/ss_dec_conv3d_16l8_fp16.safetensors --quiet >/dev/null
hf download facebook/dinov3-vitl16-pretrain-lvd1689m --quiet >/dev/null
hf download briaai/RMBG-2.0 --quiet >/dev/null || echo "RMBG-2.0 gated: accept license on HF"
mkdir -p $W/ckpts
for r in microsoft/TRELLIS.2-4B microsoft/TRELLIS-image-large; do
  find -L "$(hf download $r --quiet | tail -1)" -name '*.safetensors' -exec ln -sf {} $W/ckpts/ \;
done
ln -sf "$(hf download facebook/dinov3-vitl16-pretrain-lvd1689m --quiet | tail -1)/model.safetensors" $W/ckpts/dinov3_vitl16.safetensors
python -c "import torch,flash_attn,flex_gemm,cumesh,o_voxel,nvdiffrast; print('ref env ok', torch.__version__, flash_attn.__version__)"
