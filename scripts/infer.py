#!/usr/bin/env python3
"""Image -> textured GLB with trellis2-cuda, in one command.

  python scripts/infer.py image.png -o out.glb

On first use it downloads the checkpoints (about 16 GB; DINOv3 is gated, so accept its licence on the Hub and log in
with `hf auth login` first) and builds the binary for this GPU with tools/build.sh. After that each call only runs
build/t2. The fast build is the default; --parity runs the build that is bitwise identical to the official pipeline.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = {  # file name expected by Pipeline::load (src/pipeline.cu) -> (repo, path in repo)
    "dinov3_vitl16.safetensors": ("facebook/dinov3-vitl16-pretrain-lvd1689m", "model.safetensors"),
    "ss_dec_conv3d_16l8_fp16.safetensors": ("microsoft/TRELLIS-image-large", "ckpts/ss_dec_conv3d_16l8_fp16.safetensors"),
    **{f"{n}.safetensors": ("microsoft/TRELLIS.2-4B", f"ckpts/{n}.safetensors") for n in (
        "ss_flow_img_dit_1_3B_64_bf16", "slat_flow_img2shape_dit_1_3B_512_bf16", "slat_flow_img2shape_dit_1_3B_1024_bf16",
        "slat_flow_imgshape2tex_dit_1_3B_1024_bf16", "shape_dec_next_dc_f16c32_fp16", "tex_dec_next_dc_f16c32_fp16")},
}


def die(msg):
    sys.exit(f"error: {msg}")


def gpu_arch():
    try:
        cc = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True,
                            text=True, check=True).stdout.split()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        die("no NVIDIA GPU found (nvidia-smi failed)")
    arch = cc.replace(".", "")
    if arch not in ("90", "120"):
        print(f"warning: compute capability {cc} is untested; building for sm_{arch} without FlashAttention-3",
              file=sys.stderr)
    return arch


def checkpoints(ckpt):
    os.makedirs(ckpt, exist_ok=True)
    missing = [n for n in WEIGHTS if not os.path.exists(os.path.join(ckpt, n))]
    if not missing:
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        die("pip install huggingface_hub (needed once, to download the checkpoints)")
    for name in missing:
        repo, path = WEIGHTS[name]
        print(f"downloading {repo}/{path}", flush=True)
        try:
            src = hf_hub_download(repo, path)
        except Exception as e:  # noqa: BLE001
            hint = " (gated: accept the licence on its Hub page, then `hf auth login`)" if "dinov3" in repo else ""
            die(f"could not download {repo}/{path}{hint}: {e}")
        dst = os.path.join(ckpt, name)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)


def binary(build_dir, arch, rebuild):
    exe = os.path.join(build_dir, "t2")
    if os.path.exists(exe) and not rebuild:
        return exe
    for tool in ("nvcc", "cmake", "ninja", "git"):
        if not shutil.which(tool):
            die(f"{tool} not found; the build needs CUDA 12.4+ (nvcc), cuDNN 9, CMake >= 3.24, Ninja and git")
    print(f"building {exe} for sm_{arch} (first run only, 10-20 min)", flush=True)
    env = dict(os.environ, T2_CMAKE_ARGS=f"-DT2_CUDA_ARCH={arch} " + os.environ.get("T2_CMAKE_ARGS", ""))
    subprocess.run([os.path.join(ROOT, "tools", "build.sh"), build_dir], env=env, check=True)
    return exe


def cut_out(image, tmp):
    """The pipeline needs a foreground mask: a transparent background is used as is, otherwise rembg removes it."""
    from PIL import Image
    im = Image.open(image)
    if im.mode in ("RGBA", "LA", "PA") and im.getchannel("A").getextrema()[0] < 255:
        return image
    try:
        from rembg import remove
    except ImportError:
        die(f"{image} has no transparent background; give a cut-out PNG, or pip install rembg onnxruntime to remove it")
    print("removing the background (rembg, u2net)", flush=True)
    remove(im.convert("RGB")).save(tmp)
    return tmp


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="PNG / JPEG / WebP of one object; a transparent background is used as the mask")
    p.add_argument("-o", "--out", help="output .glb (default: <image>.glb)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--parity", action="store_true", help="bitwise-identical build instead of the fast build")
    p.add_argument("--profile", action="store_true", help="print per-stage times (T2_PROF=1)")
    p.add_argument("--ckpt", default=os.environ.get("T2_CKPT", os.path.expanduser("~/.cache/trellis2-cuda/ckpts")),
                   help="checkpoint directory (default: $T2_CKPT or ~/.cache/trellis2-cuda/ckpts)")
    p.add_argument("--build-dir", help="default: build (sm_90) or build<arch>")
    p.add_argument("--rebuild", action="store_true", help="run tools/build.sh even if the binary exists")
    a = p.parse_args()
    if not os.path.exists(a.image):
        die(f"{a.image} not found")
    out = os.path.abspath(a.out or os.path.splitext(a.image)[0] + ".glb")

    arch = gpu_arch()
    build_dir = os.path.abspath(a.build_dir or os.path.join(ROOT, "build" if arch == "90" else f"build{arch}"))
    exe = binary(build_dir, arch, a.rebuild)
    checkpoints(a.ckpt)
    image = cut_out(a.image, os.path.splitext(out)[0] + "_input.png")

    env = dict(os.environ, T2_CKPT=a.ckpt, T2_FAST="0" if a.parity else "1")
    if a.profile:
        env["T2_PROF"] = "1"
    env.setdefault("T2_FLEXGEMM_CACHE", os.path.join(ROOT, "assets", "flexgemm_autotune.json"))
    print(f"{'parity' if a.parity else 'fast'} build, seed {a.seed}: {image} -> {out}", flush=True)
    t = time.perf_counter()
    r = subprocess.run([exe, image, out, str(a.seed)], env=env)
    if r.returncode != 0:
        die(f"build/t2 exited with code {r.returncode}")
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB) in {time.perf_counter() - t:.1f} s, including the weight load")


if __name__ == "__main__":
    main()
