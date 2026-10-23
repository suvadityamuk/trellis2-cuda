import spaces  # before torch: ZeroGPU patches it

import json
import os
import random
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.update(OPENCV_IO_ENABLE_OPENEXR="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", ATTN_BACKEND="flash_attn",
                  FLEX_GEMM_AUTOTUNE_CACHE_PATH=os.path.join(HERE, "assets", "autotune_cache.json"))

import gradio as gr
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from rembg import new_session, remove

import trellis2.pipelines.rembg as trellis2_rembg

trellis2_rembg.BiRefNet = lambda **_: None  # inputs arrive with a mask; skips loading the gated briaai/RMBG-2.0
import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline

CKPT = "/tmp/t2-ckpts"
WEIGHTS = {  # file name expected by Pipeline::load -> (repo, path in repo)
    "dinov3_vitl16.safetensors": ("facebook/dinov3-vitl16-pretrain-lvd1689m", "model.safetensors"),
    "ss_dec_conv3d_16l8_fp16.safetensors": ("microsoft/TRELLIS-image-large", "ckpts/ss_dec_conv3d_16l8_fp16.safetensors"),
    **{f"{n}.safetensors": ("microsoft/TRELLIS.2-4B", f"ckpts/{n}.safetensors") for n in (
        "ss_flow_img_dit_1_3B_64_bf16", "slat_flow_img2shape_dit_1_3B_512_bf16", "slat_flow_img2shape_dit_1_3B_1024_bf16",
        "slat_flow_imgshape2tex_dit_1_3B_1024_bf16", "shape_dec_next_dc_f16c32_fp16", "tex_dec_next_dc_f16c32_fp16")},
}
os.makedirs(CKPT, exist_ok=True)
for name, (repo, path) in WEIGHTS.items():
    dst = os.path.join(CKPT, name)
    if not os.path.exists(dst):
        os.symlink(hf_hub_download(repo, path), dst)

# libt2.so needs the CUDA 12 runtime, cuBLAS and cuDNN; torch here is CUDA 13, so install them apart from site-packages.
CU12 = "/tmp/t2-cu12"
CU12_PKGS = ["nvidia-cuda-runtime-cu12==12.8.90", "nvidia-cublas-cu12==12.8.4.1", "nvidia-cudnn-cu12==9.10.2.21",
             "nvidia-cuda-nvrtc-cu12==12.8.93", "nvidia-nvjitlink-cu12==12.8.93"]
if not os.path.exists(os.path.join(CU12, ".done")):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", CU12, *CU12_PKGS], check=True)
    open(os.path.join(CU12, ".done"), "w").close()
CU12_LIBS = [os.path.join(CU12, "nvidia", d, "lib") for d in ("cuda_runtime", "cublas", "cudnn", "cuda_nvrtc", "nvjitlink")]

# Kept in CPU memory: the port peaks near 37 GB, so both models do not fit the 48 GB slice at once.
PIPE = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
PIPE.low_vram = False

REMBG = new_session("u2net")
STAGES = ("image → mesh", "mesh → textured mesh", "GLB export")
IMPLS = ("CUDA C++ port", "Official PyTorch")


def prepare(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    return img if (np.asarray(img)[..., 3] < 255).any() else remove(img.convert("RGB"), session=REMBG)


def heavy(faces, texture):
    return int(texture) >= 4096 and int(faces) > 500_000


# Measured worst case on the slice (0e49 example, 1M faces / 4096²): 85 s for the port with its weight load, 165 s for
# PyTorch with the weight transfer. The defaults below fit a free account's 300 s daily quota.
def cuda_seconds(path, seed, faces, texture):
    return 120 if heavy(faces, texture) else 90


@spaces.GPU(duration=cuda_seconds)
def run_cuda(path: str, seed: int, faces: int, texture: int):
    out = tempfile.NamedTemporaryFile(suffix=".glb", delete=False).name
    args = dict(ckpt=CKPT, lib_dirs=CU12_LIBS, image=path, out=out, seed=seed, faces=faces, texture=texture)
    env = dict(os.environ, LD_LIBRARY_PATH=":".join(CU12_LIBS + [os.environ.get("LD_LIBRARY_PATH", "")]))
    p = subprocess.run([sys.executable, os.path.join(HERE, "t2_run.py"), json.dumps(args)], capture_output=True, text=True,
                       env=env, cwd=HERE)
    lines = p.stdout.strip().splitlines()
    r = json.loads(lines[-1]) if lines and lines[-1].startswith("{") else {"error": (p.stderr or p.stdout)[-600:]}
    if "error" in r:
        raise gr.Error(f"CUDA C++ port failed: {r['error'].split(': ', 1)[-1]}")
    if not r["fast"]:
        raise gr.Error("CUDA C++ port ran its parity build instead of the fast build.")
    t = r["times"]
    stages = [t["pipeline"], t["remesh"] + t["simplify"] + t["uv_unwrap"] + t["bake"], t["export"]]
    return out, stages, r["load"], r["vertices"], r["faces"]


def python_seconds(path, seed, faces, texture):
    return 210 if heavy(faces, texture) else 150


@spaces.GPU(duration=python_seconds)
def run_python(path: str, seed: int, faces: int, texture: int):
    """TRELLIS.2's own image -> GLB path (pipeline.run, then o_voxel.postprocess.to_glb), timed like the port."""
    def now():
        torch.cuda.synchronize()
        return time.perf_counter()

    out = tempfile.NamedTemporaryFile(suffix=".glb", delete=False).name
    t0 = now()
    PIPE.cuda()
    load = now() - t0
    try:
        s = now()
        mesh = PIPE.run(Image.open(path), seed=seed)[0]
        t1 = now()
        mesh.simplify(16777216)
        glb = o_voxel.postprocess.to_glb(vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
                                         attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5] * 3, [0.5] * 3],
                                         decimation_target=faces, texture_size=texture, remesh=True, remesh_band=1,
                                         remesh_project=0)
        t2 = now()
        glb.export(out, extension_webp=True)
        t3 = now()
        nv, nf = len(glb.vertices), len(glb.faces)
    finally:
        mesh = glb = None
        PIPE.cpu()
        torch.cuda.empty_cache()
    return out, [t1 - s, t2 - t1, t3 - t2], load, nv, nf


def chart(cuda=None, python=None):
    rows = [(impl, st, sec) for impl, r in zip(IMPLS, (cuda, python)) if r for st, sec in zip(STAGES, r)]
    return pd.DataFrame(rows, columns=["implementation", "stage", "seconds"])


def card(stages, load, nv, nf, what):
    rows = "\n".join(f"| {k} | {v:.1f} s |" for k, v in zip(STAGES, stages))
    return (f"### {sum(stages):.1f} s\n{nv:,} vertices, {nf:,} faces\n\n| stage | time |\n|---|---|\n{rows}\n\n"
            f"<sub>{what} {load:.1f} s, not counted.</sub>")


def summary(c, p):
    lines = [f"## {sum(p) / sum(c):.1f}× faster end to end",
             f"**{sum(c):.1f} s** with the CUDA C++ port vs **{sum(p):.1f} s** with the official PyTorch code, same model, "
             "image, seed, settings and GPU.\n", "| stage | CUDA C++ | PyTorch | speedup |", "|---|---|---|---|"]
    lines += [f"| {k} | {a:.1f} s | {b:.1f} s | {b / a:.1f}× |" for k, a, b in zip(STAGES, c, p)]
    return "\n".join(lines)


def compare(img: Image.Image, seed: int, randomize: bool, faces: int, texture: str):
    """Image to textured GLB with TRELLIS.2-4B, run by the CUDA C++ port and then by the official PyTorch code, timed.

    Args:
        img: object image; a transparent background is used as the mask, otherwise the background is removed first.
        seed: sampling seed.
        randomize: pick a random seed instead.
        faces: target face count after simplification.
        texture: base color / metallic-roughness texture size in pixels.
    """
    if img is None:
        raise gr.Error("Upload an image first.")
    seed = random.randint(0, 2**31 - 1) if randomize else int(seed)
    faces, texture = int(faces), int(texture)
    rgba = prepare(img)
    path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    rgba.save(path)
    wait = gr.update()
    yield rgba, seed, None, None, "Running the CUDA C++ port…", None, None, "*Waiting for the CUDA C++ port.*", chart(), \
        "*Running the CUDA C++ port first, then the official PyTorch code.*"
    c_glb, c, c_load, c_nv, c_nf = run_cuda(path, seed, faces, texture)
    c_card = card(c, c_load, c_nv, c_nf, "Weight load:")
    yield wait, wait, c_glb, c_glb, c_card, None, None, "Running the official PyTorch code…", chart(c), \
        f"*CUDA C++ port: {sum(c):.1f} s. Now running the official PyTorch code.*"
    try:
        p_glb, p, p_load, p_nv, p_nf = run_python(path, seed, faces, texture)
    except Exception as e:
        yield rgba, seed, c_glb, c_glb, c_card, None, None, f"**Failed:** {e}", chart(c), \
            f"*CUDA C++ port: {sum(c):.1f} s. The PyTorch run did not complete; see its column.*"
        return
    # Complete, since cached examples keep only the last update.
    yield rgba, seed, c_glb, c_glb, c_card, p_glb, p_glb, card(p, p_load, p_nv, p_nf, "Moving weights to the GPU:"), \
        chart(c, p), summary(c, p)


def compare_example(img: Image.Image):
    yield from compare(img, 42, False, 1_000_000, "4096")


EXAMPLES = sorted(os.path.join(HERE, "examples", f) for f in os.listdir(os.path.join(HERE, "examples")))

with gr.Blocks(title="TRELLIS.2: CUDA C++ vs PyTorch", delete_cache=(600, 600)) as demo:
    gr.Markdown(
        "# TRELLIS.2 image → 3D: CUDA C++ vs PyTorch\n"
        "Each run generates the same textured GLB twice on the same ZeroGPU slice (half an RTX PRO 6000 Blackwell): first with "
        "a C++/CUDA port of the whole [TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) pipeline (no PyTorch in "
        "the inference path), then with Microsoft's own PyTorch code, as their "
        "[TRELLIS.2 Space](https://huggingface.co/spaces/microsoft/TRELLIS.2) runs it (eager, FlashAttention-2, the same "
        "prebuilt CUDA extensions). Both get the same image, seed and settings, and both are timed from the image to the "
        "written GLB with the weights already in memory. The two meshes are close but not identical: the port's fast build "
        "reorders floating-point work. Images without transparency get their background removed on CPU first (rembg). "
        "Examples run at 1M faces with 4096² textures; once one has run, its result is cached and shown instantly.")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(label="Image", type="pil", image_mode="RGBA", height=300)
            with gr.Accordion("Settings", open=False):
                seed = gr.Number(42, label="Seed", precision=0)
                randomize = gr.Checkbox(False, label="Random seed")
                faces = gr.Slider(100_000, 1_000_000, 500_000, step=50_000, label="Target faces")
                texture = gr.Radio(["1024", "2048", "4096"], value="2048", label="Texture size")
                gr.Markdown("1M faces with 4096² textures (the examples' settings) reserves 330 s of ZeroGPU quota, more "
                            "than a free account's daily 300 s.")
            btn = gr.Button("Generate with both", variant="primary")
            used = gr.Image(label="Model input", type="pil", image_mode="RGBA", height=160, interactive=False)
        with gr.Column(scale=3):
            status = gr.Markdown("*Pick an example or upload an image.*")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## CUDA C++ port (fast build)")
                    c_view = gr.Model3D(label="CUDA C++ GLB", height=380, clear_color=(0.25, 0.25, 0.25, 1.0))
                    c_dl = gr.DownloadButton("Download GLB")
                    c_info = gr.Markdown()
                with gr.Column():
                    gr.Markdown("## Official PyTorch")
                    p_view = gr.Model3D(label="PyTorch GLB", height=380, clear_color=(0.25, 0.25, 0.25, 1.0))
                    p_dl = gr.DownloadButton("Download GLB")
                    p_info = gr.Markdown()
            plot = gr.BarPlot(chart(), x="implementation", y="seconds", color="stage", title="Time per stage",
                              y_title="seconds", x_title="", height=260)
    outputs = [used, seed, c_view, c_dl, c_info, p_view, p_dl, p_info, plot, status]
    gr.Examples(EXAMPLES, inputs=[inp], outputs=outputs, fn=compare_example,
                cache_examples=True, cache_mode="lazy")
    btn.click(compare, [inp, seed, randomize, faces, texture], outputs)

demo.launch(mcp_server=True)
