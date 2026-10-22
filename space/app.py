import spaces  # before torch: ZeroGPU patches it

import os
import random
import tempfile
import threading

import gradio as gr
import numpy as np
import torch  # also installs the CUDA 12 runtime, cuBLAS and cuDNN that libt2.so links against
from huggingface_hub import hf_hub_download
from PIL import Image
from rembg import new_session, remove

from t2 import STAGES, T2

HERE = os.path.dirname(os.path.abspath(__file__))
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

REMBG = new_session("u2net")


def remove_background(img: Image.Image) -> Image.Image:
    return remove(img.convert("RGB"), session=REMBG)


def prepare(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    a = np.asarray(img)[..., 3]
    return img if (a < 255).any() else remove_background(img)


_model, _model_lock = None, threading.Lock()


def model() -> T2:
    """libt2.so and the weights live in the GPU worker; a warm worker reuses them."""
    global _model
    with _model_lock:
        if _model is None:
            _model = T2(CKPT)
    return _model


def gpu_seconds(img, seed, faces, texture):
    # Full RTX PRO 6000, warm: 16 s at 1024² / 200k faces, 23-52 s at 4096² / 1M; a half-GPU slice can take ~2x that.
    return 120 if int(texture) >= 4096 and faces > 500_000 else 90


@spaces.GPU(duration=gpu_seconds)
def image_to_glb(img: Image.Image, seed: int, faces: int, texture: int):
    if not torch.cuda.is_available():
        raise gr.Error("No GPU attached: this Space must run on ZeroGPU (or another NVIDIA GPU) hardware.")
    out = tempfile.NamedTemporaryFile(suffix=".glb", delete=False).name
    times, nv, nf = model().image_to_glb(np.asarray(img), out, seed=seed, faces=faces, texture=texture)
    return out, times, nv, nf


def generate(img: Image.Image, seed: int, randomize: bool, faces: int, texture: str):
    """Image to textured GLB with TRELLIS.2-4B, run by a CUDA C++ port of the pipeline.

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
    rgba = prepare(img)
    try:
        path, times, nv, nf = image_to_glb(rgba, seed, int(faces), int(texture))
    except RuntimeError as e:
        raise gr.Error(str(e).split(": ", 1)[-1])
    rows = "\n".join(f"| {k} | {times[k]:.1f} |" for k in STAGES)
    info = f"**{sum(times.values()):.1f} s** on GPU, seed {seed}, {nv:,} vertices, {nf:,} faces\n\n| stage | s |\n|---|---|\n{rows}"
    return rgba, path, path, seed, info


EXAMPLES = sorted(os.path.join(HERE, "examples", f) for f in os.listdir(os.path.join(HERE, "examples")))

with gr.Blocks(title="TRELLIS.2 in CUDA C++", delete_cache=(600, 600)) as demo:
    gr.Markdown(
        "# TRELLIS.2 → GLB in pure CUDA C++\n"
        "[TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) image-to-3D, run end to end by a C++/CUDA port "
        "(no PyTorch in the inference path): DINOv3, the four flow transformers, the sparse decoders, remeshing, UV "
        "unwrapping, texture baking and GLB export. This Space runs the port's fast build. "
        "Images without transparency get their background removed on CPU first (rembg).")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(label="Image", type="pil", image_mode="RGBA", height=360)
            with gr.Accordion("Settings", open=False):
                seed = gr.Number(42, label="Seed", precision=0)
                randomize = gr.Checkbox(False, label="Random seed")
                faces = gr.Slider(100_000, 1_000_000, 1_000_000, step=50_000, label="Target faces")
                texture = gr.Radio(["1024", "2048", "4096"], value="4096", label="Texture size")
            btn = gr.Button("Generate", variant="primary")
            used = gr.Image(label="Model input", type="pil", image_mode="RGBA", height=200, interactive=False)
        with gr.Column():
            view = gr.Model3D(label="GLB", height=480, clear_color=(0.25, 0.25, 0.25, 1.0))
            dl = gr.DownloadButton("Download GLB")
            info = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=[inp], outputs=[used, view, dl, seed, info], fn=lambda i: generate(i, 42, False, 1_000_000, "4096"),
                cache_examples=True, cache_mode="lazy")
    btn.click(generate, [inp, seed, randomize, faces, texture], [used, view, dl, seed, info])

demo.launch(mcp_server=True)
