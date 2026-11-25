import spaces  # before torch: ZeroGPU patches it

import html
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.update(OPENCV_IO_ENABLE_OPENEXR="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", ATTN_BACKEND="flash_attn",
                  FLEX_GEMM_AUTOTUNE_CACHE_PATH=os.path.join(HERE, "assets", "autotune_cache.json"))

import gradio as gr
import numpy as np
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
GITHUB = "https://github.com/suvadityamuk/trellis2-cuda"
KERNEL_HUB = "https://huggingface.co/kernels/suvadityamuk/trellis2-cuda"
# measured offline: README.md, H200, 5 example images, warm process, image to written GLB, sums in seconds
H200 = {"eager": 558.4, "compile": 472.7, "fast": 61.4}


def prepare(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    return img if (np.asarray(img)[..., 3] < 255).any() else remove(img.convert("RGB"), session=REMBG)


def heavy(faces, texture):
    return int(texture) >= 4096 and int(faces) > 500_000


# Measured worst case on the slice (0e49 example, 1M faces / 4096²): 85 s for the port with its weight load, 165 s for
# PyTorch with the weight transfer. The defaults below fit a free account's 300 s daily quota.
def cuda_seconds(path, seed, faces, texture):
    return 120 if heavy(faces, texture) else 90


def python_seconds(path, seed, faces, texture):
    return 210 if heavy(faces, texture) else 150


@spaces.GPU(duration=cuda_seconds)
def run_cuda(path: str, seed: int, faces: int, texture: int):
    """libt2.so in a subprocess (see t2_run.py). Yields ('gpu', name), ('tick', elapsed) and finally ('done', result)."""
    yield ("gpu", torch.cuda.get_device_name(0))
    out = tempfile.NamedTemporaryFile(suffix=".glb", delete=False).name
    args = dict(ckpt=CKPT, lib_dirs=CU12_LIBS, image=path, out=out, seed=seed, faces=faces, texture=texture)
    env = dict(os.environ, LD_LIBRARY_PATH=":".join(CU12_LIBS + [os.environ.get("LD_LIBRARY_PATH", "")]))
    with tempfile.TemporaryFile("w+") as so, tempfile.TemporaryFile("w+") as se:
        p = subprocess.Popen([sys.executable, os.path.join(HERE, "t2_run.py"), json.dumps(args)], stdout=so, stderr=se,
                             env=env, cwd=HERE, text=True)
        t0 = time.perf_counter()
        while p.poll() is None:
            time.sleep(0.25)
            yield ("tick", time.perf_counter() - t0)
        so.seek(0), se.seek(0)
        stdout, stderr = so.read(), se.read()
    lines = stdout.strip().splitlines()
    r = json.loads(lines[-1]) if lines and lines[-1].startswith("{") else {"error": (stderr or stdout)[-600:]}
    if "error" in r:
        raise gr.Error(f"trellis2-cuda failed: {r['error'].split(': ', 1)[-1]}")
    if not r["fast"]:
        raise gr.Error("trellis2-cuda ran its parity build instead of the fast build.")
    t = r["times"]
    stages = [t["pipeline"], t["remesh"] + t["simplify"] + t["uv_unwrap"] + t["bake"], t["export"]]
    yield ("done", dict(glb=out, stages=stages, load=r["load"], nv=r["vertices"], nf=r["faces"]))


def _python_glb(path, seed, faces, texture):
    """TRELLIS.2's own image -> GLB path (pipeline.run, then o_voxel.postprocess.to_glb), timed like the port."""
    def now():
        torch.cuda.synchronize()
        return time.perf_counter()

    out = tempfile.NamedTemporaryFile(suffix=".glb", delete=False).name
    t0 = now()
    PIPE.cuda()
    load = now() - t0
    mesh = glb = None
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
    return dict(glb=out, stages=[t1 - s, t2 - t1, t3 - t2], load=load, nv=nv, nf=nf)


POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="t2-torch")


@spaces.GPU(duration=python_seconds)
def run_python(path: str, seed: int, faces: int, texture: int):
    """The official PyTorch code on a worker thread, so the UI can show the elapsed time. Same events as run_cuda."""
    yield ("gpu", torch.cuda.get_device_name(0))
    fut = POOL.submit(_python_glb, path, seed, faces, texture)
    t0 = time.perf_counter()
    while True:
        try:
            r = fut.result(timeout=0.25)
            break
        except FutureTimeout:
            yield ("tick", time.perf_counter() - t0)
    yield ("done", r)


RUN = {"cuda": run_cuda, "torch": run_python}

# ------------------------------------------------------------------------------------------------ UI
# One idea per screen: the same image through both implementations, two big numbers, the speed-up. Settings and
# explanation sit behind two quiet disclosures. No number is shown until it is measured, except the labelled H200 line.
NAME = {"cuda": ("trellis2-cuda", "fast build"), "torch": ("Official PyTorch", "eager")}


def card(impl, state):
    st = state.get(impl) or {}
    name, tag = NAME[impl]
    status = st.get("status")
    big, note, cls = "", "", ""
    if status == "done":
        big = f'{sum(st["stages"]):.1f}<small> s</small>'
        note = f'image to GLB · {st["load"]:.1f} s weight {"load" if impl == "cuda" else "transfer"} not counted'
    elif status == "running":
        big, cls = f'{st.get("elapsed", 0):.0f}<small> s</small>', " is-live"
        note = "running · elapsed, weight " + ("load" if impl == "cuda" else "transfer") + " included"
    elif status == "waiting":
        big, cls, note = "Next", " is-word is-dim", "starts when trellis2-cuda finishes"
    elif status == "skipped":
        big, cls, note = "Not run", " is-word is-dim", st.get("why", "Compare runs both")
    elif status == "failed":
        big, cls, note = "Failed", " is-word is-dim", html.escape(st.get("error", ""))[:160]
    else:
        cls, note = " is-empty", "press Compare to time it on this GPU"
    return (f'<div class="tc-card tc-{impl}"><div class="tc-name">{name} <span>{tag}</span></div>'
            f'<div class="tc-num{cls}">{big or "&nbsp;"}</div><div class="tc-note">{note}</div></div>')


def short_gpu(name):
    return (name or "this GPU").replace("NVIDIA ", "").replace(" Server Edition", "").split(" MIG ")[0]


def hero(state):
    c, p = state.get("cuda") or {}, state.get("torch") or {}
    if c.get("status") == "done" and p.get("status") == "done":
        ratio = sum(p["stages"]) / sum(c["stages"])
        return (f'<div class="tc-hero is-result"><h1><span class="tc-x">{ratio:.1f}×</span> faster</h1>'
                f'<p>trellis2-cuda vs the official TRELLIS.2 code, same image, seed and settings, image to textured '
                f'GLB, measured just now on {html.escape(short_gpu(state.get("gpu")))}.</p></div>')
    return ('<div class="tc-hero"><h1>One image in.<br><span class="tc-x">A textured 3D model out.</span></h1>'
            '<p>TRELLIS.2 rebuilt in CUDA C++, no PyTorch at runtime. Compare runs it and the official PyTorch code '
            'on this GPU with the same seed, and times both.</p></div>')


def details(state):
    rows = []
    if state.get("gpu"):
        rows.append(f'GPU: {html.escape(state["gpu"])}; seed {state.get("seed")}; target {state["faces"]:,} faces, '
                    f'{state["texture"]}² textures.')
    for impl in ("cuda", "torch"):
        st = state.get(impl) or {}
        if st.get("status") == "done":
            stages = ", ".join(f"{k} {v:.1f} s" for k, v in zip(STAGES, st["stages"]))
            rows.append(f'{NAME[impl][0]}: {sum(st["stages"]):.2f} s ({stages}); {st["nv"]:,} vertices, '
                        f'{st["nf"]:,} faces; weight {"load" if impl == "cuda" else "transfer to the GPU"} '
                        f'{st["load"]:.1f} s, not counted.')
        elif st.get("status") == "failed":
            rows.append(f'{NAME[impl][0]}: failed: {html.escape(st.get("error", ""))}')
    return '<div class="tc-run">' + "".join(f"<p>{r}</p>" for r in rows) + "</div>" if rows else ""


def generate(img, seed, randomize, faces, texture, compare=True):
    if img is None:
        raise gr.Error("Upload an image or pick an example first.")
    seed = random.randint(0, 2**31 - 1) if randomize else int(seed)
    faces, texture = int(faces), int(texture)
    rgba = prepare(img)
    path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    rgba.save(path)
    state = {"cuda": None, "torch": {"status": "waiting" if compare else "skipped"}, "seed": seed, "faces": faces,
             "texture": texture}
    files, sent = {"cuda": None, "torch": None}, {}

    def emit(first=False):
        # Send each GLB once: they are tens of MB at high settings, and re-sending one with every tick reloads it.
        def once(impl):
            if impl in sent and sent[impl] == files[impl]:
                return gr.skip()
            sent[impl] = files[impl]
            return files[impl]
        return (rgba if first else gr.skip(), once("cuda"), once("torch"), card("cuda", state), card("torch", state),
                hero(state), details(state), seed)

    yield emit(first=True)
    for impl in (["cuda", "torch"] if compare else ["cuda"]):
        state[impl] = {"status": "running", "elapsed": 0}
        yield emit()
        try:
            for ev in RUN[impl](path, seed, faces, texture):
                if ev[0] == "gpu":
                    state["gpu"] = ev[1]
                elif ev[0] == "tick":
                    state[impl]["elapsed"] = ev[1]
                elif ev[0] == "done":
                    state[impl] = dict(ev[1], status="done")
                    files[impl] = ev[1]["glb"]
                yield emit()
        except Exception as e:  # noqa: BLE001
            msg = str(e) or type(e).__name__
            state[impl] = {"status": "failed", "error": msg}
            if impl == "cuda":  # nothing to compare against: show it on the card, then as a toast
                if compare:
                    state["torch"] = {"status": "skipped", "why": "not run: trellis2-cuda did not finish"}
                yield emit()
                raise gr.Error(msg) from e
            yield emit()


def compare(img: Image.Image, seed: int, randomize: bool, faces: int, texture: str):
    """Image to textured GLB with TRELLIS.2-4B, run by trellis2-cuda and then by the official PyTorch code, timed.

    Args:
        img: object image; a transparent background is used as the mask, otherwise the background is removed first.
        seed: sampling seed.
        randomize: pick a random seed instead.
        faces: target face count after simplification.
        texture: base color / metallic-roughness texture size in pixels.
    """
    yield from generate(img, seed, randomize, faces, texture, compare=True)


def cuda_only(img: Image.Image, seed: int, randomize: bool, faces: int, texture: str):
    """Image to textured GLB with TRELLIS.2-4B, run by trellis2-cuda only, timed. Arguments as for compare."""
    yield from generate(img, seed, randomize, faces, texture, compare=False)


EXAMPLES = ["T.png"] + sorted(f for f in os.listdir(os.path.join(HERE, "examples")) if f != "T.png")
PRE = os.path.join(HERE, "precomputed")  # GLBs from one live Compare of T.png at the default settings, if present
CLEAR = (0.0, 0.0, 0.0, 1.0)  # = --tc-stage below

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700;800&'
         'family=Geist+Mono:wght@400;500&display=swap" rel="stylesheet">')

CSS = """
:root, .light { --tc-bg: #fbfbfd; --tc-fg: #111114; --tc-mute: #6e6e76; --tc-line: #e4e4ea; --tc-accent: #4c8c00;
  --tc-panel: #f0f0f3; --tc-stage: #000; }
.dark { --tc-bg: #121214; --tc-fg: #f5f5f7; --tc-mute: #94949c; --tc-line: #2a2a2f; --tc-accent: #76b900;
  --tc-panel: #1c1c1f; --tc-stage: #000; }
body, gradio-app, .gradio-container, .main, .wrap.app { background: var(--tc-bg) !important; }
.gradio-container { font-family: 'Geist', ui-sans-serif, system-ui, -apple-system, sans-serif !important;
  color: var(--tc-fg) !important; max-width: 1480px !important; margin: 0 auto !important; padding: 0 clamp(16px, 3vw, 40px) !important; }
.gradio-container * { font-family: inherit; }
footer { display: none !important; }
.gradio-container, .gradio-container .main, .gradio-container .wrap, .gradio-container main.contain { min-width: 0 !important; }
.gradio-container .main { padding: 12px 0 0 !important; }
.html-container { padding: 0 !important; }

/* top bar */
.tc-bar { display: flex; align-items: center; justify-content: space-between; gap: 16px;
  padding: clamp(8px, 2vh, 18px) 0 clamp(4px, 1vh, 8px); }
.tc-mark { font-weight: 700; font-size: 17px; letter-spacing: -0.02em; color: var(--tc-fg) !important; }
.tc-mark b { color: var(--tc-accent) !important; font-weight: 700; }
.tc-links { display: flex; gap: 22px; font-size: 14px; }
.tc-links a { color: var(--tc-mute) !important; text-decoration: none !important; }
.tc-links a:hover { color: var(--tc-fg) !important; }

/* hero */
.tc-hero { padding: clamp(4px, min(1.6vw, 2.4vh), 26px) 0 clamp(14px, min(2vw, 3.4vh), 30px); }
.tc-hero h1 { font-size: clamp(36px, min(5.2vw, 8.6vh), 78px) !important; line-height: 1.02 !important; letter-spacing: -0.045em;
  font-weight: 700 !important; margin: 0 0 14px !important; color: var(--tc-fg) !important; text-wrap: balance; }
.tc-hero.is-result h1 { font-size: clamp(52px, min(8vw, 13vh), 124px) !important; letter-spacing: -0.05em; }
.tc-x { color: var(--tc-accent) !important; font-variant-numeric: tabular-nums; }
.tc-hero p { font-size: clamp(15px, 1.35vw, 20px) !important; line-height: 1.4 !important; color: var(--tc-mute) !important;
  margin: 0 !important; max-width: 72ch; text-wrap: pretty; }

/* stage: input + two dark panes */
#tc-stage { gap: clamp(12px, 1.4vw, 20px) !important; align-items: stretch; flex-wrap: nowrap; }
#tc-stage > .column { gap: 0 !important; }
.tc-pane { background: var(--tc-stage) !important; border-radius: 22px !important; overflow: hidden; }
.tc-pane .block, .tc-pane .html-container, .tc-pane .prose { background: transparent !important; border: 0 !important;
  box-shadow: none !important; padding: 0 !important; }
.tc-pane .model3D { height: clamp(190px, calc(100vh - 500px), 480px) !important; }
.tc-pane > .block, .tc-input > .block { height: auto !important; }
.tc-pane canvas { display: block; width: 100% !important; height: 100% !important; }
.tc-pane .icon-button-wrapper { background: transparent !important; border: 0 !important; box-shadow: none !important;
  top: 10px !important; right: 12px !important; }
.tc-pane .icon-button { color: #8f8f98 !important; background: transparent !important; border: 0 !important; }
.tc-pane .empty, .tc-pane .unpadded_box { background: transparent !important; color: #3a3a40 !important; min-height: 0 !important; }
.tc-card { padding: clamp(14px, 1.6vw, 24px) clamp(16px, 1.8vw, 28px) 0; color: #f5f5f7 !important; }
.tc-name { font-size: clamp(15px, 1.35vw, 22px); font-weight: 600; letter-spacing: -0.01em; color: #f5f5f7 !important; }
.tc-name span { color: #8f8f98 !important; font-weight: 500; }
.tc-cuda .tc-name { color: #8fd61a !important; }
.tc-num { font-size: clamp(48px, min(5.8vw, 8.6vh), 96px); font-weight: 700; letter-spacing: -0.045em; line-height: 1.02;
  font-variant-numeric: tabular-nums; color: #fff !important; margin-top: 2px; white-space: nowrap; }
.tc-num small { font-size: 0.42em; letter-spacing: -0.01em; font-weight: 600; color: #a1a1aa !important; }
.tc-cuda .tc-num { color: #8fd61a !important; }
.tc-num.is-live, .tc-cuda .tc-num.is-live { color: #d4d4d8 !important; }
.tc-num.is-word { font-size: clamp(30px, 3vw, 48px); letter-spacing: -0.035em; padding: 0.18em 0 0.12em; }
.tc-num.is-dim, .tc-cuda .tc-num.is-dim { color: #6e6e76 !important; }
.tc-num.is-empty { display: none; }
.tc-note { font-size: clamp(12px, 1vw, 15px); color: #8f8f98 !important; margin-top: 4px; min-height: 1.4em;
  overflow-wrap: anywhere; }
.tc-input { background: var(--tc-panel) !important; border-radius: 22px !important; padding: 12px !important;
  justify-content: space-between; gap: 12px !important; }
.tc-input > .block { flex: 1 1 auto; display: flex; align-items: flex-start; justify-content: center; }
.tc-input .image-container { height: auto !important; position: relative; }
.tc-input img { border-radius: 14px; max-height: clamp(150px, calc(100vh - 460px), 420px); }
.tc-input .icon-button-wrapper { position: absolute !important; top: 6px !important; right: 6px !important; z-index: 5;
  background: rgba(0,0,0,.45) !important; border: 0 !important; border-radius: 999px !important; }
.tc-input .icon-button { color: #fff !important; }
.tc-input .block { background: transparent !important; border: 0 !important; box-shadow: none !important; }
.tc-input .image-container, .tc-input .upload-container { border-radius: 14px; overflow: hidden; }
#tc-go { flex: 0 0 auto !important; border-radius: 999px !important; font-size: 17px !important; font-weight: 600 !important;
  padding: 14px 18px !important; background: var(--tc-accent) !important; color: #fff !important; border: 0 !important;
  box-shadow: none !important; }
.dark #tc-go { color: #0b0b0c !important; }
#tc-go:hover { filter: brightness(1.08); }
.tc-fine { font-size: 13px; color: var(--tc-mute) !important; margin: 14px 2px 0; line-height: 1.45; }

/* examples strip */
.tc-strip-label { font-size: 14px; color: var(--tc-mute) !important; margin: 30px 2px 8px; }
#tc-examples { background: transparent !important; border: 0 !important; padding: 0 !important; max-width: 760px; }
#tc-examples .grid-wrap { padding: 0 !important; overflow: visible !important; max-height: none !important;
  height: auto !important; min-height: 0 !important; }
#tc-examples, #tc-examples .gallery-container { height: auto !important; min-height: 0 !important; }
#tc-examples .grid-container { gap: 10px !important; grid-template-columns: repeat(5, minmax(0, 1fr)) !important;
  min-width: 0 !important; }
#tc-examples .gallery-item, #tc-examples .thumbnail-item { min-width: 0 !important; width: 100% !important;
  height: auto !important; aspect-ratio: 1; }
#tc-examples .thumbnail-item { border-radius: 14px !important; border: 0 !important; box-shadow: none !important;
  background: var(--tc-panel) !important; }
#tc-examples .thumbnail-item img { object-fit: contain !important; padding: 6px; }
#tc-examples .thumbnail-item:hover { transform: translateY(-2px); }
#tc-examples .icon-button-wrapper, #tc-examples .block-label { display: none !important; }

/* quiet disclosures */
.tc-more { margin-top: 28px !important; gap: 10px !important; }
.tc-more > .block, .tc-acc { background: transparent !important; border: 0 !important; border-top: 1px solid var(--tc-line) !important;
  border-radius: 0 !important; box-shadow: none !important; padding: 6px 0 !important; }
.tc-acc .label-wrap { font-size: 15px !important; color: var(--tc-mute) !important; }
.tc-acc .label-wrap span { font-weight: 500 !important; }
.tc-doc { font-size: 14px; line-height: 1.55; color: var(--tc-mute) !important; max-width: 75ch; }
.tc-doc p { margin: 0 0 10px; }
.tc-doc a { color: var(--tc-accent) !important; padding: 0 !important; margin: 0 !important; }
.tc-acc .form { max-width: 720px; }
.tc-run { font-size: 13px; color: var(--tc-mute) !important; font-family: 'Geist Mono', ui-monospace, monospace !important;
  overflow-wrap: anywhere; }
.tc-run p { margin: 0 0 4px; }
.tc-end { height: 40px; }

/* narrow screens: stack, input first, panes full width */
@media (max-width: 900px) {
  #tc-stage { flex-wrap: wrap !important; }
  #tc-stage > .tc-input { flex: 1 1 100% !important; min-width: 100% !important; flex-direction: row !important;
    gap: 14px !important; align-items: center !important; flex-wrap: nowrap !important; }
  #tc-stage > .tc-input > .block { flex: 0 0 120px !important; max-width: 120px; justify-content: flex-start !important; }
  .tc-input img { max-height: 120px; }
  #tc-stage > .tc-input #tc-go { flex: 1 1 0 !important; min-width: 0 !important; align-self: center; }
  .tc-pane .model3D { height: 300px !important; }
  #tc-stage > .tc-pane { flex: 1 1 calc(50% - 10px) !important; min-width: 280px !important; }
  .tc-links { gap: 14px; }
}
@media (max-width: 620px) {
  #tc-stage > .tc-pane { flex: 1 1 100% !important; min-width: 100% !important; }
}
"""

BAR = (f'<div class="tc-bar"><div class="tc-mark">trellis2<b>-</b>cuda</div><div class="tc-links">'
       f'<a href="{GITHUB}" target="_blank" rel="noopener">GitHub</a>'
       f'<a href="{KERNEL_HUB}" target="_blank" rel="noopener">Kernels</a></div></div>')
FINE = (f'Baseline: the official TRELLIS.2 PyTorch code (eager), on this GPU. Measured offline on an H200 over five '
        f'images: official eager {H200["eager"]:.0f} s, trellis2-cuda {H200["fast"]:.1f} s, '
        f'{H200["eager"] / H200["fast"]:.1f}× faster.')
DOC = f"""<div class="tc-doc">
<p>TRELLIS.2 turns one image into a textured 3D mesh. trellis2-cuda runs the whole pipeline, from the image encoder
to the GLB writer, as one C++/CUDA library with no PyTorch: hand-written kernels for the transformers and sparse
convolutions, a GPU BVH for remeshing, parallel UV unwrapping, GPU texture inpainting and fused DiT kernels.</p>
<p>The baseline is Microsoft's own code as their TRELLIS.2 Space runs it: eager PyTorch with FlashAttention-2 and the
same prebuilt CUDA extensions. Both get the same image, seed and settings. The meshes are close but not identical,
because the fast build reorders floating-point work. Measured offline on an H200 over five images: official eager
{H200["eager"]:.1f} s, official torch.compile {H200["compile"]:.1f} s, trellis2-cuda {H200["fast"]:.1f} s.</p>
<p>Times run from the image to the written GLB. Loading the port's weights and moving the PyTorch weights to the GPU
are not counted; both are listed below after a run. Images without transparency get their background removed on the
CPU first (not timed). Source, method and benchmarks: <a href="{GITHUB}">GitHub</a>.</p></div>"""

theme = gr.themes.Base(
    font=[gr.themes.Font(f) for f in ("Geist", "ui-sans-serif", "system-ui", "sans-serif")],
    font_mono=[gr.themes.Font(f) for f in ("Geist Mono", "ui-monospace", "monospace")],
    radius_size=gr.themes.sizes.radius_lg,
).set(body_background_fill="#fbfbfd", body_background_fill_dark="#121214",
      block_background_fill="transparent", block_background_fill_dark="transparent",
      block_border_width="0px", block_shadow="none", block_label_border_width="0px",
      button_primary_background_fill="#4c8c00", button_primary_background_fill_dark="#76b900",
      button_primary_text_color="#ffffff", button_primary_text_color_dark="#0b0b0c",
      color_accent="#4c8c00", color_accent_soft="#eef6e3", color_accent_soft_dark="#1c2a0b",
      checkbox_background_color_selected="#4c8c00", slider_color="#4c8c00", slider_color_dark="#76b900")


def pre(name):
    p = os.path.join(PRE, name)
    return p if os.path.exists(p) else None


def pre_card(impl):
    c = card(impl, {})
    return c.replace("press Compare to time it on this GPU", "example output · not timed yet") if pre(f"T_{impl}.glb") else c


with gr.Blocks(title="trellis2-cuda: TRELLIS.2 in CUDA C++", delete_cache=(600, 600)) as demo:
    gr.HTML(BAR, padding=False)
    hero_h = gr.HTML(hero({}), padding=False)
    with gr.Row(elem_id="tc-stage", equal_height=True):
        with gr.Column(scale=3, min_width=200, elem_classes="tc-input"):
            inp = gr.Image(value=os.path.join(HERE, "examples", "T.png"), type="pil", image_mode="RGBA", show_label=False,
                           sources=["upload"], height=300, container=False, buttons=[],
                           placeholder="Drop an image of one object")
            btn = gr.Button("Compare", variant="primary", elem_id="tc-go")
        with gr.Column(scale=5, min_width=280, elem_classes="tc-pane"):
            card_c = gr.HTML(pre_card("cuda"), padding=False)
            view_c = gr.Model3D(value=pre("T_cuda.glb"), show_label=False, height=360, clear_color=CLEAR, container=False)
        with gr.Column(scale=5, min_width=280, elem_classes="tc-pane"):
            card_p = gr.HTML(pre_card("torch"), padding=False)
            view_p = gr.Model3D(value=pre("T_torch.glb"), show_label=False, height=360, clear_color=CLEAR, container=False)
    gr.HTML(f'<p class="tc-fine">{FINE}</p>', padding=False)
    gr.HTML('<div class="tc-strip-label">Or try one of these</div>', padding=False)
    gallery = gr.Gallery([os.path.join(HERE, "examples", f) for f in EXAMPLES], columns=5, rows=1, height="auto",
                         allow_preview=False, show_label=False, object_fit="contain", elem_id="tc-examples",
                         interactive=False, buttons=[], container=False)
    with gr.Column(elem_classes="tc-more"):
        with gr.Accordion("Options", open=False, elem_classes="tc-acc"):
            faces = gr.Slider(100_000, 1_000_000, 500_000, step=50_000, label="Target faces")
            texture = gr.Radio(["1024", "2048", "4096"], value="2048", label="Texture size")
            seed = gr.Number(42, precision=0, label="Seed (both use it)")
            randomize = gr.Checkbox(False, label="Random seed")
            gr.HTML('<div class="tc-doc"><p>Compare reserves up to 240 s of ZeroGPU time at these defaults, and 330 s '
                    'at 1M faces with 4096² textures. trellis2-cuda alone reserves 90 s.</p></div>', padding=False)
            btn_cuda = gr.Button("Run trellis2-cuda only", variant="secondary", size="sm")
        with gr.Accordion("Details", open=False, elem_classes="tc-acc"):
            gr.HTML(DOC, padding=False)
            info = gr.HTML("", padding=False)
    gr.HTML('<div class="tc-end"></div>', padding=False)

    def pick(evt: gr.SelectData):
        return os.path.join(HERE, "examples", EXAMPLES[evt.index])
    gallery.select(pick, None, inp, show_progress="hidden", api_name=False)

    ins = [inp, seed, randomize, faces, texture]
    outs = [inp, view_c, view_p, card_c, card_p, hero_h, info, seed]
    btn.click(compare, ins, outs, concurrency_limit=1, concurrency_id="gpu", show_progress="hidden")
    btn_cuda.click(cuda_only, ins, outs, concurrency_limit=1, concurrency_id="gpu", show_progress="hidden")

demo.queue(max_size=20, default_concurrency_limit=1).launch(css=CSS, head=FONTS, theme=theme, footer_links=[],
                                                           mcp_server=True)
