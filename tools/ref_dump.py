"""Run the official TRELLIS.2 pipeline on one image and dump every stage boundary (and block-0 internals of each
model's first call) to safetensors. Offline verification tool only; never used by the CUDA runtime."""
import os, sys, gc, json, time, argparse
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import numpy as np, torch
from safetensors.torch import save_file
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--image", default="/workspace/TRELLIS.2/assets/example_image/T.png")
ap.add_argument("--out", default="/workspace/ref")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--deep", type=int, default=1, help="dump block-0 submodule I/O for first call of each model")
ap.add_argument("--glb", type=int, default=1)
a = ap.parse_args()
sys.path.insert(0, "/workspace/TRELLIS.2")
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.modules.sparse import SparseTensor, VarLenTensor
import o_voxel, cumesh

D = {}
def put(k, v):
    if isinstance(v, (SparseTensor,)):
        put(k + ".feats", v.feats); put(k + ".coords", v.coords); return
    if isinstance(v, VarLenTensor):
        put(k + ".feats", v.feats); return
    if isinstance(v, (list, tuple)):
        for i, x in enumerate(v): put(f"{k}.{i}", x)
        return
    if isinstance(v, np.ndarray): v = torch.from_numpy(np.ascontiguousarray(v))
    if isinstance(v, (int, float)): v = torch.tensor(v, dtype=torch.float64)
    if isinstance(v, torch.Tensor):
        if v.dtype == torch.bool: v = v.to(torch.uint8)
        if v.is_complex(): v = torch.view_as_real(v)
        D[k] = v.detach().to("cpu").contiguous().clone()

def flush(stage):
    global D
    if D:
        save_file(D, f"{a.out}/{stage}.safetensors"); print(f"[dump] {stage}: {len(D)} tensors", flush=True)
    D = {}

os.makedirs(a.out, exist_ok=True)
pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipe.low_vram = False
pipe.cuda()
from trellis2.models.sparse_structure_flow import TimestepEmbedder
put("const/ss_rope_phases", pipe.models["sparse_structure_flow_model"].rope_phases)
put("const/rope_freqs", pipe.models["shape_slat_flow_model_512"].blocks[0].self_attn.rope.freqs)
put("const/t_freqs", torch.exp(-np.log(10000) * torch.arange(start=0, end=128, dtype=torch.float32) / 128))
dino = pipe.image_cond_model
put("const/dino_inv_freq", dino.model.rope_embeddings.inv_freq)
print("[dino]", dino.model.config._attn_implementation, next(dino.model.parameters()).dtype)
_ef = type(dino).extract_features
def ef(self, image):
    r = image.shape[-1]; put(f"dino{r}/in", image)
    hs = [self.model.embeddings.register_forward_hook(lambda m, i, o: put(f"dino{r}/embeddings", o[0])),
          self.model.rope_embeddings.register_forward_hook(lambda m, i, o: (put(f"dino{r}/rope.cos", o[0]), put(f"dino{r}/rope.sin", o[1])) and None)]
    for n in ["norm1", "attention.q_proj", "attention", "mlp"]:
        hs.append(self.model.layer[0].get_submodule(n).register_forward_hook(
            lambda m, i, o, n=n: put(f"dino{r}/layer.0.{n}", (o[0] if isinstance(o, tuple) else o)[0])))
    for li, l in enumerate(self.model.layer):
        hs.append(l.register_forward_hook(lambda m, i, o, li=li: put(f"dino{r}/layer.{li}", o[0])))
    out = _ef(self, image); put(f"dino{r}/out", out[0])
    for h in hs: h.remove()
    return out
type(dino).extract_features = ef
print("[cfg]", {k: str(next(m.parameters()).dtype) for k, m in pipe.models.items()}, "tf32:", torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)

# --- RNG capture -------------------------------------------------------------------------------
_randn, NOISE = torch.randn, []
def randn(*s, **k):
    x = _randn(*s, **k); NOISE.append(x.clone()); return x
torch.randn = randn

# --- flow model I/O + deep hooks ----------------------------------------------------------------
def hook_model(name, m):
    calls = [0]
    def pre(mod, args, kwargs):
        i = calls[0]; p = f"{name}/call{i}"
        x, t, cond = args[:3]
        put(p + "/x", x); put(p + "/t", t)
        if i < 2: put(p + "/cond", cond)
        if "concat_cond" in kwargs and i == 0: put(p + "/concat_cond", kwargs["concat_cond"])
    def post(mod, args, kwargs, out):
        put(f"{name}/call{calls[0]}/out", out); calls[0] += 1
    m.register_forward_pre_hook(pre, with_kwargs=True); m.register_forward_hook(post, with_kwargs=True)
    if a.deep:
        for sn, sm in m.named_modules():
            if sn.startswith("blocks.") and not sn.startswith("blocks.0.") and sn.count(".") == 1:
                sm.register_forward_hook(lambda mod, i_, o, sn=sn: None if calls[0] else put(f"{name}/deep/{sn}.out", o))
            if sn.startswith("blocks.0") or sn in ("input_layer", "t_embedder", "adaLN_modulation", "out_layer"):
                def h(mod, i_, o, sn=sn):
                    if calls[0] == 0:
                        put(f"{name}/deep/{sn}.in", i_[0] if len(i_) else torch.zeros(0)); put(f"{name}/deep/{sn}.out", o)
                sm.register_forward_hook(h)

for k, m in pipe.models.items():
    if "flow" in k: hook_model(k, m)
    elif a.deep:
        for sn, sm in m.named_modules():
            if sn.count(".") <= 2 and sn:
                sm.register_forward_hook(lambda mod, i_, o, sn=sn, k=k: put(f"{k}/deep/{sn}.out", o) if not isinstance(o, tuple) else put(f"{k}/deep/{sn}.out", o[0]))

# --- pipeline stage boundaries -----------------------------------------------------------------
P = Trellis2ImageTo3DPipeline
def wrap(cls, fn, tag, post=None):
    orig = getattr(cls, fn)
    def w(self, *args, **kw):
        t0 = time.time(); out = orig(self, *args, **kw); torch.cuda.synchronize()
        print(f"[time] {tag}: {time.time()-t0:.3f}s", flush=True)
        (post or (lambda o, A: put(tag + "/out", o)))(out, args); return out
    setattr(cls, fn, w)

wrap(P, "preprocess_image", "preprocess", lambda o, A: put("preprocess/out", np.array(o)))
wrap(P, "get_cond", "cond", lambda o, A: (put(f"cond{A[1]}/cond", o["cond"])))
wrap(P, "sample_sparse_structure", "ss", lambda o, A: put("ss/coords", o))
wrap(P, "sample_shape_slat_cascade", "shape", lambda o, A: (put("shape/slat", o[0]), put("shape/res", o[1])))
wrap(P, "sample_tex_slat", "tex", lambda o, A: put("tex/slat", o))
wrap(P, "decode_shape_slat", "dec_shape", lambda o, A: (put("dec_shape/v", o[0][0].vertices), put("dec_shape/f", o[0][0].faces), put("dec_shape/subs", o[1])))
wrap(P, "decode_tex_slat", "dec_tex", lambda o, A: put("dec_tex/out", o[0] if isinstance(o, list) else o))
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler as FS
_fs, NS = FS.sample, [0]
def fs(self, model, noise, *A, **K):
    o = _fs(self, model, noise, *A, **K); put(f"sample{NS[0]}/noise", noise); put(f"sample{NS[0]}/out", o.samples); NS[0] += 1; return o
FS.sample = fs
dec = pipe.models["shape_slat_decoder"]; _up = dec.upsample
def up(x, upsample_times):
    o = _up(x, upsample_times); put("shape/hr_coords", o); return o
dec.upsample = up

img = Image.open(a.image)
torch.cuda.synchronize(); t0 = time.time()
mesh = pipe.run(img, seed=a.seed)[0]
torch.cuda.synchronize(); print(f"[time] pipeline.run: {time.time()-t0:.3f}s", flush=True)
put("final/v", mesh.vertices); put("final/f", mesh.faces); put("final/attrs", mesh.attrs); put("final/coords", mesh.coords)
for i, n in enumerate(NOISE): put(f"noise/{i}", n)
flush("pipeline")

# --- autotune choices (FlexGEMM) ----------------------------------------------------------------
from flex_gemm.utils import autotuner as AT
tune = {}
for o in gc.get_objects():
    try:
        if isinstance(o, tuple(v for v in vars(AT).values() if isinstance(v, type))) and hasattr(o, "cache"):
            nm = getattr(getattr(o, "fn", None), "__name__", str(type(o)))
            tune[nm] = {str(k): str(v) for k, v in o.cache.items()}
    except Exception: pass
json.dump(tune, open(f"{a.out}/autotune.json", "w"), indent=1)

# --- GLB export (official example.py settings) -------------------------------------------------
if a.glb:
    mesh.simplify(16777216)
    put("glb/in_v", mesh.vertices); put("glb/in_f", mesh.faces)
    import o_voxel.postprocess as PP, cv2
    _inp = cv2.inpaint
    def inp(img, m, r, f):
        o = _inp(img, m, r, f); put(f"glb/inpaint{len([k for k in D if 'inpaint' in k])//2}", o); put(f"glb/inpaint_in{len([k for k in D if 'inpaint_in' in k])}", img); return o
    PP.cv2.inpaint = inp
    _gs = PP.grid_sample_3d
    def gs(*A, **K):
        o = _gs(*A, **K); put("glb/grid", K["grid"]); put("glb/attrs", o); return o
    PP.grid_sample_3d = gs
    for fn in ["fill_holes", "simplify", "uv_unwrap", "init"]:
        orig = getattr(cumesh.CuMesh, fn)
        def w(self, *A, _o=orig, _n=fn, **K):
            t0 = time.time(); r = _o(self, *A, **K); torch.cuda.synchronize(); print(f"[time] cumesh.{_n}: {time.time()-t0:.3f}s", flush=True)
            if _n == "uv_unwrap": put("glb/uv_unwrap", list(r))
            else: put(f"glb/{_n}{len([k for k in D if f'glb/{_n}' in k])//2}", list(self.read()))
            return r
        setattr(cumesh.CuMesh, fn, w)
    torch.cuda.synchronize(); t0 = time.time()
    glb = PP.to_glb(vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
                    attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5]*3, [0.5]*3],
                    decimation_target=1000000, texture_size=4096, remesh=True, remesh_band=1, remesh_project=0)
    torch.cuda.synchronize(); print(f"[time] to_glb: {time.time()-t0:.3f}s", flush=True)
    glb.export(f"{a.out}/ref.glb", extension_webp=True)
    flush("glb")
print("done")
