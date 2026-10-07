"""Official TRELLIS.2 image -> GLB timing (example.py settings), eager or torch.compile. Offline baseline only.
Usage: python bench_ref.py --mode {eager,compile} --images a.png b.webp ... --out times.json"""
import os, sys, time, json, argparse
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import torch
from PIL import Image
sys.path.insert(0, "/workspace/TRELLIS.2")
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel.postprocess as PP

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="eager", choices=["eager", "compile"])
ap.add_argument("--images", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--warmup", type=int, default=1)
a = ap.parse_args()

pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipe.low_vram = False
pipe.cuda()
if a.mode == "compile":
    torch._dynamo.config.cache_size_limit = 256
    for k, m in pipe.models.items():
        if "flow" in k: m.forward = torch.compile(m.forward, dynamic=True)
    pipe.image_cond_model.model.forward = torch.compile(pipe.image_cond_model.model.forward, dynamic=True)

def sync(): torch.cuda.synchronize(); return time.perf_counter()
def run(path):
    t = {}; t0 = s = sync()
    mesh = pipe.run(Image.open(path), seed=42)[0]
    t["pipeline"] = (e := sync()) - s; s = e
    mesh.simplify(16777216)
    glb = PP.to_glb(vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords, attr_layout=mesh.layout,
                    voxel_size=mesh.voxel_size, aabb=[[-0.5] * 3, [0.5] * 3], decimation_target=1000000, texture_size=4096,
                    remesh=True, remesh_band=1, remesh_project=0)
    t["to_glb"] = (e := sync()) - s; s = e
    glb.export(f"/tmp/ref_{a.mode}_{os.path.basename(path)}.glb", extension_webp=True)
    t["export"] = (e := sync()) - s
    t["total"] = e - t0
    return t

for i in range(a.warmup): print("warmup", run(a.images[0]), flush=True)
R = {}
for p in a.images:
    R[os.path.basename(p)] = run(p); print(a.mode, os.path.basename(p), R[os.path.basename(p)], flush=True)
json.dump(R, open(a.out, "w"), indent=1)
