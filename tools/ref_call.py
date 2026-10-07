"""Replay one dumped flow-model call with deep hooks: python ref_call.py <model_name> <ckpt> <call> <cond_key>.
Writes {ref}/call.safetensors with {model}/deep/* keys for that call (parity merges it over pipeline.safetensors)."""
import os, sys, torch
from safetensors.torch import load_file, save_file
sys.path.insert(0, "/workspace/TRELLIS.2")
from trellis2 import models
from trellis2.modules.sparse import SparseTensor
name, ck, ci, ck_cond = sys.argv[1:5]
R, D = load_file("/workspace/ref/pipeline.safetensors"), {}
m = models.from_pretrained(f"microsoft/TRELLIS.2-4B/ckpts/{ck}").cuda().eval()
def put(k, v):
    if isinstance(v, SparseTensor): v = v.feats
    if isinstance(v, (tuple, list)): v = v[0]
    if isinstance(v, torch.Tensor): D[f"{name}/deep/{k}"] = v.detach().cpu().contiguous().clone()
for sn, sm in m.named_modules():
    if (sn.startswith("blocks.") and sn.count(".") == 1) or sn.startswith(f"blocks.{os.environ.get('T2_TAPBLK', '0')}.") or sn.startswith("t_embedder") or sn in ("input_layer", "adaLN_modulation", "out_layer"):
        sm.register_forward_hook(lambda mod, i, o, sn=sn: (put(sn + ".in", i[0]) if len(i) else None, put(sn + ".out", o)) and None)
p, g = f"{name}/call{ci}/", lambda k: R[k].cuda()
dense = p + "x" in R
x = g(p + "x") if dense else SparseTensor(g(p + "x.feats"), g(p + "x.coords"))
kw = {}
if f"{name}/call0/concat_cond.feats" in R:
    kw["concat_cond"] = SparseTensor(g(f"{name}/call0/concat_cond.feats"), g(p + "x.coords"))
cond = g(p + "cond") if p + "cond" in R else g(ck_cond)
with torch.no_grad():
    o = m(x, g(p + "t"), cond, **kw)
o = o if dense else o.feats
ref = R[p + ("out" if dense else "out.feats")]
print("replay matches dump:", torch.equal(o.cpu(), ref))
save_file(D, "/workspace/ref/call.safetensors"); print(len(D), "tensors")
