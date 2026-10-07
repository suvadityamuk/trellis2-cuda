"""Run the shape SLat decoder on the dumped latent and save the I/O of the first sparse conv for every distinct
(N, Ci, Co) to {ref}/spconv.safetensors (keys conv{j}/in, conv{j}/out, conv{j}/coords, conv{j}/name)."""
import sys, torch
from safetensors.torch import load_file, save_file
sys.path.insert(0, "/workspace/TRELLIS.2")
from trellis2 import models
from trellis2.modules.sparse import SparseTensor
from trellis2.modules.sparse.conv import SparseConv3d
R = load_file("/workspace/ref/pipeline.safetensors")
dec = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16").cuda().eval()
dec.set_resolution(1024)
D, seen = {}, set()
for name, m in dec.named_modules():
    if isinstance(m, SparseConv3d):
        def hook(mod, i, o, name=name):
            x = i[0]; key = (x.feats.shape[0], x.feats.shape[1], o.feats.shape[1])
            if key in seen: return
            j = len(seen); seen.add(key)
            D[f"conv{j}/in"], D[f"conv{j}/out"], D[f"conv{j}/coords"] = x.feats.cpu(), o.feats.cpu(), x.coords.cpu()
            D[f"conv{j}/name"] = torch.tensor(list(name.encode()), dtype=torch.uint8)
            print(j, name, key, flush=True)
        m.register_forward_hook(hook)
slat = SparseTensor(R["shape/slat.feats"].cuda(), R["shape/slat.coords"].cuda())
with torch.no_grad():
    h = dec.from_latent(slat)
    print("from_latent matches dump:", torch.equal(h.feats.cpu(), R["shape_slat_decoder/deep/from_latent.out.feats"]))
    dec(slat, return_subs=True)
save_file(D, "/workspace/ref/spconv.safetensors")
