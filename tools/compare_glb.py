"""Offline tolerance check between two textured GLBs: symmetric Chamfer distance (relative to the bbox diagonal) and
mean |base color| difference at nearest surface samples. Usage: python compare_glb.py a.glb b.glb [n_samples]"""
import sys
import numpy as np, trimesh
from scipy.spatial import cKDTree


def samples(path, n, seed=0):
    m = trimesh.load(path, force="mesh")
    pts, fid = trimesh.sample.sample_surface(m, n, seed=seed)
    bary = trimesh.triangles.points_to_barycentric(m.triangles[fid], pts)
    uv = (m.visual.uv[m.faces[fid]] * bary[:, :, None]).sum(1)
    tex = np.asarray(m.visual.material.baseColorTexture.convert("RGB"), dtype=np.float32)
    h, w = tex.shape[:2]
    x = np.clip((uv[:, 0] * w).astype(int), 0, w - 1); y = np.clip(((1 - uv[:, 1]) * h).astype(int), 0, h - 1)
    return pts, tex[y, x], np.linalg.norm(m.bounds[1] - m.bounds[0])


a, b = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 500000
pa, ca, diag = samples(a, n)
pb, cb, _ = samples(b, n, seed=1)
dab, iab = cKDTree(pb).query(pa); dba, iba = cKDTree(pa).query(pb)
cd = (dab.mean() + dba.mean()) / 2 / diag
col = (np.abs(ca - cb[iab]).mean() + np.abs(cb - ca[iba]).mean()) / 2
print(f"chamfer {cd:.2e} (x diag)  p99 {max(np.percentile(dab, 99), np.percentile(dba, 99)) / diag:.2e}  color L1 {col:.2f}/255")
