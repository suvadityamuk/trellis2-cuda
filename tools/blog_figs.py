# Offline: figures for docs/blog.md -> docs/blog/*.png.  uv run --with matplotlib --with numpy python tools/blog_figs.py
import glob, json, os, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

O = 'docs/blog'; os.makedirs(O, exist_ok=True)
def save(fig, n): fig.tight_layout(); fig.savefig(f'{O}/{n}.png', dpi=130); plt.close(fig)
def clean(ax): ax.spines[['top', 'right']].set_visible(False)

# 1. Fast-build changes: before/after (T.png, measured; README "Fast build" table)
ch = [('GPU LBVH\n(bvh build)', 7.1, 0.005), ('chart split\n(xatlas ComputeCharts)', 36, 1.3), ('GPU TELEA\n(inpaint)', 8.2, 0.15),
      ('FA3\n(shape flow)', 7.7, 5.6), ('FA3\n(tex flow)', 3.7, 2.6), ('WebP method 0\n(export)', 6.0, 1.4)]
fig, ax = plt.subplots(figsize=(10, 4)); x = np.arange(len(ch))
ax.bar(x - 0.2, [c[1] for c in ch], 0.38, color='#bbbbbb', label='before'); ax.bar(x + 0.2, [c[2] for c in ch], 0.38, color='#e76f51', label='fast build')
for i, (_, a, b) in enumerate(ch): ax.text(i + 0.2, b * 1.15, f'{a / b:.0f}×' if a / b >= 10 else f'{a / b:.1f}×', ha='center', fontsize=9)
ax.set_yscale('log'); ax.set_ylabel('seconds (log)'); ax.set_xticks(x, [c[0] for c in ch], fontsize=8); ax.legend(); clean(ax)
ax.set_title('Each fast-build change on its own stage (T.png, H200)'); save(fig, 'fast_changes')

# 2. Tolerance: GLB distance to official eager (README compare_glb table)
img = ['0a34', '0e49', '0f16', '130c', 'T']
tol = {'official torch.compile': ([2.4, 4.0, 2.9, 4.5, 3.0], [16.2, 23.3, 26.9, 25.8, 25.2], '#e9c46a', 'o'),
       'CUDA parity': ([1.3, 1.7, 1.7, 0.9, 1.6], [3.3, 7.9, 5.6, 4.0, 6.1], '#8ab17d', 's'),
       'CUDA fast': ([2.4, 3.8, 3.0, 4.6, 2.8], [20.5, 23.6, 24.1, 26.6, 28.7], '#e76f51', '^')}
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for k, (c, l, col, m) in tol.items():
    ax.scatter(c, l, s=60, color=col, marker=m, label=k, edgecolor='k', lw=0.5)
    for xi, yi, n in zip(c, l, img): ax.annotate(n, (xi, yi), fontsize=7, xytext=(4, 3), textcoords='offset points')
ax.set_xlabel('Chamfer distance to eager GLB (×1e-3 of bbox diagonal)'); ax.set_ylabel('base-color L1 to eager (/255)')
ax.set_title('Distance to the official eager output'); ax.legend(fontsize=8); clean(ax); save(fig, 'tolerance')

# 3. CuMesh chart sizes for T.png (T2_CHARTS dump from the fast build), with the xatlas split threshold
sz = np.loadtxt('bench/charts_T.txt', dtype=np.int64); S = 1024; big = sz > S
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
bins = np.logspace(0, np.log10(sz.max()) + 0.1, 40)
axs[0].hist(sz, bins=bins, color='#457b9d'); axs[0].set_xscale('log'); axs[0].set_yscale('log'); axs[0].axvline(S, color='#e76f51', ls='--')
axs[0].set_xlabel('faces per CuMesh chart'); axs[0].set_ylabel('charts')
axs[0].set_title(f'{len(sz)} charts; {big.sum()} above {S} faces ({100 * big.mean():.1f}%)'); clean(axs[0])
o = np.sort(sz)[::-1]; cf = np.cumsum(o) / o.sum()
axs[1].plot(np.arange(1, len(o) + 1), cf, color='#457b9d'); axs[1].set_xscale('log'); axs[1].axvline(big.sum(), color='#e76f51', ls='--')
axs[1].set_xlabel('charts, largest first'); axs[1].set_ylabel('fraction of all faces')
axs[1].set_title(f'the {big.sum()} charts above {S} faces hold {100 * sz[big].sum() / sz.sum():.0f}% of faces'); clean(axs[1])
save(fig, 'charts')

# 4. GPU TELEA levels for T.png (T2_TELEA dump): texels filled per level, one set of launches per level
cnt = np.loadtxt('bench/telea_T.txt', dtype=np.int64)
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.bar(np.arange(1, len(cnt) + 1), cnt, width=0.9, color='#2a9d8f'); ax.set_yscale('log'); ax.set_xticks(np.arange(1, len(cnt) + 1))
ax.set_xlabel('distance level L'); ax.set_ylabel('texels filled in parallel')
ax.set_title(f'T.png 4096² atlas: {cnt.sum():,} texels inpainted in {len(cnt)} levels'); clean(ax); save(fig, 'telea_levels')

# 5. Code size by component
grp = {'DiT flows + sampler': ['src/dit.cu', 'src/dit.h'], 'DINOv3': ['src/dino.cu'], 'attention (FA2/FA3/mem-eff)': ['src/flash.cu', 'src/fa3.cu', 'src/mea.cu'],
       'GEMM + elementwise ops': ['src/gemm.cu', 'src/ops.cu', 'src/ops.h', 'src/core.cu', 'src/core.h'],
       'sparse conv + decoders': ['src/spconv.cu', 'src/spconv.h', 'src/sdec.cu', 'src/ssdec.cu', 'src/conv.cu'],
       'image + RNG': ['src/image.cu', 'src/rng.cpp'], 'mesh + to_glb': ['src/mesh.cu', 'src/glb.cu', 'src/nvdr.cpp'],
       'TELEA + GLB writer': ['src/inpaint.cpp', 'src/gltf.cpp'], 'pipeline + models.h': ['src/pipeline.cu', 'src/models.h'],
       'apps (CLI, parity)': ['apps/t2.cu', 'apps/parity.cu']}
loc = {k: sum(sum(1 for _ in open(f)) for f in v) for k, v in grp.items()}
fig, ax = plt.subplots(figsize=(8, 3.8)); ks = sorted(loc, key=loc.get)
ax.barh(ks, [loc[k] for k in ks], color='#457b9d')
for i, k in enumerate(ks): ax.text(loc[k] + 5, i, str(loc[k]), va='center', fontsize=8)
ax.set_xlabel('lines of C++/CUDA'); ax.set_title(f'{sum(loc.values())} lines total (excluding reused third-party kernels)'); clean(ax); save(fig, 'loc')
