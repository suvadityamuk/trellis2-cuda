# Offline: bench/*.json -> docs/timing.png (per-image stacked stages) and docs/speedup.png (sum).
# `plot_bench.py v2` plots the second benchmark session (bench/*_v2.json, fast_old.json) to docs/{timing,speedup}_v2.png.
import json, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np

V = [('eager', 'Official eager'), ('compile', 'Official torch.compile'), ('cuda', 'CUDA parity'), ('fast', 'CUDA fast')]
S = [('pipeline', 'pipeline (models)', '#4C72B0'), ('to_glb', 'to_glb (remesh/UV/bake)', '#DD8452'), ('export', 'GLB export', '#55A868')]
SFX = ''
if sys.argv[1:] == ['v2']:
    SFX = '_v2'
    V = [('eager_v2', 'Official eager'), ('eager_warm_v2', 'Official eager, warm autotune cache'), ('compile_v2', 'Official torch.compile'),
         ('cuda_v2', 'CUDA parity'), ('fast_old', 'CUDA fast (before)'), ('fast_v2', 'CUDA fast')]
D = {k: json.load(open(f'bench/{k}.json')) for k, _ in V}; NV = len(V)
imgs = list(D[V[0][0]]); names = [i[:4] if i != 'T.png' else 'T' for i in imgs]

fig, ax = plt.subplots(figsize=(11, 4.5)); w = 0.8 / NV
for j, (k, lab) in enumerate(V):
    x = np.arange(len(imgs)) + (j - (NV - 1) / 2) * w; b = np.zeros(len(imgs))
    for s, sl, c in S:
        h = np.array([D[k][i][s] for i in imgs])
        ax.bar(x, h, w * 0.92, bottom=b, color=c, alpha=0.4 + 0.6 * j / (NV - 1), label=sl if j == NV - 1 else None); b += h
    for xi, t in zip(x, b): ax.text(xi, t + 1, f'{t:.0f}', ha='center', fontsize=7)
ax.set_xticks(np.arange(len(imgs)), names); ax.set_ylabel('seconds (H200)')
ax.set_title('Per-image time; bars left→right: ' + ', '.join(l for _, l in V), fontsize=8 if SFX else 10)
ax.legend(); ax.spines[['top', 'right']].set_visible(False); fig.tight_layout(); fig.savefig(f'docs/timing{SFX}.png', dpi=130)

fig, ax = plt.subplots(figsize=(7, 3.2))
tot = [sum(D[k][i]['total'] for i in imgs) for k, _ in V]
bars = ax.barh([l for _, l in V][::-1], tot[::-1], color=['#2a9d8f', '#5fa98a', '#8ab17d', '#e9c46a', '#f4a261', '#e76f51'][-NV:])
for b, t in zip(bars, tot[::-1]): ax.text(t + 5, b.get_y() + b.get_height() / 2, f'{t:.0f} s  ({tot[0] / t:.1f}×)', va='center')
ax.set_xlabel('total seconds, 5 example images'); ax.set_xlim(0, max(tot) * 1.25)
ax.spines[['top', 'right']].set_visible(False); fig.tight_layout(); fig.savefig(f'docs/speedup{SFX}.png', dpi=130)
