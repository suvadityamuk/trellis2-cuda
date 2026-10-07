# Offline: bench/*.json -> docs/timing.png (per-image stacked stages) and docs/speedup.png (sum).
import json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np

V = [('eager', 'Official eager'), ('compile', 'Official torch.compile'), ('cuda', 'CUDA parity'), ('fast', 'CUDA fast')]
S = [('pipeline', 'pipeline (models)', '#4C72B0'), ('to_glb', 'to_glb (remesh/UV/bake)', '#DD8452'), ('export', 'GLB export', '#55A868')]
D = {k: json.load(open(f'bench/{k}.json')) for k, _ in V}
imgs = list(D['eager']); names = [i[:4] if i != 'T.png' else 'T' for i in imgs]

fig, ax = plt.subplots(figsize=(11, 4.5)); w = 0.2
for j, (k, lab) in enumerate(V):
    x = np.arange(len(imgs)) + (j - 1.5) * w; b = np.zeros(len(imgs))
    for s, sl, c in S:
        h = np.array([D[k][i][s] for i in imgs])
        ax.bar(x, h, w * 0.92, bottom=b, color=c, alpha=[0.45, 0.65, 0.85, 1][j], label=sl if j == 3 else None); b += h
    for xi, t in zip(x, b): ax.text(xi, t + 1, f'{t:.0f}', ha='center', fontsize=7)
ax.set_xticks(np.arange(len(imgs)), names); ax.set_ylabel('seconds (H200)')
ax.set_title('Per-image time; bars left→right: eager, torch.compile, CUDA parity, CUDA fast (lighter = official)')
ax.legend(); ax.spines[['top', 'right']].set_visible(False); fig.tight_layout(); fig.savefig('docs/timing.png', dpi=130)

fig, ax = plt.subplots(figsize=(7, 3.2))
tot = [sum(D[k][i]['total'] for i in imgs) for k, _ in V]
bars = ax.barh([l for _, l in V][::-1], tot[::-1], color=['#2a9d8f', '#8ab17d', '#e9c46a', '#e76f51'])
for b, t in zip(bars, tot[::-1]): ax.text(t + 5, b.get_y() + b.get_height() / 2, f'{t:.0f} s  ({tot[0] / t:.1f}×)', va='center')
ax.set_xlabel('total seconds, 5 example images'); ax.set_xlim(0, max(tot) * 1.25)
ax.spines[['top', 'right']].set_visible(False); fig.tight_layout(); fig.savefig('docs/speedup.png', dpi=130)
