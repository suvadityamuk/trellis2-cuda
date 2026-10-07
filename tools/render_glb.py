# Offline: render GLBs with nvdiffrast into a comparison grid (and an optional turntable GIF).
# python tools/render_glb.py grid out.png "Col A=dirA/{}.glb" "Col B=..." -- img1 img2 ...
# python tools/render_glb.py gif out.gif mesh.glb
# python tools/render_glb.py atlas out.png "Title A=a.glb" "Title B=b.glb"   (base-color atlas, full + 1:1 crop)
import sys, math, numpy as np, torch, trimesh, nvdiffrast.torch as dr
from PIL import Image, ImageDraw

R = 384
ctx = dr.RasterizeCudaContext()

def load(p):
    m = trimesh.load(p, force='mesh')
    v = torch.tensor(m.vertices, dtype=torch.float32, device='cuda')
    v = (v - (v.max(0)[0] + v.min(0)[0]) / 2) / (v.max(0)[0] - v.min(0)[0]).max()
    tex = np.asarray(m.visual.material.baseColorTexture.convert('RGBA'), np.float32) / 255
    return dict(v=v, f=torch.tensor(m.faces, dtype=torch.int32, device='cuda'),
                uv=torch.tensor(m.visual.uv, dtype=torch.float32, device='cuda') * torch.tensor([1., -1.], device='cuda') + torch.tensor([0., 1.], device='cuda'),
                n=torch.tensor(m.vertex_normals, dtype=torch.float32, device='cuda'),
                tex=torch.tensor(tex, device='cuda')[None].contiguous())

def mvp(yaw, pitch=20, d=2.0, fov=40):
    y, p = math.radians(yaw), math.radians(pitch)
    eye = np.array([d * math.cos(p) * math.sin(y), d * math.sin(p), d * math.cos(p) * math.cos(y)])
    z = eye / np.linalg.norm(eye); x = np.cross([0, 1, 0], z); x /= np.linalg.norm(x); yv = np.cross(z, x)
    V = np.eye(4); V[:3, :3] = np.stack([x, yv, z]); V[:3, 3] = -V[:3, :3] @ eye
    t = 1 / math.tan(math.radians(fov) / 2); n_, f_ = 0.1, 10
    P = np.array([[t, 0, 0, 0], [0, t, 0, 0], [0, 0, -(f_ + n_) / (f_ - n_), -2 * f_ * n_ / (f_ - n_)], [0, 0, -1, 0]])
    return torch.tensor(P @ V, dtype=torch.float32, device='cuda'), torch.tensor(z, dtype=torch.float32, device='cuda')

def render(m, yaw):
    M, view = mvp(yaw)
    clip = (torch.cat([m['v'], torch.ones_like(m['v'][:, :1])], 1) @ M.T)[None]
    rast, _ = dr.rasterize(ctx, clip, m['f'], (R, R))
    uv, _ = dr.interpolate(m['uv'][None], rast, m['f'])
    n, _ = dr.interpolate(m['n'][None], rast, m['f'])
    c = dr.texture(m['tex'], uv, filter_mode='linear')
    light = torch.nn.functional.normalize(view + torch.tensor([0.3, 0.5, 0.], device='cuda'), dim=0)
    shade = 0.55 + 0.45 * (torch.nn.functional.normalize(n, dim=-1) @ light).abs()[..., None]
    a = (rast[..., 3:] > 0).float()
    img = c[..., :3] * shade * a + (1 - a)
    img = dr.antialias(img.contiguous(), rast, clip, m['f'])
    return Image.fromarray((img[0].flip(0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))

def label(im, s):
    d = ImageDraw.Draw(im); d.rectangle([0, 0, R, 22], fill='white'); d.text((6, 5), s, fill='black'); return im

if sys.argv[1] == 'grid':
    out, rest = sys.argv[2], sys.argv[3:]
    k = rest.index('--'); cols = [c.split('=', 1) for c in rest[:k]]; imgs = rest[k + 1:]
    G = Image.new('RGB', (R * (len(cols) + 1), R * len(imgs)), 'white')
    for i, p in enumerate(imgs):
        src = Image.open(p).convert('RGBA'); src.thumbnail((R, R))
        bg = Image.new('RGB', (R, R), 'white'); bg.paste(src, ((R - src.width) // 2, (R - src.height) // 2), src)
        G.paste(label(bg, 'input' if i == 0 else ''), (0, i * R))
        name = p.split('/')[-1]
        for j, (title, pat) in enumerate(cols):
            G.paste(label(render(load(pat.format(name)), 35), title if i == 0 else ''), ((j + 1) * R, i * R))
    G.save(out)
elif sys.argv[1] == 'atlas':
    S = 768; cols = [c.split('=', 1) for c in sys.argv[3:]]
    G = Image.new('RGB', (S * len(cols), 2 * S), 'white')
    for j, (title, p) in enumerate(cols):
        t = trimesh.load(p, force='mesh').visual.material.baseColorTexture.convert('RGB'); W = t.width
        G.paste(t.resize((S, S), Image.LANCZOS), (j * S, 0))
        c = W // 2 - S // 2; G.paste(t.crop((c, c, c + S, c + S)), (j * S, S))
        d = ImageDraw.Draw(G); d.rectangle([j * S, 0, j * S + S, 22], fill='white'); d.text((j * S + 6, 5), f'{title}: {W}x{W} atlas', fill='black')
        d.rectangle([j * S, S, j * S + S, S + 22], fill='white'); d.text((j * S + 6, S + 5), f'{title}: center crop at 1:1', fill='black')
    G.save(sys.argv[2])
else:
    m = load(sys.argv[3])
    fr = [render(m, a) for a in range(0, 360, 10)]
    fr[0].save(sys.argv[2], save_all=True, append_images=fr[1:], duration=80, loop=0)
