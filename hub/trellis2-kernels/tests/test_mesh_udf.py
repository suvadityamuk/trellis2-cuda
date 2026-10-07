import kernels
import pytest
import torch

pytestmark = pytest.mark.kernels_ci

k = kernels.get_kernel("suvadityamuk/trellis2-kernels", version=1, trust_remote_code=True)


def seg_d2(p, a, b):
    d = b - a
    t = (((p - a) * d).sum(-1) / (d * d).sum(-1).clamp_min(1e-300)).clamp(0, 1)
    return ((a + t[..., None] * d - p) ** 2).sum(-1)


def brute_d2(V, F, P):
    """Float64 point-triangle squared distances [M, F]: plane distance when the projection is inside, else nearest edge."""
    V, P = V.double(), P.double()
    a, b, c = (V[F[:, i].long()][None] for i in range(3))
    p = P[:, None]
    n = torch.cross(b - a, c - a, dim=-1)
    nn = (n * n).sum(-1)
    t = ((p - a) * n).sum(-1) / nn.clamp_min(1e-300)
    q = p - t[..., None] * n
    inside = (nn > 0).expand_as(t).clone()
    for x, y in ((a, b), (b, c), (c, a)):
        inside &= (torch.cross(y - x, q - x, dim=-1) * n).sum(-1) >= 0
    plane = t * t * nn
    edges = torch.minimum(torch.minimum(seg_d2(p, a, b), seg_d2(p, b, c)), seg_d2(p, c, a))
    return torch.where(inside, plane, edges)


def check(V, F, P):
    d, f, b = k.mesh_udf(V, F, P)
    D2 = brute_d2(V, F, P)
    ref = D2.min(1).values.sqrt()
    torch.testing.assert_close(d.double(), ref, atol=1e-5, rtol=1e-5)
    # the returned face is (one of) the nearest
    torch.testing.assert_close(D2.gather(1, f[:, None]).squeeze(1).sqrt(), ref, atol=1e-5, rtol=1e-5)
    # barycentrics reconstruct a point at that distance
    tri = V[F[f].long()]
    q = (b[..., None] * tri).sum(1)
    torch.testing.assert_close((q - P).norm(dim=-1), d, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(b.sum(-1), torch.ones_like(d), atol=1e-5, rtol=0)
    assert (b >= -1e-5).all()


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_random_soup(dtype):
    g = torch.Generator(device="cuda").manual_seed(0)
    V = torch.rand(3000, 3, device="cuda", generator=g) * 2 - 1
    F = torch.randint(0, 3000, (2000, 3), device="cuda", generator=g).to(dtype)
    P = torch.randn(4000, 3, device="cuda", generator=g)
    check(V, F, P)


@pytest.mark.kernels_ci
def test_grid_surface():
    n = 48
    y, x = torch.meshgrid(torch.linspace(-1, 1, n, device="cuda"), torch.linspace(-1, 1, n, device="cuda"), indexing="ij")
    V = torch.stack([x, y, 0.2 * torch.sin(3 * x) * torch.cos(3 * y)], -1).reshape(-1, 3)
    i = torch.arange(n - 1, device="cuda")
    q = (i[:, None] * n + i[None]).reshape(-1)
    F = torch.cat([torch.stack([q, q + 1, q + n], 1), torch.stack([q + 1, q + n + 1, q + n], 1)]).int()
    P = torch.rand(5000, 3, device="cuda") * 3 - 1.5
    check(V, F, P)


@pytest.mark.kernels_ci
def test_degenerate_and_single():
    V = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=torch.float32, device="cuda")
    P = torch.randn(256, 3, device="cuda")
    check(V, torch.tensor([[0, 1, 2]], device="cuda"), P)  # zero-area triangle
    check(V, torch.tensor([[0, 0, 3]], device="cuda"), P)  # two coincident vertices
    check(V, torch.tensor([[0, 1, 3]], device="cuda"), P)
    check(V, torch.tensor([[0, 1, 2], [0, 1, 3], [0, 0, 0]], device="cuda"), P)
