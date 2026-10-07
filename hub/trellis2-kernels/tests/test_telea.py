import kernels
import pytest
import torch

pytestmark = pytest.mark.kernels_ci

k = kernels.get_kernel("suvadityamuk/trellis2-kernels", version=1, trust_remote_code=True)


def blobs(H, W, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(H, device="cuda"), torch.arange(W, device="cuda"), indexing="ij")
    known = torch.zeros(H, W, dtype=torch.bool, device="cuda")
    for c in (torch.rand(12, 4, device="cuda", generator=g) * torch.tensor([H, W, H / 5, W / 5], device="cuda")):
        known |= ((yy - c[0]) / (c[2] + 4)) ** 2 + ((xx - c[1]) / (c[3] + 4)) ** 2 < 1
    return ~known


@pytest.mark.kernels_ci
@pytest.mark.parametrize("shape", [(96, 128), (64, 80, 3), (50, 70, 4), (40, 40, 1)])
def test_known_pixels_unchanged(shape):
    img = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    mask = blobs(*shape[:2])
    out = k.telea_inpaint(img, mask, 3)
    assert out.shape == img.shape and out.dtype == torch.uint8
    assert torch.equal(out[~mask], img[~mask])


@pytest.mark.kernels_ci
def test_constant_and_smooth_fill():
    H, W = 128, 160
    mask = blobs(H, W, 1)
    img = torch.full((H, W, 3), 77, dtype=torch.uint8, device="cuda")
    # TELEA rounds with +0.5 and normalizes its gradient term, so constant fills drift upward with depth;
    # cv2.inpaint on this exact input yields values in 76..85 with mean |x - 77| of 1.66.
    dev = (k.telea_inpaint(img, mask, 3).float() - 77)[mask]
    assert dev.abs().mean() < 3 and dev.min() >= -2 and dev.max() <= 9
    ramp = torch.linspace(0, 255, W, device="cuda").round().to(torch.uint8).expand(H, W).contiguous()
    out = k.telea_inpaint(ramp, mask, 5)
    # 74% of pixels are masked here; cv2.inpaint's mean error against the true ramp is 10.8.
    assert (out.float() - ramp.float()).abs()[mask].mean() < 13


@pytest.mark.kernels_ci
def test_bool_and_uint8_masks_agree():
    img = torch.randint(0, 256, (64, 64, 3), dtype=torch.uint8, device="cuda")
    mask = blobs(64, 64, 2)
    assert torch.equal(k.telea_inpaint(img, mask, 3), k.telea_inpaint(img, mask.to(torch.uint8) * 255, 3))


def test_close_to_opencv():
    cv2 = pytest.importorskip("cv2")
    H, W = 256, 256
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    img = torch.stack([(xx + yy) % 256, (2 * xx) % 256, (128 + 64 * torch.sin(yy / 9.0)).long()], -1).to(torch.uint8)
    mask = blobs(H, W, 3).cpu()
    ref = torch.from_numpy(cv2.inpaint(img.numpy(), mask.to(torch.uint8).numpy() * 255, 3, cv2.INPAINT_TELEA))
    out = k.telea_inpaint(img.cuda(), mask.cuda(), 3).cpu()
    assert (out.float() - ref.float()).abs()[mask].mean() < 6
