"""GPU mesh and texture kernels from the TRELLIS.2 CUDA port (https://github.com/suvadityamuk/trellis2-cuda)."""

from typing import Tuple

import torch

from ._ops import ops


def mesh_udf(vertices: torch.Tensor, faces: torch.Tensor, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unsigned distance from each point to a triangle mesh, via a GPU linear BVH built on the fly.

    Args:
        vertices: float32 [V, 3] CUDA tensor.
        faces: int32 or int64 [F, 3] CUDA tensor of vertex indices.
        points: float32 [M, 3] CUDA tensor of query points.

    Returns:
        (distance float32 [M], face_id int64 [M], barycentric float32 [M, 3]) of the closest point on the mesh.
        Ties between equally close faces resolve to the lowest face index.
    """
    d, f, b = ops.mesh_udf(vertices, faces, points)
    return d, f, b


def telea_inpaint(image: torch.Tensor, mask: torch.Tensor, radius: int = 3) -> torch.Tensor:
    """Fill masked pixels with TELEA inpainting, a GPU alternative to cv2.inpaint(..., cv2.INPAINT_TELEA).

    Pixels are filled in parallel by integer distance level rather than in fast-marching order, so results are close
    to OpenCV's but not identical.

    Args:
        image: uint8 [H, W] or [H, W, C] (C <= 4) CUDA tensor.
        mask: [H, W] CUDA tensor; nonzero marks pixels to inpaint.
        radius: neighborhood radius in pixels.

    Returns:
        Inpainted uint8 tensor with the same shape as image.
    """
    return ops.telea_inpaint(image, mask, radius)


__all__ = ["mesh_udf", "telea_inpaint"]
