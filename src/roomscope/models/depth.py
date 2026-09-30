"""Monocular metric depth: MoGe-2 (Microsoft, MIT licence, Ruicheng/moge-2-vitl-normal).

Chosen as the photo/video tiers' metric-scale source after measuring on real iPad images
with laser ground truth (ARKitScenes 47429912, 14 views): median scale error -0.5% with
known FOV, per-view spread 6.9%. MapAnything alone was -33% with 17% spread on the same
views, so its metric scale is not trusted on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import cache
from .device import torch_device

MODEL_ID = "Ruicheng/moge-2-vitl-normal"
_model = None


@dataclass
class MetricDepth:
    depth: np.ndarray
    mask: np.ndarray
    normal: np.ndarray | None
    K: np.ndarray


def _load():
    global _model
    if _model is None:
        from moge.model.v2 import MoGeModel
        _model = MoGeModel.from_pretrained(MODEL_ID).to(torch_device()).eval()
    return _model


def metric_depth(image: np.ndarray, fov_x_deg: float | None) -> MetricDepth:
    """image: uint8 RGB at processing resolution. fov_x from EXIF when known."""
    name = cache.key(MODEL_ID, image, None if fov_x_deg is None else round(float(fov_x_deg), 4), "fp32")
    hit = cache.load("moge2", name)
    if hit is not None:
        return MetricDepth(hit["depth"], hit["mask"].astype(bool), hit.get("normal"), hit["K"])

    import torch
    model = _load()
    tensor = torch.from_numpy(image).float().permute(2, 0, 1).div(255).to(torch_device())
    with torch.no_grad():
        out = model.infer(tensor, fov_x=fov_x_deg, use_fp16=False)
    depth = out["depth"].float().cpu().numpy()
    mask = out["mask"].cpu().numpy().astype(bool) & np.isfinite(depth) & (depth > 0)
    depth = np.where(mask, depth, 0.0).astype(np.float32)
    normal = out["normal"].float().cpu().numpy().astype(np.float32) if "normal" in out else None
    # MoGe intrinsics are normalised by image size.
    height, width = depth.shape
    K = out["intrinsics"].float().cpu().numpy().astype(np.float64)
    K[0] *= width
    K[1] *= height
    arrays = {"depth": depth, "mask": mask, "K": K}
    if normal is not None:
        arrays["normal"] = normal
    cache.store("moge2", name, arrays)
    return MetricDepth(depth, mask, normal, K)
