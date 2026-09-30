"""Multi-view reconstruction: MapAnything (Meta, facebook/map-anything-apache, Apache-2.0).

Given images, intrinsics and MoGe-2 metric depth as conditioning, it returns consistent
camera poses (OpenCV cam-to-world, first view's frame) and refined per-view depth. Fed
MoGe depth, its depth scale on real ARKitScenes views was 0.978 of laser truth (5.9%
spread) versus 0.669 (16.8%) from images alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import cache
from .device import torch_device

MODEL_ID = "facebook/map-anything-apache"
_model = None


@dataclass
class ViewPrediction:
    depth: np.ndarray
    mask: np.ndarray
    confidence: np.ndarray
    K: np.ndarray
    pose: np.ndarray


def _load():
    global _model
    if _model is None:
        from mapanything.models import MapAnything
        _model = MapAnything.from_pretrained(MODEL_ID).to(torch_device()).eval()
    return _model


def reconstruct(images: list[np.ndarray], intrinsics: list[np.ndarray],
                depths: list[np.ndarray | None] | None = None) -> list[ViewPrediction]:
    """All images at the same (W, H), multiples of 14. depths may hold None for views
    reconstructed from the image alone (MapAnything accepts mixed inputs per view)."""
    height, width = images[0].shape[:2]
    name = cache.key(MODEL_ID, width, height, *images, *intrinsics,
                     *[d if d is not None else "none" for d in (depths or [])], "bf16", "mask_edges")
    hit = cache.load("mapanything", name)
    if hit is not None:
        return [ViewPrediction(hit["depth"][i], hit["mask"][i].astype(bool), hit["conf"][i], hit["K"][i], hit["pose"][i])
                for i in range(len(images))]

    import torch
    from mapanything.utils.image import preprocess_inputs

    torch.manual_seed(0)
    views = []
    for k, (image, K) in enumerate(zip(images, intrinsics)):
        view = {"img": torch.from_numpy(image), "intrinsics": torch.from_numpy(K.astype(np.float32))}
        if depths is not None and depths[k] is not None:
            view["depth_z"] = torch.from_numpy(depths[k].astype(np.float32))
            view["is_metric_scale"] = torch.tensor([True])
        views.append(view)
    model = _load()
    with torch.no_grad():
        preds = model.infer(preprocess_inputs(views, resize_mode="fixed_size", size=(width, height)),
                            memory_efficient_inference=False, use_amp=True, amp_dtype="bf16",
                            apply_mask=True, mask_edges=True)
    out = [ViewPrediction(p["depth_z"][0, ..., 0].float().cpu().numpy(), p["mask"][0, ..., 0].cpu().numpy().astype(bool),
                          p["conf"][0].float().cpu().numpy(), p["intrinsics"][0].float().cpu().numpy(),
                          p["camera_poses"][0].float().cpu().numpy()) for p in preds]
    cache.store("mapanything", name, {
        "depth": np.stack([v.depth for v in out]), "mask": np.stack([v.mask for v in out]),
        "conf": np.stack([v.confidence for v in out]), "K": np.stack([v.K for v in out]),
        "pose": np.stack([v.pose for v in out])})
    if torch_device() == "mps":
        torch.mps.empty_cache()
    return out
