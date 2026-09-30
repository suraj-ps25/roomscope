"""Damage regions on a surface orthophoto.

Walls are mostly one paint colour, so damage is found as deviation from the wall's own
background (a large robust median in CIELAB), then typed by shape and colour:

  crack         thin, long, dark ridge (multi-scale Sato filter), skeleton length >= 8 cm
  water_stain   compact blob shifted yellow/brown (b* up) and slightly darker
  mold          dark, speckled (high local variance), clustered
  peeling_paint irregular, lighter, high edge density
Flush objects are rejected before typing: posters and frames are saturated and
rectangular, switch plates are small and bright. Openings and mirrors are masked out.

Everything is measured in texels of known size, so areas and lengths are metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage
from skimage.filters import sato
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

from .ortho import TEXEL_M, Orthophoto

BACKGROUND_M = 0.30
STAIN_DELTA_E = 7.0
MIN_STAIN_M2 = 0.004
CRACK_RIDGE = 0.06
MIN_CRACK_M = 0.08
EDGE_MARGIN_M = 0.04


@dataclass
class Detection:
    cls: str
    polygon_uv: np.ndarray
    area_m2: float
    length_m: float | None
    confidence: float
    bbox_uv: tuple[float, float, float, float]


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(rgb / 255.0, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)


def _background(lab: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Robust local median per channel, computed on a coarse grid and upsampled."""
    rows, cols = valid.shape
    factor = 8
    small_valid = cv2.resize(valid.astype(np.float32), (max(cols // factor, 1), max(rows // factor, 1)),
                             interpolation=cv2.INTER_AREA) > 0.5
    background = np.zeros_like(lab)
    kernel = max(3, int(BACKGROUND_M / (TEXEL_M * factor)) | 1)
    for c in range(3):
        small = cv2.resize(lab[..., c], small_valid.shape[::-1], interpolation=cv2.INTER_AREA)
        filled = small.copy()
        if small_valid.any():
            _, (ii, jj) = ndimage.distance_transform_edt(~small_valid, return_indices=True)
            filled = small[ii, jj]
        median = ndimage.median_filter(filled, size=kernel)
        background[..., c] = cv2.resize(median, (cols, rows), interpolation=cv2.INTER_LINEAR)
    return background


def _to_uv(rows: np.ndarray, cols: np.ndarray, step: int) -> np.ndarray:
    return np.stack([(cols + 0.5) * TEXEL_M * step, (rows + 0.5) * TEXEL_M * step], axis=-1)


def _polygon(mask: np.ndarray, step: int) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)[:, 0, :]
    if len(contour) < 3:
        ys, xs = np.nonzero(mask)
        contour = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]])
    epsilon = max(1.0, 0.01 * cv2.arcLength(contour.reshape(-1, 1, 2).astype(np.float32), True))
    simple = cv2.approxPolyDP(contour.reshape(-1, 1, 2).astype(np.float32), epsilon, True)[:, 0, :]
    if len(simple) < 3:
        simple = contour
    return _to_uv(simple[:, 1].astype(float), simple[:, 0].astype(float), step)


def detect(ortho: Orthophoto, exclude_uv: list[tuple[float, float, float, float]] | None = None,
           step: int = 1) -> list[Detection]:
    rgb, valid = ortho.rgb, ortho.valid.copy()
    rows, cols = valid.shape
    texel = TEXEL_M * step
    margin = int(EDGE_MARGIN_M / texel)
    valid[:margin] = valid[-margin:] = False
    valid[:, :margin] = valid[:, -margin:] = False
    for u0, u1, v0, v1 in exclude_uv or []:
        c0, c1 = int(max(u0 - 0.03, 0) / texel), int((u1 + 0.03) / texel) + 1
        r0, r1 = int(max(v0 - 0.03, 0) / texel), int((v1 + 0.03) / texel) + 1
        valid[r0:r1, c0:c1] = False
    if valid.sum() < 500:
        return []

    lab = _lab(rgb)
    background = _background(lab, valid)
    delta = lab - background
    delta_e = np.linalg.norm(delta, axis=-1)
    detections: list[Detection] = []

    # Cracks: thin dark ridges on lightness.
    lightness = np.where(valid, lab[..., 0], background[..., 0])
    ridge = sato(lightness / 100.0, sigmas=[1, 2, 3], black_ridges=True)
    crack_mask = (ridge > CRACK_RIDGE) & valid & (delta[..., 0] < -8)
    crack_mask = ndimage.binary_opening(crack_mask, structure=np.ones((2, 2)))
    for region in regionprops(label(crack_mask, connectivity=2)):
        piece = np.zeros_like(crack_mask)
        piece[tuple(region.coords.T)] = True
        length = float(skeletonize(piece).sum() * texel)
        width = region.area * texel ** 2 / max(length, 1e-6)
        if length < MIN_CRACK_M or length / max(width, texel) < 8:
            continue
        detections.append(_make("crack", piece, step, region, length, confidence=min(1.0, length / 0.3)))

    # Blobs: stains, mould, peeling paint; reject flush objects.
    blob_mask = (delta_e > STAIN_DELTA_E) & valid & ~ndimage.binary_dilation(crack_mask, iterations=3)
    blob_mask = ndimage.binary_opening(blob_mask, structure=np.ones((3, 3)))
    blob_mask = ndimage.binary_closing(blob_mask, structure=np.ones((5, 5)))
    for region in regionprops(label(blob_mask, connectivity=2)):
        area = region.area * texel ** 2
        if area < MIN_STAIN_M2:
            continue
        piece = np.zeros_like(blob_mask)
        piece[tuple(region.coords.T)] = True
        d = delta[piece]
        chroma = float(np.hypot(lab[piece][:, 1], lab[piece][:, 2]).mean())
        rectangularity = region.area / max(region.bbox_area, 1)
        speckle = float(np.std(lab[piece][:, 0]))
        if chroma > 28 or (rectangularity > 0.9 and region.eccentricity < 0.99 and area > 0.05):
            continue
        dl, db = float(d[:, 0].mean()), float(d[:, 2].mean())
        if dl < -15 and speckle > 8:
            cls, confidence = "mold", min(1.0, speckle / 20)
        elif db > 3 and dl < 2:
            cls, confidence = "water_stain", min(1.0, db / 10)
        elif dl > 4:
            cls, confidence = "peeling_paint", min(1.0, dl / 15)
        else:
            cls, confidence = "water_stain", 0.4
        detections.append(_make(cls, piece, step, region, None, confidence))
    return detections


def _make(cls: str, piece: np.ndarray, step: int, region, length: float | None, confidence: float) -> Detection:
    texel = TEXEL_M * step
    r0, c0, r1, c1 = region.bbox
    return Detection(cls, _polygon(piece, step), float(region.area * texel ** 2), length, float(confidence),
                     (c0 * texel, c1 * texel, r0 * texel, r1 * texel))
