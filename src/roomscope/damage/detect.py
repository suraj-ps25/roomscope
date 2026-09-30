"""Damage regions on a surface orthophoto.

Walls are mostly one paint colour, so damage is found as deviation from the wall's own
background (a large robust median in CIELAB). The order matters, because each detector's
false positives are another's true positives:

  1. flush objects  multicoloured or high-contrast, rectangular (posters, frames): masked
  2. mould          dense clusters of dark specks (speck density over 3 cm windows)
  3. cracks         thin, long, dark ridges (multi-scale Sato), outside mould and objects
  4. stains         hysteresis on colour deviation (seed strong, grow weak) so a stain's
                    pale interior is kept, not only its darker tide line; a water stain is
                    a yellow/brown shift
Peeling paint is not reported: "lighter than the wall around it" is also what a lighting
gradient looks like, and a phantom damage claim is worse than a missing class. Textured
surfaces (floors) only report strong, compact stains.
Openings and mirrors are masked out by the caller. Everything is measured in texels of
known size, so areas and lengths are metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage
from skimage.filters import apply_hysteresis_threshold, sato
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

from .ortho import TEXEL_M, Orthophoto

BACKGROUND_M = 0.30
STAIN_BACKGROUND_M = 1.2
STRAIGHT_TORTUOSITY = 1.04
AXIS_TOLERANCE_DEG = 4.0
STAIN_SEED_DELTA_E = 7.0
STAIN_GROW_DELTA_E = 3.5
MIN_REGION_M2 = 0.01
CRACK_RIDGE = 0.06
MIN_CRACK_M = 0.08
EDGE_MARGIN_M = 0.04
SPECK_DARK_L = 18.0
SPECK_WINDOW_M = 0.03
SPECK_DENSITY = 0.12
OBJECT_COLOUR_SPREAD = 9.0
MIN_SPECKS = 15


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


def _background(lab: np.ndarray, valid: np.ndarray, window_m: float = BACKGROUND_M) -> np.ndarray:
    """Robust local median per channel, computed on a coarse grid and upsampled. The
    window must be larger than the damage it is a background for: a median ignores
    anything covering less than half of it."""
    rows, cols = valid.shape
    factor = 8
    small_valid = cv2.resize(valid.astype(np.float32), (max(cols // factor, 1), max(rows // factor, 1)),
                             interpolation=cv2.INTER_AREA) > 0.5
    background = np.zeros_like(lab)
    kernel = max(3, int(window_m / (TEXEL_M * factor)) | 1)
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


def _objects(lab: np.ndarray, delta: np.ndarray, valid: np.ndarray, texel: float) -> np.ndarray:
    """Flush decor: regions whose colour deviates in many directions at once (a poster's
    print) rather than one consistent tint (a stain)."""
    strong = (np.linalg.norm(delta, axis=-1) > STAIN_SEED_DELTA_E) & valid
    strong = ndimage.binary_closing(strong, structure=np.ones((5, 5)))
    mask = np.zeros_like(valid)
    for region in regionprops(label(strong, connectivity=2)):
        if region.area * texel ** 2 < MIN_REGION_M2:
            continue
        coords = tuple(region.coords.T)
        spread = float(np.std(lab[coords][:, 1]) + np.std(lab[coords][:, 2]))
        fill = region.area / max(region.area_bbox, 1)
        if spread > OBJECT_COLOUR_SPREAD or (fill > 0.9 and np.std(lab[coords][:, 0]) > 12):
            mask[coords] = True
    return ndimage.binary_dilation(mask, iterations=max(1, int(0.02 / texel)))


def _mould(lab: np.ndarray, delta: np.ndarray, valid: np.ndarray, texel: float) -> np.ndarray:
    dark = (delta[..., 0] < -SPECK_DARK_L) & valid
    window = max(3, int(SPECK_WINDOW_M / texel) | 1)
    density = cv2.boxFilter(dark.astype(np.float32), -1, (window, window))
    clustered = (density > SPECK_DENSITY) & valid
    return ndimage.binary_closing(clustered, structure=np.ones((window, window)))


def detect(ortho: Orthophoto, exclude_uv: list[tuple[float, float, float, float]] | None = None,
           step: int = 1, textured: bool = False) -> list[Detection]:
    rgb, valid = ortho.rgb, ortho.valid.copy()
    texel = TEXEL_M * step
    margin = int(EDGE_MARGIN_M / texel)
    valid[:margin] = valid[-margin:] = False
    valid[:, :margin] = valid[:, -margin:] = False
    for u0, u1, v0, v1 in exclude_uv or []:
        c0, c1 = int(max(u0 - 0.02, 0) / texel), int((u1 + 0.02) / texel) + 1
        r0, r1 = int(max(v0 - 0.02, 0) / texel), int((v1 + 0.02) / texel) + 1
        valid[r0:r1, c0:c1] = False
    if valid.sum() < 500:
        return []

    lab = _lab(rgb)
    background = _background(lab, valid)
    delta = lab - background
    delta_e = np.linalg.norm(delta, axis=-1)
    detections: list[Detection] = []

    objects = _objects(lab, delta, valid, texel)
    usable = valid & ~objects

    mould = _mould(lab, delta, usable, texel)
    dark = (delta[..., 0] < -SPECK_DARK_L) & usable
    kept_mould = np.zeros_like(mould)
    for region in regionprops(label(mould, connectivity=2)):
        if region.area * texel ** 2 < 2 * MIN_REGION_M2:
            continue
        piece = np.zeros_like(mould)
        piece[tuple(region.coords.T)] = True
        # Mould is many separate small specks; a crack or a shadow edge is one long one.
        specks = label(dark & piece, connectivity=2).max()
        if specks < MIN_SPECKS:
            continue
        kept_mould |= piece
        dark_share = float(np.mean(delta[piece][:, 0] < -SPECK_DARK_L))
        detections.append(_make("mold", piece, step, region, None, min(1.0, 2 * dark_share)))
    mould_zone = ndimage.binary_dilation(kept_mould, iterations=max(1, int(0.03 / texel)))

    lightness = np.where(usable, lab[..., 0], background[..., 0])
    ridge = sato(lightness / 100.0, sigmas=[1, 2, 3], black_ridges=True)
    crack_mask = (ridge > CRACK_RIDGE) & usable & ~mould_zone & (delta[..., 0] < -8)
    crack_mask = ndimage.binary_closing(crack_mask, structure=np.ones((3, 3)))
    for region in regionprops(label(crack_mask, connectivity=2)) if not textured else []:
        piece = np.zeros_like(crack_mask)
        piece[tuple(region.coords.T)] = True
        skeleton = skeletonize(piece)
        length = float(skeleton.sum() * texel)
        width = region.area * texel ** 2 / max(length, 1e-6)
        if length < MIN_CRACK_M or length / max(width, texel) < 8 or _straight_edge(skeleton, length, texel):
            continue
        detections.append(_make("crack", piece, step, region, length, confidence=min(1.0, length / 0.3)))
    crack_zone = ndimage.binary_dilation(crack_mask, iterations=3)

    open_area = usable & ~mould_zone & ~crack_zone
    stain_delta = lab - _background(lab, usable, STAIN_BACKGROUND_M)
    delta = stain_delta
    blob_mask = apply_hysteresis_threshold(np.where(open_area, np.linalg.norm(stain_delta, axis=-1), 0),
                                           STAIN_GROW_DELTA_E, STAIN_SEED_DELTA_E)
    blob_mask = ndimage.binary_fill_holes(ndimage.binary_closing(blob_mask, structure=np.ones((5, 5))))
    blob_mask = ndimage.binary_opening(blob_mask, structure=np.ones((3, 3)))
    min_area = 3 * MIN_REGION_M2 if textured else MIN_REGION_M2
    min_shift = 4.0 if textured else 2.0
    for region in regionprops(label(blob_mask, connectivity=2)):
        if region.area * texel ** 2 < min_area:
            continue
        piece = np.zeros_like(blob_mask)
        piece[tuple(region.coords.T)] = True
        d = delta[piece]
        dl, db = float(d[:, 0].mean()), float(d[:, 2].mean())
        if db > min_shift and dl < 3 and region.solidity > 0.6:
            detections.append(_make("water_stain", piece, step, region, None, min(1.0, db / 8)))
    return detections


def _straight_edge(skeleton: np.ndarray, length: float, texel: float) -> bool:
    """Junctions, skirting, frames and cables make dead-straight axis-aligned lines; real
    cracks meander or run diagonally."""
    rows, cols = np.nonzero(skeleton)
    points = np.stack([cols, rows], axis=1).astype(float)
    centred = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    along = centred @ vt[0]
    chord = float((along.max() - along.min()) * texel)
    angle = np.degrees(np.arctan2(abs(vt[0][1]), abs(vt[0][0])))
    axis_aligned = min(angle, 90 - angle) < AXIS_TOLERANCE_DEG
    return axis_aligned and length / max(chord, 1e-6) < STRAIGHT_TORTUOSITY


def _make(cls: str, piece: np.ndarray, step: int, region, length: float | None, confidence: float) -> Detection:
    texel = TEXEL_M * step
    r0, c0, r1, c1 = region.bbox
    return Detection(cls, _polygon(piece, step), float(region.area * texel ** 2), length, float(confidence),
                     (c0 * texel, c1 * texel, r0 * texel, r1 * texel))
