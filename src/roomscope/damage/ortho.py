"""Metric orthophotos of room surfaces.

Each wall (u along the wall from its start corner, v height above the floor), the floor
and the ceiling are sampled on a TEXEL_M grid. Every texel takes its colour from the view
that sees it best: frontal, close, and not occluded, the last judged against that view's
depth (anything more than OCCLUSION_M in front of the surface blocks it, so a bed or a
wardrobe never paints onto the wall behind it). Pixel sizes are metric by construction,
so damage extent comes straight out in m2 and m.

Views differ in exposure and white balance (a phone adjusts both as it turns), so a
patchwork of best views has seams, and a seam between a darker and a lighter view is a
"stain" to any detector. Each view's colour gain is solved from what overlapping views
see at the same texels, and applied before the patchwork is assembled.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..capture import Frame

TEXEL_M = 0.005
OCCLUSION_M = 0.05
MAX_VIEW_M = 4.0
MIN_COSINE = 0.35


@dataclass
class SurfaceGrid:
    """Maps texel (row, col) to world points. Rows run along v (bottom row = low v)."""
    key: str
    kind: str
    origin: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    normal: np.ndarray
    width_m: float
    height_m: float

    @property
    def shape(self) -> tuple[int, int]:
        return int(np.ceil(self.height_m / TEXEL_M)), int(np.ceil(self.width_m / TEXEL_M))

    def points(self) -> np.ndarray:
        rows, cols = self.shape
        u = (np.arange(cols) + 0.5) * TEXEL_M
        v = (np.arange(rows) + 0.5) * TEXEL_M
        uu, vv = np.meshgrid(u, v)
        return self.origin + uu[..., None] * self.u_axis + vv[..., None] * self.v_axis


@dataclass
class Orthophoto:
    grid: SurfaceGrid
    rgb: np.ndarray
    valid: np.ndarray
    quality: np.ndarray


def _view_gains(samples: list[tuple[int, np.ndarray, np.ndarray]], views: int) -> np.ndarray:
    """Per-view, per-channel colour gains making overlapping views agree: colour = gain x
    surface, in logs, solved by alternating medians (robust to the occlusion leaks and
    highlights a few samples always carry), median gain fixed at 1."""
    gains = np.ones((max(views, 1), 3), dtype=np.float32)
    if not samples:
        return gains
    view_of = np.concatenate([np.full(len(index), v) for v, index, _ in samples])
    texel = np.concatenate([index for _, index, _ in samples])
    logs = np.log(np.maximum(np.concatenate([rgb for _, _, rgb in samples]), 1.0))
    order = np.unique(texel, return_inverse=True)[1]
    shared = np.bincount(order)[order] >= 2
    if shared.sum() < 50:
        return gains
    view_of, order, logs = view_of[shared], order[shared], logs[shared]
    g = np.zeros((views, 3))
    for _ in range(GAIN_ITERATIONS):
        residual = logs - g[view_of]
        surface = np.stack([_group_median(order, residual[:, c]) for c in range(3)], axis=1)
        offset = logs - surface[order]
        g = np.stack([_group_median(view_of, offset[:, c], views) for c in range(3)], axis=1)
        seen = np.bincount(view_of, minlength=views) > 0
        g[seen] -= np.median(g[seen], axis=0)
        g[~seen] = 0.0
    return np.exp(-g).astype(np.float32)


def _group_median(groups: np.ndarray, values: np.ndarray, size: int | None = None) -> np.ndarray:
    size = int(groups.max()) + 1 if size is None else size
    order = np.lexsort((values, groups))
    groups, values = groups[order], values[order]
    starts = np.searchsorted(groups, np.arange(size))
    ends = np.searchsorted(groups, np.arange(size), side="right")
    out = np.zeros(size)
    present = ends > starts
    middle = (starts + ends - 1) // 2
    middle_hi = (starts + ends) // 2
    out[present] = 0.5 * (values[middle[present]] + values[middle_hi[present]])
    return out


def _bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    top = image[y0, x0] * (1 - fx) + image[y0, x0 + 1] * fx
    bottom = image[y0 + 1, x0] * (1 - fx) + image[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def wall_grid(key: str, start: np.ndarray, end: np.ndarray, inward: np.ndarray, floor_z: float, ceiling_z: float) -> SurfaceGrid:
    direction = np.append((end - start) / np.linalg.norm(end - start), 0.0)
    return SurfaceGrid(key, "wall", np.array([start[0], start[1], floor_z]), direction, np.array([0.0, 0.0, 1.0]),
                       np.append(inward, 0.0), float(np.linalg.norm(end - start)), ceiling_z - floor_z)


def level_grid(key: str, kind: str, polygon: np.ndarray, z: float) -> SurfaceGrid:
    lo, hi = polygon.min(axis=0), polygon.max(axis=0)
    normal = np.array([0.0, 0.0, 1.0 if kind == "floor" else -1.0])
    return SurfaceGrid(key, kind, np.array([lo[0], lo[1], z]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                       normal, float(hi[0] - lo[0]), float(hi[1] - lo[1]))


GAIN_LATTICE = 8
GAIN_ITERATIONS = 10


def render(grid: SurfaceGrid, frames: list[Frame], poses: dict[int, np.ndarray], step: int = 1) -> Orthophoto:
    rows, cols = grid.shape
    world = grid.points()[::step, ::step].reshape(-1, 3)
    out_rows, out_cols = int(np.ceil(rows / step)), int(np.ceil(cols / step))
    lattice = np.zeros((out_rows, out_cols), dtype=bool)
    lattice[::GAIN_LATTICE, ::GAIN_LATTICE] = True
    lattice = lattice.reshape(-1)
    best_quality = np.zeros(len(world))
    best_view = np.full(len(world), -1)
    colour = np.zeros((len(world), 3), dtype=np.float32)
    samples: list[tuple[int, np.ndarray, np.ndarray]] = []
    view = -1
    for frame in frames:
        if frame.load_rgb is None:
            continue
        pose = poses[frame.index]
        centre = pose[:3, 3]
        to_point = world - centre
        distance = np.linalg.norm(to_point, axis=1)
        cosine = -(to_point @ grid.normal) / np.maximum(distance, 1e-9)
        candidate = (cosine > MIN_COSINE) & (distance < MAX_VIEW_M)
        if not candidate.any():
            continue
        cam = (world[candidate] - centre) @ pose[:3, :3]
        in_front = cam[:, 2] > 0.2
        rgb_K = frame.K
        width, height = frame.image_size
        px = rgb_K[0, 0] * cam[:, 0] / np.maximum(cam[:, 2], 1e-6) + rgb_K[0, 2]
        py = rgb_K[1, 1] * cam[:, 1] / np.maximum(cam[:, 2], 1e-6) + rgb_K[1, 2]
        inside = in_front & (px >= 0) & (py >= 0) & (px < width - 1) & (py < height - 1)
        if not inside.any():
            continue
        depth = frame.depth()
        if depth is not None:
            # Min-filtered depth: a texel next to a foreground edge counts as occluded, so
            # the silhouette of a wardrobe never bleeds onto the wall behind it.
            valid_depth = np.where(depth > 0, depth, np.inf).astype(np.float32)
            depth = cv2.erode(valid_depth, np.ones((5, 5), np.uint8))
            depth[~np.isfinite(depth)] = 0
            dh, dw = depth.shape
            dx = np.clip((px * dw / width).astype(int), 0, dw - 1)
            dy = np.clip((py * dh / height).astype(int), 0, dh - 1)
            measured = depth[dy, dx]
            visible = (measured <= 0) | (measured > cam[:, 2] - OCCLUSION_M)
            inside &= visible
        quality = cosine[candidate] / np.maximum(distance[candidate], 0.3)
        better = inside & (quality > best_quality[candidate])
        on_lattice = inside & lattice[candidate]
        if not better.any() and not on_lattice.any():
            continue
        view += 1
        image = frame.rgb().astype(np.float32)
        if image.shape[1] != width or image.shape[0] != height:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        if on_lattice.any():
            samples.append((view, np.nonzero(candidate)[0][on_lattice], _bilinear(image, px[on_lattice], py[on_lattice])))
        if not better.any():
            continue
        sampled = _bilinear(image, px[better], py[better])
        index = np.nonzero(candidate)[0][better]
        colour[index] = sampled
        best_quality[index] = quality[better]
        best_view[index] = view
    gains = _view_gains(samples, view + 1)
    painted = best_view >= 0
    colour[painted] *= gains[best_view[painted]]
    return Orthophoto(grid, colour.reshape(out_rows, out_cols, 3), (best_quality > 0).reshape(out_rows, out_cols),
                      best_quality.reshape(out_rows, out_cols))
