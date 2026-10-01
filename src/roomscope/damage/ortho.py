"""Metric orthophotos of room surfaces.

Each wall (u along the wall from its start corner, v height above the floor), the floor
and the ceiling are sampled on a TEXEL_M grid. Every texel takes its colour from the view
that sees it best: frontal, close, and not occluded, the last judged against that view's
depth (anything more than OCCLUSION_M in front of the surface blocks it, so a bed or a
wardrobe never paints onto the wall behind it). Pixel sizes are metric by construction,
so damage extent comes straight out in m2 and m.
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


def render(grid: SurfaceGrid, frames: list[Frame], poses: dict[int, np.ndarray], step: int = 1) -> Orthophoto:
    rows, cols = grid.shape
    world = grid.points()[::step, ::step].reshape(-1, 3)
    best_quality = np.zeros(len(world))
    colour = np.zeros((len(world), 3), dtype=np.float32)
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
        if not better.any():
            continue
        image = frame.rgb().astype(np.float32)
        if image.shape[1] != width or image.shape[0] != height:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        sampled = _bilinear(image, px[better], py[better])
        index = np.nonzero(candidate)[0][better]
        colour[index] = sampled
        best_quality[index] = quality[better]
    out_rows, out_cols = int(np.ceil(rows / step)), int(np.ceil(cols / step))
    return Orthophoto(grid, colour.reshape(out_rows, out_cols, 3), (best_quality > 0).reshape(out_rows, out_cols),
                      best_quality.reshape(out_rows, out_cols))
