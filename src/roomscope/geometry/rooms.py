"""World alignment and room segmentation.

Interior comes from free-space carving: every depth ray proves the space between the
camera and the surface it hit is empty. Walls come from the band above door-head height,
where every doorway still has wall (the header), so each room is enclosed by a closed
loop of wall evidence even with its doors open. Rooms are connected free space minus
walls, grown a little (never across a wall) to fill notches behind tall furniture.

A region the camera never stood in is not a room we measured: that rejects the phantom
space behind a mirror and rooms only glimpsed through a doorway.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from .cloud import Cloud
from .drift import yaw_matrix

CELL = 0.05
HEADER_MIN_Z = 2.15
MIN_ROOM_AREA = 1.0
MIN_FRAMES_INSIDE = 8
GROW_LIMIT_M = 0.6
CARVE_STRIDE = 6
CARVE_STEP_M = 0.04


def dominant_wall_angle(normals: np.ndarray) -> tuple[float, float]:
    """Dominant wall direction modulo 90 deg, via the circular mean of 4*theta (all four
    wall directions vote together), and how strongly normals agree (1 = all Manhattan)."""
    horizontal = normals[np.abs(normals[:, 2]) < 0.2]
    if len(horizontal) < 200:
        return 0.0, 0.0
    theta = np.arctan2(horizontal[:, 1], horizontal[:, 0])
    c, s = np.cos(4 * theta).sum(), np.sin(4 * theta).sum()
    return float(np.arctan2(s, c) / 4), float(np.hypot(c, s) / len(theta))


@dataclass
class Alignment:
    """World -> plan frame: walls on the x/y axes, floor at z = 0."""
    transform: np.ndarray
    floor_z: float
    wall_angle_deg: float

    def apply(self, cloud: Cloud) -> Cloud:
        rot, shift = self.transform[:3, :3], self.transform[:3, 3]
        return Cloud(cloud.points @ rot.T + shift, cloud.normals @ rot.T, cloud.frame_ids)

    def pose(self, pose: np.ndarray) -> np.ndarray:
        return self.transform @ pose


def robust_level(values: np.ndarray, window: float = 0.03) -> float:
    """Mode-seeking level estimate: histogram peak, then mean of values within window."""
    if len(values) == 0:
        return float("nan")
    hist, edges = np.histogram(values, bins=np.arange(values.min(), values.max() + 0.02, 0.01))
    peak = edges[np.argmax(hist)] + 0.005
    near = values[np.abs(values - peak) < window]
    return float(np.mean(near)) if len(near) else float(peak)


def align(cloud: Cloud) -> Alignment:
    angle, _ = dominant_wall_angle(cloud.normals)
    rot = yaw_matrix(-angle)
    up = cloud.normals[:, 2] > 0.9
    if up.sum() < 20:
        # No floor in view: keep the frame's own zero; layout levels say so later.
        transform = np.eye(4)
        transform[:3, :3] = rot
        return Alignment(transform, 0.0, float(np.rad2deg(angle)))
    z = cloud.points[up, 2]
    # The floor is the lowest large upward-facing level (tables and beds are higher).
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.02, 0.02))
    big = np.nonzero(hist > 0.15 * hist.max())[0]
    floor_guess = edges[big[0]] + 0.01
    floor_z = robust_level(z[np.abs(z - floor_guess) < 0.08])
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[2, 3] = -floor_z
    return Alignment(transform, floor_z, float(np.rad2deg(angle)))


@dataclass
class RoomRegion:
    id: str
    mask: np.ndarray
    origin: np.ndarray
    cell: float
    frames_inside: int
    first_frame: int

    def contains(self, xy: np.ndarray, dilate_cells: int = 0) -> np.ndarray:
        mask = self.mask if dilate_cells == 0 else cv2.dilate(
            self.mask.astype(np.uint8), np.ones((2 * dilate_cells + 1,) * 2, np.uint8)) > 0
        ij = np.floor((xy - self.origin) / self.cell).astype(int)
        inside = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < mask.shape[0]) & (ij[:, 1] < mask.shape[1])
        out = np.zeros(len(xy), dtype=bool)
        out[inside] = mask[ij[inside, 0], ij[inside, 1]]
        return out

    @property
    def area(self) -> float:
        return float(self.mask.sum() * self.cell ** 2)


SEGMENT_BIN = 0.02
SEGMENT_MIN_POINTS = 40
SEGMENT_GAP_M = 0.4
SEGMENT_MIN_M = 0.3
EXTEND_M = 1.0
BARRIER_MIN_M = 0.5


DIRECTION_BIN_DEG = 2.0
MIN_DIRECTION_SHARE = 0.04


@dataclass
class WallSegment:
    """A straight wall trace in the plan: points n . p = offset with n = (cos a, sin a),
    running from `start` to `end` along t = (-sin a, cos a). Any direction, not only the
    dominant axes: real homes have angled walls, bays and chamfered corners."""
    angle: float
    offset: float
    start: float
    end: float

    @property
    def normal(self) -> np.ndarray:
        return np.array([np.cos(self.angle), np.sin(self.angle)])

    @property
    def tangent(self) -> np.ndarray:
        return np.array([-np.sin(self.angle), np.cos(self.angle)])

    def point(self, s: float) -> np.ndarray:
        return self.normal * self.offset + self.tangent * s


def _wall_directions(normals: np.ndarray) -> list[float]:
    """Peaks of the wall-normal direction histogram, modulo 180 deg."""
    theta = np.mod(np.arctan2(normals[:, 1], normals[:, 0]), np.pi)
    bins = np.arange(0, np.pi + 1e-9, np.deg2rad(DIRECTION_BIN_DEG))
    hist, edges = np.histogram(theta, bins=bins)
    circular = np.concatenate([hist[-2:], hist, hist[:2]])
    smooth = np.convolve(circular, [1, 2, 3, 2, 1], mode="same")[2:-2]
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(np.concatenate([smooth[-3:], smooth, smooth[:3]]), height=MIN_DIRECTION_SHARE * len(theta),
                          distance=int(15 / DIRECTION_BIN_DEG))
    angles = sorted({round(float(edges[(p - 3) % len(hist)] + np.deg2rad(DIRECTION_BIN_DEG) / 2), 4) for p in peaks})
    return angles


def wall_segments(points: np.ndarray, normals: np.ndarray) -> list[WallSegment]:
    from scipy.signal import find_peaks

    segments = []
    for angle in _wall_directions(normals):
        n = np.array([np.cos(angle), np.sin(angle)])
        t = np.array([-np.sin(angle), np.cos(angle)])
        on_direction = np.abs(normals @ n) > 0.95
        coord, along = points[on_direction] @ n, points[on_direction] @ t
        if len(coord) < SEGMENT_MIN_POINTS:
            continue
        bins = np.arange(coord.min() - SEGMENT_BIN, coord.max() + 2 * SEGMENT_BIN, SEGMENT_BIN)
        hist, edges = np.histogram(coord, bins=bins)
        smooth = np.convolve(hist, [1, 2, 1], mode="same")
        peaks, _ = find_peaks(smooth, height=SEGMENT_MIN_POINTS, distance=4)
        for peak in peaks:
            centre = edges[peak] + SEGMENT_BIN / 2
            near = np.abs(coord - centre) < 0.03
            offset = float(np.median(coord[near]))
            spans = np.sort(along[near])
            breaks = np.nonzero(np.diff(spans) > SEGMENT_GAP_M)[0]
            starts = np.concatenate([[spans[0]], spans[breaks + 1]])
            ends = np.concatenate([spans[breaks], [spans[-1]]])
            for a, b in zip(starts, ends):
                if b - a >= SEGMENT_MIN_M:
                    segments.append(WallSegment(angle, offset, float(a), float(b)))
    return segments


def complete_corners(segments: list[WallSegment]) -> list[WallSegment]:
    """Extend each trace to where it meets a non-parallel trace, if that point lies within
    EXTEND_M of both traces' observed ends. Rooms are closed polygons, so a trace that stops
    short of a corner was just not observed there."""
    completed = []
    for seg in segments:
        start, end = seg.start, seg.end
        for other in segments:
            if abs(np.sin(seg.angle - other.angle)) < 0.3:
                continue
            matrix = np.array([seg.normal, other.normal])
            corner = np.linalg.solve(matrix, np.array([seg.offset, other.offset]))
            s_here, s_there = corner @ seg.tangent, corner @ other.tangent
            if not (other.start - EXTEND_M <= s_there <= other.end + EXTEND_M):
                continue
            if seg.start - EXTEND_M <= s_here < seg.start:
                start = min(start, s_here)
            if seg.end < s_here <= seg.end + EXTEND_M:
                end = max(end, s_here)
        completed.append(WallSegment(seg.angle, seg.offset, start, end))
    return completed


def _draw_segments(segments: list[WallSegment], origin: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for seg in segments:
        along = np.arange(seg.start, seg.end + CELL / 2, CELL / 2)
        xy = seg.normal * seg.offset + along[:, None] * seg.tangent
        ij = np.floor((xy - origin) / CELL).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
        mask[ij[ok, 0], ij[ok, 1]] = True
    return mask


MIN_WALL_SUPPORT = 0.4


def _wall_support(mask: np.ndarray, walls: np.ndarray) -> float:
    """Share of a region's boundary that runs along wall evidence. A room is enclosed;
    a fan of free space carved through a doorway or window is not."""
    boundary = mask & ~ndimage.binary_erosion(mask)
    near_wall = ndimage.binary_dilation(walls, iterations=3)
    return float((boundary & near_wall).sum() / max(boundary.sum(), 1))


CEILING_CELL_M = 0.25
UNDER_CEILING_M = 0.25


def _under_ceiling(cloud: Cloud) -> np.ndarray:
    """Vertical-surface points in the band just below the local ceiling. Ceiling height is
    mapped per 25 cm cell (rooms differ; bathrooms are often lower) and holes are filled
    from the nearest observed cell. Falls back to a fixed above-door band without ceiling."""
    vertical = np.abs(cloud.normals[:, 2]) < 0.3
    ceiling = (cloud.normals[:, 2] < -0.9) & (cloud.points[:, 2] > 1.8)
    if ceiling.sum() < 200:
        return vertical & (cloud.points[:, 2] > HEADER_MIN_Z)
    xy = cloud.points[:, :2]
    origin = xy.min(axis=0)
    shape = tuple(np.floor((xy.max(axis=0) - origin) / CEILING_CELL_M).astype(int) + 1)
    cells = np.floor((xy[ceiling] - origin) / CEILING_CELL_M).astype(int)
    total = np.zeros(shape)
    count = np.zeros(shape)
    np.add.at(total, (cells[:, 0], cells[:, 1]), cloud.points[ceiling, 2])
    np.add.at(count, (cells[:, 0], cells[:, 1]), 1)
    level = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    _, (ii, jj) = ndimage.distance_transform_edt(np.isnan(level), return_indices=True)
    level = level[ii, jj]
    ij = np.floor((xy - origin) / CEILING_CELL_M).astype(int)
    local = level[ij[:, 0], ij[:, 1]]
    z = cloud.points[:, 2]
    return vertical & (z > local - UNDER_CEILING_M) & (z < local - 0.02) & (z > HEADER_MIN_Z - 0.1)


def _grow_to_walls(mask: np.ndarray, walls: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Geodesic growth up to GROW_LIMIT_M, never into wall cells or another kept room:
    fills the unseen strip behind a wardrobe or over a sink without leaking through gaps.
    Discarded scraps of free space (too small, never entered) do not block growth."""
    blocked = walls | others
    grown = mask.copy()
    for _ in range(int(GROW_LIMIT_M / CELL)):
        grown = ndimage.binary_dilation(grown, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]]) & ~blocked
    return ndimage.binary_fill_holes(grown | mask)


def _rasterise(xy: np.ndarray, origin: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    ij = np.floor((xy - origin) / CELL).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    grid = np.zeros(shape, dtype=np.int32)
    np.add.at(grid, (ij[ok, 0], ij[ok, 1]), 1)
    return grid


def carve_free_space(frames, poses: dict[int, np.ndarray], origin: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Count, per plan cell, depth rays that passed through it (endpoint excluded)."""
    counts = np.zeros(shape[0] * shape[1], dtype=np.int64)
    for frame in frames:
        depth = frame.depth()[::CARVE_STRIDE, ::CARVE_STRIDE].astype(np.float64)
        K = frame.depth_K
        v, u = np.nonzero(depth > 0)
        z = depth[v, u]
        cam = np.stack([(u * CARVE_STRIDE - K[0, 2]) / K[0, 0] * z, (v * CARVE_STRIDE - K[1, 2]) / K[1, 1] * z, z], 1)
        pose = poses[frame.index]
        ends = cam @ pose[:3, :3].T + pose[:3, 3]
        start = pose[:2, 3]
        span = ends[:, :2] - start
        length = np.linalg.norm(span, axis=1)
        steps = np.maximum((length / CARVE_STEP_M).astype(int), 1)
        ray = np.repeat(np.arange(len(ends)), steps)
        frac = (np.arange(len(ray)) - np.repeat(np.cumsum(steps) - steps, steps)) / np.repeat(steps, steps)
        samples = start + span[ray] * (frac[:, None] * 0.97)
        ij = np.floor((samples - origin) / CELL).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
        counts += np.bincount(ij[ok, 0] * shape[1] + ij[ok, 1], minlength=len(counts))
    return counts.reshape(shape)


def segment_rooms(cloud: Cloud, frames, poses: dict[int, np.ndarray],
                  min_frames_inside: int = MIN_FRAMES_INSIDE, single_room: bool = False) -> tuple[list[RoomRegion], list[str]]:
    """single_room: the capture is known to be one room (a photo folder). Nothing is dropped
    for enclosure or camera presence; the best-supported region is returned, else all of
    the carved free space."""
    """cloud and poses in the plan frame."""
    xy = cloud.points[:, :2]
    origin = xy.min(axis=0) - 0.5
    shape = tuple((np.ceil((xy.max(axis=0) + 0.5 - origin) / CELL)).astype(int))
    wall = _under_ceiling(cloud)
    # Barriers are long straight wall traces that reach the ceiling. Real walls and door
    # headers do; clutter above door height mostly doesn't (shower screens and rails,
    # tall cabinets, pendant lamps) and would otherwise cut a room into pieces.
    traces = complete_corners([s for s in wall_segments(cloud.points[wall, :2], cloud.normals[wall, :2])
                               if s.end - s.start >= BARRIER_MIN_M])
    wall_grid = ndimage.binary_dilation(_draw_segments(traces, origin, shape), structure=np.ones((3, 3)))
    free = (carve_free_space(frames, poses, origin, shape) >= 3) & ~wall_grid
    free = ndimage.binary_opening(free, structure=np.ones((3, 3)))
    labels, count = ndimage.label(free, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    camera_xy = np.array([poses[f.index][:2, 3] for f in frames])
    frame_order = np.array([f.index for f in frames])

    cam_ij = np.floor((camera_xy - origin) / CELL).astype(int)
    cam_ok = (cam_ij[:, 0] >= 0) & (cam_ij[:, 1] >= 0) & (cam_ij[:, 0] < shape[0]) & (cam_ij[:, 1] < shape[1])
    cam_label = np.zeros(len(camera_xy), dtype=int)
    cam_label[cam_ok] = labels[cam_ij[cam_ok, 0], cam_ij[cam_ok, 1]]

    kept, notes = [], []
    for label in range(1, count + 1):
        mask = labels == label
        area = mask.sum() * CELL ** 2
        inside = np.nonzero(cam_label == label)[0]
        if area < MIN_ROOM_AREA:
            continue
        if len(inside) < min_frames_inside and not single_room:
            notes.append(f"dropped a {area:.1f} m2 region the camera never entered "
                         "(seen through a doorway or in a mirror)")
            continue
        support = _wall_support(mask, wall_grid)
        if support < MIN_WALL_SUPPORT and not single_room:
            notes.append(f"dropped a {area:.1f} m2 region only {support:.0%} enclosed by walls (open space seen "
                         "through an opening, not a room)")
            continue
        kept.append((label, inside))

    if single_room:
        if kept:
            kept = [max(kept, key=lambda item: (len(item[1]), (labels == item[0]).sum()))]
        elif free.any():
            labels = ndimage.label(ndimage.binary_closing(free, structure=np.ones((5, 5))))[0]
            biggest = int(np.argmax(np.bincount(labels.ravel())[1:])) + 1
            kept = [(biggest, np.nonzero(cam_label >= 0)[0])]
            notes.append("photo room: no enclosed region; using all carved free space")
    kept_cells = np.isin(labels, [label for label, _ in kept])
    regions = []
    for label, inside in kept:
        own = labels == label
        mask = _grow_to_walls(ndimage.binary_fill_holes(own), wall_grid, kept_cells & ~own)
        regions.append(RoomRegion("", mask, origin, CELL, len(inside), int(frame_order[inside.min()])))
    regions.sort(key=lambda r: r.first_frame)
    for number, region in enumerate(regions, start=1):
        region.id = f"room_{number}"
    return regions, notes
