"""Per-room layout: footprint polygon from fitted wall lines, floor and ceiling levels.

The coarse room mask only seeds the polygon. Each wall line is then fitted to that
wall's own points: first from the band just under the ceiling, where furniture rarely
reaches, then refined over all heights within a few centimetres of that estimate. Wall
angles are fitted, not snapped to 90 degrees; rooms are often a fraction of a degree out
of square, and snapping a 4 m wall by 0.5 deg moves its corners ~1.7 cm.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from .cloud import Cloud
from .rooms import RoomRegion, robust_level

AXIS_SNAP_DEG = 12.0
MIN_EDGE_M = 0.25
SEARCH_M = 0.30
CORNER_MARGIN_M = 0.15
INLIER_M = 0.03
UPPER_BAND_M = 0.6
EFFECTIVE_POINTS_CAP = 400


@dataclass
class WallLine:
    """Line of a wall's interior face. normal points into the room; points p on the line
    satisfy normal . p = offset."""
    normal: np.ndarray
    offset: float
    sigma: float
    points: int
    spread: float

    @property
    def direction(self) -> np.ndarray:
        # CCW polygon: walking along the wall the room is on the left, i.e. the inward
        # normal is the direction rotated +90 deg.
        return np.array([self.normal[1], -self.normal[0]])


@dataclass
class RoomLayout:
    id: str
    polygon: np.ndarray
    lines: list[WallLine]
    floor_z: float
    ceiling_z: float
    floor_sigma: float
    ceiling_sigma: float
    points: Cloud
    notes: list[str]

    @property
    def ceiling_height(self) -> float:
        return self.ceiling_z - self.floor_z


def _intersect(a: WallLine, b: WallLine) -> np.ndarray | None:
    matrix = np.array([a.normal, b.normal])
    if abs(np.linalg.det(matrix)) < 0.1:
        return None
    return np.linalg.solve(matrix, np.array([a.offset, b.offset]))


def _mask_polygon(region: RoomRegion) -> np.ndarray:
    mask = ndimage.binary_opening(region.mask, structure=np.ones((3, 3))).astype(np.uint8)
    contours, _ = cv2.findContours(mask.T.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(contour, epsilon=2.0, closed=True)[:, 0, :].astype(float)
    # contour is in (col=i, row=j) of mask.T, i.e. (x cell, y cell); cell centres.
    polygon = region.origin + (approx + 0.5) * region.cell
    area = 0.5 * np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - polygon[:, 1] * np.roll(polygon[:, 0], -1))
    return polygon if area > 0 else polygon[::-1]


def _regularise(polygon: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Polygon -> list of (inward normal, point on edge) with near-axis edges snapped to
    the axis as a starting guess, tiny edges and collinear runs merged."""
    edges = []
    n = len(polygon)
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        span = b - a
        length = np.linalg.norm(span)
        if length < 1e-6:
            continue
        direction = span / length
        angle = np.degrees(np.arctan2(direction[1], direction[0])) % 90
        if min(angle, 90 - angle) < AXIS_SNAP_DEG:
            axis = int(abs(direction[1]) > abs(direction[0]))
            direction = np.zeros(2)
            direction[axis] = np.sign(span[axis])
        inward = np.array([-direction[1], direction[0]])
        edges.append([inward, (a + b) / 2, length])

    merged: list[list] = []
    for inward, mid, length in edges:
        if merged and np.dot(merged[-1][0], inward) > 0.99 and \
                abs(np.dot(mid - merged[-1][1], inward)) < 0.12:
            prev = merged[-1]
            weight = prev[2] + length
            prev[1] = (prev[1] * prev[2] + mid * length) / weight
            prev[2] = weight
        else:
            merged.append([inward, mid, length])
    if len(merged) > 2 and np.dot(merged[0][0], merged[-1][0]) > 0.99 and \
            abs(np.dot(merged[0][1] - merged[-1][1], merged[0][0])) < 0.12:
        first, last = merged[0], merged.pop()
        weight = first[2] + last[2]
        first[1] = (first[1] * first[2] + last[1] * last[2]) / weight
        first[2] = weight
    kept = [(inward, mid) for inward, mid, length in merged if length >= MIN_EDGE_M]
    return kept


def _tukey_line(points: np.ndarray, normal: np.ndarray, offset: float, iterations: int = 8) -> tuple[np.ndarray, float, np.ndarray]:
    """Robust total-least-squares line: returns (unit normal, offset, final weights)."""
    weights = np.ones(len(points))
    for _ in range(iterations):
        residual = points @ normal - offset
        c = 4.685 * max(1.4826 * np.median(np.abs(residual)), 0.002)
        weights = np.where(np.abs(residual) < c, (1 - (residual / c) ** 2) ** 2, 0.0)
        if weights.sum() < 10:
            break
        centre = (points * weights[:, None]).sum(0) / weights.sum()
        centred = (points - centre) * np.sqrt(weights)[:, None]
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        fitted = vt[1] if np.dot(vt[1], normal) > 0 else -vt[1]
        normal, offset = fitted, float(fitted @ centre)
    return normal, offset, weights


def fit_wall(points: Cloud, inward: np.ndarray, anchor: np.ndarray, span: tuple[float, float],
             floor_z: float, ceiling_z: float) -> WallLine | None:
    """Fit the wall whose interior face passes near `anchor` with inward normal `inward`,
    using only points on that face between span (along-wall coordinates)."""
    direction = np.array([inward[1], -inward[0]])
    rel = points.points[:, :2] - anchor
    across = rel @ inward
    along = rel @ direction
    z = points.points[:, 2]
    facing = (points.normals[:, :2] @ inward > 0.8) & (np.abs(points.normals[:, 2]) < 0.3)
    base = facing & (np.abs(across) < SEARCH_M) & (along > span[0] + CORNER_MARGIN_M) & \
        (along < span[1] - CORNER_MARGIN_M) & (z > floor_z + 0.15) & (z < ceiling_z - 0.05)
    upper = base & (z > ceiling_z - UPPER_BAND_M)
    seed_rows = upper if upper.sum() >= 30 else base
    if seed_rows.sum() < 30:
        return None
    seed = robust_level(across[seed_rows], window=INLIER_M)
    rows = base & (np.abs(across - seed) < INLIER_M)
    if rows.sum() < 30:
        return None
    xy = points.points[rows, :2]
    normal, offset, weights = _tukey_line(xy, inward, float(inward @ anchor + seed))
    residual = xy @ normal - offset
    good = weights > 0
    spread = float(1.4826 * np.median(np.abs(residual[good])))
    effective = min(int(good.sum()), EFFECTIVE_POINTS_CAP)
    return WallLine(normal, offset, spread / np.sqrt(effective), int(good.sum()), spread)


def _level(values: np.ndarray) -> tuple[float, float]:
    level = robust_level(values, window=0.02)
    near = values[np.abs(values - level) < 0.02]
    spread = 1.4826 * np.median(np.abs(near - level)) if len(near) else 0.02
    return level, float(spread / np.sqrt(min(len(near), EFFECTIVE_POINTS_CAP)) if len(near) else 0.02)


def room_layout(region: RoomRegion, cloud: Cloud) -> RoomLayout | None:
    notes: list[str] = []
    near_room = region.contains(cloud.points[:, :2], dilate_cells=int(SEARCH_M / region.cell) + 2)
    points = cloud.subset(near_room)
    eroded = RoomRegion(region.id, ndimage.binary_erosion(region.mask, iterations=4), region.origin,
                        region.cell, region.frames_inside, region.first_frame)
    core = eroded.contains(points.points[:, :2])
    floor_rows = core & (points.normals[:, 2] > 0.9) & (points.points[:, 2] < 0.3)
    ceiling_rows = core & (points.normals[:, 2] < -0.9) & (points.points[:, 2] > 1.8)
    if floor_rows.sum() < 50 or ceiling_rows.sum() < 50:
        notes.append("floor or ceiling barely observed; heights unreliable")
    floor_z, floor_sigma = _level(points.points[floor_rows, 2]) if floor_rows.sum() else (0.0, 0.05)
    ceiling_z, ceiling_sigma = _level(points.points[ceiling_rows, 2]) if ceiling_rows.sum() else (2.5, 0.2)

    edges = _regularise(_mask_polygon(region))
    lines: list[WallLine] = []
    guesses = []
    for k, (inward, mid) in enumerate(edges):
        prev_mid = edges[k - 1][1]
        next_mid = edges[(k + 1) % len(edges)][1]
        direction = np.array([inward[1], -inward[0]])
        ends = sorted([float((prev_mid - mid) @ direction), float((next_mid - mid) @ direction)])
        # Along-wall extent: from the previous edge's line to the next edge's line.
        span = (min(ends[0], -0.05), max(ends[1], 0.05))
        # The mask sits inside the dilated wall band, so the true face is outward of `mid`.
        line = fit_wall(points, inward, mid, span, floor_z, ceiling_z)
        if line is None:
            notes.append(f"wall {k}: too few points on its face; kept the coarse mask edge")
            line = WallLine(inward, float(inward @ mid) - 0.08, 0.05, 0, 0.05)
        lines.append(line)
        guesses.append(mid)

    # Clean-up, repeated until stable:
    #  - consecutive walls can't be parallel (same or opposite facing); keep the one with
    #    more support. This also happens after an artefact edge between them is dropped.
    #  - a real wall has material along most of its length just under the ceiling (above
    #    doors, windows and furniture); edges that don't are mask artefacts.
    while len(lines) > 3:
        parallel = next((k for k in range(len(lines)) if abs(lines[k - 1].normal @ lines[k].normal) > 0.97), None)
        if parallel is not None:
            weaker = parallel if lines[parallel].points < lines[parallel - 1].points else parallel - 1
            notes.append("merged two consecutive parallel wall lines")
            lines.pop(weaker)
            continue
        polygon = _corners(lines)
        if polygon is None:
            break
        coverage = [_upper_coverage(points, line, polygon[k], polygon[(k + 1) % len(polygon)], ceiling_z)
                    for k, line in enumerate(lines)]
        worst = int(np.argmin(coverage))
        if coverage[worst] >= MIN_UPPER_COVERAGE:
            break
        notes.append(f"dropped an edge with {coverage[worst]:.0%} wall coverage under the ceiling (mask artefact)")
        lines.pop(worst)

    polygon = _corners(lines)
    if polygon is None or len(polygon) < 3:
        return None
    # Vertex k is the start of wall k (intersection of wall k-1 and wall k).
    return RoomLayout(region.id, polygon, lines, floor_z, ceiling_z, floor_sigma, ceiling_sigma, points, notes)


MIN_UPPER_COVERAGE = 0.35
COVERAGE_BIN_M = 0.1


def _corners(lines: list[WallLine]) -> np.ndarray | None:
    corners = []
    for k in range(len(lines)):
        corner = _intersect(lines[k - 1], lines[k])
        if corner is None:
            return None
        corners.append(corner)
    return np.array(corners)


def _upper_coverage(points: Cloud, line: WallLine, start: np.ndarray, end: np.ndarray, ceiling_z: float) -> float:
    direction = line.direction
    length = float((end - start) @ direction)
    if length < COVERAGE_BIN_M:
        return 0.0
    xy = points.points[:, :2]
    on_face = (np.abs(xy @ line.normal - line.offset) < INLIER_M) & \
        (points.normals[:, :2] @ line.normal > 0.8) & (points.points[:, 2] > ceiling_z - UPPER_BAND_M) & \
        (points.points[:, 2] < ceiling_z - 0.03)
    along = (xy[on_face] - start) @ direction
    bins = np.floor(along[(along >= 0) & (along < length)] / COVERAGE_BIN_M).astype(int)
    filled = np.count_nonzero(np.bincount(bins, minlength=int(np.ceil(length / COVERAGE_BIN_M))) >= 2)
    return filled / np.ceil(length / COVERAGE_BIN_M)
