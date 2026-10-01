"""Room layout from sparse views (photo tier): outermost strong wall per side.

From 2-8 photos the fused geometry is noisy and full of gaps, and it includes glimpses
through doorways, so the dense LiDAR wall fitter has too little to work with. A stronger
shape prior works better. In the Manhattan-aligned frame, each side's wall is the
nearest substantial vertical surface beyond the camera positions facing back into the
room: substantial meaning real support and an along-wall extent spanning much of the
room (furniture fronts fail this), nearest because walls glimpsed through doorways and
the fake planes a depth network puts behind a window's sky or inside a mirror all lie
farther out.
Each wall's offset is the robust mean of its points; its sigma comes from their spread.
The result is the room's bounding rectangle: L-shaped rooms come out as their rectangle
(a documented limitation of the photo tier).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .cloud import Cloud
from .layout import RoomLayout, WallLine, _level

BIN_M = 0.02
STRONG_SHARE = 0.15
SUPPORT_SHARE = 0.25
EXTENT_SHARE = 0.5
INLIER_M = 0.06
MIN_SIGMA_M = 0.02
EFFECTIVE_POINTS = 50
UNSEEN_SIGMA_M = 0.15
FOOTPRINT_MARGIN_M = 0.05


def _side(points: np.ndarray, normals: np.ndarray, axis: int, sign: int, beyond: float) -> tuple[float, float, int] | None:
    """Wall on the +axis (sign=1) or -axis (sign=-1) side: points beyond `beyond` whose
    normal faces back into the room."""
    coord = points[:, axis]
    facing = normals[:, axis] * -sign > 0.85
    rows = facing & ((coord - beyond) * sign > 0.05)
    if rows.sum() < 30:
        return None
    values = coord[rows]
    hist, edges = np.histogram(values, bins=np.arange(values.min() - BIN_M, values.max() + 2 * BIN_M, BIN_M))
    smooth = np.convolve(hist, [1, 2, 3, 2, 1], mode="same")
    peaks, _ = find_peaks(smooth, height=max(10, STRONG_SHARE * smooth.max()))
    if len(peaks) == 0:
        return None
    centres = edges[peaks] + BIN_M / 2
    along = points[rows, 1 - axis]
    support, extent = [], []
    for centre in centres:
        inliers = np.abs(values - centre) < INLIER_M
        support.append(int(inliers.sum()))
        extent.append(np.percentile(along[inliers], 95) - np.percentile(along[inliers], 5) if inliers.sum() > 10 else 0.0)
    support, extent = np.array(support), np.array(extent)
    # Substantial candidates: real support and spanning much of the room. Of those, the
    # nearest is the wall; glimpses through doors, window "sky" planes and mirror
    # phantoms all lie farther out, furniture fronts fail the extent test.
    extent = np.nan_to_num(extent)
    substantial = (support >= SUPPORT_SHARE * support.max()) & (extent >= EXTENT_SHARE * extent.max())
    if not substantial.any():
        substantial = support == support.max()
    chosen = centres[substantial][np.argmin((centres[substantial] - beyond) * sign)]
    near = values[np.abs(values - chosen) < INLIER_M]
    offset = float(np.median(near))
    spread = float(1.4826 * np.median(np.abs(near - offset)))
    return offset, max(spread / np.sqrt(min(len(near), EFFECTIVE_POINTS)), MIN_SIGMA_M), len(near)


def rectangle_layout(room: str, cloud: Cloud, cameras_xy: np.ndarray) -> RoomLayout | None:
    z = cloud.points[:, 2]
    floor_rows = cloud.normals[:, 2] > 0.9
    ceiling_rows = (cloud.normals[:, 2] < -0.9) & (z > 1.8)
    floor_z, floor_sigma = _level(z[floor_rows & (z < 0.4)]) if (floor_rows & (z < 0.4)).sum() > 50 else (0.0, 0.05)
    ceiling_z, ceiling_sigma = _level(z[ceiling_rows]) if ceiling_rows.sum() > 50 else (float(np.percentile(z, 99)), 0.15)
    vertical = (np.abs(cloud.normals[:, 2]) < 0.3) & (z > floor_z + 0.3) & (z < ceiling_z - 0.08)
    points, normals = cloud.points[vertical, :2], cloud.normals[vertical, :2]
    lo, hi = cameras_xy.min(axis=0), cameras_xy.max(axis=0)
    sides, unseen = {}, []
    everything = cloud.points[(z > floor_z + 0.1) & (z < ceiling_z - 0.05), :2]
    for axis in (0, 1):
        for sign, beyond in ((1, hi[axis]), (-1, lo[axis])):
            found = _side(points, normals, axis, sign, beyond)
            if found is None:
                # No wall seen on this side: take the extent of whatever was observed there,
                # with a wide sigma, rather than lose the room.
                extent = float(np.percentile(everything[:, axis] * sign, 98) * sign) if len(everything) else beyond + sign
                found = (extent, UNSEEN_SIGMA_M, 0)
                unseen.append({(0, 1): "east", (0, -1): "west", (1, 1): "north", (1, -1): "south"}[(axis, sign)])
            sides[(axis, sign)] = found
    x0, x1 = sides[(0, -1)][0], sides[(0, 1)][0]
    y0, y1 = sides[(1, -1)][0], sides[(1, 1)][0]
    if x1 - x0 < 0.8 or y1 - y0 < 0.8:
        return None
    polygon = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    # Levels again from this room's own footprint (to the wall faces): through a doorway
    # the neighbour's ceiling, or a step down to its floor, is in view too.
    xy = cloud.points[:, :2]
    margin = FOOTPRINT_MARGIN_M
    inside = (xy[:, 0] > x0 - margin) & (xy[:, 0] < x1 + margin) & (xy[:, 1] > y0 - margin) & (xy[:, 1] < y1 + margin)
    if (floor_rows & inside & (z < floor_z + 0.3)).sum() > 50:
        floor_z, floor_sigma = _level(z[floor_rows & inside & (z < floor_z + 0.3)])
    if (ceiling_rows & inside).sum() > 50:
        ceiling_z, ceiling_sigma = _level(z[ceiling_rows & inside])
    # CCW walls: south (inward +y), east (inward -x), north (inward -y), west (inward +x).
    lines = [WallLine(np.array([0.0, 1.0]), y0, sides[(1, -1)][1], sides[(1, -1)][2], sides[(1, -1)][1]),
             WallLine(np.array([-1.0, 0.0]), -x1, sides[(0, 1)][1], sides[(0, 1)][2], sides[(0, 1)][1]),
             WallLine(np.array([0.0, -1.0]), -y1, sides[(1, 1)][1], sides[(1, 1)][2], sides[(1, 1)][1]),
             WallLine(np.array([1.0, 0.0]), x0, sides[(0, -1)][1], sides[(0, -1)][2], sides[(0, -1)][1])]
    notes = ["sparse-view layout: best-supported wall per side (rectangle)"]
    if unseen:
        notes.append(f"no wall seen on the {', '.join(unseen)} side; taken from the observed extent (15 cm sigma)")
    if not (ceiling_rows & inside).sum() > 50:
        notes.append("ceiling barely seen; height from the highest points (wide interval)")
    return RoomLayout(room, polygon, lines, floor_z, ceiling_z, floor_sigma, ceiling_sigma, cloud, notes)
