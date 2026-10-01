"""Ground truth for a real capture, measured on the laser scan of the same rooms.

  python benchmark/real/laser_truth.py runs/bench/real/47429914/plan.json data/public/arkitscenes/laser/471428 \
      --out benchmark/ground_truth/arkitscenes_471428.yaml --overlay docs/real/laser_471428.png

A survey, done the way the benchmark protocol asks a person with a laser meter to do it:
the plan says which wall is which, the scan says where it is. The plan is placed on the
scan by a yaw search and a chamfer refinement over a slice at tape height. Each wall's face
is then found in the scan near the plan's wall, as the densest offset in each 10 cm strip,
and fitted as a line through the strips that agree (so a strip through a window recess or
a wardrobe front moves neither its position nor its direction). Corners are where
neighbouring faces meet; wall lengths, floor area, ceiling height (floor to ceiling layer,
median over the room) and openings all come from the scan, never from the plan's numbers.
A wall length is marked measured only if its face and both neighbours' were found.
The overlay and the wall elevations are the check that the plan named real walls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage, optimize, signal

PLY_DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1"),
                      ("quality", "<f8"), ("radius", "<f8")])
CELL_M = 0.02
TAPE_BAND_M = (0.9, 1.6)
UPPER_BAND_M = (0.30, 0.08)
WALL_SEARCH_M = 0.30
STRIP_M = 0.10
MIN_STRIP_POINTS = 8
CHAMFER_CAP_M = 0.10


def load_scans(folder: Path, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """The scans of a visit share one frame already (their pose files map scanner to that
    frame and are applied in the published points); a pose's translation is where the
    scanner stood."""
    clouds, stations = [], []
    for ply in sorted(folder.glob("*.ply")):
        head = ply.open("rb").read(4096)
        offset = head.index(b"end_header\n") + len(b"end_header\n")
        points = np.memmap(ply, dtype=PLY_DTYPE, mode="r", offset=offset)[::stride]
        clouds.append(np.stack([points["x"], points["y"], points["z"]], axis=1).astype(np.float64))
        stations.append(np.loadtxt(ply.with_name(ply.stem + "_pose.txt"), delimiter=",")[3, :3])
    return np.vstack(clouds), np.array(stations)


def _layers(points: np.ndarray) -> tuple[float, float]:
    """Lowest and highest horizontal layers that cover a large area. Counting points
    would not do: the scanner sits close under the ceiling, which gets many times the
    floor's points, and a table top is dense but small."""
    z = points[:, 2]
    bins = np.arange(z.min(), z.max() + 0.01, 0.01)
    which = np.digitize(z, bins)
    cells = np.floor(points[:, :2] / 0.05).astype(np.int64)
    key = (cells[:, 0] * 1_000_003 + cells[:, 1]) * 4096 + which
    unique = np.unique(key)
    area = np.bincount(unique % 4096, minlength=len(bins) + 1) * 0.0025
    wide = np.nonzero(area >= 0.3 * area.max())[0]
    return _layer(z, bins[wide[0] - 1] + 0.005), _layer(z, bins[wide[-1] - 1] + 0.005)


def level(points: np.ndarray, stations: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Rotate so the floor's normal is +z; return points, scanner stations, floor z and
    ceiling z."""
    for _ in range(2):
        floor_z, _ = _layers(points)
        layer = points[np.abs(points[:, 2] - floor_z) < 0.02]
        centred = layer - layer.mean(axis=0)
        normal = np.linalg.svd(centred[:: max(1, len(centred) // 200000)], full_matrices=False)[2][2]
        normal = normal if normal[2] > 0 else -normal
        axis = np.cross(normal, [0, 0, 1.0])
        angle = np.arcsin(min(1.0, np.linalg.norm(axis)))
        if angle > 1e-5:
            axis /= np.linalg.norm(axis)
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            rotation = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
            points, stations = points @ rotation.T, stations @ rotation.T
    floor_z, ceiling_z = _layers(points)
    return points, stations, floor_z, ceiling_z


def _layer(z: np.ndarray, near: float) -> float:
    close = z[np.abs(z - near) < 0.03]
    counts, edges = np.histogram(close, bins=np.arange(near - 0.03, near + 0.031, 0.002))
    peak = edges[np.argmax(counts)] + 0.001
    return float(np.median(close[np.abs(close - peak) < 0.008]))


def _band(points: np.ndarray, low: float, high: float) -> np.ndarray:
    return points[(points[:, 2] > low) & (points[:, 2] < high), :2]


def _samples(plan: dict) -> np.ndarray:
    out = []
    for room in plan["rooms"]:
        for wall in room["walls"]:
            a, b = np.array(wall["start"]), np.array(wall["end"])
            n = max(2, int(np.linalg.norm(b - a) / CELL_M))
            out.append(a + np.linspace(0, 1, n)[:, None] * (b - a))
    return np.vstack(out)


def _rotate(xy: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return xy @ np.array([[c, s], [-s, c]])


def place(plan: dict, slice_xy: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Yaw and shift taking plan coordinates onto the scan: exhaustive yaw at 1 degree with
    the best shift for each from a cross-correlation, then a capped chamfer refinement."""
    lo = slice_xy.min(axis=0) - 1.0
    shape = tuple(((slice_xy.max(axis=0) + 1.0 - lo) / CELL_M).astype(int) + 1)
    occupied = np.zeros(shape, dtype=np.float32)
    cells = ((slice_xy - lo) / CELL_M).astype(int)
    np.add.at(occupied, (cells[:, 0], cells[:, 1]), 1)
    occupied = ndimage.binary_dilation(occupied > 2).astype(np.float32)
    distance = ndimage.distance_transform_edt(occupied == 0) * CELL_M
    samples = _samples(plan)
    best = (-1.0, 0.0, np.zeros(2))
    for yaw in np.radians(np.arange(0, 360, 1.0)):
        rotated = _rotate(samples, yaw)
        origin = rotated.min(axis=0)
        template_shape = tuple(((rotated.max(axis=0) - origin) / CELL_M).astype(int) + 1)
        template = np.zeros(template_shape, dtype=np.float32)
        idx = ((rotated - origin) / CELL_M).astype(int)
        template[idx[:, 0], idx[:, 1]] = 1
        score = signal.fftconvolve(occupied, template[::-1, ::-1], mode="valid")
        peak = np.unravel_index(np.argmax(score), score.shape)
        if score[peak] > best[0]:
            best = (float(score[peak]), yaw, lo + np.array(peak) * CELL_M - origin)

    def cost(params: np.ndarray) -> float:
        moved = _rotate(samples, params[0]) + params[1:]
        rc = (moved - lo) / CELL_M
        d = ndimage.map_coordinates(distance, rc.T, order=1, mode="nearest")
        return float(np.mean(np.minimum(d, CHAMFER_CAP_M) ** 2))

    start = np.array([best[1], *best[2]])
    simplex = np.vstack([start, start + [0.02, 0, 0], start + [0, 0.05, 0], start + [0, 0, 0.05]])
    refined = optimize.minimize(cost, start, method="Nelder-Mead",
                                options={"xatol": 1e-5, "fatol": 1e-10, "initial_simplex": simplex, "maxiter": 2000})
    yaw, shift = refined.x[0], refined.x[1:]
    moved = _rotate(samples, yaw) + shift
    d = ndimage.map_coordinates(distance, ((moved - lo) / CELL_M).T, order=1, mode="nearest")
    return float(yaw), shift, float(np.mean(d < 0.05))


def wall_face(a: np.ndarray, b: np.ndarray, outward: np.ndarray, points_xy: np.ndarray) -> dict:
    direction = (b - a) / np.linalg.norm(b - a)
    length = float(np.linalg.norm(b - a))
    rel = points_xy - a
    along, offset = rel @ direction, rel @ outward
    trim = min(0.10, 0.2 * length)
    near = (along > trim) & (along < length - trim) & (np.abs(offset) < WALL_SEARCH_M)
    along, offset = along[near], offset[near]
    modes = []
    for start in np.arange(trim, length - trim, STRIP_M):
        strip = offset[(along >= start) & (along < start + STRIP_M)]
        if len(strip) < MIN_STRIP_POINTS:
            continue
        counts, edges = np.histogram(strip, bins=np.arange(-WALL_SEARCH_M, WALL_SEARCH_M + 0.005, 0.005))
        peak = edges[np.argmax(counts)] + 0.0025
        modes.append((start + STRIP_M / 2, float(np.mean(strip[np.abs(strip - peak) < 0.01]))))
    if len(modes) < 2:
        return {"offset": None, "slope": 0.0, "strips": len(modes)}
    centres, modes = np.array(modes).T
    # The face is a line, not the plan's direction: fit it to the strips that agree with
    # the median (a window recess or a wardrobe strip is an outlier, not a tilt).
    slope, intercept = 0.0, float(np.median(modes))
    for tolerance in (0.03, 0.01, 0.005):
        inlier = np.abs(modes - (intercept + slope * centres)) < tolerance
        if inlier.sum() < 3:
            break
        slope, intercept = np.polyfit(centres[inlier], modes[inlier], 1)
    residual = modes[inlier] - (intercept + slope * centres[inlier])
    return {"offset": float(intercept), "slope": float(slope), "strips": int(len(modes)),
            "spread": float(1.4826 * np.median(np.abs(residual)))}


def _intersect(p: np.ndarray, d: np.ndarray, q: np.ndarray, e: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    matrix = np.array([d, -e]).T
    if abs(np.linalg.det(matrix)) < np.sin(np.radians(10)):
        return fallback
    t = np.linalg.solve(matrix, q - p)
    return p + t[0] * d


FACE_M = 0.05
THROUGH_M = 0.05
ALONG_CELL_M = 0.01
HEIGHT_CELL_M = 0.02
MIN_OPENING_M = 0.30
DOOR_SILL_M = 0.10
BOUNDED_SHARE = 0.3
MIN_FILL = 0.8
SHADOW_STRIDE = 5
SHADOW_HIT_M = 0.03
RECESS_DEPTH_M = 1.0
RECESS_COVER = 0.6
MIRROR_MATCH_M = 0.03
MIRROR_SHARE = 0.5
MULLION_SHARE = 0.7
DEBUG = False


def _mirror(beyond: np.ndarray, p: np.ndarray, outward: np.ndarray, tree) -> bool:
    """A mirror's laser points lie behind the wall (the reflected path); reflected back
    across the wall plane they land on the room itself. A window's land on nothing."""
    if len(beyond) < 50:
        return False
    if len(beyond) > 3000:
        beyond = beyond[np.random.default_rng(0).choice(len(beyond), 3000, replace=False)]
    signed = (beyond[:, :2] - p) @ outward
    best = 0.0
    # The glass hangs a little proud of the wall face; search the reflecting plane.
    for proud in np.arange(-0.04, 0.0451, 0.005):
        reflected = beyond.copy()
        reflected[:, :2] -= 2 * (signed + proud)[:, None] * outward
        distance, _ = tree.query(reflected, distance_upper_bound=MIRROR_MATCH_M)
        best = max(best, float(np.isfinite(distance).mean()))
    return best >= MIRROR_SHARE


def _reachable(target: np.ndarray, stations: np.ndarray, tree) -> bool:
    for station in stations:
        span = target - station
        distance = float(np.linalg.norm(span))
        steps = np.linspace(0.0, 1.0 - 0.1 / max(distance, 0.2), max(2, int(distance / 0.03)))
        if not np.any(tree.query_ball_point(station + steps[:, None] * span, SHADOW_HIT_M, return_length=True)[3:] > 0):
            return True
    return False


def _shadow_mask(clear: np.ndarray, p: np.ndarray, d: np.ndarray, floor_z: float, stations: np.ndarray, tree) -> np.ndarray:
    """Clear cells the laser could not have reached from any station, because the ray hits
    a surface first. An unscanned patch behind a wardrobe or in a corner is empty in the
    scan exactly like a window pane; only the ray tells them apart. Tested on a 5-cell
    lattice, each sample standing for its block."""
    shadow = np.zeros_like(clear)
    s = SHADOW_STRIDE
    for i in range(s // 2, clear.shape[0], s):
        for j in range(s // 2, clear.shape[1], s):
            if not clear[i, j]:
                continue
            target = np.array([*(p + (i + 0.5) * ALONG_CELL_M * d), floor_z + (j + 0.5) * HEIGHT_CELL_M])
            if not _reachable(target, stations, tree):
                shadow[max(0, i - s // 2): i + s // 2 + 1, max(0, j - s // 2): j + s // 2 + 1] = True
    return shadow


def _split_at_mullions(cols: np.ndarray, rows: np.ndarray, back: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """A window of several panes is one clear region (its mullions are frame, not wall):
    split it at columns that are frame over most of the region's height, so each pane is
    measured on its own, as its own opening."""
    if rows.min() * HEIGHT_CELL_M <= DOOR_SILL_M:
        return [(cols, rows)]
    span = range(rows.min(), rows.max() + 1)
    columns = np.arange(cols.min(), cols.max() + 1)
    share = np.array([back[c, span].mean() for c in columns])
    mullion = share > MULLION_SHARE
    labels, count = ndimage.label(~mullion)
    if count <= 1:
        return [(cols, rows)]
    parts = []
    for piece in range(1, count + 1):
        keep = np.isin(cols, columns[labels == piece])
        if keep.any():
            parts.append((cols[keep], rows[keep]))
    return parts


def survey_openings(lines: list, lengths: list[float], points: np.ndarray, floor_z: float, stations: np.ndarray,
                    tree) -> tuple[list[dict], list]:
    """Openings are regions of a wall's elevation (1 cm along, 2 cm up) where the laser saw
    no wall face and nothing in front of it, and could have (see _shadow_mask): through the
    opening, or no return at all (glass). Door if it reaches the floor, with its width the
    face-to-face run; a window's width is its pane, inside the frame, and a window of
    several panes is several openings. Widths are medians over the region's middle rows,
    a row counting only when both ends are wall face (or, for a pane, frame); if few rows
    are, something stands in front of a jamb and the opening is reported as occluded,
    without a width. A region whose see-through points are a reflection of the room is a
    mirror; one with a back wall close behind is a recess."""
    found, elevations, mirrors = [], [], []
    for k, ((p, d), length) in enumerate(zip(lines, lengths)):
        outward = np.array([d[1], -d[0]])
        rel = points[:, :2] - p
        along, offset = rel @ d, rel @ outward
        near = (along > 0) & (along < length) & (offset > -0.6) & (offset < 3.0)
        z = points[near, 2] - floor_z
        along, offset = along[near], offset[near]
        elevations.append((along, z, offset))
        top = float(np.percentile(z, 99.5)) if len(z) else 0.0
        shape = (int(length / ALONG_CELL_M) + 1, int(top / HEIGHT_CELL_M) + 1)
        cells = (np.minimum((along / ALONG_CELL_M).astype(int), shape[0] - 1),
                 np.clip((z / HEIGHT_CELL_M).astype(int), 0, shape[1] - 1))
        face, front, back = (np.zeros(shape, dtype=bool) for _ in range(3))
        is_face, is_front = np.abs(offset) < FACE_M, offset < -THROUGH_M
        is_back = (offset > THROUGH_M) & (offset < RECESS_DEPTH_M)
        face[cells[0][is_face], cells[1][is_face]] = True
        front[cells[0][is_front], cells[1][is_front]] = True
        back[cells[0][is_back], cells[1][is_back]] = True
        face = ndimage.binary_closing(face, structure=np.ones((3, 3)))
        back = ndimage.binary_closing(back, structure=np.ones((3, 3)))
        # Opening the mask cuts the slivers between pieces of furniture that would
        # otherwise join a window to the floor.
        clear = ndimage.binary_opening(~face & ~front, structure=np.ones((9, 5)))
        clear &= ~_shadow_mask(clear, p, d, floor_z, stations, tree)
        clear = ndimage.binary_opening(clear, structure=np.ones((9, 5)))
        labels, count = ndimage.label(clear)
        parts = []
        for region in range(1, count + 1):
            cols, rows = np.nonzero(labels == region)
            parts += _split_at_mullions(cols, rows, back)
        for cols, rows in parts:
            width_cells, height_cells = cols.max() - cols.min() + 1, rows.max() - rows.min() + 1
            if width_cells * ALONG_CELL_M < MIN_OPENING_M or height_cells * HEIGHT_CELL_M < MIN_OPENING_M:
                continue
            middle = (rows >= rows.min() + height_cells // 4) & (rows <= rows.max() - height_cells // 4)
            middle_fill = middle.sum() / max(width_cells * (height_cells - 2 * (height_cells // 4)), 1)
            if DEBUG:
                print(f"  wall {k} region at {cols.min() * ALONG_CELL_M:.2f}+{width_cells * ALONG_CELL_M:.2f} "
                      f"z {rows.min() * HEIGHT_CELL_M:.2f}+{height_cells * HEIGHT_CELL_M:.2f} fill {middle_fill:.2f} "
                      f"back {np.mean(back[cols, rows]):.2f}")
            # Reaching the top of the scan, or a ragged gap between things: not an opening. Fill
            # is judged on the middle rows, since a plant on a sill only bites a pane's bottom.
            if rows.max() >= shape[1] - 2 or middle_fill < MIN_FILL:
                continue
            if np.mean(back[cols, rows]) > RECESS_COVER:
                continue  # a back wall close behind: a recess or bay, part of the room's shape
            mid = range(rows.min() + height_cells // 4, rows.max() - height_cells // 4 + 1)
            sill = rows.min() * HEIGHT_CELL_M
            door = sill <= DOOR_SILL_M
            widths = []
            for row in mid:
                inside = cols[rows == row]
                lo, hi = inside.min(), inside.max()
                # A pane ends at wall face, or at the frame member it shares with the next.
                ends = face if door else face | back
                if not (lo > 0 and hi < shape[0] - 1 and ends[lo - 1, row] and ends[hi + 1, row]):
                    continue
                if door:
                    widths.append((hi - lo + 1) * ALONG_CELL_M)
                else:
                    # A window's clear width is inside its frame: the pane, where the
                    # laser got nothing back from the frame and reveal depth.
                    pane = ~back[lo:hi + 1, row]
                    runs, n_runs = ndimage.label(pane)
                    if n_runs:
                        widths.append(np.bincount(runs)[1:].max() * ALONG_CELL_M)
            beyond = (offset > THROUGH_M) & (along >= cols.min() * ALONG_CELL_M) & (along <= (cols.max() + 1) * ALONG_CELL_M) & \
                (z >= rows.min() * HEIGHT_CELL_M) & (z <= (rows.max() + 1) * HEIGHT_CELL_M)
            if _mirror(np.column_stack([p + np.outer(along[beyond], d) + np.outer(offset[beyond], outward), z[beyond] + floor_z]),
                       p, outward, tree):
                mirrors.append({"wall": k, "offset": round(float(cols.min() * ALONG_CELL_M), 3),
                                "width": round(float(width_cells * ALONG_CELL_M), 3)})
                continue
            bounded = len(widths) >= BOUNDED_SHARE * len(mid)
            if bounded and np.median(widths) < MIN_OPENING_M:
                continue
            found.append({"type": "door" if door else "window",
                          "width": round(float(np.median(widths)), 3) if bounded else None, "wall": k,
                          "offset": round(float(cols.min() * ALONG_CELL_M), 3), "sill": round(float(sill), 3),
                          "height": round(float(height_cells * HEIGHT_CELL_M), 3), "bounded": bool(bounded)})
    return found, elevations, mirrors


def survey_room(room: dict, yaw: float, shift: np.ndarray, bands: list[np.ndarray], floor_z: float,
                points: np.ndarray, stations: np.ndarray, tree) -> dict:
    corners = _rotate(np.array([w["start"] for w in room["walls"]]), yaw) + shift
    ends = _rotate(np.array([w["end"] for w in room["walls"]]), yaw) + shift
    polygon = _rotate(np.array(room["polygon"]), yaw) + shift
    ccw = 0.5 * np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - np.roll(polygon[:, 0], -1) * polygon[:, 1]) > 0
    slice_xy = np.vstack(bands)
    faces, lines = [], []
    for a, b in zip(corners, ends):
        d = (b - a) / np.linalg.norm(b - a)
        inward = np.array([-d[1], d[0]]) if ccw else np.array([d[1], -d[0]])
        face = wall_face(a, b, -inward, slice_xy)
        faces.append(face)
        along_face = d - inward * face["slope"]
        lines.append((a - inward * (face["offset"] or 0.0), along_face / np.linalg.norm(along_face)))
    n = len(lines)
    truth_corners = []
    for k in range(n):
        (p, d), (q, e) = lines[k - 1], lines[k]
        # Consecutive walls that are parallel (a jog) meet where the plan corner projects onto the wall.
        truth_corners.append(_intersect(p, d, q, e, q + ((corners[k] - q) @ e) * e))
    truth_corners = np.array(truth_corners)
    lengths = [float(np.linalg.norm(truth_corners[(k + 1) % n] - truth_corners[k])) for k in range(n)]
    area = float(abs(0.5 * np.sum(truth_corners[:, 0] * np.roll(truth_corners[:, 1], -1)
                                  - np.roll(truth_corners[:, 0], -1) * truth_corners[:, 1])))
    from matplotlib.path import Path as MplPath
    inside = MplPath(truth_corners).contains_points(points[:, :2])
    # Ceiling height everywhere in the room, 25 cm cells at least 30 cm from the walls:
    # floor and ceiling layers per cell, median of the differences.
    clearance = np.min([np.abs((points[:, :2] - p) @ np.array([d[1], -d[0]])) for p, d in lines], axis=0)
    interior = inside & (clearance > 0.3)
    xy, z = points[interior, :2], points[interior, 2]
    keys = np.floor(xy / 0.25).astype(np.int64)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    keys, z = keys[order], z[order]
    starts = np.r_[0, np.nonzero(np.any(np.diff(keys, axis=0) != 0, axis=1))[0] + 1, len(keys)]
    heights = []
    for i0, i1 in zip(starts[:-1], starts[1:]):
        cell = z[i0:i1]
        low, high = cell[cell < floor_z + 0.05], cell[cell > floor_z + 2.0]
        if len(low) >= 20 and len(high) >= 20:
            top = high[high > np.percentile(high, 50)]
            heights.append(_layer(high, float(np.median(top))) - _layer(low, float(np.median(low))))
    heights = np.array(heights)
    shots = [float(np.median(heights))] if len(heights) else []
    measured = [f for f in faces if f["offset"] is not None]
    edges = [(truth_corners[k], (truth_corners[(k + 1) % n] - truth_corners[k]) / max(lengths[k], 1e-9)) for k in range(n)]
    openings, elevations, mirrors = survey_openings(edges, lengths, points, floor_z, stations, tree)
    return {"id": room["id"], "walls": [round(v, 4) for v in lengths], "floor_area": round(area, 4),
            "ceiling_height": [round(v, 4) for v in shots],
            "openings": [{k: v for k, v in o.items() if k != "bounded"} for o in openings if o["bounded"]],
            "survey": {"wall_offsets_m": [None if f["offset"] is None else round(f["offset"], 4) for f in faces],
                       "strips": [f["strips"] for f in faces],
                       "strip_spread_m": [round(f.get("spread", float("nan")), 4) for f in faces],
                       "walls_measured": f"{len(measured)}/{n}",
                       # A length is laser-measured when its own face and both neighbours'
                       # (which make its corners) were found; otherwise it leans on the plan.
                       "length_measured": [all(faces[j % n]["offset"] is not None for j in (k - 1, k, k + 1))
                                           for k in range(n)],
                       "ceiling_cells": int(len(heights)),
                       "ceiling_p10_p90_m": [round(float(v), 4) for v in np.percentile(heights, [10, 90])] if len(heights) else [],
                       "occluded_openings": [o for o in openings if not o["bounded"]],
                       "mirrors": mirrors},
            "_corners": truth_corners, "_plan": polygon, "_elevations": elevations, "_lengths": lengths}


def elevations(path: Path, room: dict, plan_room: dict) -> None:
    """Every wall as seen by the laser, face-on: grey = wall face, blue = beyond it,
    red = in front. Green boxes are the openings measured, orange the plan's."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    walls = room["_elevations"]
    fig, axes = plt.subplots((len(walls) + 1) // 2, 2, figsize=(16, 3.4 * ((len(walls) + 1) // 2)))
    axes = np.atleast_1d(axes).ravel()
    ids = [w["id"] for w in plan_room["walls"]]
    for k, (ax, (along, z, offset)) in enumerate(zip(axes, walls)):
        pick = slice(None, None, max(1, len(along) // 60000))
        colour = np.where(np.abs(offset) < FACE_M, 0, np.where(offset > THROUGH_M, 1, 2))[pick]
        ax.scatter(along[pick], z[pick], s=0.1, c=np.array(["0.6", "tab:blue", "tab:red"])[colour])
        for o, bounded in [(o, True) for o in room["openings"]] + [(o, False) for o in room["survey"]["occluded_openings"]]:
            if o["wall"] == k:
                ax.add_patch(plt.Rectangle((o["offset"], o["sill"]), o["width"] or 0.05, o["height"],
                                           fill=False, ec="tab:green" if bounded else "k", lw=1.5))
        for o in plan_room["openings"]:
            if o["wall_id"] == ids[k]:
                sill = o.get("sill_height", {}).get("value", 0.0) if o.get("sill_height") else 0.0
                ax.add_patch(plt.Rectangle((o["offset"]["value"], sill), o["width"]["value"], o["height"]["value"],
                                           fill=False, ec="tab:orange", lw=1.2, ls="--"))
        ax.set_xlim(0, room["_lengths"][k])
        ax.set_ylim(0, 2.8)
        ax.set_aspect("equal")
        ax.set_title(f"{room['id']} wall {k}", fontsize=8)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def overlay(path: Path, slice_xy: np.ndarray, rooms: list[dict], title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    span = np.vstack([r["_corners"] for r in rooms])
    lo, hi = span.min(axis=0) - 1.0, span.max(axis=0) + 1.0
    keep = np.all((slice_xy > lo) & (slice_xy < hi), axis=1)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(*slice_xy[keep][:: max(1, keep.sum() // 300000)].T, s=0.05, c="0.45")
    for r in rooms:
        ax.plot(*np.vstack([r["_plan"], r["_plan"][:1]]).T, c="tab:red", lw=1.2, label="plan")
        ax.plot(*np.vstack([r["_corners"], r["_corners"][:1]]).T, c="tab:green", lw=1.2, ls="--", label="laser faces")
        ax.text(*r["_corners"].mean(axis=0), r["id"], ha="center")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc="upper right")
    ax.set_aspect("equal")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_title(title)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    parser.add_argument("laser", help="folder of the visit's laser scans")
    parser.add_argument("--out", required=True)
    parser.add_argument("--overlay")
    parser.add_argument("--rooms", nargs="*", help="plan rooms to survey (default: all)")
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    from scipy.spatial import cKDTree
    points, stations, floor_z, ceiling_z = level(*load_scans(Path(args.laser), args.stride))
    tree = cKDTree(points[::4])
    bands = [_band(points, floor_z + TAPE_BAND_M[0], floor_z + TAPE_BAND_M[1]),
             _band(points, ceiling_z - UPPER_BAND_M[0], ceiling_z - UPPER_BAND_M[1])]
    yaw, shift, fit = place(plan, bands[0])
    print(f"placed plan: yaw {np.degrees(yaw):.2f} deg, {fit:.0%} of plan wall samples within 5 cm of the scan")
    rooms = [survey_room(r, yaw, shift, bands, floor_z, points, stations, tree) for r in plan["rooms"]
             if not args.rooms or r["id"] in args.rooms]
    for r in rooms:
        print(f"{r['id']}: walls {r['walls']} area {r['floor_area']} ceiling {r['ceiling_height']} "
              f"offsets {r['survey']['wall_offsets_m']}")
        print(f"  openings {r['openings']}  occluded {r['survey']['occluded_openings']}")
    if args.overlay:
        Path(args.overlay).parent.mkdir(parents=True, exist_ok=True)
        for r in rooms:
            plan_room = next(p for p in plan["rooms"] if p["id"] == r["id"])
            elevations(Path(args.overlay).with_name(Path(args.overlay).stem + f"_{r['id']}_walls.png"), r, plan_room)
        overlay(Path(args.overlay), bands[0], rooms, f"{plan['capture']['id']} on laser scan {Path(args.laser).name}")
    truth = {"property": f"arkitscenes visit {Path(args.laser).name}",
             "source": f"laser scans in {args.laser}, plan placed with {fit:.0%} of wall samples within 5 cm",
             "rooms": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rooms], "adjacency": []}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(yaml.safe_dump(truth, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
