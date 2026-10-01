"""Per-room damage analysis: orthophotos -> detections -> concealed-damage rules."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..capture import Frame
from ..geometry.layout import RoomLayout
from .detect import Detection, detect
from .ortho import level_grid, render, wall_grid
from .rules import Flag, OpeningExtent, SurfaceDamage, evaluate_rules

VIEWS_PER_SURFACE = 40
ORTHO_STEP = 1
# Reporting thresholds, set from undamaged scanned rooms (every detection there is false)
# and the staged synthetic damage (true detections score 0.74-1.0):
REPORT_MIN_CONFIDENCE = 0.6     # crack confidence is length / 0.3 m: nothing under 18 cm
CRACK_MAX_M = 2.5               # longer "cracks" are structure: blind slats, frames, skirting
CRACK_MIN_M = 0.25              # shorter ones are too easily the edge of something in front


@dataclass
class RoomDamage:
    detections: dict[str, list[Detection]] = field(default_factory=dict)
    flags: list[Flag] = field(default_factory=list)
    ids: dict[str, list[str]] = field(default_factory=dict)


COVERAGE_STEP_M = 0.15


def _best_views(frames: list[Frame], poses: dict[int, np.ndarray], grid) -> list[Frame]:
    """Views that together see the whole surface well, picked greedily: each pick is the
    frame that most improves the best viewing quality (frontal, close) over a 15 cm
    lattice on the surface. Ranking views by how well they see the surface's centre
    leaves its corners and the strip above a door to whatever the top views happen to
    catch; a crack running up from a door head is exactly there."""
    from .ortho import MAX_VIEW_M, MIN_COSINE
    u = np.arange(COVERAGE_STEP_M / 2, grid.width_m, COVERAGE_STEP_M)
    v = np.arange(COVERAGE_STEP_M / 2, grid.height_m, COVERAGE_STEP_M)
    uu, vv = np.meshgrid(u, v)
    lattice = (grid.origin + uu.reshape(-1, 1) * grid.u_axis + vv.reshape(-1, 1) * grid.v_axis)
    qualities, candidates = [], []
    for frame in frames:
        pose = poses[frame.index]
        to_point = lattice - pose[:3, 3]
        distance = np.linalg.norm(to_point, axis=1)
        cosine = -(to_point @ grid.normal) / np.maximum(distance, 1e-9)
        cam = to_point @ pose[:3, :3]
        width, height = frame.image_size
        z = np.maximum(cam[:, 2], 1e-6)
        px = frame.K[0, 0] * cam[:, 0] / z + frame.K[0, 2]
        py = frame.K[1, 1] * cam[:, 1] / z + frame.K[1, 2]
        seen = (cam[:, 2] > 0.2) & (px >= 0) & (py >= 0) & (px < width) & (py < height) & \
            (cosine > MIN_COSINE) & (distance < MAX_VIEW_M)
        if seen.any():
            qualities.append(np.where(seen, cosine / np.maximum(distance, 0.3), 0.0))
            candidates.append(frame)
    if not candidates:
        return []
    quality = np.array(qualities)
    best = np.zeros(quality.shape[1])
    picked = []
    for _ in range(min(VIEWS_PER_SURFACE, len(candidates))):
        gain = np.maximum(quality - best, 0).sum(axis=1)
        gain[picked] = -1
        pick = int(np.argmax(gain))
        if gain[pick] <= 0:
            break
        picked.append(pick)
        best = np.maximum(best, quality[pick])
    return [candidates[k] for k in picked]


ALL_CLASSES = ("water_stain", "mold", "crack")
# Video frames (720 px across, compressed) can't tell a 1-3 mm crack from a frame or cabinet
# edge: on undamaged scanned rooms they did, by the dozen. Stains and mould are larger.
VIDEO_CLASSES = ("water_stain", "mold")


def analyse_room(layout: RoomLayout, openings: list, mirrors: list, frames: list[Frame], poses: dict[int, np.ndarray],
                 label: str, neighbours_on_wall: dict[int, list[str]], warm_rgb=None,
                 classes: tuple[str, ...] = ALL_CLASSES) -> RoomDamage:
    polygon = layout.polygon
    n = len(polygon)
    grids = []
    for k, line in enumerate(layout.lines):
        grids.append((f"w{k}", k, wall_grid(f"w{k}", polygon[k], polygon[(k + 1) % n], line.normal,
                                           layout.floor_z, layout.ceiling_z)))
    # Floors are not analysed: rugs, wood grain, furniture shadows and grazing views make
    # every detector there unreliable (on the synthetic flat it reported 0.9 m2 "stains").
    grids.append(("ceiling", None, level_grid("ceiling", "ceiling", polygon, layout.ceiling_z - 0.002)))

    extents = [OpeningExtent(f"o{i}", o.kind, o.wall, o.u0, o.u1, o.v0 - layout.floor_z, o.v1 - layout.floor_z)
               for i, o in enumerate(openings)]
    masks = {k: [] for k in range(n)}
    for o in [*openings, *mirrors]:
        masks[o.wall].append((o.u0, o.u1, o.v0 - layout.floor_z, o.v1 - layout.floor_z))

    result = RoomDamage()
    surfaces = []
    for key, wall_index, grid in grids:
        views = _best_views(frames, poses, grid)
        if not views:
            continue
        if warm_rgb is not None:
            warm_rgb(views)
        ortho = render(grid, views, poses, ORTHO_STEP)
        ortho.valid &= ~_in_front(grid, layout.points.points, ortho.valid.shape)
        exclude = masks.get(wall_index, []) if wall_index is not None else []
        if key != "floor" and key != "ceiling":
            found = detect(ortho, exclude, ORTHO_STEP)
        else:
            # Floor/ceiling detections are reported in plan x/y (schema), not grid-local uv.
            found = [d for d in detect(ortho, [], ORTHO_STEP, textured=(key == "floor")) if _inside(d, polygon, grid)]
            for d in found:
                d.polygon_uv = d.polygon_uv + grid.origin[:2]
                u0, u1, v0, v1 = d.bbox_uv
                d.bbox_uv = (u0 + grid.origin[0], u1 + grid.origin[0], v0 + grid.origin[1], v1 + grid.origin[1])
        found = [d for d in found if d.cls in classes and d.confidence >= REPORT_MIN_CONFIDENCE
                 and not (d.cls == "crack" and not CRACK_MIN_M <= (d.length_m or 0) <= CRACK_MAX_M)]
        ids = [f"{key}/d{i}" for i in range(len(found))]
        result.detections[key] = found
        result.ids[key] = ids
        surfaces.append(SurfaceDamage(key, grid.kind, wall_index, found, ids))
    result.flags = evaluate_rules(label, layout.floor_z, surfaces, extents, neighbours_on_wall)
    return result


IN_FRONT_MIN_M = 0.03
IN_FRONT_MAX_M = 1.0
IN_FRONT_GROW_M = 0.03
IN_FRONT_CELL_M = 0.02
IN_FRONT_MIN_CELLS = 12
IN_FRONT_RADIUS_M = 0.04
IN_FRONT_NEIGHBOURS = 8


def _in_front(grid, points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Texels with geometry standing in front of the surface: shelves, toys, plants,
    furniture against a wall, a lamp under a ceiling. Their colours are not the surface's,
    and the depth-based occlusion test lets them leak in at low depth resolution (on a real
    bedroom, 54 'cracks' were shelf and toy edges). Flush decor stays for the colour test."""
    from scipy import ndimage

    from .ortho import TEXEL_M
    offset = points - grid.origin
    ahead = offset @ grid.normal
    u, v = offset @ grid.u_axis, offset @ grid.v_axis
    on_grid = (u >= 0) & (u < grid.width_m) & (v >= 0) & (v < grid.height_m)
    near = (ahead > IN_FRONT_MIN_M) & (ahead < IN_FRONT_MAX_M) & on_grid
    # A recess (a shelf niche, a window reveal) has no wall face at all; what the
    # orthophoto shows there is whatever stands inside it.
    behind = (ahead < -IN_FRONT_MIN_M) & (ahead > -IN_FRONT_MAX_M) & on_grid
    face = (np.abs(ahead) <= IN_FRONT_MIN_M) & on_grid
    step = grid.height_m / shape[0] if shape[0] else TEXEL_M
    # Objects are surfaces: dense in 3-D. Depth edges smear "flying" points through the
    # space in front of every wall; they are sparse there, and masking each one (or
    # closing the gaps between them) would cover the whole wall.
    from scipy.spatial import cKDTree
    cells = (int(np.ceil(grid.height_m / IN_FRONT_CELL_M)) + 1, int(np.ceil(grid.width_m / IN_FRONT_CELL_M)) + 1)

    def dense_cells(select: np.ndarray) -> np.ndarray:
        found = np.stack([u[select], v[select], ahead[select]], axis=1)
        grid_cells = np.zeros(cells, dtype=bool)
        if len(found) >= IN_FRONT_MIN_CELLS:
            dense = cKDTree(found).query_ball_point(found, IN_FRONT_RADIUS_M, return_length=True) >= IN_FRONT_NEIGHBOURS
            grid_cells[(found[dense, 1] / IN_FRONT_CELL_M).astype(int), (found[dense, 0] / IN_FRONT_CELL_M).astype(int)] = True
        return grid_cells

    wall_face = ndimage.binary_dilation(dense_cells(face), iterations=1)
    occupied = dense_cells(near) | (dense_cells(behind) & ~wall_face)
    labels, _ = ndimage.label(occupied, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())
    solid = np.isin(labels, np.nonzero(sizes >= IN_FRONT_MIN_CELLS)[0]) & (labels > 0)
    rows = np.minimum((np.arange(shape[0]) * step / IN_FRONT_CELL_M).astype(int), cells[0] - 1)
    cols = np.minimum((np.arange(shape[1]) * step / IN_FRONT_CELL_M).astype(int), cells[1] - 1)
    mask = solid[np.ix_(rows, cols)]
    grow = max(1, int(round(IN_FRONT_GROW_M / step)))
    return ndimage.binary_dilation(mask, iterations=grow)


def _inside(detection: Detection, polygon: np.ndarray, grid) -> bool:
    from matplotlib.path import Path
    centre = detection.polygon_uv.mean(axis=0)
    point = grid.origin[:2] + grid.u_axis[:2] * centre[0] + grid.v_axis[:2] * centre[1]
    return bool(Path(polygon).contains_point(point))
