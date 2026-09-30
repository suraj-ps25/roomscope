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


@dataclass
class RoomDamage:
    detections: dict[str, list[Detection]] = field(default_factory=dict)
    flags: list[Flag] = field(default_factory=list)
    ids: dict[str, list[str]] = field(default_factory=dict)


def _best_views(frames: list[Frame], poses: dict[int, np.ndarray], centre: np.ndarray, normal: np.ndarray) -> list[Frame]:
    scored = []
    for frame in frames:
        pose = poses[frame.index]
        to_surface = centre - pose[:3, 3]
        distance = float(np.linalg.norm(to_surface))
        facing = float(pose[:3, 2] @ (to_surface / max(distance, 1e-9)))
        frontal = float(-(to_surface / max(distance, 1e-9)) @ normal)
        if facing > 0.3 and frontal > 0.2 and distance < 5.0:
            scored.append((facing * frontal / max(distance, 0.5), frame))
    scored.sort(key=lambda s: -s[0])
    return [frame for _, frame in scored[:VIEWS_PER_SURFACE]]


def analyse_room(layout: RoomLayout, openings: list, mirrors: list, frames: list[Frame], poses: dict[int, np.ndarray],
                 label: str, neighbours_on_wall: dict[int, list[str]], warm_rgb=None) -> RoomDamage:
    polygon = layout.polygon
    n = len(polygon)
    grids = []
    for k, line in enumerate(layout.lines):
        grids.append((f"w{k}", k, wall_grid(f"w{k}", polygon[k], polygon[(k + 1) % n], line.normal,
                                           layout.floor_z, layout.ceiling_z)))
    grids.append(("floor", None, level_grid("floor", "floor", polygon, layout.floor_z + 0.002)))
    grids.append(("ceiling", None, level_grid("ceiling", "ceiling", polygon, layout.ceiling_z - 0.002)))

    extents = [OpeningExtent(f"o{i}", o.kind, o.wall, o.u0, o.u1, o.v0 - layout.floor_z, o.v1 - layout.floor_z)
               for i, o in enumerate(openings)]
    masks = {k: [] for k in range(n)}
    for o in [*openings, *mirrors]:
        masks[o.wall].append((o.u0, o.u1, o.v0 - layout.floor_z, o.v1 - layout.floor_z))

    result = RoomDamage()
    surfaces = []
    for key, wall_index, grid in grids:
        centre = grid.origin + grid.u_axis * grid.width_m / 2 + grid.v_axis * grid.height_m / 2
        views = _best_views(frames, poses, centre, grid.normal)
        if not views:
            continue
        if warm_rgb is not None:
            warm_rgb(views)
        ortho = render(grid, views, poses, ORTHO_STEP)
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
        ids = [f"{key}/d{i}" for i in range(len(found))]
        result.detections[key] = found
        result.ids[key] = ids
        surfaces.append(SurfaceDamage(key, grid.kind, wall_index, found, ids))
    result.flags = evaluate_rules(label, layout.floor_z, surfaces, extents, neighbours_on_wall)
    return result


def _inside(detection: Detection, polygon: np.ndarray, grid) -> bool:
    from matplotlib.path import Path
    centre = detection.polygon_uv.mean(axis=0)
    point = grid.origin[:2] + grid.u_axis[:2] * centre[0] + grid.v_axis[:2] * centre[1]
    return bool(Path(polygon).contains_point(point))
