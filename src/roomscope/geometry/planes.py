"""Plane-anchored drift correction (a small bundle adjustment with planes as landmarks).

The pose graph only aligns fragments pairwise. What makes a room's dimensions right is
that every view of a wall agrees where that wall is, including the far-away view from the
opposite side of the room. Here the landmarks are the rooms' fitted wall planes plus each
room's floor and ceiling; the unknowns are a 4-DoF correction per fragment and every
plane's parameters. Residuals are point-to-plane distances of each fragment's points on
the planes they belong to, plus the same VIO odometry priors the pose graph uses.

Points from one fragment on one plane share correlated depth error, so each such block
counts as PLANE_BLOCK_EFFECTIVE points regardless of how many points it has.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

from ..capture import Frame
from .cloud import Cloud, frame_points, voxelize
from .drift import (FRAGMENT_SECONDS, VIO_SIGMA_FLOOR_M, VIO_SIGMA_M_PER_M, VIO_YAW_DEG_PER_SQRT_M,
                    VIO_YAW_FLOOR_DEG, Pose4, _fragments_of, yaw_matrix)
from .layout import RoomLayout

POINT_SIGMA_M = 0.01
PLANE_BLOCK_EFFECTIVE = 50
POINTS_PER_BLOCK = 250
ASSOCIATE_M = (0.06, 0.03)
EXTENT_MARGIN_M = 0.2


@dataclass
class Plane:
    kind: str          # "wall" (also door/window jambs) or "level"
    room: str
    angle: float       # wall: normal direction in the plan; level: unused
    offset: float      # wall: normal . xy; level: z
    up: float          # level: +1 floor (faces up), -1 ceiling
    start: np.ndarray | None = None
    end: np.ndarray | None = None
    mask_room: object = None
    z_range: tuple[float, float] | None = None
    margin: float = EXTENT_MARGIN_M


JAMB_DEPTH_M = 0.30


def planes_from_layouts(layouts: list[RoomLayout], regions: dict, openings: list | None = None) -> list[Plane]:
    """Walls, floors and ceilings of every room, plus the jambs of detected openings.
    Jambs are perpendicular to their wall, so they pin the along-wall sliding a wall
    plane cannot (the weak direction of a long hallway)."""
    planes = []
    by_room = {layout.id: layout for layout in layouts}
    for opening in openings or []:
        layout = by_room.get(opening.room)
        if layout is None or opening.jambs_found < 2:
            continue
        line, start = layout.lines[opening.wall], layout.polygon[opening.wall]
        behind = -line.normal * JAMB_DEPTH_M
        for u, facing in ((opening.u0, line.direction), (opening.u1, -line.direction)):
            face_point = start + line.direction * u
            planes.append(Plane("wall", opening.room, float(np.arctan2(facing[1], facing[0])),
                                float(facing @ face_point), 0.0, face_point, face_point + behind,
                                z_range=(opening.v0 + 0.1, opening.v1 - 0.1), margin=0.01))
    for layout in layouts:
        n = len(layout.polygon)
        for k, line in enumerate(layout.lines):
            planes.append(Plane("wall", layout.id, float(np.arctan2(line.normal[1], line.normal[0])),
                                line.offset, 0.0, layout.polygon[k], layout.polygon[(k + 1) % n]))
        planes.append(Plane("level", layout.id, 0.0, layout.floor_z, 1.0, mask_room=regions[layout.id]))
        planes.append(Plane("level", layout.id, 0.0, layout.ceiling_z, -1.0, mask_room=regions[layout.id]))
    return planes


def _associate(cloud: Cloud, planes: list[Plane], layouts: dict[str, RoomLayout], tolerance: float,
               rng: np.random.Generator) -> list[tuple[int, np.ndarray]]:
    """(plane index, point row indices) blocks for one fragment."""
    blocks = []
    xy, z = cloud.points[:, :2], cloud.points[:, 2]
    for index, plane in enumerate(planes):
        layout = layouts[plane.room]
        if plane.kind == "wall":
            normal = np.array([np.cos(plane.angle), np.sin(plane.angle)])
            direction = np.array([normal[1], -normal[0]])
            along = (xy - plane.start) @ direction
            length = float((plane.end - plane.start) @ direction)
            z_lo, z_hi = plane.z_range or (layout.floor_z + 0.1, layout.ceiling_z - 0.05)
            rows = (cloud.normals[:, :2] @ normal > 0.9) & (np.abs(xy @ normal - plane.offset) < tolerance) & \
                (along > plane.margin) & (along < length - plane.margin) & (z > z_lo) & (z < z_hi)
        else:
            rows = (cloud.normals[:, 2] * plane.up > 0.9) & (np.abs(z - plane.offset) < tolerance)
            rows &= plane.mask_room.contains(xy)
        picked = np.nonzero(rows)[0]
        if len(picked) < (8 if plane.margin < EXTENT_MARGIN_M else 20):
            continue
        if len(picked) > POINTS_PER_BLOCK:
            picked = rng.choice(picked, POINTS_PER_BLOCK, replace=False)
        blocks.append((index, picked))
    return blocks


def _odometry_sigmas(distance: float, scale: float = 1.0) -> tuple[float, float]:
    sigma_t = (VIO_SIGMA_FLOOR_M + VIO_SIGMA_M_PER_M * distance) * scale
    sigma_yaw = np.deg2rad(VIO_YAW_FLOOR_DEG + VIO_YAW_DEG_PER_SQRT_M * np.sqrt(distance)) * scale
    return sigma_t, sigma_yaw


@dataclass(frozen=True)
class PlaneSettings:
    associate_m: tuple[float, float] = ASSOCIATE_M
    point_sigma_m: float = POINT_SIGMA_M
    vio_scale: float = 1.0
    min_confidence: int = 1


LIDAR_PLANES = PlaneSettings()
VIDEO_PLANES = PlaneSettings(associate_m=(0.15, 0.08), point_sigma_m=0.04, vio_scale=3.0)


def plane_adjust(frames: list[Frame], poses: dict[int, np.ndarray], layouts: list[RoomLayout],
                 regions: dict, openings: list | None = None,
                 settings: PlaneSettings = LIDAR_PLANES) -> tuple[dict[int, np.ndarray], list[str]]:
    """poses and layouts in the plan frame. Returns corrected plan-frame poses."""
    groups = _fragments_of(frames, FRAGMENT_SECONDS)
    clouds, centres, mids = [], [], []
    for group in groups:
        clouds.append(voxelize(Cloud.concat([frame_points(f, poses[f.index], min_confidence=settings.min_confidence,
                                                          stride=2, max_depth=5.0)
                                             for f in group]), 0.02))
        centres.append(np.mean([poses[f.index][:3, 3] for f in group], axis=0))
        mids.append(float(np.mean([f.timestamp for f in group])))
    centres = np.array(centres)
    planes = planes_from_layouts(layouts, regions, openings)
    by_room = {layout.id: layout for layout in layouts}
    n_frag, n_planes = len(groups), len(planes)
    rng = np.random.default_rng(0)

    frag_params = np.zeros((n_frag, 4))   # yaw, tx, ty, tz about the fragment centre
    plane_params = np.array([[p.angle, p.offset] for p in planes])

    for tolerance in settings.associate_m:
        # Associate in the current corrected frame.
        rows_frag, rows_plane, rows_points = [], [], []
        for f, cloud in enumerate(clouds):
            moved = _move(cloud, frag_params[f], centres[f])
            for plane_index, picked in _associate(moved, planes, by_room, tolerance, rng):
                rows_frag.append(np.full(len(picked), f))
                rows_plane.append(np.full(len(picked), plane_index))
                rows_points.append(cloud.points[picked])
        if not rows_points:
            return dict(poses), ["plane adjustment: no plane associations; skipped"]
        point_frag = np.concatenate(rows_frag)
        point_plane = np.concatenate(rows_plane)
        points = np.concatenate(rows_points)
        block_size = np.array([len(r) for r in rows_frag for _ in range(len(r))])
        point_weight = np.sqrt(np.minimum(1.0, PLANE_BLOCK_EFFECTIVE / block_size)) / settings.point_sigma_m
        is_wall = np.array([planes[k].kind == "wall" for k in point_plane])

        odo_pairs = [(f, f + 1) for f in range(n_frag - 1)]
        odo_sigmas = [_odometry_sigmas(float(np.linalg.norm(centres[b] - centres[a])), settings.vio_scale)
                      for a, b in odo_pairs]

        def unpack(x):
            frag = np.vstack([np.zeros(4), x[:4 * (n_frag - 1)].reshape(-1, 4)])
            plane = x[4 * (n_frag - 1):].reshape(-1, 2)
            return frag, plane

        def residuals_and_jacobian(x, want_jac):
            frag, plane = unpack(x)
            yaw = frag[point_frag, 0]
            c, s = np.cos(yaw), np.sin(yaw)
            rel = points - centres[point_frag]
            rx = c * rel[:, 0] - s * rel[:, 1]
            ry = s * rel[:, 0] + c * rel[:, 1]
            moved = np.stack([centres[point_frag, 0] + rx + frag[point_frag, 1],
                              centres[point_frag, 1] + ry + frag[point_frag, 2],
                              points[:, 2] + frag[point_frag, 3]], axis=1)
            angle = plane[point_plane, 0]
            nx, ny = np.cos(angle), np.sin(angle)
            wall_res = nx * moved[:, 0] + ny * moved[:, 1] - plane[point_plane, 1]
            level_res = moved[:, 2] - plane[point_plane, 1]
            point_res = np.where(is_wall, wall_res, level_res) * point_weight

            odo_res = []
            for (a, b), (sigma_t, sigma_yaw) in zip(odo_pairs, odo_sigmas):
                ta = frag[a, 1:] + frag[a, 0] * np.array([centres[a, 1], -centres[a, 0], 0.0])
                tb = frag[b, 1:] + frag[b, 0] * np.array([centres[b, 1], -centres[b, 0], 0.0])
                odo_res.append(np.concatenate([[(frag[a, 0] - frag[b, 0]) / sigma_yaw],
                                               (ta[:2] - tb[:2]) / sigma_t, [(ta[2] - tb[2]) / (0.5 * sigma_t)]]))
            residual = np.concatenate([point_res, np.concatenate(odo_res)])
            if not want_jac:
                return residual

            n_points = len(points)
            n_vars = 4 * (n_frag - 1) + 2 * n_planes
            r_idx, c_idx, vals = [], [], []
            has_frag = point_frag > 0
            base = 4 * (point_frag - 1)
            d_yaw = np.where(is_wall, nx * (-ry) + ny * rx, 0.0)
            grads = [d_yaw, np.where(is_wall, nx, 0.0), np.where(is_wall, ny, 0.0), np.where(is_wall, 0.0, 1.0)]
            rows = np.arange(n_points)
            for k, g in enumerate(grads):
                r_idx.append(rows[has_frag]); c_idx.append(base[has_frag] + k); vals.append((g * point_weight)[has_frag])
            plane_base = 4 * (n_frag - 1) + 2 * point_plane
            d_angle = np.where(is_wall, -np.sin(angle) * moved[:, 0] + np.cos(angle) * moved[:, 1], 0.0)
            r_idx += [rows, rows]; c_idx += [plane_base, plane_base + 1]
            vals += [d_angle * point_weight, -point_weight]

            row = n_points
            for (a, b), (sigma_t, sigma_yaw) in zip(odo_pairs, odo_sigmas):
                lever = {a: np.array([centres[a, 1], -centres[a, 0], 0.0]), b: np.array([centres[b, 1], -centres[b, 0], 0.0])}
                scales = [sigma_t, sigma_t, 0.5 * sigma_t]
                for node, sign in ((a, 1.0), (b, -1.0)):
                    if node == 0:
                        continue
                    col = 4 * (node - 1)
                    r_idx.append(np.array([row])); c_idx.append(np.array([col])); vals.append(np.array([sign / sigma_yaw]))
                    for axis in range(3):
                        r_idx.append(np.array([row + 1 + axis, row + 1 + axis]))
                        c_idx.append(np.array([col + 1 + axis, col]))
                        vals.append(np.array([sign / scales[axis], sign * lever[node][axis] / scales[axis]]))
                row += 4
            jac = coo_matrix((np.concatenate(vals), (np.concatenate(r_idx), np.concatenate(c_idx))),
                             shape=(len(residual), n_vars)).tocsr()
            return residual, jac

        x0 = np.concatenate([frag_params[1:].ravel(), plane_params.ravel()])
        result = least_squares(lambda x: residuals_and_jacobian(x, False), x0,
                               jac=lambda x: residuals_and_jacobian(x, True)[1],
                               loss="huber", f_scale=2.0, method="trf", tr_solver="lsmr", x_scale="jac",
                               max_nfev=60)
        frag_params, plane_params = unpack(result.x)
        frag_params = frag_params.copy()

    corrected = _interpolate(frames, poses, frag_params, centres, np.array(mids))
    shifts = np.linalg.norm(frag_params[:, 1:3], axis=1)
    notes = [f"plane adjustment: {n_frag} fragments x {n_planes} planes, {len(points)} point residuals; "
             f"median fragment shift {np.median(shifts) * 100:.1f} cm, max {shifts.max() * 100:.1f} cm"]
    return corrected, notes


def _move(cloud: Cloud, params: np.ndarray, centre: np.ndarray) -> Cloud:
    rot = yaw_matrix(params[0])
    points = centre + (cloud.points - centre) @ rot.T + params[1:]
    return Cloud(points, cloud.normals @ rot.T, cloud.frame_ids)


def _interpolate(frames: list[Frame], poses: dict[int, np.ndarray], params: np.ndarray, centres: np.ndarray,
                 mids: np.ndarray) -> dict[int, np.ndarray]:
    # Fragment corrections are about fragment centres; re-express about the origin, then
    # blend between fragment mid-times.
    corrections = [Pose4(p[0], c - yaw_matrix(p[0]) @ c + p[1:]) for p, c in zip(params, centres)]
    yaws = np.unwrap([c.yaw for c in corrections])
    shifts = np.array([c.t for c in corrections])
    out = {}
    for frame in frames:
        t = frame.timestamp
        node = Pose4(float(np.interp(t, mids, yaws)), np.array([np.interp(t, mids, shifts[:, k]) for k in range(3)]))
        out[frame.index] = node.matrix() @ poses[frame.index]
    return out
