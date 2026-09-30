"""Camera paths through a synthetic scene.

walkthrough(): continuous handheld scan (video / LiDAR tiers) that visits rooms through
their doors and ends where it started, as the capture protocol asks. apply_drift() turns
the true path into what visual-inertial odometry would report.

photo_views(): the per-room stills the photo-tier protocol prescribes (corner-to-corner
shots plus one shot facing each doorway).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .scene import OpeningSpec, RoomSpec, SceneSpec


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    eye, target, up = (np.asarray(a, dtype=float) for a in (eye, target, up))
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, down, forward], axis=1)
    pose[:3, 3] = eye
    return pose


def pose_from_yaw_pitch(eye: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    direction = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    return look_at(eye, eye + direction)


def inset_polygon(polygon: np.ndarray, distance: float) -> np.ndarray:
    """Offset a CCW polygon inward by intersecting each edge's inward-shifted line."""
    n = len(polygon)
    lines = []
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        d = (b - a) / np.linalg.norm(b - a)
        inward = np.array([-d[1], d[0]])
        lines.append((a + inward * distance, d))
    corners = []
    for i in range(n):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denom) < 1e-9:
            corners.append(p2)
            continue
        t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / denom
        corners.append(p1 + d1 * t)
    return np.asarray(corners)


def standable(scene: SceneSpec, room: RoomSpec, xy: np.ndarray, margin: float = 0.3) -> np.ndarray:
    """Nudge a standing position toward the room centre until it is clear of furniture a
    person can't stand in or lean over (anything taller than ~0.9 m)."""
    centre = room.polygon.mean(axis=0)
    point = np.asarray(xy, dtype=float)
    for _ in range(40):
        blocked = False
        for box in scene.furniture:
            if box.room != room.id or box.size[2] < 0.9:
                continue
            c, s = np.cos(-box.yaw), np.sin(-box.yaw)
            local = np.array([[c, -s], [s, c]]) @ (point - box.center)
            if np.all(np.abs(local) < box.size[:2] / 2 + margin):
                blocked = True
                break
        if not blocked:
            return point
        point = point + (centre - point) * 0.1
    return point


def _door_points(scene: SceneSpec, opening: OpeningSpec, standoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Points `standoff` metres either side of a door's centre (inside own room, inside the other)."""
    room = scene.room(opening.room)
    start, direction, _, inward = room.wall_frame(opening.wall)
    centre = start + direction * (opening.offset + opening.width / 2)
    return centre + inward * standoff, centre - inward * (scene.wall_thickness + standoff)


def _door_between(scene: SceneSpec, room_a: str, room_b: str) -> OpeningSpec:
    return next(o for o in scene.openings if o.room == room_a and o.connects_to == room_b)


def default_route(scene: SceneSpec) -> list[str]:
    """Depth-first tour of the door graph from the first room, walking back through each
    door, so the scan ends where it started (the protocol's loop-closure requirement)."""
    neighbours: dict[str, list[str]] = {r.id: [] for r in scene.rooms}
    for opening in scene.openings:
        if opening.connects_to and opening.connects_to not in neighbours[opening.room]:
            neighbours[opening.room].append(opening.connects_to)
    route: list[str] = []
    seen: set[str] = set()

    def visit(room_id: str) -> None:
        seen.add(room_id)
        route.append(room_id)
        for other in neighbours[room_id]:
            if other not in seen:
                visit(other)
                route.append(room_id)

    visit(scene.rooms[0].id)
    return route


@dataclass
class Trajectory:
    timestamps: np.ndarray
    poses: np.ndarray
    room_of_frame: list[str]


SPIN_SECONDS = 9.0


def walkthrough(scene: SceneSpec, route: list[str], fps: float = 6.0, speed: float = 0.35,
                eye_height: float = 1.45, seed: int = 0) -> Trajectory:
    """route: rooms in visiting order, e.g. [hallway, living, hallway, bedroom, hallway].
    Follows the capture protocol: on entering a room, walk to its middle and turn slowly
    through a full circle (sees opposite walls seconds apart, so room dimensions do not
    rest on long-term odometry), then walk the perimeter facing the walls. Transitions go
    through the connecting door."""
    rng = np.random.default_rng(seed)
    waypoints: list[tuple[np.ndarray, str, str]] = []  # (xy, room, mode)

    def add_loop(room: RoomSpec, entry: np.ndarray) -> np.ndarray:
        dims = room.polygon.max(axis=0) - room.polygon.min(axis=0)
        middle = standable(scene, room, room.polygon.mean(axis=0))
        waypoints.append((middle, room.id, "walk"))
        waypoints.append((middle, room.id, "spin"))
        inset = min(0.9, 0.3 * float(dims.min()))
        loop = inset_polygon(room.polygon, inset)
        start = int(np.argmin(np.linalg.norm(loop - middle, axis=1)))
        ordered = np.vstack([loop[start:], loop[:start], loop[start:start + 1]])
        for point in ordered:
            waypoints.append((standable(scene, room, point), room.id, "scan"))
        return ordered[-1]

    current = scene.room(route[0]).polygon.mean(axis=0)
    visited = set()
    for index, room_id in enumerate(route):
        room = scene.room(room_id)
        if room_id not in visited:
            current = add_loop(room, current)
            visited.add(room_id)
        if index + 1 < len(route):
            door = _door_between(scene, room_id, route[index + 1])
            near, far = _door_points(scene, door, 0.6)
            waypoints += [(near, room_id, "walk"), (far, route[index + 1], "walk")]
            current = far
    waypoints.append((waypoints[0][0], route[-1], "walk"))

    positions, rooms, modes, headings, spins = [], [], [], [], []
    step = speed / fps
    for (a, room_a, mode_a), (b, _, _) in zip(waypoints[:-1], waypoints[1:]):
        length = float(np.linalg.norm(b - a))
        tangent = (b - a) / max(length, 1e-9)
        if mode_a == "spin":
            turns = int(SPIN_SECONDS * fps)
            for k in range(turns):
                positions.append(a.copy())
                rooms.append(room_a)
                modes.append("spin")
                headings.append(tangent)
                spins.append(2 * np.pi * k / turns)
            mode_a = "walk"
        count = max(1, int(np.ceil(length / step)))
        for k in range(count):
            positions.append(a + (b - a) * (k / count))
            rooms.append(room_a)
            modes.append(mode_a)
            headings.append(tangent)
            spins.append(0.0)

    poses = []
    travelled = 0.0
    for i, (xy, mode, tangent, spin) in enumerate(zip(positions, modes, headings, spins)):
        if i:
            travelled += float(np.linalg.norm(xy - positions[i - 1]))
        bob = 0.015 * np.sin(travelled * 2 * np.pi / 0.7)
        eye = np.array([xy[0], xy[1], eye_height + bob]) + rng.normal(0, 0.004, 3)
        base_yaw = np.arctan2(tangent[1], tangent[0])
        if mode == "spin":
            # Slow full turn on the spot, tilting up and down twice to catch the floor and
            # ceiling junctions.
            yaw = base_yaw + spin
            pitch = 0.4 * np.sin(2 * spin)
        elif mode == "scan":
            # Face the walls (right of travel on a CCW loop) and sweep up/down and sideways.
            yaw = base_yaw - np.pi / 2 + 0.45 * np.sin(travelled * 2 * np.pi / 1.6)
            pitch = 0.55 * np.sin(travelled * 2 * np.pi / 2.3)
        else:
            yaw = base_yaw + 0.1 * np.sin(travelled * 2 * np.pi / 1.5)
            pitch = -0.05
        poses.append(pose_from_yaw_pitch(eye, yaw, pitch))
    poses = _smooth_rotations(np.asarray(poses), window=5)
    timestamps = np.arange(len(poses)) / fps
    return Trajectory(timestamps, poses, rooms)


def _smooth_rotations(poses: np.ndarray, window: int) -> np.ndarray:
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    smoothed = poses.copy()
    half = window // 2
    for i in range(len(poses)):
        lo, hi = max(0, i - half), min(len(poses), i + half + 1)
        smoothed[i, :3, :3] = rotations[lo:hi].mean().as_matrix()
    return smoothed


@dataclass
class DriftModel:
    """Error of visual-inertial odometry.

    Real VIO poses are smooth frame to frame (millimetre jitter) and drift slowly: a
    velocity bias wanders as a random walk, so position error is an integrated random
    walk (smooth), heading wanders as a random walk, and metric scale is off by a small
    constant. Roll/pitch are observable from gravity and stay clean. level=1 lands around
    0.5% of path length at the end of a ~50 m indoor walk, in line with ARKit reports.
    """
    bias_per_sqrt_m: float = 0.0012
    yaw_deg_per_sqrt_m: float = 0.2
    scale_error: float = 0.003
    jitter_m: float = 0.0015

    @classmethod
    def level(cls, level: float) -> DriftModel:
        return cls(0.0012 * level, 0.2 * level, 0.003 * level, 0.0015 * min(level, 1.0) if level else 0.0)


def apply_drift(poses: np.ndarray, model: DriftModel, seed: int = 0) -> np.ndarray:
    """Dead-reckoning error: each true step is re-integrated with the heading error and
    velocity bias of that moment. A heading error bends only the motion after it, never the
    path already walked."""
    rng = np.random.default_rng(seed)
    scale = 1.0 + rng.normal(0, 1) * model.scale_error if model.scale_error else 1.0
    estimated = [poses[0].copy()]
    heading_error = 0.0
    bias = np.zeros(3)
    position = poses[0][:3, 3].copy()
    for prev, cur in zip(poses[:-1], poses[1:]):
        displacement = cur[:3, 3] - prev[:3, 3]
        step = float(np.linalg.norm(displacement))
        heading_error += np.deg2rad(rng.normal(0, model.yaw_deg_per_sqrt_m * np.sqrt(step)))
        bias += rng.normal(0, model.bias_per_sqrt_m * np.sqrt(step), 3) * np.array([1.0, 1.0, 0.3])
        yaw_rot = Rotation.from_euler("z", heading_error).as_matrix()
        position = position + yaw_rot @ (displacement * scale) + bias * step
        est = np.eye(4)
        est[:3, :3] = yaw_rot @ cur[:3, :3]
        est[:3, 3] = position + rng.normal(0, model.jitter_m, 3)
        estimated.append(est)
    return np.asarray(estimated)


def photo_views(scene: SceneSpec, room_id: str, eye_height: float = 1.5) -> list[tuple[str, np.ndarray]]:
    """Protocol stills for one room: each corner toward the opposite side of the room, then
    one shot per doorway taken ~1 m inside the room facing the doorway."""
    room = scene.room(room_id)
    dims = room.polygon.max(axis=0) - room.polygon.min(axis=0)
    corners = inset_polygon(room.polygon, min(0.35, 0.2 * float(dims.min())))
    centroid = room.polygon.mean(axis=0)
    views = []
    for i, corner in enumerate(corners):
        eye = standable(scene, room, corner)
        target = centroid + (centroid - eye) * 0.6
        views.append((f"corner{i + 1}", look_at([*eye, eye_height], [*target, eye_height - 0.35])))
    doors = [o for o in scene.openings if o.room == room_id and o.type == "door"]
    for door in doors:
        start, direction, _, inward = room.wall_frame(door.wall)
        centre = start + direction * (door.offset + door.width / 2)
        standoff = min(1.0, 0.45 * float(dims.min()))
        eye = standable(scene, room, centre + inward * standoff)
        views.append((f"door_{door.id.rstrip('~')}", look_at([*eye, eye_height], [*centre, eye_height - 0.15])))
    return views[:8]
