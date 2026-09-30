"""Whole-property stitching for rooms reconstructed separately (photo tier).

Every room comes back in its own frame. A door in room A and a door of similar width in
room B are a candidate doorway, and a candidate fixes B's pose relative to A completely:
the two door walls face each other across a wall's thickness and the door centres line up.

Candidates are verified geometrically. The doorway photo taken inside A glimpses part of B
through the door; under the right placement those glimpsed points land on B's surfaces,
under a wrong one they don't. Rooms are then assembled greedily by score (a maximum
spanning tree), accepting a doorway only if it joins two separate groups, reuses no door,
and creates no overlap between rooms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree

WALL_THICKNESS_M = 0.15
WIDTH_TOLERANCE_M = 0.12
GLIMPSE_INLIER_M = 0.12
MAX_OVERLAP_M2 = 0.25


@dataclass
class Transform2D:
    angle: float
    t: np.ndarray

    @property
    def R(self) -> np.ndarray:
        c, s = np.cos(self.angle), np.sin(self.angle)
        return np.array([[c, -s], [s, c]])

    def apply(self, xy: np.ndarray) -> np.ndarray:
        return xy @ self.R.T + self.t

    def compose(self, other: Transform2D) -> Transform2D:
        return Transform2D(self.angle + other.angle, self.R @ other.t + self.t)

    def inverse(self) -> Transform2D:
        return Transform2D(-self.angle, -(Transform2D(-self.angle, np.zeros(2)).R @ self.t))

    @classmethod
    def identity(cls) -> Transform2D:
        return cls(0.0, np.zeros(2))


@dataclass
class DoorRef:
    room: str
    index: int
    wall: int
    centre: np.ndarray
    inward: np.ndarray
    width: float


@dataclass
class Candidate:
    a: DoorRef
    b: DoorRef
    transform: Transform2D
    score: float
    glimpse: tuple[float, float]


@dataclass
class StitchResult:
    transforms: dict[str, Transform2D]
    links: list[Candidate]
    unlinked: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _doors(room: str, layout, openings) -> list[DoorRef]:
    refs = []
    for index, o in enumerate(openings):
        if o.kind not in ("door", "opening"):
            continue
        line, start = layout.lines[o.wall], layout.polygon[o.wall]
        refs.append(DoorRef(room, index, o.wall, start + line.direction * (o.u0 + o.u1) / 2, line.normal, o.u1 - o.u0))
    return refs


def _place(a: DoorRef, b: DoorRef) -> Transform2D:
    """Pose of room B in room A's frame that makes door b the far side of door a."""
    target = np.arctan2(-a.inward[1], -a.inward[0])
    angle = target - np.arctan2(b.inward[1], b.inward[0])
    R = Transform2D(angle, np.zeros(2)).R
    far_side = a.centre - a.inward * WALL_THICKNESS_M
    return Transform2D(angle, far_side - R @ b.centre)


def _glimpse_points(door: DoorRef, cloud, layout) -> np.ndarray:
    """Points of this room's reconstruction beyond the door's wall, in front of the door."""
    xy = cloud.points[:, :2]
    beyond = -(xy - door.centre) @ door.inward
    along = (xy - door.centre) @ np.array([door.inward[1], -door.inward[0]])
    z = cloud.points[:, 2]
    keep = (beyond > 0.35) & (beyond < 3.0) & (np.abs(along) < door.width / 2 + 0.8) & (z > 0.2) & (z < layout.ceiling_z - 0.1)
    points = cloud.points[keep]
    if len(points) > 4000:
        points = points[np.random.default_rng(0).choice(len(points), 4000, replace=False)]
    return points


def _inlier_share(points_xyz: np.ndarray, transform: Transform2D, target_tree: cKDTree) -> float:
    if len(points_xyz) < 50:
        return float("nan")
    moved = np.column_stack([transform.apply(points_xyz[:, :2]), points_xyz[:, 2]])
    distance, _ = target_tree.query(moved, distance_upper_bound=GLIMPSE_INLIER_M)
    return float(np.isfinite(distance).mean())


def _overlap(polygons: list[np.ndarray]) -> float:
    if len(polygons) < 2:
        return 0.0
    corners = np.vstack(polygons)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    xs, ys = np.meshgrid(np.arange(lo[0], hi[0], 0.04), np.arange(lo[1], hi[1], 0.04))
    grid = np.stack([xs.ravel(), ys.ravel()], axis=1)
    count = np.zeros(len(grid), dtype=int)
    for polygon in polygons:
        # Shrink slightly: rooms that share a wall line must not count as overlapping.
        centre = polygon.mean(axis=0)
        count += MplPath(centre + (polygon - centre) * 0.97).contains_points(grid)
    return float(np.sum(count > 1) * 0.04 ** 2)


def stitch(rooms: dict[str, tuple]) -> StitchResult:
    """rooms: name -> (layout, openings, cloud). Returns each room's pose in the property frame."""
    doors = {name: _doors(name, layout, openings) for name, (layout, openings, _) in rooms.items()}
    trees = {name: cKDTree(cloud.points) for name, (_, _, cloud) in rooms.items()}
    glimpses = {(d.room, d.index): _glimpse_points(d, rooms[d.room][2], rooms[d.room][0])
                for refs in doors.values() for d in refs}

    candidates: list[Candidate] = []
    names = list(rooms)
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            for a in doors[name_a]:
                for b in doors[name_b]:
                    if abs(a.width - b.width) > WIDTH_TOLERANCE_M:
                        continue
                    b_in_a = _place(a, b)
                    a_in_b = b_in_a.inverse()
                    forward = _inlier_share(glimpses[(a.room, a.index)], a_in_b, trees[name_b])
                    backward = _inlier_share(glimpses[(b.room, b.index)], b_in_a, trees[name_a])
                    shares = [s for s in (forward, backward) if np.isfinite(s)]
                    visual = float(np.mean(shares)) if shares else 0.25
                    width_term = 1.0 - abs(a.width - b.width) / WIDTH_TOLERANCE_M
                    candidates.append(Candidate(a, b, b_in_a, visual * (0.5 + 0.5 * width_term), (forward, backward)))
    candidates.sort(key=lambda c: -c.score)

    group = {name: name for name in names}

    def find(name: str) -> str:
        while group[name] != name:
            group[name] = group[group[name]]
            name = group[name]
        return name

    transforms = {names[0]: Transform2D.identity()} if names else {}
    local = {name: Transform2D.identity() for name in names}  # pose within its group's root frame
    used: set[tuple[str, int]] = set()
    links: list[Candidate] = []
    notes: list[str] = []
    for cand in candidates:
        ga, gb = find(cand.a.room), find(cand.b.room)
        if ga == gb or (cand.a.room, cand.a.index) in used or (cand.b.room, cand.b.index) in used:
            continue
        if cand.score < 0.15:
            notes.append(f"doorway {cand.a.room}<->{cand.b.room} rejected: glimpses do not match (score {cand.score:.2f})")
            continue
        # Move group gb into ga's frame: pose_b_in_ga = local[a] . (b in a) . local[b]^-1
        b_group_to_a = local[cand.a.room].compose(cand.transform).compose(local[cand.b.room].inverse())
        moved = {n: b_group_to_a.compose(local[n]) for n in names if find(n) == gb}
        polygons = [local[n].apply(rooms[n][0].polygon) for n in names if find(n) == ga] + \
                   [moved[n].apply(rooms[n][0].polygon) for n in moved]
        if _overlap(polygons) > MAX_OVERLAP_M2:
            notes.append(f"doorway {cand.a.room}<->{cand.b.room} rejected: placement overlaps another room")
            continue
        for n, pose in moved.items():
            local[n] = pose
        group[gb] = ga
        used |= {(cand.a.room, cand.a.index), (cand.b.room, cand.b.index)}
        links.append(cand)

    roots = {find(n) for n in names}
    main = max(roots, key=lambda r: sum(find(n) == r for n in names)) if names else None
    unlinked = [n for n in names if find(n) != main]
    offset = 0.0
    for n in names:
        if find(n) == main:
            transforms[n] = local[n]
    if unlinked:
        extent = np.vstack([transforms[n].apply(rooms[n][0].polygon) for n in transforms]).max(axis=0)
        for n in unlinked:
            polygon = local[n].apply(rooms[n][0].polygon)
            shift = np.array([extent[0] + 1.0 + offset - polygon.min(axis=0)[0], -polygon.min(axis=0)[1]])
            transforms[n] = Transform2D(local[n].angle, local[n].t + shift)
            offset += polygon.max(axis=0)[0] - polygon.min(axis=0)[0] + 1.0
        notes.append(f"no verified doorway to: {', '.join(unlinked)} (placed beside the plan, unconnected)")
    return StitchResult(transforms, links, unlinked, notes)
