"""Door and window detection from ray evidence, with jamb-refined widths.

A hole in a point cloud is ambiguous: unseen, or open? So instead of looking for missing
points, every depth ray that reaches a wall's plane casts a vote in that wall's (u, v)
grid (u along the wall, v height):

  face      the ray stopped on the wall face           -> wall
  recess    stopped a little behind it (jamb, glass)   -> weakly open
  through   passed well beyond the face                -> open
  noreturn  no depth at all (glass, open air)          -> open
  (a ray stopped in front of the face is occluded by furniture and casts no vote)

Openings are regions where open votes dominate. Edges are then refined from the jamb and
head (reveal) surfaces inside the wall thickness, which pin a door edge far more precisely
than the blurred depth edge. A candidate whose see-through points, reflected back across
the wall plane, land on the room's own geometry is a mirror, not an opening.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ..capture import Frame
from .cloud import Cloud
from .layout import RoomLayout, WallLine
from .rooms import RoomRegion
from .tolerances import LIDAR_TOL, Tolerances

GRID_M = 0.02
RAY_STRIDE = 2
MAX_RANGE_M = 4.5
OPEN_SCORE = 0.55
MIN_SIZE_M = 0.45
DOOR_SILL_MAX_M = 0.10
MIRROR_MATCH_M = 0.04
MIRROR_FRACTION = 0.5
JAMB_MIN_POINTS = 20
JAMB_BASE_SIGMA_M = 0.003
EDGE_FALLBACK_SIGMA_M = 0.015
THROUGH_SAMPLES = 4000


@dataclass
class WallEvidence:
    face: np.ndarray
    recess: np.ndarray
    through: np.ndarray
    noreturn: np.ndarray
    through_points: list = field(default_factory=list)
    through_cells: list = field(default_factory=list)


@dataclass
class Opening:
    room: str
    wall: int
    kind: str
    u0: float
    u1: float
    v0: float
    v1: float
    sigma_u0: float
    sigma_u1: float
    sigma_v0: float
    sigma_v1: float
    confidence: float
    jambs_found: int
    connects_to: str | None = None
    partner: Opening | None = None

    @property
    def width(self) -> float:
        return self.u1 - self.u0

    @property
    def width_sigma(self) -> float:
        return float(np.hypot(self.sigma_u0, self.sigma_u1))


@dataclass
class _Wall:
    index: int
    line: WallLine
    start: np.ndarray
    length: float
    floor_z: float
    ceiling_z: float
    evidence: WallEvidence


def _walls_of(layout: RoomLayout) -> list[_Wall]:
    walls = []
    n = len(layout.polygon)
    for k, line in enumerate(layout.lines):
        start, end = layout.polygon[k], layout.polygon[(k + 1) % n]
        length = float((end - start) @ line.direction)
        nu = int(np.ceil(length / GRID_M))
        nv = int(np.ceil((layout.ceiling_z - layout.floor_z) / GRID_M))
        shape = (max(nu, 1), max(nv, 1))
        walls.append(_Wall(k, line, start, length, layout.floor_z, layout.ceiling_z,
                           WallEvidence(*(np.zeros(shape, np.int32) for _ in range(4)))))
    return walls


def _accumulate(frame: Frame, pose: np.ndarray, walls: list[_Wall], rng: np.random.Generator,
                tol: Tolerances = LIDAR_TOL) -> None:
    depth = frame.depth()[::RAY_STRIDE, ::RAY_STRIDE].astype(np.float64)
    K = frame.depth_K
    rows, cols = np.indices(depth.shape)
    cam_dirs = np.stack([(cols * RAY_STRIDE - K[0, 2]) / K[0, 0], (rows * RAY_STRIDE - K[1, 2]) / K[1, 1],
                         np.ones(depth.shape)], axis=-1).reshape(-1, 3)
    measured = depth.reshape(-1)
    dirs = cam_dirs @ pose[:3, :3].T
    centre = pose[:3, 3]
    for wall in walls:
        normal = np.array([wall.line.normal[0], wall.line.normal[1], 0.0])
        ahead = float(normal @ centre - wall.line.offset)
        if ahead < 0.1:
            continue
        toward = dirs @ normal
        ok = toward < -1e-3
        if not ok.any():
            continue
        t_wall = np.full(len(dirs), np.inf)
        t_wall[ok] = -ahead / toward[ok]
        hit = centre + dirs * np.where(ok, t_wall, 0.0)[:, None]
        ranges = t_wall * np.linalg.norm(dirs, axis=1)
        u = (hit[:, :2] - wall.start) @ wall.line.direction
        v = hit[:, 2]
        keep = ok & (ranges < MAX_RANGE_M) & (u >= 0) & (u < wall.length) & (v >= wall.floor_z) & (v < wall.ceiling_z)
        if not keep.any():
            continue
        shape = wall.evidence.face.shape
        iu = np.minimum((u[keep] / GRID_M).astype(int), shape[0] - 1)
        iv = np.minimum(((v[keep] - wall.floor_z) / GRID_M).astype(int), shape[1] - 1)
        flat = iu * shape[1] + iv
        d = measured[keep]
        behind = (d - t_wall[keep]) * (-toward[keep])
        noreturn = d <= 0
        end_z = centre[2] + dirs[keep][:, 2] * d
        # A ray that lands on the floor beyond the wall plane went through a doorway, even
        # if it landed only a few centimetres past the face (the threshold, the next room's
        # floor seen at a steep angle).
        on_floor_beyond = ~noreturn & (behind > tol.floor_beyond_m) & (end_z < wall.floor_z + 0.05)
        face = ~noreturn & (np.abs(behind) <= tol.opening_face_m) & ~on_floor_beyond
        recess = ~noreturn & (behind > tol.opening_face_m) & (behind <= tol.opening_recess_m) & ~on_floor_beyond
        through = ~noreturn & ((behind > tol.opening_recess_m) | on_floor_beyond)
        size = shape[0] * shape[1]
        for grid, sel in ((wall.evidence.face, face), (wall.evidence.recess, recess),
                          (wall.evidence.through, through), (wall.evidence.noreturn, noreturn)):
            grid += np.bincount(flat[sel], minlength=size).reshape(shape).astype(np.int32)
        if through.any():
            pts = centre + dirs[keep][through] * d[through][:, None]
            cells = np.stack([iu[through], iv[through]], 1)
            if len(pts) > 200:
                pick = rng.choice(len(pts), 200, replace=False)
                pts, cells = pts[pick], cells[pick]
            wall.evidence.through_points.append(pts)
            wall.evidence.through_cells.append(cells)


def detect_openings(layout: RoomLayout, region: RoomRegion, frames: list[Frame],
                    poses: dict[int, np.ndarray], tol: Tolerances = LIDAR_TOL) -> tuple[list[Opening], list[str]]:
    walls = _walls_of(layout)
    rng = np.random.default_rng(0)
    inside = [f for f in frames if region.contains(poses[f.index][None, :2, 3], dilate_cells=2)[0]] or frames
    for frame in inside:
        _accumulate(frame, poses[frame.index], walls, rng, tol)
    room_tree = cKDTree(layout.points.points)
    openings, notes = [], []
    for wall in walls:
        found, wall_notes = _extract(layout, wall, room_tree, tol)
        openings += found
        notes += wall_notes
    return openings, notes


def _extract(layout: RoomLayout, wall: _Wall, room_tree: cKDTree,
             tol: Tolerances = LIDAR_TOL) -> tuple[list[Opening], list[str]]:
    ev = wall.evidence
    total = ev.face + ev.recess + ev.through + ev.noreturn
    score = np.divide(ev.through + ev.noreturn + 0.5 * ev.recess, total, out=np.zeros(total.shape), where=total > 0)
    mask = (total >= tol.min_votes) & (score > OPEN_SCORE)
    mask = ndimage.binary_opening(mask, structure=np.ones((2, 2)))
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))
    labels, count = ndimage.label(mask)
    through_pts = np.concatenate(ev.through_points) if ev.through_points else np.zeros((0, 3))
    through_cells = np.concatenate(ev.through_cells) if ev.through_cells else np.zeros((0, 2), int)
    found, notes = [], []
    for label in range(1, count + 1):
        cells = np.argwhere(labels == label)
        (iu0, iv0), (iu1, iv1) = cells.min(axis=0), cells.max(axis=0) + 1
        u0, u1 = iu0 * GRID_M, iu1 * GRID_M
        v0, v1 = wall.floor_z + iv0 * GRID_M, wall.floor_z + iv1 * GRID_M
        if u1 - u0 < MIN_SIZE_M or v1 - v0 < MIN_SIZE_M:
            continue
        fill = len(cells) / ((iu1 - iu0) * (iv1 - iv0))
        if fill < 0.55:
            continue
        member = labels[through_cells[:, 0], through_cells[:, 1]] == label if len(through_cells) else np.zeros(0, bool)
        if _is_mirror(through_pts[member], wall, room_tree):
            notes.append(f"wall {wall.index}: {u1 - u0:.2f} x {v1 - v0:.2f} m see-through patch is a mirror "
                         "(reflected points land on this room); not an opening")
            # Kept (kind "mirror") so later stages can mask it: a reflection is not a stain.
            found.append(Opening(layout.id, wall.index, "mirror", u0, u1, v0, v1, 0.02, 0.02, 0.02, 0.02, 1.0, 0))
            continue
        touches_floor = v0 - wall.floor_z < DOOR_SILL_MAX_M
        if not touches_floor:
            # A window has wall below its sill. If the strip under this patch shows no wall
            # face (only unobserved or open cells), the bottom of a doorway simply wasn't
            # seen, as when facing a door from under a metre away.
            # The bottom 10 cm is skipped: rays grazing a door threshold land right at the
            # wall plane and look like wall face.
            lowest = int(0.10 / GRID_M)
            strip = (slice(iu0 + (iu1 - iu0) // 4, iu1 - (iu1 - iu0) // 4), slice(lowest, max(iv0 - 1, lowest)))
            below_face = ev.face[strip].sum()
            below_votes = total[strip].sum()
            touches_floor = iv0 <= lowest + 1 or below_votes < 50 or below_face < 0.3 * below_votes
            if touches_floor:
                v0 = wall.floor_z
        kind = "opening" if touches_floor and v1 > wall.ceiling_z - 0.05 else ("door" if touches_floor else "window")
        opening = _refine(layout, wall, kind, u0, u1, v0, v1, tol)
        observed = float((total[iu0:iu1, iv0:iv1] >= tol.min_votes).mean())
        opening.confidence = float(np.clip(observed * min(1.0, fill / 0.8), 0, 1))
        found.append(opening)
    return found, notes


def _is_mirror(points: np.ndarray, wall: _Wall, room_tree: cKDTree) -> bool:
    """Reflect see-through points across the wall plane; a mirror's phantom lands back on
    this room. Only surfaces facing the wall are informative: floor, ceiling and walls
    perpendicular to it are unchanged by the reflection, and neighbouring rooms often
    share such walls (both north walls on one line), which would fake a match."""
    if len(points) > THROUGH_SAMPLES:
        points = points[np.random.default_rng(1).choice(len(points), THROUGH_SAMPLES, replace=False)]
    if len(points) < 60:
        return False
    normal = np.array([wall.line.normal[0], wall.line.normal[1], 0.0])
    _, neighbours = cKDTree(points).query(points, k=12)
    local = points[neighbours] - points[neighbours].mean(axis=1, keepdims=True)
    _, _, vt = np.linalg.svd(local, full_matrices=False)
    surface_normals = vt[:, -1, :]
    facing = (np.abs(surface_normals @ normal) > 0.7) & (points[:, 2] > wall.floor_z + 0.15) & \
        (points[:, 2] < wall.ceiling_z - 0.15)
    points = points[facing]
    if len(points) < 30:
        return False
    signed = points @ normal - wall.line.offset
    reflected = points - 2 * signed[:, None] * normal
    distance, _ = room_tree.query(reflected, distance_upper_bound=MIRROR_MATCH_M)
    return float(np.isfinite(distance).mean()) >= MIRROR_FRACTION


def _refine(layout: RoomLayout, wall: _Wall, kind: str, u0: float, u1: float, v0: float, v1: float,
            tol: Tolerances = LIDAR_TOL) -> Opening:
    """Snap opening edges to the reveal surfaces inside the wall thickness."""
    pts, nrm = layout.points.points, layout.points.normals
    direction = wall.line.direction
    u = (pts[:, :2] - wall.start) @ direction
    behind = -(pts[:, :2] @ wall.line.normal - wall.line.offset)
    z = pts[:, 2]
    in_wall = (behind > 0.01) & (behind < tol.opening_recess_m)
    facing_u = nrm[:, :2] @ direction
    mid_height = (z > v0 + 0.1) & (z < v1 - 0.1)
    mid_span = (u > u0 + 0.1) & (u < u1 - 0.1)

    def edge(select: np.ndarray, coord: np.ndarray, fallback: float) -> tuple[float, float, int]:
        values = coord[select]
        if len(values) < JAMB_MIN_POINTS:
            return fallback, EDGE_FALLBACK_SIGMA_M, 0
        # Density mode, not median: the reveal surface is a sharp peak; residual flying
        # pixels spread around it.
        hist, bins = np.histogram(values, bins=np.arange(values.min(), values.max() + 0.01, 0.005))
        peak = bins[np.argmax(hist)] + 0.0025
        near = values[np.abs(values - peak) < 0.01]
        centre = float(np.mean(near))
        mad = 1.4826 * float(np.median(np.abs(near - centre)))
        return centre, float(np.hypot(mad / np.sqrt(min(len(near), 200)), JAMB_BASE_SIGMA_M)), 1

    # Flying pixels from densified depth are removed upstream by using only high-confidence
    # returns for measurement (ARKit marks depth-edge pixels low/medium).
    window = tol.jamb_window_m
    reach = tol.jamb_reach_m
    left = in_wall & mid_height & (facing_u > 0.85) & (u > u0 - window - reach) & (u < u0 + window)
    right = in_wall & mid_height & (facing_u < -0.85) & (u > u1 - window) & (u < u1 + window + reach)
    head = in_wall & mid_span & (nrm[:, 2] < -0.85) & (np.abs(z - v1) < window)
    sill = in_wall & mid_span & (nrm[:, 2] > 0.85) & (np.abs(z - v0) < window)
    ru0, su0, f0 = edge(left, u, u0)
    ru1, su1, f1 = edge(right, u, u1)
    rv1, sv1, f2 = edge(head, z, v1)
    if kind == "window":
        rv0, sv0, f3 = edge(sill, z, v0)
    else:
        rv0, sv0, f3 = wall.floor_z, 0.003, 1
    return Opening(layout.id, wall.index, kind, ru0, ru1, rv0, rv1, su0, su1, sv0, sv1, 1.0, f0 + f1 + f2 + f3)


def match_doors(openings: list[Opening], layouts: dict[str, RoomLayout]) -> list[tuple[Opening, Opening]]:
    """Pair doors on facing walls of two rooms across a plausible wall thickness."""
    pairs = []
    doors = [o for o in openings if o.kind in ("door", "opening")]
    for i, a in enumerate(doors):
        la = layouts[a.room]
        line_a = la.lines[a.wall]
        start_a = la.polygon[a.wall]
        a_ends = [start_a + line_a.direction * a.u0, start_a + line_a.direction * a.u1]
        best = None
        for b in doors[i + 1:]:
            if b.room == a.room or b.partner is not None:
                continue
            lb = layouts[b.room]
            line_b = lb.lines[b.wall]
            if line_a.normal @ line_b.normal > -0.95:
                continue
            gap = -(np.mean(a_ends, axis=0) @ line_b.normal - line_b.offset)
            if not 0.03 < gap < 0.45:
                continue
            start_b = lb.polygon[b.wall]
            b_along = sorted(float((start_b + line_b.direction * s - start_a) @ line_a.direction) for s in (b.u0, b.u1))
            overlap = min(a.u1, b_along[1]) - max(a.u0, b_along[0])
            if overlap < 0.5 * min(a.width, b.width):
                continue
            if best is None or gap < best[0]:
                best = (gap, b)
        if best is not None:
            b = best[1]
            a.connects_to, b.connects_to = b.room, a.room
            a.partner, b.partner = b, a
            _fuse_pair(a, b, la, layouts[b.room])
            pairs.append((a, b))
    return pairs


def _fuse_pair(a: Opening, b: Opening, la: RoomLayout, lb: RoomLayout) -> None:
    """Both sides of a doorway see the same jambs and head: combine each edge by inverse
    variance and write the result back into each side's own wall coordinates."""
    line_a, start_a = la.lines[a.wall], la.polygon[a.wall]
    line_b, start_b = lb.lines[b.wall], lb.polygon[b.wall]

    def to_a(u_b: float) -> float:
        return float((start_b + line_b.direction * u_b - start_a) @ line_a.direction)

    def to_b(u_a: float) -> float:
        return float((start_a + line_a.direction * u_a - start_b) @ line_b.direction)

    # b's walls run the opposite way, so b's u1 edge is a's u0 edge.
    edges = [((a.u0, a.sigma_u0), (to_a(b.u1), b.sigma_u1)), ((a.u1, a.sigma_u1), (to_a(b.u0), b.sigma_u0))]
    fused = []
    for (va, sa), (vb, sb) in edges:
        wa, wb = 1 / sa ** 2, 1 / sb ** 2
        fused.append(((va * wa + vb * wb) / (wa + wb), (wa + wb) ** -0.5))
    (u0, s0), (u1, s1) = fused
    a.u0, a.sigma_u0, a.u1, a.sigma_u1 = u0, s0, u1, s1
    b.u0, b.sigma_u0, b.u1, b.sigma_u1 = to_b(u1), s1, to_b(u0), s0
    # Heads are at the same height on both sides (the floor may step, the head does not).
    wa, wb = 1 / a.sigma_v1 ** 2, 1 / b.sigma_v1 ** 2
    head = (a.v1 * wa + b.v1 * wb) / (wa + wb)
    a.v1 = b.v1 = head
    a.sigma_v1 = b.sigma_v1 = (wa + wb) ** -0.5
