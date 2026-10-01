"""Photo tier: 2-8 stills per room folder, no depth, no poses -> stitched, dimensioned plan.

Per room:
  MoGe-2 metric depth per photo (FOV from EXIF when present)
  -> MapAnything multi-view reconstruction conditioned on that depth (poses, refined depth)
  -> gravity from geometry (camera-down prior refined by floor/ceiling normals)
  -> the shared geometry pipeline on pseudo-LiDAR frames with photo tolerances
Then rooms are stitched by door correspondences (geometry/stitch.py).

Scale is the dominant error. Metric scale comes from MoGe-2, measured on real iPad
views with laser truth at -0.5% bias and 6.9% per-view spread, so a room's scale sigma
is sqrt((6.9% / sqrt(n_photos))^2 + BIAS^2): fewer photos, wider intervals.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..capture import Frame
from ..geometry.cloud import fuse
from ..geometry.layout import RoomLayout, room_layout
from ..geometry.sparse_layout import rectangle_layout
from ..geometry.openings import Opening, detect_openings
from ..geometry.rooms import RoomRegion, align, segment_rooms
from ..geometry.tolerances import PHOTO_TOL
from ..io.photos import Photo

MOGE_VIEW_SPREAD = 0.069
CONSISTENT_SHARE = 0.5
INCONSISTENT_SCALE_SIGMA = 0.25
SCALE_BIAS_BUDGET = 0.02
PHOTO_VOXEL = 0.03
GRAVITY_PRIOR_WEIGHT = 0.02
# Benchmark ablation hook (roomscope run --oracle-depth): photo -> MetricDepth.
depth_source = None


@dataclass
class RoomReconstruction:
    room: str
    frames: list[Frame]
    poses: dict[int, np.ndarray]
    region: RoomRegion | None
    layout: RoomLayout | None
    openings: list[Opening]
    scale_sigma: float
    cloud: object
    notes: list[str] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)


def scale_sigma(n_views: int) -> float:
    return float(np.hypot(MOGE_VIEW_SPREAD / np.sqrt(max(n_views, 1)), SCALE_BIAS_BUDGET))


def view_agreement(frames: list[Frame]) -> float:
    """Multi-view consistency: project each view's points into the others and count how
    often their depth agrees within 10%. Broken poses (a narrow, repetitive corridor can
    defeat multi-view reconstruction) show up as low agreement."""
    agree, total = 0, 0
    for a in frames:
        depth_a = a.depth()
        rows, cols = np.nonzero(depth_a > 0)
        if len(rows) == 0:
            continue
        pick = np.random.default_rng(a.index).choice(len(rows), min(2000, len(rows)), replace=False)
        rows, cols = rows[pick], cols[pick]
        z = depth_a[rows, cols]
        K = a.depth_K
        cam = np.stack([(cols - K[0, 2]) / K[0, 0] * z, (rows - K[1, 2]) / K[1, 1] * z, z], 1)
        world = cam @ a.pose[:3, :3].T + a.pose[:3, 3]
        for b in frames:
            if b is a:
                continue
            local = (world - b.pose[:3, 3]) @ b.pose[:3, :3]
            Kb = b.depth_K
            w, h = b.depth_size
            u = Kb[0, 0] * local[:, 0] / np.maximum(local[:, 2], 1e-6) + Kb[0, 2]
            v = Kb[1, 1] * local[:, 1] / np.maximum(local[:, 2], 1e-6) + Kb[1, 2]
            seen = (local[:, 2] > 0.2) & (u >= 0) & (v >= 0) & (u < w - 1) & (v < h - 1)
            if seen.sum() < 50:
                continue
            measured = b.depth()[v[seen].astype(int), u[seen].astype(int)]
            ok = measured > 0
            agree += int(np.sum(ok & (np.abs(local[seen, 2] - measured) < 0.1 * measured)))
            total += int(ok.sum())
    return agree / total if total else 0.0


def gravity_rotation(normals: np.ndarray, poses: list[np.ndarray]) -> np.ndarray:
    """World rotation taking gravity-up to +z. Phones are held roughly upright, so the
    cameras' mean down axis is a good start. Then up is refined against both kinds of
    surface: floor and ceiling normals are parallel to it, wall normals perpendicular. Walls
    matter: from a few photos or one turn there is often far more wall than floor in view."""
    up = -np.mean([pose[:3, 1] for pose in poses], axis=0)
    up /= np.linalg.norm(up)
    prior = up.copy()
    for horizontal_cos, vertical_cos in ((0.8, 0.4), (0.9, 0.25), (0.95, 0.15)):
        along = np.abs(normals @ up)
        level, wall = normals[along > horizontal_cos], normals[along < vertical_cos]
        if len(level) + len(wall) < 200:
            break
        # Up maximises (n.up)^2 over level surfaces and minimises it over walls; each set
        # is averaged so neither dominates by point count.
        scatter = np.zeros((3, 3))
        if len(level):
            scatter += level.T @ level / len(level)
        if len(wall):
            scatter -= wall.T @ wall / len(wall)
        # A weak pull to the camera prior: in a corridor the walls fix only one horizontal
        # axis, and with little floor in view up would be ambiguous with the other.
        scatter += GRAVITY_PRIOR_WEIGHT * np.outer(prior, prior)
        values, vectors = np.linalg.eigh(scatter)
        candidate = vectors[:, -1] * np.sign(vectors[:, -1] @ prior)
        if candidate @ prior < np.cos(np.radians(30)):
            break
        up = candidate
    x = np.cross([0.0, 1.0, 0.0], up) if abs(up[1]) < 0.9 else np.cross([1.0, 0.0, 0.0], up)
    x /= np.linalg.norm(x)
    y = np.cross(up, x)
    return np.stack([x, y, up])


def _confidence_levels(confidence: np.ndarray, mask: np.ndarray) -> np.ndarray:
    levels = np.zeros(confidence.shape, dtype=np.uint8)
    if mask.any():
        lo, hi = np.percentile(confidence[mask], [15, 40])
        levels[mask & (confidence >= lo)] = 1
        levels[mask & (confidence >= hi)] = 2
    return levels


def reconstruct_room(room: str, photos: list[Photo], first_index: int = 0) -> RoomReconstruction:
    from ..models.depth import metric_depth
    from ..models.multiview import reconstruct

    timing: dict[str, float] = {}
    start = time.perf_counter()
    metric = [depth_source(p) if depth_source else metric_depth(p.image, p.fov_x_deg) for p in photos]
    intrinsics = [p.K if p.K is not None else m.K for p, m in zip(photos, metric)]
    timing["depth"] = round(time.perf_counter() - start, 2)
    if all(p.rotation is not None for p in photos):
        return _panorama_room(room, photos, metric, intrinsics, first_index, timing)
    start = time.perf_counter()
    views = reconstruct([p.image for p in photos], intrinsics, [m.depth for m in metric])
    timing["multiview"] = round(time.perf_counter() - start, 2)

    # MapAnything is conditioned on MoGe depth, so their per-view ratio should be ~1; its
    # median corrects what's left and its spread is reported.
    ratios = []
    for view, m in zip(views, metric):
        valid = view.mask & m.mask & (view.depth > 0.1)
        if valid.sum() > 500:
            ratios.append(float(np.median(m.depth[valid] / view.depth[valid])))
    correction = float(np.median(ratios)) if ratios else 1.0

    frames = []
    for k, (photo, view) in enumerate(zip(photos, views)):
        depth = np.where(view.mask, view.depth * correction, 0.0).astype(np.float32)
        confidence = _confidence_levels(view.confidence, view.mask)
        pose = view.pose.copy()
        pose[:3, 3] *= correction
        height, width = depth.shape
        # Colour comes from the full-resolution original (cracks need the pixels); depth
        # and geometry stay at processing resolution.
        full_h, full_w = photo.original.shape[:2]
        full_K = view.K.copy()
        full_K[0] *= full_w / width
        full_K[1] *= full_h / height
        frames.append(Frame(first_index + k, float(k), full_K, (full_w, full_h), pose, room,
                            load_rgb=(lambda im=photo.original: im), load_depth=(lambda d=depth: d),
                            load_confidence=(lambda c=confidence: c), depth_K=view.K, depth_size=(width, height)))

    agreement = view_agreement(frames)
    notes = [f"{len(photos)} photos, scale correction {correction:.3f}"
             + (f", per-view ratio spread {np.std(ratios):.1%}" if len(ratios) > 1 else "")
             + f", multi-view agreement {agreement:.0%}"]
    return _finish_room(room, photos, frames, agreement, notes, timing)


def _edge_free(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Confidence levels for a monocular depth map: depth discontinuities (where flying
    pixels sit) drop to level 0."""
    gy, gx = np.gradient(np.where(mask, depth, 0.0))
    relative = np.hypot(gx, gy) / np.maximum(depth, 0.1)
    levels = np.where(mask, 2, 0).astype(np.uint8)
    levels[relative > 0.04] = 0
    return levels


ARM_LENGTHS_M = np.arange(0.0, 0.46, 0.05)


def _pair_log_ratios(frames: list[Frame]) -> list[tuple[int, int, float, int]]:
    """For each overlapping view pair (a, b): log of the median ratio of b's measured depth
    to the depth b would see if a's points were right, over points that roughly agree."""
    pairs = []
    for i, a in enumerate(frames):
        depth_a = a.depth()
        rows, cols = np.nonzero(a.confidence() >= 2)
        if len(rows) == 0:
            continue
        pick = np.random.default_rng(a.index).choice(len(rows), min(4000, len(rows)), replace=False)
        rows, cols = rows[pick], cols[pick]
        z = depth_a[rows, cols]
        K = a.depth_K
        cam = np.stack([(cols - K[0, 2]) / K[0, 0] * z, (rows - K[1, 2]) / K[1, 1] * z, z], 1)
        world = cam @ a.pose[:3, :3].T + a.pose[:3, 3]
        for j, b in enumerate(frames):
            if j == i:
                continue
            local = (world - b.pose[:3, 3]) @ b.pose[:3, :3]
            Kb = b.depth_K
            w, h = b.depth_size
            u = Kb[0, 0] * local[:, 0] / np.maximum(local[:, 2], 1e-6) + Kb[0, 2]
            v = Kb[1, 1] * local[:, 1] / np.maximum(local[:, 2], 1e-6) + Kb[1, 2]
            seen = (local[:, 2] > 0.2) & (u >= 0) & (v >= 0) & (u < w - 1) & (v < h - 1)
            if seen.sum() < 200:
                continue
            measured = b.depth()[v[seen].astype(int), u[seen].astype(int)]
            ok = measured > 0
            if ok.sum() < 200:
                continue
            ratio = np.log(measured[ok] / local[seen, 2][ok])
            ratio = ratio[np.abs(ratio - np.median(ratio)) < 0.15]
            if len(ratio) >= 100:
                pairs.append((i, j, float(np.median(ratio)), len(ratio)))
    return pairs


PAIR_CAUCHY = 0.1
BAD_VIEW_RESIDUAL = 0.2


def _relative_scales(pairs: list[tuple[int, int, float, int]], count: int) -> tuple[np.ndarray, set[int]]:
    """Per-view scale corrections c with log c_b - log c_a = -log ratio_ab, solved by
    Cauchy-weighted least squares (IRLS), gauge: geometric mean 1, so the room keeps the
    views' average metric scale. A weak prior (sigma 0.3 in log) keeps a view with no
    usable overlap at its own scale. Views whose pairs stay inconsistent after the solve
    (a depth map that is distorted, not just mis-scaled: a window, a mirror) are returned
    as bad."""
    if not pairs:
        return np.ones(count), set()
    A = np.zeros((len(pairs) + count + 1, count))
    b = np.zeros(len(pairs) + count + 1)
    base = np.zeros(len(pairs) + count + 1)
    for row, (i, j, log_ratio, n) in enumerate(pairs):
        A[row, j], A[row, i] = 1.0, -1.0
        b[row] = -log_ratio
        base[row] = np.sqrt(min(n, 2000))
    for k in range(count):
        A[len(pairs) + k, k] = 1.0
        base[len(pairs) + k] = np.sqrt(200.0)
    A[-1] = 1.0
    base[-1] = 100.0
    weights = base.copy()
    for _ in range(8):
        solution = np.linalg.lstsq(A * weights[:, None], b * weights, rcond=None)[0]
        residual = A[:len(pairs)] @ solution - b[:len(pairs)]
        weights[:len(pairs)] = base[:len(pairs)] / np.sqrt(1 + (residual / PAIR_CAUCHY) ** 2)
    involved = {k: [abs(r) for (i, j, _, _), r in zip(pairs, residual) if k in (i, j)] for k in range(count)}
    bad = {k for k, rs in involved.items() if rs and np.median(rs) > BAD_VIEW_RESIDUAL}
    return np.exp(np.clip(solution, np.log(0.5), np.log(2.0))), bad


def _panorama_frames(room, photos, metric, intrinsics, first_index, arm, scales=None):
    frames = []
    for k, (photo, m, K) in enumerate(zip(photos, metric, intrinsics)):
        depth = np.where(m.mask, m.depth * (1.0 if scales is None else scales[k]), 0.0).astype(np.float32)
        height, width = depth.shape
        pose = np.eye(4)
        pose[:3, :3] = photo.rotation
        # The phone circles the body's turning axis, held `arm` in front of it.
        pose[:3, 3] = arm * photo.rotation[:, 2]
        full_h, full_w = photo.original.shape[:2]
        full_K = K.copy()
        full_K[0] *= full_w / width
        full_K[1] *= full_h / height
        levels = _edge_free(depth, m.mask)
        frames.append(Frame(first_index + k, float(k), full_K, (full_w, full_h), pose, room,
                            load_rgb=(lambda im=photo.original: im), load_depth=(lambda d=depth: d),
                            load_confidence=(lambda c=levels: c), depth_K=K, depth_size=(width, height)))
    return frames


def _panorama_room(room, photos, metric, intrinsics, first_index, timing) -> RoomReconstruction:
    """Views from one on-the-spot turn: rotations are known from the video's homography
    chain and the camera centres lie on a small circle round the turning axis, so no
    multi-view pose solving is needed. Each view keeps its own MoGe-2 metric depth; the arm
    length (phone to turning axis) is the one under which the views agree best."""
    start = time.perf_counter()
    scored = []
    for arm in ARM_LENGTHS_M:
        scales = np.ones(len(photos))
        for _ in range(3):
            frames = _panorama_frames(room, photos, metric, intrinsics, first_index, arm, scales)
            step, bad = _relative_scales(_pair_log_ratios(frames), len(photos))
            scales = np.clip(scales * step, 0.4, 2.5)
            if np.max(np.abs(np.log(step))) < 0.005:
                break
        frames = _panorama_frames(room, photos, metric, intrinsics, first_index, arm, scales)
        kept = [f for k, f in enumerate(frames) if k not in bad]
        scored.append((view_agreement(kept), arm, kept, scales, bad))
    agreement, arm, frames, scales, bad = max(scored, key=lambda item: item[0])
    timing["panorama"] = round(time.perf_counter() - start, 2)
    notes = [f"{len(photos)} views from one turn, arm {arm:.2f} m, per-view scale spread {np.std(np.log(scales)):.1%}"
             f", multi-view agreement {agreement:.0%}"]
    if bad:
        notes.append(f"{len(bad)} view(s) dropped: depth inconsistent with the overlapping views")
    return _finish_room(room, photos, frames, agreement, notes, timing)


def _finish_room(room, photos, frames, agreement, notes, timing) -> RoomReconstruction:
    raw = fuse(frames, {f.index: f.pose for f in frames}, voxel=PHOTO_VOXEL, stride=1, min_confidence=1)
    gravity = np.eye(4)
    gravity[:3, :3] = gravity_rotation(raw.normals, [f.pose for f in frames])
    poses = {f.index: gravity @ f.pose for f in frames}
    cloud = fuse(frames, poses, voxel=PHOTO_VOXEL, stride=1, min_confidence=1)
    alignment = align(cloud)
    cloud = alignment.apply(cloud)
    poses = {k: alignment.pose(v) for k, v in poses.items()}

    room_sigma = scale_sigma(len(photos))
    if agreement < CONSISTENT_SHARE:
        room_sigma = max(room_sigma, INCONSISTENT_SCALE_SIGMA)
        notes.append(f"photos do not agree with each other ({agreement:.0%} consistent): reconstruction "
                     f"unreliable, intervals widened to {room_sigma:.0%} scale")
    regions, seg_notes = segment_rooms(cloud, frames, poses, min_frames_inside=PHOTO_TOL.min_room_frames, single_room=True)
    notes += seg_notes
    region = max(regions, key=lambda r: (r.frames_inside, r.area)) if regions else None
    layout = openings = None
    cameras = np.array([poses[f.index][:2, 3] for f in frames])
    layout = rectangle_layout(room, cloud, cameras)
    if region is not None:
        region.id = room
        if layout is None:
            layout = room_layout(region, cloud, PHOTO_TOL)
    if layout is not None:
        openings, opening_notes = detect_openings(layout, region, frames, poses, PHOTO_TOL)
        notes += opening_notes
    else:
        notes.append("room layout could not be recovered from these photos")
    return RoomReconstruction(room, frames, poses, region, layout, openings or [], room_sigma, cloud, notes, timing)


def run_photo(root) -> "PhotoResult":
    from pathlib import Path

    from ..build import PHOTO
    from ..io.photos import read_photo_folders

    return plan_from_photo_sets(read_photo_folders(root), Path(root).name, PHOTO, "Camera (photos)")


def room_headings(reconstructions: dict, orientations: dict) -> dict[str, float]:
    """Each room frame's rotation (about the vertical) into the first room's frame, from
    camera rotations known in one common frame (the video's visual compass). Room frames are
    wall-aligned, so the relative heading is snapped to a multiple of 90 degrees; drift of
    up to ~45 degrees between rooms is harmless."""
    global_from_room = {}
    for name, rec in reconstructions.items():
        if name not in orientations or not rec.frames:
            continue
        # A frame's timestamp is its slot in the room's photo list.
        estimates = [orientations[name][int(f.timestamp)] @ rec.poses[f.index][:3, :3].T for f in rec.frames]
        u, _, vt = np.linalg.svd(np.sum(estimates, axis=0))
        global_from_room[name] = u @ vt
    if not global_from_room:
        return {}
    reference = next(iter(global_from_room.values()))
    headings = {}
    for name, rotation in global_from_room.items():
        relative = reference.T @ rotation
        yaw = np.arctan2(relative[1, 0], relative[0, 0])
        headings[name] = float(np.round(yaw / (np.pi / 2)) * (np.pi / 2))
    return headings


def plan_from_photo_sets(folders: dict, capture_id: str, profile, source_app: str,
                         extra_notes: list[str] | None = None, orientations: dict | None = None) -> "PhotoResult":
    """Per-room photo sets (from folders, or from the room turns found in a video) ->
    reconstruct each room -> stitch by verified doorways -> plan. orientations (video tier):
    per room, each photo's camera-to-common-frame rotation."""

    from .. import model
    from ..build import OpeningGeometry, RoomGeometry, build_plan
    from ..damage.stage import analyse_room
    from ..geometry.stitch import stitch
    from ..log import log

    start = time.perf_counter()
    reconstructions, index = {}, 0
    for name, photos in folders.items():
        reconstructions[name] = reconstruct_room(name, photos, first_index=index)
        index += len(photos)
        log("rooms", f"{name}: {reconstructions[name].notes[0]}")
    timing = {"reconstruct": round(time.perf_counter() - start, 2)}

    usable = {n: r for n, r in reconstructions.items() if r.layout is not None}
    start = time.perf_counter()
    headings = room_headings(usable, orientations) if orientations else None
    if headings:
        log("stitch", "room headings from the video: " + ", ".join(f"{n} {np.degrees(h):.0f}" for n, h in headings.items()))
    stitched = stitch({n: (r.layout, r.openings, r.cloud) for n, r in usable.items()}, headings)
    timing["stitch"] = round(time.perf_counter() - start, 2)

    partner = {}
    for link in stitched.links:
        partner[(link.a.room, link.a.index)] = (link.b.room, link.b.index)
        partner[(link.b.room, link.b.index)] = (link.a.room, link.a.index)

    start = time.perf_counter()
    geometries = []
    for name, rec in usable.items():
        transform = stitched.transforms[name]
        layout = rec.layout
        R = transform.R
        normals = np.array([R @ line.normal for line in layout.lines])
        offsets = np.array([line.offset + (R @ line.normal) @ transform.t for line in layout.lines])
        sigmas = np.array([np.hypot(line.sigma, 0.25 * line.spread) for line in layout.lines])
        neighbours = {}
        converted = []
        mirrors = [o for o in rec.openings if o.kind == "mirror"]
        for k, o in enumerate(rec.openings):
            if o.kind == "mirror":
                continue
            other = partner.get((name, k))
            if other:
                neighbours.setdefault(o.wall, []).append(other[0])
            converted.append(OpeningGeometry(o.wall, o.kind, o.u0, o.u1, o.v0, o.v1, o.sigma_u0, o.sigma_u1,
                                             o.sigma_v0, o.sigma_v1, o.confidence, other[0] if other else None, other))
        damage = analyse_room(layout, [o for o in rec.openings if o.kind != "mirror"], mirrors, rec.frames, rec.poses,
                              name, neighbours)
        for key in ("floor", "ceiling"):
            for det in damage.detections.get(key, []):
                det.polygon_uv = transform.apply(det.polygon_uv)
        geometries.append(RoomGeometry(name, transform.apply(layout.polygon), normals, offsets, sigmas,
                                       layout.floor_z, layout.ceiling_z, layout.floor_sigma, layout.ceiling_sigma,
                                       converted, notes=rec.notes + layout.notes, damage=damage, label=name,
                                       scale_sigma=rec.scale_sigma))
    timing["damage"] = round(time.perf_counter() - start, 2)

    failed = [n for n, r in reconstructions.items() if r.layout is None]
    drift = model.DriftReport(
        "rooms reconstructed independently (no long trajectory to drift); placed by verified doorways",
        False, notes=stitched.notes)
    plan = build_plan(capture_id, profile, geometries, drift, source_app=source_app,
                      frames_used=index, capture_notes=(extra_notes or [])
                      + ([f"no layout recovered for: {', '.join(failed)}"] if failed else []) + stitched.notes)
    plan.timing_s = timing
    return PhotoResult(plan, reconstructions, stitched)


@dataclass
class PhotoResult:
    plan: object
    reconstructions: dict
    stitched: object
