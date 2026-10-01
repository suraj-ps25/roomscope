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
    cameras' mean down axis is a good start; floor and ceiling normals then refine it."""
    up = -np.mean([pose[:3, 1] for pose in poses], axis=0)
    up /= np.linalg.norm(up)
    for threshold in (0.8, 0.9, 0.95):
        near = np.abs(normals @ up) > threshold
        if near.sum() < 200:
            break
        signed = normals[near] * np.sign(normals[near] @ up)[:, None]
        up = signed.mean(axis=0)
        up /= np.linalg.norm(up)
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
    metric = [metric_depth(p.image, p.fov_x_deg) for p in photos]
    intrinsics = [p.K if p.K is not None else m.K for p, m in zip(photos, metric)]
    timing["depth"] = round(time.perf_counter() - start, 2)
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
    raw = fuse(frames, {f.index: f.pose for f in frames}, voxel=PHOTO_VOXEL, stride=1, min_confidence=1)
    gravity = np.eye(4)
    gravity[:3, :3] = gravity_rotation(raw.normals, [f.pose for f in frames])
    poses = {f.index: gravity @ f.pose for f in frames}
    cloud = fuse(frames, poses, voxel=PHOTO_VOXEL, stride=1, min_confidence=1)
    alignment = align(cloud)
    cloud = alignment.apply(cloud)
    poses = {k: alignment.pose(v) for k, v in poses.items()}

    notes = [f"{len(photos)} photos, scale correction {correction:.3f}"
             + (f", per-view ratio spread {np.std(ratios):.1%}" if len(ratios) > 1 else "")
             + f", multi-view agreement {agreement:.0%}"]
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
    from ..build import PHOTO, OpeningGeometry, RoomGeometry, build_plan
    from ..damage.stage import analyse_room
    from ..geometry.stitch import stitch
    from ..io.photos import read_photo_folders
    from .. import model
    from pathlib import Path

    start = time.perf_counter()
    folders = read_photo_folders(root)
    reconstructions, index = {}, 0
    for name, photos in folders.items():
        reconstructions[name] = reconstruct_room(name, photos, first_index=index)
        index += len(photos)
    timing = {"reconstruct": round(time.perf_counter() - start, 2)}

    usable = {n: r for n, r in reconstructions.items() if r.layout is not None}
    start = time.perf_counter()
    stitched = stitch({n: (r.layout, r.openings, r.cloud) for n, r in usable.items()})
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
        for k, o in enumerate(rec.openings):
            other = partner.get((name, k))
            if other:
                neighbours.setdefault(o.wall, []).append(other[0])
            converted.append(OpeningGeometry(o.wall, o.kind, o.u0, o.u1, o.v0, o.v1, o.sigma_u0, o.sigma_u1,
                                             o.sigma_v0, o.sigma_v1, o.confidence, other[0] if other else None, other))
        damage = analyse_room(layout, rec.openings, [], rec.frames, rec.poses, name, neighbours)
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
        "photo tier: rooms reconstructed independently (no trajectory to drift); placed by verified doorways",
        False, notes=stitched.notes)
    plan = build_plan(Path(root).name, PHOTO, geometries, drift, source_app="Camera (photos)",
                      frames_used=index, capture_notes=([f"no layout recovered for: {', '.join(failed)}"] if failed else [])
                      + stitched.notes)
    plan.timing_s = timing
    return PhotoResult(plan, reconstructions, stitched)


@dataclass
class PhotoResult:
    plan: object
    reconstructions: dict
    stitched: object
