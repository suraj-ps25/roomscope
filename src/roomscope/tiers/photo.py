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
from ..geometry.openings import Opening, detect_openings
from ..geometry.rooms import RoomRegion, align, segment_rooms
from ..geometry.tolerances import PHOTO_TOL
from ..io.photos import Photo

MOGE_VIEW_SPREAD = 0.069
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
        frames.append(Frame(first_index + k, float(k), view.K, (width, height), pose, room,
                            load_rgb=(lambda im=photo.image: im), load_depth=(lambda d=depth: d),
                            load_confidence=(lambda c=confidence: c), depth_K=view.K, depth_size=(width, height)))

    raw = fuse(frames, {f.index: f.pose for f in frames}, voxel=PHOTO_VOXEL, stride=1, min_confidence=1)
    gravity = np.eye(4)
    gravity[:3, :3] = gravity_rotation(raw.normals, [f.pose for f in frames])
    poses = {f.index: gravity @ f.pose for f in frames}
    cloud = fuse(frames, poses, voxel=PHOTO_VOXEL, stride=1, min_confidence=1)
    alignment = align(cloud)
    cloud = alignment.apply(cloud)
    poses = {k: alignment.pose(v) for k, v in poses.items()}

    notes = [f"{len(photos)} photos, scale correction {correction:.3f}"
             + (f", per-view ratio spread {np.std(ratios):.1%}" if len(ratios) > 1 else "")]
    regions, seg_notes = segment_rooms(cloud, frames, poses, min_frames_inside=PHOTO_TOL.min_room_frames)
    notes += seg_notes
    region = max(regions, key=lambda r: (r.frames_inside, r.area)) if regions else None
    layout = openings = None
    if region is not None:
        region.id = room
        layout = room_layout(region, cloud, PHOTO_TOL)
    if layout is not None:
        openings, opening_notes = detect_openings(layout, region, frames, poses, PHOTO_TOL)
        notes += opening_notes
    else:
        notes.append("room layout could not be recovered from these photos")
    return RoomReconstruction(room, frames, poses, region, layout, openings or [], scale_sigma(len(photos)),
                              cloud, notes, timing)
