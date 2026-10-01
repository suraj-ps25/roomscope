"""Video tier: one handheld walkthrough clip -> stitched, dimensioned plan.

  decode + motion keyframes -> MoGe-2 metric depth per keyframe
  -> MapAnything over overlapping windows, conditioned on that depth
  -> windows chained through their shared keyframes (rigid fit on camera centres)
  -> gravity from geometry -> pseudo-LiDAR frames -> the LiDAR tier's core with video
     noise settings (pose graph, plane/jamb adjustment, rooms, walls, openings)

Scale sigma: MoGe-2's 6.9% per-view spread averaged over the keyframes that saw each
room, plus a 2% bias budget, floored at 1.5% for correlated error across a clip.
"""

from __future__ import annotations

import numpy as np

from ..build import TierProfile
from ..capture import CaptureBundle, Frame
from ..geometry.drift import VIDEO_DRIFT
from ..geometry.planes import VIDEO_PLANES
from ..geometry.tolerances import VIDEO_TOL
from ..io.video import intrinsics_from_focal, read_video, select_keyframes
from ..log import log
from .lidar import LidarOptions, LidarResult, run_lidar
from .photo import MOGE_VIEW_SPREAD, SCALE_BIAS_BUDGET, _confidence_levels, gravity_rotation

WINDOW = 20
OVERLAP = 6
SCALE_FLOOR = 0.015
DEPTH_EVERY = 3


def _mean_transform(transforms: list[np.ndarray]) -> np.ndarray:
    """Chordal mean of rotations plus mean translation. Uses full poses, so it stays
    well-posed when the shared keyframes come from an on-the-spot turn (camera centres
    coincide and a fit on positions alone could not fix the rotation)."""
    u, _, vt = np.linalg.svd(np.sum([t[:3, :3] for t in transforms], axis=0))
    rot = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = np.mean([t[:3, 3] for t in transforms], axis=0)
    return out


def _chain_windows(images, intrinsics, depths):
    from ..log import log
    from ..models.multiview import reconstruct

    n = len(images)
    starts = list(range(0, max(n - OVERLAP, 1), WINDOW - OVERLAP))
    poses = [None] * n
    view_depth = [None] * n
    view_mask = [None] * n
    view_conf = [None] * n
    view_K = [None] * n
    for start in starts:
        end = min(start + WINDOW, n)
        views = reconstruct(images[start:end], intrinsics[start:end], depths[start:end])
        log("multiview", f"window {starts.index(start) + 1}/{len(starts)} (keyframes {start}-{end - 1})")
        local = [v.pose for v in views]
        shared = [k for k in range(start, end) if poses[k] is not None]
        to_global = _mean_transform([poses[k] @ np.linalg.inv(local[k - start]) for k in shared]) if shared else np.eye(4)
        for k in range(start, end):
            if poses[k] is None:
                poses[k] = to_global @ local[k - start]
                v = views[k - start]
                view_depth[k], view_mask[k], view_conf[k], view_K[k] = v.depth, v.mask, v.confidence, v.K
        if end == n:
            break
    return poses, view_depth, view_mask, view_conf, view_K


def run_video(path, drift_correction: bool = True) -> LidarResult:
    from ..models.depth import metric_depth

    video = read_video(path)
    log("ingest", f"decoded {len(video.images)} frames")
    keys = select_keyframes(video.images, video.timestamps)
    images = [video.images[k] for k in keys]
    stamps = video.timestamps[keys]
    decoded = len(video.images)
    video.images = []
    log("ingest", f"{len(keys)} keyframes")
    height, width = images[0].shape[:2]
    K = intrinsics_from_focal(video.focal_35mm, video.source_size, (width, height)) if video.focal_35mm else None
    fov = None if K is None else float(np.degrees(2 * np.arctan(width / (2 * K[0, 0]))))
    # Metric depth on every DEPTH_EVERY-th keyframe anchors scale; the rest go in as images.
    metric = [metric_depth(image, fov) if k % DEPTH_EVERY == 0 else None for k, image in enumerate(images)]
    anchored = [m for m in metric if m is not None]
    log("depth", f"MoGe-2 metric depth on {len(anchored)} anchor keyframes")
    if K is None:
        # One lens for the whole clip: take the median of MoGe's per-frame estimates.
        fx = float(np.median([m.K[0, 0] for m in anchored]))
        fy = float(np.median([m.K[1, 1] for m in anchored]))
        K = np.array([[fx, 0, (width - 1) / 2], [0, fy, (height - 1) / 2], [0, 0, 1.0]])
    poses, depth, mask, conf, view_K = _chain_windows(images, [K] * len(images),
                                                      [None if m is None else m.depth for m in metric])

    from ..models import depth as depth_model, multiview
    depth_model.release()
    multiview.release()
    log("multiview", "poses chained over all windows; models released")
    ratios = [float(np.median(m.depth[v & m.mask & (d > 0.1)] / d[v & m.mask & (d > 0.1)]))
              for m, d, v in zip(metric, depth, mask) if m is not None and (v & m.mask & (d > 0.1)).sum() > 500]
    correction = float(np.median(ratios)) if ratios else 1.0

    frames = []
    for k, (image, pose) in enumerate(zip(images, poses)):
        scaled = pose.copy()
        scaled[:3, 3] *= correction
        d = np.where(mask[k], depth[k] * correction, 0.0).astype(np.float32)
        levels = _confidence_levels(conf[k], mask[k])
        frames.append(Frame(k, float(stamps[k]), view_K[k], (width, height), scaled, None,
                            load_rgb=(lambda im=image: im), load_depth=(lambda dd=d: dd),
                            load_confidence=(lambda c=levels: c), depth_K=view_K[k], depth_size=(width, height)))

    from ..geometry.cloud import fuse
    raw = fuse(frames[:: max(1, len(frames) // 60)], {f.index: f.pose for f in frames}, voxel=0.04, stride=2, min_confidence=1)
    gravity = np.eye(4)
    gravity[:3, :3] = gravity_rotation(raw.normals, [f.pose for f in frames])
    for frame in frames:
        frame.pose = gravity @ frame.pose

    per_room_views = max(len(anchored) / 4, 1)
    scale_sigma = max(float(np.hypot(MOGE_VIEW_SPREAD / np.sqrt(per_room_views), SCALE_BIAS_BUDGET)), SCALE_FLOOR)
    profile = TierProfile("video", scale_sigma, "video multi-view reconstruction, MoGe-2 metric scale")
    bundle = CaptureBundle(str(getattr(path, "stem", path)).split("/")[-1], "video", frames, source_app="Camera (video)",
                           device=video.device,
                           notes=[f"{len(keys)} keyframes from {decoded} decoded frames, scale correction "
                                  f"{correction:.3f}, focal {'from metadata' if video.focal_35mm else 'estimated'}"])
    options = LidarOptions(drift_correction=drift_correction, profile=profile, tol=VIDEO_TOL, drift=VIDEO_DRIFT,
                           planes=VIDEO_PLANES, measure_confidence=1, select_keyframes=False)
    return run_lidar(bundle, options)
