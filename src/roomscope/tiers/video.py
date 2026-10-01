"""Video tier: one handheld walkthrough clip -> stitched, dimensioned plan.

The capture protocol starts every room with a slow on-the-spot turn. A turn is a ready-made
panorama: many views from one point, so rotation is well-conditioned and MoGe-2 gives
metric depth per view regardless of wall texture. So the video tier is:

  decode -> per-frame rotation (homography on tracked features) -> find the turns
  -> per turn: ~10 frames evenly spaced in yaw = that room's photo set
  -> the photo tier's per-room reconstruction (with its multi-view consistency check)
  -> stitch rooms by verified doorways -> plan

Why not chain the whole clip: MapAnything over overlapping windows of walking keyframes
gave ATE of 10-57 cm on real ARKitScenes video and broke on the synthetic walk, while turns
are found reliably (flat_a: 4/4 at the right times, no false positives) and reconstruct
like photos. With no long trajectory there is no accumulated drift to correct.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from ..build import TierProfile
from ..io.photos import Photo
from ..io.video import (STANDING_STILL_PX, find_spins, focal_from_rotation, intrinsics_from_focal, read_video,
                        turn_rotations, turn_views, yaw_steps, clip_rotations)
from ..log import log
from .photo import PhotoResult, plan_from_photo_sets

VIEWS_PER_REVOLUTION = 14
VIEWS_PER_TURN = 10
VIDEO_PROFILE = TierProfile("video", 0.02, "video: rooms from on-the-spot turns, MoGe-2 + MapAnything")


def _clip(path) -> Path:
    path = Path(path)
    if path.is_dir():
        from ..io.detect import video_files
        return video_files(path)[0]
    return path


def _full_resolution(clip: Path, times: list[float]) -> dict[float, np.ndarray]:
    """Re-decode just the chosen views at full resolution (damage needs the pixels)."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", str(clip)], capture_output=True, text=True)
    width, height = (int(v) for v in probe.stdout.strip().split(",")[:2])
    frames = {}
    for t in times:
        out = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(clip), "-frames:v", "1",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
        pixels = len(out.stdout) // 3
        if pixels == width * height:
            frames[t] = np.frombuffer(out.stdout, np.uint8).reshape(height, width, 3)
        elif pixels:
            # Rotated clips decode with width and height swapped.
            frames[t] = np.frombuffer(out.stdout, np.uint8).reshape(width, height, 3)
    return frames


def _intrinsics(video, size: tuple[int, int]) -> tuple[np.ndarray, float | None]:
    width, height = size
    if video.focal_35mm:
        K = intrinsics_from_focal(video.focal_35mm, video.source_size, size)
        return K, float(np.degrees(2 * np.arctan(width / (2 * K[0, 0]))))
    # One lens for the whole clip: the median of MoGe-2's estimates on a few frames.
    from ..models.depth import metric_depth
    estimates = [metric_depth(video.images[k], None).K for k in np.linspace(0, len(video.images) - 1, 6).astype(int)]
    fx = float(np.median([e[0, 0] for e in estimates]))
    fy = float(np.median([e[1, 1] for e in estimates]))
    K = np.array([[fx, 0, (width - 1) / 2], [0, fy, (height - 1) / 2], [0, 0, 1.0]])
    return K, float(np.degrees(2 * np.arctan(width / (2 * fx))))


def run_video(path) -> PhotoResult:
    clip = _clip(path)
    video = read_video(clip)
    height, width = video.images[0].shape[:2]
    log("ingest", f"decoded {len(video.images)} frames at {width}x{height}")
    K, fov = _intrinsics(video, (width, height))
    # Decoding resizes to patch-multiple sizes, so pixels are not exactly square any more.
    aspect = (height / video.source_size[1]) / (width / video.source_size[0])
    homographies = {}
    steps, residuals = yaw_steps(video.images, K, homographies)
    turns = find_spins(steps, video.timestamps, residuals)
    log("rooms", f"{len(turns)} on-the-spot turns found: " +
        ", ".join(f"{video.timestamps[a]:.0f}-{video.timestamps[b]:.0f}s" for a, b in turns))
    notes = [f"{len(turns)} room turns found in a {video.timestamps[-1]:.0f} s clip"]
    if not video.focal_35mm and turns:
        # Without a lens tag, the turns themselves calibrate the focal; metric depth scales
        # with it, so a monocular estimate's few-percent focal error would land on every wall.
        focals = [focal_from_rotation([homographies[k] for k in range(a + 1, b + 1)
                                       if k in homographies and residuals[k] <= STANDING_STILL_PX], K, aspect)
                  for a, b in turns]
        focals = [f for f in focals if f]
        if focals:
            fx = float(np.median(focals))
            log("rooms", f"focal from the turns: {fx:.1f} px (monocular estimate {K[0, 0]:.1f} px)")
            notes.append(f"focal self-calibrated from {len(focals)} turns: {fx:.1f} px at {width} px width")
            K = np.array([[fx, 0, K[0, 2]], [0, fx * aspect, K[1, 2]], [0, 0, 1.0]])
            fov = float(np.degrees(2 * np.arctan(width / (2 * fx))))
    if not turns:
        notes.append("no on-the-spot turns found (capture protocol not followed): the clip is treated as one room")
        chosen = [list(np.linspace(0, len(video.images) - 1, VIEWS_PER_TURN).astype(int))]
    else:
        chosen = []
    rotations = {}
    for turn in turns:
        turn_chain = turn_rotations(video.images, turn, K)
        rotations.update(turn_chain)
        chosen.append(turn_views(turn_chain, VIEWS_PER_REVOLUTION))

    times = sorted({float(video.timestamps[k]) for picks in chosen for k in picks})
    full = _full_resolution(clip, times)
    rooms = {}
    for number, picks in enumerate(chosen, start=1):
        name = f"room_{number}"
        rooms[name] = [Photo(clip, name, video.images[k], K, fov, full.get(float(video.timestamps[k]), video.images[k]),
                             video.device, rotations.get(k), float(video.timestamps[k])) for k in picks]
    compass = clip_rotations(clip, K, (width, height), video.timestamps)
    orientations = {f"room_{number}": [compass[k] for k in picks] for number, picks in enumerate(chosen, start=1)}
    return plan_from_photo_sets(rooms, clip.stem, VIDEO_PROFILE, "Camera (video)", notes, orientations)
