"""Stray Scanner (iOS, free, MIT) capture reader.

Layout (docs/format.md in strayrobots/scanner):
  odometry.csv      timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, ...
  camera_matrix.csv 3x3 K at RGB resolution (last frame; per-frame K is in odometry.csv)
  depth/NNNNNN.png  uint16 millimetres, 256x192
  confidence/NNNNNN.png  uint8 0/1/2 (ARConfidenceLevel)
  rgb.mp4           HEVC, frames aligned with odometry rows
Poses are camera(OpenCV)-to-world with an ARKit y-up world.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ..capture import CaptureBundle, Frame, stray_pose_to_internal


def is_stray(path: Path) -> bool:
    return (path / "odometry.csv").exists() and (path / "depth").is_dir()


def _video_size(video: Path) -> tuple[int, int] | None:
    if not video.exists():
        return None
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(video)], capture_output=True, text=True)
    if probe.returncode != 0 or not probe.stdout.strip():
        return None
    width, height = (int(v) for v in probe.stdout.strip().split(",")[:2])
    return width, height


def _depth_file(folder: Path, frame: int) -> Path | None:
    for suffix in (".png", ".npy"):
        candidate = folder / f"{frame:06d}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _load_depth(path: Path) -> np.ndarray:
    raw = np.load(path) if path.suffix == ".npy" else cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return raw.astype(np.float32) / 1000.0


PREFETCH_ROWS_PER_PASS = 5000


class VideoFrames:
    """Decodes only the RGB frames the pipeline asks for, once, into a JPEG cache."""

    def __init__(self, video: Path, cache: Path):
        self.video = video
        self.cache = cache

    def path(self, row: int) -> Path:
        return self.cache / f"{row:06d}.jpg"

    def prefetch(self, rows: list[int]) -> None:
        missing = sorted({r for r in rows if not self.path(r).exists()})
        if not missing:
            return
        self.cache.mkdir(parents=True, exist_ok=True)
        # One pass for everything asked for: each pass decodes the video from its start, so
        # a few hundred rows in chunks cost a full decode per chunk (minutes on a long scan).
        for start in range(0, len(missing), PREFETCH_ROWS_PER_PASS):
            chunk = missing[start:start + PREFETCH_ROWS_PER_PASS]
            expr = "+".join(f"eq(n\\,{r})" for r in chunk)
            staging = self.cache / "staging"
            staging.mkdir(exist_ok=True)
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(self.video), "-vf", f"select='{expr}'",
                 "-vsync", "0", "-q:v", "2", str(staging / "%06d.jpg")], check=True)
            for produced, row in zip(sorted(staging.glob("*.jpg")), chunk):
                produced.rename(self.path(row))
            staging.rmdir()

    def load(self, row: int) -> np.ndarray:
        if not self.path(row).exists():
            self.prefetch([row])
        image = cv2.imread(str(self.path(row)), cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_stray(path: str | Path, cache_dir: str | Path | None = None) -> CaptureBundle:
    root = Path(path)
    with open(root / "odometry.csv", newline="") as handle:
        rows = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"{root}: odometry.csv has no rows")

    first_depth = next((p for p in (_depth_file(root / "depth", int(r["frame"])) for r in rows) if p), None)
    if first_depth is None:
        raise ValueError(f"{root}: no depth maps found")
    depth_h, depth_w = _load_depth(first_depth).shape
    video_size = _video_size(root / "rgb.mp4") or (1920, 1440)
    video = VideoFrames(root / "rgb.mp4", Path(cache_dir or root / ".roomscope_cache") / "rgb")

    frames = []
    notes = []
    row_of_frame: dict[int, int] = {}
    for row_index, row in enumerate(rows):
        frame_no = int(row["frame"])
        depth_path = _depth_file(root / "depth", frame_no)
        if depth_path is None:
            continue
        conf_path = root / "confidence" / f"{frame_no:06d}.png"
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat([float(row[k]) for k in ("qx", "qy", "qz", "qw")]).as_matrix()
        pose[:3, 3] = [float(row[k]) for k in ("x", "y", "z")]
        K = np.array([[float(row["fx"]), 0, float(row["cx"])], [0, float(row["fy"]), float(row["cy"])], [0, 0, 1.0]])
        depth_K = K.copy()
        depth_K[0] *= depth_w / video_size[0]
        depth_K[1] *= depth_h / video_size[1]
        frames.append(Frame(
            index=frame_no, timestamp=float(row["timestamp"]), K=K, image_size=video_size,
            pose=stray_pose_to_internal(pose), depth_K=depth_K, depth_size=(depth_w, depth_h),
            load_depth=(lambda p=depth_path: _load_depth(p)),
            load_confidence=(lambda p=conf_path: cv2.imread(str(p), cv2.IMREAD_UNCHANGED) if p.exists() else None),
            load_rgb=(lambda r=row_index: video.load(r))))
        row_of_frame[frame_no] = row_index
    if len(frames) < len(rows):
        notes.append(f"{len(rows) - len(frames)} odometry rows had no depth map and were skipped")
    return CaptureBundle(root.name, "lidar", frames, source_app="Stray Scanner", notes=notes,
                         prefetch_rgb=lambda wanted: video.prefetch([row_of_frame[f.index] for f in wanted]))
