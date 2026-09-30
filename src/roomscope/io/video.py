"""Video-tier ingest: one handheld walkthrough clip (any iPhone 15+, Camera app).

Frames are decoded by ffmpeg (which applies the clip's rotation tag) at DECODE_FPS and
the working resolution. Keyframes are chosen by motion, from sparse optical flow: a new
keyframe once the view has shifted KEYFRAME_SHIFT of its width, so a slow on-the-spot turn
still gets overlapping keyframes while standing still adds none.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .photos import FULL_FRAME_DIAGONAL_MM, processing_size

DECODE_FPS = 5.0
KEYFRAME_SHIFT = 0.25
MAX_KEYFRAME_GAP_S = 2.0
MIN_KEYFRAME_GAP_S = 0.2


@dataclass
class VideoFrames:
    images: list[np.ndarray]
    timestamps: np.ndarray
    source_size: tuple[int, int]
    focal_35mm: float | None
    device: str | None


def _probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _metadata(info: dict) -> tuple[tuple[int, int], float | None, str | None]:
    stream = next(s for s in info["streams"] if s.get("codec_type") == "video")
    width, height = int(stream["width"]), int(stream["height"])
    rotation = 0
    for side in stream.get("side_data_list", []):
        rotation = int(side.get("rotation", rotation) or rotation)
    rotation = int(stream.get("tags", {}).get("rotate", rotation) or rotation)
    if abs(rotation) % 180 == 90:
        width, height = height, width
    tags = {k.lower(): v for k, v in {**info.get("format", {}).get("tags", {}), **stream.get("tags", {})}.items()}
    focal = next((float(v) for k, v in tags.items() if "focal_length" in k and "35" in k), None)
    device = tags.get("com.apple.quicktime.model")
    return (width, height), focal, device


def read_video(path: str | Path) -> VideoFrames:
    path = Path(path)
    if path.is_dir():
        from .detect import video_files
        path = video_files(path)[0]
    (width, height), focal, device = _metadata(_probe(path))
    w, h = processing_size(width, height)
    decode = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"fps={DECODE_FPS},scale={w}:{h}:flags=area",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    raw = decode.stdout.read()
    decode.wait()
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w, 3)
    timestamps = np.arange(len(frames)) / DECODE_FPS
    return VideoFrames(list(frames), timestamps, (width, height), focal, device)


KEYFRAME_BUDGET = 160


def _frame_shifts(images: list[np.ndarray]) -> np.ndarray:
    """Median sparse optical-flow displacement between consecutive frames, as a fraction
    of the image width."""
    shifts = np.zeros(len(images))
    width = images[0].shape[1]
    grey_prev = cv2.cvtColor(images[0], cv2.COLOR_RGB2GRAY)
    for k in range(1, len(images)):
        grey = cv2.cvtColor(images[k], cv2.COLOR_RGB2GRAY)
        points = cv2.goodFeaturesToTrack(grey_prev, 300, 0.01, 8)
        if points is not None:
            moved, status, _ = cv2.calcOpticalFlowPyrLK(grey_prev, grey, points, None)
            ok = status.ravel() == 1
            if ok.sum() > 10:
                shifts[k] = float(np.median(np.linalg.norm((moved - points)[ok].reshape(-1, 2), axis=1))) / width
        grey_prev = grey
    return shifts


def _pick(shifts: np.ndarray, timestamps: np.ndarray, threshold: float) -> list[int]:
    chosen, accumulated = [0], 0.0
    for k in range(1, len(shifts)):
        accumulated += shifts[k]
        gap = timestamps[k] - timestamps[chosen[-1]]
        if (accumulated >= threshold and gap >= MIN_KEYFRAME_GAP_S) or gap >= MAX_KEYFRAME_GAP_S:
            chosen.append(k)
            accumulated = 0.0
    return chosen


def select_keyframes(images: list[np.ndarray], timestamps: np.ndarray, budget: int = KEYFRAME_BUDGET) -> list[int]:
    """Motion-based keyframes, with the shift threshold raised until the count fits the
    budget, so runtime stays predictable however much the walker sweeps the phone."""
    shifts = _frame_shifts(images)
    threshold = KEYFRAME_SHIFT
    chosen = _pick(shifts, timestamps, threshold)
    while len(chosen) > budget and threshold < 2.0:
        threshold *= 1.2
        chosen = _pick(shifts, timestamps, threshold)
    return chosen


def intrinsics_from_focal(focal_35mm: float, source_size: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
    width, height = source_size
    w, h = size
    fx = focal_35mm * np.hypot(width, height) / FULL_FRAME_DIAGONAL_MM
    return np.array([[fx * w / width, 0, (w - 1) / 2], [0, fx * h / height, (h - 1) / 2], [0, 0, 1.0]])
