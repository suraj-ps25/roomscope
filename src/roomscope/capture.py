"""Normalised capture bundle: what every tier's ingest produces and every stage consumes.

Conventions: camera OpenCV (x right, y down, z forward); poses camera-to-world 4x4; world
z-up (gravity), metres. Depth is z-depth in metres at depth resolution, 0 = invalid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Frame:
    index: int
    timestamp: float
    K: np.ndarray
    image_size: tuple[int, int]
    pose: np.ndarray | None = None
    group: str | None = None
    load_rgb: Callable[[], np.ndarray] | None = None
    load_depth: Callable[[], np.ndarray] | None = None
    load_confidence: Callable[[], np.ndarray] | None = None
    depth_K: np.ndarray | None = None
    depth_size: tuple[int, int] | None = None

    def rgb(self) -> np.ndarray:
        return self.load_rgb()

    def depth(self) -> np.ndarray | None:
        return None if self.load_depth is None else self.load_depth()

    def confidence(self) -> np.ndarray | None:
        return None if self.load_confidence is None else self.load_confidence()


@dataclass
class CaptureBundle:
    capture_id: str
    tier: str
    frames: list[Frame]
    source_app: str | None = None
    device: str | None = None
    notes: list[str] = field(default_factory=list)
    prefetch_rgb: Callable[[list[Frame]], None] | None = None

    def warm_rgb(self, frames: list[Frame]) -> None:
        """Decode a batch of RGB frames up front (video sources seek slowly one by one)."""
        if self.prefetch_rgb is not None:
            self.prefetch_rgb(frames)

    @property
    def has_depth(self) -> bool:
        return any(f.load_depth is not None for f in self.frames)

    @property
    def has_poses(self) -> bool:
        return all(f.pose is not None for f in self.frames)

    def groups(self) -> dict[str, list[Frame]]:
        grouped: dict[str, list[Frame]] = {}
        for frame in self.frames:
            grouped.setdefault(frame.group or "all", []).append(frame)
        return grouped


# ARKit camera axes are x right, y up, z backward; OpenCV's are x right, y down, z forward.
ARKIT_TO_OPENCV_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])
# ARKit world is y-up; ours is z-up. Rotate +90 deg about x: (x, y, z) -> (x, -z, y).
ARKIT_TO_ZUP_WORLD = np.array([[1.0, 0, 0, 0], [0, 0, -1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]])


def arkit_pose_to_internal(arkit_camera_to_world: np.ndarray) -> np.ndarray:
    return ARKIT_TO_ZUP_WORLD @ arkit_camera_to_world @ ARKIT_TO_OPENCV_CAMERA


def internal_pose_to_arkit(pose: np.ndarray) -> np.ndarray:
    return np.linalg.inv(ARKIT_TO_ZUP_WORLD) @ pose @ np.linalg.inv(ARKIT_TO_OPENCV_CAMERA)


# Stray Scanner already rotates the camera axes to OpenCV before writing odometry.csv
# (q_WA * q_AC with q_AC a 180 deg turn about x); only its world stays ARKit y-up.
def stray_pose_to_internal(stray_camera_to_world: np.ndarray) -> np.ndarray:
    return ARKIT_TO_ZUP_WORLD @ stray_camera_to_world


def internal_pose_to_stray(pose: np.ndarray) -> np.ndarray:
    return np.linalg.inv(ARKIT_TO_ZUP_WORLD) @ pose
