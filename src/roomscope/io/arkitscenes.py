"""ARKitScenes raw reader (Apple; real iPad Pro LiDAR, used for development and validation).

  lowres_depth/<vid>_<ts>.png      uint16 mm, 256x192, ~60 Hz
  confidence/<vid>_<ts>.png        uint8 0..2
  lowres_wide_intrinsics/*.pincam  "w h fx fy cx cy"
  lowres_wide.traj                 "ts ax ay az tx ty tz" per ~10 Hz, world->camera
                                   (axis-angle); the reference loader inverts it
Frames are kept where a trajectory sample exists. The ARKitScenes world is already
gravity-aligned with +z up: checked on 47429912, where the cameras' image-up axis (-x for
sky_direction=Left) averages to world +z (0.99). Counting surface normals is not a safe
substitute: in a close-range scan walls outnumber floor and win.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ..capture import CaptureBundle, Frame


def is_arkitscenes(path: Path) -> bool:
    return (path / "lowres_wide.traj").exists() and (path / "lowres_depth").is_dir()


def _stamp(path: Path) -> float:
    return float(path.stem.split("_")[-1])


def _pincam(path: Path) -> np.ndarray:
    _, _, fx, fy, cx, cy = map(float, path.read_text().split())
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def read_arkitscenes(path: str | Path) -> CaptureBundle:
    root = Path(path)
    traj = np.loadtxt(root / "lowres_wide.traj")
    depth_files = sorted((root / "lowres_depth").glob("*.png"), key=_stamp)
    stamps = np.array([_stamp(p) for p in depth_files])
    intrinsics = {round(_stamp(p), 3): p for p in (root / "lowres_wide_intrinsics").glob("*.pincam")}
    # Optional 640x480 colour stream (vga_wide), matched by timestamp, for damage.
    rgb_files = sorted((root / "vga_wide").glob("*.png"), key=_stamp) if (root / "vga_wide").is_dir() else []
    rgb_stamps = np.array([_stamp(p) for p in rgb_files])
    rgb_K_files = {round(_stamp(p), 3): p for p in (root / "vga_wide_intrinsics").glob("*.pincam")} \
        if (root / "vga_wide_intrinsics").is_dir() else {}
    frames = []
    for row in traj:
        k = int(np.argmin(np.abs(stamps - row[0])))
        if abs(stamps[k] - row[0]) > 1 / 90:
            continue
        depth_path = depth_files[k]
        conf_path = root / "confidence" / depth_path.name
        K_path = intrinsics.get(round(stamps[k], 3))
        if K_path is None or not conf_path.exists():
            continue
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = Rotation.from_rotvec(row[1:4]).as_matrix()
        world_to_camera[:3, 3] = row[4:7]
        K = _pincam(K_path)
        rgb_K, rgb_size, load_rgb = K, (256, 192), None
        if len(rgb_files):
            r = int(np.argmin(np.abs(rgb_stamps - row[0])))
            rgb_K_path = rgb_K_files.get(round(rgb_stamps[r], 3))
            if abs(rgb_stamps[r] - row[0]) < 1 / 50 and rgb_K_path is not None:
                rgb_K, rgb_size = _pincam(rgb_K_path), (640, 480)
                load_rgb = (lambda p=rgb_files[r]: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
        frames.append(Frame(
            index=len(frames), timestamp=float(row[0]), K=rgb_K, image_size=rgb_size, load_rgb=load_rgb,
            pose=np.linalg.inv(world_to_camera), depth_K=K, depth_size=(256, 192),
            load_depth=(lambda p=depth_path: cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0),
            load_confidence=(lambda p=conf_path: cv2.imread(str(p), cv2.IMREAD_UNCHANGED))))
    if not frames:
        raise ValueError(f"{root}: no depth frames aligned with the trajectory")
    t0 = frames[0].timestamp
    for frame in frames:
        frame.timestamp -= t0
    return CaptureBundle(root.name, "lidar", frames, source_app="ARKitScenes (iPad Pro LiDAR)",
                         notes=[f"{len(frames)} frames with poses"])
