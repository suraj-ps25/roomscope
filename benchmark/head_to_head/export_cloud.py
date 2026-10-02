"""Export a real LiDAR recording as a coloured point cloud for a third-party floor-plan tool.

  python benchmark/head_to_head/export_cloud.py 47429914 --out runs/head_to_head/47429914.ply

The cloud is the recording's raw input as any tool would get it: LiDAR depth (confidence
2 only), placed with the device's own ARKit poses, with none of our drift correction or
depth calibration. Each 1.5 cm voxel takes its mean position and colour. Metres, z up
(ARKitScenes' world is gravity-aligned). Binary PLY with x, y, z, red, green, blue.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data" / "public" / "arkitscenes"
VOXEL_M = 0.015
FRAMES_PER_SECOND = 6.0
MAX_DEPTH_M = 4.0


def coloured_points(frame) -> tuple[np.ndarray, np.ndarray]:
    depth = frame.depth().astype(np.float64)
    confidence = frame.confidence()
    rows, cols = np.nonzero((depth > 0.15) & (depth < MAX_DEPTH_M) & (confidence >= 2))
    z = depth[rows, cols]
    K = frame.depth_K
    cam = np.stack([(cols - K[0, 2]) / K[0, 0] * z, (rows - K[1, 2]) / K[1, 1] * z, z], axis=1)
    colour = np.full((len(cam), 3), 128, dtype=np.uint8)
    if frame.load_rgb is not None:
        rgb = frame.rgb()
        Kc = frame.K
        u = np.clip((Kc[0, 0] * cam[:, 0] / z + Kc[0, 2]).astype(int), 0, rgb.shape[1] - 1)
        v = np.clip((Kc[1, 1] * cam[:, 1] / z + Kc[1, 2]).astype(int), 0, rgb.shape[0] - 1)
        colour = rgb[v, u]
    world = cam @ frame.pose[:3, :3].T + frame.pose[:3, 3]
    return world, colour


def export(video: str, out: Path) -> int:
    from roomscope.io.arkitscenes import read_arkitscenes
    from roomscope.tiers.lidar import select_keyframes

    frames = select_keyframes(read_arkitscenes(DATA / video).frames, FRAMES_PER_SECOND)
    sums: dict[int, np.ndarray] = {}
    keys_all, xyz_all, rgb_all = [], [], []
    for frame in frames:
        xyz, rgb = coloured_points(frame)
        xyz_all.append(xyz)
        rgb_all.append(rgb.astype(np.float64))
    xyz = np.concatenate(xyz_all)
    rgb = np.concatenate(rgb_all)
    cells = np.floor(xyz / VOXEL_M).astype(np.int64)
    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    n = len(counts)
    mean_xyz = np.stack([np.bincount(inverse, xyz[:, k], n) for k in range(3)], axis=1) / counts[:, None]
    mean_rgb = np.stack([np.bincount(inverse, rgb[:, k], n) for k in range(3)], axis=1) / counts[:, None]
    keep = counts >= 2   # a voxel seen once is more often noise than surface
    mean_xyz, mean_rgb = mean_xyz[keep], np.clip(mean_rgb[keep], 0, 255).astype(np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    vertex = np.empty(len(mean_xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                            ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = mean_xyz[:, 0], mean_xyz[:, 1], mean_xyz[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"comment ARKitScenes recording {video}: LiDAR depth on ARKit poses, metres, z up\n"
              f"element vertex {len(vertex)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with out.open("wb") as handle:
        handle.write(header.encode())
        handle.write(vertex.tobytes())
    return len(vertex)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    count = export(args.recording, Path(args.out))
    print(f"{args.out}: {count} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
