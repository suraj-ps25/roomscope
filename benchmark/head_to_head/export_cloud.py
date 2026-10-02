"""Export a real LiDAR recording as a coloured point cloud for a third-party floor-plan tool.

  python benchmark/head_to_head/export_cloud.py 47429914 --out runs/head_to_head/47429914.ply
  python benchmark/head_to_head/export_cloud.py 47429914 --out runs/head_to_head/47429914.las


The cloud is the recording's raw input as any tool would get it: LiDAR depth (confidence
2 only), placed with the device's own ARKit poses, with none of our drift correction or
depth calibration. Each 1.5 cm voxel takes its mean position and colour. Metres, z up
(ARKitScenes' world is gravity-aligned). Binary PLY with x, y, z, red, green, blue, or,
for tools that take only LAS (Pointorama), LAS 1.2 point format 2 at 1 mm, written here
without a LAS library.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import struct

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data" / "public" / "arkitscenes"
VOXEL_M = 0.015
FRAMES_PER_SECOND = 6.0
MAX_DEPTH_M = 4.0
LAS_SCALE_M = 0.001


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
    if out.suffix.lower() == ".las":
        write_las(out, mean_xyz, mean_rgb)
        return len(mean_xyz)
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


def write_las(out: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    xyz = xyz.astype("<f4").astype(np.float64)   # the same float32 positions the PLY carries
    offset = np.floor(xyz.min(axis=0))
    ints = np.round((xyz - offset) / LAS_SCALE_M).astype("<i4")
    record = np.zeros(len(xyz), dtype=[("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("intensity", "<u2"),
                                       ("flags", "u1"), ("classification", "u1"), ("scan_angle", "i1"),
                                       ("user_data", "u1"), ("point_source", "<u2"),
                                       ("red", "<u2"), ("green", "<u2"), ("blue", "<u2")])
    record["x"], record["y"], record["z"] = ints[:, 0], ints[:, 1], ints[:, 2]
    record["red"], record["green"], record["blue"] = (rgb.astype("<u2") * 257).T
    low, high = offset + ints.min(axis=0) * LAS_SCALE_M, offset + ints.max(axis=0) * LAS_SCALE_M
    header_size, record_size = 227, record.dtype.itemsize
    header = struct.pack(
        "<4sHHIHH8sBB32s32sHHHIIBHI5I3d3d6d",
        b"LASF", 0, 0, 0, 0, 0, b"\0" * 8, 1, 2, b"roomscope".ljust(32, b"\0"),
        b"export_cloud.py".ljust(32, b"\0"), 1, 2026, header_size, header_size, 0, 2, record_size,
        len(record), len(record), 0, 0, 0, 0,
        LAS_SCALE_M, LAS_SCALE_M, LAS_SCALE_M, *offset,
        high[0], low[0], high[1], low[1], high[2], low[2])
    assert len(header) == header_size
    with out.open("wb") as handle:
        handle.write(header)
        handle.write(record.tobytes())


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
