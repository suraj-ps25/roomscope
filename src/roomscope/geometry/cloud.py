"""Depth frames -> world points with normals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from ..capture import Frame

MAX_DEPTH = 4.0


@dataclass
class Cloud:
    points: np.ndarray
    normals: np.ndarray
    frame_ids: np.ndarray

    def __len__(self) -> int:
        return len(self.points)

    def subset(self, mask: np.ndarray) -> Cloud:
        return Cloud(self.points[mask], self.normals[mask], self.frame_ids[mask])

    @classmethod
    def concat(cls, clouds: list[Cloud]) -> Cloud:
        if not clouds:
            return cls(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, dtype=np.int32))
        return cls(np.concatenate([c.points for c in clouds]), np.concatenate([c.normals for c in clouds]),
                   np.concatenate([c.frame_ids for c in clouds]))

    def to_o3d(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.points))
        pcd.normals = o3d.utility.Vector3dVector(self.normals)
        return pcd


def frame_points(frame: Frame, pose: np.ndarray, min_confidence: int = 1, stride: int = 1,
                 max_depth: float = MAX_DEPTH) -> Cloud:
    """Back-project one depth frame. Normals from the organised depth grid (central
    differences, 2-px baseline), flipped to face the camera, then rotated to world."""
    depth = frame.depth().astype(np.float64)
    confidence = frame.confidence()
    K = frame.depth_K
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    cam = np.stack([x, y, depth], axis=-1)

    du = np.zeros_like(cam)
    dv = np.zeros_like(cam)
    du[:, 2:-2] = cam[:, 4:] - cam[:, :-4]
    dv[2:-2, :] = cam[4:, :] - cam[:-4, :]
    normals = np.cross(du, dv)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(norm, 1e-12)
    facing = np.sum(normals * cam, axis=-1) > 0
    normals[facing] *= -1

    valid = (depth > 0.15) & (depth < max_depth) & (norm[..., 0] > 0)
    if confidence is not None:
        valid &= confidence >= min_confidence
    # A normal built across a depth discontinuity is meaningless; drop those pixels.
    neighbours_valid = np.zeros_like(valid)
    neighbours_valid[2:-2, 2:-2] = (depth[2:-2, 4:] > 0) & (depth[2:-2, :-4] > 0) & \
                                   (depth[4:, 2:-2] > 0) & (depth[:-4, 2:-2] > 0)
    jump = np.zeros_like(depth)
    jump[2:-2, 2:-2] = np.maximum.reduce([np.abs(depth[2:-2, 4:] - depth[2:-2, :-4]),
                                          np.abs(depth[4:, 2:-2] - depth[:-4, 2:-2])])
    valid &= neighbours_valid & (jump < 0.08 * depth + 0.02)
    if stride > 1:
        keep = np.zeros_like(valid)
        keep[::stride, ::stride] = True
        valid &= keep

    rot, origin = pose[:3, :3], pose[:3, 3]
    pts = cam[valid] @ rot.T + origin
    nrm = normals[valid] @ rot.T
    return Cloud(pts, nrm, np.full(len(pts), frame.index, dtype=np.int32))


def fuse(frames: list[Frame], poses: dict[int, np.ndarray], voxel: float = 0.02,
         min_confidence: int = 1, stride: int = 1) -> Cloud:
    clouds = [frame_points(f, poses[f.index], min_confidence, stride) for f in frames]
    return voxelize(Cloud.concat(clouds), voxel)


def voxelize(cloud: Cloud, voxel: float) -> Cloud:
    """Average points and normals per voxel; keep the most common frame id (for provenance)."""
    if len(cloud) == 0:
        return cloud
    cells = np.floor(cloud.points / voxel).astype(np.int64)
    cells -= cells.min(axis=0)
    extent = cells.max(axis=0) + 1
    keys = (cells[:, 0] * extent[1] + cells[:, 1]) * extent[2] + cells[:, 2]
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    n = len(counts)
    points = np.stack([np.bincount(inverse, cloud.points[:, k], n) for k in range(3)], axis=1)
    normals = np.stack([np.bincount(inverse, cloud.normals[:, k], n) for k in range(3)], axis=1)
    points /= counts[:, None]
    lengths = np.linalg.norm(normals, axis=1)
    # Opposite faces never share a 2 cm voxel, but noisy normals can nearly cancel.
    good = lengths > 0.3 * counts
    normals = normals / np.maximum(lengths[:, None], 1e-12)
    frame_ids = np.zeros(n, dtype=np.int32)
    frame_ids[inverse] = cloud.frame_ids
    return Cloud(points[good], normals[good], frame_ids[good])
