"""Render real scanned rooms (vertex-coloured meshes such as Replica's) as phone imagery.

The synthetic flat's renders are clean geometry with procedural texture; a monocular depth
model treats them as out of distribution (MoGe-2 per-view scale 0.46-1.55 on them, against
6.9% spread on real iPad frames). A scanned room carries real materials, clutter and baked
real lighting, with exact metric geometry, so the photo and video tiers can be measured
on imagery that looks like a home while the truth stays exact.

Cameras follow the pipeline's convention: OpenCV axes, camera-to-world poses, z-up world.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


class MeshScene:
    def __init__(self, path: str | Path):
        mesh = o3d.io.read_triangle_mesh(str(path))
        self.vertices = np.asarray(mesh.vertices)
        self.triangles = np.asarray(mesh.triangles)
        self.colors = np.asarray(mesh.vertex_colors)
        self._scene = o3d.t.geometry.RaycastingScene()
        self._scene.add_triangles(o3d.core.Tensor(self.vertices.astype(np.float32)),
                                  o3d.core.Tensor(self.triangles.astype(np.uint32)))

    def cast(self, origins: np.ndarray, directions: np.ndarray) -> dict:
        rays = o3d.core.Tensor(np.hstack([origins, directions]).astype(np.float32))
        return {k: v.numpy() for k, v in self._scene.cast_rays(rays).items()}

    def render(self, K: np.ndarray, size: tuple[int, int], pose: np.ndarray,
               exposure: float = 1.0, noise: float = 0.0, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
        """RGB (uint8) and z-depth (metres, 0 where nothing is hit)."""
        width, height = size
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        cam = np.stack([(cols - K[0, 2]) / K[0, 0], (rows - K[1, 2]) / K[1, 1], np.ones_like(cols, float)], -1).reshape(-1, 3)
        directions = cam @ pose[:3, :3].T
        hits = self.cast(np.tile(pose[:3, 3], (len(directions), 1)), directions / np.linalg.norm(directions, axis=1, keepdims=True))
        t = hits["t_hit"]
        hit = np.isfinite(t)
        ids = np.where(hit, hits["primitive_ids"], 0).astype(np.int64)
        uv = hits["primitive_uvs"]
        corners = self.triangles[ids]
        weights = np.stack([1 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]], -1)
        colour = np.einsum("nk,nkc->nc", weights, self.colors[corners])
        colour = np.where(hit[:, None], colour, 0.0) * exposure
        if noise and rng is not None:
            colour = colour + rng.normal(0, noise, colour.shape)
        rgb = (np.clip(colour, 0, 1) * 255).round().astype(np.uint8).reshape(height, width, 3)
        # z-depth: range along the unit ray times the ray's z component in the camera frame.
        z = np.where(hit, t / np.linalg.norm(cam, axis=1), 0.0).reshape(height, width)
        return rgb, z.astype(np.float32)
