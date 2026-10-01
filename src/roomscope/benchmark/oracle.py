"""Oracle depth for synthetic captures (benchmark ablation only).

Replaces the monocular depth model with depth rendered from the scene's true geometry at
each photo's true pose, read from the capture's oracle/ sidecar. Running a tier both ways
splits its error into what the depth model contributes and what the rest of the pipeline
does. Real captures have no sidecar, so this can never touch a real run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..models.depth import MetricDepth


class OracleDepth:
    def __init__(self, capture: Path):
        from ..sim.render import Camera, Renderer
        from ..sim.scene import build_mesh, load_scene

        folder = Path(capture) / "oracle"
        if not folder.is_dir():
            raise FileNotFoundError(f"{capture} has no oracle/ sidecar (only synthetic captures do)")
        self.mesh = self.renderer = None
        if (folder / "mesh.txt").exists():
            from ..sim.mesh_render import MeshScene
            self.mesh = MeshScene((folder / "mesh.txt").read_text().strip())
        else:
            scene = load_scene(folder / "scene.yaml")
            self.renderer = Renderer(scene, build_mesh(scene))
        data = np.load(folder / "poses.npz")
        self.poses = dict(zip(data["keys"].tolist(), data["poses"]))
        self.camera = json.loads((folder / "camera.json").read_text())
        self._Camera = Camera

    def _pose(self, photo) -> np.ndarray:
        if photo.timestamp is not None:
            return self.poses[str(int(round(photo.timestamp * self.camera["fps"])))]
        return self.poses[f"{Path(photo.path).parent.name}/{Path(photo.path).name}"]

    def __call__(self, photo) -> MetricDepth:
        height, width = photo.image.shape[:2]
        c = self.camera
        sx, sy = width / c["width"], height / c["height"]
        camera = self._Camera(width, height, c["fx"] * sx, c["fy"] * sy, c["cx"] * sx, c["cy"] * sy)
        if self.mesh is not None:
            _, depth = self.mesh.render(camera.K, (width, height), self._pose(photo))
        else:
            depth = self.renderer.trace(camera, self._pose(photo)).zdepth
        depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0)
        mask = depth > 0.05
        return MetricDepth(np.where(mask, depth, 0.0), mask, None, camera.K)
