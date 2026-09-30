"""Simulated LiDAR-tier capture: per-frame depth + confidence (+ optional RGB) with the
drifted odometry a phone would report. Depth is rendered from the *true* pose; the pose
handed to the pipeline is the drifted one."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..capture import CaptureBundle, Frame
from .render import Camera, Renderer, lidar_depth
from .scene import build_mesh, load_scene
from .trajectory import DriftModel, apply_drift, default_route, walkthrough

RGB_SIZE = (1920, 1440)
DEPTH_SIZE = (256, 192)


@dataclass
class SimLidar:
    bundle: CaptureBundle
    true_poses: np.ndarray
    depth: np.ndarray
    confidence: np.ndarray
    rgb_camera: Camera
    depth_camera: Camera


def simulate_lidar(scene_path: str | Path, drift_level: float = 1.0, seed: int = 0,
                   fps: float = 6.0, route: list[str] | None = None) -> SimLidar:
    scene = load_scene(scene_path)
    renderer = Renderer(scene, build_mesh(scene))
    trajectory = walkthrough(scene, route or default_route(scene), fps=fps, seed=seed)
    odometry = apply_drift(trajectory.poses, DriftModel.level(drift_level), seed=seed + 1)
    rgb_camera = Camera.iphone_main(*RGB_SIZE)
    depth_camera = rgb_camera.scaled(*DEPTH_SIZE)
    rng = np.random.default_rng(seed + 2)
    scale_bias = rng.normal(0, 0.002)

    depths, confidences, frames = [], [], []
    for index, (pose, est) in enumerate(zip(trajectory.poses, odometry)):
        hits = renderer.trace(depth_camera, pose)
        depth, confidence = lidar_depth(hits, rng, scale_bias=scale_bias)
        depths.append(depth)
        confidences.append(confidence)
        frames.append(Frame(
            index=index, timestamp=float(trajectory.timestamps[index]), K=rgb_camera.K,
            image_size=RGB_SIZE, pose=est, depth_K=depth_camera.K, depth_size=DEPTH_SIZE,
            load_depth=(lambda d=depth: d), load_confidence=(lambda c=confidence: c)))
    bundle = CaptureBundle(f"sim_{scene.name}_lidar_s{seed}", "lidar", frames, source_app="roomscope-sim")
    return SimLidar(bundle, trajectory.poses, np.asarray(depths), np.asarray(confidences),
                    rgb_camera, depth_camera)
