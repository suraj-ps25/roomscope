"""Protocol captures of a real scanned room (photo and video tiers), with truth from the mesh.

Ground truth comes from the mesh alone, not from the pipeline: floor and ceiling are the
area-weighted modes of up- and down-facing triangles, the wall direction is the
area-weighted mode of vertical normals modulo 90 degrees, and each side's wall is the
outermost inward-facing vertical plane with real area behind it. Rooms with a small jog
(a pillar, a notch) are scored against their bounding rectangle; the tiers that use
`rectangle_layout` produce the same.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .mesh_render import MeshScene
from .render import Camera
from .trajectory import DOUBLE_SPIN_PITCH_RAD, SPIN_SECONDS, pose_from_yaw_pitch

EYE_HEIGHT = 1.45
CLEARANCE_M = 0.6


def _faces(scene: MeshScene):
    tri = scene.vertices[scene.triangles]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(normal, axis=1) / 2
    normal = normal / np.maximum(2 * area[:, None], 1e-12)
    return tri.mean(axis=1), normal, area


def _mode(values: np.ndarray, weights: np.ndarray, bin_m: float = 0.01) -> float:
    hist, edges = np.histogram(values, bins=np.arange(values.min() - bin_m, values.max() + 2 * bin_m, bin_m), weights=weights)
    peak = edges[np.argmax(hist)] + bin_m / 2
    near = np.abs(values - peak) < 0.02
    return float(np.average(values[near], weights=weights[near]))


def room_truth(scene: MeshScene, name: str) -> dict:
    centre, normal, area = _faces(scene)
    z = centre[:, 2]
    floor = _mode(z[(normal[:, 2] > 0.95) & (z < z.min() + 0.5)], area[(normal[:, 2] > 0.95) & (z < z.min() + 0.5)])
    up_band = (normal[:, 2] < -0.95) & (z > floor + 2.0)
    ceiling = _mode(z[up_band], area[up_band])
    vertical = (np.abs(normal[:, 2]) < 0.1) & (z > floor + 0.3) & (z < ceiling - 0.2)
    angles = np.mod(np.arctan2(normal[vertical, 1], normal[vertical, 0]), np.pi / 2)
    hist, edges = np.histogram(angles, bins=np.radians(np.arange(0, 90.25, 0.25)), weights=area[vertical])
    theta = float(edges[np.argmax(hist)] + np.radians(0.125))
    c, s = np.cos(theta), np.sin(theta)
    to_room = np.array([[c, s], [-s, c]])                     # world xy -> wall-aligned frame
    local = centre[:, :2] @ to_room.T
    local_normal = normal[:, :2] @ to_room.T
    sides = {}
    for axis in (0, 1):
        for sign in (1, -1):
            facing = vertical & (local_normal[:, axis] * -sign > 0.95)
            coord = local[facing, axis]
            hist, edges = np.histogram(coord, bins=np.arange(coord.min() - 0.01, coord.max() + 0.02, 0.01), weights=area[facing])
            strong = np.nonzero(hist > 0.15 * hist.max())[0]
            outer = strong[-1] if sign > 0 else strong[0]
            sides[(axis, sign)] = _mode(coord[np.abs(coord - (edges[outer] + 0.005)) < 0.05],
                                        area[facing][np.abs(coord - (edges[outer] + 0.005)) < 0.05])
    x0, x1, y0, y1 = sides[(0, -1)], sides[(0, 1)], sides[(1, -1)], sides[(1, 1)]
    corners_local = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    polygon = corners_local @ to_room          # back to world xy
    return {"id": name, "polygon": polygon.round(4).tolist(), "ceiling_height": round(ceiling - floor, 4),
            "floor_area": round((x1 - x0) * (y1 - y0), 4),
            "walls": [{"index": k, "length": round(float(v), 4)} for k, v in enumerate([x1 - x0, y1 - y0, x1 - x0, y1 - y0])],
            "openings": [], "_floor_z": floor, "_theta": theta, "_local": [x0, x1, y0, y1]}


def _clearance(scene: MeshScene, point_xy: np.ndarray, floor: float) -> float:
    angles = np.radians(np.arange(0, 360, 10))
    nearest = np.inf
    for height in (0.3, 0.8, EYE_HEIGHT):
        origins = np.tile([point_xy[0], point_xy[1], floor + height], (len(angles), 1))
        dirs = np.c_[np.cos(angles), np.sin(angles), np.zeros(len(angles))]
        t = scene.cast(origins, dirs)["t_hit"]
        nearest = min(nearest, float(np.min(t)))
    return nearest


def standpoint(scene: MeshScene, truth: dict, near_xy: np.ndarray) -> np.ndarray:
    """The free spot (clear of furniture by CLEARANCE_M at shin, waist and eye height)
    closest to near_xy."""
    floor = truth["_floor_z"]
    x0, x1, y0, y1 = truth["_local"]
    theta = truth["_theta"]
    c, s = np.cos(theta), np.sin(theta)
    to_world = np.array([[c, -s], [s, c]])
    candidates = []
    for x in np.arange(x0 + 0.4, x1 - 0.4, 0.2):
        for y in np.arange(y0 + 0.4, y1 - 0.4, 0.2):
            candidates.append(to_world @ np.array([x, y]))
    candidates = np.array(candidates)
    order = np.argsort(np.linalg.norm(candidates - near_xy, axis=1))
    for k in order[:200]:
        if _clearance(scene, candidates[k], floor) >= CLEARANCE_M:
            return candidates[k]
    return candidates[order[0]]


def _iphone_camera(size: tuple[int, int], video: bool) -> Camera:
    # Main camera ~67 deg across the long side of the 4:3 sensor; video crops 16:9 (+8%).
    fx = 0.75 * max(size) * (1.08 if video else 1.0)
    return Camera(size[0], size[1], fx, fx, size[0] / 2 - 0.5, size[1] / 2 - 0.5)


def _write_truth(out: Path, truth: dict, scene_path: Path) -> None:
    public = {k: v for k, v in truth.items() if not k.startswith("_")}
    (out / "ground_truth.json").write_text(json.dumps({
        "property": truth["id"], "source": f"mesh {scene_path.name} (Replica, culled): planes fitted to the mesh",
        "rooms": [public], "adjacency": [], "footprint_area": public["floor_area"]}, indent=2))


def _write_oracle(out: Path, scene_path: Path, camera: Camera, poses: dict, fps: float | None) -> None:
    folder = out / "oracle"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "mesh.txt").write_text(str(scene_path.resolve()))
    np.savez_compressed(folder / "poses.npz", keys=np.array(list(poses)), poses=np.asarray(list(poses.values())))
    (folder / "camera.json").write_text(json.dumps({"width": camera.width, "height": camera.height, "fx": camera.fx,
                                                    "fy": camera.fy, "cx": camera.cx, "cy": camera.cy, "fps": fps}))


def write_video(scene_path: Path, out: Path, fps: float = 10.0, size: tuple[int, int] = (720, 1280), seed: int = 0) -> dict:
    import subprocess

    scene = MeshScene(scene_path)
    name = scene_path.stem
    truth = room_truth(scene, name)
    rng = np.random.default_rng(seed)
    corners = np.array(truth["polygon"])
    spot = standpoint(scene, truth, corners.mean(axis=0) + rng.normal(0, 0.3, 2))
    start = standpoint(scene, truth, corners[seed % 4])
    eye_z = truth["_floor_z"] + EYE_HEIGHT
    camera = _iphone_camera(size, video=True)
    poses = []
    # Walk in from near a corner (as from a doorway), turn twice (down, then up), walk on.
    heading = np.arctan2(*(spot - start)[::-1])
    walk = np.linalg.norm(spot - start)
    for k in range(int(walk / 0.5 * fps)):
        xy = start + (spot - start) * k / max(int(walk / 0.5 * fps), 1)
        poses.append(pose_from_yaw_pitch(np.r_[xy, eye_z + 0.015 * np.sin(k)], heading, np.radians(-5)))
    turns = int(2 * SPIN_SECONDS * fps)
    for k in range(turns):
        spin = 4 * np.pi * k / turns
        pitch = DOUBLE_SPIN_PITCH_RAD * np.tanh((spin - 2 * np.pi) / 1.0)
        poses.append(pose_from_yaw_pitch(np.r_[spot, eye_z] + rng.normal(0, 0.004, 3), heading + spin, pitch))
    for k in range(int(1.5 * fps)):
        poses.append(pose_from_yaw_pitch(np.r_[spot, eye_z], heading + 4 * np.pi, np.radians(-5)))
    out.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                "-pix_fmt", "yuv420p", "-crf", "18", str(out / "walkthrough.mp4")], stdin=subprocess.PIPE)
    for pose in poses:
        rgb, _ = scene.render(camera.K, size, pose, noise=0.01, rng=rng)
        encoder.stdin.write(rgb.tobytes())
    encoder.stdin.close()
    encoder.wait()
    _write_truth(out, truth, scene_path)
    _write_oracle(out, scene_path, camera, {str(k): p for k, p in enumerate(poses)}, fps)
    return truth


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, down, forward, eye
    return pose


def write_photos(scene_path: Path, out: Path, size: tuple[int, int] = (2016, 1512), seed: int = 0) -> dict:
    from .writers import _iphone_exif

    scene = MeshScene(scene_path)
    name = scene_path.stem
    truth = room_truth(scene, name)
    rng = np.random.default_rng(seed)
    corners = np.array(truth["polygon"])
    centre = corners.mean(axis=0)
    eye_z = truth["_floor_z"] + 1.5
    camera = _iphone_camera(size, video=False)
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    poses = {}
    for k, corner in enumerate(corners, start=1):
        inset = corner + 0.45 * (centre - corner) / np.linalg.norm(centre - corner)
        eye_xy = standpoint(scene, truth, inset)
        target_xy = centre + (centre - eye_xy) * 0.6
        drop = np.tan(np.radians(5)) * float(np.linalg.norm(target_xy - eye_xy))
        pose = look_at(np.r_[eye_xy, eye_z], np.r_[target_xy, eye_z - drop])
        rgb, _ = scene.render(camera.K, size, pose, noise=0.01, rng=rng)
        file = f"IMG_{k:04d}.JPG"
        Image.fromarray(rgb).save(folder / file, quality=92, exif=_iphone_exif(camera, "iPhone 15"))
        poses[f"{name}/{file}"] = pose
    _write_truth(out, truth, scene_path)
    _write_oracle(out, scene_path, camera, poses, None)
    return truth
