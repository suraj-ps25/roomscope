"""Write synthetic captures to disk in the same layout real captures arrive in.

Photo tier: <out>/<room>/IMG_0001.JPG ... (JPEG, iPhone-style EXIF, no depth, no poses)
Video tier: <out>/walkthrough.mp4 (H.264, no depth, no poses)
Every tier also gets <out>/ground_truth.json (never read by the pipeline).
"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .render import Camera, Renderer
from .scene import build_mesh, ground_truth, load_scene
from .trajectory import photo_views, walkthrough

_worker_renderer: Renderer | None = None

EXIF_MAKE, EXIF_MODEL, EXIF_FOCAL, EXIF_FOCAL35 = 0x010F, 0x0110, 0x920A, 0xA405
EXIF_IFD = 0x8769


def _init_worker(scene_path: str) -> None:
    global _worker_renderer
    scene = load_scene(scene_path)
    _worker_renderer = Renderer(scene, build_mesh(scene))


def _render_rgb(args: tuple) -> np.ndarray:
    camera, pose, seed, exposure, noise = args
    rng = np.random.default_rng(seed)
    hits = _worker_renderer.trace(camera, pose)
    return (_worker_renderer.shade(hits, exposure=exposure, rng=rng, noise_sigma=noise) * 255).astype(np.uint8)


def render_many(scene_path: str, camera: Camera, poses: np.ndarray, seed: int = 0,
                exposure: float = 1.0, noise: float = 0.01, workers: int = 6):
    jobs = [(camera, pose, seed * 100003 + i, exposure, noise) for i, pose in enumerate(poses)]
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(str(scene_path),)) as pool:
        yield from pool.map(_render_rgb, jobs, chunksize=4)


def _iphone_exif(camera: Camera, model: str) -> Image.Exif:
    exif = Image.Exif()
    exif[EXIF_MAKE] = "Apple"
    exif[EXIF_MODEL] = model
    diag = float(np.hypot(camera.width, camera.height))
    ifd = exif.get_ifd(EXIF_IFD)
    ifd[EXIF_FOCAL35] = int(round(camera.fx * 43.27 / diag))
    ifd[EXIF_FOCAL] = 5.96
    return exif


def write_oracle(scene_path: str | Path, out: Path, camera: Camera, poses: dict, fps: float | None = None) -> None:
    """Benchmark-only sidecar: the true camera and poses, so `roomscope run --oracle-depth`
    can swap the monocular depth model for rendered truth and separate pipeline error from
    depth-model error. The pipeline never reads it otherwise."""
    import shutil
    folder = out / "oracle"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(scene_path, folder / "scene.yaml")
    np.savez_compressed(folder / "poses.npz", keys=np.array(list(poses)), poses=np.asarray(list(poses.values())))
    (folder / "camera.json").write_text(json.dumps({"width": camera.width, "height": camera.height, "fx": camera.fx,
                                                    "fy": camera.fy, "cx": camera.cx, "cy": camera.cy, "fps": fps}))


def write_ground_truth(scene_path: str | Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "ground_truth.json").write_text(json.dumps(ground_truth(load_scene(scene_path)), indent=2))


def write_photo_tier(scene_path: str | Path, out: Path, rooms: list[str] | None = None,
                     size: tuple[int, int] = (2016, 1512), model: str = "iPhone 15",
                     seed: int = 0, exposure: float = 1.0, noise: float = 0.01) -> None:
    scene = load_scene(scene_path)
    camera = Camera.iphone_main(*size)
    counter = 1
    oracle_poses = {}
    for room_id in rooms or [r.id for r in scene.rooms]:
        views = photo_views(scene, room_id)
        folder = out / room_id
        folder.mkdir(parents=True, exist_ok=True)
        poses = np.asarray([pose for _, pose in views])
        for rgb in render_many(str(scene_path), camera, poses, seed + counter, exposure, noise):
            Image.fromarray(rgb).save(folder / f"IMG_{counter:04d}.JPG", quality=92,
                                      exif=_iphone_exif(camera, model))
            counter += 1
        oracle_poses.update({f"{room_id}/IMG_{counter - len(poses) + k:04d}.JPG": pose for k, pose in enumerate(poses)})
    write_ground_truth(scene_path, out)
    write_oracle(scene_path, out, camera, oracle_poses)


def write_lidar_tier(scene_path: str | Path, out: Path, route: list[str], seed: int = 0,
                     drift_level: float = 1.0, fps: float = 5.0, rgb_size: tuple[int, int] = (1920, 1440),
                     exposure: float = 1.0, noise: float = 0.01) -> None:
    """Stray Scanner layout: depth/ confidence/ odometry.csv camera_matrix.csv rgb.mp4.
    The poses written are the drifted odometry; the true poses go to a ground-truth file."""
    import cv2
    from scipy.spatial.transform import Rotation

    from ..capture import internal_pose_to_stray
    from .lidar import simulate_lidar

    sim = simulate_lidar(scene_path, drift_level=drift_level, seed=seed, fps=fps, route=route)
    rgb_camera = Camera.iphone_main(*rgb_size)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "confidence").mkdir(exist_ok=True)
    rows = []
    for frame, depth, confidence in zip(sim.bundle.frames, sim.depth, sim.confidence):
        cv2.imwrite(str(out / "depth" / f"{frame.index:06d}.png"), np.round(depth * 1000).astype(np.uint16))
        cv2.imwrite(str(out / "confidence" / f"{frame.index:06d}.png"), confidence)
        stray = internal_pose_to_stray(frame.pose)
        qx, qy, qz, qw = Rotation.from_matrix(stray[:3, :3]).as_quat()
        x, y, z = stray[:3, 3]
        rows.append(f"{frame.timestamp:.6f}, {frame.index}, {x:.6f}, {y:.6f}, {z:.6f}, "
                    f"{qx:.8f}, {qy:.8f}, {qz:.8f}, {qw:.8f}, {rgb_camera.fx:.4f}, {rgb_camera.fy:.4f}, "
                    f"{rgb_camera.cx:.4f}, {rgb_camera.cy:.4f}")
    (out / "odometry.csv").write_text(
        "timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy\n" + "\n".join(rows) + "\n")
    np.savetxt(out / "camera_matrix.csv", rgb_camera.K, delimiter=",", fmt="%.6f")

    encoder = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{rgb_size[0]}x{rgb_size[1]}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out / "rgb.mp4")],
        stdin=subprocess.PIPE)
    for rgb in render_many(str(scene_path), rgb_camera, sim.true_poses, seed, exposure, noise):
        encoder.stdin.write(rgb.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg failed while encoding the synthetic RGB stream")
    write_ground_truth(scene_path, out)
    np.save(out / "ground_truth_poses.npy", sim.true_poses)


def write_video_tier(scene_path: str | Path, out: Path, route: list[str], fps: float = 10.0,
                     size: tuple[int, int] = (1280, 720), seed: int = 0,
                     exposure: float = 1.0, noise: float = 0.01) -> None:
    scene = load_scene(scene_path)
    trajectory = walkthrough(scene, route, fps=fps, seed=seed, perimeter=False)
    # Video is a 16:9 crop of the 4:3 sensor at the same horizontal FOV, plus a little
    # stabilisation crop.
    camera = Camera.iphone_main(*size)
    camera = Camera(camera.width, camera.height, camera.fx * 1.08, camera.fy * 1.08, camera.cx, camera.cy)
    out.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out / "walkthrough.mp4")],
        stdin=subprocess.PIPE)
    for rgb in render_many(str(scene_path), camera, trajectory.poses, seed, exposure, noise):
        encoder.stdin.write(rgb.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg failed while encoding the synthetic walkthrough")
    write_ground_truth(scene_path, out)
    write_oracle(scene_path, out, camera, {f"{k}": pose for k, pose in enumerate(trajectory.poses)}, fps)
