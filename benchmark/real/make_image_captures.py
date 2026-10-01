"""Photo-tier and video-tier captures made from a real recording's colour stream.

  python benchmark/real/make_image_captures.py 47429912 [...]

ARKitScenes recordings carry the iPad's own 640x480 colour stream next to the LiDAR, so the
same real rooms that have laser truth can also be fed to the image-only tiers:

  photo  <video>_photo/<room>/IMG_nnnn.JPG   the stills a person following the photo
         protocol would take in each room: sharp frames, looking across the room (far
         median depth), at headings spread around the room; EXIF carries the 35 mm focal.
         Rooms are assigned by camera position, from the LiDAR tier's plan of the same
         recording; depth and poses are used only to choose frames, never passed on.
  video  <video>_video/walkthrough.mp4      the whole colour stream, upright, at its own
         frame rate. The recordings are walk-arounds, not the protocol's on-the-spot turns,
         so this tests the video tier off-protocol, and is reported as such.

Frames are rotated upright by the recording's sky direction, as a phone would save them.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data" / "public" / "arkitscenes"
OUT = ROOT / "data" / "public" / "real_captures"
PHOTOS_PER_ROOM = 6
MIN_HEADING_GAP_DEG = 40
MIN_MEDIAN_DEPTH_M = 1.0
CORNER_REACH_M = 1.6
FULL_FRAME_DIAGONAL_MM = 43.27
TURNS = {"Up": 0, "Left": -1, "Right": 1, "Down": 2}


def _sky(video: str) -> str:
    for row in csv.DictReader((DATA / "metadata.csv").open()):
        if row["video_id"] == video:
            return row["sky_direction"]
    return "Up"


def _upright(image: np.ndarray, sky: str) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(image, TURNS.get(sky, 0)))


def _save_photo(rgb: np.ndarray, K: np.ndarray, path: Path, taken: float) -> None:
    height, width = rgb.shape[:2]
    f35 = int(round(K[0, 0] * FULL_FRAME_DIAGONAL_MM / np.hypot(width, height)))
    exif = Image.Exif()
    exif[0x010F] = "Apple"
    exif[0x0110] = "iPad Pro (ARKitScenes)"
    exif[0x0132] = "2021:01:01 12:00:00"
    exif.get_ifd(0x8769)[0xA405] = f35
    exif.get_ifd(0x8769)[0x9003] = f"2021:01:01 12:{int(taken // 60) % 60:02d}:{int(taken % 60):02d}"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, quality=95, exif=exif)


def photos(video: str) -> Path:
    from matplotlib.path import Path as MplPath

    from roomscope.io.arkitscenes import read_arkitscenes
    from roomscope.tiers.lidar import LidarOptions, run_lidar

    bundle = read_arkitscenes(DATA / video)
    result = run_lidar(bundle, LidarOptions(damage=False))
    plan = result.plan.to_dict()
    sky = _sky(video)
    out = OUT / f"{video}_photo"
    # Every frame with colour, not only the LiDAR keyframes: the plan-frame pose of the rest
    # comes from its raw ARKit pose through the keyframes' median raw-to-plan transform.
    keyed = {f.index: f for f in result.frames}
    raw = {f.index: f.pose for f in bundle.frames}
    transforms = [result.poses[k] @ np.linalg.inv(raw[k]) for k in keyed]
    to_plan = transforms[len(transforms) // 2]
    frames = [f for f in bundle.frames if f.load_rgb is not None]
    plan_pose = {f.index: result.poses.get(f.index, to_plan @ f.pose) for f in frames}
    for room in plan["rooms"]:
        polygon = np.array(room["polygon"])
        inside = MplPath(polygon)
        centre = polygon.mean(axis=0)
        candidates = []
        for frame in frames:
            pose = plan_pose[frame.index]
            position, forward = pose[:2, 3], pose[:3, 2]
            if not inside.contains_point(position) or abs(forward[2]) > 0.6:
                continue   # a phone photo of a room is taken in it, looking across it
            to_centre = centre - position
            facing = float(forward[:2] @ to_centre / max(np.linalg.norm(forward[:2]) * np.linalg.norm(to_centre), 1e-6))
            depth = frame.depth()
            median = float(np.median(depth[depth > 0])) if (depth > 0).any() else 0.0
            grey = cv2.cvtColor(frame.rgb(), cv2.COLOR_RGB2GRAY)
            sharpness = min(float(cv2.Laplacian(grey, cv2.CV_64F).var()), 150.0)
            candidates.append({"frame": frame, "position": position, "facing": facing, "median": median,
                               "sharpness": sharpness, "heading": float(np.degrees(np.arctan2(forward[1], forward[0])))})
        picked = []
        # The protocol's corner shots: from near each corner, looking at the room's centre.
        for corner in polygon:
            near = [c for c in candidates if np.linalg.norm(c["position"] - corner) < CORNER_REACH_M and c["facing"] > 0.8
                    and all(c["frame"] is not p["frame"] for p in picked)]
            if near:
                picked.append(max(near, key=lambda c: c["sharpness"] * c["median"]))
        for c in sorted(candidates, key=lambda c: -c["sharpness"] * c["median"]):
            if len(picked) >= PHOTOS_PER_ROOM:
                break
            if c["median"] >= MIN_MEDIAN_DEPTH_M and all(
                    abs((c["heading"] - p["heading"] + 180) % 360 - 180) >= MIN_HEADING_GAP_DEG for p in picked):
                picked.append(c)
        picked = [(0, 0, c["frame"]) for c in picked]
        if len(picked) < 2:
            print(f"{video} {room['id']}: only {len(picked)} usable views; folder skipped")
            continue
        for n, (_, _, frame) in enumerate(sorted(picked, key=lambda p: p[2].timestamp), start=1):
            rgb = _upright(frame.rgb(), sky)
            K = frame.K.copy()
            if TURNS.get(sky, 0) % 2:
                K = K[[1, 0, 2]][:, [1, 0, 2]]
            _save_photo(rgb, K, out / room["id"] / f"IMG_{n:04d}.JPG", frame.timestamp)
        print(f"{video} {room['id']}: {len(picked)} photos")
    return out


def video(video: str) -> Path:
    sky = _sky(video)
    files = sorted((DATA / video / "vga_wide").glob("*.png"), key=lambda p: float(p.stem.split("_")[-1]))
    stamps = np.array([float(p.stem.split("_")[-1]) for p in files])
    fps = round(1.0 / float(np.median(np.diff(stamps))))
    out = OUT / f"{video}_video"
    out.mkdir(parents=True, exist_ok=True)
    first = _upright(cv2.imread(str(files[0])), sky)
    height, width = first.shape[:2]
    encoder = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                "-pix_fmt", "yuv420p", "-crf", "18", str(out / "walkthrough.mp4")], stdin=subprocess.PIPE)
    for path in files:
        encoder.stdin.write(_upright(cv2.imread(str(path)), sky).tobytes())
    encoder.stdin.close()
    encoder.wait()
    print(f"{video}: {len(files)} frames at {fps} fps")
    return out


if __name__ == "__main__":
    for name in sys.argv[1:]:
        photos(name)
        video(name)
