"""How far off is the monocular depth model's metric scale, per scene?

  python benchmark/depth_model_scale.py --out runs/depth_scale.json

For each scanned-room video (rendered truth from the capture's oracle sidecar) and each
ARKitScenes sequence (its LiDAR depth as reference), MoGe-2's per-view median ratio to the
reference over ~16 views. The per-scene median is the bias a room inherits (views of one
room share it, so averaging views does not remove it); the spread across scenes is what
the photo and video intervals have to carry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def scanned_room(capture: Path) -> list[float]:
    from roomscope.benchmark.oracle import OracleDepth
    from roomscope.io.photos import Photo
    from roomscope.io.video import read_video
    from roomscope.models.depth import metric_depth

    oracle = OracleDepth(capture)
    video = read_video(capture / "walkthrough.mp4")
    height, width = video.images[0].shape[:2]
    camera = oracle.camera
    fx = camera["fx"] * width / camera["width"]
    fov = float(np.degrees(2 * np.arctan(width / (2 * fx))))
    ratios = []
    for k in np.linspace(len(video.images) * 0.2, len(video.images) - 10, 16).astype(int):
        photo = Photo(capture, "probe", video.images[k], None, fov, video.images[k], None, None, float(video.timestamps[k]))
        truth = oracle(photo).depth
        estimate = metric_depth(video.images[k], fov)
        ok = estimate.mask & (truth > 0)
        if ok.sum() > 500:
            ratios.append(float(np.median(estimate.depth[ok] / truth[ok])))
    return ratios


def arkitscenes(sequence: Path) -> list[float]:
    from roomscope.io.arkitscenes import read_arkitscenes
    from roomscope.models.depth import metric_depth

    frames = [f for f in read_arkitscenes(sequence).frames if f.load_rgb is not None]
    ratios = []
    for frame in frames[::max(1, len(frames) // 16)][:16]:
        rgb = frame.rgb()
        # Upright, as a phone photo would be (sequences are stored in sensor orientation).
        down = frame.pose[:3, :3].T @ np.array([0.0, 0.0, -1.0])
        turns = 0 if abs(down[1]) >= abs(down[0]) and down[1] > 0 else (2 if abs(down[1]) >= abs(down[0]) else
                                                                           (3 if down[0] > 0 else 1))
        fov = float(np.degrees(2 * np.arctan(rgb.shape[1] / (2 * frame.K[0, 0]))))
        upright = np.ascontiguousarray(np.rot90(rgb, turns))
        if turns % 2:
            fov = float(np.degrees(2 * np.arctan(rgb.shape[0] / (2 * frame.K[1, 1]))))
        estimate = metric_depth(upright, fov)
        depth, confidence = frame.depth(), frame.confidence()
        depth = np.rot90(depth, turns)
        confidence = np.rot90(confidence, turns) if confidence is not None else np.full(depth.shape, 2)
        reference = cv2.resize(np.ascontiguousarray(depth), estimate.depth.shape[::-1], interpolation=cv2.INTER_NEAREST)
        conf = cv2.resize(np.ascontiguousarray(confidence).astype(np.uint8), estimate.depth.shape[::-1],
                          interpolation=cv2.INTER_NEAREST)
        ok = estimate.mask & (reference > 0.3) & (conf >= 2)
        if ok.sum() > 500:
            ratios.append(float(np.median(estimate.depth[ok] / reference[ok])))
    return ratios


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replica", default=str(ROOT / "data" / "captures" / "replica"))
    parser.add_argument("--arkitscenes", default=str(ROOT / "data" / "public" / "arkitscenes"))
    parser.add_argument("--out", default=str(ROOT / "runs" / "depth_scale.json"))
    args = parser.parse_args()
    rows = []
    for capture in sorted(Path(args.replica).glob("*_video")):
        r = scanned_room(capture)
        rows.append({"scene": capture.name.replace("_video", ""), "source": "Replica render, true depth", "views": len(r),
                     "median_ratio": float(np.median(r)), "view_spread": float(np.std(r))})
        print(json.dumps(rows[-1]), flush=True)
    for sequence in sorted(p for p in Path(args.arkitscenes).iterdir() if p.is_dir()):
        r = arkitscenes(sequence)
        rows.append({"scene": f"ARKitScenes {sequence.name}", "source": "real iPad frames, LiDAR depth", "views": len(r),
                     "median_ratio": float(np.median(r)), "view_spread": float(np.std(r))})
        print(json.dumps(rows[-1]), flush=True)
    bias = np.array([row["median_ratio"] for row in rows])
    summary = {"scenes": rows, "bias_median": float(np.median(bias)), "bias_sd_log": float(np.std(np.log(bias)))}
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"per-scene scale: median {summary['bias_median']:.3f}, spread (sd of log) {summary['bias_sd_log']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
