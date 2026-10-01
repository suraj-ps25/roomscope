"""Measure a LiDAR device's depth scale against laser scans; write calibration/lidar_depth.json.

  python benchmark/real/depth_scale.py            # the dev visits in dev_visits.txt

For each dev recording: the LiDAR tier's fused cloud (no depth correction), placed on the
laser scan, then registered by similarity ICP. The scale that registration needs is how
much the device's depth reads short or long. A registration that matches too little of the
cloud (fitness below MIN_FITNESS) says more about overlap than scale and is not used. The
device's correction is the median over its recordings. Held-out visits are never used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import laser_truth  # noqa: E402

DATA = ROOT / "data" / "public" / "arkitscenes"
MIN_FITNESS = 0.5


def measure(video: str, visit: str) -> dict:
    import open3d as o3d

    from roomscope.geometry.cloud import fuse
    from roomscope.io.arkitscenes import read_arkitscenes
    from roomscope.tiers.lidar import LidarOptions, run_lidar

    bundle = read_arkitscenes(DATA / video)
    result = run_lidar(bundle, LidarOptions(damage=False, depth_scale=1.0))
    plan = result.plan.to_dict()
    cloud = fuse(result.frames, result.poses, voxel=0.02, stride=2, min_confidence=2).points
    points, stations, floor_z, _ = laser_truth.level(*laser_truth.load_scans(DATA / "laser" / visit, 8))
    yaw, shift, _ = laser_truth.place(plan, laser_truth._band(points, floor_z + 0.9, floor_z + 1.6))
    ours = np.c_[laser_truth._rotate(cloud[:, :2], yaw) + shift, cloud[:, 2] + floor_z]
    lo, hi = ours.min(axis=0) - 0.3, ours.max(axis=0) + 0.3
    laser = points[np.all((points > lo) & (points < hi), axis=1)]
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ours))
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(laser)).voxel_down_sample(0.01)
    reg = o3d.pipelines.registration
    rigid = reg.registration_icp(source, target, 0.05, np.eye(4), reg.TransformationEstimationPointToPoint(False),
                                 reg.ICPConvergenceCriteria(max_iteration=200))
    similar = reg.registration_icp(source, target, 0.03, rigid.transformation, reg.TransformationEstimationPointToPoint(True),
                                   reg.ICPConvergenceCriteria(max_iteration=300))
    return {"recording": video, "visit": visit, "source_app": bundle.source_app,
            "scale": round(float(np.cbrt(np.linalg.det(similar.transformation[:3, :3]))), 5),
            "fitness": round(float(similar.fitness), 3), "rmse_m": round(float(similar.inlier_rmse), 4)}


def main() -> int:
    visits = (Path(__file__).with_name("dev_visits.txt")).read_text().split()
    rows = []
    for visit in visits:
        for marker in sorted(DATA.glob("*/visit.txt")):
            if marker.read_text().strip() != visit:
                continue
            try:
                rows.append(measure(marker.parent.name, visit))
            except Exception as error:   # a recording that never closes a room has no plan to place
                rows.append({"recording": marker.parent.name, "visit": visit, "error": str(error)})
            print(json.dumps(rows[-1]), flush=True)
    devices = {}
    for app in {r["source_app"] for r in rows if "scale" in r}:
        used = [r for r in rows if r.get("source_app") == app and r["fitness"] >= MIN_FITNESS]
        if used:
            devices[app] = {"scale": round(float(np.median([r["scale"] for r in used])), 5),
                            "recordings": [r["recording"] for r in used]}
    table = {"id": f"lidar-depth-laser-{len(rows)}recordings", "method": "similarity ICP of the fused cloud on laser scans",
             "min_fitness": MIN_FITNESS, "devices": devices, "measurements": rows}
    (ROOT / "calibration" / "lidar_depth.json").write_text(json.dumps(table, indent=2))
    print(json.dumps(devices))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
