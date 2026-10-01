"""The real-capture benchmark: iPad Pro LiDAR recordings scored against laser-scan truth.

  python benchmark/real/run_real.py --out runs/bench/real

For every recording of every surveyed visit (benchmark/ground_truth/arkitscenes_<visit>.yaml,
made by laser_truth.py and checked against its overlay): run the LiDAR tier, score it
against the visit's truth, and compare recordings of the same visit with each other
(repeatability). Recordings that never closed a room are reported as such, not dropped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from roomscope.benchmark.evaluate import evaluate, load_ground_truth, repeatability  # noqa: E402

DATA = ROOT / "data" / "public" / "arkitscenes"
IMAGE_CAPTURES = ROOT / "data" / "public" / "real_captures"


def _measured_walls_only(metrics: dict, truth: dict) -> None:
    """Wall errors on lengths the laser measured end to end (see laser_truth.py)."""
    import numpy as np
    for room in metrics["per_room"]:
        surveyed = next(r for r in truth["rooms"] if r["id"] == room["truth"])
        flags = surveyed.get("survey", {}).get("length_measured") or [True] * len(room["wall_errors_m"])
        room["wall_errors_measured_m"] = [e for e, ok in zip(room["wall_errors_m"], flags) if ok]
    errors = np.abs([e for r in metrics["per_room"] for e in r["wall_errors_measured_m"]])
    metrics["walls_measured"] = {"count": int(len(errors)),
                                 "median_abs_m": float(np.median(errors)) if len(errors) else None,
                                 "max_abs_m": float(np.max(errors)) if len(errors) else None}


def recordings(visit: str) -> list[str]:
    return sorted(p.parent.name for p in DATA.glob("*/visit.txt") if p.read_text().strip() == visit)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "runs" / "bench" / "real"))
    parser.add_argument("--visits", nargs="*", help="default: every surveyed visit")
    parser.add_argument("--tiers", nargs="*", default=["lidar", "photo", "video"],
                        help="photo and video run on captures made by make_image_captures.py, where present")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    truths = sorted((ROOT / "benchmark" / "ground_truth").glob("arkitscenes_*.yaml"))
    rows, pairs = [], []
    for truth_path in truths:
        visit = truth_path.stem.split("_")[1]
        if args.visits and visit not in args.visits:
            continue
        truth = load_ground_truth(truth_path)
        plans = {}
        for video, tier in [(v, t) for v in recordings(visit) for t in args.tiers]:
            capture = DATA / video if tier == "lidar" else IMAGE_CAPTURES / f"{video}_{tier}"
            if not capture.exists():
                continue
            run = out / (video if tier == "lidar" else f"{video}_{tier}")
            if not (run / "plan.json").exists():
                done = subprocess.run([str(ROOT / ".venv" / "bin" / "roomscope"), "run", str(capture), "--out", str(run),
                                       "--tier", tier], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                if done.returncode != 0:
                    rows.append({"visit": visit, "recording": video, "tier": tier, "rooms": 0,
                                 "notes": [done.stderr.strip().splitlines()[-1] if done.stderr.strip() else "failed"]})
                    print(json.dumps(rows[-1]), flush=True)
                    continue
            plan = json.loads((run / "plan.json").read_text())
            row = {"visit": visit, "recording": video, "tier": tier, "rooms": len(plan["rooms"])}
            if plan["rooms"]:
                metrics = evaluate(plan, truth)
                _measured_walls_only(metrics, truth)
                (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
                row["metrics"] = metrics
                if tier == "lidar":
                    plans[video] = plan
            else:
                row["notes"] = plan["capture"]["notes"]
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "metrics"}), flush=True)
        for a, b in combinations(sorted(plans), 2):
            pairs.append({"visit": visit, "a": a, "b": b, "walls": repeatability(plans[a], plans[b])})
    (out / "summary_real.json").write_text(json.dumps({"runs": rows, "repeatability": pairs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
