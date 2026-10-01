"""Scanned-room benchmark: photo and video tiers on Replica rooms, with the real depth model
and with rendered true depth (the ablation that isolates the depth model's share).

  python benchmark/run_replica.py --out runs/replica_bench [--rooms room0 office1] [--depth model oracle]

Captures come from `roomscope sim data/public/replica/cull_replica_mesh/<room>.ply --tier
photo|video --out data/captures/replica/<room>_<tier>` (see benchmark/README.md). Each run
is scored against truth fitted to the mesh; the summary has one row per room x tier x depth.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from roomscope.benchmark.evaluate import evaluate, load_ground_truth  # noqa: E402

ROOMS = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rooms", nargs="+", default=ROOMS)
    parser.add_argument("--tiers", nargs="+", default=["video", "photo"])
    parser.add_argument("--depth", nargs="+", default=["model", "oracle"], choices=["model", "oracle"])
    parser.add_argument("--captures", default=str(ROOT / "data" / "captures" / "replica"))
    parser.add_argument("--out", default=str(ROOT / "runs" / "replica_bench"))
    args = parser.parse_args()
    roomscope = str(Path(sys.executable).parent / "roomscope")
    out = Path(args.out)
    rows = []
    for room in args.rooms:
        for tier in args.tiers:
            capture = Path(args.captures) / f"{room}_{tier}"
            truth = load_ground_truth(capture / "ground_truth.json")
            gt = truth["rooms"][0]
            for depth in args.depth:
                run_dir = out / f"{room}_{tier}_{depth}"
                flags = ["--oracle-depth"] if depth == "oracle" else []
                if (run_dir / "plan.json").exists() and (run_dir / "metrics.json").exists():
                    done = subprocess.CompletedProcess([], 0)    # resumable: already run
                else:
                    done = subprocess.run([roomscope, "run", str(capture), "--out", str(run_dir), *flags],
                                          capture_output=True, text=True)
                if done.returncode != 0:
                    rows.append({"room": room, "tier": tier, "depth": depth, "error": done.stderr.strip().splitlines()[-1]})
                    print(json.dumps(rows[-1]), flush=True)
                    continue
                plan = json.loads((run_dir / "plan.json").read_text())
                result = evaluate(plan, truth)
                (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=float))
                predicted = max(plan["rooms"], key=lambda r: r["floor_area"]["value"])
                rel = [abs(e) / t for r in result["per_room"] for e, t in zip(r["wall_errors_m"], gt["walls"])]
                rows.append({
                    "room": room, "tier": tier, "depth": depth,
                    "walls_true": [round(w, 3) for w in gt["walls"][:2]],
                    "walls_pred": [round(w["length"]["value"], 3) for w in predicted["walls"][:2]],
                    "wall_max_cm": round(100 * result["walls"]["max_abs_m"], 2) if result["walls"]["max_abs_m"] is not None else None,
                    "wall_max_rel": round(max(rel), 4) if rel else None,
                    "ceiling_err_cm": round(100 * (predicted["ceiling_height"]["value"] - gt["ceiling_height"]), 2),
                    "area_err": round(predicted["floor_area"]["value"] / gt["floor_area"] - 1, 4),
                    "coverage": result["calibration"]["coverage"],
                    "seconds": round(sum(plan.get("timing_s", {}).values()), 1),
                })
                print(json.dumps(rows[-1]), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    for tier in args.tiers:
        for depth in args.depth:
            chosen = [r for r in rows if r.get("tier") == tier and r.get("depth") == depth and "error" not in r]
            if chosen:
                rel = np.array([r["wall_max_rel"] for r in chosen if r["wall_max_rel"] is not None])
                print(f"{tier:5s} {depth:6s}: {len(chosen)} rooms, worst-wall error median {np.median(rel):.1%}, "
                      f"max {rel.max():.1%}; ceiling |err| median {np.median([abs(r['ceiling_err_cm']) for r in chosen]):.1f} cm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
