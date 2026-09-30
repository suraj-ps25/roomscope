"""Synthetic benchmark sweep: generate captures, run the pipeline, score every gate.

  python benchmark/run_sim.py --tier lidar --seeds 0 1 2 --drift 1.0 1.5 --out runs/simbench

Each seed is an independent capture of the same property, so seed pairs double as the
repeatability test (same space, same tier, two captures). Synthetic numbers are for
development only; the reported benchmark uses real captures with tape/laser truth.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from roomscope.benchmark.evaluate import evaluate, load_ground_truth, repeatability  # noqa: E402


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=str(ROOT / "benchmark" / "sim" / "flat_a.yaml"))
    parser.add_argument("--tier", default="lidar", choices=["lidar", "photo", "video"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--drift", type=float, nargs="+", default=[1.0])
    parser.add_argument("--rgb-width", type=int, default=480)
    parser.add_argument("--out", default=str(ROOT / "runs" / "simbench"))
    parser.add_argument("--ablation", action="store_true", help="also run with drift correction off")
    args = parser.parse_args()

    roomscope = str(Path(sys.executable).parent / "roomscope")
    out = Path(args.out)
    rows = []
    for drift in args.drift:
        plans = {}
        for seed in args.seeds:
            name = f"{Path(args.scene).stem}_{args.tier}_d{drift:g}_s{seed}"
            capture = ROOT / "data" / "captures" / "simbench" / name
            if not capture.exists():
                _run([roomscope, "sim", args.scene, "--tier", args.tier, "--out", str(capture), "--seed", str(seed),
                      "--drift", str(drift), "--rgb-width", str(args.rgb_width)])
            variants = [("corrected", [])] + ([("as-is", ["--no-drift-correction"])] if args.ablation else [])
            for label, flags in variants:
                run_dir = out / f"{name}_{label}"
                _run([roomscope, "run", str(capture), "--out", str(run_dir), *flags])
                plan = json.loads((run_dir / "plan.json").read_text())
                result = evaluate(plan, load_ground_truth(capture / "ground_truth.json"))
                (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=float))
                if label == "corrected":
                    plans[seed] = plan
                rows.append({"drift": drift, "seed": seed, "variant": label,
                             "wall_max_cm": round(result["walls"]["max_abs_m"] * 100, 2),
                             "walls_1cm": result["walls"]["within_1cm"],
                             "ceiling_max_cm": round(result["ceiling"]["max_abs_m"] * 100, 2),
                             "openings": f"{result['openings']['passing']}/{result['openings']['scored']}",
                             "adjacency": result["stitch"]["adjacency_correct"],
                             "footprint_err": round(result["stitch"]["footprint_error"] * 100, 2),
                             "coverage": result["calibration"]["coverage"],
                             "seconds": plan.get("timing_s", {}).get("total")})
                print(json.dumps(rows[-1]), flush=True)
        for a, b in combinations(sorted(plans), 2):
            rep = repeatability(plans[a], plans[b])
            passing = sum(r.get("pass", False) for r in rep)
            worst = max((abs(r["difference"]) for r in rep if "difference" in r), default=float("nan"))
            rows.append({"drift": drift, "repeatability": f"s{a} vs s{b}", "walls_passing": f"{passing}/{len(rep)}",
                         "worst_cm": round(worst * 100, 2)})
            print(json.dumps(rows[-1]), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"summary_{args.tier}.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
