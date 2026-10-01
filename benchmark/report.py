"""Build docs/benchmark_report.md from the outputs of benchmark/run_all.sh.

  python benchmark/report.py --bench runs/bench --out docs/benchmark_report.md

Nothing in the report is typed in by hand: every number is read from a metrics file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _pct(x, digits=1):
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def _cm(x, digits=2):
    return "–" if x is None else f"{100 * x:.{digits}f}"


def _gates(metrics: dict) -> dict:
    walls, ceiling, openings, stitch = metrics["walls"], metrics["ceiling"], metrics["openings"], metrics["stitch"]
    return {
        "wall median (cm)": _cm(walls["median_abs_m"]), "wall max (cm)": _cm(walls["max_abs_m"]),
        "wall max (%)": _pct(walls["max_rel"]), "ceiling max (cm)": _cm(ceiling["max_abs_m"]),
        "openings ≤ 2 cm": f"{openings['passing']}/{openings['scored']}" + (" ✓" if openings["gate_pass"] else ""),
        "adjacency": "✓" if stitch["adjacency_correct"] else "✗", "overlap (m²)": f"{stitch['overlap_m2']:.2f}",
        "footprint": _pct(stitch["footprint_error"]), "interval coverage": _pct(metrics["calibration"]["coverage"], 0),
    }


def _table(rows: list[dict], first: str) -> list[str]:
    if not rows:
        return ["(no results)"]
    keys = [k for k in rows[0] if k != first]
    out = [f"| {first} | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
    out += [f"| {r[first]} | " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    return out


def lidar_section(bench: Path) -> list[str]:
    summary = _load(bench / "lidar" / "summary_lidar.json") or []
    lines = ["## LiDAR tier (synthetic flat, three independent captures)", ""]
    rows = []
    for item in summary:
        if "seed" not in item:
            continue
        metrics = _load(bench / "lidar" / f"flat_a_lidar_d{item['drift']:g}_s{item['seed']}_{item['variant']}" / "metrics.json")
        if metrics:
            rows.append({"capture": f"seed {item['seed']}, drift correction {'on' if item['variant'] == 'corrected' else 'OFF'}",
                         **_gates(metrics), "run (s)": item.get("seconds")})
    lines += _table(rows, "capture") + [""]
    rep = [r for r in summary if "repeatability" in r]
    if rep:
        lines += ["Repeatability (same space, same tier, two captures): walls agreeing within 1 cm or 0.5%.", ""]
        lines += _table([{"pair": f"{r['repeatability']} ({r.get('variant', 'corrected')})", "walls passing": r["walls_passing"],
                          "worst difference (cm)": r["worst_cm"]} for r in rep], "pair") + [""]
    return lines


def flat_section(bench: Path) -> list[str]:
    lines = ["## Photo and video tiers on the synthetic flat", "",
             "With the depth model (as shipped) and with rendered true depth (`--oracle-depth`), which isolates the",
             "pipeline from the depth model. The synthetic renders are out of distribution for MoGe-2 (see the depth",
             "model section), so the model-depth rows here understate real-imagery accuracy; the scanned-room rows below",
             "are the better guide.", ""]
    rows = []
    for tier in ("video", "photo"):
        for depth in ("model", "oracle"):
            metrics = _load(bench / f"flat_a_{tier}_{depth}" / "metrics.json")
            if metrics:
                rows.append({"run": f"{tier}, {'depth model' if depth == 'model' else 'true depth'}", **_gates(metrics)})
    return lines + _table(rows, "run") + [""]


def replica_section(bench: Path) -> list[str]:
    summary = _load(bench / "replica" / "summary.json") or []
    lines = ["## Photo and video tiers on scanned rooms (Replica)", "",
             "Real scanned rooms (real materials, clutter, baked real lighting) rendered as protocol captures, with truth",
             "fitted to the mesh at tape height. One room per scene, so no stitching is scored here. Walls are scored",
             "against the room's bounding rectangle; rooms with a jog (office1) are marked.", ""]
    rows = []
    for item in summary:
        if "error" in item:
            rows.append({"room": f"{item['room']} {item['tier']} {item['depth']}", "worst wall": "failed", "ceiling (cm)": "",
                         "floor area": "", "coverage": ""})
            continue
        rows.append({"room": f"{item['room']}, {item['tier']}, {'depth model' if item['depth'] == 'model' else 'true depth'}",
                     "worst wall": _pct(item["wall_max_rel"]), "ceiling (cm)": f"{item['ceiling_err_cm']:+.1f}",
                     "floor area": _pct(item["area_err"]), "coverage": _pct(item["coverage"], 0)})
    lines += _table(rows, "room") + [""]
    for tier in ("video", "photo"):
        for depth in ("model", "oracle"):
            chosen = [r for r in summary if r.get("tier") == tier and r.get("depth") == depth and "error" not in r]
            if chosen:
                worst = np.array([r["wall_max_rel"] for r in chosen])
                lines.append(f"- **{tier}, {'depth model' if depth == 'model' else 'true depth'}**: worst-wall error median "
                             f"{np.median(worst):.1%}, {np.mean(worst <= (0.03 if tier == 'video' else 0.08)):.0%} of rooms within the "
                             f"{'±3%' if tier == 'video' else '±8%'} gate; floor area error median "
                             f"{np.median([abs(r['area_err']) for r in chosen]):.1%}.")
    return lines + [""]


def depth_section(bench: Path) -> list[str]:
    data = _load(bench / "depth_scale.json")
    if not data:
        return []
    lines = ["## Depth model: metric scale per scene", "",
             "MoGe-2's median depth ratio to the reference over ~16 views per scene. Views of one room share this bias, so",
             "averaging them doesn't remove it; it is the floor on photo/video wall accuracy, and the intervals carry it.", ""]
    rows = [{"scene": r["scene"], "reference": r["source"], "views": r["views"], "scale": f"{r['median_ratio']:.3f}",
             "per-view spread": _pct(r["view_spread"])} for r in data["scenes"]]
    lines += _table(rows, "scene")
    lines += ["", f"Across scenes: median {data['bias_median']:.3f}, spread (sd of log) {data['bias_sd_log']:.1%}.", ""]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default=str(ROOT / "runs" / "bench"))
    parser.add_argument("--out", default=str(ROOT / "docs" / "benchmark_report.md"))
    args = parser.parse_args()
    bench = Path(args.bench)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    lines = ["# Benchmark report", "",
             f"Generated by `benchmark/report.py` from `benchmark/run_all.sh` outputs (commit `{commit}`). Every number is",
             "read from a metrics file; regenerate with `benchmark/run_all.sh`.", "",
             "**What this is and isn't.** No real phone captures with laser truth exist yet (no iPhone was available to",
             "build this), so the brief's benchmark set, head-to-head and walk-in test are still to be run",
             "(`benchmark/README.md`, `benchmark/head_to_head/`). What is here: a synthetic flat with exact truth for all",
             "three tiers (multi-room, stitching, openings, drift), real scanned rooms for the photo and video tiers, and",
             "the depth model's measured scale error on real iPad imagery. Gates are the brief's: openings ≤ 2 cm on",
             "≥ 85% (misses and phantoms count), ceiling ≤ 1.5 cm, walls ±3% (video) / ±8% (photo), footprint ±8%.", ""]
    lines += lidar_section(bench) + flat_section(bench) + replica_section(bench) + depth_section(bench)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
