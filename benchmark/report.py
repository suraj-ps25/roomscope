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
             "MoGe-2's median depth ratio to the reference over ~16 views per scene, and the robust (MAD) spread across",
             "views. Views of one room share the bias, so",
             "averaging them doesn't remove it; it is the floor on photo/video wall accuracy, and the intervals carry it.", ""]
    rows = [{"scene": r["scene"], "reference": r["source"], "views": r["views"], "scale": f"{r['median_ratio']:.3f}",
             "per-view spread": _pct(r["view_spread"])} for r in data["scenes"]]
    lines += _table(rows, "scene")
    lines += ["", f"Across scenes: median {data['bias_median']:.3f}, spread (sd of log) {data['bias_sd_log']:.1%}.", ""]
    return lines


def damage_section(bench: Path) -> list[str]:
    lines = ["## Damage", "",
             "Staged damage on the synthetic flat (a water stain and a crack in the bedroom, mould on the bathroom ceiling),",
             "and the undamaged scanned rooms, where every reported region is a false positive. Extent error is area for",
             "stains and mould, length for cracks.", ""]
    rows = []
    for name, label in (("flat_a_lidar_fullres", "LiDAR, full-res RGB"), ("flat_a_video_model", "video, depth model"),
                        ("flat_a_photo_model", "photo, depth model")):
        metrics = _load(bench / name / "metrics.json")
        if not metrics or "damage" not in metrics:
            continue
        d = metrics["damage"]
        extents = ", ".join(f"{x['class']} {x['relative_error']:+.0%}" for x in d["detail"] if x["relative_error"] is not None)
        rows.append({"synthetic flat": label, "found": f"{d['found']}/{d['staged']}", "false positives": d["false_positives"],
                     "extent error": extents or "–"})
    lines += _table(rows, "synthetic flat") + [""]
    fp = {"photo": [], "video": []}
    for path in sorted((bench / "replica").glob("*_model/plan.json")):
        tier = "video" if "_video_" in path.parent.name else "photo"
        plan = json.loads(path.read_text())
        fp[tier].append(sum(len(s.get("damage_regions", [])) for r in plan["rooms"] for s in r["surfaces"]))
    for tier, counts in fp.items():
        if counts:
            lines.append(f"- Undamaged scanned rooms, {tier}: {sum(counts)} false regions over {len(counts)} rooms "
                         f"({sum(c == 0 for c in counts)} rooms clean).")
    return lines + [""]


def real_section(bench: Path) -> list[str]:
    summary = _load(bench / "real" / "summary_real.json")
    lines = ["## Real captures: iPad Pro LiDAR against laser scans (ARKitScenes)", ""]
    if not summary:
        return lines + ["(not run: fetch the visits with `benchmark/real/fetch_arkitscenes.py`)", ""]
    dev = set((ROOT / "benchmark" / "real" / "dev_visits.txt").read_text().split())
    lines += ["Real iPad Pro recordings of real rooms (Apple's ARKitScenes: LiDAR depth, ARKit poses, 640x480 colour),",
              "scored against laser-scanner truth surveyed with `benchmark/real/laser_truth.py`: the plan names the walls,",
              "the laser scan measures them (faces, corners, floor-to-ceiling height over the whole room, openings with a",
              "mirror test). Every survey was checked against its overlay (`docs/real/`). The same LiDAR tier as for",
              "Stray Scanner captures, unchanged. The photo and video tiers run on the same recordings' own colour",
              "stream (`benchmark/real/make_image_captures.py`): stills chosen as the photo protocol would take them,",
              "and the whole stream as a clip. These recordings are walk-arounds with the device mostly tilted up, not",
              "the protocol's corner shots and on-the-spot turns, so the image tiers run off-protocol here. **dev** visits were used to find and fix bugs and to fit the LiDAR",
              "interval calibration; **held-out** visits were chosen by a fixed rule before being looked at (one laser",
              "scan, three or more recordings) and run once, with no change made after. Wall errors are on lengths the",
              "laser measured end to end; rooms are undamaged, so every damage region reported is a false positive.", ""]
    rows = []
    for run in summary["runs"]:
        split = "dev" if run["visit"] in dev else "held-out"
        label = f"{run['visit']} / {run['recording']} / {run.get('tier', 'lidar')} ({split})"
        metrics = run.get("metrics")
        if not metrics:
            rows.append({"visit / recording": label, "walls median (cm)": "no room", "walls max (cm)": "–",
                         "ceiling (cm)": "–", "openings ≤ 2 cm": "–", "damage regions": "–", "interval coverage": "–"})
            continue
        walls, o = metrics["walls_measured"], metrics["openings"]
        rows.append({"visit / recording": label, "walls median (cm)": _cm(walls["median_abs_m"], 1),
                     "walls max (cm)": _cm(walls["max_abs_m"], 1),
                     "ceiling (cm)": " ".join(f"{100 * v:+.1f}" for v in metrics["ceiling"]["per_room_m"].values()),
                     "openings ≤ 2 cm": f"{o['passing']}/{o['scored']} (missed {o['missed']}, phantom {o['phantom']})",
                     "damage regions": metrics["damage"]["false_positives"],
                     "interval coverage": _pct(metrics["calibration"]["coverage"], 0)})
    lines += _table(rows, "visit / recording") + [""]
    for tier, split in [(t, sp) for t in ("lidar", "photo", "video") for sp in ("dev", "held-out")]:
        chosen = [r["metrics"] for r in summary["runs"] if r.get("metrics") and (r["visit"] in dev) == (split == "dev")
                  and r.get("tier", "lidar") == tier]
        if not chosen:
            continue
        walls = [m["walls_measured"]["max_abs_m"] for m in chosen if m["walls_measured"]["max_abs_m"] is not None]
        ceilings = [abs(v) for m in chosen for v in m["ceiling"]["per_room_m"].values()]
        covered = [m["calibration"]["coverage"] for m in chosen if m["calibration"]["coverage"] is not None]
        opening_pass = sum(m["openings"]["passing"] for m in chosen)
        opening_all = sum(m["openings"]["scored"] for m in chosen)
        lines.append(f"- **{tier}, {split}** ({len(chosen)} recordings with a room, walls scored on {len(walls)}): worst wall median "
                     f"{_cm(float(np.median(walls)), 1) if walls else '–'} cm; "
                     f"ceiling within 1.5 cm on {sum(c <= 0.015 for c in ceilings)}/{len(ceilings)}; openings within 2 cm "
                     f"{opening_pass}/{opening_all}; interval coverage median {_pct(float(np.median(covered)), 0)}"
                     + (" (in-sample: these runs fitted the calibration)" if split == "dev" and tier == "lidar" else "") + ".")
    lines += ["", "Repeatability (recordings of the same room): walls agreeing within 1 cm or 0.5%.", ""]
    rows = []
    for pair in summary["repeatability"]:
        scored = [w for w in pair["walls"] if "pass" in w]
        rows.append({"pair": f"{pair['visit']}: {pair['a']} vs {pair['b']}",
                     "walls passing": f"{sum(w['pass'] for w in scored)}/{len(scored)}" if scored else "different wall count",
                     "worst difference (cm)": _cm(max(abs(w["difference"]) for w in scored), 1) if scored else "–"})
    return lines + _table(rows, "pair") + [""]


def samples_section(bench: Path) -> list[str]:
    plans = sorted((bench / "samples").glob("*/plan.json"))
    lines = ["## Assessors' sample captures", "",
             "The three Stray Scanner recordings sent with the brief (`data/sample/`). They come without measurements, so",
             "nothing here is scored: this is what the pipeline makes of them, and how long it takes, cold, on an M2.", ""]
    if not plans:
        return lines + ["(not run: unzip the samples into `data/sample/`)", ""]
    rows = []
    for path in plans:
        plan = json.loads(path.read_text())
        timing = plan.get("timing_s", {})
        unseen = sum(any("ceiling not observed" in n for n in r["quality"]["notes"]) for r in plan["rooms"])
        rows.append({"sample": path.parent.name, "recording (s)": f"{plan['capture'].get('frames_used', 0) / 6:.0f}",
                     "rooms": len(plan["rooms"]), "connections": len(plan["adjacency"]),
                     "footprint (m²)": f"{plan['property']['footprint_area']['value']:.1f}",
                     "ceiling not seen in": f"{unseen} rooms", "run (s)": f"{timing.get('total', 0):.0f}",
                     "of which damage (s)": f"{timing.get('damage', 0):.0f}"})
    return lines + _table(rows, "sample") + [""]


def calibration_section(bench: Path) -> list[str]:
    lines = ["## Interval calibration", "",
             "Propagated intervals (fit uncertainty through the geometry, plus the tier's scale budget) are scaled per tier",
             "and quantity by split-conformal multipliers fitted on these runs and shipped in `calibration/`. The",
             "'interval coverage' columns above are before calibration (the runs were made without tables); below is",
             "coverage after it, leave-one-capture-out, so a capture never calibrates the intervals it is scored on.",
             "Target: 90%. The video multipliers are large because the depth model's per-scene bias (above) dwarfs",
             "the propagated fit uncertainty.", ""]
    rows = []
    for tier in ("lidar", "video", "photo"):
        table = _load(bench / f"calibration_{tier}.json")
        if not table:
            continue
        for quantity, k in table["multipliers"].items():
            rows.append({"tier / quantity": f"{tier} / {quantity}", "multiplier": f"{k:.2f}",
                         "samples": table["samples"].get(quantity), "held-out coverage":
                         _pct(table["held_out_coverage"].get(quantity), 0)})
    return lines + _table(rows, "tier / quantity") + [""]


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
             "**What this is and isn't.** There are no captures from our own iPhone (none was available), so the",
             "head-to-head and the walk-in on real phone captures are still to be run (`benchmark/head_to_head/`;",
             "the walk-in is rehearsed by `scripts/walk_in.sh`). What is here: a synthetic flat with exact truth for",
             "all three tiers (multi-room, stitching, openings, drift); real scanned rooms for the photo and video",
             "tiers; the depth model's measured scale error on real iPad imagery; and real iPad Pro LiDAR recordings",
             "of real rooms scored against laser scans, on dev and held-out visits. Gates are the brief's: openings ≤ 2 cm on",
             "≥ 85% (misses and phantoms count), ceiling ≤ 1.5 cm, walls ±3% (video) / ±8% (photo), footprint ±8%.", ""]
    lines += (lidar_section(bench) + flat_section(bench) + replica_section(bench) + depth_section(bench)
              + damage_section(bench) + real_section(bench) + samples_section(bench) + calibration_section(bench))
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
