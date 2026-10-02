"""Regenerate a fix's before/after over every real recording, each side at its own commit.

  python benchmark/fix_loop/regenerate_real.py 7e5e7be 1b84c55     # the declared opening fix
  python benchmark/fix_loop/regenerate_real.py 7f5bd43 561189d     # the depth-scale calibration

Both commits are checked out into git worktrees and run on the same ARKitScenes
recordings (fetched once by benchmark/real/fetch_arkitscenes.py), each with its own code.
Both are scored by the current evaluator and the current laser truth, so before and after
are measured with one ruler. It prints the opening, ceiling and wall tallies that
docs/fix_loop.md reports, and writes runs/fix_loop/real/<before>_<after>.md.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmark" / "real"))

from roomscope.benchmark.evaluate import evaluate, load_ground_truth  # noqa: E402

DEV = set((ROOT / "benchmark" / "real" / "dev_visits.txt").read_text().split())
DATA = ROOT / "data" / "public" / "arkitscenes"


def _worktree(commit: str) -> Path:
    path = ROOT / "runs" / "fix_loop" / "real" / f"src_{commit}"
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", "--force", str(path), commit],
                       check=True, stdout=subprocess.DEVNULL)
    return path


def _plans(commit: str) -> dict[str, dict]:
    src = _worktree(commit)
    out = ROOT / "runs" / "fix_loop" / "real" / commit
    env = {**os.environ, "PYTHONPATH": str(src / "src")}
    plans = {}
    for marker in sorted(DATA.glob("*/visit.txt")):
        video = marker.parent.name
        run = out / video
        if not (run / "plan.json").exists():
            subprocess.run([sys.executable, "-m", "roomscope.cli", "run", str(marker.parent), "--out", str(run)],
                           env=env, cwd=src, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if (run / "plan.json").exists():
            plans[video] = json.loads((run / "plan.json").read_text())
    return plans


def tally(commit: str) -> dict:
    counts = {"within 2 cm": 0, "matched, width off": 0, "missed": 0, "phantom": 0}
    ceilings = {"dev": [], "held-out": []}
    for video, plan in _plans(commit).items():
        visit = (DATA / video / "visit.txt").read_text().strip()
        truth_path = ROOT / "benchmark" / "ground_truth" / f"arkitscenes_{visit}.yaml"
        if not plan["rooms"] or not truth_path.exists():
            continue
        truth = load_ground_truth(truth_path)
        if not truth["rooms"]:
            continue
        metrics = evaluate(plan, truth)
        for o in metrics["openings"]["detail"]:
            if o["status"] == "matched":
                counts["within 2 cm" if o["pass"] else "matched, width off"] += 1
            else:
                counts[o["status"]] += 1
        ceilings["dev" if visit in DEV else "held-out"] += list(metrics["ceiling"]["per_room_m"].values())
    return {"openings": counts, "ceilings": ceilings}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()
    sides = {name: tally(sha) for name, sha in (("before", args.before), ("after", args.after))}
    lines = [f"# Real recordings: `{args.before}` -> `{args.after}`", "", "| | before | after |", "|---|---|---|"]
    for key in sides["before"]["openings"]:
        lines.append(f"| openings {key} | {sides['before']['openings'][key]} | {sides['after']['openings'][key]} |")
    for split in ("dev", "held-out"):
        within = [f"{sum(abs(c) <= 0.015 for c in s['ceilings'][split])}/{len(s['ceilings'][split])}" for s in sides.values()]
        median = [f"{100 * np.median(s['ceilings'][split]):+.2f} cm" if s["ceilings"][split] else "–" for s in sides.values()]
        lines.append(f"| {split} ceilings within 1.5 cm | {within[0]} | {within[1]} |")
        lines.append(f"| {split} ceiling error, median | {median[0]} | {median[1]} |")
    text = "\n".join(lines) + "\n"
    out = ROOT / "runs" / "fix_loop" / "real" / f"{args.before}_{args.after}.md"
    out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
