"""Regenerate a fix-loop entry: the same capture, run at the commit before and after a fix.

  python benchmark/fix_loop/regenerate.py <fix id>      # one entry from fixes.yaml
  python benchmark/fix_loop/regenerate.py --all

Each commit is checked out into its own git worktree under runs/fix_loop/<id>/<side>/src
and run on the identical capture with that commit's own code; both plans are then scored
by the *current* evaluator, so before and after are measured by the same ruler. Output:
runs/fix_loop/<id>/{before,after}/plan.json, metrics.json, and diff.md.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from roomscope.benchmark.evaluate import evaluate, load_ground_truth  # noqa: E402

FIXES = Path(__file__).with_name("fixes.yaml")


def _worktree(commit: str, path: Path) -> Path:
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", "--force", str(path), commit],
                       check=True, stdout=subprocess.DEVNULL)
    return path


def _run(src: Path, capture: Path, out: Path, flags: list[str]) -> dict:
    env = {**os.environ, "PYTHONPATH": str(src / "src")}
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "roomscope.cli", "run", str(capture), "--out", str(out), *flags],
                   check=True, env=env, cwd=src)
    return json.loads((out / "plan.json").read_text())


def _headline(metrics: dict) -> dict:
    walls, ceiling, openings, stitch = metrics["walls"], metrics["ceiling"], metrics["openings"], metrics["stitch"]
    return {
        "wall median (cm)": None if walls["median_abs_m"] is None else round(100 * walls["median_abs_m"], 2),
        "wall max (cm)": None if walls["max_abs_m"] is None else round(100 * walls["max_abs_m"], 2),
        "walls within 1 cm": walls["within_1cm"],
        "ceiling max (cm)": None if ceiling["max_abs_m"] is None else round(100 * ceiling["max_abs_m"], 2),
        "openings within 2 cm": f'{openings["passing"]}/{openings["scored"]}',
        "adjacency correct": stitch["adjacency_correct"],
        "footprint error": round(stitch["footprint_error"], 4),
        "interval coverage": metrics["calibration"]["coverage"],
    }


def regenerate(fix: dict) -> Path:
    base = ROOT / "runs" / "fix_loop" / fix["id"]
    capture = ROOT / fix["capture"]
    truth = load_ground_truth(capture / "ground_truth.json")
    rows = {}
    for side in ("before", "after"):
        src = _worktree(fix[side], base / side / "src")
        plan = _run(src, capture, base / side, fix.get("flags", []))
        metrics = evaluate(plan, truth)
        (base / side / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
        rows[side] = _headline(metrics)
    lines = [f"# {fix['id']}: {fix['title']}", "", f"Symptom: {fix['symptom']}", "", f"Cause: {fix['cause']}", "",
             f"Fix: {fix['fix']} (`{fix['before']}` -> `{fix['after']}`)", "", "| metric | before | after |", "|---|---|---|"]
    lines += [f"| {k} | {rows['before'][k]} | {rows['after'][k]} |" for k in rows["before"]]
    (base / "diff.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fix", nargs="?")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    fixes = yaml.safe_load(FIXES.read_text())
    chosen = fixes if args.all else [f for f in fixes if f["id"] == args.fix]
    if not chosen:
        print(f"unknown fix; known: {', '.join(f['id'] for f in fixes)}", file=sys.stderr)
        return 2
    for fix in chosen:
        regenerate(fix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
