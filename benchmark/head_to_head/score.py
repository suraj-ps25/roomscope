"""Head-to-head: a consumer app's measurements vs ours, against the same ground truth.

  python benchmark/head_to_head/score.py app.yaml runs/<name>/plan.json benchmark/ground_truth/<property>.yaml

app.yaml holds the app's numbers as read off its export (magicplan: Export -> PDF/CSV
room report; Polycam: the room's measurement overlay or DXF), keyed like ground truth:

  app: magicplan 9.x, iPhone 15 Pro, LiDAR on
  rooms:
    bedroom:
      walls: [3.61, 3.39, 3.60, 3.41]       # same wall order as ground truth
      ceiling_height: 2.70
      openings: [0.86, 1.19]                # widths, ground-truth order

A quantity is a win when our error is smaller, a tie when both are within TIE_M of each
other; the brief asks for >= 70% of quantities beaten or tied.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from roomscope.benchmark.evaluate import _cyclic_alignment, load_ground_truth, match_rooms  # noqa: E402

TIE_M = 0.005
TARGET_SHARE = 0.70


def compare(app: dict, plan: dict, truth: dict) -> dict:
    by_truth = {t["id"]: p for p, t in match_rooms(plan, truth)}
    rows = []
    for room_id, measured in app["rooms"].items():
        gt = next(t for t in truth["rooms"] if t["id"] == room_id)
        ours = by_truth.get(room_id)
        if ours is None:
            continue
        ours_walls = [w["length"]["value"] for w in ours["walls"]]
        shift = _cyclic_alignment(ours_walls, gt["walls"], [], [])
        for k, true_length in enumerate(gt["walls"]):
            mine = ours_walls[(k + shift) % len(ours_walls)] if shift is not None else np.nan
            rows.append((room_id, f"wall {k}", true_length, measured["walls"][k], mine))
        rows.append((room_id, "ceiling", gt["ceiling_height"], measured.get("ceiling_height", np.nan),
                     ours["ceiling_height"]["value"]))
        ours_openings = sorted(o["width"]["value"] for o in ours["openings"])
        for k, (g, theirs) in enumerate(zip(gt.get("openings", []), measured.get("openings", []))):
            mine = min(ours_openings, key=lambda w: abs(w - g["width"])) if ours_openings else np.nan
            rows.append((room_id, f"opening {k}", g["width"], theirs, mine))
    table, outcomes = [], []
    for room_id, quantity, true_value, theirs, mine in rows:
        e_app, e_ours = abs(theirs - true_value), abs(mine - true_value)
        outcome = "tie" if abs(e_app - e_ours) <= TIE_M else ("win" if e_ours < e_app else "loss")
        if not np.isfinite(e_ours):
            outcome = "loss"
        outcomes.append(outcome)
        table.append({"room": room_id, "quantity": quantity, "truth": true_value, "app": theirs, "ours": mine,
                      "app_error_cm": round(100 * e_app, 2), "our_error_cm": round(100 * e_ours, 2), "outcome": outcome})
    share = float(np.mean([o != "loss" for o in outcomes])) if outcomes else float("nan")
    return {"app": app.get("app"), "quantities": len(outcomes), "beat_or_tie": share,
            "pass": share >= TARGET_SHARE, "rows": table}


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    app = yaml.safe_load(Path(sys.argv[1]).read_text())
    plan = json.loads(Path(sys.argv[2]).read_text())
    result = compare(app, plan, load_ground_truth(sys.argv[3]))
    print(f"{result['app']}: {result['beat_or_tie']:.0%} of {result['quantities']} quantities beaten or tied "
          f"(target {TARGET_SHARE:.0%}) -> {'PASS' if result['pass'] else 'FAIL'}")
    for row in result["rows"]:
        print(f"  {row['room']:10s} {row['quantity']:10s} truth {row['truth']:.3f}  app {row['app_error_cm']:6.2f} cm"
              f"  ours {row['our_error_cm']:6.2f} cm  {row['outcome']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
