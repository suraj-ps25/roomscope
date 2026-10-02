"""Split-conformal calibration of interval widths, per tier and quantity.

For every measurement with ground truth, the normalised score is |value - truth| / sigma,
where sigma is the propagated (uncalibrated) half width / z, with an absolute floor tau
added in quadrature, sqrt(sigma^2 + tau^2). The multiplier is the
(n + 1)-corrected 90th percentile of those scores divided by z, so an interval of
multiplier * propagated half width covers the truth with >= 90% probability on
exchangeable data. Coverage is reported leave-one-property-out (captures sharing a truth
file are one property), so neither a capture nor a repeat capture of the same rooms
calibrates the intervals it is scored on. Wall lengths the truth marks as not measured
end to end (laser surveys: a face that could not be found) are left out.

The floor is chosen per quantity from FLOOR_GRID: the narrowest median interval whose
leave-one-property-out coverage reaches the target, or the best-covering one if none does.
Without it, a plane fit's millimetre sigma has to be multiplied by tens to reach a wall
placed a few centimetres off, and the intervals of well-measured walls balloon with it.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..model import DEFAULT_CI_LEVEL, z_for_level
from .evaluate import _cyclic_alignment, _gt_area, _score_openings, load_ground_truth, match_rooms

QUANTITIES = ("wall_length", "ceiling_height", "opening_width", "floor_area")
FLOOR_GRID = {"wall_length": (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08),
              "ceiling_height": (0.0, 0.005, 0.01, 0.02, 0.03, 0.05),
              "opening_width": (0.0, 0.005, 0.01, 0.02, 0.03, 0.05),
              "floor_area": (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)}


def _uncalibrated_sigma(measurement: dict, multiplier: float, floor: float = 0.0) -> float:
    half = (measurement["ci_high"] - measurement["ci_low"]) / 2
    calibrated = half / (z_for_level(measurement["ci_level"]) * multiplier)
    return float(np.sqrt(max(calibrated ** 2 - floor ** 2, 0.0)))


def pairs(plan: dict, gt: dict, applied: dict[str, float] | None = None,
          applied_floors: dict[str, float] | None = None) -> dict[str, list[tuple[float, float]]]:
    """(error, uncalibrated sigma) per quantity. `applied` and `applied_floors` undo the
    calibration already baked into the plan's intervals, so re-calibration starts from the
    propagated sigmas."""
    applied = applied or {}
    floors = applied_floors or {}
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for pred, truth in match_rooms(plan, gt):
        lengths = [w["length"]["value"] for w in pred["walls"]]
        wall_ids = [w["id"] for w in pred["walls"]]
        # Same alignment as the evaluator: door walls break rectangle symmetry.
        pred_doors = [wall_ids.index(o["wall_id"]) for o in pred["openings"] if o["type"] == "door" and o["wall_id"] in wall_ids]
        truth_doors = [o["wall"] for o in truth.get("openings", []) if o.get("type") == "door" and "wall" in o]
        shift = _cyclic_alignment(lengths, truth["walls"], pred_doors, truth_doors)
        if shift is not None:
            n = len(truth["walls"])
            measured = truth.get("survey", {}).get("length_measured") or [True] * n
            for k in range(n):
                if not measured[k]:
                    continue
                m = pred["walls"][(k + shift) % n]["length"]
                out["wall_length"].append((m["value"] - truth["walls"][k], _uncalibrated_sigma(m, applied.get("wall_length", 1), floors.get("wall_length", 0.0))))
        m = pred["ceiling_height"]
        out["ceiling_height"].append((m["value"] - truth["ceiling_height"], _uncalibrated_sigma(m, applied.get("ceiling_height", 1), floors.get("ceiling_height", 0.0))))
        area = _gt_area(truth)
        if math.isfinite(area):
            m = pred["floor_area"]
            out["floor_area"].append((m["value"] - area, _uncalibrated_sigma(m, applied.get("floor_area", 1), floors.get("floor_area", 0.0))))
        wall_ids = [w["id"] for w in pred["walls"]]
        by_width = {round(o["width"]["value"], 6): o for o in pred["openings"]}
        for scored in _score_openings(pred, truth, shift, wall_ids):
            if scored["status"] == "matched":
                o = by_width[round(scored["measured"], 6)]
                out["opening_width"].append((scored["error"], _uncalibrated_sigma(o["width"], applied.get("opening_width", 1), floors.get("opening_width", 0.0))))
    return out


def multiplier(scores: list[tuple[float, float]], level: float = DEFAULT_CI_LEVEL) -> float:
    normalised = np.sort([abs(e) / max(s, 1e-9) for e, s in scores])
    n = len(normalised)
    rank = min(n - 1, math.ceil((n + 1) * level) - 1)
    return float(max(normalised[rank] / z_for_level(level), 1.0))


def _floored(values: list[tuple[float, float]], floor: float) -> list[tuple[float, float]]:
    return [(e, float(np.hypot(s, floor))) for e, s in values]


def _left_out_coverage(per_capture: list[dict], groups: list[str], q: str, floor: float) -> list[bool]:
    """Coverage of each property's values under a multiplier fitted on the other properties."""
    z = z_for_level(DEFAULT_CI_LEVEL)
    covered = []
    for i, capture in enumerate(per_capture):
        rest = [v for j, other in enumerate(per_capture) if groups[j] != groups[i] for v in other.get(q, [])]
        if len(rest) < 5 or not capture.get(q):
            continue
        k = multiplier(_floored(rest, floor))
        covered += [abs(e) <= z * s * k for e, s in _floored(capture[q], floor)]
    return covered


def calibrate(runs: list[tuple[dict, dict]], tier: str, applied: dict[str, float] | None = None,
              groups: list[str] | None = None, applied_floors: dict[str, float] | None = None) -> dict:
    per_capture = [pairs(plan, gt, applied, applied_floors) for plan, gt in runs]
    groups = groups or [str(i) for i in range(len(runs))]
    pooled: dict[str, list] = defaultdict(list)
    for capture in per_capture:
        for q, values in capture.items():
            pooled[q] += values
    z = z_for_level(DEFAULT_CI_LEVEL)
    multipliers, floors, held_out = {}, {}, {}
    for q in QUANTITIES:
        if len(pooled[q]) < 5:
            continue
        candidates = []
        for floor in FLOOR_GRID[q]:
            covered = _left_out_coverage(per_capture, groups, q, floor)
            k = multiplier(_floored(pooled[q], floor))
            width = float(np.median([z * s * k for _, s in _floored(pooled[q], floor)]))
            coverage = float(np.mean(covered)) if covered else float("nan")
            candidates.append((coverage, width, floor, k))
        reaching = [c for c in candidates if c[0] >= DEFAULT_CI_LEVEL]
        coverage, _, floor, k = min(reaching, key=lambda c: c[1]) if reaching else max(candidates, key=lambda c: (c[0], -c[1]))
        multipliers[q], floors[q] = k, floor
        if np.isfinite(coverage):
            held_out[q] = coverage
    return {
        "id": f"{tier}-conformal-{len(runs)}captures",
        "tier": tier,
        "level": DEFAULT_CI_LEVEL,
        "multipliers": multipliers,
        "floors_m": floors,
        "samples": {q: len(pooled[q]) for q in QUANTITIES},
        "held_out_coverage": held_out,
    }


def run(tier: str, specs: list[str], out: str | None) -> dict:
    runs, applied, groups = [], None, []
    for spec in specs:
        plan_path, truth_path = spec.split(":", 1)
        runs.append((json.loads(Path(plan_path).read_text()), load_ground_truth(truth_path)))
        groups.append(str(Path(truth_path).resolve()))
    existing = Path(__file__).resolve().parents[3] / "calibration" / f"{tier}.json"
    applied_floors = None
    if existing.exists():
        shipped = json.loads(existing.read_text())
        applied, applied_floors = shipped["multipliers"], shipped.get("floors_m", {})
    table = calibrate(runs, tier, applied, groups, applied_floors)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(table, indent=2))
    return table
