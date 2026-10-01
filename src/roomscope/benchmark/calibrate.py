"""Split-conformal calibration of interval widths, per tier and quantity.

For every measurement with ground truth, the normalised score is |value - truth| / sigma,
where sigma is the propagated (uncalibrated) half width / z. The multiplier is the
(n + 1)-corrected 90th percentile of those scores divided by z, so an interval of
multiplier * propagated half width covers the truth with >= 90% probability on
exchangeable data. Coverage is reported leave-one-property-out (captures sharing a truth
file are one property), so neither a capture nor a repeat capture of the same rooms
calibrates the intervals it is scored on. Wall lengths the truth marks as not measured
end to end (laser surveys: a face that could not be found) are left out.
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


def _uncalibrated_sigma(measurement: dict, multiplier: float) -> float:
    half = (measurement["ci_high"] - measurement["ci_low"]) / 2
    return half / (z_for_level(measurement["ci_level"]) * multiplier)


def pairs(plan: dict, gt: dict, applied: dict[str, float] | None = None) -> dict[str, list[tuple[float, float]]]:
    """(error, uncalibrated sigma) per quantity. `applied` undoes multipliers already baked
    into the plan's intervals, so re-calibration starts from the propagated sigmas."""
    applied = applied or {}
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
                out["wall_length"].append((m["value"] - truth["walls"][k], _uncalibrated_sigma(m, applied.get("wall_length", 1))))
        m = pred["ceiling_height"]
        out["ceiling_height"].append((m["value"] - truth["ceiling_height"], _uncalibrated_sigma(m, applied.get("ceiling_height", 1))))
        area = _gt_area(truth)
        if math.isfinite(area):
            m = pred["floor_area"]
            out["floor_area"].append((m["value"] - area, _uncalibrated_sigma(m, applied.get("floor_area", 1))))
        wall_ids = [w["id"] for w in pred["walls"]]
        by_width = {round(o["width"]["value"], 6): o for o in pred["openings"]}
        for scored in _score_openings(pred, truth, shift, wall_ids):
            if scored["status"] == "matched":
                o = by_width[round(scored["measured"], 6)]
                out["opening_width"].append((scored["error"], _uncalibrated_sigma(o["width"], applied.get("opening_width", 1))))
    return out


def multiplier(scores: list[tuple[float, float]], level: float = DEFAULT_CI_LEVEL) -> float:
    normalised = np.sort([abs(e) / max(s, 1e-9) for e, s in scores])
    n = len(normalised)
    rank = min(n - 1, math.ceil((n + 1) * level) - 1)
    return float(max(normalised[rank] / z_for_level(level), 1.0))


def calibrate(runs: list[tuple[dict, dict]], tier: str, applied: dict[str, float] | None = None,
              groups: list[str] | None = None) -> dict:
    per_capture = [pairs(plan, gt, applied) for plan, gt in runs]
    groups = groups or [str(i) for i in range(len(runs))]
    pooled: dict[str, list] = defaultdict(list)
    for capture in per_capture:
        for q, values in capture.items():
            pooled[q] += values
    multipliers = {q: multiplier(pooled[q]) for q in QUANTITIES if len(pooled[q]) >= 5}

    # Leave one property out: calibrate on the rest, check coverage on the one left out.
    held_out: dict[str, list[bool]] = defaultdict(list)
    z = z_for_level(DEFAULT_CI_LEVEL)
    for i, capture in enumerate(per_capture):
        rest: dict[str, list] = defaultdict(list)
        for j, other in enumerate(per_capture):
            if groups[j] != groups[i]:
                for q, values in other.items():
                    rest[q] += values
        for q, values in capture.items():
            if len(rest[q]) < 5:
                continue
            k = multiplier(rest[q])
            held_out[q] += [abs(e) <= z * s * k for e, s in values]
    return {
        "id": f"{tier}-conformal-{len(runs)}captures",
        "tier": tier,
        "level": DEFAULT_CI_LEVEL,
        "multipliers": multipliers,
        "samples": {q: len(pooled[q]) for q in QUANTITIES},
        "held_out_coverage": {q: float(np.mean(v)) for q, v in held_out.items() if v},
    }


def run(tier: str, specs: list[str], out: str | None) -> dict:
    runs, applied, groups = [], None, []
    for spec in specs:
        plan_path, truth_path = spec.split(":", 1)
        runs.append((json.loads(Path(plan_path).read_text()), load_ground_truth(truth_path)))
        groups.append(str(Path(truth_path).resolve()))
    existing = Path(__file__).resolve().parents[3] / "calibration" / f"{tier}.json"
    if existing.exists():
        applied = json.loads(existing.read_text())["multipliers"]
    table = calibrate(runs, tier, applied, groups)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(table, indent=2))
    return table
