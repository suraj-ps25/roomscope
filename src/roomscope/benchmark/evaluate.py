"""Score a plan.json against ground truth, gate by gate.

Ground truth is what a person with a tape or laser can record (benchmark/README.md):

  rooms:
    - id: bedroom                     # photo tier: must equal the room's folder name
      ceiling_height: [2.708, 2.712]  # one or more shots, averaged
      walls: [3.600, 3.400, 3.600, 3.400]   # counter-clockwise, any starting wall
      openings:
        - {type: door, width: 0.850, height: 2.050, wall: 1}
  adjacency: [[bedroom, hallway], ...]
  footprint_area: 44.04               # optional; default = sum of room areas

Matching: rooms by folder name when names agree, else by shape (Hungarian on area,
ceiling and sorted wall lengths); walls by the best cyclic alignment of the CCW sequences,
using door positions to break the ties rectangles create; openings by type, wall and
width. A missed opening and a phantom opening each count as a miss.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

OPENING_GATE_M = 0.02
OPENING_GATE_SHARE = 0.85
CEILING_GATE_M = 0.015
FOOTPRINT_GATE = 0.08


def load_ground_truth(path: str | Path) -> dict:
    text = Path(path).read_text()
    gt = yaml.safe_load(text) if str(path).endswith((".yaml", ".yml")) else json.loads(text)
    for room in gt["rooms"]:
        if isinstance(room.get("walls", [None])[0], dict):
            room["walls"] = [w["length"] for w in room["walls"]]
        ceiling = room.get("ceiling_height")
        room["ceiling_height"] = float(np.mean(ceiling)) if isinstance(ceiling, list) else float(ceiling)
    gt.setdefault("adjacency", [])
    if not gt.get("footprint_area"):
        gt["footprint_area"] = float(sum(_gt_area(r) for r in gt["rooms"]))
    return gt


def _gt_area(room: dict) -> float:
    if "floor_area" in room:
        return float(room["floor_area"])
    walls = room["walls"]
    return float(walls[0] * walls[1]) if len(walls) == 4 else float("nan")


@dataclass
class RoomScore:
    predicted: str
    truth: str
    wall_errors: list[float]
    wall_truth: list[float]
    wall_covered: list[bool]
    ceiling_error: float
    ceiling_covered: bool
    area_error: float
    openings: list[dict] = field(default_factory=list)


def _cyclic_alignment(pred: list[float], truth: list[float], pred_doors: list[int], truth_doors: list[int]) -> int | None:
    if len(pred) != len(truth):
        return None
    n = len(truth)
    best, best_cost = None, np.inf
    for shift in range(n):
        cost = sum(abs(pred[(k + shift) % n] - truth[k]) for k in range(n))
        # Rectangles give identical costs for opposite shifts; door walls decide.
        mapped = {(d - shift) % n for d in pred_doors}
        cost -= 1e-3 * len(mapped & set(truth_doors))
        if cost < best_cost:
            best, best_cost = shift, cost
    return best


def _room_cost(pred: dict, truth: dict) -> float:
    area_p, area_t = pred["floor_area"]["value"], _gt_area(truth)
    lengths_p = sorted(w["length"]["value"] for w in pred["walls"])
    lengths_t = sorted(truth["walls"])
    shape = abs(len(lengths_p) - len(lengths_t)) * 1.0
    if len(lengths_p) == len(lengths_t):
        shape += float(np.mean(np.abs(np.array(lengths_p) - np.array(lengths_t))))
    return abs(area_p - area_t) / max(area_t, 1e-6) + shape + abs(pred["ceiling_height"]["value"] - truth["ceiling_height"])


def match_rooms(plan: dict, gt: dict) -> list[tuple[dict, dict]]:
    predicted, truth = plan["rooms"], gt["rooms"]
    by_name = {r["id"]: r for r in truth}
    if all(p["id"] in by_name for p in predicted):
        return [(p, by_name[p["id"]]) for p in predicted]
    cost = np.array([[_room_cost(p, t) for t in truth] for p in predicted])
    rows, cols = linear_sum_assignment(cost)
    return [(predicted[r], truth[c]) for r, c in zip(rows, cols)]


def _score_room(pred: dict, truth: dict) -> RoomScore:
    pred_lengths = [w["length"]["value"] for w in pred["walls"]]
    wall_ids = [w["id"] for w in pred["walls"]]
    pred_doors = [wall_ids.index(o["wall_id"]) for o in pred["openings"] if o["type"] == "door" and o["wall_id"] in wall_ids]
    truth_doors = [o["wall"] for o in truth.get("openings", []) if o.get("type") == "door" and "wall" in o]
    shift = _cyclic_alignment(pred_lengths, truth["walls"], pred_doors, truth_doors)
    errors, covered, truth_lengths = [], [], []
    if shift is not None:
        n = len(truth["walls"])
        for k in range(n):
            measured = pred["walls"][(k + shift) % n]["length"]
            errors.append(measured["value"] - truth["walls"][k])
            covered.append(measured["ci_low"] <= truth["walls"][k] <= measured["ci_high"])
            truth_lengths.append(truth["walls"][k])
    ceiling = pred["ceiling_height"]
    score = RoomScore(pred["id"], truth["id"], errors, truth_lengths, covered, ceiling["value"] - truth["ceiling_height"],
                      ceiling["ci_low"] <= truth["ceiling_height"] <= ceiling["ci_high"],
                      pred["floor_area"]["value"] - _gt_area(truth))
    score.openings = _score_openings(pred, truth, shift, wall_ids)
    return score


def _unmeasured(o: dict, truth: dict, shift: int | None, wall_ids: list[str]) -> bool:
    """A predicted opening where the truth survey found an opening it could not measure (a
    laser survey with something in front of a jamb) is neither right nor a phantom."""
    if shift is None or o["wall_id"] not in wall_ids:
        return False
    wall = (wall_ids.index(o["wall_id"]) - shift) % len(truth["walls"])
    u0 = o["offset"]["value"]
    u1 = u0 + o["width"]["value"]
    return any(g["wall"] == wall and u0 < g["offset"] + g.get("span", 0) and g["offset"] < u1
               for g in truth.get("survey", {}).get("occluded_openings", []))


def _score_openings(pred: dict, truth: dict, shift: int | None, wall_ids: list[str]) -> list[dict]:
    gt_openings = truth.get("openings", [])
    pr_openings = [o for o in pred["openings"] if not _unmeasured(o, truth, shift, wall_ids)]
    n = len(truth["walls"])
    cost = np.full((len(pr_openings), len(gt_openings)), 10.0)
    for i, o in enumerate(pr_openings):
        pred_wall = wall_ids.index(o["wall_id"]) if o["wall_id"] in wall_ids else None
        for j, g in enumerate(gt_openings):
            if (o["type"] == "window") != (g.get("type") == "window"):
                continue
            wall_penalty = 0.0
            if shift is not None and pred_wall is not None and "wall" in g and (pred_wall - shift) % n != g["wall"]:
                wall_penalty = 1.0
            cost[i, j] = abs(o["width"]["value"] - g["width"]) + wall_penalty
    results = []
    used_pred, used_gt = set(), set()
    if cost.size:
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if cost[i, j] >= 1.0:
                continue
            o, g = pr_openings[i], gt_openings[j]
            error = o["width"]["value"] - g["width"]
            results.append({"status": "matched", "type": g.get("type"), "truth": g["width"], "measured": o["width"]["value"],
                            "error": error, "covered": o["width"]["ci_low"] <= g["width"] <= o["width"]["ci_high"],
                            "pass": abs(error) <= OPENING_GATE_M})
            used_pred.add(i)
            used_gt.add(j)
    for j, g in enumerate(gt_openings):
        if j not in used_gt:
            results.append({"status": "missed", "type": g.get("type"), "truth": g["width"], "pass": False})
    for i, o in enumerate(pr_openings):
        if i not in used_pred:
            results.append({"status": "phantom", "type": o["type"], "measured": o["width"]["value"], "pass": False})
    return results


def _overlap_area(polygons: list[np.ndarray]) -> float:
    """Total pairwise overlap between room polygons, by rasterisation at 1 cm."""
    if len(polygons) < 2:
        return 0.0
    from matplotlib.path import Path as MplPath
    corners = np.vstack(polygons)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    xs, ys = np.meshgrid(np.arange(lo[0], hi[0], 0.01), np.arange(lo[1], hi[1], 0.01))
    grid = np.stack([xs.ravel(), ys.ravel()], axis=1)
    count = np.zeros(len(grid), dtype=int)
    for polygon in polygons:
        count += MplPath(polygon).contains_points(grid)
    return float(np.sum(count > 1) * 1e-4)


DAMAGE_REACH_M = 0.15


def _damage_shift(pred: dict, truth: dict) -> int | None:
    wall_ids = [w["id"] for w in pred["walls"]]
    pred_doors = [wall_ids.index(o["wall_id"]) for o in pred["openings"] if o["type"] == "door" and o["wall_id"] in wall_ids]
    truth_doors = [o["wall"] for o in truth.get("openings", []) if o.get("type") == "door" and "wall" in o]
    return _cyclic_alignment([w["length"]["value"] for w in pred["walls"]], truth["walls"], pred_doors, truth_doors)


def _at_staged_place(region: dict, surface: dict, staged: dict, shift: int | None, n_walls: int) -> bool:
    """A reported region counts for a staged one only where it is: on the same wall (via
    the room's wall alignment) and within DAMAGE_REACH_M of the staged extent. Without
    this, any crack-shaped edge on any wall of the room "finds" the staged crack. Truth
    without an extent (older captures), ceilings and floors match by surface kind."""
    extent = staged.get("extent_uv")
    if extent is None or not staged["surface"].startswith("wall:") or shift is None:
        return True
    wall = (int(staged["surface"].split(":")[1]) + shift) % n_walls
    if not surface["id"].endswith(f"/w{wall}"):
        return False
    centre = np.mean(np.asarray(region["polygon_uv"], dtype=float), axis=0)
    u0, v0, u1, v1 = extent
    return u0 - DAMAGE_REACH_M <= centre[0] <= u1 + DAMAGE_REACH_M and v0 - DAMAGE_REACH_M <= centre[1] <= v1 + DAMAGE_REACH_M


def _score_damage(pairs: list[tuple[dict, dict]], plan: dict) -> dict:
    """Staged damage vs reported regions, matched by room, class and place (wall and
    position; ceilings and floors by surface kind). A truth region is found if a reported
    region of its class is there; every reported region left over is a false positive."""
    found, missed, false_positive = [], [], []
    matched_rooms = set()
    for pred, truth in pairs:
        matched_rooms.add(pred["id"])
        shift = _damage_shift(pred, truth)
        reported = [(s, d) for s in pred["surfaces"] for d in s.get("damage_regions", [])]
        used = set()
        for staged in truth.get("damage", []):
            kind = "ceiling" if staged["surface"] == "ceiling" else ("floor" if staged["surface"] == "floor" else "wall")
            options = [(k, d) for k, (surface, d) in enumerate(reported)
                       if k not in used and surface["kind"] == kind and d["class"] == staged["class"]
                       and _at_staged_place(d, surface, staged, shift, len(pred["walls"]))]
            if not options:
                missed.append({"room": truth["id"], "class": staged["class"], "surface": staged["surface"]})
                continue
            size = "length" if staged["class"] == "crack" else "area"
            k, best = min(options, key=lambda item: abs((item[1].get(size) or {}).get("value", 0) - staged.get(size, 0)))
            used.add(k)
            measured = (best.get(size) or {}).get("value")
            found.append({"room": truth["id"], "class": staged["class"], "quantity": size, "truth": staged.get(size),
                          "measured": measured, "relative_error": (measured / staged[size] - 1) if measured and staged.get(size) else None})
        false_positive += [{"room": truth["id"], "class": d["class"], "surface": surface["kind"]}
                           for k, (surface, d) in enumerate(reported) if k not in used]
    for room in plan["rooms"]:
        if room["id"] not in matched_rooms:
            false_positive += [{"room": room["id"], "class": d["class"], "surface": s["kind"]}
                               for s in room["surfaces"] for d in s.get("damage_regions", [])]
    staged = len(found) + len(missed)
    reported = len(found) + len(false_positive)
    return {"staged": staged, "found": len(found), "false_positives": len(false_positive),
            "recall": len(found) / staged if staged else None, "precision": len(found) / reported if reported else None,
            "detail": found, "missed": missed, "false_positive_detail": false_positive}


def evaluate(plan: dict, gt: dict) -> dict:
    pairs = match_rooms(plan, gt)
    rooms = [_score_room(p, t) for p, t in pairs]
    name_of = {p["id"]: t["id"] for p, t in pairs}

    wall_errors = np.array([e for r in rooms for e in r.wall_errors])
    wall_truth = np.array([t for r in rooms for t in r.wall_truth])
    openings = [o for r in rooms for o in r.openings]
    unmatched_gt_rooms = [t["id"] for t in gt["rooms"] if t["id"] not in {r.truth for r in rooms}]
    for room_id in unmatched_gt_rooms:
        truth = next(t for t in gt["rooms"] if t["id"] == room_id)
        openings += [{"status": "missed", "type": o.get("type"), "truth": o["width"], "pass": False}
                     for o in truth.get("openings", [])]
    opening_pass = sum(o["pass"] for o in openings)

    predicted_edges = {tuple(sorted((name_of.get(a["from"], a["from"]), name_of.get(a["to"], a["to"]))))
                       for a in plan["adjacency"]}
    truth_edges = {tuple(sorted(e)) for e in gt["adjacency"]}
    footprint = plan["property"]["footprint_area"]
    footprint_error = (footprint["value"] - gt["footprint_area"]) / gt["footprint_area"]
    overlap = _overlap_area([np.array(r["polygon"]) for r in plan["rooms"]])

    covered = [c for r in rooms for c in r.wall_covered] + [r.ceiling_covered for r in rooms] + \
        [o["covered"] for o in openings if o["status"] == "matched"]
    ci_level = plan["calibration"]["ci_level"]
    return {
        "capture": plan["capture"]["id"],
        "tier": plan["capture"]["tier"],
        "rooms": {"predicted": len(plan["rooms"]), "truth": len(gt["rooms"]), "unmatched_truth": unmatched_gt_rooms},
        "walls": {
            "count": int(len(wall_errors)),
            "median_abs_m": float(np.median(np.abs(wall_errors))) if len(wall_errors) else None,
            "p90_abs_m": float(np.percentile(np.abs(wall_errors), 90)) if len(wall_errors) else None,
            "max_abs_m": float(np.max(np.abs(wall_errors))) if len(wall_errors) else None,
            "max_rel": float(np.max(np.abs(wall_errors) / wall_truth)) if len(wall_errors) else None,
            "within_1cm": float(np.mean(np.abs(wall_errors) <= 0.01)) if len(wall_errors) else None,
        },
        "ceiling": {
            "max_abs_m": float(max(abs(r.ceiling_error) for r in rooms)) if rooms else None,
            "per_room_m": {r.truth: round(r.ceiling_error, 4) for r in rooms},
            "gate_pass": bool(rooms) and all(abs(r.ceiling_error) <= CEILING_GATE_M for r in rooms),
        },
        "openings": {
            "scored": len(openings),
            "passing": int(opening_pass),
            "share": opening_pass / len(openings) if openings else None,
            "missed": sum(o["status"] == "missed" for o in openings),
            "phantom": sum(o["status"] == "phantom" for o in openings),
            "gate_pass": bool(openings) and opening_pass / len(openings) >= OPENING_GATE_SHARE,
            "detail": openings,
        },
        "stitch": {
            "adjacency_correct": predicted_edges == truth_edges,
            "adjacency_missing": sorted(truth_edges - predicted_edges),
            "adjacency_extra": sorted(predicted_edges - truth_edges),
            "overlap_m2": round(overlap, 3),
            "footprint_error": footprint_error,
            "footprint_covered": footprint["ci_low"] <= gt["footprint_area"] <= footprint["ci_high"],
        },
        "calibration": {"ci_level": ci_level, "coverage": float(np.mean(covered)) if covered else None,
                        "measurements": len(covered)},
        "damage": _score_damage(pairs, plan),
        "per_room": [{"predicted": r.predicted, "truth": r.truth,
                      "wall_errors_m": [round(e, 4) for e in r.wall_errors],
                      "ceiling_error_m": round(r.ceiling_error, 4), "area_error_m2": round(r.area_error, 3)} for r in rooms],
    }


def repeatability(plan_a: dict, plan_b: dict) -> list[dict]:
    """Per-wall agreement between two captures of the same space at the same tier."""
    rows = []
    for pa, pb in match_rooms(plan_a, {"rooms": [_as_truth(r) for r in plan_b["rooms"]]}):
        lengths_a = [w["length"]["value"] for w in pa["walls"]]
        lengths_b = pb["walls"]
        shift = _cyclic_alignment(lengths_a, lengths_b, [], [])
        if shift is None:
            rows.append({"room": pa["id"], "status": "different wall count"})
            continue
        n = len(lengths_b)
        for k in range(n):
            a, b = lengths_a[(k + shift) % n], lengths_b[k]
            limit = max(0.01, 0.005 * b)
            rows.append({"room": pa["id"], "wall": k, "a": a, "b": b, "difference": a - b, "limit": limit,
                         "pass": abs(a - b) <= limit})
    return rows


def _as_truth(room: dict) -> dict:
    return {"id": room["id"], "walls": [w["length"]["value"] for w in room["walls"]],
            "ceiling_height": room["ceiling_height"]["value"], "floor_area": room["floor_area"]["value"],
            "openings": []}


def summary_table(result: dict) -> str:
    walls, ceiling, openings, stitch = result["walls"], result["ceiling"], result["openings"], result["stitch"]
    cm = lambda v: "n/a" if v is None else f"{v * 100:.2f} cm"
    lines = [
        f"{result['capture']} ({result['tier']} tier): {result['rooms']['predicted']} rooms predicted, "
        f"{result['rooms']['truth']} in truth",
        f"  walls      median {cm(walls['median_abs_m'])}, p90 {cm(walls['p90_abs_m'])}, max {cm(walls['max_abs_m'])}"
        + (f", within 1 cm {walls['within_1cm']:.0%}" if walls["within_1cm"] is not None else ""),
        f"  ceiling    max {cm(ceiling['max_abs_m'])}  gate(<=1.5 cm) {'PASS' if ceiling['gate_pass'] else 'FAIL'}",
        f"  openings   {openings['passing']}/{openings['scored']} within 2 cm (missed {openings['missed']}, "
        f"phantom {openings['phantom']})  gate(>=85%) {'PASS' if openings['gate_pass'] else 'FAIL'}",
        f"  stitch     adjacency {'correct' if stitch['adjacency_correct'] else 'WRONG'}, overlap {stitch['overlap_m2']:.2f} m2, "
        f"footprint {stitch['footprint_error']:+.1%}",
        f"  intervals  {result['calibration']['coverage']:.0%} of {result['calibration']['measurements']} truths inside "
        f"the stated {result['calibration']['ci_level']:.0%} interval" if result["calibration"]["coverage"] is not None else "",
    ]
    return "\n".join(line for line in lines if line)
