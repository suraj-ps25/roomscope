"""Turn a floor-plan tool's DXF room outline into the app.yaml that score.py reads.

  python benchmark/head_to_head/dxf_to_app.py export.dxf runs/<name>/plan.json bathroom \
      --ceiling 2.4955 --app "Pointorama, 2026-10-02" --out app.yaml

The tool's outline and our plan sit in one frame (both come from the recording's ARKit
world), so each ground-truth wall is identified by our plan's matching wall line. A wall's
app length is how far the tool's wall runs along that line: the extent, end to end, of
the outline edges parallel to it (within MAX_ANGLE_DEG), near it (within MAX_OFFSET_M) and
mostly alongside it (at least MIN_OVERLAP of the edge projects inside the wall's span). A
wall the tool broke into steps is read as one wall from its first piece to its last. The offset
allowance is wide because the tool saw the raw cloud, whose drift we correct and it can't:
in the large room its walls sit up to 0.5 m from ours. The
outline's own area and edge count are recorded alongside. When the tool drew several rooms,
the one sharing the most floor with our matched room is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from roomscope.benchmark.evaluate import _cyclic_alignment, load_ground_truth, match_rooms  # noqa: E402

MAX_ANGLE_DEG = 10.0
MAX_OFFSET_M = 0.6
MIN_OVERLAP = 0.5


def read_outlines(dxf: Path) -> list[np.ndarray]:
    """One array of (x, y) vertices per POLYLINE, i.e. per room the tool drew."""
    lines = [line.strip() for line in dxf.read_text().splitlines()]
    pairs = list(zip(lines[0::2], lines[1::2]))
    outlines: list[list] = []
    vertex = None
    for code, value in pairs:
        if code == "0":
            if vertex is not None and len(vertex) == 2:
                outlines[-1].append(vertex)
            vertex = None
            if value == "POLYLINE":
                outlines.append([])
            elif value == "VERTEX" and outlines:
                vertex = []
        elif vertex is not None and code in ("10", "20"):
            vertex.append(float(value))
    return [np.array(outline) for outline in outlines if len(outline) >= 3]


def shared_area(outline: np.ndarray, polygon: np.ndarray, cell_m: float = 0.05) -> float:
    low, high = polygon.min(axis=0), polygon.max(axis=0)
    xs, ys = np.meshgrid(np.arange(low[0], high[0], cell_m), np.arange(low[1], high[1], cell_m))
    cells = np.column_stack([xs.ravel(), ys.ravel()])
    inside = MplPath(polygon).contains_points(cells) & MplPath(outline).contains_points(cells)
    return float(inside.sum()) * cell_m ** 2


def shoelace(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def matched_length(outline: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    span = float(np.linalg.norm(end - start))
    direction = (end - start) / span
    normal = np.array([-direction[1], direction[0]])
    ends: list[float] = []
    for a, b in zip(outline, np.roll(outline, -1, axis=0)):
        edge = b - a
        length = float(np.linalg.norm(edge))
        angle = np.degrees(np.arccos(min(1.0, abs(float(edge @ direction)) / length)))
        offset = abs(float(((a + b) / 2 - start) @ normal))
        along = sorted((float((a - start) @ direction), float((b - start) @ direction)))
        inside = max(0.0, min(along[1], span) - max(along[0], 0.0)) / max(along[1] - along[0], 1e-9)
        if angle <= MAX_ANGLE_DEG and offset <= MAX_OFFSET_M and inside >= MIN_OVERLAP:
            ends += along
    return max(ends) - min(ends) if ends else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dxf")
    parser.add_argument("plan")
    parser.add_argument("room", help="ground-truth room id")
    parser.add_argument("--truth", required=True)
    parser.add_argument("--ceiling", type=float)
    parser.add_argument("--app", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    truth_plan = load_ground_truth(args.truth)
    room, truth = next((p, t) for p, t in match_rooms(json.loads(Path(args.plan).read_text()), truth_plan)
                       if t["id"] == args.room)
    ours_polygon = np.array(room["polygon"])
    outline = max(read_outlines(Path(args.dxf)), key=lambda o: shared_area(o, ours_polygon))
    ours = [w["length"]["value"] for w in room["walls"]]
    shift = _cyclic_alignment(ours, truth["walls"], [], [])
    walls = []
    for k in range(len(truth["walls"])):
        wall = room["walls"][(k + shift) % len(ours)]
        walls.append(round(matched_length(outline, np.array(wall["start"]), np.array(wall["end"])), 4))
    measured = {"walls": walls, "outline_edges": len(outline), "floor_area": round(shoelace(outline), 4),
                "openings": []}
    if args.ceiling is not None:
        measured["ceiling_height"] = args.ceiling
    Path(args.out).write_text(yaml.safe_dump({"app": args.app, "rooms": {args.room: measured}}, sort_keys=False))
    print(Path(args.out).read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
