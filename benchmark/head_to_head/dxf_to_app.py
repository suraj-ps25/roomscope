"""Turn a floor-plan tool's DXF room outline into the app.yaml that score.py reads.

  python benchmark/head_to_head/dxf_to_app.py export.dxf runs/<name>/plan.json bathroom \
      --ceiling 2.4955 --app "Pointorama, 2026-10-02" --out app.yaml

The tool's outline and our plan sit in one frame (both come from the recording's ARKit
world), so each ground-truth wall is identified by our plan's matching wall line. A wall's
app length is the summed length of the outline edges parallel to that line (within
MAX_ANGLE_DEG) and lying on it (within MAX_OFFSET_M): a wall the tool broke into steps
gets credit for every piece, which is the reading most favourable to the tool. The
outline's own area and edge count are recorded alongside.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from roomscope.benchmark.evaluate import _cyclic_alignment, load_ground_truth  # noqa: E402

MAX_ANGLE_DEG = 10.0
MAX_OFFSET_M = 0.25


def read_outline(dxf: Path) -> np.ndarray:
    lines = [line.strip() for line in dxf.read_text().splitlines()]
    pairs = list(zip(lines[0::2], lines[1::2]))
    vertices, current = [], None
    for code, value in pairs:
        if code == "0":
            if current is not None and len(current) == 2:
                vertices.append(current)
            current = [] if value == "VERTEX" else None
        elif current is not None and code in ("10", "20"):
            current.append(float(value))
    return np.array(vertices)


def shoelace(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def matched_length(outline: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = (end - start) / np.linalg.norm(end - start)
    normal = np.array([-direction[1], direction[0]])
    total = 0.0
    for a, b in zip(outline, np.roll(outline, -1, axis=0)):
        edge = b - a
        length = float(np.linalg.norm(edge))
        angle = np.degrees(np.arccos(min(1.0, abs(float(edge @ direction)) / length)))
        offset = abs(float(((a + b) / 2 - start) @ normal))
        if angle <= MAX_ANGLE_DEG and offset <= MAX_OFFSET_M:
            total += length
    return total


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

    outline = read_outline(Path(args.dxf))
    room = json.loads(Path(args.plan).read_text())["rooms"][0]
    truth = next(r for r in load_ground_truth(args.truth)["rooms"] if r["id"] == args.room)
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
