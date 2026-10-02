"""Draw a tool's outline and our plan over the raw cloud both were made from.

  python benchmark/head_to_head/overlay.py cloud.ply results/<rec>_pointorama.dxf runs/<ours>/plan.json out.png "title"
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dxf_to_app import read_outlines  # noqa: E402

PLY_VERTEX = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
WALL_BAND_M = (0.9, 1.6)   # above the floor: walls, not furniture tops or ceiling


def main() -> int:
    cloud, dxf, plan, out, title = sys.argv[1:6]
    raw = Path(cloud).read_bytes()
    points = np.frombuffer(raw[raw.index(b"end_header\n") + len(b"end_header\n"):], dtype=PLY_VERTEX)
    height = points["z"] - np.percentile(points["z"], 2)
    band = (height > WALL_BAND_M[0]) & (height < WALL_BAND_M[1])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(points["x"][band], points["y"][band], s=0.1, c="0.6", label="raw LiDAR, wall band")
    for k, outline in enumerate(read_outlines(Path(dxf))):
        closed = np.vstack([outline, outline[:1]])
        ax.plot(closed[:, 0], closed[:, 1], "-", c="tab:purple", lw=2, label="Pointorama" if k == 0 else None)
    for k, room in enumerate(json.loads(Path(plan).read_text())["rooms"]):
        polygon = np.array(room["polygon"] + room["polygon"][:1])
        ax.plot(polygon[:, 0], polygon[:, 1], "-", c="tab:green", lw=2, label="roomscope LiDAR tier" if k == 0 else None)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
