"""Time the photo and video tiers on the live path: the model-output cache off, so every
depth and multi-view inference runs, as it will on a capture never seen before.

  python benchmark/time_live.py --out runs/bench/timing_live.json

The benchmark's photo and video plans replay cached model outputs (deterministic, keyed by
the exact inputs), so their run times are not what a fresh capture costs. The LiDAR tier
uses no model, so its benchmark timings are already live.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPTURES = {
    "photo, scanned room (Replica office0)": "data/captures/replica/office0_photo",
    "video, scanned room (Replica office0)": "data/captures/replica/office0_video",
    "photo, real iPad frames (ARKitScenes 47429914)": "data/public/real_captures/47429914_photo",
    "video, real iPad frames (ARKitScenes 47429914)": "data/public/real_captures/47429914_video",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    env = {**os.environ, "ROOMSCOPE_CACHE": "off"}
    rows = []
    for label, capture in CAPTURES.items():
        if not (ROOT / capture).exists():
            continue
        with tempfile.TemporaryDirectory() as run:
            subprocess.run([str(ROOT / ".venv" / "bin" / "roomscope"), "run", str(ROOT / capture), "--out", run],
                           env=env, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            plan = json.loads((Path(run) / "plan.json").read_text())
        rows.append({"capture": label, "tier": plan["capture"]["tier"], "frames": plan["capture"]["frames_used"],
                     "timing_s": plan["timing_s"]})
        print(f"{label}: {plan['timing_s']['total']:.0f} s")
    Path(args.out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
