"""Thin photo captures: every photo capture cut to 2 and 3 photos a room, run through the photo tier.

  python benchmark/thin_photos.py --out runs/bench/thin

The brief's photo tier is 2 to 8 stills a room, and the few-photos interval table
(calibration/photo_few_views.json) is fitted on these. For each capture with ground truth
(the eight scanned rooms and the synthetic flat), each room folder keeps k photos (k = 2, 3)
in two deterministic selections: evenly spaced through the folder starting at its first
photo, and starting at its second. Each variant shares its parent's truth file, so
leave-one-property-out coverage never scores a variant on a table fitted on its siblings.
Writes <out>/<variant>/plan.json and prints the plan:truth pairs for `roomscope calibrate`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from roomscope.io.detect import image_files  # noqa: E402

CAPTURES = sorted((ROOT / "data" / "captures" / "replica").glob("*_photo")) + [ROOT / "data" / "captures" / "sim_flat_a_photo"]
KEEP = (2, 3)
OFFSETS = (0, 1)


def pick(photos: list[Path], keep: int, offset: int) -> list[Path]:
    if len(photos) <= keep:
        return photos
    step = (len(photos) - offset) / keep
    return [photos[offset + int(k * step)] for k in range(keep)]


def make_variant(capture: Path, keep: int, offset: int, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    for room in sorted(d for d in capture.iterdir() if d.is_dir() and image_files(d)):
        (dest / room.name).mkdir(parents=True)
        for photo in pick(image_files(room), keep, offset):
            shutil.copy2(photo, dest / room.name / photo.name)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    pairs = []
    for capture in CAPTURES:
        truth = capture / "ground_truth.json"
        if not truth.exists():
            continue
        for keep in KEEP:
            for offset in OFFSETS:
                name = f"{capture.name}_k{keep}_o{offset}"
                variant = make_variant(capture, keep, offset, ROOT / "data" / "captures" / "thin" / name)
                run = out / name
                if not (run / "plan.json").exists():
                    subprocess.run([str(ROOT / ".venv" / "bin" / "roomscope"), "run", str(variant), "--out", str(run),
                                    "--tier", "photo"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if (run / "plan.json").exists():
                    pairs.append(f"{run / 'plan.json'}:{truth}")
                    print(f"{name}: done", file=sys.stderr)
                else:
                    print(f"{name}: no plan", file=sys.stderr)
    print(" ".join(pairs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
