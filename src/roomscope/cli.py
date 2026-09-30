"""roomscope command-line entrypoint.

One command per capture:

    roomscope run <capture_dir> --out runs/<name>

The capture directory's shape selects the tier (photo folders / video clip /
LiDAR log); a tier can be forced with --tier. The run produces plan.json (to
schema/floorplan.schema.json) and plan.png.
"""

from __future__ import annotations

import argparse
import sys

STAGES = [
    "ingest",        # normalize capture -> frames, intrinsics, poses/depth if present
    "reconstruct",   # per-room geometry (SfM / depth fusion / point cloud)
    "measure",       # walls, ceiling height, floor area, openings + intervals
    "stitch",        # place rooms into one plan, adjacency, drift correction
    "damage",        # per-surface damage regions + concealed-damage rules
    "scope",         # scope line items keyed to surfaces
    "render",        # plan.json + plan.png
]

TIERS = ("photo", "video", "lidar")


def _run(args: argparse.Namespace) -> int:
    print(f"capture: {args.capture_dir}")
    print(f"tier:    {args.tier or 'auto (detect from capture layout)'}")
    print(f"out:     {args.out}")
    print("pipeline stages:", " -> ".join(STAGES))
    print("\nnot implemented yet: pipeline is being built tier by tier "
          "(see docs/architecture.md).", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="roomscope")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the pipeline on one capture")
    run.add_argument("capture_dir", help="path to a capture (photo folders, video, or LiDAR log)")
    run.add_argument("--out", required=True, help="output run directory")
    run.add_argument("--tier", choices=TIERS, help="force a tier (default: auto-detect)")
    run.set_defaults(func=_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
