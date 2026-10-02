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
    import json
    import time
    from pathlib import Path

    from .io.detect import detect_tier
    from .render import render_plan
    from .schema import validate

    started = time.perf_counter()
    capture = Path(args.capture_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tier = args.tier or detect_tier(capture)
    print(f"roomscope: {capture} -> {out} ({tier} tier)")
    if args.oracle_depth:
        from .benchmark.oracle import OracleDepth
        from .tiers import photo as photo_tier
        photo_tier.depth_source = OracleDepth(capture)
        print("  oracle depth: monocular depth replaced by rendered truth (benchmark ablation)")

    if tier == "lidar":
        from .io.arkitscenes import is_arkitscenes, read_arkitscenes
        from .io.stray import read_stray
        from .tiers.lidar import LidarOptions, run_lidar
        bundle = read_arkitscenes(capture) if is_arkitscenes(capture) else read_stray(capture, cache_dir=out / "cache")
        result = run_lidar(bundle, LidarOptions(drift_correction=not args.no_drift_correction,
                                                 depth_scale=args.depth_scale))
        plan = result.plan
    elif tier == "photo":
        from .tiers.photo import run_photo
        plan = run_photo(capture).plan
    elif tier == "video":
        from .tiers.video import run_video
        plan = run_video(capture).plan
    else:
        print(f"{tier} tier is not wired into the CLI yet (see docs/architecture.md).", file=sys.stderr)
        return 2

    plan.timing_s["total"] = round(time.perf_counter() - started, 2)
    document = plan.to_dict()
    validate(document)
    (out / "plan.json").write_text(json.dumps(document, indent=2))
    render_plan(document, out / "plan.png")
    print(f"  {len(document['rooms'])} rooms, {len(document['adjacency'])} connections, "
          f"footprint {document['property']['footprint_area']['value']:.2f} m2 "
          f"in {plan.timing_s['total']:.0f} s")
    if not document["rooms"]:
        print("  no room was measured: no region the camera stood in was enclosed by walls. "
              "Capture notes:", file=sys.stderr)
        for note in document["capture"]["notes"]:
            print(f"    - {note}", file=sys.stderr)
        print("  The protocol asks for a full turn in each room so every wall is seen "
              "(docs/capture_protocol.md).", file=sys.stderr)
    print(f"  wrote {out / 'plan.json'} and {out / 'plan.png'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="roomscope")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the pipeline on one capture")
    run.add_argument("capture_dir", help="path to a capture (photo folders, video, or LiDAR log)")
    run.add_argument("--out", required=True, help="output run directory")
    run.add_argument("--tier", choices=TIERS, help="force a tier (default: auto-detect)")
    run.add_argument("--no-drift-correction", action="store_true",
                     help="use capture poses as-is (ablation for the drift report)")
    run.add_argument("--depth-scale", type=float,
                     help="LiDAR only: fixed depth scale instead of the device calibration (1.0 = raw)")
    run.add_argument("--oracle-depth", action="store_true",
                     help="synthetic captures only: true depth instead of the depth model (ablation)")
    run.set_defaults(func=_run)

    ev = sub.add_parser("eval", help="score a plan.json against ground truth (gates, intervals)")
    ev.add_argument("plan", help="plan.json from roomscope run")
    ev.add_argument("truth", help="ground truth (.yaml or .json, see benchmark/README.md)")
    ev.add_argument("--out", help="write the full metrics JSON here")
    ev.set_defaults(func=_eval)

    cal = sub.add_parser("calibrate", help="fit per-tier interval multipliers (split conformal) on ground truth")
    cal.add_argument("tier", choices=TIERS)
    cal.add_argument("runs", nargs="+", help="plan.json:truth.yaml pairs")
    cal.add_argument("--out", help="write the calibration table (e.g. calibration/lidar.json)")
    cal.set_defaults(func=_calibrate)

    sim = sub.add_parser("sim", help="generate a synthetic capture with exact ground truth (dev tool)")
    sim.add_argument("scene", help="scene spec YAML (see benchmark/sim/)")
    sim.add_argument("--tier", choices=TIERS, required=True)
    sim.add_argument("--out", required=True, help="capture directory to write")
    sim.add_argument("--seed", type=int, default=0)
    sim.add_argument("--drift", type=float, default=1.0, help="odometry drift level (lidar tier; 0 = perfect)")
    sim.add_argument("--low-light", action="store_true", help="underexposed, noisier frames")
    sim.add_argument("--rgb-width", type=int, default=1920, help="lidar tier RGB width (4:3); smaller is faster")
    sim.set_defaults(func=_sim)

    return parser


def _eval(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from .benchmark.evaluate import evaluate, load_ground_truth, summary_table

    result = evaluate(json.loads(Path(args.plan).read_text()), load_ground_truth(args.truth))
    print(summary_table(result))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=float))
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    import json

    from .benchmark.calibrate import run

    print(json.dumps(run(args.tier, args.runs, args.out), indent=2))
    return 0


def _sim(args: argparse.Namespace) -> int:
    if str(args.scene).endswith(".ply"):
        # A scanned room (e.g. Replica): photo and video protocol captures, truth from the mesh.
        from pathlib import Path

        from .sim import mesh_capture
        if args.tier not in ("photo", "video"):
            print("mesh scenes give photo and video captures", file=sys.stderr)
            return 2
        writer = mesh_capture.write_photos if args.tier == "photo" else mesh_capture.write_video
        truth = writer(Path(args.scene), Path(args.out), seed=args.seed)
        print(f"wrote {args.tier} capture of {truth['id']} to {args.out}")
        return 0
    from pathlib import Path

    from .sim import writers
    from .sim.scene import load_scene
    from .sim.trajectory import default_route

    exposure, noise = (0.35, 0.035) if args.low_light else (1.0, 0.01)
    out = Path(args.out)
    if args.tier == "photo":
        writers.write_photo_tier(args.scene, out, seed=args.seed, exposure=exposure, noise=noise)
    elif args.tier == "video":
        route = default_route(load_scene(args.scene))
        writers.write_video_tier(args.scene, out, route, seed=args.seed, exposure=exposure, noise=noise)
    else:
        route = default_route(load_scene(args.scene))
        writers.write_lidar_tier(args.scene, out, route, seed=args.seed, drift_level=args.drift,
                                 rgb_size=(args.rgb_width, args.rgb_width * 3 // 4), exposure=exposure, noise=noise)
    print(f"wrote {args.tier} capture to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
