"""LiDAR tier: depth + poses + intrinsics (Stray Scanner) -> stitched, dimensioned plan.

  keyframes -> pose graph (loop closures) -> fuse -> align (walls on axes, floor z=0)
  -> segment rooms -> layout -> plane-anchored adjustment (x2) -> layout
  -> openings + door pairing -> RoomGeometry -> plan
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .. import model
from ..build import LIDAR, OpeningGeometry, RoomGeometry, TierProfile, build_plan
from ..capture import CaptureBundle, Frame
from ..geometry.cloud import fuse
from ..geometry.drift import LIDAR_DRIFT, DriftSettings, correct_drift
from ..geometry.layout import RoomLayout, room_layout
from ..geometry.openings import carry_unpaired_doors, detect_openings, match_doors
from ..geometry.planes import LIDAR_PLANES, PlaneSettings, plane_adjust
from ..geometry.rooms import align, segment_rooms
from ..geometry.tolerances import LIDAR_TOL, Tolerances
from ..log import log

KEYFRAME_FPS = 6.0
PLANE_ROUNDS = 3
MEASURE_VOXEL = 0.02


@dataclass
class LidarOptions:
    drift_correction: bool = True
    plane_rounds: int = PLANE_ROUNDS
    keyframe_fps: float = KEYFRAME_FPS
    profile: TierProfile = LIDAR
    tol: Tolerances = LIDAR_TOL
    drift: DriftSettings = LIDAR_DRIFT
    planes: PlaneSettings = LIDAR_PLANES
    measure_confidence: int = 2
    select_keyframes: bool = True
    damage: bool = True
    # Depth scale for the capture device; None looks it up in calibration/lidar_depth.json.
    depth_scale: float | None = None


@dataclass
class LidarResult:
    plan: model.Plan
    layouts: dict[str, RoomLayout]
    poses: dict[int, np.ndarray]
    frames: list[Frame]
    timing: dict[str, float] = field(default_factory=dict)


def select_keyframes(frames: list[Frame], fps: float) -> list[Frame]:
    """Uniform in time: the capture app may record at 60 fps; tracking and fusion don't
    need more than a few frames per second of walking."""
    picked, next_time = [], -np.inf
    for frame in sorted(frames, key=lambda f: f.timestamp):
        if frame.timestamp >= next_time:
            picked.append(frame)
            next_time = frame.timestamp + 1.0 / fps - 1e-6
    return picked


class _Clock:
    def __init__(self):
        self.timing: dict[str, float] = {}
        self._start = time.perf_counter()

    def lap(self, stage: str) -> None:
        now = time.perf_counter()
        self.timing[stage] = round(self.timing.get(stage, 0.0) + now - self._start, 2)
        self._start = now


def device_depth_scale(source_app: str | None) -> tuple[float, str | None]:
    """A device's depth-scale correction, measured against laser scans
    (benchmark/real/depth_scale.py). Devices never measured get none: an uncalibrated
    phone is not corrected with another device's bias."""
    import json
    from ..build import CALIBRATION_DIR
    path = CALIBRATION_DIR / "lidar_depth.json"
    if not path.exists() or source_app is None:
        return 1.0, None
    table = json.loads(path.read_text())
    entry = table.get("devices", {}).get(source_app)
    return (float(entry["scale"]), table.get("id")) if entry else (1.0, None)


def _scaled(load, scale: float):
    return lambda: load() * np.float32(scale)


def run_lidar(bundle: CaptureBundle, options: LidarOptions | None = None) -> LidarResult:
    options = options or LidarOptions()
    clock = _Clock()
    scale, table = (options.depth_scale, None) if options.depth_scale is not None else device_depth_scale(bundle.source_app)
    if scale != 1.0:
        for frame in bundle.frames:
            if frame.load_depth is not None:
                frame.load_depth = _scaled(frame.load_depth, scale)
        bundle.notes.append(f"depth scaled by {scale:.4f} for {bundle.source_app} ({table})")
    frames = select_keyframes(bundle.frames, options.keyframe_fps) if options.select_keyframes else list(bundle.frames)
    clock.lap("ingest")
    log("ingest", f"{len(frames)} frames")

    drift = correct_drift(frames, enabled=options.drift_correction, settings=options.drift)
    clock.lap("drift")
    log("drift", drift.notes[-1] if drift.notes else "done")

    cloud = fuse(frames, drift.poses, voxel=MEASURE_VOXEL, stride=2, min_confidence=options.measure_confidence)
    alignment = align(cloud)
    plan_cloud = alignment.apply(cloud)
    poses = {k: alignment.pose(v) for k, v in drift.poses.items()}
    regions, segment_notes = segment_rooms(plan_cloud, frames, poses, min_frames_inside=options.tol.min_room_frames)
    clock.lap("reconstruct")
    log("rooms", f"{len(regions)} rooms segmented")

    drift_notes = list(drift.notes)
    if options.drift_correction:
        found = None
        for round_index in range(options.plane_rounds):
            layouts = {r.id: room_layout(r, plan_cloud, options.tol, absorb_slivers=False) for r in regions}
            layouts = {k: v for k, v in layouts.items() if v is not None}
            if round_index > 0:
                # From the second round on, door and window jambs join the landmarks.
                found = [o for r in regions if r.id in layouts
                         for o in detect_openings(layouts[r.id], r, frames, poses, options.tol, as_landmarks=True)[0]
                         if o.kind != "mirror"]
            poses, notes = plane_adjust(frames, poses, list(layouts.values()),
                                        {region.id: region for region in regions}, found, options.planes)
            drift_notes += notes
            plan_cloud = fuse(frames, poses, voxel=MEASURE_VOXEL, stride=2, min_confidence=options.measure_confidence)
    clock.lap("stitch")
    log("stitch", "plane-anchored adjustment done")

    layouts = {}
    for region in regions:
        layout = room_layout(region, plan_cloud, options.tol)
        if layout is not None:
            layouts[region.id] = layout
    clock.lap("measure")

    openings_by_room, mirrors_by_room, opening_notes = {}, {}, []
    for region in regions:
        if region.id not in layouts:
            continue
        found, notes = detect_openings(layouts[region.id], region, frames, poses, options.tol)
        openings_by_room[region.id] = [o for o in found if o.kind != "mirror"]
        mirrors_by_room[region.id] = [o for o in found if o.kind == "mirror"]
        opening_notes += [f"{region.id}: {n}" for n in notes]
    match_doors([o for found in openings_by_room.values() for o in found], layouts)
    for o in carry_unpaired_doors([o for found in openings_by_room.values() for o in found], layouts):
        openings_by_room[o.room].append(o)
        opening_notes.append(f"{o.room}: door taken from the {o.connects_to} side of the shared doorway ({o.width:.2f} m)")
    clock.lap("openings")
    log("openings", f"{sum(len(v) for v in openings_by_room.values())} openings")

    damage_by_room = {}
    if options.damage:
        from ..damage.stage import analyse_room
        # Decode every keyframe's colour once, up front; surfaces then pick from the cache.
        bundle.warm_rgb(frames)
        for rid, layout in layouts.items():
            neighbours: dict[int, list[str]] = {}
            for o in openings_by_room.get(rid, []):
                if o.connects_to:
                    neighbours.setdefault(o.wall, []).append(o.connects_to)
            damage_by_room[rid] = analyse_room(layout, openings_by_room.get(rid, []), mirrors_by_room.get(rid, []),
                                               frames, poses, rid, neighbours, bundle.warm_rgb)
    clock.lap("damage")
    log("damage", f"{sum(len(d) for r in damage_by_room.values() for d in r.detections.values())} regions, "
                  f"{sum(len(r.flags) for r in damage_by_room.values())} flags")

    key_of = {id(o): (rid, k) for rid, found in openings_by_room.items() for k, o in enumerate(found)}
    geometries = [_geometry(layouts[rid], openings_by_room.get(rid, []), key_of, damage_by_room.get(rid))
                  for rid in layouts]
    report = model.DriftReport(
        method="4-DoF pose graph with ICP loop closures, then plane-anchored bundle adjustment"
               if options.drift_correction else "none (poses used as-is; ablation)",
        enabled=options.drift_correction,
        residual_before_m=None if drift.loop_residual_before_m is None else round(drift.loop_residual_before_m, 4),
        residual_after_m=None if drift.loop_residual_after_m is None else round(drift.loop_residual_after_m, 4),
        notes=drift_notes)
    plan = build_plan(bundle.capture_id, options.profile, geometries, report, source_app=bundle.source_app,
                      frames_used=len(frames), capture_notes=bundle.notes + segment_notes + opening_notes)
    clock.lap("assemble")
    plan.timing_s = clock.timing
    return LidarResult(plan, layouts, poses, frames, clock.timing)


def _geometry(layout: RoomLayout, openings, key_of: dict[int, tuple[str, int]], damage=None) -> RoomGeometry:
    converted = []
    for o in openings:
        partner = key_of.get(id(o.partner)) if o.partner is not None else None
        converted.append(OpeningGeometry(o.wall, o.kind, o.u0, o.u1, o.v0, o.v1, o.sigma_u0, o.sigma_u1,
                                         o.sigma_v0, o.sigma_v1, o.confidence, o.connects_to, partner))
    normals = np.array([line.normal for line in layout.lines])
    offsets = np.array([line.offset for line in layout.lines])
    sigmas = np.array([np.hypot(line.sigma, 0.25 * line.spread) for line in layout.lines])
    return RoomGeometry(layout.id, layout.polygon, normals, offsets, sigmas, layout.floor_z, layout.ceiling_z,
                        layout.floor_sigma, layout.ceiling_sigma, converted, notes=layout.notes, damage=damage)
