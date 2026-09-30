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
from ..build import LIDAR, OpeningGeometry, RoomGeometry, build_plan
from ..capture import CaptureBundle, Frame
from ..geometry.cloud import fuse
from ..geometry.drift import correct_drift
from ..geometry.layout import RoomLayout, room_layout
from ..geometry.openings import detect_openings, match_doors
from ..geometry.planes import plane_adjust
from ..geometry.rooms import align, segment_rooms

KEYFRAME_FPS = 6.0
PLANE_ROUNDS = 2
MEASURE_VOXEL = 0.02


@dataclass
class LidarOptions:
    drift_correction: bool = True
    plane_rounds: int = PLANE_ROUNDS
    keyframe_fps: float = KEYFRAME_FPS


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


def run_lidar(bundle: CaptureBundle, options: LidarOptions | None = None) -> LidarResult:
    options = options or LidarOptions()
    clock = _Clock()
    frames = select_keyframes(bundle.frames, options.keyframe_fps)
    clock.lap("ingest")

    drift = correct_drift(frames, enabled=options.drift_correction)
    clock.lap("drift")

    cloud = fuse(frames, drift.poses, voxel=MEASURE_VOXEL, stride=2, min_confidence=2)
    alignment = align(cloud)
    plan_cloud = alignment.apply(cloud)
    poses = {k: alignment.pose(v) for k, v in drift.poses.items()}
    regions, segment_notes = segment_rooms(plan_cloud, frames, poses)
    clock.lap("reconstruct")

    drift_notes = list(drift.notes)
    if options.drift_correction:
        for _ in range(options.plane_rounds):
            layouts = [room_layout(region, plan_cloud) for region in regions]
            poses, notes = plane_adjust(frames, poses, [l for l in layouts if l is not None],
                                        {region.id: region for region in regions})
            drift_notes += notes
            plan_cloud = fuse(frames, poses, voxel=MEASURE_VOXEL, stride=2, min_confidence=2)
    clock.lap("stitch")

    layouts = {}
    for region in regions:
        layout = room_layout(region, plan_cloud)
        if layout is not None:
            layouts[region.id] = layout
    clock.lap("measure")

    openings_by_room, opening_notes = {}, []
    for region in regions:
        if region.id not in layouts:
            continue
        found, notes = detect_openings(layouts[region.id], region, frames, poses)
        openings_by_room[region.id] = found
        opening_notes += [f"{region.id}: {n}" for n in notes]
    match_doors([o for found in openings_by_room.values() for o in found], layouts)
    clock.lap("openings")

    key_of = {id(o): (rid, k) for rid, found in openings_by_room.items() for k, o in enumerate(found)}
    geometries = [_geometry(layouts[rid], openings_by_room.get(rid, []), key_of) for rid in layouts]
    report = model.DriftReport(
        method="4-DoF pose graph with ICP loop closures, then plane-anchored bundle adjustment"
               if options.drift_correction else "none (poses used as-is; ablation)",
        enabled=options.drift_correction,
        residual_before_m=None if drift.loop_residual_before_m is None else round(drift.loop_residual_before_m, 4),
        residual_after_m=None if drift.loop_residual_after_m is None else round(drift.loop_residual_after_m, 4),
        notes=drift_notes)
    plan = build_plan(bundle.capture_id, LIDAR, geometries, report, source_app=bundle.source_app,
                      frames_used=len(frames), capture_notes=bundle.notes + segment_notes + opening_notes)
    clock.lap("assemble")
    plan.timing_s = clock.timing
    return LidarResult(plan, layouts, poses, frames, clock.timing)


def _geometry(layout: RoomLayout, openings, key_of: dict[int, tuple[str, int]]) -> RoomGeometry:
    converted = []
    for o in openings:
        partner = key_of.get(id(o.partner)) if o.partner is not None else None
        converted.append(OpeningGeometry(o.wall, o.kind, o.u0, o.u1, o.v0, o.v1, o.sigma_u0, o.sigma_u1,
                                         o.sigma_v0, o.sigma_v1, o.confidence, o.connects_to, partner))
    normals = np.array([line.normal for line in layout.lines])
    offsets = np.array([line.offset for line in layout.lines])
    sigmas = np.array([np.hypot(line.sigma, 0.25 * line.spread) for line in layout.lines])
    return RoomGeometry(layout.id, layout.polygon, normals, offsets, sigmas, layout.floor_z, layout.ceiling_z,
                        layout.floor_sigma, layout.ceiling_sigma, converted, notes=layout.notes)
