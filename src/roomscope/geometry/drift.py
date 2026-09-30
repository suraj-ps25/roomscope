"""Accumulated-drift correction for a continuous multi-room scan.

Stage 1, heading: every short fragment measures the dominant wall direction from its own
wall normals. Heading drift shows up as that direction slowly rotating over time; we
estimate it per fragment and re-integrate the odometry with the heading error removed
step by step. Rooms need not be rectangular; only fragments whose walls agree with the
global axes within HEADING_ACCEPT_DEG vote.

Stage 2, translation: fragments that overlap (consecutive ones, repeat passes through the
hallway, end-of-scan vs start) are registered with point-to-plane ICP, and a 4-DoF pose
graph (x, y, z, yaw; roll/pitch are observable from gravity) is optimised with a robust
line process that prunes wrong loop closures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ..capture import Frame
from .cloud import Cloud, fuse

FRAGMENT_SECONDS = 6.0
HEADING_ACCEPT_DEG = 6.0
OVERLAP_CELL = 0.25


@dataclass
class Fragment:
    index: int
    frame_indices: list[int]
    mid_time: float
    cloud: Cloud
    cells: set = field(default_factory=set)


@dataclass
class DriftResult:
    poses: dict[int, np.ndarray]
    heading_correction_deg: np.ndarray
    loop_residual_before_m: float | None
    loop_residual_after_m: float | None
    loop_edges: int
    notes: list[str]


def yaw_matrix(yaw: float) -> np.ndarray:
    return Rotation.from_euler("z", yaw).as_matrix()


def dominant_wall_angle(normals: np.ndarray) -> tuple[float, float]:
    """Dominant wall direction modulo 90 deg, via the circular mean of 4*theta, and how
    strongly the normals agree with it (1 = perfectly Manhattan)."""
    horizontal = normals[np.abs(normals[:, 2]) < 0.2]
    if len(horizontal) < 200:
        return 0.0, 0.0
    theta = np.arctan2(horizontal[:, 1], horizontal[:, 0])
    c, s = np.cos(4 * theta).sum(), np.sin(4 * theta).sum()
    return float(np.arctan2(s, c) / 4), float(np.hypot(c, s) / len(theta))


def _wrap_quarter(angle: float) -> float:
    return (angle + np.pi / 4) % (np.pi / 2) - np.pi / 4


def _fragments_of(frames: list[Frame], seconds: float) -> list[list[Frame]]:
    groups, current, start = [], [], frames[0].timestamp
    for frame in frames:
        if current and frame.timestamp - start >= seconds:
            groups.append(current)
            current, start = [], frame.timestamp
        current.append(frame)
    if current:
        if groups and len(current) < 0.4 * len(groups[-1]):
            groups[-1].extend(current)
        else:
            groups.append(current)
    return groups


def correct_heading(frames: list[Frame], poses: dict[int, np.ndarray]) -> tuple[dict[int, np.ndarray], np.ndarray, list[str]]:
    groups = _fragments_of(frames, FRAGMENT_SECONDS)
    angles, strengths, times = [], [], []
    for group in groups:
        cloud = fuse(group, poses, voxel=0.04, stride=4)
        angle, strength = dominant_wall_angle(cloud.normals)
        angles.append(angle)
        strengths.append(strength)
        times.append(np.mean([f.timestamp for f in group]))
    angles, strengths, times = np.array(angles), np.array(strengths), np.array(times)

    weights = strengths ** 2
    reference = np.arctan2((weights * np.sin(4 * angles)).sum(), (weights * np.cos(4 * angles)).sum()) / 4
    residual = np.array([_wrap_quarter(a - reference) for a in angles])
    valid = (strengths > 0.35) & (np.abs(residual) < np.deg2rad(HEADING_ACCEPT_DEG))
    notes = [f"heading: {valid.sum()}/{len(valid)} fragments voted"]
    if valid.sum() < 2:
        notes.append("heading: too few Manhattan-consistent fragments; heading left as-is")
        return dict(poses), np.zeros(len(frames)), notes

    # Heading error is a slow random walk: interpolate between voting fragments, smooth,
    # and remove it step by step.
    per_fragment = np.interp(times, times[valid], residual[valid])
    per_fragment = gaussian_filter1d(per_fragment, sigma=0.8, mode="nearest")
    frame_times = np.array([f.timestamp for f in frames])
    per_frame = np.interp(frame_times, times, per_fragment)

    corrected = {}
    previous_raw = previous_new = None
    for frame, error in zip(frames, per_frame):
        raw = poses[frame.index]
        fix = yaw_matrix(-error)
        new = np.eye(4)
        new[:3, :3] = fix @ raw[:3, :3]
        new[:3, 3] = raw[:3, 3] if previous_raw is None else \
            previous_new[:3, 3] + fix @ (raw[:3, 3] - previous_raw[:3, 3])
        corrected[frame.index] = new
        previous_raw, previous_new = raw, new
    return corrected, np.rad2deg(per_frame), notes


def _to_4dof(transform: np.ndarray) -> np.ndarray:
    yaw = np.arctan2(transform[1, 0], transform[0, 0])
    out = np.eye(4)
    out[:3, :3] = yaw_matrix(yaw)
    out[:3, 3] = transform[:3, 3]
    return out


def _register(source: Cloud, target: Cloud) -> tuple[np.ndarray, float, float, np.ndarray]:
    src, tgt = source.to_o3d(), target.to_o3d()
    transform = np.eye(4)
    result = None
    for distance in (0.15, 0.07, 0.035):
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(k=distance))
        result = o3d.pipelines.registration.registration_icp(
            src, tgt, distance, transform, estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
        transform = _to_4dof(result.transformation)
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(src, tgt, 0.035, transform)
    return transform, float(result.fitness), float(result.inlier_rmse), info


def _build_fragments(frames: list[Frame], poses: dict[int, np.ndarray]) -> list[Fragment]:
    fragments = []
    for index, group in enumerate(_fragments_of(frames, FRAGMENT_SECONDS)):
        cloud = fuse(group, poses, voxel=0.03, stride=2)
        cells = set(map(tuple, np.floor(cloud.points[:, :2] / OVERLAP_CELL).astype(int).tolist()))
        fragments.append(Fragment(index, [f.index for f in group],
                                  float(np.mean([f.timestamp for f in group])), cloud, cells))
    return fragments


def _loop_pairs(fragments: list[Fragment]) -> list[tuple[int, int]]:
    pairs = []
    for i in range(len(fragments)):
        for j in range(i + 2, len(fragments)):
            shared = len(fragments[i].cells & fragments[j].cells)
            smaller = min(len(fragments[i].cells), len(fragments[j].cells))
            if smaller and shared / smaller > 0.3:
                pairs.append((i, j))
    return pairs


def _apply(transform: np.ndarray, cloud: Cloud) -> Cloud:
    return Cloud(cloud.points @ transform[:3, :3].T + transform[:3, 3], cloud.normals @ transform[:3, :3].T,
                 cloud.frame_ids)


def optimise_pose_graph(frames: list[Frame], poses: dict[int, np.ndarray]) -> tuple[dict[int, np.ndarray], float | None, float | None, int, list[str]]:
    fragments = _build_fragments(frames, poses)
    reg = o3d.pipelines.registration
    graph = reg.PoseGraph()
    for _ in fragments:
        graph.nodes.append(reg.PoseGraphNode(np.eye(4)))

    for i in range(len(fragments) - 1):
        transform, fitness, rmse, info = _register(fragments[i].cloud, fragments[i + 1].cloud)
        if fitness < 0.2 or np.linalg.norm(transform[:3, 3]) > 0.3:
            transform, info = np.eye(4), np.eye(6) * 1e3
        graph.edges.append(reg.PoseGraphEdge(i, i + 1, transform, info, uncertain=False))

    loop_before = []
    accepted = []
    for i, j in _loop_pairs(fragments):
        transform, fitness, rmse, info = _register(fragments[i].cloud, fragments[j].cloud)
        yaw = abs(np.rad2deg(np.arctan2(transform[1, 0], transform[0, 0])))
        if fitness > 0.3 and rmse < 0.025 and np.linalg.norm(transform[:3, 3]) < 0.8 and yaw < 4:
            graph.edges.append(reg.PoseGraphEdge(i, j, transform, info, uncertain=True))
            loop_before.append(float(np.linalg.norm(transform[:3, 3])))
            accepted.append((i, j))

    notes = [f"pose graph: {len(fragments)} fragments, {len(accepted)} loop closures"]
    if not accepted:
        notes.append("pose graph: no loop closures found; translation drift uncorrected")
        return dict(poses), None, None, 0, notes

    reg.global_optimization(
        graph, reg.GlobalOptimizationLevenbergMarquardt(), reg.GlobalOptimizationConvergenceCriteria(),
        reg.GlobalOptimizationOption(max_correspondence_distance=0.035, edge_prune_threshold=0.25,
                                     preference_loop_closure=2.0, reference_node=0))
    corrections = [_to_4dof(node.pose) for node in graph.nodes]

    loop_after = []
    for i, j in accepted:
        src = _apply(corrections[i], fragments[i].cloud)
        tgt = _apply(corrections[j], fragments[j].cloud)
        transform, _, _, _ = _register(src, tgt)
        loop_after.append(float(np.linalg.norm(transform[:3, 3])))

    # Blend corrections between fragment centres so frames never jump at a boundary.
    mids = np.array([f.mid_time for f in fragments])
    yaws = np.unwrap([np.arctan2(c[1, 0], c[0, 0]) for c in corrections])
    shifts = np.array([c[:3, 3] for c in corrections])
    corrected = {}
    for frame in frames:
        t = frame.timestamp
        yaw = np.interp(t, mids, yaws)
        shift = np.array([np.interp(t, mids, shifts[:, k]) for k in range(3)])
        fix = np.eye(4)
        fix[:3, :3] = yaw_matrix(yaw)
        fix[:3, 3] = shift
        corrected[frame.index] = fix @ poses[frame.index]
    return corrected, float(np.mean(loop_before)), float(np.mean(loop_after)), len(accepted), notes


TRACK_RADIUS = 5.0
TRACK_VOXEL = 0.03
TRACK_MAX_DEPTH = 3.0
TRACK_REBUILD_EVERY = 5
DEGENERATE_RATIO = 0.02


@dataclass
class IcpResult:
    yaw: float
    shift: np.ndarray
    fitness: float
    rmse: float
    constrained_dims: int

    def matrix(self, pivot: np.ndarray) -> np.ndarray:
        """Correction as a world transform: rotate by yaw about pivot, then shift."""
        rot = yaw_matrix(self.yaw)
        out = np.eye(4)
        out[:3, :3] = rot
        out[:3, 3] = pivot - rot @ pivot + self.shift
        return out


def icp_4dof(points: np.ndarray, normals: np.ndarray, tree: cKDTree, model_points: np.ndarray,
             model_normals: np.ndarray, pivot: np.ndarray,
             distances: tuple[float, ...] = (0.08, 0.035, 0.02), iterations: int = 5) -> IcpResult:
    """Point-to-plane ICP over (yaw, x, y, z) about `pivot`, Tukey-weighted, with
    degeneracy-aware updates: the step is solved only in eigen-directions of the normal
    matrix the geometry constrains (solution remapping), everything else stays put."""
    yaw, shift = 0.0, np.zeros(3)
    residual = np.zeros(0)
    matched = 0
    dims = 0
    lever0 = points - pivot
    for distance in distances:
        for _ in range(iterations):
            rot = yaw_matrix(yaw)
            lever = lever0 @ rot.T
            moved = pivot + lever + shift
            gap, index = tree.query(moved, distance_upper_bound=distance, workers=-1)
            ok = np.isfinite(gap)
            if ok.sum() < 50:
                return IcpResult(0.0, np.zeros(3), 0.0, np.inf, 0)
            n_model = model_normals[index[ok]]
            compatible = np.sum((normals[ok] @ rot.T) * n_model, axis=1) > 0.7
            sel = np.nonzero(ok)[0][compatible]
            n_model = n_model[compatible]
            q = model_points[index[sel]]
            residual = np.sum(n_model * (moved[sel] - q), axis=1)
            lv = lever[sel]
            jac = np.column_stack([n_model[:, 1] * lv[:, 0] - n_model[:, 0] * lv[:, 1], n_model])
            c = distance / 2
            weight = np.where(np.abs(residual) < c, (1 - (residual / c) ** 2) ** 2, 0.0)
            hessian = (jac * weight[:, None]).T @ jac
            gradient = (jac * weight[:, None]).T @ residual
            values, vectors = np.linalg.eigh(hessian)
            keep = values > DEGENERATE_RATIO * max(values.max(), 1e-12)
            dims = int(keep.sum())
            step = -vectors[:, keep] @ ((vectors[:, keep].T @ gradient) / values[keep])
            yaw += step[0]
            shift += step[1:]
            matched = len(sel)
            if abs(step[0]) < 1e-5 and np.linalg.norm(step[1:]) < 1e-4:
                break
    inliers = np.abs(residual) < distances[-1]
    rmse = float(np.sqrt(np.mean(residual[inliers] ** 2))) if inliers.any() else np.inf
    return IcpResult(float(yaw), shift, matched / len(points), rmse, dims)


class _LocalMap:
    """Voxelised map of everything tracked so far, with a KD-tree over the part near the
    camera. Rebuilt every few frames; new frames are appended in between."""

    def __init__(self):
        self.points = np.zeros((0, 3))
        self.normals = np.zeros((0, 3))
        self.pending: list[Cloud] = []
        self.tree: cKDTree | None = None
        self.local_points = self.local_normals = None

    def add(self, cloud: Cloud) -> None:
        self.pending.append(cloud)

    def rebuild(self, centre: np.ndarray) -> None:
        from .cloud import voxelize
        merged = voxelize(Cloud.concat([Cloud(self.points, self.normals, np.zeros(len(self.points), np.int32)),
                                        *self.pending]), TRACK_VOXEL)
        self.points, self.normals, self.pending = merged.points, merged.normals, []
        near = np.linalg.norm(self.points - centre, axis=1) < TRACK_RADIUS
        self.local_points, self.local_normals = self.points[near], self.normals[near]
        self.tree = cKDTree(self.local_points) if near.sum() > 0 else None


def track_frame_to_model(frames: list[Frame], poses: dict[int, np.ndarray]) -> tuple[dict[int, np.ndarray], list[str]]:
    """Register each frame to the map built so far, seeded by the odometry step since the
    previous frame. Revisited areas therefore snap back onto the geometry already mapped."""
    from .cloud import frame_points

    tracked: dict[int, np.ndarray] = {}
    world = _LocalMap()
    accepted = 0
    previous = None
    for count, frame in enumerate(frames):
        odom = poses[frame.index]
        guess = odom if previous is None else tracked[previous[0]] @ np.linalg.inv(previous[1]) @ odom
        pose = guess
        if count and count % TRACK_REBUILD_EVERY == 1:
            world.rebuild(guess[:3, 3])
        # Close-range returns only: iPhone LiDAR is accurate to ~1 cm within 2-3 m and
        # degrades quickly beyond.
        local = frame_points(frame, np.eye(4), min_confidence=2, stride=4, max_depth=TRACK_MAX_DEPTH)
        if world.tree is not None and len(world.local_points) > 2000 and len(local) > 300:
            source = _apply(guess, local)
            result = icp_4dof(source.points, source.normals, world.tree, world.local_points,
                              world.local_normals, guess[:3, 3])
            if (result.fitness > 0.35 and result.rmse < 0.012 and np.linalg.norm(result.shift) < 0.08
                    and abs(result.yaw) < np.deg2rad(1.5)):
                pose = result.matrix(guess[:3, 3]) @ guess
                accepted += 1
        tracked[frame.index] = pose
        previous = (frame.index, odom)
        world.add(_apply(pose, local))
        if count == 0:
            world.rebuild(pose[:3, 3])
    return tracked, [f"frame-to-model tracking: {accepted}/{len(frames) - 1} frames refined"]


def correct_drift(frames: list[Frame], enabled: bool = True, heading: bool = True,
                  tracking: bool = True, pose_graph: bool = True) -> DriftResult:
    raw = {f.index: f.pose for f in frames}
    if not enabled:
        return DriftResult(raw, np.zeros(len(frames)), None, None, 0, ["drift correction disabled (ablation)"])
    notes: list[str] = []
    poses, heading_deg = raw, np.zeros(len(frames))
    if heading:
        poses, heading_deg, heading_notes = correct_heading(frames, poses)
        notes += heading_notes
    if tracking:
        poses, track_notes = track_frame_to_model(frames, poses)
        notes += track_notes
    before = after = None
    loops = 0
    if pose_graph:
        poses, before, after, loops, graph_notes = optimise_pose_graph(frames, poses)
        notes += graph_notes
    return DriftResult(poses, heading_deg, before, after, loops, notes)
