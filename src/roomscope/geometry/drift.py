"""Accumulated-drift correction for a continuous multi-room scan.

The phone's visual-inertial odometry (VIO) is excellent over a second or two and drifts
slowly over tens of metres. Depth geometry is the opposite: registering two views of the
same surfaces pins their relative pose to millimetres, but chaining registrations frame
after frame random-walks just like odometry does. So each source is used for what it is
good at:

  - The scan is cut into short fragments (FRAGMENT_SECONDS). Inside a fragment VIO is
    trusted as-is.
  - Odometry edges link consecutive fragments with the VIO relative motion and a
    covariance from a stated drift model (VIO_* constants).
  - Loop-closure edges link any two fragments that see the same surfaces (repeat passes,
    the end of a room loop against its start, the end of the scan against its start),
    measured by point-to-plane ICP. Their information comes from the ICP Hessian, so a
    corridor match is confident across the corridor and weak along it, normalised so the
    best-constrained direction is trusted to LOOP_SIGMA_M.
  - A 4-DoF pose graph (yaw, x, y, z; roll and pitch are observable from gravity) is
    solved with a Cauchy loss, which down-weights a wrong loop closure instead of letting
    it bend the plan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ..capture import Frame
from .cloud import Cloud, fuse

FRAGMENT_SECONDS = 3.0
FRAGMENT_MAX_DEPTH = 5.0
OVERLAP_CELL = 0.25
OVERLAP_MIN = 0.25
VIO_SIGMA_M_PER_M = 0.01
VIO_SIGMA_FLOOR_M = 0.002
VIO_YAW_DEG_PER_SQRT_M = 0.2
VIO_YAW_FLOOR_DEG = 0.02
LOOP_SIGMA_M = 0.004
DEGENERATE_RATIO = 0.02


def yaw_matrix(yaw: float) -> np.ndarray:
    return Rotation.from_euler("z", yaw).as_matrix()


@dataclass(frozen=True)
class Pose4:
    """World-frame 4-DoF transform: x -> Rz(yaw) x + t."""
    yaw: float
    t: np.ndarray

    @classmethod
    def identity(cls) -> Pose4:
        return cls(0.0, np.zeros(3))

    def compose(self, other: Pose4) -> Pose4:
        return Pose4(self.yaw + other.yaw, yaw_matrix(self.yaw) @ other.t + self.t)

    def inverse(self) -> Pose4:
        return Pose4(-self.yaw, -(yaw_matrix(-self.yaw) @ self.t))

    def matrix(self) -> np.ndarray:
        out = np.eye(4)
        out[:3, :3] = yaw_matrix(self.yaw)
        out[:3, 3] = self.t
        return out


def _wrap(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class IcpResult:
    correction: Pose4
    fitness: float
    rmse: float
    information: np.ndarray
    constrained_dims: int


def icp_4dof(points: np.ndarray, normals: np.ndarray, tree: cKDTree, target_points: np.ndarray,
             target_normals: np.ndarray, distances: tuple[float, ...] = (0.25, 0.10, 0.04, 0.02),
             iterations: int = 8) -> IcpResult:
    """Point-to-plane ICP over (yaw, x, y, z), Tukey-weighted. Updates are solved only in
    eigen-directions the geometry constrains (solution remapping), so unobservable motion
    (sliding along a bare corridor) stays at the odometry value instead of wandering.

    Returns the correction (world frame, about the origin) and its 4x4 information matrix
    in (yaw, x, y, z) about the origin, from the final weighted correspondences."""
    pivot = points.mean(axis=0)
    lever0 = points - pivot
    yaw, shift = 0.0, np.zeros(3)
    residual = weight = jac = None
    dims = 0
    for distance in distances:
        for _ in range(iterations):
            rot = yaw_matrix(yaw)
            lever = lever0 @ rot.T
            moved = pivot + lever + shift
            gap, index = tree.query(moved, distance_upper_bound=distance, workers=-1)
            ok = np.isfinite(gap)
            if ok.sum() < 100:
                return IcpResult(Pose4.identity(), 0.0, np.inf, np.zeros((4, 4)), 0)
            n_target = target_normals[index[ok]]
            compatible = np.sum((normals[ok] @ rot.T) * n_target, axis=1) > 0.7
            sel = np.nonzero(ok)[0][compatible]
            n_target = n_target[compatible]
            residual = np.sum(n_target * (moved[sel] - target_points[index[sel]]), axis=1)
            lv = lever[sel]
            jac = np.column_stack([n_target[:, 1] * lv[:, 0] - n_target[:, 0] * lv[:, 1], n_target])
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
            if abs(step[0]) < 1e-6 and np.linalg.norm(step[1:]) < 1e-5:
                break

    rot = yaw_matrix(yaw)
    moved = pivot + lever0 @ rot.T + shift
    inliers = np.abs(residual) < distances[-1]
    rmse = float(np.sqrt(np.mean(residual[inliers] ** 2))) if inliers.any() else np.inf
    fitness = float(inliers.sum() / len(points))

    # Information about the world origin (how the pose graph parameterises yaw).
    sel_points = moved[np.isfinite(tree.query(moved, distance_upper_bound=distances[-1], workers=-1)[0])]
    gap, index = tree.query(sel_points, workers=-1)
    n_target = target_normals[index]
    jac_origin = np.column_stack([n_target[:, 1] * sel_points[:, 0] - n_target[:, 0] * sel_points[:, 1], n_target])
    information = jac_origin.T @ jac_origin
    trans_max = np.linalg.eigvalsh(information[1:, 1:]).max() if len(sel_points) else 0.0
    if trans_max > 0:
        information *= (1.0 / LOOP_SIGMA_M ** 2) / trans_max

    # Rotation about pivot + shift, re-expressed about the origin.
    correction = Pose4(yaw, pivot - rot @ pivot + shift)
    return IcpResult(correction, fitness, rmse, information, dims)


@dataclass
class Fragment:
    index: int
    frames: list[Frame]
    mid_time: float
    cloud: Cloud
    tree: cKDTree
    centre: np.ndarray
    cells: set


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


def build_fragments(frames: list[Frame], poses: dict[int, np.ndarray]) -> list[Fragment]:
    fragments = []
    for index, group in enumerate(_fragments_of(frames, FRAGMENT_SECONDS)):
        from .cloud import frame_points, voxelize
        cloud = voxelize(Cloud.concat([frame_points(f, poses[f.index], min_confidence=2, stride=2,
                                                    max_depth=FRAGMENT_MAX_DEPTH) for f in group]), 0.025)
        centre = np.mean([poses[f.index][:3, 3] for f in group], axis=0)
        cells = set(map(tuple, np.floor(cloud.points[:, :2] / OVERLAP_CELL).astype(int).tolist()))
        fragments.append(Fragment(index, group, float(np.mean([f.timestamp for f in group])), cloud,
                                  cKDTree(cloud.points), centre, cells))
    return fragments


def _odometry_information(distance: float) -> np.ndarray:
    sigma_t = VIO_SIGMA_FLOOR_M + VIO_SIGMA_M_PER_M * distance
    sigma_yaw = np.deg2rad(VIO_YAW_FLOOR_DEG + VIO_YAW_DEG_PER_SQRT_M * np.sqrt(distance))
    return np.diag([1 / sigma_yaw ** 2, 1 / sigma_t ** 2, 1 / sigma_t ** 2, 1 / (0.5 * sigma_t) ** 2])


@dataclass
class Edge:
    i: int
    j: int
    measurement: Pose4
    sqrt_information: np.ndarray
    loop: bool


def _edge_residual(ci: Pose4, cj: Pose4, edge: Edge) -> np.ndarray:
    # Fragment j's points, moved by the measurement, coincide with fragment i's:
    # Ci . M = Cj, so Cj^-1 . Ci . M should be the identity.
    error = cj.inverse().compose(ci).compose(edge.measurement)
    return edge.sqrt_information @ np.array([_wrap(error.yaw), *error.t])


def _solve(n_nodes: int, edges: list[Edge]) -> list[Pose4]:
    def unpack(x: np.ndarray) -> list[Pose4]:
        nodes = [Pose4.identity()]
        for k in range(n_nodes - 1):
            nodes.append(Pose4(x[4 * k], x[4 * k + 1:4 * k + 4]))
        return nodes

    def residuals(x: np.ndarray) -> np.ndarray:
        nodes = unpack(x)
        return np.concatenate([_edge_residual(nodes[e.i], nodes[e.j], e) for e in edges])

    from scipy.sparse import lil_matrix

    sparsity = lil_matrix((4 * len(edges), 4 * (n_nodes - 1)), dtype=int)
    for row, edge in enumerate(edges):
        for node in (edge.i, edge.j):
            if node > 0:
                sparsity[4 * row:4 * row + 4, 4 * (node - 1):4 * node] = 1

    # Cauchy scale 3: residuals are whitened, so an edge more than ~3 sigma off starts
    # losing influence.
    result = least_squares(residuals, np.zeros(4 * (n_nodes - 1)), loss="cauchy", f_scale=3.0,
                           method="trf", x_scale="jac", jac_sparsity=sparsity, max_nfev=200)
    return unpack(result.x)


def _loop_candidates(fragments: list[Fragment]) -> list[tuple[int, int]]:
    pairs = []
    for i in range(len(fragments)):
        for j in range(i + 2, len(fragments)):
            shared = len(fragments[i].cells & fragments[j].cells)
            smaller = min(len(fragments[i].cells), len(fragments[j].cells))
            if smaller and shared / smaller > OVERLAP_MIN:
                pairs.append((i, j))
    return pairs


ICP_SOURCE_POINTS = 5000


def _register(target: Fragment, source: Fragment) -> IcpResult:
    points, normals = source.cloud.points, source.cloud.normals
    if len(points) > ICP_SOURCE_POINTS:
        pick = np.random.default_rng(source.index).choice(len(points), ICP_SOURCE_POINTS, replace=False)
        points, normals = points[pick], normals[pick]
    return icp_4dof(points, normals, target.tree, target.cloud.points, target.cloud.normals)


def _acceptable(result: IcpResult) -> bool:
    return (result.fitness > 0.3 and result.rmse < 0.015 and result.constrained_dims >= 3
            and np.linalg.norm(result.correction.t) < 0.6 and abs(result.correction.yaw) < np.deg2rad(4))


@dataclass
class DriftResult:
    poses: dict[int, np.ndarray]
    loop_residual_before_m: float | None
    loop_residual_after_m: float | None
    loop_edges: int
    notes: list[str]


def optimise_pose_graph(frames: list[Frame], poses: dict[int, np.ndarray]) -> DriftResult:
    fragments = build_fragments(frames, poses)
    edges = []
    for a, b in zip(fragments[:-1], fragments[1:]):
        info = _odometry_information(float(np.linalg.norm(b.centre - a.centre)))
        edges.append(Edge(a.index, b.index, Pose4.identity(), np.linalg.cholesky(info).T, loop=False))

    before, loops = [], []
    for i, j in _loop_candidates(fragments):
        result = _register(fragments[i], fragments[j])
        if not _acceptable(result):
            continue
        info = result.information + np.eye(4) * 1e-6
        edges.append(Edge(i, j, result.correction, np.linalg.cholesky(info).T, loop=True))
        before.append(float(np.linalg.norm(result.correction.t)))
        loops.append((i, j))

    notes = [f"pose graph: {len(fragments)} fragments of {FRAGMENT_SECONDS:.0f} s, {len(loops)} loop closures"]
    if not loops:
        notes.append("no loop closures found; drift left uncorrected (capture did not revisit any area)")
        return DriftResult(dict(poses), None, None, 0, notes)

    nodes = _solve(len(fragments), edges)

    after = []
    for i, j in loops:
        moved_i = _moved(fragments[i], nodes[i])
        moved_j = _moved(fragments[j], nodes[j])
        check = icp_4dof(moved_j.points, moved_j.normals, cKDTree(moved_i.points), moved_i.points, moved_i.normals)
        after.append(float(np.linalg.norm(check.correction.t)))

    corrected = _interpolate(frames, poses, fragments, nodes)
    return DriftResult(corrected, float(np.median(before)), float(np.median(after)), len(loops), notes)


def _moved(fragment: Fragment, node: Pose4) -> Cloud:
    rot = yaw_matrix(node.yaw)
    return Cloud(fragment.cloud.points @ rot.T + node.t, fragment.cloud.normals @ rot.T, fragment.cloud.frame_ids)


def _interpolate(frames: list[Frame], poses: dict[int, np.ndarray], fragments: list[Fragment],
                 nodes: list[Pose4]) -> dict[int, np.ndarray]:
    """Blend node corrections between fragment mid-times so no frame jumps at a boundary."""
    mids = np.array([f.mid_time for f in fragments])
    yaws = np.unwrap([n.yaw for n in nodes])
    shifts = np.array([n.t for n in nodes])
    corrected = {}
    for frame in frames:
        t = frame.timestamp
        node = Pose4(float(np.interp(t, mids, yaws)), np.array([np.interp(t, mids, shifts[:, k]) for k in range(3)]))
        corrected[frame.index] = node.matrix() @ poses[frame.index]
    return corrected


def correct_drift(frames: list[Frame], enabled: bool = True) -> DriftResult:
    raw = {f.index: f.pose for f in frames}
    if not enabled:
        return DriftResult(raw, None, None, 0, ["drift correction disabled (ablation: poses used as-is)"])
    return optimise_pose_graph(frames, raw)
