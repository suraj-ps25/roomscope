"""Registering a room's photos with metric depth: SIFT matches lifted to 3D, then a
similarity per view (rotation, translation, depth scale) from a robust joint fit.

A learned multi-view model has to infer poses from a handful of wide-baseline photos; here
each photo already carries metric depth, so a matched keypoint is a 3D point in both
views and three of them fix the relative pose and the relative depth scale. Pairwise
3-point RANSAC rejects bad matches, a maximum spanning tree over inlier counts gives the
start, and least squares over all views' inlier points (Cauchy-weighted, sigma growing
with depth) settles everything at once. The scale gauge is the views' geometric mean, so
the room keeps the depth model's average metric scale while per-view scale disagreement
is removed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial.transform import Rotation

RATIO = 0.8
MIN_INLIERS = 20
MIN_INLIER_SHARE = 0.15
RANSAC_ITERATIONS = 600
POINTS_PER_EDGE = 150


@dataclass
class Registration:
    poses: list[np.ndarray]          # camera-to-world 4x4, depth already scaled
    scales: np.ndarray               # per-view depth scale
    registered: list[int]            # views in the largest connected component
    edges: list[tuple[int, int, int]]  # (i, j, inliers)
    rms_m: float


def _features(image: np.ndarray, depth: np.ndarray, mask: np.ndarray, K: np.ndarray):
    grey = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    keypoints, descriptors = cv2.SIFT_create(3000, contrastThreshold=0.01).detectAndCompute(grey, None)
    if descriptors is None:
        return np.zeros((0, 3)), np.zeros((0, 128), np.float32)
    uv = np.array([k.pt for k in keypoints])
    u = np.clip(np.round(uv[:, 0]).astype(int), 0, depth.shape[1] - 1)
    v = np.clip(np.round(uv[:, 1]).astype(int), 0, depth.shape[0] - 1)
    z = depth[v, u]
    # Depth at a keypoint is only trusted away from depth edges (flying pixels).
    patch = np.stack([depth[np.clip(v + dv, 0, depth.shape[0] - 1), np.clip(u + du, 0, depth.shape[1] - 1)]
                      for dv in (-2, 0, 2) for du in (-2, 0, 2)])
    steady = (np.ptp(patch, axis=0) < 0.05 * np.maximum(z, 0.1)) & mask[v, u] & (z > 0.2)
    points = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0] * z, (uv[:, 1] - K[1, 2]) / K[1, 1] * z, z]
    return points[steady], descriptors[steady]


def _umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R, t minimising |target - (s R source + t)|."""
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    a, b = source - mu_s, target - mu_t
    u, singular, vt = np.linalg.svd(b.T @ a / len(source))
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    R = u @ d @ vt
    variance = (a ** 2).sum() / len(source)
    s = float(np.trace(np.diag(singular) @ d) / max(variance, 1e-12))
    return s, R, mu_t - s * R @ mu_s


def _pair(features_i, features_j, rng) -> tuple[np.ndarray, np.ndarray] | None:
    points_i, desc_i = features_i
    points_j, desc_j = features_j
    if len(points_i) < MIN_INLIERS or len(points_j) < MIN_INLIERS:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(desc_i, desc_j, k=2)
    backward = {m.queryIdx: m.trainIdx for m in matcher.match(desc_j, desc_i)}
    pairs = [(m.queryIdx, m.trainIdx) for m, n in (p for p in forward if len(p) == 2)
             if m.distance < RATIO * n.distance and backward.get(m.trainIdx) == m.queryIdx]
    if len(pairs) < MIN_INLIERS:
        return None
    a = points_i[[p for p, _ in pairs]]
    b = points_j[[q for _, q in pairs]]
    tolerance = np.maximum(0.05, 0.04 * a[:, 2])
    best = None
    for _ in range(RANSAC_ITERATIONS):
        pick = rng.choice(len(a), 3, replace=False)
        s, R, t = _umeyama(b[pick], a[pick])
        if not 0.5 < s < 2.0:
            continue
        inliers = np.linalg.norm(a - (s * b @ R.T + t), axis=1) < tolerance
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.sum() < max(MIN_INLIERS, MIN_INLIER_SHARE * len(a)):
        return None
    return a[best], b[best]


def _to_matrix(params: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    return float(np.exp(params[6])), Rotation.from_rotvec(params[:3]).as_matrix(), params[3:6]


def register(images: list[np.ndarray], depths: list[np.ndarray], masks: list[np.ndarray],
             intrinsics: list[np.ndarray], seed: int = 0) -> Registration | None:
    rng = np.random.default_rng(seed)
    n = len(images)
    features = [_features(im, d, m, K) for im, d, m, K in zip(images, depths, masks, intrinsics)]
    matches = {}
    for i in range(n):
        for j in range(i + 1, n):
            found = _pair(features[i], features[j], rng)
            if found is not None:
                matches[(i, j)] = found
    if not matches:
        return None
    weight = np.zeros((n, n))
    for (i, j), (a, _) in matches.items():
        weight[i, j] = weight[j, i] = len(a)
    _, labels = connected_components(weight > 0, directed=False)
    component = max(set(labels), key=lambda c: int(np.sum(labels == c)))
    views = [k for k in range(n) if labels[k] == component]
    if len(views) < 2:
        return None

    # Start: maximum spanning tree (minimum over negated weights) from the best-connected view.
    tree = minimum_spanning_tree(-weight * (labels[:, None] == component) * (labels[None, :] == component)).toarray()
    tree = (tree != 0) | (tree.T != 0)
    root = max(views, key=lambda k: weight[k].sum())
    params = np.zeros((n, 7))
    placed, frontier = {root}, [root]
    while frontier:
        i = frontier.pop()
        for j in np.nonzero(tree[i])[0]:
            if j in placed:
                continue
            a, b = matches[(i, j)] if (i, j) in matches else matches[(j, i)][::-1]
            s, R, t = _umeyama(b, a)       # view j -> view i
            si, Ri, ti = _to_matrix(params[i])
            params[j, :3] = Rotation.from_matrix(Ri @ R).as_rotvec()
            params[j, 3:6] = si * Ri @ t + ti
            params[j, 6] = np.log(si * s)
            placed.add(j)
            frontier.append(j)

    edges = []
    for (i, j), (a, b) in matches.items():
        if i in placed and j in placed:
            pick = rng.choice(len(a), min(len(a), POINTS_PER_EDGE), replace=False)
            edges.append((i, j, a[pick], b[pick]))
    free = [k for k in views if k != root]
    index = {k: m for m, k in enumerate(free)}

    def unpack(x: np.ndarray) -> np.ndarray:
        # The root's pose is the frame of reference; its depth scale is free like the others'.
        full = params.copy()
        for k, m in index.items():
            full[k] = x[7 * m:7 * m + 7]
        full[root, 6] = x[-1]
        return full

    def residuals(x: np.ndarray) -> np.ndarray:
        full = unpack(x)
        out = []
        for i, j, a, b in edges:
            si, Ri, ti = _to_matrix(full[i])
            sj, Rj, tj = _to_matrix(full[j])
            # Divided by the pair's scale: shrinking the whole room would otherwise shrink
            # every residual and win.
            sigma = (0.02 + 0.03 * a[:, 2:3]) * 0.5 * (si + sj)
            out.append(((si * a @ Ri.T + ti) - (sj * b @ Rj.T + tj)) / sigma)
        # Scale gauge: geometric mean of the registered views' scales is 1.
        out.append(np.array([[20.0 * np.mean(full[views, 6])]]))
        return np.concatenate([r.ravel() for r in out])

    x0 = np.concatenate([params[k] for k in free] + [params[root, 6:7]])
    solution = least_squares(residuals, x0, loss="cauchy", f_scale=1.0, max_nfev=200)
    final = unpack(solution.x)
    # The gauge exactly: scale the whole configuration (every depth scale and translation)
    # so the registered views' geometric-mean scale is 1. A similarity of the whole room,
    # so no residual changes.
    shift = float(np.mean(final[views, 6]))
    final[:, 6] -= shift
    final[:, 3:6] *= np.exp(-shift)
    poses, scales = [], np.ones(n)
    for k in range(n):
        s, R, t = _to_matrix(final[k])
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = R, t
        poses.append(pose)
        scales[k] = s
    gaps = []
    for i, j, a, b in edges:
        si, Ri, ti = _to_matrix(final[i])
        sj, Rj, tj = _to_matrix(final[j])
        gaps.append(np.linalg.norm((si * a @ Ri.T + ti) - (sj * b @ Rj.T + tj), axis=1))
    rms = float(np.sqrt(np.median(np.concatenate(gaps) ** 2))) if gaps else float("nan")
    return Registration(poses, scales, sorted(placed), [(i, j, len(a)) for (i, j), (a, _) in matches.items()], rms)
