import numpy as np
from scipy.spatial.transform import Rotation

from roomscope.io.video import (_rotation_ransac, find_spins, focal_from_rotation, turn_views)

WIDTH, HEIGHT = 518, 294


def _K(fx):
    return np.array([[fx, 0, (WIDTH - 1) / 2], [0, fx, (HEIGHT - 1) / 2], [0, 0, 1.0]])


def test_focal_from_rotation_recovers_true_focal():
    rng = np.random.default_rng(0)
    true_K = _K(420.0)
    homographies = []
    for _ in range(30):
        R = Rotation.from_rotvec(rng.normal(0, 0.05, 3) + [0, 0.12, 0]).as_matrix()
        H = true_K @ R @ np.linalg.inv(true_K)
        homographies.append(H * (1 + rng.normal(0, 0.002, (3, 3))))
    # Start from a monocular-model guess 15% off.
    estimate = focal_from_rotation(homographies, _K(483.0))
    assert abs(estimate / 420.0 - 1) < 0.01


def test_focal_from_rotation_needs_enough_turns():
    assert focal_from_rotation([np.eye(3)] * 3, _K(420.0)) is None


def _steps(segments):
    """segments: (seconds, deg/step, residual px) at 5 fps."""
    steps, residuals = [], []
    for seconds, rate, residual in segments:
        n = int(seconds * 5)
        steps += [rate] * n
        residuals += [residual] * n
    return np.array(steps, float), np.arange(len(steps)) / 5.0, np.array(residuals, float)


def test_find_spins_finds_one_turn_between_walks():
    steps, t, res = _steps([(6, 1.0, 12.0), (10, 8.0, 0.4), (6, -1.5, 15.0)])
    spins = find_spins(steps, t, res)
    assert len(spins) == 1
    start, end = spins[0]
    assert abs(t[start] - 6.0) < 0.6 and abs(t[end] - 16.0) < 0.6


def test_find_spins_ignores_walking_and_short_glances():
    steps, t, res = _steps([(8, 2.0, 20.0), (2, 8.0, 0.4), (8, -2.0, 25.0)])
    assert find_spins(steps, t, res) == []


def test_find_spins_bridges_featureless_dropouts():
    steps, t, res = _steps([(3, 0.5, 10.0), (4, 8.0, 0.4), (0.6, 0.0, np.inf), (5, 8.0, 0.4), (3, 0.5, 10.0)])
    assert len(find_spins(steps, t, res)) == 1


def test_rotation_ransac_rejects_outliers():
    rng = np.random.default_rng(1)
    R = Rotation.from_rotvec([0.02, 0.15, -0.01]).as_matrix()
    a = rng.normal(size=(200, 3))
    a[:, 2] = np.abs(a[:, 2]) + 2
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = a @ R.T
    b[:60] = rng.normal(size=(60, 3))
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    estimate, inliers = _rotation_ransac(a, b, 0.003, rng)
    assert inliers >= 140
    assert np.degrees(Rotation.from_matrix(estimate.T @ R).magnitude()) < 0.05


def test_turn_views_are_even_in_heading_despite_tilt():
    rotations = {}
    for k in range(80):
        heading = np.radians(9.0 * k)
        pitch = np.radians(25) * np.tanh((heading - 2 * np.pi) / 1.0)
        # camera-to-world, OpenCV camera: yaw about world -y (camera y is down), then pitch.
        rotations[k] = Rotation.from_euler("yx", [heading, -pitch]).as_matrix()
    views = turn_views(rotations, per_revolution=14)
    headings = np.array([9.0 * k for k in views])
    gaps = np.diff(headings)
    # 720 degrees less the trimmed turn ends, at 14 views per revolution.
    assert len(views) >= 24
    assert gaps.max() - gaps.min() <= 18.0
