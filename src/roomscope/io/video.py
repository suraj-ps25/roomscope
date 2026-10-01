"""Video-tier ingest: one handheld walkthrough clip (any iPhone 15+, Camera app).

Frames are decoded by ffmpeg (which applies the clip's rotation tag) at DECODE_FPS and
the working resolution. Keyframes are chosen by motion, from sparse optical flow: a new
keyframe once the view has shifted KEYFRAME_SHIFT of its width, so a slow on-the-spot turn
still gets overlapping keyframes while standing still adds none.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .photos import FULL_FRAME_DIAGONAL_MM, processing_size

DECODE_FPS = 5.0
KEYFRAME_SHIFT = 0.25
MAX_KEYFRAME_GAP_S = 2.0
MIN_KEYFRAME_GAP_S = 0.2


@dataclass
class VideoFrames:
    images: list[np.ndarray]
    timestamps: np.ndarray
    source_size: tuple[int, int]
    focal_35mm: float | None
    device: str | None


def _probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _metadata(info: dict) -> tuple[tuple[int, int], float | None, str | None]:
    stream = next(s for s in info["streams"] if s.get("codec_type") == "video")
    width, height = int(stream["width"]), int(stream["height"])
    rotation = 0
    for side in stream.get("side_data_list", []):
        rotation = int(side.get("rotation", rotation) or rotation)
    rotation = int(stream.get("tags", {}).get("rotate", rotation) or rotation)
    if abs(rotation) % 180 == 90:
        width, height = height, width
    tags = {k.lower(): v for k, v in {**info.get("format", {}).get("tags", {}), **stream.get("tags", {})}.items()}
    focal = next((float(v) for k, v in tags.items() if "focal_length" in k and "35" in k), None)
    device = tags.get("com.apple.quicktime.model")
    return (width, height), focal, device


def read_video(path: str | Path) -> VideoFrames:
    path = Path(path)
    if path.is_dir():
        from .detect import video_files
        path = video_files(path)[0]
    (width, height), focal, device = _metadata(_probe(path))
    w, h = processing_size(width, height)
    decode = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"fps={DECODE_FPS},scale={w}:{h}:flags=area",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    raw = decode.stdout.read()
    decode.wait()
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w, 3)
    timestamps = np.arange(len(frames)) / DECODE_FPS
    return VideoFrames(list(frames), timestamps, (width, height), focal, device)


KEYFRAME_BUDGET = 160


def _frame_shifts(images: list[np.ndarray]) -> np.ndarray:
    """Median sparse optical-flow displacement between consecutive frames, as a fraction
    of the image width."""
    shifts = np.zeros(len(images))
    width = images[0].shape[1]
    grey_prev = cv2.cvtColor(images[0], cv2.COLOR_RGB2GRAY)
    for k in range(1, len(images)):
        grey = cv2.cvtColor(images[k], cv2.COLOR_RGB2GRAY)
        points = cv2.goodFeaturesToTrack(grey_prev, 300, 0.01, 8)
        if points is not None:
            moved, status, _ = cv2.calcOpticalFlowPyrLK(grey_prev, grey, points, None)
            ok = status.ravel() == 1
            if ok.sum() > 10:
                shifts[k] = float(np.median(np.linalg.norm((moved - points)[ok].reshape(-1, 2), axis=1))) / width
        grey_prev = grey
    return shifts


def _pick(shifts: np.ndarray, timestamps: np.ndarray, threshold: float) -> list[int]:
    chosen, accumulated = [0], 0.0
    for k in range(1, len(shifts)):
        accumulated += shifts[k]
        gap = timestamps[k] - timestamps[chosen[-1]]
        if (accumulated >= threshold and gap >= MIN_KEYFRAME_GAP_S) or gap >= MAX_KEYFRAME_GAP_S:
            chosen.append(k)
            accumulated = 0.0
    return chosen


def select_keyframes(images: list[np.ndarray], timestamps: np.ndarray, budget: int = KEYFRAME_BUDGET) -> list[int]:
    """Motion-based keyframes, with the shift threshold raised until the count fits the
    budget, so runtime stays predictable however much the walker sweeps the phone."""
    shifts = _frame_shifts(images)
    threshold = KEYFRAME_SHIFT
    chosen = _pick(shifts, timestamps, threshold)
    while len(chosen) > budget and threshold < 2.0:
        threshold *= 1.2
        chosen = _pick(shifts, timestamps, threshold)
    return chosen


def intrinsics_from_focal(focal_35mm: float, source_size: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
    width, height = source_size
    w, h = size
    fx = focal_35mm * np.hypot(width, height) / FULL_FRAME_DIAGONAL_MM
    return np.array([[fx * w / width, 0, (w - 1) / 2], [0, fx * h / height, (h - 1) / 2], [0, 0, 1.0]])


SPIN_MIN_DEG = 280.0
# Up to two revolutions (the video protocol's down-then-up double turn) plus overlap.
SPIN_FULL_DEG = 760.0
SPIN_MIN_SECONDS = 5.5
STANDING_STILL_PX = 1.0
SPIN_MAX_SECONDS = 40.0
SPIN_STEP_DEG = 2.5
SPIN_MAX_DROPOUTS = 6
ROTATION_ONLY_PX = 3.0


def yaw_steps(images: list[np.ndarray], K: np.ndarray, homographies: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rotation about the camera's vertical axis between consecutive frames, and how well a
    pure rotation explains the tracked features (median reprojection residual, px).
    Standing still and turning, a rotation explains every point; walking, parallax leaves
    pixels of residual. The yaw comes from a 3-DoF rotation fit (robust on low-texture walls where an
    8-DoF homography degenerates); homographies are kept for focal self-calibration."""
    steps = np.zeros(len(images))
    residuals = np.full(len(images), np.inf)
    K_inv = np.linalg.inv(K)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    rng = np.random.default_rng(0)
    grey_prev = clahe.apply(cv2.cvtColor(images[0], cv2.COLOR_RGB2GRAY))
    for k in range(1, len(images)):
        grey = clahe.apply(cv2.cvtColor(images[k], cv2.COLOR_RGB2GRAY))
        points = cv2.goodFeaturesToTrack(grey_prev, 400, 0.01, 7)
        if points is not None and len(points) >= 12:
            moved, status, _ = cv2.calcOpticalFlowPyrLK(grey_prev, grey, points, None, winSize=(25, 25), maxLevel=4)
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(grey, grey_prev, moved, None, winSize=(25, 25), maxLevel=4)
            # Forward-backward check: a track that does not come back to where it started is
            # drifting on low texture, and would read as parallax.
            round_trip = np.linalg.norm((back - points).reshape(-1, 2), axis=1)
            ok = (status.ravel() == 1) & (back_status.ravel() == 1) & (round_trip < 1.0)
            if ok.sum() >= 12:
                a = np.c_[points[ok].reshape(-1, 2), np.ones(ok.sum())] @ K_inv.T
                b = np.c_[moved[ok].reshape(-1, 2), np.ones(ok.sum())] @ K_inv.T
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                b /= np.linalg.norm(b, axis=1, keepdims=True)
                # The stand-still test uses the homography residual, which does not depend on
                # K (the focal is not calibrated yet when turns are being found).
                H, _ = cv2.findHomography(points[ok], moved[ok], cv2.RANSAC, 3.0)
                if H is not None:
                    projected = cv2.perspectiveTransform(points[ok].reshape(-1, 1, 2), H).reshape(-1, 2)
                    residuals[k] = float(np.median(np.linalg.norm(projected - moved[ok].reshape(-1, 2), axis=1)))
                    if homographies is not None:
                        homographies[k] = H
                R, count = _rotation_ransac(a, b, 3.0 / K[0, 0], rng, iterations=100)
                if R is not None and count >= 8:
                    steps[k] = float(np.arctan2(R[0, 2], R[2, 2]))
        grey_prev = grey
    return np.degrees(steps), residuals


def find_spins(steps_deg: np.ndarray, timestamps: np.ndarray, residuals: np.ndarray) -> list[tuple[int, int]]:
    """The protocol's on-the-spot turn: a run of steady same-direction rotation steps
    (each >= SPIN_STEP_DEG, i.e. a full turn in under ~25 s at 5 fps) with homography-level
    residuals, accumulating >= SPIN_MIN_DEG. Measured on the synthetic walkthrough: turns
    run 7.6 deg/step with sign consistency 1.0; walking 2.3 deg/step at 0.67 and up to
    34 px of parallax residual."""
    # Featureless steps (bare walls) are filled from their neighbours and single-step
    # outliers removed with a 5-step median, so a steady turn reads as a steady rate.
    from scipy.ndimage import median_filter
    valid = np.isfinite(residuals) & (residuals <= ROTATION_ONLY_PX)
    filled = steps_deg.astype(float).copy()
    if valid.sum() >= 2:
        index = np.arange(len(steps_deg))
        filled[~valid] = np.interp(index[~valid], index[valid], steps_deg[valid])
        # Only short gaps are bridged; a long stretch with no rotation-only fit is walking.
        gap_id = np.cumsum(valid)
        for gap in np.unique(gap_id[~valid]):
            members = ~valid & (gap_id == gap)
            if members.sum() > SPIN_MAX_DROPOUTS:
                filled[members] = 0.0
    steps_deg = median_filter(filled, size=5, mode="nearest")
    raw_residuals = residuals
    residuals = np.zeros_like(steps_deg)
    spins, k, n = [], 1, len(steps_deg)
    while k < n:
        sign = np.sign(steps_deg[k])
        if abs(steps_deg[k]) < SPIN_STEP_DEG or residuals[k] > ROTATION_ONLY_PX:
            k += 1
            continue
        start, total, misses, good_steps, j = k, 0.0, 0, 0, k
        while j < n:
            good = abs(steps_deg[j]) >= SPIN_STEP_DEG and np.sign(steps_deg[j]) == sign and residuals[j] <= ROTATION_ONLY_PX
            if good:
                total += steps_deg[j]
                good_steps += 1
                misses = 0
            else:
                misses += 1
                if misses > SPIN_MAX_DROPOUTS:
                    break
                # A turn is steady: bridge a short dropout at the run's own mean rate.
                total += total / max(good_steps, 1)
            if abs(total) >= SPIN_FULL_DEG:
                break
            j += 1
        # The run ends on the step that broke it; trailing bridged dropouts are not part of it.
        j = min(j, n - 1)
        while j > start and misses and not (abs(steps_deg[j]) >= SPIN_STEP_DEG and np.sign(steps_deg[j]) == sign):
            j -= 1
            misses -= 1
        duration = timestamps[j] - timestamps[start - 1]
        run = raw_residuals[start:j + 1]
        still = np.median(run[np.isfinite(run)]) <= STANDING_STILL_PX if np.isfinite(run).any() else False
        if abs(total) >= SPIN_MIN_DEG and SPIN_MIN_SECONDS <= duration <= SPIN_MAX_SECONDS and still:
            spins.append((start - 1, j))
            k = j + 1
        else:
            k += 1
    return spins


def spin_views(steps_deg: np.ndarray, spin: tuple[int, int], count: int = 10) -> list[int]:
    """Frames evenly spaced in yaw around one turn."""
    start, end = spin
    yaw = np.cumsum(np.abs(steps_deg[start:end + 1]))
    # One revolution's worth, evenly spaced; a short turn spreads its views over what it has.
    targets = np.linspace(0, min(yaw[-1], 360.0), count, endpoint=yaw[-1] < 360.0)
    return sorted({start + int(np.argmin(np.abs(yaw - t))) for t in targets})


def _rotation_error(homographies: list[np.ndarray], fx: float, aspect: float, cx: float, cy: float) -> float:
    K = np.array([[fx, 0, cx], [0, fx * aspect, cy], [0, 0, 1.0]])
    K_inv = np.linalg.inv(K)
    errors = []
    for H in homographies:
        R = K_inv @ H @ K
        R /= np.cbrt(np.linalg.det(R))
        errors.append(np.linalg.norm(R.T @ R - np.eye(3)))
    return float(np.median(errors))


def focal_from_rotation(homographies: list[np.ndarray], K: np.ndarray, aspect: float = 1.0) -> float | None:
    """Focal length from a turning camera (Hartley 1997): for pure rotation H = K R K^-1, so
    only the true K makes every K^-1 H K orthonormal. fy = aspect x fx (the decode's resize
    is not exactly uniform) and the principal point at the centre are assumed; the focal is a 1-D search over [0.4, 2.5] x the image width.
    Needs a turn with real rotation: small steps hardly constrain the focal."""
    usable = [H for H in homographies if np.isfinite(H).all()]
    if len(usable) < 8:
        return None
    width = 2 * (K[0, 2] + 0.5)
    candidates = np.geomspace(0.4 * width, 2.5 * width, 160)
    errors = [_rotation_error(usable, f, aspect, K[0, 2], K[1, 2]) for f in candidates]
    best = int(np.argmin(errors))
    if best in (0, len(candidates) - 1):
        return None
    fine = np.geomspace(candidates[best - 1], candidates[best + 1], 60)
    return float(fine[np.argmin([_rotation_error(usable, f, aspect, K[0, 2], K[1, 2]) for f in fine])])


def _orthonormal(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    return R if np.linalg.det(R) > 0 else -R


def _kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation R minimising |b - R a| for unit bearing rows a, b."""
    u, _, vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def _rotation_ransac(a: np.ndarray, b: np.ndarray, threshold: float, rng: np.random.Generator,
                     iterations: int = 300) -> tuple[np.ndarray | None, int]:
    best, best_count = None, 0
    for _ in range(iterations):
        pick = rng.choice(len(a), 2, replace=False)
        R = _kabsch(a[pick], b[pick])
        count = int(np.sum(np.linalg.norm(a @ R.T - b, axis=1) < threshold))
        if count > best_count:
            best, best_count = R, count
    if best is None:
        return None, 0
    inliers = np.linalg.norm(a @ best.T - b, axis=1) < threshold
    return _kabsch(a[inliers], b[inliers]), int(inliers.sum())


def matched_rotation(grey_a: np.ndarray, grey_b: np.ndarray, K: np.ndarray,
                     rng: np.random.Generator) -> tuple[np.ndarray | None, int]:
    """Pure rotation between two frames far apart in time (x_b = R x_a): SIFT matches with
    Lowe's ratio test, then the same 2-point Kabsch RANSAC as frame-to-frame tracking."""
    sift = cv2.SIFT_create(1500, contrastThreshold=0.01)
    # Local contrast equalisation first: painted walls in soft light are very flat.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    ka, da = sift.detectAndCompute(clahe.apply(grey_a), None)
    kb, db = sift.detectAndCompute(clahe.apply(grey_b), None)
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        return None, 0
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 10:
        return None, 0
    K_inv = np.linalg.inv(K)

    def bearings(px):
        rays = np.c_[px, np.ones(len(px))] @ K_inv.T
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    a = bearings(np.array([ka[m.queryIdx].pt for m in good]))
    b = bearings(np.array([kb[m.trainIdx].pt for m in good]))
    return _rotation_ransac(a, b, 2.0 / K[0, 0], rng)


def rotation_between(grey_prev: np.ndarray, grey: np.ndarray, K: np.ndarray, rng: np.random.Generator,
                     iterations: int = 200, allow_translation: bool = True) -> tuple[np.ndarray | None, int]:
    """Pure-rotation fit between two frames: 3 DoF, so far better conditioned on a sparse,
    low-texture wall than an 8-DoF homography. RANSAC over 2-point Kabsch samples on unit
    bearings, refit on inliers (within 1.5 px). Returns (R with x_cur = R x_prev, inliers)."""
    points = cv2.goodFeaturesToTrack(grey_prev, 600, 0.003, 6)
    if points is None or len(points) < 8:
        return None, 0
    moved, status, _ = cv2.calcOpticalFlowPyrLK(grey_prev, grey, points, None, winSize=(25, 25), maxLevel=4)
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(grey, grey_prev, moved, None, winSize=(25, 25), maxLevel=4)
    round_trip = np.linalg.norm((back - points).reshape(-1, 2), axis=1)
    ok = (status.ravel() == 1) & (back_status.ravel() == 1) & (round_trip < 1.0)
    if ok.sum() < 8:
        return None, 0
    K_inv = np.linalg.inv(K)

    def bearings(px):
        rays = np.c_[px.reshape(-1, 2), np.ones(len(px))] @ K_inv.T
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    a, b = bearings(points[ok]), bearings(moved[ok])
    threshold = 1.5 / K[0, 0]
    best, best_count = None, 0
    for _ in range(iterations):
        pick = rng.choice(len(a), 2, replace=False)
        R = _kabsch(a[pick], b[pick])
        count = int(np.sum(np.linalg.norm(a @ R.T - b, axis=1) < threshold))
        if count > best_count:
            best, best_count = R, count
    if best is not None and (best_count >= max(8, 0.6 * len(a)) or (not allow_translation and best_count >= 8)):
        inliers = np.linalg.norm(a @ best.T - b, axis=1) < threshold
        return _kabsch(a[inliers], b[inliers]), int(inliers.sum())
    if not allow_translation:
        return None, best_count
    # Too much parallax for a pure rotation (walking): the essential matrix separates
    # rotation from translation.
    if len(a) < 12:
        return None, best_count
    E, mask = cv2.findEssentialMat(points[ok].reshape(-1, 2), moved[ok].reshape(-1, 2), K, cv2.RANSAC, 0.999, 1.0)
    if E is None or E.shape != (3, 3):
        return (_kabsch(a, b) if best is None else best), best_count
    count, R, _, _ = cv2.recoverPose(E, points[ok].reshape(-1, 2), moved[ok].reshape(-1, 2), K, mask=mask)
    return R, int(count)


def turn_rotations(images: list[np.ndarray], turn: tuple[int, int], K: np.ndarray) -> dict[int, np.ndarray]:
    """Camera-to-world rotations for every frame of one turn (world = the first frame's
    camera), chaining frame-to-frame rotation fits. A step that cannot be fitted is
    bridged by fitting across it (k-1 -> k+1) and halving."""
    start, end = turn
    rng = np.random.default_rng(0)
    greys = {k: cv2.cvtColor(images[k], cv2.COLOR_RGB2GRAY) for k in range(start, min(end + 2, len(images)))}
    rotations = {start: np.eye(3)}
    current = np.eye(3)
    previous_step = np.eye(3)
    sizes = []
    for k in range(start + 1, end + 1):
        R, count = rotation_between(greys[k - 1], greys[k], K, rng, allow_translation=False)
        size = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))) if R is not None else np.inf
        # A turn is steady: a step far off the recent rate, either way, is a bad fit (a blank
        # wall filling the frame reads as no rotation at all), and so is one resting on a
        # handful of tracks. Carry the previous step; averaging refines it below.
        recent = np.median(sizes[-8:]) if len(sizes) >= 3 else None
        if (R is None or count < TURN_MIN_TRACKS
                or (recent is not None and not (TURN_SLOW_SHARE * recent <= size <= max(2.5 * recent, 6.0)))):
            R = previous_step
        else:
            sizes.append(size)
        current = current @ R.T
        previous_step = R
        rotations[k] = current.copy()
    return _average_turn_rotations(_close_turn_loop(rotations, greys, K, rng), greys, K, rng)


AVERAGING_SPAN = 4
TURN_MIN_TRACKS = 30
TURN_SLOW_SHARE = 0.35
TURN_EDGE_FRAMES = 3


def _average_turn_rotations(rotations: dict[int, np.ndarray], greys: dict, K: np.ndarray,
                            rng: np.random.Generator) -> dict[int, np.ndarray]:
    """Rotation averaging over the turn: relative rotations from wide-baseline matches
    between every frame and its next AVERAGING_SPAN frames, plus the loop across 360
    degrees, then robust chordal Gauss-Seidel from the chain estimate. A single bad
    frame-to-frame step no longer propagates into everything after it."""
    frames, heading = _turn_heading(rotations)
    edges = []
    for i, a in enumerate(frames):
        partners = frames[i + 1:i + 1 + AVERAGING_SPAN]
        partners += [b for b, h in zip(frames, heading) if abs(h - heading[i] - 2 * np.pi) < np.radians(15)]
        for b in partners:
            Q, count = matched_rotation(greys[a], greys[b], K, rng)
            if Q is not None and count >= 25:
                edges.append((a, b, Q, float(count)))
    if not edges:
        return rotations
    estimate = dict(rotations)
    for _ in range(15):
        for k in frames[1:]:
            total = np.zeros((3, 3))
            for a, b, Q, count in edges:
                if b == k:
                    proposal = estimate[a] @ Q.T
                elif a == k:
                    proposal = estimate[b] @ Q
                else:
                    continue
                gap = np.linalg.norm(proposal - estimate[k])
                total += count / (1 + (gap / 0.05) ** 2) * proposal
            if np.any(total):
                u, _, vt = np.linalg.svd(total)
                estimate[k] = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    return estimate


def _turn_heading(rotations: dict[int, np.ndarray]) -> tuple[list[int], np.ndarray]:
    """Frames of a turn and each one's heading (radians from the first) about the turn axis."""
    frames = sorted(rotations)
    forward = np.array([rotations[k][:, 2] for k in frames])
    steps = []
    for a, b in zip(frames[:-1], frames[1:]):
        relative = rotations[a].T @ rotations[b]
        axis = np.array([relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0], relative[1, 0] - relative[0, 1]])
        steps.append(rotations[a] @ axis)
    axis = np.sum(steps, axis=0) if steps else np.array([0.0, 1.0, 0.0])
    axis /= max(np.linalg.norm(axis), 1e-12)
    flat = forward - np.outer(forward @ axis, axis)
    reference = flat[0] / max(np.linalg.norm(flat[0]), 1e-12)
    heading = np.unwrap(np.arctan2(flat @ np.cross(axis, reference), flat @ reference))
    return frames, np.abs(heading - heading[0])


def _close_turn_loop(rotations: dict[int, np.ndarray], greys: dict, K: np.ndarray,
                     rng: np.random.Generator) -> dict[int, np.ndarray]:
    """A full turn sees its start again: fit the start frame directly against the frames
    near 360 degrees and spread the chain's discrepancy along the turn by heading."""
    from scipy.spatial.transform import Rotation

    frames, heading = _turn_heading(rotations)
    if heading[-1] < np.radians(350):
        return rotations
    near = [k for k, h in zip(frames, heading) if np.radians(345) <= h <= np.radians(375)]
    best, best_count = None, 0
    for k in near:
        R, count = matched_rotation(greys[frames[0]], greys[k], K, rng)
        if R is not None and count > best_count:
            best, best_count, closing = R, count, k
    if best is None or best_count < 30:
        return rotations
    # Direct estimate of the closing frame's rotation vs the chained one.
    correction = Rotation.from_matrix(best.T @ rotations[closing].T).as_rotvec()
    if np.degrees(np.linalg.norm(correction)) > 15:
        return rotations
    end_heading = heading[frames.index(closing)]
    closed = {}
    for k, h in zip(frames, heading):
        share = min(h / end_heading, 1.0)
        closed[k] = Rotation.from_rotvec(share * correction).as_matrix() @ rotations[k]
    return closed


CHAIN_MAX_FPS = 15.0
CHAIN_LONG_SIDE = 384
MAX_TURN_RATE_DEG_S = 240.0
LINE_MIN_PX = 70.0
LINE_INLIER = np.sin(np.radians(2.5))
MIN_LINE_INLIERS = 12


def line_normals(grey: np.ndarray, K: np.ndarray, detector) -> np.ndarray:
    """Unit normals of the interpretation planes of the frame's line segments (camera
    coordinates): a segment's 3D direction is orthogonal to its normal."""
    lines = detector.detect(grey)[0]
    if lines is None:
        return np.zeros((0, 3))
    lines = lines.reshape(-1, 4)
    lines = lines[np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1]) >= LINE_MIN_PX]
    K_inv = np.linalg.inv(K)
    a = np.c_[lines[:, :2], np.ones(len(lines))] @ K_inv.T
    b = np.c_[lines[:, 2:], np.ones(len(lines))] @ K_inv.T
    normals = np.cross(a, b)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def manhattan_refine(M: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, int]:
    """Refine M (columns = the three Manhattan axes in camera coordinates) so each line is
    orthogonal to its nearest axis: Gauss-Newton on the small rotation, Cauchy-weighted."""
    count = 0
    for _ in range(4):
        if len(normals) < MIN_LINE_INLIERS:
            return M, 0
        dots = normals @ M
        axis = np.argmin(np.abs(dots), axis=1)
        residual = dots[np.arange(len(normals)), axis]
        inlier = np.abs(residual) < 2 * LINE_INLIER
        count = int(np.sum(np.abs(residual) < LINE_INLIER))
        used = np.unique(axis[inlier])
        if inlier.sum() < MIN_LINE_INLIERS or len(used) < 2:
            return M, 0
        J = np.cross(M[:, axis[inlier]].T, normals[inlier])
        weights = 1.0 / (1.0 + (residual[inlier] / LINE_INLIER) ** 2)
        omega = np.linalg.lstsq(J * weights[:, None], -residual[inlier] * weights, rcond=None)[0]
        angle = np.linalg.norm(omega)
        if angle > np.radians(10):
            return M, 0
        if angle > 0:
            k = omega / angle
            Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            M = (np.eye(3) + np.sin(angle) * Kx + (1 - np.cos(angle)) * Kx @ Kx) @ M
        u, _, vt = np.linalg.svd(M)
        M = u @ vt
    return M, count


def _bootstrap_manhattan(normals: np.ndarray) -> np.ndarray:
    """First frame: phone roughly upright (camera -y is up), so search the heading."""
    best, best_count = np.eye(3), -1
    for theta in np.radians(np.arange(0, 90, 2.0)):
        c, s_ = np.cos(theta), np.sin(theta)
        M = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
        M, count = manhattan_refine(M, normals)
        if count > best_count:
            best, best_count = M, count
    return best


def clip_rotations(path: Path, K: np.ndarray, size: tuple[int, int], timestamps: np.ndarray) -> np.ndarray:
    """Camera-to-Manhattan-world rotation at each of `timestamps`, over the whole clip at its
    own frame rate (up to CHAIN_MAX_FPS) so walking turns stay trackable.

    A visual compass: frame-to-frame tracking predicts the rotation, the frame's line
    segments then snap it onto the room's Manhattan axes. Heading therefore cannot drift as
    long as each step is right to well under 45 degrees; frames without enough lines (bare
    walls) coast on tracking alone until lines return."""
    info = _probe(path)
    stream = next(st for st in info["streams"] if st.get("codec_type") == "video")
    num, den = (float(v) for v in stream.get("avg_frame_rate", "30/1").split("/"))
    fps = min(num / den if den else 30.0, CHAIN_MAX_FPS)
    width, height = size
    scale = CHAIN_LONG_SIDE / max(width, height)
    w, h = int(round(width * scale / 2)) * 2, int(round(height * scale / 2)) * 2
    decode = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"fps={fps},scale={w}:{h}:flags=area",
                             "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True)
    greys = np.frombuffer(decode.stdout, np.uint8).reshape(-1, h, w)
    K_small = K.copy()
    K_small[0] *= w / width
    K_small[1] *= h / height
    rng = np.random.default_rng(0)
    detector = cv2.createLineSegmentDetector()
    limit = np.radians(MAX_TURN_RATE_DEG_S / fps)
    M = _bootstrap_manhattan(line_normals(greys[0], K_small, detector))
    chain = [M.T]
    previous_step = np.eye(3)
    for k in range(1, len(greys)):
        R, _ = rotation_between(greys[k - 1], greys[k], K_small, rng)
        if R is None or np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)) > limit:
            R = previous_step
        previous_step = R
        M, _ = manhattan_refine(R @ M, line_normals(greys[k], K_small, detector))
        chain.append(M.T)
    chain_times = np.arange(len(chain)) / fps
    picks = np.clip(np.searchsorted(chain_times, timestamps), 0, len(chain) - 1)
    return np.array([chain[i] for i in picks])


def turn_views(rotations: dict[int, np.ndarray], per_revolution: int = 14) -> list[int]:
    """Frames evenly spaced in heading over the turn (up to two revolutions), from the
    turn's chained rotations.
    Heading is measured about the turn axis (the dominant rotation axis, i.e. vertical),
    so the up-and-down tilting of the protocol does not distort the spacing."""
    frames, heading = _turn_heading(rotations)
    # The ends of a detected turn are where the walker is still stepping in or already
    # stepping out: the camera is off the turning axis there.
    trim = TURN_EDGE_FRAMES if len(frames) > 4 * TURN_EDGE_FRAMES else 0
    frames, heading = frames[trim:len(frames) - trim], heading[trim:len(heading) - trim] - heading[trim]
    total = min(float(np.degrees(heading[-1])), 720.0)
    views = max(4, int(round(per_revolution * total / 360.0)))
    targets = np.radians(np.linspace(0, total, views, endpoint=total not in (360.0, 720.0)))
    return sorted({frames[int(np.argmin(np.abs(heading - t)))] for t in targets})

