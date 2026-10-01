import numpy as np
from scipy.spatial.transform import Rotation

from roomscope.geometry.cloud import Cloud
from roomscope.geometry.layout import RoomLayout, WallLine
from roomscope.geometry.openings import Opening
from roomscope.geometry.sparse_layout import rectangle_layout
from roomscope.geometry.stitch import stitch
from roomscope.tiers.photo import _relative_scales, gravity_rotation


def _plane(rng, n, origin, u, v, normal):
    s, t = rng.random((2, n))
    points = origin + s[:, None] * u + t[:, None] * v
    return points, np.tile(normal, (n, 1))


def _box_room(rng, x1=4.0, y1=3.0, ceiling=2.6):
    parts = [
        _plane(rng, 3000, np.array([0, 0, 0.0]), np.array([x1, 0, 0]), np.array([0, y1, 0]), [0, 0, 1.0]),
        _plane(rng, 3000, np.array([0, 0, ceiling]), np.array([x1, 0, 0]), np.array([0, y1, 0]), [0, 0, -1.0]),
        _plane(rng, 2000, np.array([0, 0, 0.0]), np.array([x1, 0, 0]), np.array([0, 0, ceiling]), [0, 1, 0.0]),
        _plane(rng, 2000, np.array([0, y1, 0.0]), np.array([x1, 0, 0]), np.array([0, 0, ceiling]), [0, -1, 0.0]),
        _plane(rng, 2000, np.array([0, 0, 0.0]), np.array([0, y1, 0]), np.array([0, 0, ceiling]), [1, 0, 0.0]),
        _plane(rng, 2000, np.array([x1, 0, 0.0]), np.array([0, y1, 0]), np.array([0, 0, ceiling]), [-1, 0, 0.0]),
    ]
    return parts


def _cloud(parts, noise_rng=None, noise=0.0):
    points = np.vstack([p for p, _ in parts])
    normals = np.vstack([n for _, n in parts])
    if noise_rng is not None:
        points = points + noise_rng.normal(0, noise, points.shape)
    return Cloud(points, normals, np.zeros(len(points), np.int32))


def test_rectangle_layout_ignores_furniture_fronts_and_doorway_glimpses():
    rng = np.random.default_rng(0)
    parts = _box_room(rng)
    # A bookshelf front 0.35 m off the west wall: real support but short.
    parts.append(_plane(rng, 1500, np.array([0.35, 1.0, 0.0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.9]), [1, 0, 0.0]))
    # Through a door in the east wall: the next room's far wall and higher ceiling.
    parts.append(_plane(rng, 1500, np.array([7.0, 0.5, 0.0]), np.array([0, 2.0, 0]), np.array([0, 0, 2.6]), [-1, 0, 0.0]))
    parts.append(_plane(rng, 1500, np.array([4.15, 0.5, 2.75]), np.array([2.8, 0, 0]), np.array([0, 2.0, 0]), [0, 0, -1.0]))
    layout = rectangle_layout("r", _cloud(parts, rng, 0.004), np.array([[2.0, 1.5]]))
    xs, ys = layout.polygon[:, 0], layout.polygon[:, 1]
    assert abs((xs.max() - xs.min()) - 4.0) < 0.02
    assert abs((ys.max() - ys.min()) - 3.0) < 0.02
    assert abs(layout.ceiling_height - 2.6) < 0.01


def test_gravity_from_walls_when_little_floor_is_seen():
    rng = np.random.default_rng(2)
    tilt = Rotation.from_rotvec([0.06, -0.04, 0.0]).as_matrix()
    parts = _box_room(rng)
    # Mostly wall: drop most of the floor and ceiling.
    parts[0] = (parts[0][0][:40], parts[0][1][:40])
    parts[1] = (parts[1][0][:40], parts[1][1][:40])
    normals = np.vstack([n for _, n in parts]) @ tilt.T
    # Cameras held roughly upright but biased: the prior alone would be 6 degrees off.
    camera = np.eye(4)
    camera[:3, :3] = Rotation.from_rotvec([0.1, 0, 0]).as_matrix() @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]])
    up = gravity_rotation(normals, [tilt_pose(tilt, camera)])[2]
    true_up = tilt @ np.array([0, 0, 1.0])
    assert np.degrees(np.arccos(np.clip(up @ true_up, -1, 1))) < 0.5


def tilt_pose(tilt, camera):
    pose = camera.copy()
    pose[:3, :3] = tilt @ camera[:3, :3]
    return pose


def test_relative_scales_recovers_and_flags_distorted_view():
    # Eight views round a turn, each overlapping its two neighbours on either side.
    true_log = np.array([0.0, 0.08, -0.05, 0.03, -0.1, 0.06, 0.02, -0.04])
    count = len(true_log)
    pairs = []
    for i in range(count):
        for step in (-2, -1, 1, 2):
            j = (i + step) % count
            # measured_j / predicted_from_i = s_j / s_i for a pure scale error ...
            ratio = true_log[j] - true_log[i]
            # ... but view 5's depth is distorted: every pair touching it reads 0.4 high.
            if 5 in (i, j):
                ratio += 0.4
            pairs.append((i, j, ratio, 1500))
    scales, bad = _relative_scales(pairs, count)
    assert bad == {5}
    good = [k for k in range(count) if k != 5]
    corrected = np.log(scales[good]) + true_log[good]
    assert np.ptp(corrected) < 0.01


def _rect_layout(name, corners):
    corners = np.asarray(corners, float)
    lines = []
    for k in range(len(corners)):
        p0, p1 = corners[k], corners[(k + 1) % len(corners)]
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        n = np.array([-d[1], d[0]])
        lines.append(WallLine(n, float(n @ p0), 0.01, 500, 0.01))
    cloud = Cloud(np.array([[0.0, 0.0, -50.0]]), np.array([[0, 0, 1.0]]), np.zeros(1, np.int32))
    return RoomLayout(name, corners, lines, 0.0, 2.6, 0.005, 0.005, cloud, [])


def _rotate(corners, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(corners) @ np.array([[c, -s], [s, c]]).T


def test_stitch_uses_known_heading_to_pick_the_right_door():
    a = _rect_layout("a", [[0, 0], [4, 0], [4, 3], [0, 3]])
    door_a = Opening("a", 1, "door", 1.0, 1.9, 0.0, 2.05, 0.01, 0.01, 0.01, 0.01, 1.0, 4)
    # Room b truly sits east of a, in its own frame turned by +90 degrees.
    b_global = [[4.15, 0], [7.15, 0], [7.15, 3], [4.15, 3]]
    b = _rect_layout("b", _rotate(b_global, np.pi / 2))
    door_b = Opening("b", 3, "door", 1.1, 2.0, 0.0, 2.05, 0.01, 0.01, 0.01, 0.01, 1.0, 4)
    decoy_b = Opening("b", 2, "door", 1.0, 1.9, 0.0, 2.05, 0.01, 0.01, 0.01, 0.01, 1.0, 4)
    rooms = {"a": (a, [door_a], a.points), "b": (b, [decoy_b, door_b], b.points)}
    result = stitch(rooms, headings={"a": 0.0, "b": -np.pi / 2})
    assert len(result.links) == 1
    assert result.links[0].b.index == 1
    placed = result.transforms["b"].apply(b.polygon)
    assert abs(placed[:, 0].min() - 4.15) < 0.05
