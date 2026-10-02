import numpy as np

from roomscope.benchmark.evaluate import _at_staged_place
from roomscope.damage.ortho import _view_gains
from roomscope.geometry.rooms import WallSegment, _drops_from_ceiling


def test_view_gains_undo_per_view_exposure():
    rng = np.random.default_rng(0)
    surface = rng.uniform(60, 200, (400, 3))
    exposure = np.array([[1.0, 1.0, 1.0], [1.25, 1.1, 0.9], [0.8, 0.85, 1.05]])
    samples = [(v, np.arange(400), surface * exposure[v]) for v in range(3)]
    gains = _view_gains(samples, 3)
    corrected = [surface * exposure[v] * gains[v] for v in range(3)]
    assert np.allclose(corrected[1], corrected[0], rtol=1e-3) and np.allclose(corrected[2], corrected[0], rtol=1e-3)


def test_ceiling_step_is_not_a_barrier_but_a_door_header_is():
    rng = np.random.default_rng(1)
    along = rng.uniform(0, 2.0, 2000)
    segment = WallSegment(0.0, 0.0, 0.0, 2.0)   # x = 0, along y
    points = np.c_[np.zeros(2000), along]
    normals = np.tile([1.0, 0.0], (2000, 1))
    step = rng.uniform(0.02, 0.10, 2000)        # a 10 cm face just under the ceiling
    header = rng.uniform(0.02, 0.45, 2000)      # a header down to door height
    assert not _drops_from_ceiling(segment, points, normals, step)
    assert _drops_from_ceiling(segment, points, normals, header)


def test_damage_counts_only_at_the_staged_place():
    staged = {"surface": "wall:1", "extent_uv": [2.8, 2.05, 3.2, 2.55]}
    region_there = {"polygon_uv": [[2.9, 2.2], [3.1, 2.4], [3.0, 2.3]]}
    region_elsewhere = {"polygon_uv": [[1.39, 1.1], [1.41, 1.6], [1.40, 1.4]]}
    # The plan's walls are the truth's shifted by one.
    assert _at_staged_place(region_there, {"id": "room_3/w2"}, staged, 1, 4)
    assert not _at_staged_place(region_elsewhere, {"id": "room_3/w2"}, staged, 1, 4)
    assert not _at_staged_place(region_there, {"id": "room_3/w3"}, staged, 1, 4)


def test_doorway_gap_between_collinear_traces_is_bridged():
    from roomscope.geometry.rooms import _doorway_bridges
    # One wall x = 0, traced from y = 0 to 2 and from y = 2.9 to 5: a 0.9 m doorway between.
    left, right = WallSegment(0.0, 0.0, 0.0, 2.0), WallSegment(0.0, 0.0, 2.9, 5.0)
    bridges = _doorway_bridges([left, right])
    assert len(bridges) == 1 and abs(bridges[0].start - 2.0) < 1e-9 and abs(bridges[0].end - 2.9) < 1e-9
    # A 2 m gap is open space, not a doorway; a parallel wall elsewhere is not on the line.
    assert not _doorway_bridges([left, WallSegment(0.0, 0.0, 4.0, 5.0)])
    assert not _doorway_bridges([left, WallSegment(0.0, 1.5, 2.9, 5.0)])


def test_ceiling_is_the_layer_over_most_of_the_room():
    from roomscope.geometry.layout import _ceiling_level
    rng = np.random.default_rng(0)
    # A 4 x 3 m ceiling at 3.05 m, lowered to 2.45 m over a 0.8 m strip (denser, fewer cells),
    # plus the bottoms of wall cabinets at 2.2 m.
    main = np.c_[rng.uniform(0, 4, 6000), rng.uniform(0, 3, 6000), 3.05 + rng.normal(0, 0.003, 6000)]
    lowered = np.c_[rng.uniform(0, 4, 7000), rng.uniform(0, 0.8, 7000), 2.45 + rng.normal(0, 0.003, 7000)]
    cabinets = np.c_[rng.uniform(0, 1, 9000), rng.uniform(0, 0.3, 9000), 2.2 + rng.normal(0, 0.003, 9000)]
    level, _ = _ceiling_level(np.vstack([main, lowered, cabinets]))
    assert abs(level - 3.05) < 0.01


def test_sliver_wall_between_collinear_faces_is_absorbed_but_a_real_step_stays():
    from roomscope.geometry.layout import WallLine, _absorb_slivers, _corners

    def line(normal, offset, points=1000):
        n = np.array(normal, float); n /= np.linalg.norm(n)
        return WallLine(n, offset, 0.001, points, 0.01)

    # A 4 x 3 room whose bottom wall has a 2 cm step (y = 0 then y = 0.02) joined by a sliver.
    stepped = [line([0, 1], 0.0), line([-1, 0], -2.0), line([0, 1], 0.02), line([-1, 0], -4.0),
               line([0, -1], -3.0), line([1, 0], 0.0)]
    absorbed, notes = _absorb_slivers(stepped, 12.0)
    assert len(absorbed) == 4 and notes
    # A 20 cm step (a pier) is a real jog and stays.
    pier = [line([0, 1], 0.0), line([-1, 0], -2.0), line([0, 1], 0.2), line([-1, 0], -2.2),
            line([0, -1], -3.0), line([1, 0], 0.0)]
    kept, _ = _absorb_slivers(pier, 12.0)
    assert len(kept) == len(pier) and _corners(kept) is not None
