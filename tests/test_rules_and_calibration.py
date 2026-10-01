import numpy as np

from roomscope.benchmark.calibrate import multiplier
from roomscope.damage.detect import Detection
from roomscope.damage.rules import OpeningExtent, SurfaceDamage, evaluate_rules


def _det(cls, polygon, area, length=None):
    polygon = np.asarray(polygon, float)
    u0, v0 = polygon.min(axis=0)
    u1, v1 = polygon.max(axis=0)
    return Detection(cls, polygon, area, length, 0.9, (u0, u1, v0, v1))


def _rules(surfaces, openings=(), neighbours=None, room="bedroom"):
    return {f.rule_id for f in evaluate_rules(room, 0.0, list(surfaces), list(openings), neighbours or {})}


def test_stain_below_window_and_large_stain():
    window = OpeningExtent("w0", "window", 2, 1.0, 2.2, 1.0, 2.2)
    stain = _det("water_stain", [[1.2, 0.4], [1.9, 0.4], [1.9, 0.9], [1.2, 0.9]], 0.35)
    fired = _rules([SurfaceDamage("wall:2", "wall", 2, [stain], ["d0"])], [window])
    assert {"R1_STAIN_BELOW_WINDOW", "R5_LARGE_STAIN"} <= fired


def test_stain_on_other_wall_does_not_blame_window():
    window = OpeningExtent("w0", "window", 2, 1.0, 2.2, 1.0, 2.2)
    stain = _det("water_stain", [[1.2, 0.4], [1.5, 0.4], [1.5, 0.6], [1.2, 0.6]], 0.06)
    assert _rules([SurfaceDamage("wall:0", "wall", 0, [stain], ["d0"])], [window]) == set()


def test_diagonal_crack_from_door_corner():
    door = OpeningExtent("o1", "door", 1, 1.0, 1.9, 0.0, 2.05)
    crack = _det("crack", [[1.92, 2.08], [2.05, 2.2], [2.2, 2.35], [2.3, 2.45]], 0.002, 0.5)
    assert "R4_CRACK_FROM_OPENING_CORNER" in _rules([SurfaceDamage("wall:1", "wall", 1, [crack], ["c0"])], [door])


def test_horizontal_crack_is_not_structural_flag():
    door = OpeningExtent("o1", "door", 1, 1.0, 1.9, 0.0, 2.05)
    crack = _det("crack", [[1.92, 2.08], [2.4, 2.1], [2.9, 2.11]], 0.002, 1.0)
    assert "R4_CRACK_FROM_OPENING_CORNER" not in _rules([SurfaceDamage("wall:1", "wall", 1, [crack], ["c0"])], [door])


def test_ceiling_mould_and_shared_wet_wall():
    mould = _det("mold", [[0, 0], [0.4, 0], [0.4, 0.4], [0, 0.4]], 0.16)
    assert {"R3_CEILING_MOISTURE", "R6_MOLD_GROWTH"} <= _rules([SurfaceDamage("ceiling", "ceiling", None, [mould], ["m0"])])
    stain = _det("water_stain", [[0.5, 1.0], [0.8, 1.0], [0.8, 1.3], [0.5, 1.3]], 0.09)
    fired = _rules([SurfaceDamage("wall:3", "wall", 3, [stain], ["s0"])], neighbours={3: ["bathroom"]})
    assert "R7_SHARED_WET_WALL" in fired


def test_rising_damp_rule():
    stain = _det("water_stain", [[0.5, 0.0], [1.0, 0.0], [1.0, 0.3], [0.5, 0.3]], 0.15)
    assert "R2_BASE_OF_WALL_MOISTURE" in _rules([SurfaceDamage("wall:0", "wall", 0, [stain], ["s0"])])


def test_conformal_multiplier_widens_overconfident_intervals():
    rng = np.random.default_rng(0)
    # Errors twice the stated sigma: the 90% multiplier should come out near 2.
    scores = [(float(e), 0.01) for e in rng.normal(0, 0.02, 400)]
    k = multiplier(scores, 0.9)
    assert 1.7 < k < 2.3


def test_conformal_multiplier_never_narrows():
    scores = [(0.001, 0.01)] * 50
    assert multiplier(scores, 0.9) == 1.0
