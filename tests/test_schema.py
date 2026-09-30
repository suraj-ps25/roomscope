import copy

import jsonschema
import pytest

from roomscope.model import (
    Adjacency, DamageRegion, DriftReport, Measurement, Opening, Plan, Room, Surface, Wall,
)
from roomscope.schema import validate


def m(value, sigma=0.01, unit="m"):
    return Measurement.from_sigma(value, sigma, unit, "test")


def _room(room_id, x0):
    corners = [[x0, 0.0], [x0 + 4.0, 0.0], [x0 + 4.0, 3.0], [x0, 3.0]]
    walls = [Wall(f"{room_id}/w{i}", corners[i], corners[(i + 1) % 4], m(4.0 if i % 2 == 0 else 3.0))
             for i in range(4)]
    stain = DamageRegion(f"{room_id}/d0", "water_stain", m(0.3, 0.02, "m2"),
                         [[0.5, 0.1], [1.0, 0.1], [1.0, 0.6], [0.5, 0.6]])
    return Room(
        id=room_id,
        polygon=corners,
        ceiling_height=m(2.7),
        floor_area=m(12.0, 0.1, "m2"),
        walls=walls,
        openings=[Opening(f"{room_id}/o0", "door", f"{room_id}/w1", m(1.0), m(0.9), m(2.1))],
        surfaces=[Surface(f"{room_id}/w0", "wall", m(10.8, 0.1, "m2"), f"{room_id}/w0", [stain])],
    )


def sample_plan() -> dict:
    plan = Plan(
        capture_id="unit",
        tier="lidar",
        rooms=[_room("room_1", 0.0), _room("room_2", 4.0)],
        adjacency=[Adjacency("room_1", "room_2", ["room_1/o0", "room_2/o0"], 0.9)],
        footprint_area=m(24.0, 0.2, "m2"),
        drift=DriftReport("pose-graph", True, 0.05, 0.01),
        calibration_method="none",
    )
    return plan.to_dict()


def test_sample_plan_validates():
    validate(sample_plan())


def test_interval_must_contain_value():
    plan = sample_plan()
    plan["rooms"][0]["ceiling_height"]["ci_high"] = 2.0
    with pytest.raises(ValueError):
        validate(plan)


def test_measurement_requires_interval():
    plan = sample_plan()
    bad = copy.deepcopy(plan)
    del bad["rooms"][0]["floor_area"]["ci_low"]
    with pytest.raises(jsonschema.ValidationError):
        validate(bad)


def test_unknown_damage_class_rejected():
    plan = sample_plan()
    plan["rooms"][0]["surfaces"][0]["damage_regions"][0]["class"] = "scuff"
    with pytest.raises(jsonschema.ValidationError):
        validate(plan)
