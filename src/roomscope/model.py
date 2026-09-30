"""Output data model. Mirrors schema/floorplan.schema.json one-to-one."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import NormalDist

DEFAULT_CI_LEVEL = 0.9


def z_for_level(level: float) -> float:
    return NormalDist().inv_cdf(0.5 + level / 2.0)


@dataclass
class Measurement:
    value: float
    unit: str
    ci_low: float
    ci_high: float
    ci_level: float
    method: str

    @classmethod
    def from_sigma(cls, value: float, sigma: float, unit: str, method: str,
                   level: float = DEFAULT_CI_LEVEL) -> Measurement:
        half_width = z_for_level(level) * max(sigma, 0.0)
        return cls(value, unit, value - half_width, value + half_width, level, method)

    @classmethod
    def from_bounds(cls, value: float, low: float, high: float, unit: str, method: str,
                    level: float = DEFAULT_CI_LEVEL) -> Measurement:
        return cls(value, unit, min(low, value), max(high, value), level, method)

    @property
    def half_width(self) -> float:
        return (self.ci_high - self.ci_low) / 2.0

    def contains(self, truth: float) -> bool:
        return self.ci_low <= truth <= self.ci_high


@dataclass
class Wall:
    id: str
    start: list[float]
    end: list[float]
    length: Measurement


@dataclass
class Opening:
    id: str
    type: str
    wall_id: str
    offset: Measurement
    width: Measurement
    height: Measurement
    sill_height: Measurement | None = None
    connects_to: str | None = None
    detection_confidence: float = 1.0


@dataclass
class DamageRegion:
    id: str
    cls: str
    area: Measurement
    polygon_uv: list[list[float]]
    length: Measurement | None = None
    confidence: float = 1.0


@dataclass
class Surface:
    id: str
    kind: str
    area: Measurement
    wall_id: str | None = None
    damage_regions: list[DamageRegion] = field(default_factory=list)


@dataclass
class ConcealedFlag:
    id: str
    rule_id: str
    rule: str
    surface_id: str
    evidence: list[str]
    suspected: str
    severity: str = "medium"


@dataclass
class Room:
    id: str
    polygon: list[list[float]]
    ceiling_height: Measurement
    floor_area: Measurement
    walls: list[Wall]
    openings: list[Opening] = field(default_factory=list)
    surfaces: list[Surface] = field(default_factory=list)
    concealed_flags: list[ConcealedFlag] = field(default_factory=list)
    perimeter: Measurement | None = None
    label: str | None = None
    coverage: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class Adjacency:
    from_room: str
    to_room: str
    via: list[str]
    confidence: float = 1.0


@dataclass
class ScopeItem:
    id: str
    surface_id: str
    code: str
    description: str
    quantity: Measurement
    reason: list[str]
    count: int | None = None


@dataclass
class DriftReport:
    method: str
    enabled: bool
    residual_before_m: float | None = None
    residual_after_m: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class Plan:
    capture_id: str
    tier: str
    rooms: list[Room]
    adjacency: list[Adjacency]
    footprint_area: Measurement
    drift: DriftReport
    calibration_method: str
    ci_level: float = DEFAULT_CI_LEVEL
    calibration_table: str | None = None
    extent_x: Measurement | None = None
    extent_y: Measurement | None = None
    scope: list[ScopeItem] = field(default_factory=list)
    source_app: str | None = None
    device: str | None = None
    frames_used: int | None = None
    capture_notes: list[str] = field(default_factory=list)
    timing_s: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _plan_to_dict(self)


def _measurement(m: Measurement | None) -> dict | None:
    return None if m is None else asdict(m)


def _damage(d: DamageRegion) -> dict:
    return {
        "id": d.id,
        "class": d.cls,
        "area": _measurement(d.area),
        "length": _measurement(d.length),
        "polygon_uv": d.polygon_uv,
        "confidence": d.confidence,
    }


def _surface(s: Surface) -> dict:
    return {
        "id": s.id,
        "kind": s.kind,
        "wall_id": s.wall_id,
        "area": _measurement(s.area),
        "damage_regions": [_damage(d) for d in s.damage_regions],
    }


def _opening(o: Opening) -> dict:
    return {
        "id": o.id,
        "type": o.type,
        "wall_id": o.wall_id,
        "offset": _measurement(o.offset),
        "width": _measurement(o.width),
        "height": _measurement(o.height),
        "sill_height": _measurement(o.sill_height),
        "connects_to": o.connects_to,
        "detection_confidence": o.detection_confidence,
    }


def _room(r: Room) -> dict:
    room = {
        "id": r.id,
        "label": r.label,
        "polygon": r.polygon,
        "ceiling_height": _measurement(r.ceiling_height),
        "floor_area": _measurement(r.floor_area),
        "walls": [{"id": w.id, "start": w.start, "end": w.end, "length": _measurement(w.length)}
                  for w in r.walls],
        "openings": [_opening(o) for o in r.openings],
        "surfaces": [_surface(s) for s in r.surfaces],
        "concealed_flags": [asdict(f) for f in r.concealed_flags],
    }
    if r.perimeter is not None:
        room["perimeter"] = _measurement(r.perimeter)
    quality = {}
    if r.coverage is not None:
        quality["coverage"] = r.coverage
    if r.notes:
        quality["notes"] = r.notes
    if quality:
        room["quality"] = quality
    return room


def _plan_to_dict(p: Plan) -> dict:
    capture = {"id": p.capture_id, "tier": p.tier, "source_app": p.source_app, "device": p.device}
    if p.frames_used is not None:
        capture["frames_used"] = p.frames_used
    if p.capture_notes:
        capture["notes"] = p.capture_notes

    prop = {"room_count": len(p.rooms), "footprint_area": _measurement(p.footprint_area)}
    if p.extent_x is not None:
        prop["extent_x"] = _measurement(p.extent_x)
    if p.extent_y is not None:
        prop["extent_y"] = _measurement(p.extent_y)

    plan = {
        "schema_version": "1.0",
        "capture": capture,
        "rooms": [_room(r) for r in p.rooms],
        "adjacency": [{"from": a.from_room, "to": a.to_room, "via": a.via, "confidence": a.confidence}
                      for a in p.adjacency],
        "property": prop,
        "scope": [{
            "id": s.id, "surface_id": s.surface_id, "code": s.code, "description": s.description,
            "quantity": _measurement(s.quantity), "count": s.count, "reason": s.reason,
        } for s in p.scope],
        "drift": asdict(p.drift),
        "calibration": {"ci_level": p.ci_level, "method": p.calibration_method,
                        "table": p.calibration_table},
    }
    if p.timing_s:
        plan["timing_s"] = p.timing_s
    return plan
