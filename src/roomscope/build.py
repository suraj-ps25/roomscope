"""Tier-agnostic plan assembly: geometry + uncertainties -> schema measurements.

Every tier hands over the same RoomGeometry. Uncertainty is propagated, not guessed:

  - a wall's length is set by where its two neighbouring walls sit, so its sigma comes
    from their line-offset sigmas pushed through the corner intersections (numerically);
  - each tier carries a relative scale uncertainty (LiDAR depth bias ~0.3%; photos, where
    metric scale comes from a learned prior, several percent) that enters every length
    proportionally and every area twice;
  - a per-tier, per-quantity calibration multiplier (fitted on ground truth, see
    calibrate.py) scales the propagated sigma so the stated coverage holds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import model
from .model import Measurement

CALIBRATION_DIR = Path(__file__).resolve().parents[2] / "calibration"


@dataclass
class OpeningGeometry:
    wall: int
    kind: str
    u0: float
    u1: float
    v0: float
    v1: float
    sigma_u0: float
    sigma_u1: float
    sigma_v0: float
    sigma_v1: float
    confidence: float = 1.0
    connects_to: str | None = None
    partner_key: tuple[str, int] | None = None


@dataclass
class RoomGeometry:
    id: str
    polygon: np.ndarray
    normals: np.ndarray
    offsets: np.ndarray
    offset_sigmas: np.ndarray
    floor_z: float
    ceiling_z: float
    floor_sigma: float
    ceiling_sigma: float
    openings: list[OpeningGeometry] = field(default_factory=list)
    coverage: float | None = None
    notes: list[str] = field(default_factory=list)
    damage: object = None
    label: str | None = None
    scale_sigma: float | None = None


@dataclass(frozen=True)
class TierProfile:
    tier: str
    scale_sigma: float
    method: str


LIDAR = TierProfile("lidar", 0.003, "lidar depth fusion, plane fit")
VIDEO = TierProfile("video", 0.02, "multi-view reconstruction, metric depth prior")
PHOTO = TierProfile("photo", 0.04, "sparse-view reconstruction, metric depth prior")


def load_calibration(tier: str) -> tuple[dict[str, float], dict[str, float], str | None]:
    """Per quantity: the conformal multiplier, and the absolute floor added in quadrature to
    the propagated sigma before it (a plane fit's millimetres can't describe a wall placed
    in the wrong spot)."""
    path = CALIBRATION_DIR / f"{tier}.json"
    if not path.exists():
        return {}, {}, None
    table = json.loads(path.read_text())
    return table["multipliers"], table.get("floors_m", {}), table.get("id")


def _polygon_from_lines(normals: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    corners = []
    for k in range(len(normals)):
        corners.append(np.linalg.solve(np.array([normals[k - 1], normals[k]]), np.array([offsets[k - 1], offsets[k]])))
    return np.array(corners)


def _area(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _lengths(polygon: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)


def _propagate(room: RoomGeometry, fn) -> float:
    """sigma of fn(polygon) from independent line-offset sigmas (central differences)."""
    variance = 0.0
    for k, sigma in enumerate(room.offset_sigmas):
        if sigma <= 0:
            continue
        step = np.zeros(len(room.offsets))
        step[k] = sigma
        hi = fn(_polygon_from_lines(room.normals, room.offsets + step))
        lo = fn(_polygon_from_lines(room.normals, room.offsets - step))
        variance += ((hi - lo) / 2) ** 2
    return float(np.sqrt(variance))


class _Sigma:
    def __init__(self, profile: TierProfile, multipliers: dict[str, float], floors: dict[str, float] | None = None):
        self.profile = profile
        self.multipliers = multipliers
        self.floors = floors or {}

    def measurement(self, quantity: str, value: float, sigma_local: float, unit: str = "m",
                    scale_power: int = 1) -> Measurement:
        scale = self.profile.scale_sigma * scale_power * abs(value)
        sigma = np.hypot(np.hypot(sigma_local, scale), self.floors.get(quantity, 0.0)) * self.multipliers.get(quantity, 1.0)
        return Measurement.from_sigma(float(value), float(sigma), unit, self.profile.method)


DAMAGE_EDGE_M = 0.0075


def _damage_regions(room: RoomGeometry, key: str, sigma: _Sigma) -> list[model.DamageRegion]:
    if room.damage is None:
        return []
    regions = []
    for det, det_id in zip(room.damage.detections.get(key, []), room.damage.ids.get(key, [])):
        polygon = det.polygon_uv
        perimeter = float(np.sum(np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)))
        area = sigma.measurement("damage_area", det.area_m2, perimeter * DAMAGE_EDGE_M, "m2", scale_power=2)
        length = None if det.length_m is None else sigma.measurement("damage_length", det.length_m, 2 * DAMAGE_EDGE_M)
        regions.append(model.DamageRegion(f"{room.id}/{det_id}", det.cls, area, np.round(polygon, 4).tolist(),
                                          length, round(det.confidence, 3)))
    return regions


def build_room(room: RoomGeometry, sigma: _Sigma) -> tuple[model.Room, list[model.ScopeItem]]:
    from dataclasses import replace

    from .damage.scope import scope_for_surface

    if room.scale_sigma is not None:
        # Photo rooms carry their own scale uncertainty (it depends on how many photos).
        sigma = _Sigma(replace(sigma.profile, scale_sigma=room.scale_sigma), sigma.multipliers, sigma.floors)

    polygon = room.polygon
    lengths = _lengths(polygon)
    walls = []
    for k in range(len(polygon)):
        local = _propagate(room, lambda p, k=k: _lengths(p)[k])
        walls.append(model.Wall(f"{room.id}/w{k}", polygon[k].round(4).tolist(),
                                polygon[(k + 1) % len(polygon)].round(4).tolist(),
                                sigma.measurement("wall_length", lengths[k], local)))
    height = room.ceiling_z - room.floor_z
    ceiling = sigma.measurement("ceiling_height", height, float(np.hypot(room.floor_sigma, room.ceiling_sigma)))
    area = _area(polygon)
    floor_area = sigma.measurement("floor_area", area, _propagate(room, _area), "m2", scale_power=2)
    perimeter = sigma.measurement("perimeter", float(lengths.sum()), _propagate(room, lambda p: _lengths(p).sum()))

    openings, surfaces = [], []
    opening_area = {k: 0.0 for k in range(len(polygon))}
    for index, o in enumerate(room.openings):
        width = sigma.measurement("opening_width", o.u1 - o.u0, float(np.hypot(o.sigma_u0, o.sigma_u1)))
        tall = sigma.measurement("opening_height", o.v1 - o.v0, float(np.hypot(o.sigma_v0, o.sigma_v1)))
        sill = None if o.kind != "window" else sigma.measurement("opening_sill", o.v0 - room.floor_z, o.sigma_v0)
        openings.append(model.Opening(f"{room.id}/o{index}", o.kind, f"{room.id}/w{o.wall}",
                                      sigma.measurement("opening_offset", o.u0, o.sigma_u0), width, tall, sill,
                                      o.connects_to, round(float(o.confidence), 3)))
        opening_area[o.wall] += (o.u1 - o.u0) * (o.v1 - o.v0)

    for k, wall in enumerate(walls):
        gross = wall.length.value * height
        net = max(gross - opening_area[k], 0.0)
        rel = np.hypot(wall.length.half_width / max(wall.length.value, 1e-9), ceiling.half_width / max(height, 1e-9))
        surfaces.append(model.Surface(f"{room.id}/w{k}", "wall", Measurement.from_bounds(
            net, net * (1 - rel), net * (1 + rel), "m2", sigma.profile.method, wall.length.ci_level), wall.id,
            _damage_regions(room, f"w{k}", sigma)))
    surfaces.append(model.Surface(f"{room.id}/floor", "floor", floor_area, None, _damage_regions(room, "floor", sigma)))
    surfaces.append(model.Surface(f"{room.id}/ceiling", "ceiling", floor_area, None,
                                  _damage_regions(room, "ceiling", sigma)))

    flags = []
    for index, flag in enumerate(room.damage.flags if room.damage is not None else []):
        flags.append(model.ConcealedFlag(f"{room.id}/f{index}", flag.rule_id, flag.rule, f"{room.id}/{flag.surface_key}",
                                         [f"{room.id}/{e}" for e in flag.evidence], flag.suspected, flag.severity))
    scope = []
    for surface in surfaces:
        surface_flags = [f for f in flags if f.surface_id == surface.id]
        for line in scope_for_surface(surface.id, surface.kind, surface.area, surface.damage_regions, surface_flags):
            scope.append(model.ScopeItem(f"{room.id}/s{len(scope)}", line.surface_id, line.code, line.description,
                                         line.quantity, line.reason, line.count))

    built = model.Room(room.id, polygon.round(4).tolist(), ceiling, floor_area, walls, openings, surfaces, flags,
                       perimeter=perimeter, label=room.label or room.id, coverage=room.coverage, notes=room.notes)
    return built, scope


def build_plan(capture_id: str, profile: TierProfile, rooms: list[RoomGeometry], drift: model.DriftReport,
               source_app: str | None = None, frames_used: int | None = None,
               capture_notes: list[str] | None = None,
               extra_adjacency: list[tuple[str, str]] | None = None) -> model.Plan:
    multipliers, floors, table_id = load_calibration(profile.tier)
    sigma = _Sigma(profile, multipliers, floors)
    results = [build_room(room, sigma) for room in rooms]
    built = [room for room, _ in results]
    scope = [item for _, items in results for item in items]

    adjacency = []
    seen = set()
    for room, geometry in zip(built, rooms):
        for opening, raw in zip(room.openings, geometry.openings):
            if raw.connects_to and raw.partner_key:
                pair = tuple(sorted([(room.id, opening.id), (raw.connects_to, f"{raw.partner_key[0]}/o{raw.partner_key[1]}")]))
                if pair in seen:
                    continue
                seen.add(pair)
                adjacency.append(model.Adjacency(room.id, raw.connects_to,
                                                 [opening.id, f"{raw.partner_key[0]}/o{raw.partner_key[1]}"],
                                                 round(float(raw.confidence), 3)))

    # Rooms placed through a doorway whose door wasn't measured on both sides (photo
    # threshold pairs): the connection is known, the opening ids are not.
    for a, b in extra_adjacency or []:
        adjacency.append(model.Adjacency(a, b, [], 0.8))

    total = float(sum(r.floor_area.value for r in built))
    total_sigma = float(np.sqrt(sum(r.floor_area.half_width ** 2 for r in built))) / model.z_for_level(model.DEFAULT_CI_LEVEL)
    footprint = Measurement.from_sigma(total, total_sigma, "m2", profile.method)
    corners = np.vstack([np.array(r.polygon) for r in built]) if built else np.zeros((1, 2))
    extent = corners.max(axis=0) - corners.min(axis=0)
    extent_x = sigma.measurement("extent", float(extent[0]), 0.01)
    extent_y = sigma.measurement("extent", float(extent[1]), 0.01)
    method = "split-conformal per tier and quantity" if multipliers else "propagated (uncalibrated)"
    return model.Plan(capture_id, profile.tier, built, adjacency, footprint, drift, method,
                      calibration_table=table_id, extent_x=extent_x, extent_y=extent_y, scope=scope,
                      source_app=source_app, frames_used=frames_used, capture_notes=capture_notes or [])
