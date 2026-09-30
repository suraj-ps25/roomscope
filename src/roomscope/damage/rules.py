"""Concealed-damage rules: visible evidence + room geometry -> what is probably hidden.

Each rule is explicit, names itself when it fires, and cites the detections (and the
opening, where one is involved) that triggered it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MOISTURE = ("water_stain", "mold")
WET_ROOM_WORDS = ("bath", "wc", "toilet", "shower", "kitchen", "laundry", "utility")


@dataclass
class SurfaceDamage:
    surface_key: str
    kind: str
    wall_index: int | None
    detections: list
    ids: list[str]


@dataclass
class OpeningExtent:
    id: str
    kind: str
    wall_index: int
    u0: float
    u1: float
    v0: float
    v1: float


@dataclass
class Flag:
    rule_id: str
    rule: str
    surface_key: str
    evidence: list[str]
    suspected: str
    severity: str


def _crack_angle(polygon_uv: np.ndarray) -> float:
    centred = polygon_uv - polygon_uv.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    direction = vt[0]
    return float(np.degrees(np.arctan2(abs(direction[1]), abs(direction[0]))))


def evaluate_rules(room_label: str, floor_z: float, surfaces: list[SurfaceDamage], openings: list[OpeningExtent],
                   neighbours_on_wall: dict[int, list[str]]) -> list[Flag]:
    flags: list[Flag] = []
    for surface in surfaces:
        for det, det_id in zip(surface.detections, surface.ids):
            u0, u1, v0, v1 = det.bbox_uv
            if surface.kind == "wall":
                wall_openings = [o for o in openings if o.wall_index == surface.wall_index]
                if det.cls in MOISTURE:
                    for o in (o for o in wall_openings if o.kind == "window"):
                        overlap = min(u1, o.u1) - max(u0, o.u0)
                        if overlap > 0.3 * (u1 - u0) and v1 >= o.v0 - 0.6 and v0 < o.v0:
                            flags.append(Flag(
                                "R1_STAIN_BELOW_WINDOW",
                                "Moisture staining directly below a window sill (within 0.6 m, overlapping its width)",
                                surface.surface_key, [det_id, o.id],
                                "failed sill or flashing seal letting water into the wall cavity below the window",
                                "high" if det.area_m2 > 0.1 else "medium"))
                    if v0 <= 0.15:
                        flags.append(Flag(
                            "R2_BASE_OF_WALL_MOISTURE", "Moisture reaching the base of a wall (within 0.15 m of the floor)",
                            surface.surface_key, [det_id],
                            "rising damp or a plumbing leak behind the skirting; subfloor may be wet", "medium"))
                    for neighbour in neighbours_on_wall.get(surface.wall_index, []):
                        if any(word in neighbour.lower() for word in WET_ROOM_WORDS):
                            flags.append(Flag(
                                "R7_SHARED_WET_WALL", f"Moisture on a wall shared with a wet room ({neighbour})",
                                surface.surface_key, [det_id],
                                "concealed plumbing leak inside the shared wall", "high"))
                if det.cls == "crack":
                    angle = _crack_angle(det.polygon_uv)
                    for o in wall_openings:
                        corners = [(o.u0, o.v1), (o.u1, o.v1), (o.u0, o.v0), (o.u1, o.v0)]
                        ends = det.polygon_uv
                        near = min(np.hypot(ends[:, 0] - cu, ends[:, 1] - cv).min() for cu, cv in corners)
                        if near < 0.15 and 20 <= angle <= 70:
                            flags.append(Flag(
                                "R4_CRACK_FROM_OPENING_CORNER",
                                f"Diagonal crack ({angle:.0f} deg) starting within 0.15 m of a {o.kind} corner",
                                surface.surface_key, [det_id, o.id],
                                "structural movement or an overstressed lintel above the opening; check the framing",
                                "high"))
                            break
            if surface.kind == "ceiling" and det.cls in MOISTURE:
                flags.append(Flag(
                    "R3_CEILING_MOISTURE", "Moisture staining or mould on a ceiling", surface.surface_key, [det_id],
                    "leak from above (roof, plumbing or a wet room overhead); the ceiling cavity is likely wet",
                    "high" if det.area_m2 > 0.2 else "medium"))
            if det.cls == "water_stain" and det.area_m2 >= 0.25:
                flags.append(Flag(
                    "R5_LARGE_STAIN", f"Visible stain of {det.area_m2:.2f} m2 (>= 0.25 m2)", surface.surface_key, [det_id],
                    "moisture has spread further inside the wall or ceiling than the visible stain", "medium"))
            if det.cls == "mold":
                flags.append(Flag(
                    "R6_MOLD_GROWTH", "Visible mould growth", surface.surface_key, [det_id],
                    "sustained moisture; concealed growth behind the lining is likely", "high"))
    return flags
