"""Repair scope: line items keyed to surfaces, quantities carrying their intervals."""

from __future__ import annotations

from dataclasses import dataclass

from ..model import Measurement

ITEMS = {
    "water_stain": ("PRM-STAIN", "Stain-blocking primer over the stain and a margin", 1.5),
    "mold": ("MLD-REM", "Mould remediation: HEPA clean, antimicrobial, dry out", 2.0),
    "peeling_paint": ("PNT-PREP", "Scrape, fill and prime peeling paint", 1.2),
    "hole": ("PLS-HOLE", "Patch and make good", 1.5),
}


@dataclass
class ScopeLine:
    surface_id: str
    code: str
    description: str
    quantity: Measurement
    reason: list[str]
    count: int | None = None


def _scaled(m: Measurement, factor: float, method: str) -> Measurement:
    return Measurement.from_bounds(m.value * factor, m.ci_low * factor, m.ci_high * factor, m.unit, method, m.ci_level)


def scope_for_surface(surface_id: str, kind: str, surface_area: Measurement, regions: list, flags: list) -> list[ScopeLine]:
    """regions: model.DamageRegion list for this surface; flags: model.ConcealedFlag list."""
    lines: list[ScopeLine] = []
    for region in regions:
        if region.cls == "crack" and region.length is not None:
            lines.append(ScopeLine(surface_id, "PLS-CRACK", "Rake out, fill, tape and sand crack", region.length,
                                   [region.id]))
        elif region.cls in ITEMS:
            code, text, margin = ITEMS[region.cls]
            lines.append(ScopeLine(surface_id, code, text, _scaled(region.area, margin, "damage extent x margin"),
                                   [region.id]))
    if regions:
        code, text = ("PNT-CEIL", "Repaint ceiling") if kind == "ceiling" else \
            (("PNT-FLOOR", "Refinish floor") if kind == "floor" else ("PNT-WALL", "Repaint wall (full surface, net of openings)"))
        lines.append(ScopeLine(surface_id, code, text, surface_area, [r.id for r in regions]))
    for flag in flags:
        structural = flag.rule_id.startswith("R4")
        code, text = ("INS-STRUCT", "Structural inspection of the opening lintel/framing") if structural else \
            ("INS-MOIST", "Moisture meter and thermal survey; open up if readings are high")
        lines.append(ScopeLine(surface_id, code, text, surface_area, [flag.id, *flag.evidence], count=1))
    return lines
