"""Stitched floor-plan rendering (plan.png), in the style of consumer scanning apps."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc, Polygon  # noqa: E402

WALL_M = 0.10
ROOM_FILLS = ["#f3efe6", "#e8f0ec", "#eceaf3", "#f4ebe9", "#e9eff4", "#f1f0e4", "#eef3e8", "#f3eaf0"]
WALL_COLOUR = "#2b2b2b"
WINDOW_COLOUR = "#4a90c2"
DOOR_COLOUR = "#8a8a8a"
DAMAGE_COLOURS = {"water_stain": "#c77d2e", "crack": "#b3261e", "mold": "#3d6b35", "peeling_paint": "#8e6bb0",
                  "hole": "#5a4636"}


def _fmt(measurement: dict, digits: int = 2) -> str:
    half = (measurement["ci_high"] - measurement["ci_low"]) / 2
    return f"{measurement['value']:.{digits}f} ±{half:.{max(digits, 2)}f}"


def render_plan(plan: dict, path: str | Path, title: str | None = None) -> None:
    rooms = plan["rooms"]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_facecolor("white")
    corners = np.vstack([np.array(r["polygon"]) for r in rooms]) if rooms else np.zeros((1, 2))

    for index, room in enumerate(rooms):
        polygon = np.array(room["polygon"])
        ax.add_patch(Polygon(polygon, closed=True, facecolor=ROOM_FILLS[index % len(ROOM_FILLS)],
                             edgecolor="none", zorder=1))
        _walls(ax, room)
        _openings(ax, room)
        _damage(ax, room)
        _dimensions(ax, room)
        centre = _label_point(polygon)
        area = room["floor_area"]
        ax.text(centre[0], centre[1], f"{room.get('label') or room['id']}\n{area['value']:.1f} m²\n"
                f"h {room['ceiling_height']['value']:.2f} m", ha="center", va="center", fontsize=9,
                color="#333", zorder=6, linespacing=1.4)

    lo, hi = corners.min(axis=0) - 0.9, corners.max(axis=0) + 0.9
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.axis("off")
    _scale_bar(ax, lo, hi)
    footprint = plan["property"]["footprint_area"]
    heading = title or f"{plan['capture']['id']}  ·  {plan['capture']['tier']} tier"
    ax.set_title(f"{heading}\n{len(rooms)} rooms  ·  {_fmt(footprint, 1)} m²  ·  "
                 f"intervals at {int(plan['calibration']['ci_level'] * 100)}%", fontsize=11, color="#222", loc="left")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _wall_segments(room: dict) -> list[tuple[np.ndarray, np.ndarray, list[dict]]]:
    return [(np.array(wall["start"]), np.array(wall["end"]),
             [o for o in room["openings"] if o["wall_id"] == wall["id"]]) for wall in room["walls"]]


def _outward(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    direction = (end - start) / max(np.linalg.norm(end - start), 1e-9)
    return np.array([direction[1], -direction[0]])


def _walls(ax, room: dict) -> None:
    for start, end, openings in _wall_segments(room):
        length = np.linalg.norm(end - start)
        direction = (end - start) / max(length, 1e-9)
        outward = _outward(start, end)
        cuts = sorted((o["offset"]["value"], o["offset"]["value"] + o["width"]["value"]) for o in openings
                      if o["type"] in ("door", "opening"))
        pieces, cursor = [], 0.0
        for a, b in cuts:
            pieces.append((cursor, max(a, cursor)))
            cursor = max(cursor, b)
        pieces.append((cursor, length))
        for a, b in pieces:
            if b - a < 1e-3:
                continue
            p0, p1 = start + direction * a, start + direction * b
            quad = np.array([p0, p1, p1 + outward * WALL_M, p0 + outward * WALL_M])
            ax.add_patch(Polygon(quad, closed=True, facecolor=WALL_COLOUR, edgecolor=WALL_COLOUR, lw=0.5, zorder=3))


def _openings(ax, room: dict) -> None:
    for start, end, openings in _wall_segments(room):
        length = np.linalg.norm(end - start)
        direction = (end - start) / max(length, 1e-9)
        outward = _outward(start, end)
        inward = -outward
        for o in openings:
            a = o["offset"]["value"]
            b = a + o["width"]["value"]
            p0, p1 = start + direction * a, start + direction * b
            if o["type"] == "window":
                for depth in (0.25, 0.75):
                    q0, q1 = p0 + outward * WALL_M * depth, p1 + outward * WALL_M * depth
                    ax.plot([q0[0], q1[0]], [q0[1], q1[1]], color=WINDOW_COLOUR, lw=1.6, zorder=4)
                ax.add_patch(Polygon(np.array([p0, p1, p1 + outward * WALL_M, p0 + outward * WALL_M]),
                                     closed=True, facecolor="white", edgecolor=WINDOW_COLOUR, lw=0.8, zorder=3.5))
            elif o["type"] == "door":
                width = o["width"]["value"]
                angle = np.degrees(np.arctan2(direction[1], direction[0]))
                side = np.degrees(np.arctan2(inward[1], inward[0]))
                leaf_end = p0 + inward * width
                ax.plot([p0[0], leaf_end[0]], [p0[1], leaf_end[1]], color=DOOR_COLOUR, lw=1.2, zorder=4)
                start_angle, end_angle = sorted([angle, side])
                if end_angle - start_angle > 180:
                    start_angle, end_angle = end_angle, start_angle + 360
                ax.add_patch(Arc(p0, 2 * width, 2 * width, theta1=start_angle, theta2=end_angle,
                                 color=DOOR_COLOUR, lw=0.8, ls="--", zorder=4))


def _damage(ax, room: dict) -> None:
    walls = {w["id"]: w for w in room["walls"]}
    for surface in room["surfaces"]:
        for region in surface["damage_regions"]:
            colour = DAMAGE_COLOURS.get(region["class"], "#b3261e")
            uv = np.array(region["polygon_uv"])
            if surface["kind"] == "wall" and surface.get("wall_id") in walls:
                wall = walls[surface["wall_id"]]
                start, end = np.array(wall["start"]), np.array(wall["end"])
                direction = (end - start) / max(np.linalg.norm(end - start), 1e-9)
                inward = -_outward(start, end)
                u0, u1 = uv[:, 0].min(), uv[:, 0].max()
                q0, q1 = start + direction * u0 + inward * 0.06, start + direction * u1 + inward * 0.06
                ax.plot([q0[0], q1[0]], [q0[1], q1[1]], color=colour, lw=4, solid_capstyle="butt", zorder=5)
            else:
                ax.add_patch(Polygon(uv, closed=True, facecolor=colour, alpha=0.35, edgecolor=colour, zorder=2))


def _dimensions(ax, room: dict) -> None:
    for wall in room["walls"]:
        start, end = np.array(wall["start"]), np.array(wall["end"])
        length = np.linalg.norm(end - start)
        if length < 0.4:
            continue
        mid = (start + end) / 2
        inward = -_outward(start, end)
        angle = np.degrees(np.arctan2(end[1] - start[1], end[0] - start[0]))
        if angle > 90 or angle < -90:
            angle += 180
        spot = mid + inward * 0.22
        ax.text(spot[0], spot[1], _fmt(wall["length"]), ha="center", va="center", rotation=angle,
                fontsize=7, color="#555", zorder=6)


def _label_point(polygon: np.ndarray) -> np.ndarray:
    """A point well inside the polygon (pole of inaccessibility on a coarse grid)."""
    lo, hi = polygon.min(axis=0), polygon.max(axis=0)
    xs, ys = np.meshgrid(np.linspace(lo[0], hi[0], 40), np.linspace(lo[1], hi[1], 40))
    candidates = np.stack([xs.ravel(), ys.ravel()], axis=1)
    inside = _inside(candidates, polygon)
    if not inside.any():
        return polygon.mean(axis=0)
    candidates = candidates[inside]
    edges = np.stack([polygon, np.roll(polygon, -1, axis=0)], axis=1)
    distance = np.full(len(candidates), np.inf)
    for a, b in edges:
        ab = b - a
        t = np.clip(((candidates - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        distance = np.minimum(distance, np.linalg.norm(candidates - (a + t[:, None] * ab), axis=1))
    return candidates[np.argmax(distance)]


def _inside(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi, xj, yj = polygon[i, 0], polygon[i, 1], polygon[j, 0], polygon[j, 1]
        crosses = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= crosses
        j = i
    return inside


def _scale_bar(ax, lo: np.ndarray, hi: np.ndarray) -> None:
    x0, y0 = lo[0] + 0.3, lo[1] + 0.3
    ax.plot([x0, x0 + 1.0], [y0, y0], color="#222", lw=2)
    ax.text(x0 + 0.5, y0 + 0.12, "1 m", ha="center", fontsize=8, color="#222")
