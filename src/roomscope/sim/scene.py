"""Synthetic property scenes with exact ground truth.

A scene is a set of rooms (CCW footprint polygons of the interior wall faces, floor at
z=0), openings cut through walls, furniture boxes, and surface decals (damage, posters,
mirrors). build_mesh() turns it into triangles tagged with the surface they belong to,
so a renderer can shade them and the ground truth can be read straight off the spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

DAMAGE_CLASSES = ("water_stain", "crack", "mold", "peeling_paint")


@dataclass
class RoomSpec:
    id: str
    polygon: np.ndarray
    ceiling: float
    wall_color: tuple[float, float, float] = (0.86, 0.84, 0.80)
    floor_color: tuple[float, float, float] = (0.55, 0.42, 0.30)
    ceiling_color: tuple[float, float, float] = (0.93, 0.93, 0.92)

    def wall_frame(self, wall: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Start corner, unit direction, length, inward normal of polygon edge `wall`."""
        start = self.polygon[wall]
        end = self.polygon[(wall + 1) % len(self.polygon)]
        span = end - start
        length = float(np.linalg.norm(span))
        direction = span / length
        inward = np.array([-direction[1], direction[0]])
        return start, direction, length, inward


@dataclass
class OpeningSpec:
    id: str
    room: str
    wall: int
    offset: float
    width: float
    height: float
    sill: float = 0.0
    type: str = "door"
    connects_to: str | None = None
    glass: bool = False


@dataclass
class BoxSpec:
    room: str
    center: np.ndarray
    size: np.ndarray
    yaw: float = 0.0
    color: tuple[float, float, float] = (0.35, 0.30, 0.28)


@dataclass
class DecalSpec:
    """Something painted onto (or mounted flush with) a surface, in surface coords (u, v)."""
    id: str
    room: str
    surface: str
    kind: str
    polygon_uv: np.ndarray | None = None
    polyline_uv: np.ndarray | None = None
    crack_width: float = 0.004


@dataclass
class SceneSpec:
    name: str
    wall_thickness: float
    rooms: list[RoomSpec]
    openings: list[OpeningSpec]
    furniture: list[BoxSpec] = field(default_factory=list)
    decals: list[DecalSpec] = field(default_factory=list)

    def room(self, room_id: str) -> RoomSpec:
        return next(r for r in self.rooms if r.id == room_id)


def load_scene(path: str | Path) -> SceneSpec:
    raw = yaml.safe_load(Path(path).read_text())
    rooms = [RoomSpec(r["id"], _ccw(np.asarray(r["polygon"], dtype=float)), float(r["ceiling"]),
                      **{k: tuple(r[k]) for k in ("wall_color", "floor_color", "ceiling_color") if k in r})
             for r in raw["rooms"]]
    openings = [OpeningSpec(**o) for o in raw.get("openings", [])]
    furniture = [BoxSpec(b["room"], np.asarray(b["center"], float), np.asarray(b["size"], float),
                         float(b.get("yaw", 0.0)), tuple(b.get("color", (0.35, 0.30, 0.28))))
                 for b in raw.get("furniture", [])]
    decals = [DecalSpec(d["id"], d["room"], d["surface"], d["kind"],
                        None if "polygon_uv" not in d else np.asarray(d["polygon_uv"], float),
                        None if "polyline_uv" not in d else np.asarray(d["polyline_uv"], float),
                        float(d.get("crack_width", 0.004)))
              for d in raw.get("decals", [])]
    scene = SceneSpec(raw["name"], float(raw.get("wall_thickness", 0.12)), rooms, openings,
                      furniture, decals)
    scene.openings += _derive_counterparts(scene)
    return scene


def _ccw(polygon: np.ndarray) -> np.ndarray:
    return polygon if polygon_area(polygon) > 0 else polygon[::-1].copy()


def polygon_area(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _derive_counterparts(scene: SceneSpec) -> list[OpeningSpec]:
    """A door declared on room A's wall also cuts room B's facing wall; find that opening."""
    derived = []
    for opening in [o for o in scene.openings if o.connects_to]:
        room_a = scene.room(opening.room)
        start_a, dir_a, _, inward_a = room_a.wall_frame(opening.wall)
        jamb_near = start_a + dir_a * opening.offset
        jamb_far = start_a + dir_a * (opening.offset + opening.width)
        room_b = scene.room(opening.connects_to)
        match = None
        for wall_b in range(len(room_b.polygon)):
            start_b, dir_b, length_b, inward_b = room_b.wall_frame(wall_b)
            if np.dot(dir_a, dir_b) > -0.999:
                continue
            gap = float(np.dot(start_b - start_a, -inward_a))
            if abs(gap - scene.wall_thickness) > 0.02:
                continue
            u_near, u_far = (float(np.dot(j - start_b, dir_b)) for j in (jamb_near, jamb_far))
            lo, hi = min(u_near, u_far), max(u_near, u_far)
            if lo >= -1e-6 and hi <= length_b + 1e-6:
                match = (wall_b, lo)
                break
        if match is None:
            raise ValueError(f"opening {opening.id}: no facing wall in {opening.connects_to}")
        derived.append(OpeningSpec(f"{opening.id}~", opening.connects_to, match[0], match[1],
                                   opening.width, opening.height, opening.sill, opening.type,
                                   opening.room, opening.glass))
    return derived


def triangulate(polygon: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear clipping for a simple CCW polygon (rooms can be L-shaped)."""
    remaining = list(range(len(polygon)))
    triangles = []
    while len(remaining) > 3:
        for k in range(len(remaining)):
            i0, i1, i2 = remaining[k - 1], remaining[k], remaining[(k + 1) % len(remaining)]
            a, b, c = polygon[i0], polygon[i1], polygon[i2]
            if _cross(b - a, c - b) <= 1e-12:
                continue
            others = [polygon[j] for j in remaining if j not in (i0, i1, i2)]
            if any(_in_triangle(p, a, b, c) for p in others):
                continue
            triangles.append((i0, i1, i2))
            remaining.pop(k)
            break
        else:
            raise ValueError("polygon is not simple; cannot triangulate")
    triangles.append(tuple(remaining))
    return triangles


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _in_triangle(p, a, b, c) -> bool:
    d1, d2, d3 = _cross(b - a, p - a), _cross(c - b, p - b), _cross(a - c, p - c)
    return (d1 >= 0 and d2 >= 0 and d3 >= 0) or (d1 <= 0 and d2 <= 0 and d3 <= 0)


@dataclass
class Surface:
    """A planar patch of the mesh. uv maps world points to surface coordinates."""
    room: str
    key: str
    color: tuple[float, float, float]
    origin: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    material: str = "paint"

    def uv(self, points: np.ndarray) -> np.ndarray:
        rel = points - self.origin
        return np.stack([rel @ self.u_axis, rel @ self.v_axis], axis=-1)


@dataclass
class SceneMesh:
    vertices: np.ndarray
    triangles: np.ndarray
    triangle_surface: np.ndarray
    surfaces: list[Surface]


class _MeshBuilder:
    def __init__(self):
        self.vertices: list[np.ndarray] = []
        self.triangles: list[tuple[int, int, int]] = []
        self.triangle_surface: list[int] = []
        self.surfaces: list[Surface] = []

    def surface(self, surface: Surface) -> int:
        self.surfaces.append(surface)
        return len(self.surfaces) - 1

    def quad(self, surface_index: int, corners: list[np.ndarray]) -> None:
        base = len(self.vertices)
        self.vertices.extend(corners)
        self.triangles += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
        self.triangle_surface += [surface_index, surface_index]

    def polygon(self, surface_index: int, points: list[np.ndarray],
                triangles: list[tuple[int, int, int]]) -> None:
        base = len(self.vertices)
        self.vertices.extend(points)
        self.triangles += [(base + a, base + b, base + c) for a, b, c in triangles]
        self.triangle_surface += [surface_index] * len(triangles)

    def build(self) -> SceneMesh:
        return SceneMesh(np.asarray(self.vertices, dtype=np.float64),
                         np.asarray(self.triangles, dtype=np.int32),
                         np.asarray(self.triangle_surface, dtype=np.int32),
                         self.surfaces)


def _xyz(xy: np.ndarray, z: float) -> np.ndarray:
    return np.array([xy[0], xy[1], z], dtype=float)


def build_mesh(scene: SceneSpec) -> SceneMesh:
    builder = _MeshBuilder()
    up = np.array([0.0, 0.0, 1.0])
    for room in scene.rooms:
        _floor_and_ceiling(builder, room)
        for wall in range(len(room.polygon)):
            _wall(builder, scene, room, wall, up)
    for index, box in enumerate(scene.furniture):
        _box(builder, box, index)
    return builder.build()


def _floor_and_ceiling(builder: _MeshBuilder, room: RoomSpec) -> None:
    tris = triangulate(room.polygon)
    floor = builder.surface(Surface(room.id, "floor", room.floor_color, np.zeros(3),
                                    np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), "wood"))
    builder.polygon(floor, [_xyz(p, 0.0) for p in room.polygon], tris)
    ceiling = builder.surface(Surface(room.id, "ceiling", room.ceiling_color,
                                      np.array([0, 0, room.ceiling]),
                                      np.array([1.0, 0, 0]), np.array([0, 1.0, 0])))
    builder.polygon(ceiling, [_xyz(p, room.ceiling) for p in room.polygon],
                    [(a, c, b) for a, b, c in tris])


def _wall(builder: _MeshBuilder, scene: SceneSpec, room: RoomSpec, wall: int, up: np.ndarray) -> None:
    start, direction, length, inward = room.wall_frame(wall)
    d3 = np.array([direction[0], direction[1], 0.0])
    n3 = np.array([inward[0], inward[1], 0.0])
    origin = _xyz(start, 0.0)
    face = builder.surface(Surface(room.id, f"wall:{wall}", room.wall_color, origin, d3, up))
    height = room.ceiling
    openings = sorted((o for o in scene.openings if o.room == room.id and o.wall == wall),
                      key=lambda o: o.offset)

    def point(u: float, v: float, depth: float = 0.0) -> np.ndarray:
        return origin + d3 * u + up * v - n3 * depth

    breaks = sorted({0.0, length, *[o.offset for o in openings],
                     *[o.offset + o.width for o in openings]})
    for u0, u1 in zip(breaks[:-1], breaks[1:]):
        if u1 - u0 < 1e-9:
            continue
        covering = next((o for o in openings if o.offset <= u0 + 1e-9 and u1 <= o.offset + o.width + 1e-9), None)
        spans = [(0.0, height)] if covering is None else \
            [(0.0, covering.sill), (covering.sill + covering.height, height)]
        for v0, v1 in spans:
            if v1 - v0 > 1e-9:
                builder.quad(face, [point(u0, v0), point(u1, v0), point(u1, v1), point(u0, v1)])

    for opening in openings:
        _reveal(builder, scene, room, opening, point, origin, d3, up, n3)


def _reveal(builder, scene, room, opening, point, origin, d3, up, n3) -> None:
    # Interior doors: each side builds half the wall thickness so the two halves meet.
    depth = scene.wall_thickness / 2 if opening.connects_to else scene.wall_thickness
    u0, u1 = opening.offset, opening.offset + opening.width
    v0, v1 = opening.sill, opening.sill + opening.height
    reveal = builder.surface(Surface(room.id, f"reveal:{opening.id}", room.wall_color, origin, d3, up))
    builder.quad(reveal, [point(u0, v0), point(u0, v0, depth), point(u0, v1, depth), point(u0, v1)])
    builder.quad(reveal, [point(u1, v0), point(u1, v1), point(u1, v1, depth), point(u1, v0, depth)])
    builder.quad(reveal, [point(u0, v1), point(u0, v1, depth), point(u1, v1, depth), point(u1, v1)])
    sill_color = room.floor_color if opening.sill == 0 else room.wall_color
    sill = builder.surface(Surface(room.id, f"sill:{opening.id}", sill_color, origin, d3, -n3,
                                   "wood" if opening.sill == 0 else "paint"))
    builder.quad(sill, [point(u0, v0), point(u1, v0), point(u1, v0, depth), point(u0, v0, depth)])
    if opening.glass:
        pane = builder.surface(Surface(room.id, f"glass:{opening.id}", (0.6, 0.7, 0.75),
                                       origin - n3 * depth * 0.5, d3, up, "glass"))
        mid = depth * 0.5
        builder.quad(pane, [point(u0, v0, mid), point(u1, v0, mid), point(u1, v1, mid), point(u0, v1, mid)])


def _box(builder: _MeshBuilder, box: BoxSpec, index: int) -> None:
    c, s = np.cos(box.yaw), np.sin(box.yaw)
    rot = np.array([[c, -s], [s, c]])
    half = box.size / 2
    corners2 = [box.center + rot @ np.array([sx * half[0], sy * half[1]])
                for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    bottom = [_xyz(p, 0.0) for p in corners2]
    top = [_xyz(p, box.size[2]) for p in corners2]
    surface = builder.surface(Surface(box.room, f"furniture:{index}", box.color,
                                      bottom[0], np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), "fabric"))
    builder.quad(surface, top)
    for i in range(4):
        j = (i + 1) % 4
        builder.quad(surface, [bottom[i], bottom[j], top[j], top[i]])


def ground_truth(scene: SceneSpec) -> dict:
    """Ground truth in the same shape a tape-measure sheet takes (see benchmark/README)."""
    rooms = []
    for room in scene.rooms:
        walls = []
        for wall in range(len(room.polygon)):
            _, _, length, _ = room.wall_frame(wall)
            walls.append({"index": wall, "length": round(length, 4)})
        openings = [{"id": o.id, "type": o.type, "wall": o.wall, "offset": round(o.offset, 4),
                     "width": o.width, "height": o.height,
                     "sill": o.sill if o.type == "window" else None, "connects_to": o.connects_to}
                    for o in scene.openings if o.room == room.id]
        damage = []
        for decal in scene.decals:
            if decal.room != room.id or decal.kind not in DAMAGE_CLASSES:
                continue
            entry = {"id": decal.id, "class": decal.kind, "surface": decal.surface}
            if decal.polyline_uv is not None:
                segs = np.diff(decal.polyline_uv, axis=0)
                crack_length = float(np.linalg.norm(segs, axis=1).sum())
                entry.update(length=round(crack_length, 4), area=round(crack_length * decal.crack_width, 6))
            else:
                entry.update(area=round(abs(polygon_area(decal.polygon_uv)), 4))
            damage.append(entry)
        rooms.append({
            "id": room.id,
            "polygon": np.round(room.polygon, 4).tolist(),
            "ceiling_height": room.ceiling,
            "floor_area": round(polygon_area(room.polygon), 4),
            "walls": walls,
            "openings": openings,
            "damage": damage,
        })
    adjacency = sorted({tuple(sorted((o.room, o.connects_to))) for o in scene.openings if o.connects_to})
    return {
        "property": scene.name,
        "source": "synthetic",
        "rooms": rooms,
        "adjacency": [list(pair) for pair in adjacency],
        "footprint_area": round(sum(r["floor_area"] for r in rooms), 4),
    }
