"""Ray-traced RGB and iPhone-LiDAR-like depth for a SceneMesh.

Camera convention everywhere in roomscope: OpenCV (x right, y down, z forward), poses
are camera-to-world 4x4, world is z-up with the floor at z=0. Depth maps are z-depth in
metres (what ARKit's sceneDepth stores), 0 where invalid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from .scene import DAMAGE_CLASSES, SceneMesh, SceneSpec

LIDAR_MAX_RANGE = 5.0


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def scaled(self, width: int, height: int) -> Camera:
        sx, sy = width / self.width, height / self.height
        return Camera(width, height, self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy)

    @classmethod
    def iphone_main(cls, width: int, height: int) -> Camera:
        # Main (wide) camera: ~67 deg horizontal FOV at 4:3, i.e. fx ~= 0.75 * width.
        fx = 0.75 * width
        return cls(width, height, fx, fx, width / 2 - 0.5, height / 2 - 0.5)


@dataclass
class HitBuffer:
    range_m: np.ndarray
    zdepth: np.ndarray
    points: np.ndarray
    surface: np.ndarray
    normals: np.ndarray
    special: np.ndarray


class Renderer:
    def __init__(self, scene: SceneSpec, mesh: SceneMesh):
        self.scene = scene
        self.mesh = mesh
        self._ray_scene = o3d.t.geometry.RaycastingScene()
        self._ray_scene.add_triangles(o3d.core.Tensor(mesh.vertices.astype(np.float32)),
                                      o3d.core.Tensor(mesh.triangles.astype(np.uint32)))
        self._decals_by_surface: dict[int, list] = {}
        for decal in scene.decals:
            for index, surface in enumerate(mesh.surfaces):
                if surface.room == decal.room and surface.key == decal.surface:
                    self._decals_by_surface.setdefault(index, []).append(decal)
        self._glass = np.array([s.material == "glass" for s in mesh.surfaces])
        self._colors = np.array([s.color for s in mesh.surfaces], dtype=np.float32)

    def _cast(self, origins: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, ...]:
        rays = np.concatenate([origins, directions], axis=1).astype(np.float32)
        result = self._ray_scene.cast_rays(o3d.core.Tensor(rays))
        t_hit = result["t_hit"].numpy().astype(np.float64)
        prim = result["primitive_ids"].numpy().astype(np.int64)
        normals = result["primitive_normals"].numpy().astype(np.float64)
        hit = np.isfinite(t_hit)
        surface = np.full(len(t_hit), -1, dtype=np.int64)
        surface[hit] = self.mesh.triangle_surface[prim[hit]]
        return t_hit, surface, normals

    def trace(self, camera: Camera, pose: np.ndarray) -> HitBuffer:
        """First opaque hit per pixel; glass is seen through, mirrors are followed once."""
        u, v = np.meshgrid(np.arange(camera.width), np.arange(camera.height))
        rays_cam = np.stack([(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy,
                             np.ones_like(u, dtype=float)], axis=-1).reshape(-1, 3)
        norms = np.linalg.norm(rays_cam, axis=1)
        rot, origin = pose[:3, :3], pose[:3, 3]
        directions = (rays_cam / norms[:, None]) @ rot.T
        origins = np.broadcast_to(origin, directions.shape).copy()

        n = len(directions)
        total_range = np.zeros(n)
        special = np.zeros(n, dtype=np.int8)  # 0 plain, 1 seen through glass, 2 via mirror
        surface = np.full(n, -1)
        points = np.zeros((n, 3))
        normals = np.zeros((n, 3))
        active = np.arange(n)
        for bounce in range(3):
            if len(active) == 0:
                break
            t_hit, surf, nrm = self._cast(origins[active], directions[active])
            hit = np.isfinite(t_hit)
            done_miss = active[~hit]
            total_range[done_miss] = np.inf
            active, t_hit, surf, nrm = active[hit], t_hit[hit], surf[hit], nrm[hit]
            hit_points = origins[active] + directions[active] * t_hit[:, None]
            total_range[active] += t_hit

            through_glass = self._glass[surf]
            mirror = np.zeros(len(active), dtype=bool)
            if bounce < 2:
                mirror = self._inside_kind(surf, hit_points, "mirror") & ~through_glass
            final = ~through_glass & ~mirror
            idx = active[final]
            surface[idx], points[idx], normals[idx] = surf[final], hit_points[final], nrm[final]

            continuing = active[through_glass | mirror]
            special[active[through_glass]] = np.maximum(special[active[through_glass]], 1)
            special[active[mirror]] = 2
            new_dirs = directions[active].copy()
            d = new_dirs[mirror]
            nm = nrm[mirror]
            new_dirs[mirror] = d - 2 * np.sum(d * nm, axis=1, keepdims=True) * nm
            directions[active] = new_dirs
            origins[active] = hit_points + new_dirs * 1e-4
            active = continuing

        cos_to_axis = (rays_cam / norms[:, None])[:, 2]
        zdepth = total_range * cos_to_axis
        shape = (camera.height, camera.width)
        return HitBuffer(total_range.reshape(shape), zdepth.reshape(shape),
                         points.reshape(*shape, 3), surface.reshape(shape),
                         normals.reshape(*shape, 3), special.reshape(shape))

    def _inside_kind(self, surf: np.ndarray, points: np.ndarray, kind: str) -> np.ndarray:
        inside = np.zeros(len(surf), dtype=bool)
        for surface_index, decals in self._decals_by_surface.items():
            rows = np.nonzero(surf == surface_index)[0]
            if len(rows) == 0:
                continue
            uv = self.mesh.surfaces[surface_index].uv(points[rows])
            for decal in decals:
                if decal.kind == kind and decal.polygon_uv is not None:
                    inside[rows] |= _points_in_polygon(uv, decal.polygon_uv)
        return inside

    def shade(self, hits: HitBuffer, exposure: float = 1.0, rng: np.random.Generator | None = None,
              noise_sigma: float = 0.01) -> np.ndarray:
        """Procedural albedo x simple lighting. Texture is a function of surface coords, so it
        is consistent across views (feature matchers can latch on to it)."""
        rng = rng or np.random.default_rng(0)
        h, w = hits.surface.shape
        surf = hits.surface.reshape(-1)
        pts = hits.points.reshape(-1, 3)
        nrm = hits.normals.reshape(-1, 3)
        rgb = np.zeros((len(surf), 3), dtype=np.float32)
        valid = surf >= 0
        rgb[~valid] = (0.92, 0.95, 1.0)  # outside through a window/door: bright daylight

        base = self._colors[np.clip(surf, 0, None)]
        texture = np.ones(len(surf), dtype=np.float32)
        for surface_index in np.unique(surf[valid]):
            rows = np.nonzero(surf == surface_index)[0]
            surface = self.mesh.surfaces[surface_index]
            uv = surface.uv(pts[rows])
            texture[rows] = _material_texture(surface.material, uv, seed=surface_index)
            for decal in self._decals_by_surface.get(surface_index, []):
                base[rows] = _apply_decal(decal, uv, base[rows])

        light = np.array([0.3, -0.5, 0.8])
        light /= np.linalg.norm(light)
        lambert = 0.62 + 0.38 * np.abs(nrm @ light)
        rgb[valid] = (base[valid] * texture[valid, None] * lambert[valid, None])
        glass = hits.special.reshape(-1) == 1
        rgb[glass] = 0.75 * rgb[glass] + 0.25 * np.array([0.55, 0.65, 0.70])
        rgb = rgb * exposure + rng.normal(0, noise_sigma, rgb.shape)
        return np.clip(rgb.reshape(h, w, 3), 0, 1)


def lidar_depth(hits: HitBuffer, rng: np.random.Generator, scale_bias: float = 0.0,
                dark: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Degrade exact z-depth the way ARKit sceneDepth does.

    - The sensor is a sparse SPAD array densified by a network, so depth edges are soft:
      discontinuities are blurred into 'flying pixels'.
    - Range noise grows roughly quadratically with distance; nothing beyond ~5 m.
    - Glass mostly returns the scene behind it (or nothing); mirrors return the reflected
      path length, i.e. a phantom surface behind the mirror plane.
    Returns (depth_m, confidence) with confidence in {0, 1, 2} like ARConfidenceLevel.
    """
    import cv2

    depth = hits.zdepth.copy()
    invalid = ~np.isfinite(depth) | (depth <= 0)
    depth[invalid] = 0.0

    glass = hits.special == 1
    glass_drop = glass & (rng.random(depth.shape) < 0.6)
    depth[glass_drop] = 0.0

    filled = depth.copy()
    filled[filled == 0] = LIDAR_MAX_RANGE + 1
    blurred = cv2.GaussianBlur(filled.astype(np.float32), (5, 5), 1.1).astype(np.float64)
    grad = np.hypot(cv2.Sobel(filled, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(filled, cv2.CV_64F, 0, 1, ksize=3))
    edge = grad > 0.25 * filled
    edge_zone = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    depth = np.where(edge_zone & (depth > 0), blurred, depth)

    sigma = 0.002 + 0.0025 * depth ** 2
    if dark is not None:
        sigma = sigma * np.where(dark, 2.5, 1.0)
    low_freq = cv2.resize(rng.normal(0, 1, (6, 8)).astype(np.float32), depth.shape[::-1],
                          interpolation=cv2.INTER_CUBIC).astype(np.float64)
    depth = depth * (1 + scale_bias) * (1 + 0.0015 * low_freq) + rng.normal(0, 1, depth.shape) * sigma
    depth[(depth > LIDAR_MAX_RANGE) | (depth < 0.1) | invalid | glass_drop] = 0.0

    # Mirrors are deliberately left at medium confidence: ARKit often reports the phantom
    # surface behind a mirror confidently, so the pipeline must reject it geometrically.
    confidence = np.full(depth.shape, 2, dtype=np.uint8)
    confidence[(depth > 3.5) | edge_zone | glass | (hits.special == 2)] = 1
    confidence[depth > 4.5] = 0
    confidence[depth == 0] = 0
    return depth.astype(np.float32), confidence


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    px, py = polygon[:, 0], polygon[:, 1]
    j = len(polygon) - 1
    for i in range(len(polygon)):
        crosses = ((py[i] > y) != (py[j] > y)) & \
                  (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i] + 1e-12) + px[i])
        inside ^= crosses
        j = i
    return inside


def _value_noise(uv: np.ndarray, scale: float, seed: int) -> np.ndarray:
    """Deterministic 2-D value noise in [-1, 1], smooth at `scale` metres."""
    g = uv / scale
    i0 = np.floor(g).astype(np.int64)
    f = g - i0
    f = f * f * (3 - 2 * f)

    def lattice(ix, iy):
        h = (ix * 374761393 + iy * 668265263 + seed * 2147483647) & 0xFFFFFFFF
        h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
        return (h / 0xFFFFFFFF) * 2 - 1

    a = lattice(i0[:, 0], i0[:, 1])
    b = lattice(i0[:, 0] + 1, i0[:, 1])
    c = lattice(i0[:, 0], i0[:, 1] + 1)
    d = lattice(i0[:, 0] + 1, i0[:, 1] + 1)
    return (a * (1 - f[:, 0]) + b * f[:, 0]) * (1 - f[:, 1]) + (c * (1 - f[:, 0]) + d * f[:, 0]) * f[:, 1]


def _material_texture(material: str, uv: np.ndarray, seed: int) -> np.ndarray:
    if material == "wood":
        grain = 0.08 * np.sin(uv[:, 0] * 40 + 3 * _value_noise(uv, 0.3, seed))
        planks = 0.06 * _value_noise(np.stack([np.floor(uv[:, 1] / 0.18), uv[:, 0] * 0.2], 1), 1.0, seed + 7)
        return (1 + grain + planks + 0.05 * _value_noise(uv, 0.02, seed + 1)).astype(np.float32)
    if material == "fabric":
        return (1 + 0.10 * _value_noise(uv, 0.05, seed) + 0.05 * _value_noise(uv, 0.01, seed + 3)).astype(np.float32)
    # Painted plaster: low-contrast mottling at several scales.
    return (1 + 0.035 * _value_noise(uv, 0.4, seed) + 0.025 * _value_noise(uv, 0.08, seed + 1)
            + 0.02 * _value_noise(uv, 0.015, seed + 2)).astype(np.float32)


def _apply_decal(decal, uv: np.ndarray, base: np.ndarray) -> np.ndarray:
    out = base.copy()
    if decal.kind == "crack":
        dist = _distance_to_polyline(uv, decal.polyline_uv)
        core = dist < decal.crack_width / 2
        halo = (dist < decal.crack_width * 1.5) & ~core
        out[core] = out[core] * 0.25
        out[halo] = out[halo] * 0.7
        return out
    inside = _points_in_polygon(uv, decal.polygon_uv)
    if not inside.any():
        return out
    if decal.kind == "water_stain":
        edge = _distance_to_polygon_edge(uv[inside], decal.polygon_uv)
        ring = np.clip(1 - edge / 0.03, 0, 1)
        stain = np.array([0.80, 0.66, 0.42])
        wobble = 0.5 + 0.5 * _value_noise(uv[inside], 0.05, 11)
        mix = np.clip(0.45 + 0.25 * wobble + 0.3 * ring, 0, 1)[:, None]
        out[inside] = out[inside] * (1 - mix) + out[inside] * stain * mix
    elif decal.kind == "mold":
        speck = _value_noise(uv[inside], 0.012, 5) + 0.6 * _value_noise(uv[inside], 0.05, 6)
        dark = speck > 0.15
        rows = np.nonzero(inside)[0][dark]
        out[rows] = np.array([0.18, 0.20, 0.14])
    elif decal.kind == "peeling_paint":
        flakes = _value_noise(uv[inside], 0.03, 9) > 0.1
        rows = np.nonzero(inside)[0][flakes]
        out[rows] = out[rows] * 0.8 + 0.2
    elif decal.kind == "poster":
        pattern = np.stack([_value_noise(uv[inside], s, 20 + k) for k, s in enumerate((0.05, 0.11, 0.023))], 1)
        out[inside] = np.clip(0.5 + 0.45 * pattern, 0, 1)
    elif decal.kind == "mirror":
        out[inside] = np.array([0.8, 0.82, 0.85])
    return out


def _distance_to_polyline(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    best = np.full(len(points), np.inf)
    for a, b in zip(polyline[:-1], polyline[1:]):
        ab = b - a
        t = np.clip(((points - a) @ ab) / (ab @ ab), 0, 1)
        best = np.minimum(best, np.linalg.norm(points - (a + t[:, None] * ab), axis=1))
    return best


def _distance_to_polygon_edge(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    closed = np.vstack([polygon, polygon[:1]])
    return _distance_to_polyline(points, closed)


def damage_mask(renderer: Renderer, hits: HitBuffer) -> np.ndarray:
    """Per-pixel damage class index (0 = none, i+1 = DAMAGE_CLASSES[i]) for ground truth."""
    h, w = hits.surface.shape
    surf = hits.surface.reshape(-1)
    pts = hits.points.reshape(-1, 3)
    mask = np.zeros(len(surf), dtype=np.uint8)
    for surface_index, decals in renderer._decals_by_surface.items():
        rows = np.nonzero(surf == surface_index)[0]
        if len(rows) == 0:
            continue
        uv = renderer.mesh.surfaces[surface_index].uv(pts[rows])
        for decal in decals:
            if decal.kind not in DAMAGE_CLASSES:
                continue
            if decal.kind == "crack":
                hit = _distance_to_polyline(uv, decal.polyline_uv) < decal.crack_width
            else:
                hit = _points_in_polygon(uv, decal.polygon_uv)
            mask[rows[hit]] = DAMAGE_CLASSES.index(decal.kind) + 1
    return mask.reshape(h, w)
