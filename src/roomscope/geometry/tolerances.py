"""Geometry tolerances per tier. LiDAR surfaces are millimetre-thin; surfaces rebuilt from
photos or video carry a few percent of depth error (~10 cm at 3 m), so the windows that
decide 'this point is on that wall' scale with the tier's noise."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tolerances:
    wall_search_m: float = 0.30
    wall_inlier_m: float = 0.03
    opening_face_m: float = 0.04
    opening_recess_m: float = 0.35
    jamb_window_m: float = 0.08
    min_votes: int = 3
    min_room_frames: int = 8


LIDAR_TOL = Tolerances()
VIDEO_TOL = Tolerances(wall_search_m=0.45, wall_inlier_m=0.07, opening_face_m=0.09, opening_recess_m=0.40,
                       jamb_window_m=0.12, min_votes=2, min_room_frames=4)
PHOTO_TOL = Tolerances(wall_search_m=0.60, wall_inlier_m=0.10, opening_face_m=0.12, opening_recess_m=0.45,
                       jamb_window_m=0.15, min_votes=1, min_room_frames=1)
