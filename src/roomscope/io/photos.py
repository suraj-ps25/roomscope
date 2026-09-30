"""Photo-tier ingest: one sub-folder per room, 2-8 stills each, any phone camera.

Handles HEIC (the iPhone default), EXIF orientation, and derives a focal-length prior
from EXIF FocalLengthIn35mmFilm when present (35 mm equivalent is defined on the frame
diagonal: fx = f35 * diagonal_px / 43.27).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .detect import image_files

PATCH = 14
LONG_SIDE = 518
FULL_FRAME_DIAGONAL_MM = 43.27
EXIF_IFD, EXIF_FOCAL35, EXIF_MODEL = 0x8769, 0xA405, 0x0110

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # HEIC unsupported; JPEG/PNG still work
    pass


@dataclass
class Photo:
    path: Path
    room: str
    image: np.ndarray
    K: np.ndarray | None
    fov_x_deg: float | None
    original: np.ndarray
    device: str | None


def processing_size(width: int, height: int, long_side: int = LONG_SIDE) -> tuple[int, int]:
    scale = long_side / max(width, height)
    w = max(PATCH, int(round(width * scale / PATCH)) * PATCH)
    h = max(PATCH, int(round(height * scale / PATCH)) * PATCH)
    return w, h


def _focal_35(image: Image.Image) -> float | None:
    try:
        value = image.getexif().get_ifd(EXIF_IFD).get(EXIF_FOCAL35)
    except Exception:
        return None
    return float(value) if value else None


def load_photo(path: Path, room: str) -> Photo:
    with Image.open(path) as raw:
        exif_model = raw.getexif().get(EXIF_MODEL)
        focal35 = _focal_35(raw)
        upright = ImageOps.exif_transpose(raw).convert("RGB")
    original = np.array(upright)
    height, width = original.shape[:2]
    w, h = processing_size(width, height)
    small = np.array(upright.resize((w, h), Image.Resampling.LANCZOS))
    K = fov = None
    if focal35:
        fx_full = focal35 * np.hypot(width, height) / FULL_FRAME_DIAGONAL_MM
        K = np.array([[fx_full * w / width, 0, (w - 1) / 2], [0, fx_full * h / height, (h - 1) / 2], [0, 0, 1.0]])
        fov = float(np.degrees(2 * np.arctan(w / (2 * K[0, 0]))))
    return Photo(path, room, small, K, fov, original, str(exif_model) if exif_model else None)


def read_photo_folders(root: str | Path) -> dict[str, list[Photo]]:
    root = Path(root)
    rooms = {}
    for folder in sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")):
        files = image_files(folder)
        if files:
            rooms[folder.name] = [load_photo(f, folder.name) for f in files]
    if not rooms:
        raise ValueError(f"{root}: no room folders with photos")
    return rooms
