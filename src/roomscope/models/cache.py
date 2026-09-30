"""Deterministic on-disk cache for model outputs.

Keyed by a hash of the exact inputs (pixels, intrinsics, model id, settings). A cache hit
replays the stored arrays bit-for-bit; a miss runs the live model and stores the result.
ROOMSCOPE_CACHE=off forces the live path.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

CACHE_DIR = Path(os.environ.get("ROOMSCOPE_CACHE_DIR", Path.home() / ".cache" / "roomscope"))


def enabled() -> bool:
    return os.environ.get("ROOMSCOPE_CACHE", "on").lower() not in ("0", "off", "false")


def key(*parts) -> str:
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            digest.update(str(part.dtype).encode())
            digest.update(str(part.shape).encode())
            digest.update(np.ascontiguousarray(part).tobytes())
        else:
            digest.update(repr(part).encode())
    return digest.hexdigest()


def load(namespace: str, name: str) -> dict[str, np.ndarray] | None:
    path = CACHE_DIR / namespace / f"{name}.npz"
    if not enabled() or not path.exists():
        return None
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def store(namespace: str, name: str, arrays: dict[str, np.ndarray]) -> None:
    if not enabled():
        return
    folder = CACHE_DIR / namespace
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"{name}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    tmp.rename(folder / f"{name}.npz")
