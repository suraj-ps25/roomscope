"""Pick the tier from what the capture folder contains."""

from __future__ import annotations

from pathlib import Path

from .arkitscenes import is_arkitscenes
from .stray import is_stray

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".heic", ".heif", ".png"}


def image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                  and not p.name.startswith("."))


def video_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
                  and not p.name.startswith("."))


def resolve_capture(path: str | Path) -> Path:
    """A Stray Scanner export copied off the phone often arrives inside a wrapper folder
    (an unzipped archive, or a folder named after the room). A folder whose only content is
    one LiDAR recording stands for that recording."""
    root = Path(path)
    if not root.is_dir() or is_stray(root) or is_arkitscenes(root):
        return root
    children = [d for d in root.iterdir() if not d.name.startswith(".")]
    recordings = [d for d in children if d.is_dir() and (is_stray(d) or is_arkitscenes(d))]
    if len(recordings) == 1 and len(children) == 1:
        return recordings[0]
    if len(recordings) > 1:
        names = ", ".join(sorted(d.name for d in recordings))
        raise ValueError(f"{root}: holds {len(recordings)} LiDAR recordings ({names}); run one at a time")
    return root


def detect_tier(path: str | Path) -> str:
    root = Path(path)
    if root.is_file() and root.suffix.lower() in VIDEO_SUFFIXES:
        return "video"
    if not root.is_dir():
        raise ValueError(f"{root}: not a capture folder or video file")
    if is_stray(root) or is_arkitscenes(root):
        return "lidar"
    rooms = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".") and image_files(d)]
    if rooms:
        return "photo"
    if video_files(root):
        return "video"
    if image_files(root):
        raise ValueError(f"{root}: photos must be in one sub-folder per room (see docs/capture_protocol.md)")
    raise ValueError(f"{root}: no Stray Scanner export, room photo folders or video found")
