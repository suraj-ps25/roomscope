"""Fetch real iPad LiDAR recordings and the laser scans of the same rooms (ARKitScenes).

  python benchmark/real/fetch_arkitscenes.py 471428 471425      # visit ids

For each visit: every recording of it (depth, confidence, ARKit trajectory, intrinsics and
the 640x480 colour stream) and its Faro laser scans, which are the ground truth. Apple's
ARKitScenes, research licence; nothing is redistributed.
"""

from __future__ import annotations

import csv
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw"
ROOT = Path(__file__).resolve().parents[2] / "data" / "public" / "arkitscenes"
ZIPPED = ["lowres_depth", "confidence", "lowres_wide_intrinsics", "vga_wide", "vga_wide_intrinsics"]


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=600) as response:
        return response.read()


def _table(url: str, cache: Path) -> list[dict]:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(_get(url))
    return list(csv.DictReader(io.StringIO(cache.read_text())))


def fetch_visit(visit: str) -> None:
    metadata = _table(f"{BASE}/metadata.csv", ROOT / "metadata.csv")
    scans = _table(f"{BASE}/laser_scanner_point_clouds/laser_scanner_point_clouds_mapping.csv", ROOT / "laser_mapping.csv")
    for row in (r for r in metadata if r["visit_id"] == visit):
        video, fold = row["video_id"], row["fold"]
        folder = ROOT / video
        folder.mkdir(parents=True, exist_ok=True)
        for asset in ZIPPED:
            if not (folder / asset).exists():
                print(f"{video}: {asset}", flush=True)
                zipfile.ZipFile(io.BytesIO(_get(f"{BASE}/{fold}/{video}/{asset}.zip"))).extractall(folder)
        if not (folder / "lowres_wide.traj").exists():
            (folder / "lowres_wide.traj").write_bytes(_get(f"{BASE}/{fold}/{video}/lowres_wide.traj"))
        (folder / "visit.txt").write_text(visit)
    laser = ROOT / "laser" / visit
    laser.mkdir(parents=True, exist_ok=True)
    for scan in (r["laser_scanner_point_clouds_id"] for r in scans if r["visit_id"] == visit):
        for suffix in (".ply", "_pose.txt"):
            target = laser / f"{scan}{suffix}"
            if not target.exists():
                print(f"visit {visit}: laser scan {scan}{suffix}", flush=True)
                urllib.request.urlretrieve(f"{BASE}/laser_scanner_point_clouds/{visit}/{scan}{suffix}", target)


if __name__ == "__main__":
    for visit_id in sys.argv[1:]:
        fetch_visit(visit_id)
