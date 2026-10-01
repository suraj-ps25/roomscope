"""Progress lines on stderr, so a live run shows what it is doing."""

from __future__ import annotations

import os
import sys
import time

_START = time.perf_counter()
QUIET = os.environ.get("ROOMSCOPE_QUIET") == "1"


def log(stage: str, message: str) -> None:
    if not QUIET:
        print(f"[{time.perf_counter() - _START:7.1f}s] {stage:<11} {message}", file=sys.stderr, flush=True)
