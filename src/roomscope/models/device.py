from __future__ import annotations

import os


def torch_device() -> str:
    forced = os.environ.get("ROOMSCOPE_DEVICE")
    if forced:
        return forced
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
