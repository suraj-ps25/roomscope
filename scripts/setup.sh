#!/usr/bin/env bash
# Fresh-machine setup. The LiDAR tier needs no model weights and is ready after step 2.
# Photo and video tiers need MoGe-2 + MapAnything (~7 GB of weights), fetched in step 3.
#
#   scripts/setup.sh            # everything
#   scripts/setup.sh --lidar    # LiDAR tier only (fast)
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
MAPANYTHING_COMMIT=3d10cf7
MOGE_COMMIT=74fbce0

echo "== 1. system tools"
command -v ffmpeg >/dev/null || {
  if command -v brew >/dev/null; then brew install ffmpeg; else echo "install ffmpeg (apt install ffmpeg)"; exit 1; fi; }
command -v uv >/dev/null || {
  if command -v brew >/dev/null; then brew install uv; else curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; fi; }

echo "== 2. roomscope (LiDAR tier ready after this)"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/roomscope --help >/dev/null && echo "   roomscope installed"

if [[ "${1:-}" == "--lidar" ]]; then
  echo "done (LiDAR tier). Run: .venv/bin/roomscope run <stray-folder> --out runs/<name>"
  exit 0
fi

echo "== 3. photo/video models (pinned commits; weights from Hugging Face Hub)"
uv pip install --python .venv/bin/python torch torchvision
mkdir -p third_party
if [[ ! -d third_party/map-anything ]]; then
  git clone -q https://github.com/facebookresearch/map-anything.git third_party/map-anything
fi
git -C third_party/map-anything checkout -q "$MAPANYTHING_COMMIT"
uv pip install --python .venv/bin/python -e third_party/map-anything
if [[ ! -d third_party/MoGe ]]; then
  git clone -q https://github.com/microsoft/MoGe.git third_party/MoGe
fi
git -C third_party/MoGe checkout -q "$MOGE_COMMIT"
# MoGe's newest release pulls a Linux-only GPU kernel (triton) that MoGe-2 inference
# never touches; install it without dependencies and add the one it does need.
uv pip install --python .venv/bin/python --no-deps -e third_party/MoGe
uv pip install --python .venv/bin/python "utils3d @ git+https://github.com/EasternJournalist/utils3d.git"

echo "   downloading weights (MoGe-2 ~1.3 GB, MapAnything ~4.9 GB + DINOv2)"
.venv/bin/python - <<'EOF'
from roomscope.models import depth, multiview
depth._load(); print("   MoGe-2 ready")
multiview._load(); print("   MapAnything ready")
EOF

echo "== 4. smoke test"
.venv/bin/python -m pytest -q

echo "done. Run: .venv/bin/roomscope run <capture> --out runs/<name>"
