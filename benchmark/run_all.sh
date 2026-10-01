#!/usr/bin/env bash
# The full benchmark, from one commit, end to end. Every number in docs/benchmark_report.md
# comes from the outputs this writes under runs/bench/ (override with OUT=...).
#
#   benchmark/run_all.sh            # ~2-3 h on an M2 (LiDAR sweep dominates)
#
# Captures are generated if missing (synthetic flat: benchmark/sim/flat_a.yaml; scanned
# rooms: data/public/replica/cull_replica_mesh, fetched by scripts/fetch_replica.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -d data/public/replica/cull_replica_mesh ]] || scripts/fetch_replica.sh
PY=.venv/bin/python
RS=.venv/bin/roomscope
OUT=${OUT:-runs/bench}
mkdir -p "$OUT"

echo "== LiDAR: 3 captures of the flat, drift correction on and off"
$PY benchmark/run_sim.py --tier lidar --seeds 0 1 2 --drift 1.0 --ablation --out "$OUT/lidar"

echo "== photo and video on the synthetic flat: depth model and rendered true depth"
for tier in photo video; do
  capture=data/captures/sim_flat_a_$tier
  [[ -f $capture/ground_truth.json ]] || $RS sim benchmark/sim/flat_a.yaml --tier $tier --out $capture
  for depth in model oracle; do
    flags=(); [[ $depth == oracle ]] && flags=(--oracle-depth)
    $RS run $capture --out "$OUT/flat_a_${tier}_$depth" "${flags[@]}"
    $RS eval "$OUT/flat_a_${tier}_$depth/plan.json" $capture/ground_truth.json --out "$OUT/flat_a_${tier}_$depth/metrics.json"
  done
done

echo "== scanned rooms (Replica): photo and video, depth model and true depth"
for room in room0 room1 room2 office0 office1 office2 office3 office4; do
  for tier in photo video; do
    capture=data/captures/replica/${room}_$tier
    [[ -f $capture/ground_truth.json ]] || $RS sim data/public/replica/cull_replica_mesh/$room.ply --tier $tier --out $capture
  done
done
$PY benchmark/run_replica.py --out "$OUT/replica"

echo "== depth model scale per scene"
$PY benchmark/depth_model_scale.py --out "$OUT/depth_scale.json"

echo "== report"
$PY benchmark/report.py --bench "$OUT" --out docs/benchmark_report.md
