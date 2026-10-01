#!/usr/bin/env bash
# The full benchmark, from one commit, end to end. Every number in docs/benchmark_report.md
# comes from the outputs this writes under runs/bench/ (override with OUT=...).
#
#   benchmark/run_all.sh            # ~2-3 h on an M2 (LiDAR sweep dominates)
#
# Resumable: a stage whose outputs already exist under $OUT is skipped (delete $OUT, or the
# stage's folder, to rerun it).
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
[[ -f $OUT/lidar/summary_lidar.json ]] || $PY benchmark/run_sim.py --tier lidar --seeds 0 1 2 --drift 1.0 --ablation --out "$OUT/lidar"

echo "== LiDAR at full RGB resolution (1920x1440, as Stray Scanner records it), for damage"
[[ -f data/captures/sim_flat_a_lidar/ground_truth.json ]] || $RS sim benchmark/sim/flat_a.yaml --tier lidar --out data/captures/sim_flat_a_lidar
if [[ ! -f $OUT/flat_a_lidar_fullres/metrics.json ]]; then
  $RS run data/captures/sim_flat_a_lidar --out "$OUT/flat_a_lidar_fullres"
  $RS eval "$OUT/flat_a_lidar_fullres/plan.json" data/captures/sim_flat_a_lidar/ground_truth.json --out "$OUT/flat_a_lidar_fullres/metrics.json"
fi

echo "== photo and video on the synthetic flat: depth model and rendered true depth"
for tier in photo video; do
  capture=data/captures/sim_flat_a_$tier
  [[ -f $capture/ground_truth.json ]] || $RS sim benchmark/sim/flat_a.yaml --tier $tier --out $capture
  for depth in model oracle; do
    [[ -f $OUT/flat_a_${tier}_$depth/metrics.json ]] && continue
    flags=""; [[ $depth == oracle ]] && flags="--oracle-depth"
    $RS run $capture --out "$OUT/flat_a_${tier}_$depth" $flags
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
[[ -f $OUT/depth_scale.json ]] || $PY benchmark/depth_model_scale.py --out "$OUT/depth_scale.json"

echo "== real iPad LiDAR recordings against laser truth (ARKitScenes; dev and held-out visits)"
# Fetch: python benchmark/real/fetch_arkitscenes.py <visit ids in benchmark/ground_truth/arkitscenes_*.yaml>
# Photo and video captures from the same recordings' colour stream (skipped if already made).
for marker in data/public/arkitscenes/*/visit.txt; do
  video=$(basename "$(dirname "$marker")")
  [[ -d data/public/real_captures/${video}_photo || -d data/public/real_captures/${video}_video ]] || $PY benchmark/real/make_image_captures.py "$video"
done
$PY benchmark/real/run_real.py --out "$OUT/real"

echo "== interval calibration (split conformal, leave-one-property-out coverage)"
lidar_runs=(); for s in 0 1 2; do lidar_runs+=("$OUT/lidar/flat_a_lidar_d1_s${s}_corrected/plan.json:data/captures/simbench/flat_a_lidar_d1_s$s/ground_truth.json"); done
# Real recordings calibrate only from the dev visits; the held-out visits never do.
while read -r visit; do
  for video in $(grep -l "^$visit$" data/public/arkitscenes/*/visit.txt | xargs -n1 dirname | xargs -n1 basename); do
    [[ -f $OUT/real/$video/metrics.json ]] && lidar_runs+=("$OUT/real/$video/plan.json:benchmark/ground_truth/arkitscenes_$visit.yaml")
  done
done < benchmark/real/dev_visits.txt
$RS calibrate lidar "${lidar_runs[@]}" --out "$OUT/calibration_lidar.json" > /dev/null
for tier in photo video; do
  runs=("$OUT/flat_a_${tier}_model/plan.json:data/captures/sim_flat_a_$tier/ground_truth.json")
  for room in room0 room1 room2 office0 office1 office2 office3 office4; do
    runs+=("$OUT/replica/${room}_${tier}_model/plan.json:data/captures/replica/${room}_$tier/ground_truth.json")
  done
  $RS calibrate $tier "${runs[@]}" --out "$OUT/calibration_$tier.json" > /dev/null
done

echo "== report"
$PY benchmark/report.py --bench "$OUT" --out docs/benchmark_report.md
