#!/usr/bin/env bash
# Walk-in rehearsal: every tier, cold, from a fresh clone (after scripts/setup.sh).
# Generates one synthetic capture per tier, runs it with one command, scores it, and
# prints the run time. With real captures, replace the `sim` lines with the phone's files.
set -euo pipefail
cd "$(dirname "$0")/.."
RS=.venv/bin/roomscope
OUT=${OUT:-runs/walk_in}
mkdir -p "$OUT" data/captures/walk_in
for tier in lidar video photo; do
  capture=data/captures/walk_in/$tier
  extra=(); [[ $tier == lidar ]] && extra=(--rgb-width 960)
  [[ -f $capture/ground_truth.json ]] || $RS sim benchmark/sim/flat_a.yaml --tier $tier --out "$capture" "${extra[@]}" > /dev/null
  start=$(date +%s)
  $RS run "$capture" --out "$OUT/$tier" | tail -2
  echo "   $tier: $(( $(date +%s) - start )) s"
  $RS eval "$OUT/$tier/plan.json" "$capture/ground_truth.json" | tail -6
done
