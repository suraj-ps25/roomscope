#!/usr/bin/env bash
# Scanned rooms for the photo/video benchmark: Replica's culled meshes as packaged by the
# NICE-SLAM authors (151 MB; Replica dataset licence, research use).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/public/replica
cd data/public/replica
[[ -d cull_replica_mesh ]] && { echo "already present"; exit 0; }
curl -L --fail -o cull_replica_mesh.zip https://cvg-data.inf.ethz.ch/nice-slam/cull_replica_mesh.zip
unzip -q cull_replica_mesh.zip && rm cull_replica_mesh.zip
echo "data/public/replica/cull_replica_mesh ready"
