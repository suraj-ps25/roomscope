# roomscope

Turn a handheld iPhone capture of a home into a **dimensioned, stitched floor plan with
per-surface damage and a repair scope**, from photos, video or LiDAR.

One command per capture produces `plan.json` (to [`schema/floorplan.schema.json`](schema/floorplan.schema.json))
and `plan.png`. The output contains:
- per room: walls, ceiling height, floor area and openings;
- the whole property stitched, with adjacency;
- damage regions per surface, with class and metric extent;
- concealed-damage flags naming the rule that fired;
- scope line items keyed to surfaces;
- a **confidence interval on every number**.

## Quick start

```
git clone git@github.com:suraj-ps25/roomscope.git && cd roomscope
scripts/setup.sh --lidar      # ~2-3 min: LiDAR tier, no model weights
scripts/setup.sh              # adds photo/video models (~7 GB of weights)
```

Capture following [`docs/capture_protocol.md`](docs/capture_protocol.md), then:

```
.venv/bin/roomscope run <capture> --out runs/<name>
```

`<capture>` can be a Stray Scanner recording folder (LiDAR), a video file (video), or a
folder with one sub-folder of photos per room (photo). The tier is detected from the
folder's shape; force it with `--tier`.

Try it without a phone, on a synthetic flat with exact ground truth:

```
.venv/bin/roomscope sim benchmark/sim/flat_a.yaml --tier lidar --out data/captures/sim_lidar --rgb-width 480
.venv/bin/roomscope run data/captures/sim_lidar --out runs/sim_lidar
.venv/bin/roomscope eval runs/sim_lidar/plan.json data/captures/sim_lidar/ground_truth.json
```

## How it works

```
capture ─ ingest ─┬─ LiDAR: depth + ARKit poses ─ drift correction ──────────────────────────┐
                  ├─ video: find room turns → lens from the turns → rotations + MoGe-2 depth ─┤─ rooms ─ walls ─ openings ─ stitch ─ damage ─ plan
                  └─ photo: per room: MoGe-2 + MapAnything ───────────────────────────────────┘
```

- **Video**: each room is the on-the-spot turn the protocol asks for. The turn calibrates
  the lens (pure rotation fixes K), gives every view's rotation, and becomes a panorama
  with per-view MoGe-2 depth made consistent across overlaps; a line-based compass gives
  each room's heading for stitching.
- **Drift** (multi-room walks): a 4-DoF pose graph with ICP loop closures, then a
  plane-anchored bundle adjustment whose landmarks are the walls, floors, ceilings and
  door jambs. `--no-drift-correction` gives the ablation.
- **Rooms**: free-space carving bounded by the wall traces that reach the ceiling.
  Regions the camera never entered (mirror reflections, glimpses) are rejected.
- **Walls**: robust line fits on each wall's own points. Angles are fitted, not snapped
  to 90°.
- **Openings**: every depth ray votes (on the wall / behind it / through it / no return).
  Widths come from the jamb surfaces; a mirror is told apart from a doorway by reflecting
  its see-through points back onto the room.
- **Damage**: metric orthophotos per surface, anomalies against the wall's own
  background, explicit concealed-damage rules, and scope.
- **Intervals**: propagated per measurement (fit uncertainty through the corner
  geometry, plus the tier's scale uncertainty), then calibrated per tier.

Details: [`docs/architecture.md`](docs/architecture.md). What each tier delivers:
[`docs/device_matrix.md`](docs/device_matrix.md). Requirement coverage:
[`docs/compliance_matrix.md`](docs/compliance_matrix.md).

## Benchmark

```
.venv/bin/roomscope eval runs/<name>/plan.json benchmark/ground_truth/<property>.yaml
```

This scores the brief's gates: opening widths ≤ 2 cm on ≥ 85% (missed and phantom
openings count against), ceiling ≤ 1.5 cm, repeatability, adjacency, overlaps, footprint,
and interval coverage. See [`benchmark/README.md`](benchmark/README.md) for how ground
truth is measured.

## Models and data (disclosure)

Everything runs locally. Weights come from Hugging Face Hub the first time they're
needed (`scripts/setup.sh` pre-fetches them).

| Component | Source | Licence | Used for |
|---|---|---|---|
| MoGe-2 (`Ruicheng/moge-2-vitl-normal`) | Microsoft | MIT | metric depth and scale (photo, video) |
| MapAnything (`facebook/map-anything-apache`) | Meta | Apache-2.0 | multi-view poses and depth (photo, video) |
| DINOv2 (loaded by MapAnything) | Meta | Apache-2.0 | image encoder |
| Stray Scanner | Stray Robots | MIT (app) | LiDAR capture |
| ARKitScenes | Apple | non-commercial research licence | development and validation only, not the benchmark |

Model outputs are cached by content hash and replay bit-for-bit. `ROOMSCOPE_CACHE=off`
forces the live path.

## Layout

```
src/roomscope/   io/ (Stray, ARKitScenes, video, photos)  tiers/  geometry/  damage/
                 models/  build.py  render.py  schema.py  cli.py  sim/ (synthetic captures)
schema/          published JSON output contract
benchmark/       ground truth, sweep script, synthetic scenes
docs/            capture protocol, device matrix, compliance matrix, architecture
scripts/         setup
```
