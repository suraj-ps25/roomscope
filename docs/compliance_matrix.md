# Compliance matrix

Every requirement in the brief → where it lives → the evidence → status.

**done**: built and measured. **partial**: built, with a named gap. **pending**: needs real
phone captures, which this build didn't have (no iPhone was available). Numbers are in
[`benchmark_report.md`](benchmark_report.md), generated from metrics files by
`benchmark/run_all.sh`; nothing here is typed in by hand.

## Part 1: capture

| Requirement | Where | Evidence | Status |
|---|---|---|---|
| Capture route: stock apps + one-page protocol (Route 2) | `docs/capture_protocol.md` | numbered steps per tier, hand-off, failure cases | done |
| Photo tier: stills per room, any iPhone 15+, one stitched plan | `io/photos.py`, `tiers/photo.py`, `geometry/registration.py` | corner shots + doorway threshold pairs; registration by matched 3D points; rooms joined through the shared threshold | done |
| Video tier: one handheld clip, any iPhone 15+ | `io/video.py`, `tiers/video.py` | rooms from on-the-spot turns; lens self-calibrated from the turns; visual compass for headings | done |
| LiDAR tier: depth + poses on Pro devices | `io/stray.py`, `tiers/lidar.py` | Stray Scanner ingest → pose graph → plane-anchored adjustment | done |
| Same output contract from every tier; intervals widen as data thins | `build.py` (`TierProfile`, per-room scale sigma) | one builder, one schema | done |
| Device matrix | `docs/device_matrix.md` | tier × hardware × measured accuracy | done (synthetic + scanned rooms); real pending |

## Part 2: output contract and gates

| Requirement | Where | Evidence | Status |
|---|---|---|---|
| Per-room plan: walls, ceiling, floor area, openings | `geometry/layout.py`, `geometry/sparse_layout.py`, `geometry/openings.py` | `rooms[]` in `plan.json` | done |
| Stitched multi-room plan, correct adjacency | `geometry/rooms.py` (LiDAR), `geometry/stitch.py` (photo, video) | adjacency scored per run | done on the synthetic flat for all three tiers; see report for per-capture results |
| Damage regions per surface, class and metric extent | `damage/ortho.py`, `damage/detect.py` | staged decals on the synthetic flat | partial: synthetic staging only; real staged damage pending |
| Concealed-damage flags naming the rule | `damage/rules.py` | `concealed_flags[]` with rule id, text and evidence; unit-tested | done |
| Scope line items keyed to surfaces | `damage/scope.py` | `scope[]` | done |
| Interval on every measurement | `model.py::Measurement`, `schema.py` | schema requires `ci_low/ci_high`; validator checks containment | done |
| One command per capture | `cli.py` | `roomscope run <capture> --out <dir>` | done |
| JSON to the published schema | `schema/floorplan.schema.json` | validated on every run | done |
| Rendered plan | `render.py` | `plan.png` | done |
| Openings ≤ 2 cm on ≥ 85% (misses and phantoms count) | `benchmark/evaluate.py` | per-run openings gate | measured per tier in the report; real pending |
| Ceiling ≤ 1.5 cm; repeat spread ≤ 1 cm | `benchmark/evaluate.py` | per-run ceiling gate; repeatability pairs | measured (synthetic); real pending |
| Repeatability ≤ 1 cm or 0.5% per wall | `evaluate.repeatability`, `benchmark/run_sim.py` | three independent LiDAR captures of the flat, pairwise | measured (synthetic); real pending |
| Drift accountability + on/off ablation | `geometry/drift.py`, `geometry/planes.py`, `--no-drift-correction` | every LiDAR capture run both ways | done |
| Photo whole-property stitch: no overlaps, footprint ±8% | `tiers/photo.py::threshold_links`, `geometry/stitch.py` | overlap and footprint per run | done (synthetic); real pending |
| Walls: photo ±8%, video ±3%; calibration scored at every tier | `build.py`, `benchmark/calibrate.py` | scanned-room and synthetic results; held-out interval coverage | partial: photo meets ±8% on most scanned rooms; video ±3% is not met with monocular depth (per-scene scale bias, measured) |

## Parts 3–5

| Requirement | Where | Evidence | Status |
|---|---|---|---|
| Head-to-head vs a consumer app on 2 rooms (≥ 70% beat or tie) | `benchmark/head_to_head/` | protocol + scorer (tested on synthetic numbers) | pending: needs an iPhone and the two rooms |
| Fix loop: declaration, regenerable before/after, diff | `docs/fix_loop.md`, `benchmark/fix_loop/` | five regenerable entries (each one commit, same capture both sides, one scorer) + history | done |
| Commit as you work | git history | small commits with the measured before/after in the message | done |
| Walk-in test: every tier runs cold | `scripts/setup.sh`, `README.md` | fresh clone → tests passing in 2.5 min (weights cached); all tiers run on generated captures | done (synthetic); real pending |

## Deliverables

| # | Deliverable | File | Status |
|---|---|---|---|
| 1 | Compliance matrix | `docs/compliance_matrix.md` | this file |
| 2 | Capture route + device matrix | `docs/capture_protocol.md`, `docs/device_matrix.md` | done |
| 3 | Repo, README to running in < 15 min, one command per capture | `README.md`, `scripts/setup.sh` | done |
| 4 | Reproduction bundle: raw → every number; cached model outputs replay; live path runs | `benchmark/run_all.sh`, `models/cache.py` | done |
| 5 | Benchmark report | `docs/benchmark_report.md` (generated) | done for synthetic + scanned rooms; real pending |
| 6 | Fix-loop bundle | `docs/fix_loop.md`, `benchmark/fix_loop/` | done |
| 7 | Technical report (≤ 6 pages) | `docs/technical_report.md` | done |
| 8 | Raw benchmark data | `data/captures/` (generated, not in git) + `scripts/fetch_replica.sh` | done (synthetic + scanned); real pending |

## Constraints

| Constraint | How it's met |
|---|---|
| Handheld consumer capture only | stock App Store apps, handheld |
| Pretrained models disclosed | MoGe-2 (Microsoft, MIT), MapAnything (Meta, Apache-2.0, fallback only); see `README.md` |
| Runs without calling our infrastructure | all local; weights from Hugging Face Hub by script |
| Weights fetched by script | `scripts/setup.sh`; cache under `~/.cache` |
| Mirrors, glass, wet-look surfaces, low light | mirror test (`geometry/openings.py`); glass returns treated as open; distorted depth views dropped (`tiers/photo.py`); protocol lighting step |
