# Compliance matrix

Every requirement in the brief → where it lives → the evidence → status.

**done**: built and measured. **partial**: built, with a named gap. **pending**: needs our
own phone captures, which this build didn't have (no iPhone was available). "Real" below
means real iPad Pro LiDAR recordings of real rooms with laser-scan truth (ARKitScenes, dev
and held-out visits); it exercises the LiDAR tier, not the photo or video tiers. Numbers are in
[`benchmark_report.md`](benchmark_report.md), generated from metrics files by
`benchmark/run_all.sh`; nothing here is typed in by hand.

## Part 1: capture

| Requirement | Where | Evidence | Status |
|---|---|---|---|
| Capture route: stock apps + one-page protocol (Route 2) | `docs/capture_protocol.md` (one page, 517 words), `docs/capture_notes.md` | what to install, how to walk, how long, what to avoid, hand-off, per tier; failure cases in the notes | done |
| Photo tier: stills per room, any iPhone 15+, one stitched plan | `io/photos.py`, `tiers/photo.py`, `geometry/registration.py` | corner shots + doorway threshold pairs; registration by matched 3D points; rooms joined through the shared threshold | done |
| Video tier: one handheld clip, any iPhone 15+ | `io/video.py`, `tiers/video.py` | rooms from on-the-spot turns; lens self-calibrated from the turns; visual compass for headings | done |
| LiDAR tier: depth + poses on Pro devices | `io/stray.py`, `tiers/lidar.py` | Stray Scanner ingest → pose graph → plane-anchored adjustment | done |
| Same output contract from every tier; intervals widen as data thins | `build.py` (`TierProfile`, per-room scale sigma) | one builder, one schema | done |
| Device matrix | `docs/device_matrix.md` | tier × hardware × measured accuracy | done (synthetic, scanned rooms, real LiDAR); iPhone photo/video pending |

## Part 2: output contract and gates

| Requirement | Where | Evidence | Status |
|---|---|---|---|
| Per-room plan: walls, ceiling, floor area, openings | `geometry/layout.py`, `geometry/sparse_layout.py`, `geometry/openings.py` | `rooms[]` in `plan.json` | done |
| Stitched multi-room plan, correct adjacency | `geometry/rooms.py` (LiDAR), `geometry/stitch.py` (photo, video) | adjacency scored per run | done on the synthetic flat for all three tiers; see report for per-capture results |
| Damage regions per surface, class and metric extent | `damage/ortho.py`, `damage/detect.py` | staged decals on the synthetic flat, scored at their staged place; undamaged real rooms for false positives | partial: 3/3 staged found; real rooms still report 2–15 false regions each; no real staged damage |
| Concealed-damage flags naming the rule | `damage/rules.py` | `concealed_flags[]` with rule id, text and evidence; unit-tested | done |
| Scope line items keyed to surfaces | `damage/scope.py` | `scope[]` | done |
| Interval on every measurement | `model.py::Measurement`, `schema.py` | schema requires `ci_low/ci_high`; validator checks containment | done |
| One command per capture | `cli.py` | `roomscope run <capture> --out <dir>` | done |
| JSON to the published schema | `schema/floorplan.schema.json` | validated on every run | done |
| Rendered plan | `render.py` | `plan.png` | done |
| Openings ≤ 2 cm on ≥ 85% (misses and phantoms count) | `benchmark/evaluate.py` | per-run openings gate | synthetic: passes at every tier with true depth, LiDAR 7–9/9; **real LiDAR: fails** (7 of 23 dev and 0 of 8 held-out openings within 2 cm) |
| Ceiling ≤ 1.5 cm; repeat spread ≤ 1 cm | `benchmark/evaluate.py` | per-run ceiling gate; repeatability pairs | synthetic passes; **real LiDAR: partial** (ceilings within 1.5 cm: dev 12 of 14, held-out 6 of 7, after the depth calibration); across captures of one real room: 3 rooms pass (spread 0.3–0.9 cm), 1 small room unrepeatable (4.0 cm); before the depth calibration, repeatable but biased (report: 'Ceiling across captures') |
| Repeatability ≤ 1 cm or 0.5% per wall | `evaluate.repeatability`, `benchmark/run_sim.py`, `benchmark/real/run_real.py` | synthetic: three captures pairwise; real: recordings of the same room | **partial**: synthetic 5–16 of 16 walls; real bathroom 2–3 of 4 (worst 5.1 cm); pairs in small rooms often differ in wall count |
| Drift accountability + on/off ablation | `geometry/drift.py`, `geometry/planes.py`, `--no-drift-correction` | every LiDAR capture run both ways | done; also on the assessors' real flat: connections 2 → 6, two captures agree on footprint to 0.1 m² (1.7 m² with drift off) |
| Photo whole-property stitch: no overlaps, footprint ±8% | `tiers/photo.py::threshold_links`, `geometry/stitch.py` | overlap and footprint per run | done (synthetic); iPhone captures pending |
| Walls: photo ±8%, video ±3%; calibration scored at every tier | `build.py`, `benchmark/calibrate.py` | scanned-room and synthetic results; held-out interval coverage | partial: on scanned rooms with the depth model, photo meets ±8% on 4 of 8 and video ±3% on 0 of 8 (the depth model's per-scene scale bias, measured at 5.9% across scenes); with true depth 6 of 8 and 5 of 8; held-out interval coverage 88–96% after calibration on synthetic and scanned rooms; **real LiDAR: 94% left-out coverage on dev, 60% median on the held-out visit (target 90%; up from 20% after an absolute error floor, `fix_loop.md`)**; photo and video on real iPad frames of the same rooms (off-protocol: walk-arounds, not corner shots or turns): walls 33–94 cm off, intervals wide enough to cover (70–100%) |

## Parts 3–5

| Requirement | Where | Evidence | Status |
|---|---|---|---|
| Head-to-head vs a consumer app on 2 rooms (≥ 70% beat or tie) | `benchmark/head_to_head/` | 2 rooms: Pointorama (point-cloud tool, automatic) on the same raw iPad LiDAR clouds, scored against laser truth: ours beats or ties 14/15 as shipped; 10/15 (67%) with our depth calibration off | met as shipped, not on identical raw input; a professional tool on an uploaded cloud, not a consumer app's own scan (no phone) |
| Fix loop: declaration, regenerable before/after, diff | `docs/fix_loop.md`, `benchmark/fix_loop/` | ten regenerable entries, three from real captures, (each one commit, same capture both sides, one scorer; tables in `benchmark/fix_loop/results/`) + history | done |
| Commit as you work | git history | small commits with the measured before/after in the message | done |
| Walk-in test: every tier runs cold | `scripts/setup.sh`, `scripts/walk_in.sh` | rehearsed on a fresh clone from GitHub: setup 1.5 min (weights cached), then every tier cold, one command each (LiDAR 6.4 min, all gates pass; video 5 min; photo 31 s) | done (synthetic); on our own iPhone captures pending |

## Deliverables

| # | Deliverable | File | Status |
|---|---|---|---|
| 1 | Compliance matrix | `docs/compliance_matrix.md` | this file |
| 2 | Capture route + device matrix | `docs/capture_protocol.md`, `docs/device_matrix.md` | done |
| 3 | Repo, README to running in < 15 min, one command per capture | `README.md`, `scripts/setup.sh` | done: fresh clone with empty caches to a first LiDAR plan in 82 s; photo/video weights add ~3.5 min of download |
| 4 | Reproduction bundle: raw → every number; cached model outputs replay; live path runs | `benchmark/run_all.sh`, `models/cache.py` | done |
| 5 | Benchmark report | `docs/benchmark_report.md` (generated) | done: gates at all three tiers (synthetic, scanned rooms, real dev + held-out), repeatability tables, head-to-head table, timing (cached and live) |
| 6 | Fix-loop bundle | `docs/fix_loop.md`, `benchmark/fix_loop/` | done |
| 7 | Technical report (≤ 6 pages) | `docs/technical_report.md` | done: 5 A4 pages rendered; architecture, tiers + device matrix, drift, error budget (§6), calibration, fix loop, failure modes |
| 8 | Raw benchmark data | app exports: `benchmark/head_to_head/results/` (Pointorama DXF/IFC); `data/captures/` (generated, not in git), `scripts/fetch_replica.sh`, `benchmark/real/fetch_arkitscenes.py`; laser truth in `benchmark/ground_truth/` | done (synthetic, scanned, real LiDAR) |

## Constraints

| Constraint | How it's met |
|---|---|
| Handheld consumer capture only | stock App Store apps, handheld |
| Pretrained models disclosed | MoGe-2 (Microsoft, MIT), MapAnything (Meta, Apache-2.0, fallback only); see `README.md` |
| Runs without calling our infrastructure | all local; weights from Hugging Face Hub by script |
| Weights fetched by script | `scripts/setup.sh`; cache under `~/.cache` |
| Mirrors, glass, wet-look surfaces, low light | mirror test (`geometry/openings.py`); glass returns treated as open; distorted depth views dropped (`tiers/photo.py`); protocol lighting step. **Real evidence:** in a real bathroom recording (ARKitScenes 47429912) the mirror over the sink was told apart from the window beside it on all three recordings, beside a glass shower screen; the laser survey applies the same reflection test |
