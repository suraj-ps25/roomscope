# Compliance matrix

Every requirement in the brief → where it lives → the artifact → status.
Status: **done** (built and exercised), **partial** (built, gaps named), **pending**
(needs real captures or not built yet).

## Part 1: capture

| Requirement | File | Artifact | Status |
|---|---|---|---|
| Capture route (Route 2: stock apps + one-page protocol) | `docs/capture_protocol.md` | numbered steps per tier, hand-off, failure cases | done |
| Photo tier: 2–8 stills per room, any iPhone 15+, folders → one stitched plan | `src/roomscope/io/photos.py`, `src/roomscope/tiers/photo.py` | per-room reconstruction | partial: cross-room stitching in progress |
| Video tier: handheld walkthrough, any iPhone 15+ | `src/roomscope/io/video.py`, `src/roomscope/tiers/video.py` | keyframes → multi-view → shared core | partial: runs; speed and accuracy being measured |
| LiDAR tier: depth, poses, intrinsics on Pro devices | `src/roomscope/io/stray.py`, `src/roomscope/tiers/lidar.py` | Stray Scanner ingest → plan | done |
| Same output contract from each tier, intervals widening as data thins | `src/roomscope/build.py` (`TierProfile`, scale sigma) | one builder for all tiers | done |
| Device matrix | `docs/device_matrix.md` | tier × hardware × honest accuracy | partial: real numbers pending |

## Part 2: output contract and gates

| Requirement | File | Artifact | Status |
|---|---|---|---|
| Per-room plan: walls, ceiling height, floor area, openings | `geometry/layout.py`, `geometry/openings.py` | `rooms[]` in `plan.json` | done |
| Stitched multi-room plan with correct adjacency | `geometry/rooms.py`, `geometry/openings.py::match_doors` | `adjacency[]`, `plan.png` | done (LiDAR); photo pending |
| Per-surface damage regions, class and metric extent | `damage/ortho.py`, `damage/detect.py` | `surfaces[].damage_regions[]` | partial: needs real staged damage |
| Concealed-damage flags with the rule that fired | `damage/rules.py` | `concealed_flags[]` (rule id + rule text + evidence) | done |
| Scope line items keyed to surfaces | `damage/scope.py` | `scope[]` | done |
| Confidence interval on every measurement | `model.py::Measurement`, `schema.py` | schema requires `ci_low/ci_high`; validator checks containment | done |
| One command per capture | `cli.py` | `roomscope run <capture> --out <dir>` | done |
| JSON to the published schema | `schema/floorplan.schema.json` | validated on every run | done |
| Rendered plan | `render.py` | `plan.png` | done |
| Opening widths ≤ 2 cm on ≥ 85% (misses and phantoms count) | `benchmark/evaluate.py` | `openings` gate | done (synthetic 9/9); real pending |
| Ceiling ≤ 1.5 cm; repeat spread ≤ 1 cm; biased vs unrepeatable stated | `benchmark/evaluate.py` | `ceiling` gate, repeatability | partial: report wording pending |
| Repeatability ≤ 1 cm or 0.5% per wall | `benchmark/evaluate.py::repeatability`, `benchmark/run_sim.py` | seed pairs / twice-captured room | partial: real pending |
| Drift accountability + on/off ablation | `geometry/drift.py`, `geometry/planes.py`, `--no-drift-correction` | pose graph + plane/jamb adjustment; ablation runs | done (synthetic ablation) |
| Photo-tier whole-property stitch, no overlaps, footprint ±8% | `tiers/photo.py`, `benchmark/evaluate.py` (overlap, footprint) | | pending |
| Photo ±8%, video ±3%, calibration scored at every tier | `build.py`, `calibration/` | per-tier interval multipliers | pending (needs real truth) |

## Parts 3–5

| Requirement | File | Artifact | Status |
|---|---|---|---|
| Head-to-head vs a consumer app on 2 rooms (≥ 70% beat or tie) | `benchmark/head_to_head/` | magicplan export + table | pending (needs captures) |
| Fix loop: declaration, before/after regenerable, diff | `docs/fix_loop.md`, `benchmark/fix_loop/` | | pending (needs real benchmark) |
| Commit as you work | git history | | done |

## Deliverables

| # | Deliverable | File | Status |
|---|---|---|---|
| 1 | Compliance matrix | `docs/compliance_matrix.md` | this file |
| 2 | Capture route + device matrix | `docs/capture_protocol.md`, `docs/device_matrix.md` | done / partial |
| 3 | Repo, README to running in < 15 min, one command per capture | `README.md`, `scripts/setup.sh` | partial: setup script pending |
| 4 | Reproduction bundle (raw → every number; cached model outputs replay; live path runs) | `models/cache.py`, `benchmark/` | partial |
| 5 | Benchmark report: gates × tiers, repeatability, head-to-head, timing | `docs/benchmark_report.md` | pending |
| 6 | Fix loop bundle | `benchmark/fix_loop/` | pending |
| 7 | Technical report (≤ 6 pages) | `docs/technical_report.md` | pending |
| 8 | Raw benchmark data | `benchmark/captures/` (fetched, not in git) | pending |

## Constraints

| Constraint | How it's met |
|---|---|
| Handheld consumer capture only | stock App Store apps, handheld |
| Pretrained models disclosed | MoGe-2 (Microsoft, MIT), MapAnything (Meta, Apache-2.0); see `README.md` |
| Runs without calling our infrastructure | all local; weights from Hugging Face Hub by script |
| Weights fetched by script | `scripts/setup.sh` (pending), cache under `~/.cache` |
| Mirrors, glass, wet-look surfaces, low light covered | mirror test (`geometry/openings.py`), glass as open, floor as dominant upward level, protocol lighting step |
