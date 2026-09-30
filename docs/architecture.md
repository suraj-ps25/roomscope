# Architecture & plan

Working design doc. Grows with the build. Reflects current thinking, not a finished
system — sections marked **(open)** are decisions still to lock.

## Goal

Handheld iPhone capture → dimensioned per-room plans → one stitched whole-property
plan with correct adjacency → per-surface damage + concealed-damage flags → scope line
items → a confidence interval on every measurement. Same output contract at three input
tiers (photo / video / LiDAR), intervals widening honestly as sensor data thins. One
command per capture. Everything runs locally; weights fetched by script.

## Capture routes (open — pick one)

- **Route 1 — custom iOS app.** ARKit/RoomPlan/raw LiDAR + camera + IMU, shipped as a
  TestFlight/dev build. Best control of raw data and best capture-route score; needs a
  Mac + Xcode + Apple Developer account.
- **Route 2 — stock app + one-page protocol.** e.g. a LiDAR logging app (Record3D / 3D
  Scanner App) for depth+poses+intrinsics, native camera for photo/video tiers. Fastest
  to end-to-end; a non-engineer follows the page literally at the defense.

Whichever we pick, the pipeline consumes a **normalized capture bundle** (frames,
intrinsics, optional poses, optional depth), so the route is swappable behind ingest.

## Tiers

All three produce the same contract; what differs is how geometry and metric scale are
recovered, and how wide the intervals get.

- **LiDAR** — depth + poses + intrinsics measured. Fuse depth into a point cloud, extract
  planes (walls/floor/ceiling), measure directly. Tightest intervals. Drives the
  head-to-head and the metric gates.
- **Video** — structure-from-motion / visual-inertial odometry recovers poses up to a
  scale that IMU + known priors resolve; densify to geometry. Medium intervals.
- **Photo** — 2–8 stills per room, no depth, no poses. SfM per room recovers geometry up
  to an unknown metric scale; a learned metric-depth prior and/or a physical reference
  resolves scale. Widest intervals. **This is the floor and the hardest gate.**

### The metric-scale problem (photo/video)

Monocular geometry is scale-free. Candidate scale sources, in preference order:
1. Learned **metric** depth (e.g. Depth-Anything-V2-Metric / UniDepth) as a scale prior.
2. A known physical reference placed in view (protocol can require a door leaf, a sheet
   of A4, a tape segment) — cheap, robust, honest.
3. A ceiling-height prior when nothing else is available — widest interval.

Scale uncertainty propagates into the confidence interval. Thin input ⇒ wide interval.
"Confident garbage on thin input caps the total score," so the intervals must be *earned*.

## Pipeline stages

`ingest → reconstruct → measure → stitch → damage → scope → render`
(mirrored in `src/roomscope/cli.py::STAGES`.)

- **ingest** — normalize any route/tier into the capture bundle; detect tier from layout.
- **reconstruct** — per-room geometry: LiDAR depth fusion, or SfM (pycolmap/GLOMAP),
  plus plane extraction (RANSAC on the cloud) for wall/floor/ceiling surfaces.
- **measure** — wall lengths, ceiling height, floor area, opening detection + widths,
  each with an interval.
- **stitch** — place rooms into one coordinate frame via shared openings (doorways) and
  adjacency; apply drift correction (pose graph / loop closure) on multi-room walks.
- **damage** — per-surface segmentation into damage classes + metric extent; rule layer
  emits concealed-damage flags naming the rule that fired.
- **scope** — map damage regions to repair line items keyed to surfaces.
- **render** — write `plan.json` (schema) and `plan.png`.

## Drift accountability

Multi-room walks accumulate pose drift. Plan: pose-graph optimization with loop closure
(and/or plane-anchored correction) over the room-to-room transform graph, with an
**ablation** (stitched footprint with correction on vs off) in the benchmark report.
"Poses used as-is" is an automatic fail on this gate, so correction is not optional.

## Calibration & confidence intervals

Every measurement carries `{value, ci_low, ci_high}`. Intervals come from propagating
the dominant uncertainty per tier (scale uncertainty, plane-fit residuals, pose
covariance) and are **calibrated** against the benchmark ground truth: a well-calibrated
interval contains the true value at its stated coverage. Calibration is scored at every
tier. The report states, per gate, whether error is **repeatable-but-biased** (systematic,
fixable by calibration) or **unrepeatable** (noise) — both fail the raw gate but demand
different fixes.

## Damage & concealed-damage rules

- Damage regions: per-surface class (≥2 classes staged in the benchmark) + metric extent.
- Concealed-damage flags fire from explicit rules (e.g. moisture staining below a window,
  discoloration pattern consistent with hidden leak) and the output names the rule.
- Real surfaces include mirrors, glass, wet-look floors, low light — handled with surface
  masking / reflection rejection so they don't corrupt geometry or damage calls. **(open)**

## Output schema

`schema/floorplan.schema.json` is the published contract. Core shape:

- `property`: tier, capture metadata, coordinate frame.
- `rooms[]`: `walls[]`, `ceiling_height` (measurement), `floor_area` (measurement),
  `openings[]` (type + `width` measurement), `surfaces[]`.
- `surfaces[]`: id, kind (wall/floor/ceiling), `damage_regions[]` (class + extent
  measurement), plus `concealed_flags[]` (rule id + description).
- `adjacency[]`: room-to-room edges through openings (the stitch graph).
- `scope[]`: line items keyed to a surface.
- Every scalar dimension is a `measurement`: `{value, unit, ci_low, ci_high, method}`.

(Schema file lands in the next commit.)

## Benchmark set

Self-built, composition fixed so it can't be flattered: one multi-room capture (3+ rooms
+ connector), one furnished room with staged damage across two classes, the same rooms at
all three tiers (photo tier as per-room folders), at least one room captured twice
(repeatability), laser/tape ground truth on everything, raw sensor data submitted. A gate
harness scores opening widths, ceiling height, repeatability, drift ablation, and the
photo-tier whole-property stitch.

## Fix loop

Pick the single worst gate on our own benchmark, state the failing number, hypothesize
root cause with evidence, predict the post-fix number, ship the fix, and submit
regenerable before/after runs plus a readable diff.

## Open decisions (to lock next)

1. Capture route (1 vs 2) and, for LiDAR, which logging app / SDK surface.
2. SfM backend for photo/video (pycolmap vs GLOMAP) and metric-depth model.
3. Damage segmentation model (pretrained vs light fine-tune) and the two staged classes.
4. Incumbent app for the head-to-head (poly.cam vs magicplan).
