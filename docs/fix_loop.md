# Fix loop

How a failing number becomes a fix, and how anyone can regenerate the before and after.

## The loop

1. **Run the benchmark** and read the failing gate, not the average. Each run scores
   every gate (`roomscope eval`).
2. **Declare** the failure before touching code: symptom (the number), suspected cause,
   and the measurement that would confirm it.
3. **Isolate** the cause with the smallest experiment that can tell causes apart, often
   an ablation (`--oracle-depth` swaps the depth model for rendered truth;
   `--no-drift-correction` turns drift correction off) or a diagnostic of one stage
   against truth.
4. **Fix in one commit**, with the measured before/after in the message.
5. **Re-run every tier the change touches.** Shared code (openings, layout, stitching)
   serves all three tiers; a fix for one can break another.
6. **Record** the entry in `benchmark/fix_loop/fixes.yaml`.

## Regenerating

```
python benchmark/fix_loop/regenerate.py <id>     # or --all
```

For each entry, both commits are checked out into git worktrees and run on the
**identical capture**, each with its own code; both plans are scored by the **current**
evaluator, so before and after are measured with the same ruler. Output goes to
`runs/fix_loop/<id>/{before,after}/` (plan.json, plan.png, metrics.json) with a
`diff.md` table; the last regeneration's tables are kept in
[`benchmark/fix_loop/results/`](../benchmark/fix_loop/results/).

## Entries

### video-door-sill: oblique doorways read as windows (`8d57a7d` → `bf0c5d5`)

- **Symptom:** video tier, flat_a, oracle depth. 4 of 5 doors seen from the hallway and
  bathroom turns came out as windows with a 0.23 m sill, so they could not pair, and the
  plan did not stitch.
- **Isolation:** evidence counts in the strip under one hallway door showed identical
  face and through votes at 0.10–0.22 m: the same rays were counted twice.
- **Cause:** the "wall face" and "floor beyond the wall" vote classes overlapped. From
  the middle of a room the bottom of a side doorway is seen at 60–70°, where those rays
  land within the face tolerance.
- **Fix:** made the classes exclusive, and tightened the floor-beyond margin for the
  single-standpoint tiers.

| metric | before | after |
|---|---|---|
| openings within 2 cm | 4/13 (4 phantom windows) | 7/9 |
| adjacency | wrong | correct |

### lidar-floor-margin: the fix above regressed LiDAR (`470c532` → `822fab3`)

- **Symptom:** re-running the LiDAR tier after the door fix (step 5): openings 8/9 → 7/9,
  the hallway's bedroom door missed, adjacency wrong.
- **Isolation:** the pre-fix commit on the same new capture still passed (8/9, adjacency
  correct), so the cause was code, not the capture.
- **Cause:** the tightened margin was a share of the tier's face tolerance: 4.8 cm for
  photos but 1.6 cm for LiDAR, inside LiDAR depth noise at the wall base.
- **Fix:** the margin became an explicit per-tier tolerance (`floor_beyond_m`). LiDAR
  keeps 4 cm, where the two classes are disjoint exactly as before; photo/video keep
  the tighter margin.

### video-pixel-aspect: every video-tier height ~1% short (`bf0c5d5` → `470c532`)

- **Symptom:** video tier, flat_a, oracle depth. Ceilings 2.5–3.8 cm low in every room,
  although each ceiling was perfectly flat (±0.5 cm) and walls were within 4 mm.
- **Isolation:** a uniform vertical-only shortfall means a vertical scale error, not
  depth or pose. The self-calibrated focal was 0.6% from truth, too little to explain it;
  the decode's resize was the remaining step.
- **Cause:** decoding 720×1280 to 294×518 (patch multiples) scales x by 0.4083 and y by
  0.4047. Self-calibration then set fy = fx.
- **Fix:** the decode's pixel aspect is carried through the rotation search and into K.

| metric | before | after |
|---|---|---|
| ceiling max (cm) | 3.81 | 1.10 |
| wall max (cm) | 1.79 | 0.39 |
| openings within 2 cm | 7/9 | 9/9 |
| footprint error | −0.2% | −0.04% |

### lidar-loop-tear: wrong loop closures tore the pose graph (`d53e04a` → `7633a27`)

- **Symptom:** LiDAR, flat_a seed 1 (seed 0 passed): openings 5/10, adjacency wrong,
  walls up to 2.6 cm, a phantom strip of hallway outside the flat.
- **Isolation:** trajectory error against the simulator's true poses was fine up to the
  last 4 s, then grew linearly to 55 cm. Each fragment's solved correction against its
  ideal one (from truth) showed the break at fragments 73 → 74; the loop closures from 74
  and 75 back to the start all said "no drift" while the ideal said 0.6 m.
- **Cause:** those ICP closures had slid along the hallway. The Cauchy loss was applied to
  every edge, so the solver switched off the one odometry edge that disagreed with them.
- **Fix:** IRLS with Cauchy weights on loop closures only; odometry is never robustified.
  Trajectory error median 5.1 → 1.1 cm (seed 1), 4.5 → 0.7 cm (seed 0).
- **What the regeneration shows:** the trajectory fix alone did not restore seed 1's gates
  (openings 5/10 → 5/9, adjacency still wrong, ceiling 1.1 → 13.8 cm), because a second,
  independent fault was in segmentation; the regenerated table records exactly that, and
  the next entry fixes it. Trajectory error isn't in the plan, so it is measured against the
  simulator's poses by the diagnostic described above, not by `regenerate.py`.

### lidar-segmentation: rooms merged through gaps in the wall traces (`b31c18c` → `2d74357`)

- **Symptom:** seed 1 with its trajectory fixed: the hallway segmented together with the
  bathroom (adjacency wrong, ceiling 13.8 cm off).
- **Isolation:** the same segmentation on the old and new poses differed only in where
  traces stopped: with the better poses, a 5 m wall's trace ended 2.2 m short.
- **Cause:** wall directions were 2° histogram bin centres; a degree off moves a 5 m trace
  9 cm, so it fitted only part of the wall. Walls never seen just under the ceiling had no
  barrier, and the observed-ceiling mask left holes where nobody looked up.
- **Fix:** directions refined to the mean of the normals; traces extended along their own
  wall wherever it continues at any height; small gaps in the ceiling mask closed and
  enclosed holes filled.

### photo-depth-cap: far walls vanished from photo reconstructions (`bdb7174` → `a1a7bf5`)

- **Symptom:** photo tier with true depth on a scanned room (Replica room0): 3.27 × 3.90 m
  for a 4.58 × 7.71 m room, "no wall seen" on two sides.
- **Isolation:** a top-down plot of the fused cloud showed furniture but no walls.
- **Cause:** every depth map was cut at 4 m, the LiDAR's useful range, for every tier; a
  corner photo of a long room is mostly far wall. The same cap, in the opening detector,
  hid the living-room door. An earlier note blaming "corridor photos that don't overlap"
  for a short synthetic hallway was wrong: it was this cap.
- **Fix:** depth limits per tier (12 m for image depth), for fusion and for opening rays.

### Real captures (ARKitScenes iPad Pro recordings, laser truth)

The first run on real recordings with laser truth (`benchmark/real/`) failed in ways no
synthetic capture had. The entries below were found on the two **dev** visits; the
held-out visits were run after them. One change was made while running them: a 7-minute
held-out recording sat in the pose-graph solve for over 20 minutes, and `cf09349` vectorised
the solver's residuals (same algebra, agreeing to 1e-15; plans move by millimetres).
Their results are from that commit; no other change was made for them.

#### real-ceiling-step: a ceiling step cut a real room in two (`1ddba65` → `f2fa61a`)

- **Symptom:** a bathroom recording came out 1.94 × 1.85 m; the laser says 2.72 × 1.98 m.
- **Isolation:** our own fused cloud, overlaid on the laser slice, showed the whole room;
  the plan stopped at y = 2.4 m. A slice just under the ceiling showed why: one end of the
  room has a ceiling 10 cm lower, and the step's vertical face spans the room's width.
- **Cause:** barriers are traced in the band under the local ceiling, where door headers
  are. The step's face is just as straight and long as a header, but 10 cm tall.
- **Fix:** a barrier must come down from the ceiling by more than 15 cm over 30% of its
  length. Synthetic flat, three captures: unchanged within 0.2 cm.

#### A fix that was reverted, and a guard that fixed nothing measurable (`14ba4c7`, `73414b4`)

`14ba4c7` added a weak "rooms are square" prior on wall directions, for one real
recording's worst wall (6.8 → 4.5 cm). Step 5 caught it moving one synthetic wall a
fraction of a degree, enough to shift a glass window's jamb search by 2 cm (openings 9/9
→ 8/9); `73414b4` removes it. The same commit stops the layout clean-up from dropping a
wall when the room left would be implausible. It was written because a recording lost its
door-and-window wall that way, but regenerating showed that happened only with the prior
active: before and after the guard, without the prior, are identical. It stays as a guard
and is not counted as a fix.

#### The damage ruler was wrong (`f57442f`)

Looking at real false positives led to the synthetic ones, and to the scorer: it matched
staged damage by class and surface kind only. The committed benchmark's "crack found,
−24% length" was a 0.49 m **door-jamb edge on another wall**; the staged crack, above the
bedroom door, was never seen. Damage now counts only on the staged wall and within 15 cm
of the staged extent. Rescored, the earlier run is 2/3 found with 1 false positive, not
3/3 with none. This is a ruler fix, not a pipeline fix, so it has no regeneration entry:
every entry is scored by the corrected ruler.

#### real-damage-false-positives / synthetic-damage-recall (`f57442f` → `ed27738`)

- **Symptom:** an undamaged real living room reported 59 damage regions; the synthetic
  staged crack was never found.
- **Isolation:** each region drawn on its orthophoto: shelf and toy edges as cracks, a
  plant and a pendant lamp as mould, patches of differently exposed views as stains; on the
  synthetic flat, the strip above the door was simply never rendered.
- **Cause:** views ranked by how well they see a surface's *centre*; objects in front of a
  wall leak colour at the LiDAR's depth resolution; a mask's outline is a dark ridge; the
  phone's exposure changes between views. A first mask from "any point in front of the
  wall" covered whole synthetic walls: depth edges scatter flying points everywhere in
  front of them. Only dense geometry is an object.
- **Fix:** views chosen greedily for coverage of a 15 cm lattice; dense geometry (8
  neighbours within 4 cm) 3 cm to 1 m in front, or in a recess with no wall face, masked;
  cracks kept 3 cm clear of masks; per-view colour gains solved from overlaps.

### The declared fix: real-capture openings (`7e5e7be` → `71b6f9c`, `1b84c55`)

Declared before any code changed, in [`fix_declaration.md`](fix_declaration.md): the
worst gate in the benchmark, real LiDAR openings within 2 cm, at 2 of 29. Two evidenced
causes, multi-sash windows split into several openings and a floor-level sill classed as
a window, and a prediction: no new passes, phantoms 12 → 9, misses 6 → 5, scored 29 → 26.

Every real recording, before (`7e5e7be`) and after (`1b84c55`), each run at its own
commit and scored by the same evaluator:

| | predicted | measured |
|---|---|---|
| within 2 cm | 2 (unchanged) | 2 |
| matched, width off | 10 | 10 |
| missed | 6 → 5 | 6 → 5 |
| phantom | 12 → 9 | 12 → 9 |
| openings scored | 29 → 26 | 29 → 26 |
| gate share | 7% → 8% | 7% → 8% |

**The first version missed one prediction**, and it was caught. In `71b6f9c` the bathroom
door came out as a 0.48 m "window", not the predicted 0.82 m door. The cause was the plane
adjustment, which also uses openings, as landmarks: there the floor-level rule replaced the
door's measured sill with an assumed floor, the poses moved, and the final opening came out
different. `1b84c55` keeps the adjustment's landmarks raw and applies the fix to the plan's
openings only; with that, every predicted number is met.

**Why the gate still fails:** the fix moved detection (18 misses and phantoms → 14), not
widths. The remaining failures are the ones the declaration set aside:
- the bathroom door, which no recording filmed above 0.75 m;
- widths ±5 cm, of both signs, at the LiDAR's 256×192 depth resolution;
- plans whose wall count differs from the survey's, so their openings can't be placed on
  the truth's walls.

Regenerate the table above with `python benchmark/fix_loop/regenerate_real.py 7e5e7be 1b84c55`
(both commits run in worktrees on the same recordings, one evaluator); the before and after
runs are in `benchmark/fix_loop/results/real/`. Per-recording tables: `real-openings-*` in
[`results/`](../benchmark/fix_loop/results/).

### real-ceiling-scale: every real ceiling read low (`7f5bd43` → `561189d`)

- **Symptom:** all 14 real-room ceilings were low against laser truth, none high. Median
  −2.0 cm on dev and −1.7 cm on held-out; 4 of 10 held-out rooms within 1.5 cm.
- **Isolation:** similarity ICP of each dev recording's fused cloud on its laser scan
  needs a scale of 1.0087, 1.0088 and 1.0106 on the three that register well. Uniform
  shrinking of that size explains the walls running short too. The ceiling fit is not
  what's wrong: the depth is.
- **Fix:** a per-device depth-scale calibration measured on the dev visits only (×1.0088),
  applied to that device's depth only. Uncalibrated devices are left alone.
- **Expected** (not a committed declaration; this was the second fix): held-out ceilings within
  1.5 cm about 4 → 7 of 10.

| every real recording | before | after |
|---|---|---|
| held-out ceilings within 1.5 cm | 4/10 | **8/10** |
| held-out median ceiling error | −1.73 cm | +0.21 cm |
| dev ceilings within 1.5 cm | 0/4 | 4/4 |
| held-out wall error, median | 6.3 cm | 3.1 cm |
| held-out interval coverage, median | 50% | 71% |

Better than predicted. The gate still fails: it asks for every room, and two held-out rooms
remain at −1.6 and −2.7 cm.

**A regression it exposed:** the dev living room (47429922) now splits into a 1.3 m²
alcove and the rest (median wall error 6.6 → 28.7 cm against laser truth). Under the
ceiling a bulkhead runs across the alcove; at the new scale its face drops from the
ceiling just past the 15 cm that makes a barrier, and furniture on the same line passes
for wall below it. A test requiring wall below over half the barrier's length fixes this
room but splits or merges two of the bathroom recordings, so it was not shipped. The
segmentation decides bulkhead against header on thresholds, and this room sits on one.

### A fix that was not shipped: a door-height scale prior for photo and video

- **The failing gates:** photo walls within ±8% on 4 of 8 scanned rooms; video within ±3%
  on 0 of 8. With rendered true depth the same pipeline gets 6/8 and 5/8, so the limit is
  the depth model's per-scene metric scale (−14% to +4% across scenes).
- **The idea:** interior door heads are about 2.03 m almost everywhere (standard leaves
  are 1.98–2.04 m in Europe and the US). A detected door's measured head height would then
  fix its room's scale, as 2.03 / h.
- **The evidence, before writing any code:**
  - **Scanned rooms (where the gates fail):** no genuine door is detected in any of the 16
    photo and video runs, with true depth or the model. What is detected are 2.3–2.9 m
    office openings and glass partitions. The prior has nothing to act on there.
  - **Synthetic flat, true depth:** door heads read 1.99–2.05 m against truth of 2.00 and
    2.05 m. The cue itself is sound.
  - **Synthetic flat, depth model:** the errors are not one scale per room. Walls and
    ceiling disagree in sign within a room, and door heads measured with model depth
    scatter by about ±4%. Applied, the door scale changes photo worst walls 3.2, 4.2, 13.1
    and 0.9% to 3.5, 2.7, 8.6 and 2.4% (no room crosses the ±8% gate either way). It makes
    video worse: living 9.7% → ~12.3%, bathroom 74% → ~91%. A 2–3% prior would also
    narrow intervals the door measurements can't support.
- **Outcome:** not shipped. Neither gate would move, and the intervals would become less
  honest. The scale lever left is the depth model itself (a stronger or ensembled
  metric-depth model), or the user's phone: the LiDAR tier.

### A fix that was not shipped: an ensemble of metric-depth models

The photo ±8% and video ±3% wall gates fail because of MoGe-2's metric scale, which differs
from scene to scene. The idea was to correct MoGe-2's scale per room with a second
metric-depth model, keeping MoGe-2's shape. The test was measured before any pipeline
change: the per-scene median depth ratio on the same 16 views per scene that
`benchmark/depth_model_scale.py` uses, over 23 scenes (8 Replica renders against true
depth, 15 real ARKitScenes iPad recordings against their LiDAR depth). A constant bias
was removed leave-one-scene-out, since a fixed bias could be calibrated away and only the
scene-to-scene spread matters.

| model (licence, size) | spread across scenes (sd of log) | worst scene |
|---|---|---|
| MoGe-2 ViT-L (MIT), as shipped | 6.9% | 16.3% |
| Depth Anything V2 Metric Indoor Small (Apache-2.0, 25 M) | 13.1% | 27.1% |
| Apple Depth Pro (Apple ML Research Model licence, 1.9 GB) | 15.2% (7 Replica scenes; dropped after them: ~7 min per scene) | – |
| MoGe-2 + Depth Anything, geometric mean, weight 0.25 / 0.5 / 0.75 | 8.0% / 9.7% / 11.3% | 17.5% / 18.3% / 21.1% |

Every ensemble is worse than MoGe-2 alone. Per view, the two models' errors correlate only
weakly (0.36), and Depth Anything's scale moves with the domain much more than MoGe-2's:
median ratio 1.28 on renders against 1.46 on real frames (MoGe-2: 0.94 against 1.03). The
bar for shipping was a spread under ~3.5%; nothing came close, so no code changed. The
photo and video wall gates remain bound by monocular metric scale. Depth Anything V2 ran
at 0.2 s a view; Depth Pro at several seconds.

### real-sliver-walls: captures of one room disagreed on its wall sequence (`23cb877` → `7c2909e`)

Developed on visits 423441, 438802 and 467326, which were held out until then and are
dev now. A fresh held-out visit (422009) was fetched afterwards by the same rule and run
once (a second, 423461, failed to download and was dropped).

- **Symptom:** repeatability on real recordings. Of 12 pairs of recordings of one room,
  5 could not be compared at all (different wall counts), and the rest agreed on 11 of 42
  walls.
- **Isolation:** the wall lists of each recording side by side. Walls of 0.01–0.26 m
  appear in some recordings of a room and not in others: a cut corner, or a few-centimetre
  step between two faces of one wall.
- **Fix:** a wall under 0.3 m is absorbed into its neighbours in the reported layout. It
  is dropped where they meet, or two nearly collinear neighbours (under 5 cm apart) are
  merged; a larger step stays.
- **Two regressions caught on the way.** The assessors' sample flats lost room connections
  (6 → 4 and 5 → 4):
  - merging parallel walls up to 0.3 m apart moved real steps by 15 cm, so the merge is
    now limited to 5 cm;
  - absorbing slivers in the layouts the pose adjustment uses moved the poses, and a
    borderline door elsewhere flipped, so the adjustment's landmark layouts keep every
    wall.

| all real recordings (dev) | before | after |
|---|---|---|
| repeatability pairs comparable | 7 of 12 | 11 of 12 |
| walls agreeing within 1 cm / 0.5% | 11 of 42 | 16 of 58 |
| phantom openings | 8 | 6 |
| openings within 2 cm | 7 | 7 |
| walls scored against laser truth | 34 | 47 |

Synthetic LiDAR (7/9, 9/9, 9/9), photo and video with true depth (9/9), full-resolution
damage (3/3, 0 false positives) and the three assessor samples (5 and 6 connections) are
unchanged.

**Why the repeatability gate still fails:** the walls now pair up, but 1–5 cm differences
remain between recordings, against a 1 cm / 0.5% gate. The largest come from one
recording's ~1° skew in the iPad depth itself.

**Not done:** refining door and window edges from the colour images, the second
planned step, was dropped to finish on time. Opening widths stay ±3–7 cm.

### real-interval-floor: real LiDAR intervals far too narrow (declared before the code, `HEAD` at declaration)

- **Symptom:** "confident garbage". On the held-out visit (422009) LiDAR wall intervals
  cover the truth on 29% of walls (median 20% per recording) against a 90% target, with
  walls up to 61 cm off; on dev, 76%.
- **Isolation:** the propagated wall sigmas are plane-fit numbers, millimetres, while real
  errors are centimetres to decimetres from walls placed in the wrong spot. A multiplier
  on a millimetre sigma cannot reach them: dev alone would need ×27 at the shipped table's
  form. The shipped table is also pooled with synthetic captures, whose errors are tiny,
  which pulls the multiplier down for real ones. Splitting by room size does not help
  (normal-sized dev rooms need wider intervals than small ones).
- **Fix:** each quantity gets an absolute floor added in quadrature, σ′ = √(σ² + τ²),
  before the split-conformal multiplier; τ is chosen per quantity from a small grid as the
  narrowest that keeps leave-one-visit-out coverage ≥ 90% on dev. The LiDAR table is
  fitted on real dev recordings only.
- **Predicted** (from the same recordings, before code): walls τ ≈ 3 cm, multiplier ≈ 4.3;
  dev leave-one-visit-out coverage ≈ 94%; held-out wall coverage 29% → ≈ 65%, median
  wall half-width ≈ 22 cm; ceilings τ ≈ 1 cm, held-out coverage ≈ 86% at ≈ 2.7 cm
  half-width. It will not reach 90% held out: one held-out wall's truth is 33 cm off (an
  object in front of the wall was surveyed as the wall; `fix_declaration.md` §4 has the
  method for checking it), and one held-out room's shape is wrong in our plan.

### Earlier fixes (in the history, found the same way)

These predate the regeneration script, so their before/after is recorded in the commit
messages rather than regenerated.

| Commit | Failing number | Cause | Fix |
|---|---|---|---|
| `d1dd3a0` | LiDAR openings 4/9: three doors 3 cm narrow | residual along-wall sliding after drift correction; wall planes can't pin it | door and window jambs as bundle-adjustment landmarks: 9/9 |
| `fcb425f` | LiDAR walls drifting tens of cm on multi-room walks | frame-to-model ICP accumulated as much drift as VIO | 4-DoF pose graph over 3 s fragments with ICP loop closures |
| `0da9d24` | video walls +12–25% | monocular focal estimate 12.6% long; metric depth scales with it | focal self-calibrated from the room turns (0.6% from truth) |
| `0da9d24` | video ATE 2.25 m (synthetic), 10–57 cm (real) | chaining a whole walk with windowed multi-view reconstruction | rooms from on-the-spot turns: no long trajectory to chain |
| `74a9670` | video turn views 15% consistent | MoGe-2 per-view scale 0.46–1.55 on synthetic frames | per-view scale solved from overlaps (Cauchy IRLS), distorted views dropped |
| `b70069d` | video walls 4–7 cm in two rooms | the first frame of a turn is mid-step, off the turning axis | trim turn ends; rotation averaging with SIFT loop closure |
| `012db77` | bathroom turn missed | homography yaw 1–2°/step on white walls during an 8°/step turn | rotation-only yaw, forward-backward track check |
| `2d74357` | LiDAR seeds 0/1: hallway merged with bedroom or bathroom | wall directions were 2° bin centres (a degree off moves a 5 m trace 9 cm, so it stopped short); walls never seen under the ceiling had no barrier | refined directions; traces extended along their own wall at any height; roof mask with hole filling |
| `dd89e05`, `b31c18c` | photo stitch 1 of 3 links, openings 4/10 (true depth) | doorway shots didn't register; side doors seen too obliquely | threshold photo pairs join rooms through a shared standpoint; doors carried through the wall: 3/3 links, 9/9 openings |
| `1e1b338` | scanned room0 one side 15.6 cm short (true depth) | nearest plane was a full-length sill bench | a wall must reach the ceiling: 0.47 cm |
| `a3d59a8` | LiDAR, a fourth capture: adjacency wrong | hallway side of the bedroom door undetected (seen obliquely) | unpaired door carried through its wall when a room's wall faces it: 9/9, adjacency correct |
| `b9ca542` | undamaged scanned rooms: 10–19 false damage regions per video room | short ridges, structure lines, floor shading; video too coarse for cracks | report confidence ≥ 0.6, cracks ≤ 2.5 m, no floor analysis, no cracks from video |
