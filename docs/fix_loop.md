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
