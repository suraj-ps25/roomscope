# roomscope: technical report

Handheld iPhone capture → a dimensioned, stitched floor plan with per-surface damage,
concealed-damage flags and a repair scope, with a confidence interval on every number.
Three capture tiers (photo, video, LiDAR), one output contract, one command per capture.
Results are in [`benchmark_report.md`](benchmark_report.md); requirement coverage in
[`compliance_matrix.md`](compliance_matrix.md); how failures became fixes in
[`fix_loop.md`](fix_loop.md).

## 1. Constraints that shaped the design

- **Route 2: stock apps and a protocol.** Stray Scanner (LiDAR: depth, ARKit poses,
  intrinsics) and the built-in Camera app (photo, video). No custom app: anyone with an
  iPhone can capture today. The protocol (`capture_protocol.md`) is part of the
  algorithm. Each tier asks for exactly the motion its method needs.
- **No iPhone was available to build this.** Development therefore leaned on four kinds
  of evidence, each used for what it can show: a synthetic flat with exact truth (every
  stage, every tier); real scanned rooms rendered as protocol captures (realistic imagery
  with exact metric truth); real iPad frames with LiDAR depth (the depth model's metric
  scale on real imagery); and **real iPad Pro LiDAR recordings of real rooms scored
  against laser scans** (Apple's ARKitScenes, §8), on dev visits and on held-out visits
  run once. The head-to-head and the walk-in on our own phone captures are scripted but
  not run.
- **Every number has an interval.** The schema requires `ci_low`/`ci_high` on every
  measurement; intervals are propagated, then calibrated per tier (§6).

## 2. One core, three ways in

All tiers end in the same geometry: rooms → wall lines → openings → stitch → damage →
`plan.json`. They differ in how they get metric depth and camera poses.

| Tier | Depth | Poses | Rooms joined by |
|---|---|---|---|
| LiDAR | ARKit depth | ARKit VIO, corrected for drift | one continuous scan |
| Video | MoGe-2 per view | from the clip itself (turns) | doorways, with headings from a visual compass |
| Photo | MoGe-2 per view | registration by matched 3D points | doorway threshold photo pairs |

## 3. LiDAR: drift is the problem

ARKit VIO is excellent over seconds and drifts over tens of metres; depth registration is
the reverse. The design uses each for what it is good at.

- **Pose graph** over 3 s fragments: odometry edges with a stated VIO budget (1%/m + 2 mm,
  0.2°/√m), loop edges from 4-DoF point-to-plane ICP between overlapping fragments, with
  information from the ICP Hessian, so a corridor match is weak along the corridor.
- **Robust loss on loop closures only.** The first version applied the Cauchy loss to
  every edge. On one capture, seven ICP closures that had slid along the hallway
  outvoted the one odometry edge they contradicted, the solver switched that edge off,
  and the graph tore: the last 4 s of poses were up to 55 cm out. IRLS with Cauchy
  weights on loop edges only fixed it (trajectory error median 5.1 → 1.1 cm on that
  capture, 4.5 → 0.7 cm on another).
- **Plane-anchored bundle adjustment.** Every room's walls, floor and ceiling are
  landmarks, and so are door and window **jambs**. Jambs are perpendicular to their wall,
  so they pin the along-wall sliding a wall plane can't: without them three of nine doors
  read 3 cm narrow.
- **Ablation.** Every LiDAR capture is benchmarked with drift correction on and off
  (`--no-drift-correction`); off, walls are off by up to ~1.8 m and adjacency fails.

**Rooms** come from free-space carving (each ray proves the cells it crossed are empty),
bounded by wall traces just under the local ceiling. Door headers close doorways, and
furniture that stops short of the ceiling doesn't cut rooms. Three faults found on repeat
captures are fixed:
- Wall directions were 2° histogram bin centres. A degree off over a 5 m wall moved the
  trace 9 cm, so it fitted part of the wall, stopped short, and the hallway leaked into
  the bedroom.
- Under-ceiling traces now continue along their own line wherever the wall continues at
  any height. A wardrobe front never lies on that line.
- A room cell must have observed ceiling over it, so space carved out through an
  exterior door isn't room; holes nobody looked up at are filled.

**Openings**: per-wall ray votes (face / recess / through / no return), so "unseen" is
never read as "open". Edges snap to jamb reveal surfaces. A mirror is told from a doorway
by reflecting its see-through points back across the wall and checking they land on the
room.

## 4. Video: rooms from on-the-spot turns

Chaining a whole walkthrough with multi-view reconstruction failed in measurement:
2.25 m trajectory error on the synthetic walk, 10–57 cm on real ARKitScenes video. A
room turn is a much better-posed problem, so the protocol asks for one per room: in the
middle of the room, phone upright, one revolution tilted down and one tilted up. The
double turn exists because each heading is seen at one tilt per revolution, and
landscape 1× can't see floor, walls and ceiling from one tilt.

1. **Find the turns**: rotation-only yaw per frame; a K-free homography residual over
   forward-backward-checked tracks says the motion was a pure rotation.
2. **Calibrate the lens from the turns.** For pure rotation H = K R K⁻¹, so only the
   true K makes every K⁻¹HK orthonormal. The result is 0.6% from truth; the monocular
   estimate was 12–18% off, and metric depth scales with it. The decode's non-square
   pixel aspect is kept; ignoring it made every height 1% short.
3. **Rotations**: 2-point Kabsch RANSAC on bearings, a guard against steps far off the
   turn's steady rate (a blank wall reads as "no rotation"), SIFT loop closure across
   360°, robust chordal averaging. Worst view 1–4° on the scanned rooms.
4. **A panorama, not a pose solve.** All views share a centre; per-view MoGe-2 scale is
   made consistent by Cauchy IRLS over overlap depth ratios, and views whose ratios fit
   no single scale (looking through glass, into a mirror) are dropped. Gravity comes
   from wall normals (perpendicular to up) and level-surface normals (parallel), with
   only a weak camera prior that breaks the corridor tie.
5. **Headings for stitching**: a visual compass over the whole clip (tracked steps
   snapped to line segments' Manhattan axes). This fixes each room's rotation exactly, to
   a multiple of 90°, so a doorway candidate with the wrong rotation is never tried.

## 5. Photo: registration by matched 3D points, rooms joined through thresholds

MapAnything's sparse-view poses were the photo tier's limiting error (rooms off by up to
3.5 m even with true depth; 19% view agreement on real stills). Each photo already
carries metric depth, so a SIFT match is a 3D point in both views:
- 3-point RANSAC similarities between pairs;
- a maximum spanning tree to start;
- a joint Cauchy least squares over every view's rotation, translation and depth scale.

Residuals are divided by the pair's scale, and the scale gauge is applied exactly
afterwards; with a soft gauge, the solver shrank whole rooms by 1.8% to lower its
residuals.

**Stitching.** At every doorway the protocol takes one photo into each room from the
threshold, back to back. Each registers with its room like a corner shot. The two share
a camera centre, paired by EXIF capture time, which places one room relative to the
other. The doorway wall's inward normals, opposite in the two rooms, fix the rotation
exactly. The threshold also says *where* a door is: an opening found there is a door,
and where one side measured the door and the other saw it too obliquely, the opening is
carried through the wall.

**Sparse-view layout.** Each side's wall is the nearest substantial vertical plane beyond
the cameras that reaches the ceiling. Substantial means real support and spanning much
of the room, so furniture fronts fail it. Reaching the ceiling is the test that rejects a
full-length sill bench 15 cm in front of a window wall. Glimpses through doorways and
"sky" planes behind windows are farther out, so nearest-first ignores them. L-shaped
rooms come out as their bounding rectangle, a stated limitation.

## 6. Uncertainty

`build.py` propagates each measurement's uncertainty:
- wall length from the neighbouring walls' fit sigmas through the corner intersections;
- ceiling from the floor and ceiling level fits;
- openings from the jamb fits;
- then the tier's scale budget.

Split-conformal multipliers per tier and quantity are fitted on ground truth, with
leave-one-capture-out coverage reported.

The photo and video scale budget rests on a measured fact: **MoGe-2's metric scale error
is a per-scene bias**: −14% to +4% across the ten scenes measured, 5.9% spread (sd of
log). Views of one room share it, so averaging views does not remove it. That bias is what
stands between the video tier and its ±3% wall gate. On the eight scanned rooms the video
tier's worst wall is a median 11.2% off with the model, and 1.9% with rendered true depth
in its place. Calibrated intervals carry the bias: walls need a ×8.3 multiplier on the
propagated sigma at the video tier, with 88% held-out coverage.

## 7. Damage

Metric orthophotos per surface at 5 mm, each texel taken from its best non-occluded
view. Views are chosen greedily to cover the surface (a 15 cm lattice), and each view's
colour gain is solved from overlaps, so an exposure step between views is not a stain.
Dense geometry in front of a surface (shelves, lamps, furniture) or in a recess is masked,
and a crack may not run along a mask's edge. Detectors run in an order that keeps each one's false positives from becoming
another's: objects masked → mould → cracks → stains. Seven explicit concealed-damage rules
fire with their evidence: stain under a window sill, moisture at the base of a wall,
ceiling moisture, a diagonal crack from an opening corner, a large stain, mould, and
moisture on a wall shared with a wet room. Scope lines are keyed to surfaces. Peeling
paint is deliberately not reported: it is indistinguishable from lighting gradients.

## 8. Real captures against laser truth

ARKitScenes ships iPad Pro recordings (LiDAR depth, ARKit poses, colour) with Faro laser
scans of the same rooms. `benchmark/real/laser_truth.py` surveys a scan as the protocol
asks a person to: the plan names the walls; the scan gives their faces, corners,
room-wide floor-to-ceiling height and openings (pane widths, mullions split, laser
shadows and recesses rejected, mirrors identified by reflecting the see-through points).
Every survey was checked against its overlay (`docs/real/`). Two visits were **dev**
(bugs found and fixed, LiDAR intervals fitted); three were **held out**, chosen by rule
before being looked at and run once (one runtime-only change while running them, §10).

- **Found on dev:** a 10 cm ceiling step was traced as a wall and cut a bathroom in two
  (1.94 m for 2.72 m); undamaged rooms reported up to 59 damage regions; and the damage
  scorer was wrong: the synthetic "crack found" was a door-jamb edge on another wall.
- **What the iPad's depth itself carries:** registered to the laser, the recordings' own
  clouds are ~1% small and ~1° out of square on a room the laser says is square to 0.2°.
  Walls come out 1–2.5% short whatever the layout does, so LiDAR intervals are now
  calibrated on real recordings (walls ×4.2), not on the simulator.
- **Held-out:** in the one large room with laser truth (≈18 m²), walls within a median
  2 cm, worst 4.5–6.5 cm; ceilings 1.2–2.5 cm low. Small irregular rooms (a 5-wall
  bathroom) score 7–25 cm, partly from phantom sliver walls that break the wall
  sequence; openings mostly miss or are phantoms (2 of 30 within 2 cm); 5–15 false damage
  regions per recording; all three recordings of one visit (27–35 s each) never closed a
  room. Interval coverage on held-out visits is a median 33% against a 90% target.

The full table is in [`benchmark_report.md`](benchmark_report.md).

## 9. How it was built: the fix loop

Every fix began as a failing number, was isolated by an ablation or a stage-level
diagnostic against truth, landed as one commit with its before/after in the message, and
was followed by a re-run of every tier the change touched. One such re-run caught a fix
for the video tier regressing LiDAR openings (8/9 → 7/9); its cause, a margin defined as
a share of a per-tier tolerance, was fixed the same day. Ten entries regenerate from
their two commits on an identical capture with one scorer (`benchmark/fix_loop/`).

## 10. Limitations and next steps

- **Real LiDAR accuracy.** On held-out real rooms the LiDAR tier misses the brief's
  gates: ceilings 1–4 cm, openings poor, intervals overconfident (33% coverage). Next:
  openings on real depth (glass gives no "no return" in densified depth, so the vote
  model needs a new cue), sliver walls in small rooms, and the depth's own skew.
- **Long recordings** were slow: a 7-minute recording sat 20+ minutes in the pose-graph
  solve until its residuals were vectorised (found while running the held-out visits;
  results moved by millimetres).
- **Our own captures.** Head-to-head and walk-in on iPhone captures are scripted but unrun;
  the photo and video tiers have no real capture with laser truth.
- **Monocular scale.** The video ±3% gate is not met. The next lever is a door-height
  prior (interior doors are ~2.03 m, σ ≈ 2%). It wasn't validated here, because the
  scanned rooms' doors are offices' and were rarely detected.
- **Photo/video layouts are rectangles.** L-shaped rooms and rooms with a deep jog read as
  their bounding rectangle.
- **LiDAR repeatability** is limited by the sensor's per-capture depth scale bias
  (simulated at 0.2%): two captures of one wall can differ by more than 0.5%.
- **Damage** recall is validated on synthetic staging only (3/3 at the staged places);
  on real undamaged rooms it still reports 2–15 false regions per recording.
