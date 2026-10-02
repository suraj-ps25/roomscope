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
  run once. The head-to-head was run on a recording's raw cloud (§8); the walk-in on our
  own phone captures is scripted but not run.
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

## 6. Error budget and calibration

| tier | dominant error | measured size | how it is handled | wall multiplier, held-out coverage |
|---|---|---|---|---|
| LiDAR | depth scale of the device | −0.88% (iPad Pro) | per-device scale ×1.0088 (§8) | 8 cm floor, ×1.63; 94% dev (left-out), 60% held-out |
| LiDAR | VIO drift | 1%/m + 2 mm budget | pose graph + plane-anchored adjustment (§3) | (in the above) |
| LiDAR | jamb edges at 256×192 | ±3–7 cm on openings | density-mode reveal fit | openings ×2.04 |
| video | depth model's per-scene scale | sd 5.9%, −14% to +4% | carried in the interval, not removable by averaging | ×8.34; 88% |
| video | geometry alone (true depth) | worst wall 1.9% median | turns + multi-view | (in the above) |
| photo | depth model's per-scene scale | as video | carried in the interval; views joined by matched 3D points (§5) | ×2.11; 92% |
| photo, ≤ 3 photos a room | wall placement, not scale | walls median 39 cm off | separate conformal table keyed on `quality.views` | ×4.77; 91% |

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
room-wide floor-to-ceiling height and openings (doors face to face, windows inside the
frame, laser shadows and recesses rejected, mirrors identified by reflecting the
see-through points). Every survey was checked against its overlay (`docs/real/`). Two
visits were **dev** from the start: bugs found and fixed, LiDAR intervals and depth scale
fitted. Three were **held out**, chosen by rule before being looked at; the last fix (§9,
sliver walls) was developed on them, so they are dev now, and a fresh visit chosen by the
same rule is the held-out set.

- **Found on dev:**
  - a 10 cm ceiling step was traced as a wall and cut a bathroom in two (1.94 m for 2.72 m);
  - undamaged rooms reported up to 59 damage regions;
  - the damage scorer was wrong: the synthetic "crack found" was a door-jamb edge on
    another wall.
- **The iPad's depth reads 0.88% short.** Registered on the laser by similarity ICP, every
  well-registered dev recording needs a scale of 1.0087–1.0106, and every real ceiling was
  low. The device calibration that corrects it (§9) took the then held-out ceilings within
  1.5 cm from 4 to 8 of 10. It is applied to that device only: a phone never measured is not
  corrected.
- **Dev, including the formerly held-out visits:**
  - the large room (≈18 m²): walls a median 2.4–3.6 cm off, worst 5.4 cm; ceilings within
    0.7 cm; the bathroom: 2.1–2.9 cm, worst 5.7 cm;
  - small rooms: worst walls 16–25 cm;
  - ceilings within 1.5 cm on 12 of 14; openings 7 of 23 within 2 cm;
  - 2–11 false damage regions per recording;
  - one visit's three recordings (27–35 s each) never closed a room.
- **Held-out (visit 422009, run once):** two very small rooms (about 2 m², walls 1.0–2.3 m)
  whose laser survey is thin (one scanner station; the plan placed with 56% of wall
  samples within 5 cm). Ceilings hold: 6 of 7 within 1.5 cm. Walls do not: each
  recording's worst wall a median 38 cm off, all 8 reported openings phantom, interval
  coverage 20%, raised to 60% by an absolute error floor in the calibration (§9). Recordings of one room disagree on its wall count, so short walls and
  cut corners in rooms this small are what the pipeline gets wrong; how much of the 38 cm
  is the thin survey is not separated.
- **Photo and video on the same rooms.** Fed the recordings' own colour frames, the image
  tiers are far off: worst walls 33–94 cm. These walk-arounds, mostly tilted up, are not
  the protocol's corner shots and turns, so this is an off-protocol test. The intervals
  were wide enough to cover the truth on most.
- **The assessors' sample captures** (Stray Scanner, no truth) exposed two assumptions:
  - Two of three were filmed with the phone held level, and segmentation needed the
    ceiling: one gave no room. Walls are now traced above furniture height and doorway
    gaps closed when no ceiling was seen, and the ceiling is then a stated prior with a
    wide interval. The flat comes out as 8 rooms either way.
  - Decoding the video took 39 of 43 minutes on a 3.5-minute scan. One sequential decode
    makes the whole run 5.4 minutes, cold.
- **Head-to-head without a phone.** Consumer scanning apps only take a live scan, so two
  rooms' raw LiDAR (iPad poses, no correction of ours) went as point clouds into
  Pointorama's automatic room tools, unedited. Against the laser we beat or tie 14 of 15
  quantities as shipped. Its outlines step out past walls (13 edges for a 4-wall room),
  inflating areas 8–14% and finding no openings, but a wall it draws cleanly is as good as
  ours (0.8 cm). With our depth calibration off, on identical input, we beat or tie 10 of
  15, under the 70% bar (`benchmark/head_to_head/`).

The full tables are in [`benchmark_report.md`](benchmark_report.md).

## 9. How it was built: the fix loop

Every fix began as a failing number and was isolated by an ablation or a stage-level
diagnostic against truth. It landed as one commit with its before and after, and was
followed by a re-run of every tier it touched. That re-run caught three regressions: a
video fix that cost LiDAR openings, a square-room prior that cost a synthetic window, and
the declared fix's first version, which lost a door.

A later declared fix put an absolute floor under every interval (σ′ = √(σ² + τ²)) and fitted
real LiDAR on real dev recordings only: held-out coverage 20% → 60%, dev 94%, at a median
wall half-width of 21.5 cm (predicted 65% held-out; the miss is explained in `fix_loop.md`).

The fix the brief scores is **declared in advance**
([`fix_declaration.md`](fix_declaration.md)).
- **The worst gate:** real openings, 2 of 29 within 2 cm.
- **Two evidenced causes:** multi-sash windows split into several openings, and a
  floor-level sill classed as a window.
- **The prediction:** no new passes; detection failures 18 → 14.
- **The result:** every predicted number was met, after one correction (the plane
  adjustment uses openings as landmarks, and the fix must not change those).
- **The gate still fails:** widths stay ±5 cm at 256×192 depth, and some doors were never
  filmed.

A second fix, the depth-scale calibration, did better than expected (above). Each shipped
entry regenerates from their two commits on an identical capture with one scorer
(`benchmark/fix_loop/`).

## 10. Limitations and next steps

- **Real openings** stay far from the 85% gate (7 of 23 on dev, 0 of 8 held-out). Next: jamb
  edges from the colour image, not 256×192 depth.
- **Segmentation sits on thresholds** where bulkheads, ceiling steps and headers look
  alike. A dev room split at a bulkhead after the depth-scale change. Small rooms sprout
  sliver walls.
- **Our own captures.** The head-to-head ran on an uploaded cloud in a professional tool,
  not a consumer app's own scan on the same phone. The photo and video tiers have no
  on-protocol real capture with laser truth.
- **Monocular scale.** The video ±3% gate is not met. A door-height prior, a ceiling-height
  prior and an ensemble of depth models were each measured and not shipped
  (`fix_loop.md`): the depth model's scale varies by 6.7% from scene to scene on real
  iPad frames, and no global correction removes a per-scene spread.
- **Photo/video layouts are rectangles.** L-shaped rooms read as their bounding rectangle.
- **Thin photo input** (2–3 photos a room) is honest, not accurate: rooms come out
  unconnected and walls tens of centimetres off; such rooms carry their own, 2.3× wider
  interval table (91% left-out coverage).
- **Low light** (35% exposure, 3× noise, simulated): LiDAR geometry is unchanged but damage
  recall drops 3/3 → 0/3; video keeps one room of four; photo degrades to 27% walls with
  52% coverage, the one place intervals still overstate confidence.
- **Damage** recall is validated on synthetic staging only (3/3 at the staged places).
  Real undamaged rooms still report 2–11 false regions.
