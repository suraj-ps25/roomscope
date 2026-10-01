# Architecture

The design and the reason behind each decision. The measured results behind these
choices are in the commit history and `docs/benchmark_report.md`.

## Contract first

Every tier produces the same `plan.json` (`schema/floorplan.schema.json`). Every
dimension is a `measurement` `{value, ci_low, ci_high, ci_level, method}`, so "a
confidence interval on every measurement" is enforced by the schema, and `schema.py` also
checks `ci_low ≤ value ≤ ci_high`. `build.py` is the one place geometry becomes
measurements, for all tiers.

## Tiers share one core

| Tier | What produces depth and poses | Then |
|---|---|---|
| LiDAR | ARKit depth + VIO poses (Stray Scanner) | shared core |
| Video | per room turn: rotations from the clip itself, MoGe-2 metric depth per view, no pose solve | per-room layout, then doorway stitching with known headings |
| Photo | per room: MoGe-2 + MapAnything on 2–8 stills | per-room layout, then doorway stitching |

The photo and video tiers turn images into **pseudo-LiDAR frames** (metric depth, pose,
intrinsics, confidence). Rooms, walls, openings and damage are then the same code as
LiDAR, with noise-appropriate tolerances (`geometry/tolerances.py`,
`DriftSettings`, `PlaneSettings`).

**Why MoGe-2 for metric scale.** Measured on real iPad views with laser ground truth
(ARKitScenes 47429912):

| Model | Scale vs truth | Per-view spread |
|---|---|---|
| MapAnything (images only) | 0.669 | 16.8% |
| MoGe-2 | −0.5% | 6.9% |
| MapAnything conditioned on MoGe-2 depth | 0.978 | 5.9% |

## Video: rooms from on-the-spot turns

Chaining a whole walkthrough with multi-view reconstruction failed in measurement:
MapAnything over overlapping keyframe windows gave 10–57 cm ATE on real ARKitScenes video
and 2.25 m on the synthetic walk. A room turn is a much better-posed problem, so the
protocol asks for one per room (portrait, one revolution tilted down, one tilted up) and
the pipeline (`io/video.py`, `tiers/video.py`, `tiers/photo.py::_panorama_room`) does:

1. **Find the turns.** Per frame, a rotation-only fit gives the yaw step; a K-free
   homography residual over forward-backward-checked tracks says whether the motion was a
   pure rotation. A turn is a steady same-direction run of ≥ 280°.
2. **Calibrate the lens from the turns.** For a pure rotation H = K R K⁻¹, so only the
   true K makes every K⁻¹HK orthonormal (Hartley). On the synthetic clip: 0.6% from truth,
   where the monocular model's estimate was 18% off. The decode's non-square pixel aspect
   is kept (ignoring it made every height 1% short).
3. **Rotations for every view.** 2-point Kabsch RANSAC on bearings frame to frame, then
   SIFT loop closure across 360° and robust chordal rotation averaging over each frame's
   next four (worst view 4° → 1.3°). The first and last 0.6 s of a turn are dropped: the
   walker is still stepping.
4. **A panorama, not a pose solve.** All views share a centre; the arm (phone to turning
   axis) is the one under which the views agree best. Each view keeps its own MoGe-2
   depth, and per-view scale is made consistent by Cauchy IRLS over the overlap depth
   ratios; a view whose ratios disagree with any single scale (looking through a window,
   into a mirror) is dropped.
5. **Gravity** from wall normals (perpendicular to up) and floor/ceiling normals
   (parallel), with only a weak pull to the cameras' mean up: in a corridor the walls fix
   one horizontal axis and the prior breaks the tie.
6. **Room headings for stitching.** A visual compass over the whole clip (tracked steps
   snapped to the line segments' Manhattan axes) gives every room's heading to well under
   45°, so the relative rotation between rooms snaps to the exact multiple of 90° and a
   doorway candidate with the wrong rotation is never tried.

With depth from the renderer instead of MoGe-2 (`--oracle-depth`), the synthetic video
tier passes every gate (walls ≤ 0.4 cm, ceilings ≤ 1.1 cm, 9/9 openings, adjacency,
footprint 0.0%): what remains in a real run is MoGe-2's error, which the intervals carry.

## Drift

The phone's VIO is excellent over seconds and drifts over tens of metres. Depth
registration is the reverse: exact between two views, but chaining registrations
random-walks. An experiment kept in the history showed that frame-to-model ICP tracking
accumulated as much drift as VIO. So:

1. **Pose graph** (`geometry/drift.py`) over 3-second fragments.
   - Odometry edges carry a stated VIO budget: 1%/m + 2 mm, and 0.2°/√m heading. An
     earlier, overconfident budget made the robust loss reject the true loop closures.
   - Loop edges come from 4-DoF point-to-plane ICP between any fragments that overlap. A
     closure is accepted on its absolute inlier count and RMSE, not on overlap fraction,
     because the end of a scan and its start usually look in different directions.
   - Information comes from the ICP Hessian, so a corridor match is weak along the
     corridor.
   - Solved with a Cauchy loss on a graduated scale, so a large true loop isn't thrown
     away as an outlier.
2. **Plane-anchored bundle adjustment** (`geometry/planes.py`).
   - Landmarks are every room's wall, floor and ceiling planes, plus door and window
     **jambs**. Jambs are perpendicular to their wall, so they pin the along-wall sliding
     a wall plane can't: without them, residual hallway sliding made 3 of 9 doors 3 cm
     narrow.
   - Point blocks are deflated to 50 effective points, because depth error within one
     view is correlated.

The **capture protocol** is part of the drift design. A full on-the-spot turn in each room
sees opposite walls seconds apart, so room dimensions don't rest on long-term odometry;
the hallway's length was 7 cm off without it. Ending at the start gives the scan-level
loop closure.

## Rooms, walls, openings

- **Rooms** (`geometry/rooms.py`). Interior comes from free-space carving: each ray
  proves the cells it crossed are empty.
  - Barriers are long straight wall traces in the band just below the *local* ceiling.
    Door headers close doorways; shower rails and cabinets that stop short of the ceiling
    don't cut rooms.
  - Traces are extended to meet at corners.
  - A region the camera never stood in isn't a room: this rejects mirror phantoms and
    glimpses.
- **Walls** (`geometry/layout.py`). A robust TLS line per wall, seeded under the ceiling
  (above furniture). Angles are **fitted, not snapped**: snapping a 4 m wall by 0.5°
  moves its corners 1.7 cm. Edges with little wall material under the ceiling are
  dropped as mask artefacts, and consecutive parallel lines are merged.
- **Openings** (`geometry/openings.py`).
  - Per-wall ray votes (face / recess / through / no-return; rays stopped in front don't
    vote), so "unseen" is never read as "open".
  - Edges snap to the density mode of jamb/head reveal points, using high-confidence
    returns only (ARKit marks edge pixels low).
  - Doors are paired across a wall thickness (adjacency) and fused by inverse variance.
  - **Mirror test**: reflect the see-through points across the wall and check they land
    on the room, using only surfaces facing the wall. Floors and aligned perpendicular
    walls are invariant under that reflection.
- **Photo stitching** (`geometry/stitch.py`). Door pairs fix the relative pose; they are
  verified by whether one room's through-door glimpse lands on the other room; then
  assembled as a greedy maximum spanning tree with no door reuse and no overlap.

## Damage

`damage/`:
- Metric orthophotos per surface at 5 mm, taking each texel from its best non-occluded
  view.
- Detectors run in an order that stops each one's false positives from becoming another's:
  objects masked → mould (speck clusters) → cracks (meandering ridges) → stains
  (hysteresis over a 1.2 m background).
- Seven explicit concealed-damage rules cite their evidence.
- Scope lines are keyed to surfaces.
- Peeling paint is deliberately not reported (it's indistinguishable from lighting
  gradients).

## Intervals

`build.py` propagates each measurement's uncertainty:
- wall length from the neighbouring walls' fit sigmas through the corner intersections;
- ceiling from the floor and ceiling level fits;
- openings from the jamb fits;
- damage from the boundary;
- then the tier's scale sigma: LiDAR 0.3%, video ≥ 1.5%, photo per room
  √((6.9%/√n)² + 2%²).

Per-tier, per-quantity calibration multipliers (`calibration/<tier>.json`) are fitted on
ground truth so the stated 90% holds, and coverage is reported per tier.

## Known limitations

- An alcove the camera never enters behind a full-height partition (a bathtub niche) is
  left out of the room.
- Closed doors read as wall; the protocol asks for doors to be opened.
- Floor-to-ceiling glass leaves a gap in the wall trace, and corner completion bridges
  only up to 1 m.
- Damage extents read low (soft edges); only water stains, cracks and mould are reported.
- Photo/video accuracy rests on MoGe-2's metric prior; intervals say so.
