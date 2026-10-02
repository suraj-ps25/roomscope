# Fix declaration

Written and committed before the fix (`7e5e7be`). **Outcome:** every predicted number
was met (phantoms 12 → 9, misses 6 → 5, scored 29 → 26, passes 2), after one correction
caught on the way. The gate still fails, for the reasons set aside in section 2. The full
before and after are in [`fix_loop.md`](fix_loop.md); per recording they regenerate with
`python benchmark/fix_loop/regenerate.py real-openings-<recording>`, and across every real
recording with `python benchmark/fix_loop/regenerate_real.py 7e5e7be 1b84c55`. The before
and after runs are in `benchmark/fix_loop/results/real/`.

## 1. The worst gate

**Opening widths on real captures, LiDAR tier: 2 of 29 within 2 cm (7%; gate 85%).**
That is the iPad Pro recordings of real rooms scored against laser truth: dev and
held-out visits, commit `537e1de`, `runs/bench/real/summary_real.json`. Missed and
phantom openings count as misses, as the brief requires. No other gate in the benchmark
is as far from passing.

| outcome | count |
|---|---|
| width within 2 cm | 2 |
| matched, width off by more than 2 cm | 9 |
| missed | 6 |
| phantom | 12 |

## 2. Root cause: hypothesis and evidence

There is more than one cause; they are separated here by evidence.

**(a) A window with several sashes is reported as several openings, and a sash that
can be seen through to the floor is reported as a door.** Each frame member between
sashes breaks the run of "open" votes in the wall's evidence grid, and each run becomes
an opening; where a sash's lower part sees through to the floor, the run touches the
floor and is classed as a door.
- 47429922 (dev, living room): one 1.36 m two-sash window is reported as windows of 0.60
  and 0.66 m, 0.16 m apart. That is one phantom and one width error of −70 cm.
- 42897688 (held-out): one 2 m four-sash window is reported as a 1.09 m window plus a
  0.90 m "door" next to it, with no gap. That is two phantoms.
- 42897678 (held-out): the same window is reported as 0.92 + 0.98 m, 0.06 m apart. The
  laser can't measure it either (a curtain), so it is unscored.

**(b) A door whose bottom was seen only at its edge is classed as a window.** The class
is decided on the vote region (bottom at 0.10 m, the edge of the "touches the floor"
test), before the edges are snapped to the reveal surfaces. The snapped sill is 0.085 m:
floor level.
- 47429912 (dev, bathroom): the 0.80 m door is reported as a 0.82 m "window" with a sill
  of 0.085 m. That is one phantom and one miss.

**What the fix does not address, and why the gate will still fail:**
- **Door never filmed (3 misses).** No recording of the bathroom looked at the door
  above 0.75 m.
- **Widths ±5 cm on matched openings (errors +6, +7, +5, −3, −5, −6, +5 cm).** Errors of
  both signs, so not a bias. The likely limit is LiDAR depth at 256×192: about 1 cm per
  pixel at 2 m, with densified, blurred edges at the jambs.
- **Openings the scorer can't place.** When a plan's wall count differs from the
  survey's, its openings can't be mapped to the truth's walls. That includes the unscored
  openings, the ones the laser couldn't measure.

## 3. The fix and the predicted number

**Fix:**
- Openings on one wall separated by at most 0.2 m, and overlapping in height, are one
  opening, measured outer edge to outer edge. If any part of it has a sill, it is a window.
- After edge refinement, an opening whose sill is within 0.10 m of the floor is a door.

**Predicted result, scored by the same evaluator before and after:**

| | count |
|---|---|
| within 2 cm | 2 (unchanged) |
| phantoms | 12 → 9 |
| misses | 6 → 5 |
| scored openings | 29 → 26 |
| gate share | 7% → 8% |

Detection failures (misses plus phantoms) go from 18 to 14. No new passes are expected:
- the bathroom door becomes matched at about +2 cm, right at the gate;
- the merged living-room window measures about 1.42 m against 1.36 m, still about +6 cm.

The gate stays failed. The remaining failures are the three causes above that the fix
does not address.

**Regression guard:** synthetic flat openings stay 9/9 at every tier with true depth, and
the LiDAR seeds stay 7/9, 9/9, 9/9 (no two openings there are within 0.2 m of each other).

## 4. Post-mortem: where the 24 remaining failures come from

Added after the fix shipped. Every failure in the after run (`1b84c55`, 26 scored, 2
within 2 cm) is attributed, from the per-opening detail the evaluator writes
(`benchmark/fix_loop/results/real/1b84c55/`):

| cause | failures | where |
|---|---|---|
| width off, **a repeatable bias on one door** | 3 | the 0.72 m door of visit 423441 reads +5.4, +6.2 and +6.7 cm in three recordings |
| width off, no common sign | 5 | windows −3, −5, −6, +5, +6 cm |
| width off by under 0.5 cm past the gate | 2 | doors +2.2 and +2.4 cm |
| plan's wall count differs from the survey's, so openings can't be placed | 3 phantoms | 42897688 (8 and 17 walls against 6 and 4) |
| door on the wrong wall after alignment | 1 miss + 1 phantom | 44358259 (a 0.825 m door for the 0.79 m one) |
| door split into pieces | 2 phantoms | 44358259 (0.58 and 0.50 m) |
| openings where the survey has none it could measure | 3 phantoms | small room of 423441 (2), large room of 42897678 (1) |
| door never filmed above 0.75 m | 2 misses | bathroom, 47429913 and 47429914 |
| window not detected | 1 miss | bathroom, 47429913 |
| door not detected | 1 miss | 44358256 |

**What the declaration got wrong.** Section 2 called the width errors "both signs, so not
a bias" and put them down to depth resolution. That holds for windows only. The 0.72 m
door is 5–7 cm wide in every recording of it, so for that door the error is systematic,
and 256×192 depth noise would not repeat to within 1.3 cm. A likely cause, not yet
tested: our edges land on the door's frame (trim) while the laser survey measures
between the jamb faces. If so, the fix is a definition (which surface is the jamb), not
better depth, and it would pass those 3 at once.

**Why the movement was small.** The fix addressed 4 of the 18 detection failures, as
predicted, and none of the width failures, as predicted. Of what is left, 10 are widths
and 14 are detection or placement, spread over seven separate causes; no single fix
reaches more than 3. Getting from 8% to the 85% gate needs the width bias, the
wall-sequence mismatches and recordings that film every door, together.

**Measured since, not part of this declaration.** Later fixes moved other gates (real
ceilings 4/10 → 8/10 held-out; repeatability pairs comparable 7/12 → 11/12) but left
openings within 2 cm unchanged (`docs/fix_loop.md`).
