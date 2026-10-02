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
