# real-openings-47429912: A floor-level door read as a window (declared fix, cause b)

Symptom: Real bathroom recording, laser truth: the 0.80 m door reported as a 0.82 m window with a 0.085 m sill: one phantom, one miss.

Cause: the door/window class was decided on the vote region, before the sill was snapped to the reveal

Fix: after refinement a window with its sill within 0.10 m of the floor is a door (plan openings only, not adjustment landmarks) (`7e5e7be` -> `1b84c55`)

| metric | before | after |
|---|---|---|
| wall median (cm) | 2.74 | 2.74 |
| wall max (cm) | 6.52 | 6.52 |
| walls within 1 cm | 0.5 | 0.5 |
| ceiling max (cm) | 2.59 | 2.59 |
| openings within 2 cm | 0/3 | 0/2 |
| adjacency correct | True | True |
| footprint error | -0.0223 | -0.0223 |
| interval coverage | 0.8333333333333334 | 0.7142857142857143 |
| damage found | 0/0 | 0/0 |
| damage false positives | 2 | 2 |
