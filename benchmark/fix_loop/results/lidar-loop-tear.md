# lidar-loop-tear: Wrong loop closures tore the pose graph at the end of a capture

Symptom: LiDAR, flat_a seed 1 (seed 0 passed): openings 5/10, adjacency wrong, walls up to 2.6 cm, and a phantom strip of hallway outside the flat. The last 4 s of corrected poses were up to 55 cm from truth.

Cause: Seven ICP loop closures from the final fragments back to the start had slid along the hallway. The Cauchy loss was applied to every edge, so the solver switched off the one odometry edge (73 -> 74) that disagreed with them by 0.6 m and tore the graph there.

Fix: robust weights (IRLS) on loop closures only; odometry is never robustified (`d53e04a` -> `7633a27`)

| metric | before | after |
|---|---|---|
| wall median (cm) | 1.1 | 1.47 |
| wall max (cm) | 2.6 | 2.14 |
| walls within 1 cm | 0.3333333333333333 | 0.125 |
| ceiling max (cm) | 1.08 | 13.84 |
| openings within 2 cm | 5/10 | 5/9 |
| adjacency correct | False | False |
| footprint error | 0.0468 | 0.039 |
| interval coverage | 0.68 | 0.7777777777777778 |
