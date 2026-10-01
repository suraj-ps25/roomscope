# photo-depth-cap: Far walls vanished from photo reconstructions

Symptom: Photo tier with true depth, Replica room0: 3.27 x 3.90 m for a 4.58 x 7.71 m room, with "no wall seen" on two sides. The synthetic hallway came out 3.1 m long for 5.0.

Cause: Every depth map was cut at 4 m, the LiDAR's useful range, for every tier. A corner photo of a long room is mostly far wall.

Fix: fuse takes a depth limit; the image tiers use 12 m (`bdb7174` -> `a1a7bf5`)

| metric | before | after |
|---|---|---|
| wall median (cm) | 278.19 | 3.95 |
| wall max (cm) | 380.49 | 7.38 |
| walls within 1 cm | 0.0 | 0.5 |
| ceiling max (cm) | 1.07 | 1.2 |
| openings within 2 cm | 0/3 | 0/0 |
| adjacency correct | True | True |
| footprint error | -0.6881 | 0.0154 |
| interval coverage | 0.2 | 1.0 |
