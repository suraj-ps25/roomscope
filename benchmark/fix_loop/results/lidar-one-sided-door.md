# lidar-one-sided-door: A door seen from one side only broke adjacency

Symptom: LiDAR, a fourth independent capture of the flat: the hallway side of the bedroom door went undetected while the bedroom side measured it (0.851 m), so adjacency failed.

Cause: Adjacency needed the door detected on both faces of the wall; the hallway saw that doorway only obliquely and briefly.

Fix: an unpaired door is carried through its wall when another room's wall faces it a wall-thickness away (`9805f2f` -> `a3d59a8`)

| metric | before | after |
|---|---|---|
| wall median (cm) | 0.13 | 0.13 |
| wall max (cm) | 0.52 | 0.52 |
| walls within 1 cm | 1.0 | 1.0 |
| ceiling max (cm) | 0.15 | 0.15 |
| openings within 2 cm | 8/9 | 9/9 |
| adjacency correct | False | True |
| footprint error | -0.0001 | -0.0001 |
| interval coverage | 1.0 | 1.0 |
