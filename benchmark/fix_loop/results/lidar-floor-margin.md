# lidar-floor-margin: The door fix regressed LiDAR openings

Symptom: Re-running the LiDAR tier after video-door-sill: openings 8/9 -> 7/9, the hallway's bedroom door missed and adjacency wrong. The pre-fix commit on the same capture passed.

Cause: The tightened floor-beyond margin was a share (0.4) of the tier's wall-face tolerance: 4.8 cm for photos but 1.6 cm for LiDAR, inside depth noise at the wall base.

Fix: explicit per-tier floor_beyond_m; LiDAR keeps 4 cm (`470c532` -> `822fab3`)

| metric | before | after |
|---|---|---|
| wall median (cm) | 0.25 | 0.25 |
| wall max (cm) | 1.79 | 1.79 |
| walls within 1 cm | 0.875 | 0.875 |
| ceiling max (cm) | 0.29 | 0.29 |
| openings within 2 cm | 7/9 | 8/9 |
| adjacency correct | False | True |
| footprint error | 0.0002 | 0.0002 |
| interval coverage | 0.9642857142857143 | 0.9655172413793104 |
