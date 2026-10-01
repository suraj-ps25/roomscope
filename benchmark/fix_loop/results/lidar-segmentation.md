# lidar-segmentation: Rooms merged through gaps in the wall traces

Symptom: LiDAR, flat_a seed 1, after lidar-loop-tear had fixed its trajectory: adjacency still wrong and a 13.8 cm ceiling error; the hallway segmented together with the bathroom.

Cause: Wall directions were 2-degree histogram bin centres, so a trace a degree off fitted only part of a 5 m wall and stopped short; walls never seen just under the ceiling had no barrier at all; the observed-ceiling mask left holes where nobody looked up.

Fix: refined wall directions; traces extended along their own wall at any height; roof mask with hole filling (`b31c18c` -> `2d74357`)

| metric | before | after |
|---|---|---|
| wall median (cm) | 1.52 | 1.37 |
| wall max (cm) | 2.16 | 2.04 |
| walls within 1 cm | 0.125 | 0.25 |
| ceiling max (cm) | 13.84 | 1.16 |
| openings within 2 cm | 6/9 | 9/9 |
| adjacency correct | False | True |
| footprint error | 0.0126 | 0.0075 |
| interval coverage | 0.8823529411764706 | 0.8275862068965517 |
