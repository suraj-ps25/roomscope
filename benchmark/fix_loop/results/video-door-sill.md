# video-door-sill: Oblique doorways read as windows

Symptom: Video tier, flat_a with oracle depth: 4 of the 5 doors seen from the hallway and bathroom turns came out as windows with a 0.23 m sill, so they could not pair and the plan did not stitch.

Cause: In the opening evidence, a ray landing on the floor just beyond the wall plane voted both "wall face" and "through": the two classes overlapped. From the middle of a room the bottom of a side doorway is seen at 60-70 degrees, where every such ray lands within the face tolerance.

Fix: face and floor-beyond classes made exclusive; floor-beyond margin 0.4 x face tolerance (`8d57a7d` -> `bf0c5d5`)

| metric | before | after |
|---|---|---|
| wall median (cm) | 0.3 | 0.3 |
| wall max (cm) | 1.79 | 1.79 |
| walls within 1 cm | 0.875 | 0.875 |
| ceiling max (cm) | 3.81 | 3.81 |
| openings within 2 cm | 4/13 | 7/9 |
| adjacency correct | False | True |
| footprint error | -0.002 | -0.002 |
| interval coverage | 1.0 | 1.0 |
