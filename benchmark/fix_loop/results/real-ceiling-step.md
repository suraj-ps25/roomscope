# real-ceiling-step: A ceiling step cut a real room in two

Symptom: Real iPad recording (ARKitScenes 47429912) of a bathroom, against laser truth: the room came out 1.94 x 1.85 m for 2.72 x 1.98 m; one end of the room was missing.

Cause: The ceiling is lowered by 10 cm over one end. The vertical face of that step sat in the band under the ceiling where barriers are traced, ran the full room width, and was taken for a wall; the free space beyond it, entered by too few frames, was dropped.

Fix: a barrier must come down from the local ceiling by more than 15 cm over 30% of its length (`1ddba65` -> `f2fa61a`)

| metric | before | after |
|---|---|---|
| wall median (cm) | 44.16 | 2.68 |
| wall max (cm) | 82.61 | 6.75 |
| walls within 1 cm | 0.0 | 0.5 |
| ceiling max (cm) | 2.47 | 2.47 |
| openings within 2 cm | 1/3 | 0/3 |
| adjacency correct | True | True |
| footprint error | -0.324 | -0.0228 |
| interval coverage | 0.16666666666666666 | 0.3333333333333333 |
| damage found | 0/0 | 0/0 |
| damage false positives | 19 | 24 |
