# video-pixel-aspect: Every video-tier height ~1% short

Symptom: Video tier, flat_a with oracle depth: ceilings 2.5-3.8 cm low in every room although each ceiling was perfectly flat and wall lengths were within 4 mm.

Cause: Decoding 720x1280 to 294x518 (patch multiples) scales x and y differently (0.4083 vs 0.4047); focal self-calibration then set fy = fx, shrinking every vertical ray.

Fix: carry the decode's pixel aspect through the rotation search and into K (`bf0c5d5` -> `470c532`)

| metric | before | after |
|---|---|---|
| wall median (cm) | 0.3 | 0.12 |
| wall max (cm) | 1.79 | 0.39 |
| walls within 1 cm | 0.875 | 1.0 |
| ceiling max (cm) | 3.81 | 1.1 |
| openings within 2 cm | 7/9 | 9/9 |
| adjacency correct | True | True |
| footprint error | -0.002 | -0.0004 |
| interval coverage | 1.0 | 1.0 |
