# real-ceiling-scale-47429914: Every real ceiling read low (dev recording)

Symptom: All 14 real-room ceilings were low against laser truth: dev median -0.9%, held-out -0.74%; this bathroom recording -2.0 cm (gate 1.5 cm).

Cause: The iPad Pro's LiDAR depth reads short: similarity ICP of the fused cloud on the laser scan needs a scale of 1.0087-1.0106 on every dev recording that registers well.

Fix: per-device depth-scale calibration from the dev visits (x1.0088), applied to that device only (`7f5bd43` -> `561189d`)

| metric | before | after |
|---|---|---|
| wall median (cm) | 3.11 | 2.12 |
| wall max (cm) | 6.43 | 3.79 |
| walls within 1 cm | 0.25 | 0.25 |
| ceiling max (cm) | 1.59 | 0.69 |
| openings within 2 cm | 0/2 | 0/2 |
| adjacency correct | True | True |
| footprint error | -0.0239 | -0.0076 |
| interval coverage | 0.6666666666666666 | 0.8333333333333334 |
| damage found | 0/0 | 0/0 |
| damage false positives | 2 | 3 |
