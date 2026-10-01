# synthetic-damage-recall: The staged crack was never found (the same fix, on the synthetic flat)

Symptom: Full-resolution LiDAR capture of the flat, damage scored at the staged place: 2/3 found and one false positive, a 0.49 m door-jamb edge that the old scorer had counted as the staged crack.

Cause: see real-damage-false-positives

Fix: see real-damage-false-positives (`f57442f` -> `ed27738`)

| metric | before | after |
|---|---|---|
| wall median (cm) | 0.13 | 0.13 |
| wall max (cm) | 0.52 | 0.52 |
| walls within 1 cm | 1.0 | 1.0 |
| ceiling max (cm) | 0.15 | 0.15 |
| openings within 2 cm | 9/9 | 9/9 |
| adjacency correct | True | True |
| footprint error | -0.0001 | -0.0001 |
| interval coverage | 1.0 | 1.0 |
| damage found | 2/3 | 3/3 |
| damage false positives | 1 | 0 |
