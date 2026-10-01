# real-damage-false-positives: Undamaged real rooms reported dozens of damage regions

Symptom: Real recording 47429922 of an undamaged living room: 59 damage regions (shelf and toy edges as cracks, a plant as mould, view seams as stains). On the synthetic flat the staged crack, above a door, was never seen (found only once the scorer checked where).

Cause: Views were ranked by how well they saw a surface's centre, so the strip above a door went unseen; objects in front of a wall leaked colour into its orthophoto at the LiDAR's depth resolution; mask outlines read as ridges; exposure steps between views read as stains.

Fix: coverage-chosen views, dense-geometry masks in front and in recesses, crack clearance from masks, per-view gain (`f57442f` -> `ed27738`)

| metric | before | after |
|---|---|---|
| wall median (cm) | 4.95 | 4.95 |
| wall max (cm) | 10.49 | 10.49 |
| walls within 1 cm | 0.0 | 0.0 |
| ceiling max (cm) | 0.86 | 0.86 |
| openings within 2 cm | 1/3 | 1/3 |
| adjacency correct | True | True |
| footprint error | -0.0265 | -0.0265 |
| interval coverage | 0.2 | 0.2 |
| damage found | 0/0 | 0/0 |
| damage false positives | 33 | 3 |
