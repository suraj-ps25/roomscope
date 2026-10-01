# real-openings-47429922: A two-sash window read as two windows (declared fix, cause a)

Symptom: Real living room: one 1.36 m window reported as 0.60 and 0.66 m windows 0.16 m apart: one phantom and a -70 cm width error.

Cause: the frame member between sashes breaks the run of open votes; each run became an opening

Fix: openings on one wall at most 0.2 m apart and overlapping in height are one opening (`7e5e7be` -> `1b84c55`)

| metric | before | after |
|---|---|---|
| wall median (cm) | 4.79 | 4.79 |
| wall max (cm) | 9.71 | 9.71 |
| walls within 1 cm | 0.0 | 0.0 |
| ceiling max (cm) | 0.93 | 0.93 |
| openings within 2 cm | 0/2 | 0/1 |
| adjacency correct | True | True |
| footprint error | -0.0235 | -0.0235 |
| interval coverage | 0.4444444444444444 | 0.5555555555555556 |
| damage found | 0/0 | 0/0 |
| damage false positives | 4 | 4 |
