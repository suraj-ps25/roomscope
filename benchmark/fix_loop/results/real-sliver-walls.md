# real-sliver-walls: Captures of one room disagreed on its wall sequence

Symptom: Repeatability on real recordings: of 12 pairs of recordings of the same room, 5 could not be compared at all (different wall counts) and the rest agreed on 11 of 42 walls; walls of 0.01-0.26 m appeared in some recordings of a room and not others.

Cause: A sliver at a cut corner, or a few-centimetre step between two faces of one wall, is fitted as its own wall in one capture and not in another; every wall after it is then paired with the wrong wall.

Fix: walls under 0.3 m absorbed into their neighbours in the reported layout (steps over 5 cm kept) (`23cb877` -> `7c2909e`)

| metric | before | after |
|---|---|---|
| wall median (cm) | None | 14.54 |
| wall max (cm) | None | 17.06 |
| walls within 1 cm | None | 0.0 |
| ceiling max (cm) | 1.37 | 1.37 |
| openings within 2 cm | 1/1 | 1/1 |
| adjacency correct | False | False |
| footprint error | 1.0026 | 1.0111 |
| interval coverage | 1.0 | 0.5714285714285714 |
| damage found | 0/0 | 0/0 |
| damage false positives | 11 | 11 |
