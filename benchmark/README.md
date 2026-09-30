# Benchmark

The benchmark set is ours to build (the brief provides no captures). Its composition is
fixed so it can't be flattered.

| Required | Our set |
|---|---|
| One multi-room capture: 3+ rooms plus a connector | `flat` (rooms + hallway) |
| One furnished room with staged damage in two classes | the bedroom: water stain + crack |
| The same rooms at all three tiers, multi-room set included | `flat` as LiDAR, video and per-room photo folders |
| At least one room captured twice at the same tier | the bedroom, LiDAR, twice (repeatability) |
| Laser or tape ground truth on everything | `ground_truth/flat.yaml` |

## Layout

```
benchmark/
  captures/<name>/        raw sensor data exactly as exported (not in git; see data/README)
  ground_truth/<property>.yaml
  manifest.yaml           capture -> tier, property, device, date
  sim/                    synthetic scenes (development only)
```

## Measuring ground truth

Use a laser distance meter if you have one (a tape is fine, but read it to the mm).
Measure each value **twice** and write both down. The evaluator averages them, and the
spread tells us how good the truth itself is.

For each room:

1. **Walls.** Choose the wall with the entrance door as the first wall, then go
   **counter-clockwise** (the room on your left as you walk along the walls). Measure each
   wall's length at about 1 m height, corner to corner along the wall face. Skirting
   boards don't count.
2. **Ceiling height.** At the room centre and at one other spot, floor to ceiling.
3. **Openings.** For every door, doorway and window: clear width between the jambs
   (inside the frame) and clear height. For windows also the sill height above the floor.
   Note which wall it's on (its index from step 1).
4. **Damage** (staged room): each region's longest extent and width, which wall it's on,
   and its class.

Then list which rooms connect to which (`adjacency`).

## Ground-truth file

```yaml
property: flat
rooms:
  - id: bedroom              # photo tier: must equal the room's folder name
    ceiling_height: [2.708, 2.712]
    walls: [3.600, 3.400, 3.600, 3.400]   # counter-clockwise from the entrance wall
    openings:
      - {type: door, width: 0.850, height: 2.050, wall: 0}
      - {type: window, width: 1.200, height: 1.200, sill: 1.000, wall: 2}
    damage:
      - {class: water_stain, wall: 2, extent: [0.65, 0.60]}
      - {class: crack, wall: 0, length: 0.62}
adjacency:
  - [bedroom, hallway]
```

## Scoring

```
roomscope eval runs/<name>/plan.json benchmark/ground_truth/<property>.yaml --out runs/<name>/metrics.json
```

This scores every gate in the brief: opening widths ≤ 2 cm on ≥ 85% of openings (missed
and phantom openings count as misses), ceiling ≤ 1.5 cm, adjacency, overlaps, footprint
and interval coverage. Repeatability comes from `evaluate.repeatability(plan_a, plan_b)`,
applied to the twice-captured room.

The synthetic sweep (`python benchmark/run_sim.py`) runs the same scoring on generated
captures. It is for development only and is never reported as the benchmark.
