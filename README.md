# roomscope

Turn handheld iPhone capture into a **dimensioned, stitched floor plan** with
**per-surface damage detection and repair scope** — from photos, video, or LiDAR.

One capture in, one command, and out comes: a measured plan of every room (walls,
ceiling height, floor area, openings), a stitched whole-property plan with correct
adjacency, damage regions with class and metric extent, concealed-damage flags with
the rule that fired, scope line items keyed to surfaces, and a confidence interval on
every measurement — as JSON to a published schema plus a rendered plan.

## Input tiers

The same output contract from all three, with intervals that widen honestly as sensor
data thins:

| Tier   | Input                                   | Depth / poses      |
|--------|-----------------------------------------|--------------------|
| Photo  | 2–8 stills per room (one folder / room) | none               |
| Video  | handheld walkthrough clip               | estimated          |
| LiDAR  | depth + poses + intrinsics (Pro device) | measured           |

## Run

```
roomscope run <capture_dir> --out runs/<name>
```

One command per capture. Produces `plan.json` (to `schema/floorplan.schema.json`) and
`plan.png` (rendered plan). See `docs/architecture.md` for the pipeline.

## Status

Early scaffold. The capture route, tier pipelines, benchmark harness, and reports are
being built incrementally — see `docs/architecture.md` for the plan and current state.

## Layout

```
src/roomscope/   pipeline package (ingest, tiers, stitch, measure, damage, render)
schema/          published JSON output contract
docs/            architecture, capture protocol, device matrix, reports
benchmark/       benchmark set manifest, ground truth, gate harness
scripts/         weight fetchers and one-off tools
```
