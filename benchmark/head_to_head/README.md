# Head-to-head vs a consumer app

Two rooms, the same session, the same phone, against the same laser/tape truth.

## Protocol

1. Measure ground truth first (`benchmark/README.md`), so nobody tunes to the app.
2. For each of the two rooms, back to back:
   - **magicplan** (free tier is enough): *New project → Scan room* with LiDAR on (Pro
     phone) or camera mode (non-Pro). Close the room and accept its dimensions without
     editing them. Export the room report (PDF or CSV).
   - **roomscope**: capture the same room with the tier being compared (LiDAR against
     magicplan-LiDAR, photo or video against magicplan's camera mode).
3. Transcribe the app's numbers into `app.yaml` (format in `score.py`): every wall in
   ground-truth order, ceiling height, and every opening width. A number the app doesn't
   report is left out, which counts in the app's favour (only quantities both report are
   scored).
4. Score:

```
python benchmark/head_to_head/score.py app.yaml runs/<name>/plan.json benchmark/ground_truth/<property>.yaml
```

A quantity is a **win** when our error is smaller and a **tie** when the two errors are
within 5 mm. The brief's bar is ≥ 70% of quantities beaten or tied.

## Without a phone: the same raw cloud through a point-cloud tool

No iPhone was available, and consumer scanning apps (magicplan, Polycam, RoomPlan apps)
only take a live scan. What was run instead uses tools that do accept an upload:

1. `export_cloud.py` writes a real ARKitScenes iPad Pro recording as a coloured point
   cloud: raw LiDAR depth on the device's own ARKit poses, none of our drift correction or
   depth calibration (converted to LAS for upload).
2. **Pointorama** (web, 2 Oct 2026) was given that cloud. Only its automatic tools were
   used: *Auto Floor* for the level and ceiling, then *Magic Room* brushed over the whole
   room and accepted. Nothing was edited. Its DXF and IFC exports are in `results/`; name, version and export record in `results/APP.md`.
3. `dxf_to_app.py` reads the DXF outline into `app.yaml`. Each truth wall is located by our
   plan's matching wall line (the two share the recording's frame), and the tool's length
   for it is how far its outline runs along that line, end to end, so a wall drawn in
   steps still reads as one wall. The allowance is 0.6 m because the tool saw the raw
   cloud, whose drift we correct and it can't: in the large room its walls sit up to
   0.5 m from ours (`results/42897678_overlay.png`). Ceiling = its IFC wall height. It
   reported no doors or windows.
4. `score.py` scores both against the laser truth (`benchmark/ground_truth/`). Walls the
   laser never measured end to end are left out, as in the main benchmark.

```
python benchmark/head_to_head/dxf_to_app.py results/47429914_pointorama.dxf runs/<ours>/plan.json bathroom \
    --truth benchmark/ground_truth/arkitscenes_471428.yaml --ceiling 2.4955 --app Pointorama --out app.yaml
python benchmark/head_to_head/score.py app.yaml runs/<ours>/plan.json benchmark/ground_truth/arkitscenes_471428.yaml
python benchmark/head_to_head/overlay.py cloud.ply results/47429914_pointorama.dxf runs/<ours>/plan.json overlay.png "title"
```

### Results against laser truth

Errors in cm (floor area in m²). *Ours* is the LiDAR tier as shipped (device depth
calibration ×1.0088). *Ours, raw* is `roomscope run --depth-scale 1.0`: identical input
to the tool's.

| room | quantity | truth | Pointorama | ours | ours, raw |
|---|---|---|---|---|---|
| bathroom | wall 0 | 2.721 m | 5.6 | 3.8 | 6.4 ✗ |
| | wall 1 | 1.983 m | 3.2 | 2.1 | 3.8 ✗ |
| | wall 2 | 2.716 m | 15.1 | 0.5 | 2.4 |
| | wall 3 | 1.888 m | 23.6 | 2.1 | 0.7 |
| | ceiling | 2.515 m | 1.95 | 0.69 | 1.59 (tie) |
| | floor area | 5.26 m² | 0.42 | 0.04 | 0.13 |
| | window | 0.64 m | not found | 6.0 | 10.4 |
| | door | 0.80 m | not found | missed (tie) | missed (tie) |
| large room | wall 0 | 4.385 m | **0.8** | 5.4 ✗ | 1.8 ✗ |
| | wall 1 | 3.530 m | 67.7 | 3.0 | 1.0 |
| | wall 5 | 4.811 m | 2.4 | 0.4 | 4.0 ✗ |
| | ceiling | 2.338 m | 0.87 | 0.43 (tie) | 1.41 ✗ |
| | floor area | 18.50 m² | 2.59 | 0.21 | 0.03 |
| | window | 1.45 m | not found | 1.0 | 1.0 |
| | door | 0.72 m | not found | 5.4 | 6.3 |
| **beaten or tied** | | | | **14/15 (93%)** | **10/15 (67%)** |

✗ = the tool's error is smaller. Per room: shipped 8/8 and 6/7, raw 6/8 and 4/7
(`results/*_calibrated.txt`, `results/*_raw.txt`).

**Reading it.**
- As shipped, the LiDAR tier clears the brief's ≥ 70% bar in both rooms. Pointorama's
  big losses are topology: its outline steps out past walls (13 edges for the 4-wall
  bathroom, 18 for the 6-wall room, one of them a spike out of the room), so walls come
  out short, the area 8–14% large, and no opening is reported.
- On identical raw input we fall to 67%, under the bar. Where Pointorama does draw a wall
  cleanly it is as accurate as we are: its large-room long wall is within 0.8 cm.
  Our margin on wall lengths comes partly from the device depth calibration, which a
  tool given only the cloud can't apply.
- The bathroom is from one of the two visits the depth calibration was fitted on; the
  large room's visit was not used for it (it was held out then, and is dev now only for
  the later sliver-wall fix).

**Caveats.** Pointorama is a professional point-cloud tool, not the consumer phone app
the brief names, and it saw the iPad's cloud rather than making its own scan. In the second
recording its other outline (1.3 m²) shares no floor with our small room, so the small room
is not scored; `dxf_to_app.py` keeps each outline separate and uses the one sharing the
most floor with our matched room.

## Status

Two rooms run and scored above. The on-phone protocol (magicplan on the same iPhone)
still needs a phone.
