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

## Status

Not yet run: it needs an iPhone and the two rooms (`docs/compliance_matrix.md`). The
scorer is tested on synthetic numbers.
