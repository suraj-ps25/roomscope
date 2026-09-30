# Device matrix

Which tier runs on which hardware, and what each tier honestly delivers. Every number
says where it comes from. **Real** means our benchmark captures with tape/laser truth
(`benchmark/`). **Synthetic** means the simulator with exact truth, used for development
only. Nothing is claimed that hasn't been measured.

## Capture hardware

| Tier | Phones | App | What it records |
|---|---|---|---|
| LiDAR | iPhone 12 Pro / Pro Max and every later **Pro** model (13, 14, 15, 16, 17 Pro/Pro Max); iPad Pro 2020+ | Stray Scanner (free) | 256×192 metric depth + confidence, ARKit camera poses, per-frame intrinsics, RGB video |
| Video | any iPhone 15 or newer, Pro or not | Camera (built in) | one walkthrough clip |
| Photo | any iPhone 15 or newer, Pro or not | Camera (built in) | 2–8 stills per room, one folder per room |

A Pro phone can run all three tiers. Stray Scanner refuses to record on a phone without
LiDAR ("Unsupported device"), so a non-Pro phone runs video and photo only.

## Processing machine

Runs locally; no service of ours is called. Tested on an Apple M2 (16 GB, macOS 15).
Models run on the Apple GPU (MPS), on CUDA when present, or on CPU (slower).

## Accuracy by tier

Intervals are stated at 90% coverage and widen as input thins. Scale error is the main
term for photo and video, because metric scale there comes from a learned prior.

| Quantity | LiDAR | Video | Photo |
|---|---|---|---|
| Wall length | Synthetic: max 0.58 cm, 100% within 1 cm (flat_a, drift ×1). Real: pending | Gate ±3%. Pending | Gate ±8%. Pending |
| Ceiling height | Synthetic: max 0.25 cm. Real: pending | pending | pending |
| Opening width | Synthetic: 9/9 within 2 cm. Real: pending | pending | pending |
| Stitched footprint | Synthetic: +0.0%, adjacency correct | pending | gate ±8% |
| Metric scale source | LiDAR depth (±0.3% budget) | MoGe-2 over many keyframes (±1.5% floor) | MoGe-2 over a room's photos: σ = √((6.9%/√n)² + 2%²), e.g. 3.7% at 5 photos |

Component measurements behind the photo/video budget, on real iPad images with laser
ground truth (ARKitScenes 47429912, 14 views): MoGe-2 metric scale −0.5% bias, 6.9%
per-view spread. MapAnything conditioned on MoGe-2 depth: 0.978 of truth.

## Timing (M2, 16 GB)

| Tier | Typical run |
|---|---|
| LiDAR | ~3.5 min for a 4-room, 3.8-minute protocol walk (synthetic) |
| Video | pending |
| Photo | ~30–70 s per room (5 photos), first run; cached re-runs are fast |
