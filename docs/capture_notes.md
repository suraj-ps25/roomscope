# Capture notes (for the operator, not the person capturing)

The capture page is [`capture_protocol.md`](capture_protocol.md). This page explains why it
says what it says, and what the pipeline does when a capture goes wrong.

## Running

```
roomscope run <recording folder | video file | folder of room folders> --out runs/<name>
```

The tier is detected from the input: a Stray Scanner folder means LiDAR, a video file
means video, and a folder of room folders means photos. One command per capture writes
`runs/<name>/plan.json` (the published schema) and `runs/<name>/plan.png`.

## Why the protocol asks what it asks

- **The turn in the middle of each room** (video): a pure rotation lets the lens be
  calibrated from the clip itself, and each room becomes one panorama, with no long
  trajectory to chain. Each heading is seen at one tilt per revolution, so the turn
  tilts both up and down. In landscape at 1×, the floor, walls and ceiling don't all fit
  at one tilt.
- **The walk ~1 m from the walls** (LiDAR): the depth sensor is best at 0.5–3 m.
- **Finishing where you started** (LiDAR): the end of the scan overlaps its start, so
  drift can be closed as a loop.
- **The threshold photos** (photos): taken back to back from one spot, so the plan knows
  exactly where each room sits relative to the next. The threshold also says where a
  door is.
- **Tilting up to the ceiling:** the band just under the ceiling is where walls and door
  headers are traced.
- **Very small rooms:** from the middle of a bathroom the floor is only in view below
  about 45°, hence "tilt well over".

## What goes wrong, and what the pipeline does about it

| Situation | Effect | Handling |
|---|---|---|
| Mirror | LiDAR sees a reflected "room" behind the wall | Detected by reflecting the see-through points back onto the room; never reported as a window or a room |
| Window glass | Depth mostly passes through | Counted as open; reported as a window, not missing wall |
| Shiny / wet-look floor | Reflections under the floor | Floor level is the dominant upward surface, so reflections below it are ignored |
| Low light | Photos and video get noisy; LiDAR is unaffected | Photo and video intervals widen when matches are poor |
| Closed door | Looks like wall | The protocol asks for every door to be open |
| Ceiling never filmed (phone kept level) | No ceiling height; door headers unseen | Walls are found above furniture height and doorway gaps closed; the ceiling is reported as a prior (2.6 m, never below the walls seen) with a wide interval and a note |
| A recording that never closes a room (too short, a wall never seen) | No room | The command says so, lists why, and points at the protocol |
