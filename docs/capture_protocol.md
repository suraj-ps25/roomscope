# Capture protocol (one page)

Pick **one** of the three ways below. Each gives a full floor plan; LiDAR is the most
accurate, photos the least (see `docs/device_matrix.md`).

**Before any capture:** switch on the lights, open every interior door fully, and don't
move furniture during the capture. Mirrors, glass and shiny floors need nothing special.

---

## A. LiDAR (iPhone 12 Pro or newer *Pro* model, or iPad Pro)

1. Install **Stray Scanner** (free, App Store, by Stray Robots). In its settings set
   **frame rate to 15 fps**.
2. Stand in the first room, near where you came in. **Remember this spot.** Press record.
3. In **every** room, one after another:
   - Walk to the **middle of the room** and **turn slowly on the spot through one full
     circle** (about 10 seconds). On the way round, tilt the phone **up until you see the
     ceiling and down until you see the floor**, twice (in a small room you have to tilt
     well over: from the middle of a bathroom the floor is only in view below ~45°).
   - Then walk **slowly around the room about 1 m from the walls**, phone pointing at
     the walls, sweeping gently up and down.
   - Go **through each doorway slowly** (2–3 seconds), phone pointing ahead.
4. Step into every space you can, including a hallway, a walk-in cupboard, a balcony.
   Where you can't step in (a bathtub, a shower), lean in and point the phone around it.
5. **Finish back at the spot from step 2**, pointing the phone the same way you started,
   and hold still for 3 seconds. Stop recording.
6. Hand-off: connect the phone to the Mac → Finder → the iPhone → *Files* → *Stray
   Scanner* → drag the newest recording folder to the Mac (or in the Files app,
   long-press the folder → Share → AirDrop).

Pace: about a minute per room. Stay 0.5–3 m from the walls. Don't run, and don't cover
the camera.

## B. Video (any iPhone 15 or newer)

1. Camera app → **Video**, lens at **1×**, phone held **sideways (landscape)**.
2. One clip for the whole property. In **every** room: walk to the **middle**, **stop**,
   and **turn slowly on the spot through one full circle** (about 10 seconds), tilting
   up to the ceiling and down to the floor twice on the way round, exactly as in A step 3.
   Then **walk normally to the next room**, phone pointing where you are going; no need
   to walk around the walls. The turns are what gets measured: each one is found in the
   clip automatically and becomes that room. Turn slowly (blur ruins video), keep turning
   until you are past where you started, and do the turn **once per room**.
3. Hand-off: AirDrop the clip to the Mac. On the phone, tap *Options* first and turn on
   **All Photos Data** so the file keeps its original quality and metadata.

## C. Photos (any iPhone 15 or newer)

1. Camera app → **Photo**, lens at **1×**, **landscape**, no Portrait mode, no zoom.
2. In **each room** take **5–8 photos**:
   - **Photos 1–4:** stand in each corner of the room and point the phone toward the
     **opposite corner**, at chest height, held **nearly level** (a touch down) so **both
     the floor and the ceiling edge** are in the picture. Check the ceiling edge is
     really in frame, especially in a small room; if not, step back into the corner.
   - **Then, at every doorway into another room:** stand **on the threshold** and take
     **one photo into each room, back to back** (turn round between them), each pointing
     across that room toward its far corner and **tilted up a little** so the ceiling is in
     it (in a small room the corner photos can't show floor and ceiling together). The two go into the two rooms' folders.
     They are how the rooms are joined: both were taken from one spot, so the plan knows
     exactly where each room sits relative to the other.
3. Hand-off: make **one folder per room** on the Mac, named after the room
   (`kitchen`, `bedroom`, `hallway`, …), and put that room's photos in it. AirDrop with
   **Options → All Photos Data** on, so each photo keeps its camera information.

---

## Run

```
roomscope run <recording folder | video file | folder of room folders> --out runs/<name>
```

One command per capture. It writes `runs/<name>/plan.json` and `runs/<name>/plan.png`.

## What goes wrong, and what the pipeline does about it

| Situation | Effect | Handling |
|---|---|---|
| Mirror | LiDAR sees a reflected "room" behind the wall | Detected by reflecting the see-through points back onto the room; never reported as a window or a room |
| Window glass | Depth mostly passes through | Counted as open; reported as a window, not missing wall |
| Shiny / wet-look floor | Reflections under the floor | Floor level is the dominant upward surface, so reflections below it are ignored |
| Low light | Photos and video get noisy; LiDAR is unaffected | Turn the lights on. Photo/video intervals widen when matches are poor |
| Closed door | Looks like wall | Open all doors (step "Before any capture") |
