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
     circle** (about 10 seconds). While turning, tilt the phone a little up and down so
     the floor edges and ceiling edges both come into view.
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
2. Walk the whole property exactly as in A, steps 2–5, in **one clip**: mid-room full
   turn in each room, slow walk around the walls, slow through doorways, finish where you
   started. Move slowly; blur ruins video.
3. Hand-off: AirDrop the clip to the Mac. On the phone, tap *Options* first and turn on
   **All Photos Data** so the file keeps its original quality and metadata.

## C. Photos (any iPhone 15 or newer)

1. Camera app → **Photo**, lens at **1×**, **landscape**, no Portrait mode, no zoom.
2. In **each room** take **5–8 photos**:
   - **Photos 1–4:** stand in each corner of the room and point the phone toward the
     **opposite corner**, at chest height, tilted slightly down so **both the floor and
     the ceiling edge** are in the picture.
   - **Then one photo per doorway:** stand about **1 m inside the room facing the
     doorway**, so the whole door frame and a glimpse of the next room are in view.
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
