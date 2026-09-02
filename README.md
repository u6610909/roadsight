# Vehicle Counter

Count vehicles from traffic footage. Upload a video, draw a region over each
movement you care about, and get counts, class breakdowns and arrival intervals.

Runs entirely on your own machine — no video is uploaded anywhere.

![four tools: detector, tally, headway, measure](docs/hub.png)

---

## Install

```bash
git clone https://github.com/<you>/vehicle-counter.git
cd vehicle-counter
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:8000**

Python 3.9+. First run downloads PyTorch via `ultralytics`, which takes a while.

---

## Use it

1. **Upload a video** — drag it onto the drop zone, or click to browse.
2. **Upload a YOLO model** — any `.pt` file trained with Ultralytics. Class names
   are read from the model, so your own taxonomy works.
3. **Load a frame**, then **draw a region** over each movement. Click to place
   points, `Enter` or double-click to close, `Esc` to cancel. Name each region.
4. **Run.** Counts appear as the video is processed.
5. **Export** CSV or JSON.

Region sets save to `rois/` and reload from the dropdown, so a camera position
only has to be set up once.

### Start with a short segment

**Length defaults to 120 seconds deliberately.** Inference on a laptop CPU runs
at roughly an eighth of real time, so a 25-minute clip takes hours. Two minutes
is enough to confirm your regions are catching the right vehicles. Set length to
`0` for the whole video once you're satisfied.

Set **Device** to `auto` and it will use an Apple Silicon GPU or CUDA if present,
typically several times faster. The device actually used is shown while running.

---

## What it counts, and how

**A vehicle is counted once** — when its tracked centre first enters a region.
Tracking uses ByteTrack, so a vehicle keeps one identity across frames.

**Parked vehicles are not traffic.** A track is counted only after it has moved
60 px from where it was first seen. Without this, anything stationary in view
inflates the count. At the junction this was built for, taxis are required to
queue across three lanes while waiting, and they would otherwise have dominated
every measurement.

**Counting windows advance by video time, never wall-clock.** Since inference is
slower than real time, a wall-clock window would cover a small fraction of the
traffic it appears to.

**Classes are grouped by passenger car unit** — how much road capacity a vehicle
consumes, which is what traffic engineering works in:

| Group | PCU | Typical classes |
|---|---|---|
| `moto` | 0.33 | motorcycle, bicycle |
| `car` | 1.0 | car, pickup, van, truck |
| `heavy` | 2.0 | bus, lorry |

Unrecognised class names fall back to `car`. Edit `PCU_GROUP` in `app.py` to
match your own model's labels.

---

## Output

**CSV** — per region: car, moto, heavy, total, cars+heavy, PCU-weighted flow,
mean arrival intervals, and vehicles per hour.

**JSON** — the same totals plus the full per-window time series and the exact
configuration used, so a result can be traced back to the settings that produced
it.

**Use the cars+heavy figure for simulation input.** Motorcycles filter to the
front of queues and occupy almost no queue space, so including them in a
car-following model overstates queue length substantially.

---

## The other three tools

Counting from video is only part of measuring a junction. The rest is included.

**Tally** — manual vehicle counting in 15-minute HCM bins, by movement and lane.
This is how you build the ground truth to validate the detector against. Do not
skip it: detection metrics like mAP do not tell you whether your counts are
right.

**Headway** — lap-time a discharging queue to measure saturation headway and
startup lost time at your own site. Published defaults (2.00 s and 2.00 s) can be
substantially wrong: at the site this was built for, startup lost time measured
2.95 s, which was enough to change the study's conclusion.

**Measure** — calibrate a scale on an overhead frame and measure real distances:
lane widths, queue storage lengths, stop-line offsets.

---

## Configuration

Defaults are at the top of `app.py`, each with the reasoning behind it:

| Constant | Default | What it does |
|---|---|---|
| `CONF_THRESHOLD` | `0.20` | detection confidence — set from an F1-curve peak, not a guess |
| `COUNTING_WINDOW_SEC` | `7.0` | bin width, in video seconds |
| `MIN_DISPLACEMENT_PX` | `60` | how far a track must move before it counts |
| `PCU_GROUP` | — | maps your model's class names to the three groups |

---

## Limitations

**Accuracy depends on your model.** This is the counting and tracking layer; a
model trained on a different viewpoint or altitude will not transfer well.

**Counts are least accurate when traffic is light.** A roughly constant
false-positive rate forms a larger share of a smaller total. On the reference
dataset the error ranged from +1% at 298 vehicles to +19% at 128.

**Validate before you trust it.** Hand-count one 15-minute bin with the Tally
tool and compare. On the reference site this gave counting accuracy within 1.3%
of ground truth — but that is a property of that model and that footage, not a
guarantee.

---

## Origin

Built for a study of the ABAC university entrance T-intersection on Bang Na–Trat,
Samut Prakan (CSX3010 Senior Project 1). Nothing in it is specific to that
junction; any fixed-camera traffic footage works.

## Licence

MIT
