# RoadSight · Vehicle Counter

Count vehicles from traffic footage. Upload a video, draw a region over each movement you care about, and get counts, class breakdowns and arrival intervals.

It runs entirely on your own machine. No video is uploaded anywhere.

**A trained model is included** (`models/best_v2_1280.pt`), so you can count overhead drone footage straight after install. See [The included model](#the-included-model).

---

## Quick start

You need **Python 3.9 or newer** and about 3 GB of free disk space (PyTorch is large).

### Mac

1. Download the repo: green **Code** button → **Download ZIP**, then unzip it. Or clone it:
   ```bash
   git clone https://github.com/u6610909/roadsight.git
   ```
2. Double-click **`Start.command`**. The first time, macOS may refuse to open it: right-click → **Open** → **Open**. You only do this once.
3. The first run installs the dependencies (a few minutes). Then your browser opens **http://127.0.0.1:8000** by itself.
4. To stop, close the Terminal window.

### Windows

1. Download and unzip the repo as above.
2. Double-click **`Start.bat`**.
3. Open **http://127.0.0.1:8000** if the browser doesn't open by itself.

### From a terminal (any system)

```bash
git clone https://github.com/u6610909/roadsight.git
cd roadsight
python3 -m pip install -r requirements.txt
python3 app.py
```

Then open **http://127.0.0.1:8000**. Stop the server with `Ctrl+C`.

The first run downloads PyTorch through `ultralytics`, which takes a while. Use a virtual environment if you prefer (`python3 -m venv .venv && source .venv/bin/activate` first).

---

## Count a video in five steps

1. **Upload a video.** Drag it onto the drop zone, or click to browse. MP4, MOV, AVI, MKV or M4V.
2. **Pick the model.** `best_v2_1280.pt` is already in the list. You can also upload any `.pt` file trained with Ultralytics; class names are read from the model, so your own classes work.
3. **Load a frame, then draw a region over each movement.** Click to place points, `Enter` or double-click to close, `Esc` to cancel. Name each region (for example `A`, `Bx`).
4. **Run.** Leave **Image size** at `1280` and **Device** at `auto`. Counts appear while the video is processed, with a live preview of the boxes and track IDs.
5. **Export** CSV or JSON.

Region sets save to `rois/` and reload from the dropdown, so a camera position only has to be set up once.

### Start with a short segment

**Length defaults to 120 seconds on purpose.** On a laptop, inference runs well below real time, so a 25-minute clip can take hours. Two minutes is enough to check your regions catch the right vehicles. Set length to `0` for the whole video once you're happy.

**Device `auto`** uses an Apple Silicon GPU (`mps`) or an NVIDIA GPU (`cuda`) when there is one, which is several times faster than CPU. The device actually used is shown while it runs.

---

## The included model

`models/best_v2_1280.pt` · 22.6 MB · SHA-256 `9260bbd6c778a2090cfbcd06b791db17c7fa9d463cd57b2901d18d332613e939`

| | |
|---|---|
| Architecture | YOLOv8s (11.1 M parameters), Ultralytics |
| Classes | `car`, `motorcycle`, `heavy` (bus, lorry and larger; pickups and light trucks count as `car`) |
| Trained on | 6,585 hand-labelled frames from about 6.6 h of overhead drone footage of the ABAC university entrance, Bang Na, Thailand. The drone hovers about 6 m above the road, 1920×1080 video |
| Training | COCO → VisDrone → ABAC v1 → ABAC v2, 30 epochs at 1280 px, batch 8, Google Colab T4. Split by whole flight session, never by random frame |
| Detection accuracy | mAP@50 **0.883** on four sessions it never saw (car 0.972, motorcycle 0.911, heavy 0.766). Only 38 of 14,493 validation boxes got the wrong class |
| Counting accuracy | within **1.7%** of a hand count over one 15-minute session (298 by hand, 303 counted). Across five sessions the hand ÷ model factor for cars + heavy is 0.935 with a 9.4% spread |
| Best settings | image size **1280**, confidence **0.20** (the F1-curve peak; already the app default) |

### Where it works and where it doesn't

- **Overhead views like the training footage:** a drone or a high pole looking down a road at roughly 6 m. A side-on street camera is a different problem and will need retraining.
- **Dry daylight only.** There is no rain or night footage in the training set (drones can't fly in rain), so it is untested in those conditions.
- **Counts are least accurate in light traffic** and far from the camera. Over-counting ranged from +1.7% at 298 vehicles to +17.2% at 128, and most of it comes from the far, narrow end of a region where cars are only a few dozen pixels tall. Keep regions on the nearer part of the road where you can.
- **It can still make mistakes.** For example, a cyclist can occasionally be boxed as a `motorcycle`. Use the counts, not single boxes.
- **"Heavy" is not "pickup".** If your hand count calls pickups heavy, the model's heavy count will look far too low. Compare `cars + heavy` instead, which is unaffected.

---

## What it counts, and how

**A vehicle is counted once,** when its tracked centre first enters a region. Tracking uses ByteTrack, so a vehicle keeps one identity across frames.

**Parked vehicles are not traffic.** A track counts only after it has moved 60 px from where it was first seen. At the junction this was built for, taxis queue across three lanes while waiting, and without this filter they would dominate every count.

**Counting windows advance by video time, never wall-clock time.** Inference is slower than real time, so a wall-clock window would cover only a fraction of the traffic it appears to.

**Classes are grouped by passenger car unit (PCU),** how much road capacity a vehicle uses:

| Group | PCU | Class names it accepts |
|---|---|---|
| `moto` | 0.33 | motorcycle, motorbike, bicycle |
| `car` | 1.0 | car, pickup, van, truck |
| `heavy` | 2.0 | heavy, bus, lorry, special type |

Unrecognised class names fall back to `car`. Edit `PCU_GROUP` in `app.py` to match your own model's labels.

---

## Output

**CSV:** per region, car, moto, heavy, total, cars + heavy, PCU-weighted flow, mean arrival intervals and vehicles per hour.

**JSON:** the same totals plus the full per-window time series and the exact settings used, so a result can be traced back to how it was made.

**Use the cars + heavy figure as simulation input.** Motorcycles filter to the front of queues and take almost no queue space, so including them in a car-following model overstates queue length.

---

## The other three tools

Counting from video is only part of measuring a junction. The rest is included on the home page.

- **Tally:** count vehicles by hand in 15-minute bins (the Highway Capacity Manual standard), by movement and lane. This is your ground truth to check the detector against. Don't skip it: a good mAP does not tell you whether your counts are right.
- **Headway:** lap-time a discharging queue to measure saturation headway and start-up lost time at your own site. Textbook defaults (2.00 s each) can be badly wrong: at the site this was built for, lost time measured 2.95 s, enough to change the study's conclusion.
- **Measure:** set a scale on an overhead frame and measure real distances, such as lane widths and queue storage lengths.

---

## Configuration

Defaults are at the top of `app.py`, each with the reasoning behind it:

| Constant | Default | What it does |
|---|---|---|
| `CONF_THRESHOLD` | `0.20` | detection confidence, from an F1-curve peak |
| `COUNTING_WINDOW_SEC` | `7.0` | bin width, in video seconds |
| `MIN_DISPLACEMENT_PX` | `60` | how far a track must move before it counts |
| `PCU_GROUP` | | maps your model's class names to the three groups |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named flask` (or `cv2`, `ultralytics`) | `python3 -m pip install -r requirements.txt` |
| Very slow, a few frames a second | Device is on `cpu`. Use `auto` on a Mac with Apple Silicon or a PC with an NVIDIA GPU, or try image size `960` |
| Device shows `cpu` on an NVIDIA PC | PyTorch was installed without CUDA. Reinstall it from [pytorch.org](https://pytorch.org/get-started/locally/) |
| Counts look far too high | The region covers the far end of the road, or doesn't match the video. Redraw it on the nearer section |
| `heavy` is always 0 | Your model uses class names not in `PCU_GROUP`. Add them in `app.py` |
| `Address already in use` | Something else is on port 8000. Close it, or stop the other copy of the app |

---

## Origin and credits

Built for a study of the ABAC university entrance T-junction on the Bang Na-Trat service road, Samut Prakan (CSX3010 Senior Project 1, project 26008, Assumption University), by **Ratchanon Puwapattarapokin**, **Warachai Aranchot** and **Badin Bangsen**, advised by Asst. Prof. Dr. Darun Kesrarat. Nothing in the app is specific to that junction; any fixed-camera traffic footage works, and the included model is a good starting point for overhead views.

## Licence

- **App code:** MIT.
- **Included model weights:** fine-tuned from Ultralytics YOLOv8s, which is licensed under AGPL-3.0, so the weights are distributed under AGPL-3.0. Check [Ultralytics' licensing](https://www.ultralytics.com/license) before any commercial use.
