# Facility Monitor POC

Proof of concept for monitoring rooms in a medical testing facility from an MP4 recording or a webcam:

- **People**: detects and tracks every person (stable IDs), reports people in frame and unique people seen.
- **Lights**: classifies room lighting as ON / OFF from frame brightness, with tunable thresholds.
- **Rules → alerts**: "minimum people", "maximum people", "lights must be on", each with a grace period.
  Violations become timestamped events with a snapshot, shown live on the dashboard and exportable as CSV.

Stack: Ultralytics [People Detection](https://platform.ultralytics.com/ultralytics/solutions/people-detection) crowd model (YOLO26x fine-tuned on MOT20) with [YOLO26n](https://docs.ultralytics.com/models/yolo26/) as the fast fallback + ByteTrack (people), OpenCV (lighting), FastAPI (API,
MJPEG stream, WebSocket metrics), single-page dashboard (no build step).

## Quick start

```bash
uv sync                                   # creates .venv with Python 3.12 and pinned deps
uv run python scripts/get_samples.py      # downloads a sample clip and builds the demo clip
uv run uvicorn app.main:app --port 8000 --timeout-graceful-shutdown 2   # first start downloads people-detection.pt (~118 MB) from Ultralytics Platform
```

Open <http://localhost:8000>, pick `demo_lights_and_people.mp4`, press **Start**.

### Demo storyline (`samples/demo_lights_and_people.mp4`, 40 s, loops)

| Time | What happens | Dashboard |
|---|---|---|
| 0–6 s | people walking, lights on | green **ALL CLEAR** |
| 6–14 s | room goes dark | brightness drops, lights **OFF**, after 3 s grace → red **ALERT · LIGHTS OFF** |
| 20–27 s | room empty (simulated) | people = 0, after 3 s grace → **ALERT · NO PERSON IN ROOM** |
| 27 s+ | back to normal | alerts resolve, events logged with thumbnails |

Talking points while it runs: press the **Dark** preset under *Lighting test* (or drag the simulated light level slider) and
watch brightness collapse, lights flip to OFF and a lights-off alert fire after the grace period, then press **As recorded**
to see it resolve; use the seek bar or the 10 s buttons (arrow keys, Shift for 30 s) to jump straight to the dark or empty
segment; drag the **OFF below** slider above the current brightness and the lights state flips live;
set **Max people** to 5 to raise a "too many people" alert; **Upload video** to run your own footage; select **Webcam**
for a live feed (macOS asks for camera permission for the terminal app the first time).

## How it works

```
source (file / webcam)
   └─ VideoPipeline thread (app/pipeline.py)
        ├─ PersonTracker  (app/detector.py)  People Detection (YOLO26x/MOT20) or YOLO26n, class=person, ByteTrack IDs
        ├─ LightingAnalyzer (app/lighting.py) mean HSV-V brightness, EMA smoothing, hysteresis
        ├─ RuleEngine (app/rules.py)          min/max people, lights required, grace period → events
        └─ SharedState (app/state.py)         latest JPEG, metrics, 5-min history, event log
FastAPI (app/main.py): /stream.mjpg (video), /ws (metrics 5 Hz + events), /api/* (control, seek, settings, upload, CSV)
static/: dashboard
```

All settings (`app/settings.py`) are live-updatable from the dashboard while a source is running. Two of them exist for
testing rather than monitoring: `light_gain` multiplies every frame before detection and lighting analysis (so the video,
the brightness meter and the Lights rule all react as if the room really dimmed), and `POST /api/seek` with `time` or
`delta` repositions file playback, closing any active alert with a "Playback position changed" event and resetting the
tracker, lighting smoothing and grace timers so nothing stale carries across the jump.

### Why these choices

- **Person detection** defaults to the Ultralytics Platform *People Detection* solution: a YOLO26x fine-tuned on the
  MOT20 crowd-tracking dataset at 1280 px, person class only (reported mAP50 0.956 / mAP50-95 0.635). It is the most
  accurate option on crowded and partially hidden people, but on a CPU-only laptop it runs at ~0.5 detections/s at
  640 px, so playback skips most frames. For a smooth real-time demo switch the dashboard **Model** to YOLO26n.
  Weights are fetched automatically on first use from the Platform files API (no API key required for public models).
- The generic COCO models need no training either. YOLO26n is the
  smallest variant; pick YOLO26s or YOLO26m in the dashboard for better accuracy on a GPU box. YOLO26 is NMS-free
  (end-to-end), so it is faster on CPU and edge devices than YOLO11 at the same size; the YOLO11 weights stay
  selectable for side-by-side comparison.
- **Lights on/off is not a detection problem.** Mean frame brightness with smoothing and hysteresis is robust,
  free, and tunable per camera. Detecting light fixtures would be slower and less reliable.
- **Rules live outside the model.** Each new use case (dwell time, restricted zone, PPE, door open) is a new rule
  or a new lightweight analyzer, not a retrained model.

## Detection quality vs speed

"Why are some people not boxed?" is almost always one of these, in order of likelihood:

1. **They are small in the frame** (far from the camera, low-resolution footage). Raise **Image size** from 640 to
   960 or 1280. On the bundled 640x360 crowd clip, YOLO11n finds ~8 people at 480, ~17 at 640, ~35 at 960 and
   ~42 at 1280 per frame. Cost: roughly half the speed per step.
2. **The model is too small.** Switch **Model** to YOLO26s (about 2x slower than n on CPU, noticeably better on
   small and partially hidden people). YOLO26m is better still but too slow for CPU-only demos.
   Note: on the bundled dense crowd clip YOLO26n out-detects YOLO11n at the same speed, but YOLO26s scored below
   YOLO11s (its NMS-free head gives lower scores on heavily overlapped people). If a scene is very crowded, compare
   both from the Model dropdown; the YOLO11 weights are kept for exactly that.

   Measured on the bundled 640x360 crowd clip (Intel i5 CPU, raw conf 0.1, people per frame):

   | Model | 640 | 960 | ms/frame at 640 |
   |---|---|---|---|
   | People Detection (YOLO26x/MOT20) | 48.0 | 49.7 | ~2000 |
   | YOLO26n | 33.7 | 40.0 | 81 |
   | YOLO11n | 27.8 | 45.6 | 84 |
   | YOLO11s | 34.4 | 40.6 | 154 |
3. **The confidence is too high.** 0.25 is the default; 0.15 finds more but adds flicker. Note the raw detector
   always runs at 0.1 and the confidence you set is applied by the tracker (ByteTrack needs low-score detections
   to keep tracks alive through occlusion), so counts are stable tracks, not raw detections.
4. **Heavy crowding.** People overlapping in a tight group merge into one or two boxes with any general-purpose
   detector. A crowd-counting model (density estimation) is the right tool if that is the actual use case.

Playback stays real time on a slow CPU by skipping frames (shown as "frames skipped" under the video). Raise
**Detect every** to 3 to reduce skipping, or tick **Max speed** to process every frame as fast as possible.

## Performance notes

- On this Intel i5 laptop (CPU only): YOLO11n at 640 runs ~5 detections/s, at 960 ~3/s; YOLO11s at 640 ~3.5/s.
  The dashboard shows throughput of published frames, which is higher because not every frame is detected.
- **Intel Macs**: PyTorch stopped shipping Intel-Mac wheels after 2.2.2, so `pyproject.toml` pins
  `torch==2.2.2`, `torchvision==0.17.2`, `numpy<2`. On Apple Silicon or Linux you can drop those pins.
- Video decoding uses OpenCV's bundled FFmpeg: MP4 (H.264) is safe; HEVC `.mov` from iPhones may not decode.

## Taking it to the edge / production

- **Edge devices**: export the model once (`yolo export model=yolo26n.pt format=onnx|openvino|engine|coreml`) and
  run the same `detector.py` interface on Jetson (TensorRT), Intel NUC (OpenVINO), Raspberry Pi 5 + Hailo, or a
  Mac mini (CoreML). Ultralytics docs: <https://docs.ultralytics.com/modes/export/>.
- **Multiple cameras**: one `VideoPipeline` per RTSP stream; the state/rules/dashboard already key by source.
- **Persistence & alerting**: write events to a database and push to Slack/SMS/email from `RuleEngine.on_event`.
- **Licensing**: Ultralytics code and YOLO26/YOLO11 weights are AGPL-3.0. For a commercial product either buy the
  Ultralytics Enterprise license or swap the detector for an Apache-2.0 model (RF-DETR, D-FINE, YOLOX, RT-DETR).
  `detector.py` is the only file that would change.
- **Accuracy**: the demo uses generic COCO weights. For real rooms, collect ~30 min of footage per camera and
  fine-tune (or at least tune `conf` and the lighting thresholds per camera).

## Layout

```
app/        backend (see diagram above)
static/     dashboard (index.html, app.js, styles.css)
scripts/    get_samples.py – demo data
samples/    demo videos (git-ignored)
uploads/    videos uploaded via the UI (git-ignored)
models/     people-detection.pt, yolo26n.pt and friends (auto-downloaded, git-ignored)
```
