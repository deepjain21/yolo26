"""Download a public people-walking clip and build a scripted demo clip with a 'lights off' segment.

Usage: uv run python scripts/get_samples.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
SRC_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4"
SRC = SAMPLES / "people_walking.mp4"
DEMO = SAMPLES / "demo_lights_and_people.mp4"

# Demo storyline (seconds): normal -> lights off (dark) -> normal -> empty room (people masked) -> normal
DARK_START, DARK_END = 6.0, 14.0
EMPTY_START, EMPTY_END = 20.0, 27.0
DARK_FACTOR = 0.12
TARGET_SECONDS = 40.0  # the source clip is ~2 s; it is ping-ponged to this length so the storyline has room


def download() -> None:
    if SRC.exists() and SRC.stat().st_size > 0:
        print(f"exists: {SRC}")
        return
    print(f"downloading {SRC_URL} ...")
    urllib.request.urlretrieve(SRC_URL, SRC)
    print(f"saved {SRC} ({SRC.stat().st_size/1e6:.1f} MB)")


def mask_people(frame: np.ndarray) -> np.ndarray:
    """Paint over any detected people so the frame reads as an empty room (simulated)."""
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "models" / "yolo26n.pt"))
    out = frame.copy()
    for _ in range(3):  # extra passes catch low-confidence leftovers
        r = model.predict(out, classes=[0], conf=0.1, imgsz=640, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            break
        for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy().astype(int):
            pad = 8
            x1, y1 = max(x1 - pad, 0), max(y1 - pad, 0)
            x2, y2 = min(x2 + pad, out.shape[1]), min(y2 + pad, out.shape[0])
            strip = out[y2:min(y2 + 12, out.shape[0]), x1:x2]
            colour = strip.reshape(-1, 3).mean(axis=0) if strip.size else out.reshape(-1, 3).mean(axis=0)
            out[y1:y2, x1:x2] = colour.astype(np.uint8)
    cv2.putText(out, "SIMULATED: EMPTY ROOM", (10, out.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1, cv2.LINE_AA)
    return out


def build_demo() -> None:
    cap = cv2.VideoCapture(str(SRC))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"source: {w}x{h} @ {fps:.1f} fps, {n/fps:.1f}s")
    out = cv2.VideoWriter(str(DEMO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    # "empty room" segment: freeze a background-ish frame (median of a few frames) so no people are visible
    bg_frames = []
    for i in range(0, min(n, 300), 15):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            bg_frames.append(f)
    background = np.median(np.stack(bg_frames), axis=0).astype(np.uint8) if bg_frames else None
    if background is not None:
        background = mask_people(background)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    # ping-pong (forward, backward, forward...) so there are no jump cuts when looping a short clip
    seq = frames + frames[-2:0:-1]
    total = int(TARGET_SECONDS * fps)
    idx = 0
    while idx < total:
        frame = seq[idx % len(seq)].copy()
        t = idx / fps
        if DARK_START <= t < DARK_END:
            frame = (frame.astype(np.float32) * DARK_FACTOR).astype(np.uint8)
        elif EMPTY_START <= t < EMPTY_END and background is not None:
            frame = background
        out.write(frame)
        idx += 1
    out.release()
    print(f"built {DEMO} ({idx/fps:.1f}s): dark {DARK_START}-{DARK_END}s, empty {EMPTY_START}-{EMPTY_END}s")


if __name__ == "__main__":
    SAMPLES.mkdir(exist_ok=True)
    try:
        download()
    except Exception as e:  # noqa: BLE001
        print("download failed:", e, file=sys.stderr)
        sys.exit(1)
    build_demo()
