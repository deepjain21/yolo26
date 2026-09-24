"""Background video pipeline: source -> YOLO track -> lighting -> rules -> annotated JPEG + metrics."""
from __future__ import annotations

import base64
import sys
from collections import deque
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .detector import PersonTracker, Track
from .lighting import LightingAnalyzer
from .rules import RuleEngine
from .state import SharedState

GREEN = (80, 200, 120)
RED = (60, 60, 230)
WHITE = (255, 255, 255)
DARK = (20, 20, 20)


class VideoPipeline(threading.Thread):
    def __init__(self, state: SharedState, tracker: PersonTracker, source: str | int, label: str,
                 loop: bool = True, max_speed: bool = False) -> None:
        super().__init__(daemon=True, name="video-pipeline")
        self.state = state
        self.tracker = tracker
        self.source = source
        self.label = label
        self.loop = loop
        self.max_speed = max_speed
        self.is_webcam = isinstance(source, int)
        self._stop_evt = threading.Event()
        self.lighting = LightingAnalyzer()
        self.rules = RuleEngine(on_event=self._on_event)
        self.error: str | None = None
        self._last_snapshot: str | None = None
        self._seek_to: float | None = None  # video seconds requested by /api/seek, consumed by the loop
        self.duration = 0.0
        self.video_time = 0.0

    def stop(self) -> None:
        self._stop_evt.set()

    def seek(self, seconds: float | None = None, delta: float | None = None) -> float:
        """Request a jump to an absolute time or relative offset (files only). Returns the clamped target."""
        if self.is_webcam:
            raise ValueError("Cannot seek a live webcam")
        base = self._seek_to if self._seek_to is not None else self.video_time
        target = seconds if seconds is not None else base + (delta or 0.0)
        target = max(target, 0.0)
        if self.duration > 0:
            target = min(target, max(self.duration - 0.2, 0.0))
        self._seek_to = target
        return target

    # ------------------------------------------------------------------
    def _on_event(self, ev: dict) -> None:
        self.state.add_event(ev)

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source)
        if self.is_webcam:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {self.label}")
        return cap

    def run(self) -> None:
        try:
            cap = self._open()
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
            self.state.add_event({"event": "ERROR", "type": "SOURCE", "label": "Source error", "detail": str(e),
                                  "video_time": 0.0, "wall_time": time.time(), "snapshot": None})
            self.state.mark_stopped()
            return

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not (1.0 <= src_fps <= 120.0):
            src_fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / src_fps if (total_frames > 0 and not self.is_webcam) else 0.0
        self.duration = duration
        self.state.reset_session(self.label, duration)
        self.tracker.reset()
        self.lighting.reset()
        self.rules.reset()

        frame_idx = 0
        last_tracks: list[Track] = []
        stamps: deque[float] = deque()  # wall-clock times of frames published in the last 2 s -> throughput
        dropped = 0
        processed = 0
        t_start = time.monotonic()
        try:
            while not self._stop_evt.is_set():
                settings = self.state.get_settings()
                if settings.model != self.tracker.model_name:
                    self.state.set_status_note(f"Loading {settings.model}")
                    try:
                        self.tracker.load(settings.model)
                    except Exception as e:  # noqa: BLE001
                        self.state.add_event({"event": "ERROR", "type": "MODEL", "label": "Model error", "detail": str(e),
                                              "video_time": 0.0, "wall_time": time.time(), "snapshot": None})
                        self.state.update_settings({"model": self.tracker.model_name})
                    self.state.set_status_note(None)
                    t_start = time.monotonic() - frame_idx / src_fps  # do not count the load as lag
                self.tracker.set_conf(settings.conf)

                # Seek requested from the dashboard: reposition, then forget stale tracks, timers and smoothing.
                if self._seek_to is not None and not self.is_webcam:
                    target, self._seek_to = self._seek_to, None
                    self.rules.clear(self.video_time, "Playback position changed")
                    frame_idx = int(target * src_fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    t_start = time.monotonic() - frame_idx / src_fps
                    self.tracker.reset_tracks()
                    self.lighting.reset()
                    last_tracks = []
                    processed = 0

                # Real-time playback on a slow CPU: if we are behind schedule, skip (grab without decoding) frames.
                if not self.is_webcam and not self.max_speed:
                    behind = int((time.monotonic() - t_start) * src_fps) - frame_idx
                    skipped = 0
                    while behind > 1 and skipped < 15 and cap.grab():
                        frame_idx += 1
                        dropped += 1
                        behind -= 1
                        skipped += 1

                ok, frame = cap.read()
                if not ok:
                    if self.loop and not self.is_webcam:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_idx = 0
                        t_start = time.monotonic()
                        self.tracker.reset()
                        self.state.clear_history()  # timeline restarts with the clip
                        continue
                    break

                video_time = (time.monotonic() - t_start) if self.is_webcam else frame_idx / src_fps
                self.video_time = video_time

                # Simulated lighting: scale pixel values so both the detector and the lighting analyzer see the
                # dimmed/brightened room, exactly as a camera would.
                gain = settings.light_gain
                if abs(gain - 1.0) > 1e-3:
                    frame = cv2.convertScaleAbs(frame, alpha=gain, beta=0)

                if processed % settings.process_every_n == 0:
                    last_tracks = self.tracker.track(frame, settings.imgsz, video_time)
                processed += 1
                tracks = last_tracks
                light = self.lighting.update(frame, src_fps, settings.lights_on_threshold,
                                             settings.lights_off_threshold, settings.lighting_smoothing_seconds)
                people = len(tracks)

                # snapshot for events: small JPEG of the *annotated* previous frame is good enough
                active = self.rules.evaluate(settings, people, light.lights_on, video_time, self._last_snapshot)
                status = "ALERT" if active else "OK"

                now = time.monotonic()
                stamps.append(now)
                while stamps and now - stamps[0] > 2.0:
                    stamps.popleft()
                fps_ema = len(stamps) / min(max(now - stamps[0], 0.25), 2.0) if len(stamps) > 1 else 0.0

                annotated = self._draw(frame, tracks, light.brightness, light.lights_on, people, status, active,
                                       video_time, fps_ema, gain)
                ok_j, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                jpeg = buf.tobytes() if ok_j else b""
                if frame_idx % 5 == 0:
                    thumb = cv2.resize(annotated, (320, int(320 * annotated.shape[0] / annotated.shape[1])))
                    ok_t, tb = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    if ok_t:
                        self._last_snapshot = "data:image/jpeg;base64," + base64.b64encode(tb).decode()

                metrics = {
                    "running": True, "source": self.label, "people_now": people,
                    "unique_total": self.tracker.unique_total,
                    "active_ids": sorted(t.track_id for t in tracks if t.track_id >= 0),
                    "brightness": light.brightness, "raw_brightness": light.raw_brightness,
                    "dark_ratio": light.dark_ratio, "lights_on": light.lights_on,
                    "fps": round(fps_ema, 1), "frame_idx": frame_idx, "video_time": round(video_time, 1),
                    "status": status, "active_alerts": active, "duration": round(duration, 1),
                    "progress": round(video_time / duration, 3) if duration else 0.0,
                    "webcam": self.is_webcam, "dropped": dropped, "model": self.tracker.model_name,
                    "skip_ratio": round(dropped / max(dropped + processed, 1), 2),
                    "seekable": not self.is_webcam and duration > 0, "light_gain": gain,
                }
                self.state.publish_frame(jpeg, metrics)
                frame_idx += 1

                if not self.is_webcam and not self.max_speed:
                    # pace playback to the source frame rate
                    target = t_start + frame_idx / src_fps
                    delay = target - time.monotonic()
                    if delay > 0:
                        self._stop_evt.wait(delay)
                    elif delay < -2.0:
                        t_start = time.monotonic() - frame_idx / src_fps  # fell behind; resync clock
        finally:
            cap.release()
            self.state.mark_stopped()

    # ------------------------------------------------------------------
    def _draw(self, frame: np.ndarray, tracks: list[Track], brightness: float, lights_on: bool, people: int,
              status: str, active: list[dict], video_time: float, fps: float, gain: float = 1.0) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        scale = max(w / 1280, 0.6)
        thick = max(int(2 * scale), 1)
        for t in tracks:
            color = GREEN if status == "OK" else RED
            cv2.rectangle(out, (t.x1, t.y1), (t.x2, t.y2), color, thick)
            label = f"#{t.track_id} {t.conf:.2f}" if t.track_id >= 0 else f"{t.conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, thick)
            cv2.rectangle(out, (t.x1, max(t.y1 - th - 8, 0)), (t.x1 + tw + 8, t.y1), color, -1)
            cv2.putText(out, label, (t.x1 + 4, max(t.y1 - 5, th)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, DARK,
                        thick, cv2.LINE_AA)

        # status bar
        bar_h = int(34 * scale) + 8
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), GREEN if status == "OK" else RED, -1)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
        text = "ALL CLEAR" if status == "OK" else "ALERT: " + " | ".join(a["label"].upper() for a in active)
        cv2.putText(out, text, (10, bar_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, WHITE, thick + 1,
                    cv2.LINE_AA)
        info = (f"People {people}  |  Lights {'ON' if lights_on else 'OFF'} ({brightness:.0f}%)  |  "
                f"{video_time:6.1f}s  |  {fps:.1f} fps")
        if abs(gain - 1.0) > 1e-3:
            info += f"  |  simulated light {gain * 100:.0f}%"
        (iw, ih), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, thick)
        cv2.rectangle(out, (0, h - ih - 16), (iw + 16, h), DARK, -1)
        cv2.putText(out, info, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, WHITE, thick, cv2.LINE_AA)
        return out


def _cli(path: str) -> None:
    """Smoke test: python -m app.pipeline samples/x.mp4"""
    state = SharedState()
    tracker = PersonTracker()
    p = VideoPipeline(state, tracker, path, Path(path).name, loop=False, max_speed=True)
    p.start()
    last_t = -1.0
    while p.is_alive():
        time.sleep(0.25)
        m = state.snapshot()["metrics"]
        if m.get("video_time", 0) - last_t >= 1.0:
            last_t = m["video_time"]
            print(f"t={m['video_time']:6.1f}s people={m['people_now']} unique={m['unique_total']} "
                  f"bright={m['brightness']:5.1f} lights={'ON' if m['lights_on'] else 'OFF'} "
                  f"status={m['status']:5} fps={m['fps']}")
        for ev in state.drain_events():
            print(f"  EVENT {ev['event']} {ev['type']} @ {ev['video_time']}s: {ev['detail']}")
    if p.error:
        print("ERROR:", p.error)


if __name__ == "__main__":
    _cli(sys.argv[1])
