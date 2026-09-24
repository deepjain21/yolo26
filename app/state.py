"""Thread-safe shared state between the pipeline thread and the web layer."""
from __future__ import annotations

import threading
from collections import deque
from typing import Any

from .settings import Settings


class SharedState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame_cond = threading.Condition(self._lock)
        self.settings = Settings()
        self.jpeg: bytes | None = None
        self.frame_seq = 0
        self.metrics: dict[str, Any] = self._idle_metrics()
        self.history: deque[dict] = deque(maxlen=5 * 60 * 5)  # 5 min at 5 Hz
        self.events: deque[dict] = deque(maxlen=500)
        self.event_seq = 0
        self.pending_events: deque[dict] = deque()
        self.last_history_t = -1.0
        self.note: str | None = None

    def set_status_note(self, note: str | None) -> None:
        with self._lock:
            self.note = note
            self.metrics = {**self.metrics, "note": note}

    @staticmethod
    def _idle_metrics() -> dict[str, Any]:
        return {
            "running": False, "source": None, "people_now": 0, "unique_total": 0, "active_ids": [],
            "brightness": 0.0, "lights_on": True, "fps": 0.0, "frame_idx": 0, "video_time": 0.0,
            "status": "IDLE", "active_alerts": [], "duration": 0.0, "progress": 0.0,
        }

    # --- pipeline side ---
    def publish_frame(self, jpeg: bytes, metrics: dict[str, Any]) -> None:
        with self._frame_cond:
            self.jpeg = jpeg
            self.frame_seq += 1
            self.metrics = {**metrics, "note": self.note}
            t = metrics.get("video_time", 0.0)
            if t - self.last_history_t >= 0.2 or t < self.last_history_t:
                self.last_history_t = t
                self.history.append({"t": round(t, 1), "people": metrics["people_now"],
                                     "brightness": metrics["brightness"], "alert": metrics["status"] == "ALERT"})
            self._frame_cond.notify_all()

    def add_event(self, ev: dict) -> None:
        with self._lock:
            self.event_seq += 1
            ev = {**ev, "id": self.event_seq}
            self.events.appendleft(ev)
            self.pending_events.append(ev)

    def reset_session(self, source: str, duration: float) -> None:
        with self._lock:
            self.history.clear()
            self.events.clear()
            self.pending_events.clear()
            self.last_history_t = -1.0
            self.metrics = {**self._idle_metrics(), "running": True, "source": source,
                            "status": "STARTING", "duration": duration}

    def clear_history(self) -> None:
        with self._lock:
            self.history.clear()
            self.last_history_t = -1.0

    def mark_stopped(self) -> None:
        with self._lock:
            self.metrics = {**self.metrics, "running": False, "status": "IDLE", "fps": 0.0, "active_alerts": []}

    # --- web side ---
    def wait_for_frame(self, last_seq: int, timeout: float = 1.0) -> tuple[bytes | None, int]:
        with self._frame_cond:
            if self.frame_seq == last_seq:
                self._frame_cond.wait(timeout)
            return self.jpeg, self.frame_seq

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"metrics": dict(self.metrics), "settings": self.settings.model_dump()}

    def drain_events(self) -> list[dict]:
        with self._lock:
            evs = list(self.pending_events)
            self.pending_events.clear()
            return evs

    def events_since(self, last_id: int) -> list[dict]:
        """Events with id > last_id, oldest first (per-connection cursor, safe for many clients)."""
        with self._lock:
            return [e for e in reversed(self.events) if e["id"] > last_id]

    def history_list(self) -> list[dict]:
        with self._lock:
            return list(self.history)

    def events_list(self) -> list[dict]:
        with self._lock:
            return list(self.events)

    def get_settings(self) -> Settings:
        with self._lock:
            return self.settings.model_copy()

    def update_settings(self, patch: dict[str, Any]) -> Settings:
        with self._lock:
            merged = self.settings.model_dump()
            merged.update(patch)
            self.settings = Settings(**merged).normalized()
            return self.settings.model_copy()
