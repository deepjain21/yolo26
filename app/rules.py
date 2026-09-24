"""Rule engine: turns per-frame observations into alerts and events, with a grace period."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .settings import Settings

NO_PERSON = "NO_PERSON"
TOO_MANY_PEOPLE = "TOO_MANY_PEOPLE"
LIGHTS_OFF = "LIGHTS_OFF"

LABELS = {
    NO_PERSON: "No person in room",
    TOO_MANY_PEOPLE: "Too many people",
    LIGHTS_OFF: "Lights off",
}


@dataclass
class RuleState:
    violated_since: float | None = None  # video time when violation started
    active: bool = False
    detail: str = ""


@dataclass
class RuleEngine:
    on_event: Callable[[dict], None]
    states: dict[str, RuleState] = field(default_factory=lambda: {k: RuleState() for k in LABELS})
    _seq: int = 0

    def reset(self) -> None:
        self.states = {k: RuleState() for k in LABELS}

    def clear(self, video_time: float, reason: str) -> None:
        """Resolve every active alert (e.g. playback jumped) so the event log stays balanced, then restart timers."""
        for kind, st in self.states.items():
            if st.active:
                self._emit("ALERT_END", kind, reason, video_time, None)
        self.reset()

    def evaluate(self, settings: Settings, people: int, lights_on: bool, video_time: float,
                 snapshot: str | None) -> list[dict]:
        """Returns the list of currently active alerts."""
        conditions = {
            NO_PERSON: (settings.min_people > 0 and people < settings.min_people,
                        f"{people} in frame, minimum {settings.min_people}"),
            TOO_MANY_PEOPLE: (settings.max_people > 0 and people > settings.max_people,
                              f"{people} in frame, maximum {settings.max_people}"),
            LIGHTS_OFF: (settings.lights_required and not lights_on, "Room brightness below threshold"),
        }
        active: list[dict] = []
        for kind, (violated, detail) in conditions.items():
            st = self.states[kind]
            if violated:
                if st.violated_since is None:
                    st.violated_since = video_time
                elapsed = video_time - st.violated_since
                if not st.active and elapsed >= settings.grace_seconds:
                    st.active = True
                    st.detail = detail
                    self._emit("ALERT_START", kind, detail, video_time, snapshot)
            else:
                if st.active:
                    self._emit("ALERT_END", kind, f"Resolved after {video_time - (st.violated_since or video_time):.1f}s",
                               video_time, snapshot)
                st.violated_since = None
                st.active = False
            if st.active:
                active.append({"type": kind, "label": LABELS[kind], "detail": detail,
                               "since": round(st.violated_since or video_time, 1)})
        return active

    def _emit(self, event: str, kind: str, detail: str, video_time: float, snapshot: str | None) -> None:
        self._seq += 1
        self.on_event({
            "id": self._seq,
            "event": event,
            "type": kind,
            "label": LABELS[kind],
            "detail": detail,
            "video_time": round(video_time, 1),
            "wall_time": time.time(),
            "snapshot": snapshot if event == "ALERT_START" else None,
        })
