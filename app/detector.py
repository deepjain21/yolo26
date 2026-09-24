"""Person detector + tracker wrapper around Ultralytics YOLO with ByteTrack.

Why tracking runs at conf=0.1: ByteTrack is a two-stage matcher. High-score detections (>= track_high_thresh)
are matched first; low-score ones (>= track_low_thresh) are used only to keep existing tracks alive through
partial occlusion. Filtering detections at a high confidence *before* the tracker starves that second stage.
So the user-facing "confidence" becomes the tracker's high/new-track threshold, and the raw detector runs low.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PERSON_CLASS = 0
MIN_HITS_FOR_UNIQUE = 8   # a track must be seen this many times before it counts as a unique person
DETECTOR_CONF = 0.1       # raw detector threshold fed to ByteTrack (see module docstring)
TRACK_BUFFER_FRAMES = 60  # keep a lost track alive this many frames (~2 s at 30 fps) before dropping its ID
# Models hosted on Ultralytics Platform (public, no API key needed for the file listing):
#   name -> (owner, project, model). Generic yolo*.pt names are fetched by Ultralytics from GitHub releases.
PLATFORM_MODELS = {
    "people-detection": ("ultralytics", "solutions", "people-detection"),  # "Crowd Detection YOLO26x", MOT20, imgsz 1280
}
PLATFORM_API = "https://platform.ultralytics.com/api/models/{owner}/{project}/{model}/files"


@dataclass
class Track:
    x1: int
    y1: int
    x2: int
    y2: int
    track_id: int
    conf: float


class PersonTracker:
    def __init__(self, model_name: str = "people-detection") -> None:
        self.model = None
        self.model_name = ""
        self.conf = 0.25
        self.seen: dict[int, dict] = {}
        self._tracker_yaml = MODELS_DIR / "bytetrack_custom.yaml"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.load(model_name)

    # ------------------------------------------------------------ configuration
    def load(self, model_name: str) -> None:
        """Load (and download on first use) a YOLO26 or YOLO11 variant. Blocks for a few seconds."""
        from ultralytics import YOLO  # deferred: slow import

        if model_name == self.model_name and self.model is not None:
            return
        weights = MODELS_DIR / f"{model_name}.pt"
        if model_name in PLATFORM_MODELS and not weights.exists():
            self._download_platform_model(model_name, weights)
        self.model = YOLO(str(weights))
        self.model_name = model_name
        self.model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=640, classes=[PERSON_CLASS], verbose=False)
        self._write_tracker_cfg()

    @staticmethod
    def _download_platform_model(model_name: str, dest: Path) -> None:
        """Fetch weights for a public Ultralytics Platform model: the files endpoint returns a signed download URL."""
        import json
        import urllib.request

        owner, project, model = PLATFORM_MODELS[model_name]
        req = urllib.request.Request(PLATFORM_API.format(owner=owner, project=project, model=model),
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            files = json.load(r)["files"]
        if not files:
            raise FileNotFoundError(f"No weights published for {owner}/{project}/{model}")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(files[0]["downloadUrl"], tmp)
        tmp.replace(dest)

    def set_conf(self, conf: float) -> None:
        if abs(conf - self.conf) > 1e-6:
            self.conf = conf
            self._write_tracker_cfg()
            self._drop_tracker()  # tracker config is read once per predictor; rebuild it

    def _write_tracker_cfg(self) -> None:
        self._tracker_yaml.write_text(
            "tracker_type: bytetrack\n"
            f"track_high_thresh: {self.conf}\n"
            f"track_low_thresh: {min(DETECTOR_CONF, self.conf)}\n"
            f"new_track_thresh: {self.conf}\n"
            f"track_buffer: {TRACK_BUFFER_FRAMES}\n"
            "match_thresh: 0.8\n"
            "fuse_score: True\n"
        )

    def _drop_tracker(self) -> None:
        if self.model is not None and getattr(self.model, "predictor", None) is not None:
            self.model.predictor.trackers = []
            self.model.predictor = None

    def reset(self) -> None:
        """New source: forget unique people and restart track IDs at 1."""
        self.seen.clear()
        self._drop_tracker()

    def reset_tracks(self) -> None:
        """Playback jumped: drop live tracks (their positions are stale) but keep the unique-people tally."""
        self._drop_tracker()

    # ------------------------------------------------------------ inference
    def track(self, frame: np.ndarray, imgsz: int, video_time: float) -> list[Track]:
        results = self.model.track(frame, persist=True, classes=[PERSON_CLASS], conf=DETECTOR_CONF, iou=0.5,
                                   imgsz=imgsz, tracker=str(self._tracker_yaml), verbose=False)
        tracks: list[Track] = []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return tracks
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else np.full(len(xyxy), -1)
        for (x1, y1, x2, y2), c, tid in zip(xyxy, confs, ids):
            tid = int(tid)
            if tid < 0:
                continue  # ByteTrack only returns confirmed tracks; anything without an ID is noise
            rec = self.seen.setdefault(tid, {"first": video_time, "last": video_time, "hits": 0})
            rec["last"] = video_time
            rec["hits"] += 1
            tracks.append(Track(int(x1), int(y1), int(x2), int(y2), tid, float(c)))
        return tracks

    @property
    def unique_total(self) -> int:
        return sum(1 for r in self.seen.values() if r["hits"] >= MIN_HITS_FOR_UNIQUE)
