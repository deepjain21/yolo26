"""Runtime settings shared by the API, pipeline, and rule engine. All fields are live-updatable."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ModelName = Literal["people-detection", "yolo26n", "yolo26s", "yolo26m", "yolo11n", "yolo11s", "yolo11m"]


class Settings(BaseModel):
    # Rules
    min_people: int = Field(1, ge=0, description="Alert if fewer people than this are in frame (0 disables)")
    max_people: int = Field(0, ge=0, description="Alert if more people than this are in frame (0 disables)")
    lights_required: bool = Field(True, description="Alert when lights are detected as OFF")
    grace_seconds: float = Field(3.0, ge=0, le=60, description="Condition must persist this long before alerting")

    # Lighting
    lights_on_threshold: float = Field(45.0, ge=0, le=100, description="Brightness % above which lights are ON")
    lights_off_threshold: float = Field(30.0, ge=0, le=100, description="Brightness % below which lights are OFF")
    lighting_smoothing_seconds: float = Field(1.0, ge=0, le=10)
    light_gain: float = Field(1.0, ge=0.0, le=2.0, description=(
        "Simulated lighting: frames are multiplied by this before analysis (1 = as recorded). "
        "Lets you test lights-off detection on footage where the lights never change"))

    # Model
    model: ModelName = Field("people-detection", description=(
        "people-detection = Ultralytics Platform crowd model (YOLO26x fine-tuned on MOT20 at 1280, person-only), most "
        "accurate but ~2 s/frame on CPU; yolo26n = fastest generic COCO model; s/m = better on small people. "
        "YOLO11 kept for comparison"))
    conf: float = Field(0.25, ge=0.05, le=0.95, description="Minimum score for a detection to start/keep a track")
    imgsz: int = Field(640, ge=320, le=1280, description="Inference size; raise for small/far people")
    process_every_n: int = Field(2, ge=1, le=10, description="Run the detector on every Nth frame")

    def normalized(self) -> "Settings":
        """Keep thresholds consistent (off < on)."""
        if self.lights_off_threshold > self.lights_on_threshold:
            self.lights_off_threshold = self.lights_on_threshold
        return self
