"""Lights on/off classification from frame luminance with smoothing + hysteresis."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LightingReading:
    brightness: float      # 0-100, smoothed mean V channel
    raw_brightness: float  # unsmoothed
    dark_ratio: float      # fraction of pixels with V < 40 (0-1)
    lights_on: bool


class LightingAnalyzer:
    def __init__(self) -> None:
        self._ema: float | None = None
        self._lights_on: bool = True

    def reset(self) -> None:
        self._ema = None
        self._lights_on = True

    def update(self, frame: np.ndarray, fps: float, on_threshold: float, off_threshold: float,
               smoothing_seconds: float) -> LightingReading:
        h, w = frame.shape[:2]
        scale = 160 / max(w, 1)
        small = cv2.resize(frame, (160, max(int(h * scale), 1)), interpolation=cv2.INTER_AREA)
        v = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 2]
        raw = float(v.mean()) / 255.0 * 100.0
        dark_ratio = float((v < 40).mean())

        window = max(smoothing_seconds * max(fps, 1.0), 1.0)
        alpha = 2.0 / (window + 1.0)
        self._ema = raw if self._ema is None else (alpha * raw + (1 - alpha) * self._ema)
        b = self._ema

        # Hysteresis: only flip when clearly past a threshold.
        if self._lights_on and b < off_threshold:
            self._lights_on = False
        elif not self._lights_on and b > on_threshold:
            self._lights_on = True

        return LightingReading(brightness=round(b, 1), raw_brightness=round(raw, 1),
                               dark_ratio=round(dark_ratio, 3), lights_on=self._lights_on)
