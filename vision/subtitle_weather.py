"""
Subtitle-based weather reader — the RELIABLE live weather signal.

Minecraft's "Show Subtitles" accessibility option captions sounds as rendered
text in the bottom-right ("Rain falls", "Thunder rumbles"). That's the SAME
clean MC-font text the F3 OCR already reads well, so detecting weather becomes
a near-100% TEXT-recognition problem instead of the brittle sky-colour guess
(``vision.weather`` mislabelled a clear forest ~64% "rain"). It also stays
pixel-only.

Robust to the captions stacking / moving: we OCR the whole bottom-right region
and scan for the keyword ANYWHERE in it (position-independent), and we keep a
short TIME WINDOW so an intermittent caption (thunder rumbles every few
seconds) is still "seen" between checks. Because the rain caption refreshes
continuously while it rains, a single occasional read reliably catches rain;
thunder is caught whenever a rumble is on screen (and thunderstorms also rain,
so they read at least as "rain" in between).

LIMITATION — snow is SILENT in Minecraft (no sound → no subtitle), so this
reader cannot see snow; it reads snow as "clear". Whether falling precipitation
is rain vs snow isn't a separate weather state at all — it's decided by the
local TEMPERATURE, i.e. the biome AND the altitude (temperature drops with
height, so the same "rain" weather falls as snow high up / in cold biomes).
So snow can't be read from the precip caption; it's handled by the COMMANDED
training label (``/weather rain`` while standing in a cold biome / high up →
``set_environment(weather="snow")``) and, live, would have to be inferred from
biome + Y. Rain + thunder — the states that change block appearance most — are
fully covered.

Returns a ``vision.world.types.WeatherObservation`` (source="subtitle") so it's
a drop-in for the existing weather plumbing in perception.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from vision.glyph_ocr import GlyphOCR, GlyphOCRConfig
from vision.world.types import WeatherObservation
from vision.weather import WeatherState


@dataclass
class SubtitleWeatherConfig:
    ui_scale: int = 2
    match_threshold: float = 0.62
    # Subtitle region (fractions of frame W/H). Captions render bottom-RIGHT,
    # stacking upward, above the hotbar. Generous so a pushed/stacked caption
    # still lands inside it. The caller can narrow it after a live look.
    x0_frac: float = 0.45
    x1_frac: float = 1.00
    y0_frac: float = 0.52
    y1_frac: float = 0.90
    y_step_px: int = 6
    read_budget_ms: int = 200
    # A keyword sighting stays "current" this long, so an intermittent caption
    # (thunder) read on one check still counts on the next. Generous because
    # checks are throttled (minutes apart) — see perception's cadence.
    window_sec: float = 90.0
    # Substrings (lower-case) that mark each precip state in a caption.
    rain_keywords: Tuple[str, ...] = ("rain",)
    thunder_keywords: Tuple[str, ...] = ("thunder",)
    # Confidence emitted for a positive caption match vs. an inferred "clear"
    # (absence of any precip caption — strong but not proof, since unusual HUD
    # layouts could in principle hide it). Both high enough to be trustworthy.
    match_confidence: float = 0.95
    clear_confidence: float = 0.85


class SubtitleWeatherReader:
    """Build once (templates load once), then call ``read(frame)`` per check.
    Holds a small time-window of recent caption sightings internally so a
    throttled, occasional call still resolves rain/thunder correctly."""

    def __init__(self,
                 templates: Dict[str, np.ndarray],
                 config: Optional[SubtitleWeatherConfig] = None):
        self.cfg = config or SubtitleWeatherConfig()
        self._ocr = GlyphOCR(
            templates=templates,
            config=GlyphOCRConfig(
                ui_scale=int(self.cfg.ui_scale),
                match_threshold=float(self.cfg.match_threshold),
            ),
        )
        self._last_rain_t: float = float("-inf")
        self._last_thunder_t: float = float("-inf")
        self._last_text: str = ""
        self._ocr_warn_emitted = False

    # -- OCR the bottom-right region into one lower-case string --------------
    def _read_region_text(self, frame: np.ndarray) -> str:
        h, w = frame.shape[:2]
        x0 = max(0, min(w - 1, int(w * self.cfg.x0_frac)))
        x1 = max(x0 + 1, min(w, int(w * self.cfg.x1_frac)))
        y0 = max(0, min(h - 1, int(h * self.cfg.y0_frac)))
        y1 = max(y0 + 1, min(h, int(h * self.cfg.y1_frac)))
        glyph_h = 8 * max(1, int(self.cfg.ui_scale))
        band_h = glyph_h + 6
        step = max(1, int(self.cfg.y_step_px))
        scan_rows = list(range(y0, y1 - band_h + 1, step))
        last_valid = y1 - band_h
        if scan_rows and scan_rows[-1] < last_valid:
            scan_rows.append(last_valid)
        elif not scan_rows and last_valid >= y0:
            scan_rows = [last_valid]

        budget_ms = float(self.cfg.read_budget_ms or 0)
        self._ocr.begin_read(budget_ms / 1000.0)
        deadline = (time.perf_counter() + budget_ms / 1000.0) if budget_ms > 0 else None
        chunks = []
        for y in scan_rows:
            if deadline is not None and time.perf_counter() > deadline:
                break
            crop = frame[y:y + band_h, x0:x1]
            try:
                text = self._ocr.recognize_line(crop)
            except (cv2.error, ValueError, IndexError, AttributeError) as e:
                if not self._ocr_warn_emitted:
                    self._ocr_warn_emitted = True
                    print(f"[subtitle_weather][WARN] OCR row y={y} failed: "
                          f"{e!r} — further errors silenced.")
                continue
            if text:
                chunks.append(text)
        return "\n".join(chunks).lower()

    def read(self, frame: np.ndarray,
             now: Optional[float] = None) -> WeatherObservation:
        """OCR the subtitle region, update the sighting window, and return the
        current weather. Pitch-independent (subtitles are HUD, always shown)."""
        if now is None:
            now = time.perf_counter()
        text = self._read_region_text(frame) if frame is not None else ""
        self._last_text = text
        saw_thunder = any(k in text for k in self.cfg.thunder_keywords)
        saw_rain = any(k in text for k in self.cfg.rain_keywords)
        if saw_thunder:
            self._last_thunder_t = now
        if saw_rain:
            self._last_rain_t = now

        thunder_recent = (now - self._last_thunder_t) <= self.cfg.window_sec
        rain_recent = (now - self._last_rain_t) <= self.cfg.window_sec
        if thunder_recent:
            state, conf = WeatherState.THUNDER, self.cfg.match_confidence
        elif rain_recent:
            state, conf = WeatherState.RAIN, self.cfg.match_confidence
        else:
            # No precip caption now or recently → clear. (Snow is silent and
            # would also land here — see module docstring.)
            state, conf = WeatherState.CLEAR, self.cfg.clear_confidence
        return WeatherObservation(
            state=state, confidence=round(float(conf), 3),
            sky_visible=1.0, source="subtitle",
            features={"caption": text[:80]},
        )


def build_subtitle_weather_reader(settings: Dict,
                                  templates: Dict[str, np.ndarray]
                                  ) -> SubtitleWeatherReader:
    cap = (settings or {}).get("capture", {}) or {}
    sub = (((settings or {}).get("vision", {}) or {})
           .get("subtitle_weather", {}) or {})
    cfg = SubtitleWeatherConfig(ui_scale=int(cap.get("ui_scale", 2)))
    for k, v in sub.items():
        if hasattr(cfg, k) and not isinstance(v, dict):
            try:
                setattr(cfg, k, type(getattr(cfg, k))(v))
            except (TypeError, ValueError):
                pass
    return SubtitleWeatherReader(templates=templates, config=cfg)
