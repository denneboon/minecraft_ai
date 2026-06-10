# control/mouse_calibration.py
"""
Mouse-to-camera-angle auto-calibration.

What this is
------------
Minecraft's "look sensitivity" plus the system mouse DPI plus any
raw-input scaling combine to a fixed ratio:

    pixels_of_mouse_motion = K_yaw * delta_yaw_degrees
    pixels_of_mouse_motion = K_pitch * delta_pitch_degrees

Where ``K_yaw`` and ``K_pitch`` are constants we cannot read directly
from the game — but we CAN observe them by sending a known mouse
delta and watching the resulting yaw / pitch change in the F3
overlay. Once measured, the agent's aim math is exact.

We learn ``K`` lazily, from real agent traffic:

  * Each tick the agent emits ``(dx, dy)`` mouse pixels.
  * Each F3 read gives us a fresh ``(yaw, pitch)``.
  * Subtracting consecutive yaw/pitch readings gives the angular
    motion that was driven by the mouse pixels sent in between.
  * We accumulate (px_emitted, deg_observed) pairs and take the
    median ratio. Median is robust to OCR jitter / outliers.

Why we don't just use the configured ``mouse_per_degree``
---------------------------------------------------------
The default 6.5 px / ° was tuned on one specific machine. It can be
off by ±50 % under different MC sensitivity settings or display
scaling. An auto-calibrated value lets the agent aim correctly
without anyone editing a config.

Persistence
-----------
The latest learned values write to
``data/calibration/mouse_calibration.json`` so successive runs reuse
the calibration without redoing it. A first-time run starts from the
config default until enough samples accumulate.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Optional, Tuple
from collections import deque


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MouseCalibrationConfig:
    # Default px-per-degree if we have no data yet. Matches the
    # historic ``mouse_per_degree`` constant for backwards
    # compatibility.
    default_px_per_deg: float = 6.5

    # We keep this many recent (px, deg) samples and use their MEDIAN
    # ratio as the current estimate. Median is robust to OCR
    # outliers — a single mis-parsed yaw of e.g. 200° (frame on top
    # of a foliage texture) doesn't bias the estimate.
    sample_window: int = 64

    # Minimum samples before we trust the calibrated value over the
    # default. Below this we use the default and slowly accumulate.
    min_samples_to_trust: int = 8

    # Reject samples whose px-per-degree falls outside this range.
    # Tighter bounds = more accurate but less data. The default
    # 1.0–30.0 covers every reasonable MC sensitivity × DPI combo.
    sane_px_per_deg_min: float = 1.0
    sane_px_per_deg_max: float = 30.0

    # Don't bother learning from tiny angular changes; they amplify
    # any OCR jitter to nonsense ratios.
    min_observed_degrees: float = 0.7

    # And don't learn from tiny mouse deltas either — too little
    # signal-to-noise.
    min_px_delta: float = 5.0


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

@dataclass
class MouseCalibrator:
    cfg: MouseCalibrationConfig = field(default_factory=MouseCalibrationConfig)

    # Disk-backed storage path. None = no persistence.
    persist_path: Optional[str] = None

    # Mouse-emitted accumulator since the last absorbed yaw / pitch.
    # When we receive a fresh pose, we use these and reset.
    _accum_dx: float = 0.0
    _accum_dy: float = 0.0
    _last_yaw:   Optional[float] = None
    _last_pitch: Optional[float] = None

    _yaw_samples:   Deque[float] = field(default_factory=lambda: deque(maxlen=64))
    _pitch_samples: Deque[float] = field(default_factory=lambda: deque(maxlen=64))

    # Cached estimates (lazy-updated when samples change).
    _px_per_deg_yaw:   Optional[float] = None
    _px_per_deg_pitch: Optional[float] = None

    # ── Public API ────────────────────────────────────────────────

    def __post_init__(self) -> None:
        # Replace the deques with the right max length from config.
        self._yaw_samples   = deque(maxlen=self.cfg.sample_window)
        self._pitch_samples = deque(maxlen=self.cfg.sample_window)
        self._try_load()

    def px_per_deg_yaw(self) -> float:
        """Current best estimate of pixels-per-degree for yaw."""
        if (self._px_per_deg_yaw is not None
                and len(self._yaw_samples) >= self.cfg.min_samples_to_trust):
            return self._px_per_deg_yaw
        return self.cfg.default_px_per_deg

    def px_per_deg_pitch(self) -> float:
        """Current best estimate of pixels-per-degree for pitch."""
        if (self._px_per_deg_pitch is not None
                and len(self._pitch_samples) >= self.cfg.min_samples_to_trust):
            return self._px_per_deg_pitch
        return self.cfg.default_px_per_deg

    def is_calibrated(self) -> bool:
        """Whether we have enough data to trust the calibrated values."""
        return (len(self._yaw_samples) >= self.cfg.min_samples_to_trust
                and len(self._pitch_samples) >= self.cfg.min_samples_to_trust)

    def stats(self) -> dict:
        return {
            "yaw_samples":      len(self._yaw_samples),
            "pitch_samples":    len(self._pitch_samples),
            "px_per_deg_yaw":   round(self.px_per_deg_yaw(), 3),
            "px_per_deg_pitch": round(self.px_per_deg_pitch(), 3),
            "calibrated":       self.is_calibrated(),
        }

    # Plan a mouse motion to achieve a target angular change.
    def pixels_for_degrees(self, d_yaw: float, d_pitch: float
                            ) -> Tuple[int, int]:
        """Return (dx_px, dy_px) needed to rotate by ``d_yaw`` /
        ``d_pitch`` degrees. Uses the current calibrated estimates.

        MC sign convention reminder:
          * positive mouse dx = yaw increases (turn right)
          * positive mouse dy = pitch increases (look down)
        """
        return (
            int(round(d_yaw   * self.px_per_deg_yaw())),
            int(round(d_pitch * self.px_per_deg_pitch())),
        )

    # ── Sample collection ────────────────────────────────────────

    def emitted(self, dx: float, dy: float) -> None:
        """Tell the calibrator about mouse motion we just emitted."""
        self._accum_dx += float(dx)
        self._accum_dy += float(dy)

    def observed_pose(self, yaw: float, pitch: float) -> None:
        """
        Tell the calibrator about a fresh F3 pose reading. If we've
        emitted mouse motion since the previous pose, learn a sample
        from the pair.
        """
        if self._last_yaw is None or self._last_pitch is None:
            self._last_yaw = float(yaw)
            self._last_pitch = float(pitch)
            return

        d_yaw   = _norm_angle_deg(float(yaw)   - self._last_yaw)
        d_pitch = float(pitch) - self._last_pitch
        # Pitch in MC is clamped to ±90 (never wraps), so straight
        # subtraction is fine. Yaw wraps at ±180 — handled above.

        self._maybe_add_yaw_sample(self._accum_dx, d_yaw)
        self._maybe_add_pitch_sample(self._accum_dy, d_pitch)

        self._last_yaw   = float(yaw)
        self._last_pitch = float(pitch)
        self._accum_dx = 0.0
        self._accum_dy = 0.0

    # ── Internals ────────────────────────────────────────────────

    def _maybe_add_yaw_sample(self, px: float, deg: float) -> None:
        if abs(px) < self.cfg.min_px_delta:
            return
        if abs(deg) < self.cfg.min_observed_degrees:
            return
        # Signs must agree — if the mouse went right but yaw decreased,
        # something else interfered (player typed, gate closed) and the
        # sample is useless.
        if (px > 0) != (deg > 0):
            return
        ratio = abs(px) / abs(deg)
        if (ratio < self.cfg.sane_px_per_deg_min
                or ratio > self.cfg.sane_px_per_deg_max):
            return
        self._yaw_samples.append(ratio)
        self._px_per_deg_yaw = _median(self._yaw_samples)
        self._try_save()

    def _maybe_add_pitch_sample(self, px: float, deg: float) -> None:
        if abs(px) < self.cfg.min_px_delta:
            return
        if abs(deg) < self.cfg.min_observed_degrees:
            return
        if (px > 0) != (deg > 0):
            return
        ratio = abs(px) / abs(deg)
        if (ratio < self.cfg.sane_px_per_deg_min
                or ratio > self.cfg.sane_px_per_deg_max):
            return
        self._pitch_samples.append(ratio)
        self._px_per_deg_pitch = _median(self._pitch_samples)
        self._try_save()

    # ── Persistence ──────────────────────────────────────────────

    def _try_load(self) -> None:
        if not self.persist_path or not os.path.isfile(self.persist_path):
            return
        try:
            data = json.loads(Path(self.persist_path).read_text(encoding="utf-8"))
        except Exception:
            return
        for v in data.get("yaw_samples", []):
            try:
                self._yaw_samples.append(float(v))
            except Exception:
                continue
        for v in data.get("pitch_samples", []):
            try:
                self._pitch_samples.append(float(v))
            except Exception:
                continue
        if self._yaw_samples:
            self._px_per_deg_yaw = _median(self._yaw_samples)
        if self._pitch_samples:
            self._px_per_deg_pitch = _median(self._pitch_samples)

    def _try_save(self) -> None:
        if not self.persist_path:
            return
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            data = {
                "px_per_deg_yaw":   self._px_per_deg_yaw,
                "px_per_deg_pitch": self._px_per_deg_pitch,
                "yaw_samples":      list(self._yaw_samples),
                "pitch_samples":    list(self._pitch_samples),
            }
            # Atomic write (tmp + fsync + replace) so a crash / power-loss
            # mid-write can't leave a truncated JSON the loader silently
            # discards (which would cold-start calibration). The helper
            # also retries the replace through Windows' transient
            # PermissionError (AV / indexer briefly holding the file).
            from utils.atomic import atomic_write_text
            atomic_write_text(self.persist_path, json.dumps(data, indent=2))
        except Exception as e:
            # First-failure log so we notice silently-failing persistence
            # (permission denied on the calibration dir, disk full, etc.).
            # Subsequent occurrences stay silent so the hot path doesn't
            # spam at 240 Hz worker rate × N agents.
            if not getattr(self, "_save_warn_emitted", False):
                self._save_warn_emitted = True
                print(f"[mouse_cal][WARN] persistence save failed: {e!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_angle_deg(d: float) -> float:
    """Normalise an angle to (-180, 180]."""
    while d >  180.0: d -= 360.0
    while d <= -180.0: d += 360.0
    return d


def _median(xs) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return float(s[n // 2])
    return float((s[n // 2 - 1] + s[n // 2]) / 2.0)


__all__ = ["MouseCalibrator", "MouseCalibrationConfig"]
