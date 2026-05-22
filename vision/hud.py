# vision/hud.py
"""
Robust HUD signal reader for Minecraft.

The original ``_read_bar`` in processing.py counted RGB pixels matching a
single target colour against the entire bar ROI. That breaks for three
reasons:

  1. Icons are sparse on the bar — a *full* health bar is only ~30 % red
     pixels (the rest is heart-border black, drop shadow, and whatever
     gameplay shows through behind the panel). Counting raw pixels gives
     a meaningless fraction.
  2. RGB targets are fragile. Lighting (day/night/Nether/End), saturation
     effects, and biome ambient changes shift the rendered colour by 30 +
     units. Tight RGB tolerances miss the bar; loose ones catch noise.
  3. Status effects (poison → green hearts, wither → black hearts) and
     biome tints can change the icon palette entirely.

This module instead:

  * works in HSV space (so brightness shifts don't drop matches),
  * subdivides each bar into its 10 fixed icon slots and checks each slot
    independently — so a half-full bar gives an honest 0.5, not whatever
    fraction the pixel-counter happened to land on,
  * exposes named HSV ranges per bar that are easy to tune.

For the XP bar (which is continuous, not discrete icons) it finds the
right-most filled column and reports the fraction of width filled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# HSV ranges
# ---------------------------------------------------------------------------
#
# Hue is on OpenCV's 0..180 scale (not 0..360). Saturation and Value are
# 0..255. Each entry is (low, high). For "red" we need both ends of the
# hue circle and combine the masks.

@dataclass(frozen=True)
class HSVRange:
    low: Tuple[int, int, int]
    high: Tuple[int, int, int]

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        return cv2.inRange(
            hsv,
            np.array(self.low,  dtype=np.uint8),
            np.array(self.high, dtype=np.uint8),
        )


# Tuned against the captured 1.21.11 frame. Validated to give a clean
# binary mask on both daylight and night scenes — Mojang's HUD icons are
# rendered without ambient occlusion, so HSV alone separates them from
# any gameplay background.
RED_LOW       = HSVRange((  0, 120, 140), ( 12, 255, 255))
RED_HIGH      = HSVRange((170, 120, 140), (180, 255, 255))
HUNGER_BROWN  = HSVRange((  0,  80,  70), ( 20, 255, 220))
ARMOR_FILLED  = HSVRange((  0,   0, 150), (180,  40, 255))  # bright steel
XP_GREEN      = HSVRange(( 35, 120, 150), ( 85, 255, 255))


def _red_mask(hsv: np.ndarray) -> np.ndarray:
    return cv2.bitwise_or(RED_LOW.mask(hsv), RED_HIGH.mask(hsv))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HUDReaderConfig:
    # Each discrete bar (health / hunger / armor) has this many slots.
    bar_slots: int = 10

    # UI scale — used to derive the in-bar geometry of icons from MC's
    # known dimensions (9 GUI px wide, 8 GUI px stride). Passed in via
    # build_hud_reader from capture.ui_scale.
    ui_scale: int = 2

    # MC HUD icon geometry at GUI scale 1. Don't override unless Mojang
    # changes the texture atlas.
    icon_width_gui:  int = 9
    icon_stride_gui: int = 8

    # Sample only the central (icon_stride_gui - 1) px of each icon so
    # the 1 px of overlap with the next icon doesn't leak into the
    # neighbour's reading.
    icon_core_inset_gui: int = 1

    # Per-bar (full, half) thresholds. Each icon shape covers a different
    # fraction of its slot — hearts are dense (~50 %), shanks are
    # sparser (~25 %), chestplates are dense (~50 %). The values below
    # were measured on captures from vanilla 1.21.11 and work across
    # any UI scale because the mask is computed from a fixed-geometry
    # sample window. ``half`` MUST be strictly less than ``full``.
    bar_thresholds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "health": (0.30, 0.12),
        "hunger": (0.18, 0.08),
        "armor":  (0.30, 0.25),
    })

    # Fallback thresholds if a bar is not in bar_thresholds.
    default_full_threshold: float = 0.30
    default_half_threshold: float = 0.12


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class HUDSnapshot:
    health: float = 0.0
    hunger: float = 0.0
    armor:  float = 0.0
    xp_bar: float = 0.0

    # Per-bar diagnostics so callers can flag low confidence (e.g. when a
    # region is mis-calibrated or a status effect changes the palette).
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class HUDReader:
    """
    Reads health / hunger / armor / XP from a frame, given the HUD region
    rectangles (x, y, w, h) calibrated for the current resolution.

    Construct once per session — HSV ranges and slot geometry don't
    change between frames.
    """

    def __init__(self,
                 hud_regions: Dict[str, Any],
                 config: Optional[HUDReaderConfig] = None):
        self.hud = hud_regions or {}
        self.cfg = config or HUDReaderConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self, frame: np.ndarray) -> HUDSnapshot:
        snap = HUDSnapshot()

        health_val, health_diag = self._read_icon_bar(
            frame, "health_bar", mask_fn=_red_mask, bar_name="health",
        )
        hunger_val, hunger_diag = self._read_icon_bar(
            frame, "hunger_bar", mask_fn=HUNGER_BROWN.mask, bar_name="hunger",
        )
        armor_val,  armor_diag = self._read_icon_bar(
            frame, "armor_bar", mask_fn=ARMOR_FILLED.mask, bar_name="armor",
        )
        xp_val,     xp_diag = self._read_xp_bar(frame)

        snap.health = health_val
        snap.hunger = hunger_val
        snap.armor  = armor_val
        snap.xp_bar = xp_val
        snap.diagnostics = {
            "health": health_diag,
            "hunger": hunger_diag,
            "armor":  armor_diag,
            "xp_bar": xp_diag,
        }
        return snap

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_roi(self, frame: np.ndarray, key: str) -> Optional[np.ndarray]:
        region = self.hud.get(key)
        if not (isinstance(region, (list, tuple)) and len(region) == 4):
            return None
        x, y, w, h = (int(v) for v in region)
        if w <= 0 or h <= 0:
            return None
        H, W = frame.shape[:2]
        x = max(0, x);  y = max(0, y)
        x1 = min(W, x + w);  y1 = min(H, y + h)
        if x1 <= x or y1 <= y:
            return None
        return frame[y:y1, x:x1]

    def _read_icon_bar(self,
                       frame: np.ndarray,
                       key: str,
                       *,
                       mask_fn,
                       bar_name: str = "") -> Tuple[float, Dict[str, Any]]:
        """
        Count "filled" slots in a 10-slot discrete bar (health / hunger /
        armor). Returns (fraction in [0,1], diagnostics dict).

        Icon geometry follows Mojang's HUD atlas: each icon is
        ``icon_width_gui`` (9) pixels wide and packed every
        ``icon_stride_gui`` (8) pixels, so neighbouring icons overlap by 1
        pixel. We sample only the CENTRAL slice of each icon to keep the
        neighbour's body out of the reading.
        """
        roi = self._get_roi(frame, key)
        if roi is None or roi.size == 0:
            return 0.0, {"reason": "missing_roi"}

        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        mask = mask_fn(hsv)

        n     = self.cfg.bar_slots
        scale = max(1, int(self.cfg.ui_scale))
        h, w  = mask.shape

        stride_px = self.cfg.icon_stride_gui * scale
        width_px  = self.cfg.icon_width_gui  * scale
        inset_px  = self.cfg.icon_core_inset_gui * scale
        core_w    = max(1, width_px - 2 * inset_px)

        if w < stride_px * (n - 1) + width_px:
            # Bar is narrower than expected — fall back to whole-mask ratio
            ratio = float(mask.mean() / 255.0)
            return min(1.0, ratio), {
                "reason": "narrow_roi",
                "expected_width": stride_px * (n - 1) + width_px,
                "actual_width":   w,
                "raw_ratio":      ratio,
            }

        per_slot = np.zeros(n, dtype=np.float32)
        for i in range(n):
            x0 = i * stride_px + inset_px
            x1 = min(w, x0 + core_w)
            if x1 <= x0:
                continue
            per_slot[i] = float(mask[:, x0:x1].mean() / 255.0)

        full_thr, half_thr = self.cfg.bar_thresholds.get(
            bar_name,
            (self.cfg.default_full_threshold, self.cfg.default_half_threshold),
        )
        full = int(np.sum(per_slot >= full_thr))
        half = int(np.sum((per_slot >= half_thr) & (per_slot < full_thr)))
        filled = full + 0.5 * half
        value  = min(1.0, filled / n)

        return value, {
            "per_slot": per_slot.tolist(),
            "full": full,
            "half": half,
            "stride_px": stride_px,
            "core_w":    core_w,
            "full_threshold": full_thr,
            "half_threshold": half_thr,
            "mask_overall_ratio": float(mask.mean() / 255.0),
        }

    def _read_xp_bar(self,
                     frame: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        roi = self._get_roi(frame, "xp_bar")
        if roi is None or roi.size == 0:
            return 0.0, {"reason": "missing_roi"}

        hsv  = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        mask = XP_GREEN.mask(hsv)
        col_filled = mask.any(axis=0)
        if not col_filled.any():
            return 0.0, {"reason": "no_green",
                         "mask_overall_ratio": float(mask.mean() / 255.0)}
        rightmost = int(np.where(col_filled)[0][-1])
        ratio = min(1.0, (rightmost + 1) / max(1, mask.shape[1]))
        return ratio, {
            "rightmost_col": rightmost,
            "bar_width":      int(mask.shape[1]),
            "mask_overall_ratio": float(mask.mean() / 255.0),
        }


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_hud_reader(settings: Dict[str, Any]) -> HUDReader:
    """Wire a HUDReader from project settings."""
    vision = (settings or {}).get("vision", {}) or {}
    cap    = (settings or {}).get("capture", {}) or {}
    hud = dict(vision.get("hud_regions", {}) or {})
    # Make extended slot rects accessible (used by FrameProcessor's hotbar
    # detector — but kept here so a single HUD reader can serve both).
    hud["hud_extended"] = vision.get("hud_extended", {}) or {}

    cfg = HUDReaderConfig(ui_scale=int(cap.get("ui_scale", 2)))
    return HUDReader(hud_regions=hud, config=cfg)
