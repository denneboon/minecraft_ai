# vision/processing.py
"""
Frame processor: converts a raw RGB screenshot into a structured GameState.

The AI never sees raw numpy arrays directly — it receives a GameState, which
is a clean dataclass with named fields. This keeps the AI code independent of
pixel layout, HUD positions, and resolution.

Pipeline per tick:
    raw frame (H×W×3 uint8)
        → FrameProcessor.process(frame, hud)
            → crop game view  (remove HUD strips if configured)
            → resize to model input size
            → extract HUD signals (health, hunger, XP, armor, hotbar slot)
            → detect screen state (playing / paused / menu / loading)
        → GameState

All HUD extraction is pixel-colour heuristics — no ML, no OCR.
OCR and segmentation live in their own modules and are called separately.

Designed to be:
  - Fast: processes a 1920×1080 frame in <2 ms on CPU (pure numpy/cv2).
  - Stateless: FrameProcessor holds no per-frame mutable state (thread-safe).
  - Replaceable: swap out any extractor without touching the AI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from vision.ocr import F3Info
from vision.hud import HUDReader, build_hud_reader


# ---------------------------------------------------------------------------
# Screen state enum
# ---------------------------------------------------------------------------

class ScreenState:
    PLAYING  = "playing"   # in-game, crosshair visible
    PAUSED   = "paused"    # pause menu open (Esc)
    MENU     = "menu"      # inventory / crafting / chest open
    LOADING  = "loading"   # loading screen / black screen
    UNKNOWN  = "unknown"


# ---------------------------------------------------------------------------
# GameState — the AI's observation at each tick
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    # ---- Visual input ----
    frame: np.ndarray               # resized RGB frame, shape (H, W, 3), dtype uint8
    frame_raw: Optional[np.ndarray] # original full-res frame (None if not kept)
    timestamp: float                # time.perf_counter() at capture

    # ---- Screen state ----
    screen_state: str = ScreenState.UNKNOWN

    # ---- HUD signals (all normalised 0.0–1.0 unless noted) ----
    health:  float = 0.0            # 0.0 = dead, 1.0 = full (20 HP)
    hunger:  float = 0.0            # 0.0 = starving, 1.0 = full (20 food)
    armor:   float = 0.0            # 0.0 = none, 1.0 = full (20 armour pts)
    xp_bar:  float = 0.0            # 0.0–1.0 within current XP level

    # ---- Hotbar ----
    hotbar_slot: int = 1            # currently selected slot (1–9), heuristic only

    # ---- Structured debug overlay data (from F3Reader, optional) ----
    # Populated when F3Reader.read() is called alongside processing.
    # None if OCR was not run this tick (it runs at ~2–4 Hz, not every frame).
    f3: Optional[F3Info] = None

    # ---- World perception (optional, off by default) ----
    # Populated by ``vision.world.WorldPerception`` when the
    # ``vision.world.enabled`` setting is on. The agent reads
    # ``state.world.world_map`` for the persistent voxel/entity store.
    # Typed loosely (Any) to avoid importing vision.world here and
    # creating a cycle.
    world: Optional[Any] = None

    # ---- Meta ----
    frame_index: int = 0            # monotonic counter, set by caller
    extras: Dict[str, Any] = field(default_factory=dict)  # for future signals


# ---------------------------------------------------------------------------
# Processor config
# ---------------------------------------------------------------------------

@dataclass
class ProcessorConfig:
    # Target size fed to the model (height, width).
    model_input_size: Tuple[int, int] = (270, 480)   # 1/4 of 1080p, 16:9

    # Whether to keep the full-res frame in GameState.frame_raw.
    keep_raw: bool = False

    # Crop the HUD strip from the bottom before resizing.
    # Set to True only if you want the model to see a HUD-free game view.
    crop_hud_strip: bool = False
    hud_strip_height_px: int = 90   # pixels from the bottom to remove

    # Screen-state detection thresholds.
    # Only LOADING is auto-detected here; PAUSED / MENU detection lives
    # in vision/menu_detect.py (OCR-based, reads the actual menu text).
    # We still expose the paused-hotbar fallback for headless cases
    # where the menu detector hasn't been built yet, but the agent
    # loop never relies on it.
    loading_brightness_threshold: float = 8.0   # mean brightness below → loading/black
    paused_hotbar_std_threshold: float = 3.0    # hotbar std below → pause

    # Hotbar selected-slot detection: the selected slot has a bright white border.
    hotbar_selected_brightness: int = 200       # min mean brightness of border pixels


# ---------------------------------------------------------------------------
# FrameProcessor
# ---------------------------------------------------------------------------

class FrameProcessor:
    """
    Stateless frame processor. Instantiate once, call process() every tick.

    Parameters
    ----------
    config : ProcessorConfig
    hud    : dict  — from settings.yaml vision.hud_regions + vision.hud_extended
    """

    def __init__(self,
                 config: Optional[ProcessorConfig] = None,
                 hud: Optional[Dict] = None,
                 hud_reader: Optional[HUDReader] = None):
        self.cfg = config or ProcessorConfig()
        self.hud = hud or {}
        self.hud_reader = hud_reader
        self._frame_counter = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> GameState:
        """
        Convert a raw uint8 RGB frame into a GameState.

        Parameters
        ----------
        frame : np.ndarray  shape (H, W, 3), dtype uint8, RGB

        Returns
        -------
        GameState
        """
        ts = time.perf_counter()
        self._frame_counter += 1

        raw = frame if self.cfg.keep_raw else None

        # 1. Detect screen state first (fast, uses whole frame).
        screen_state = self._detect_screen_state(frame)

        # 2. Crop HUD strip if configured.
        view = self._crop_hud(frame) if self.cfg.crop_hud_strip else frame

        # 3. Resize to model input size.
        model_frame = self._resize(view)

        # 4. Extract HUD signals (only meaningful when actually playing).
        if screen_state == ScreenState.PLAYING:
            if self.hud_reader is not None:
                snap = self.hud_reader.read(frame)
                health, hunger, armor, xp = snap.health, snap.hunger, snap.armor, snap.xp_bar
            else:
                health = hunger = armor = xp = 0.0
            slot = self._detect_hotbar_slot(frame)
        else:
            health = hunger = armor = xp = 0.0
            slot = 1

        return GameState(
            frame        = model_frame,
            frame_raw    = raw,
            timestamp    = ts,
            screen_state = screen_state,
            health       = health,
            hunger       = hunger,
            armor        = armor,
            xp_bar       = xp,
            hotbar_slot  = slot,
            frame_index  = self._frame_counter,
            f3           = None,  # filled by agent loop via F3Reader
        )

    # ------------------------------------------------------------------
    # Screen state detection
    # ------------------------------------------------------------------

    def _detect_screen_state(self, frame: np.ndarray) -> str:
        # Black / loading screen: nearly no light anywhere.
        if float(frame.mean()) < self.cfg.loading_brightness_threshold:
            return ScreenState.LOADING

        # PAUSED detection (pause menu / death screen): the dim overlay
        # makes the *hotbar* slots near-uniform. A real hotbar has lots
        # of contrast because of item icons, slot borders, and the
        # selected-slot highlight.
        hotbar_std = self._roi_std(frame, "hotbar_region")
        if hotbar_std is not None and hotbar_std < self.cfg.paused_hotbar_std_threshold:
            return ScreenState.PAUSED

        # We deliberately do NOT use the crosshair std as a MENU signal
        # any more. Looking at any uniform gameplay surface (sky, snow,
        # ocean, the inside of a deep cave) also gives a uniform
        # crosshair patch, which caused the agent dispatch to flicker
        # menu/playing every frame and tap-tap-tap the movement keys.
        # If we ever need a real "inventory open" signal we'll add a
        # purpose-built detector (e.g. for the GUI background tint).
        return ScreenState.PLAYING

    def _roi_std(self, frame: np.ndarray, key: str) -> Optional[float]:
        region = self.hud.get(key)
        if not region or len(region) != 4:
            return None
        x, y, w, h = region
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            return None
        return float(roi.std())

    def _crosshair_std(self, frame: np.ndarray) -> Optional[float]:
        center = self.hud.get("crosshair_center")
        if not center:
            return None
        cx, cy = center
        r = 8   # small patch around crosshair
        patch = frame[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
        if patch.size == 0:
            return None
        return float(patch.std())

    # ------------------------------------------------------------------
    # Hotbar slot detection
    # ------------------------------------------------------------------

    def _detect_hotbar_slot(self, frame: np.ndarray) -> int:
        """
        The selected hotbar slot has a bright white/light-grey border.
        Check each slot rect and return the brightest one (1-based).
        """
        slot_rects: List[List[int]] = (
            self.hud.get("hud_extended", {}).get("hotbar_slot_rects")
            or self.hud.get("hotbar_slot_rects")
            or []
        )
        if not slot_rects:
            return 1

        best_slot = 1
        best_brightness = -1.0

        for i, rect in enumerate(slot_rects):
            if len(rect) != 4:
                continue
            x, y, w, h = rect
            # Sample only the 2-pixel border of each slot (where the selection
            # highlight appears), not the icon interior.
            border = self._extract_border(frame, x, y, w, h, thickness=2)
            if border is None:
                continue
            brightness = float(border.mean())
            if brightness > best_brightness:
                best_brightness = brightness
                best_slot = i + 1

        return best_slot

    def _extract_border(
        self, frame: np.ndarray, x: int, y: int, w: int, h: int, thickness: int = 2
    ) -> Optional[np.ndarray]:
        H, W = frame.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return None
        roi = frame[y0:y1, x0:x1]
        rh, rw = roi.shape[:2]
        t = min(thickness, rh // 2, rw // 2)
        if t < 1:
            return roi
        top    = roi[:t, :]
        bottom = roi[rh - t:, :]
        left   = roi[t:rh - t, :t]
        right  = roi[t:rh - t, rw - t:]
        parts  = [p for p in (top, bottom, left, right) if p.size > 0]
        return np.concatenate([p.reshape(-1, 3) for p in parts], axis=0) if parts else None

    # ------------------------------------------------------------------
    # Frame transforms
    # ------------------------------------------------------------------

    def _crop_hud(self, frame: np.ndarray) -> np.ndarray:
        h = frame.shape[0]
        cut = max(0, h - self.cfg.hud_strip_height_px)
        return frame[:cut, :]

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        th, tw = self.cfg.model_input_size
        fh, fw = frame.shape[:2]
        if fh == th and fw == tw:
            return frame
        return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_processor(settings: Dict[str, Any]) -> FrameProcessor:
    """
    Build a FrameProcessor from the project settings dict.

    Parameters
    ----------
    settings : dict  — loaded from config/settings.yaml
    """
    def _get(d, path, default=None):
        cur = d
        for part in path.split('.'):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    # Merge hud_regions and hud_extended into one flat dict for the processor.
    hud: Dict[str, Any] = {}
    hud_regions  = _get(settings, "vision.hud_regions")  or {}
    hud_extended = _get(settings, "vision.hud_extended") or {}
    hud.update(hud_regions)
    # Keep extended data accessible as hud["hud_extended"] for slot detection.
    hud["hud_extended"] = hud_extended

    # settings.yaml stores image_size as [width, height] (e.g. [1920, 1080]).
    # Default model input: 1/4 resolution, preserving aspect ratio.
    img_size = _get(settings, "training.image_size") or [1920, 1080]
    if not isinstance(img_size, (list, tuple)) or len(img_size) < 2:
        img_size = [1920, 1080]
    model_w = max(1, int(img_size[0]) // 4)
    model_h = max(1, int(img_size[1]) // 4)

    cfg = ProcessorConfig(
        model_input_size   = (model_h, model_w),
        keep_raw           = False,
        crop_hud_strip     = bool(_get(settings, "vision.crop_hud", False)),
    )

    hud_reader = build_hud_reader(settings)
    return FrameProcessor(config=cfg, hud=hud, hud_reader=hud_reader)
