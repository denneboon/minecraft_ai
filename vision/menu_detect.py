# vision/menu_detect.py
"""
Detect Minecraft's pause / inventory / advancement menus by reading
their distinctive text via the same glyph-OCR pipeline used for F3.

Why OCR rather than texture / pixel checks
------------------------------------------
MC's menus all draw the same 200×20 grey button widget; matching that
texture would only tell us "some menu is open", not which one. Reading
the button labels lets us:

  * know we're in the Pause menu specifically (and tap Escape to exit),
  * later identify Options / Advancements / Save and Quit / Inventory
    screens by their unique button labels and act on them,
  * not depend on UI-scale-specific texture sizes — the same MC default
    font is used everywhere, so the existing glyph templates work for
    any scale we set in build_glyph_ocr.

Detection strategy
------------------
* Pre-build a GlyphOCR with the cached MC default font templates.
* Sweep candidate Y bands across the upper portion of the screen where
  the pause-menu title and the top buttons live. (We don't need to be
  pixel-perfect — at UI scale 2 the title is 16 px tall and the buttons
  are 20 px tall, both well above the OCR threshold.)
* Recognise each band and check the concatenated text against a small
  keyword set (``"menu"``, ``"back to game"``, ``"save and quit"``…).
* Any positive hit means a menu is open. We surface the keyword that
  matched so the caller can decide what to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import time

import cv2
import numpy as np

from vision.glyph_ocr import GlyphOCR, GlyphOCRConfig


# ---------------------------------------------------------------------------
# Keyword tables
# ---------------------------------------------------------------------------

# Each entry is (menu_name, keywords_that_uniquely_identify_it). All
# keywords are matched case-insensitively against the concatenated OCR
# output of the scanned rows.
_MENU_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "pause": (
        "back to game",
        "save and quit to title",
        "game menu",
    ),
    "advancements": ("advancements",),
    "statistics":   ("statistics",),
    "options":      ("options",),
    # The vanilla survival inventory screen has the literal text
    # "Crafting" above its 2×2 grid; "inventory" only appears in the
    # creative tab strip. "creative inventory" is the creative-mode
    # full item grid (a different screen).
    "inventory":          ("crafting",),
    "creative_inventory": ("creative inventory",),
    "chest":              ("chest", "large chest"),
    "furnace":            ("furnace",),
    "crafting_table":     ("crafting",),  # same as 'inventory' for now
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MenuDetectorConfig:
    ui_scale: int = 2

    # Y range (in screen pixels) to sweep. The pause menu title sits
    # near the top, but the buttons span most of the upper half on
    # smaller resolutions and taller modded screens — scanning into
    # y=h*0.7 covers every variant cheaply.
    y_start_frac:  float = 0.04
    y_end_frac:    float = 0.70

    # Stride between scanned rows (in screen pixels). 8 is fine — even
    # if a glyph straddles a row boundary the neighbouring row picks it
    # up. Smaller = slower; larger = more chance of missing short text.
    y_step_px:     int   = 8

    # Match threshold passed to GlyphOCR. Slightly stricter than F3
    # reads (0.90) because menu buttons sit on a darker drop-shadow
    # background and we want to avoid mistaking a stray bright pixel
    # for a glyph and producing phantom keywords.
    match_threshold: float = 0.88

    # Wall-clock budget (ms) for one detect() sweep. detect() OCRs ~100
    # full-WIDTH rows top-to-bottom; on a busy/garbled scene every row pays
    # the glyph-OCR retry cascade and the sweep measured 0.4–2.7s — far too
    # slow to call even occasionally in a control loop. Rows are scanned
    # top-first and the pause-menu title/buttons sit in the upper band, so a
    # budget that stops partway still catches the menu. 0 disables.
    read_budget_ms: int = 300

    keywords: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: dict(_MENU_KEYWORDS)
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class MenuDetection:
    open: bool = False
    menu: Optional[str] = None       # "pause" / "inventory" / ...
    matched_keyword: Optional[str] = None
    recognised_text: str = ""

    def __bool__(self) -> bool:
        return self.open


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class MenuDetector:
    """
    Build once at startup (so the glyph templates are loaded only
    once), then call ``detect(frame)`` per check. Each check OCRs a
    handful of rows from the upper screen and returns a structured
    result the caller can act on.
    """

    def __init__(self,
                 templates: Dict[str, np.ndarray],
                 config: Optional[MenuDetectorConfig] = None):
        self.cfg = config or MenuDetectorConfig()
        self._ocr = GlyphOCR(
            templates=templates,
            config=GlyphOCRConfig(
                ui_scale=int(self.cfg.ui_scale),
                match_threshold=float(self.cfg.match_threshold),
            ),
        )

    def detect(self, frame: np.ndarray) -> MenuDetection:
        h, w = frame.shape[:2]
        y0 = max(0, int(h * self.cfg.y_start_frac))
        y1 = min(h, int(h * self.cfg.y_end_frac))
        step = max(1, int(self.cfg.y_step_px))
        glyph_h = 8 * max(1, int(self.cfg.ui_scale))
        band_h = glyph_h + 6   # ample margin for descenders and bevels

        recognised_chunks = []
        # Inclusive end: the last valid scan row is exactly
        # ``y1 - band_h`` (its crop ends at y1). The previous
        # ``range(y0, y1 - band_h + 1, step)`` was correct in
        # principle, but with ``step`` > 1 the loop could END at
        # ``y0 + k*step`` < the last valid row, missing a menu
        # whose text sits at the very bottom of the scan band.
        # We add a final explicit-row pass so the bottom is
        # always sampled regardless of step alignment.
        scan_rows = list(range(y0, y1 - band_h + 1, step))
        last_valid = y1 - band_h
        if scan_rows and scan_rows[-1] < last_valid:
            scan_rows.append(last_valid)
        # Budget the whole sweep: on a busy/garbled scene each of the ~100
        # full-width rows pays the glyph-OCR retry cascade and the sweep hit
        # 0.4–2.7s. begin_read bounds each row's cascade; the deadline below
        # also stops the ROW loop (otherwise 100 primary decodes alone are
        # slow). Rows are scanned top-first, where pause-menu text sits, so a
        # partial sweep still detects the menu.
        budget_ms = float(getattr(self.cfg, "read_budget_ms", 0) or 0)
        self._ocr.begin_read(budget_ms / 1000.0)
        deadline = (time.perf_counter() + budget_ms / 1000.0) if budget_ms > 0 else None
        for y in scan_rows:
            if deadline is not None and time.perf_counter() > deadline:
                break
            crop = frame[y:y + band_h, :]
            try:
                text = self._ocr.recognize_line(crop)
            except (cv2.error, ValueError, IndexError, AttributeError) as e:
                # Glyph OCR can raise on malformed crops (zero-size,
                # wrong dtype) or internal state issues. Surface the
                # first occurrence so a SYSTEMATIC failure doesn't
                # leave the agent thinking it's PLAYING while a menu
                # is actually open. Subsequent failures silenced to
                # avoid 20 Hz console spam.
                if not getattr(self, "_ocr_warn_emitted", False):
                    self._ocr_warn_emitted = True
                    print(f"[menu_detect][WARN] OCR row y={y} failed: "
                          f"{e!r} — menu detection may be unreliable. "
                          f"Further errors silenced.")
                continue
            if text:
                recognised_chunks.append(text)

        joined = "\n".join(recognised_chunks).lower()
        for menu_name, kws in self.cfg.keywords.items():
            for kw in kws:
                if kw in joined:
                    return MenuDetection(
                        open=True,
                        menu=menu_name,
                        matched_keyword=kw,
                        recognised_text=joined,
                    )
        return MenuDetection(open=False, recognised_text=joined)

    def is_pause_menu(self, frame: np.ndarray) -> bool:
        d = self.detect(frame)
        return d.open and d.menu == "pause"


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_menu_detector(settings: Dict, templates: Dict[str, np.ndarray]
                       ) -> MenuDetector:
    cap = (settings or {}).get("capture", {}) or {}
    cfg = MenuDetectorConfig(ui_scale=int(cap.get("ui_scale", 2)))
    return MenuDetector(templates=templates, config=cfg)
