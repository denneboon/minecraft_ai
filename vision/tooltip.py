# vision/tooltip.py
"""
Detect and OCR Minecraft inventory tooltips.

Why we need this
----------------
Phase 1 of the inventory pipeline (``vision/inventory.py``) only commits
to an item identification when its template-matching confidence is high
— intentionally, so we never assert a confident-but-wrong id. Anything
weaker comes out as ``unknown``. Phase 2 (this module + the
``InventoryInspector`` agent) closes those unknowns by hovering the
cursor over the slot, waiting for Minecraft to draw the tooltip, and
OCRing it.

The user has **Advanced Tooltips** enabled (F3 + H in the game), so
every tooltip carries a third line that reads literally
``minecraft:<id>`` — that's our ground-truth label for the slot. It's
stable across rename + enchant + damage (the first line is the display
name, which the player can change; the id line is fixed by the item
type).

What we read
------------
1. **Display name** — first line of the tooltip. Coloured by rarity
   (white for common, aqua for rare, gold for epic, etc.).
2. **Durability** — when present, "Durability: <cur> / <max>" rendered
   in light grey.
3. **``minecraft:<id>``** — the canonical item id (advanced tooltips).
   Rendered in dark grey.
4. **Component / NBT lines** — e.g. "18 component(s)", "[+Sharpness V]"
   in dark grey or coloured. Not parsed today; kept as raw lines.

How detection works
-------------------
Minecraft draws the tooltip body with ``fillGradient`` at alpha 0xF0
on top of whatever's behind it, then adds a thin purple-gradient border
around the edges. The body's RGB on the standard inventory grey panel
comes out as roughly (20, 5, 20) — *very* dark with a noticeable purple
tint. Almost nothing else in MC's UI looks like this, so a simple
HSV-range mask + connected-components find finds the tooltip reliably.

Caveats
-------
* Tooltip layout shifts when you hover near a screen edge — MC flips
  it left/up to keep it on-screen. The detector handles that by
  searching the whole frame, not a fixed offset from the cursor.
* If multiple tooltips were somehow visible at once (shouldn't happen
  in vanilla but mods can do it), we pick the one nearest ``near_xy``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------

# HSV range that flags tooltip-body pixels.
# Empirically: tooltip fill ≈ (20, 5, 20) RGB → HSV (~150, ~190, ~20).
# Hue around 150 in OpenCV's 0..180 scale is purple/magenta.
_TIP_HSV_LOW  = (130,  60,   8)
_TIP_HSV_HIGH = (170, 255,  50)

# Lines must be at least this tall (in screen px) and the tooltip body
# must contain at least this many flagged pixels to count as a tooltip
# (not just a stray purple speck somewhere on screen).
_MIN_TIP_AREA_PX = 600
_MIN_TIP_W_PX    = 40
_MIN_TIP_H_PX    = 14


# Standard MC vertical layout (in GUI px at scale 1):
#   * 4 px top padding above first line of text
#   * 10 px line stride (8 px glyph + 2 px spacing)
#   * 4 px bottom padding below last line
_LINE_STRIDE_GUI = 10
_LINE_GLYPH_H_GUI = 8
_TIP_TOP_PAD_GUI = 4
_TIP_LEFT_PAD_GUI = 4


@dataclass
class TooltipLine:
    text: str                                # OCR'd characters
    y_top_px: int                            # row in the FULL frame
    bbox: Tuple[int, int, int, int]          # (x, y, w, h) of the line


@dataclass
class TooltipInfo:
    """
    Parsed view of a single Minecraft tooltip.

    Fields are best-effort; ``raw_lines`` is the authoritative output
    and downstream code can parse it differently if MC's layout changes.
    """
    bbox: Tuple[int, int, int, int]          # (x, y, w, h) on the frame
    raw_lines: List[TooltipLine] = field(default_factory=list)
    display_name: Optional[str]  = None      # first line
    item_id:      Optional[str]  = None      # "minecraft:<id>"
    durability:   Optional[Tuple[int, int]] = None   # (cur, max)
    component_count: Optional[int] = None    # from "18 component(s)"

    @property
    def is_valid(self) -> bool:
        return bool(self.raw_lines)


# ---------------------------------------------------------------------------
# TooltipReader
# ---------------------------------------------------------------------------

class TooltipReader:
    """
    Find and OCR a Minecraft tooltip drawn over a captured frame.

    Construct once per process (loads the cached MC font templates), then
    call ``read(frame, near_xy=...)`` for each hover.

    Parameters
    ----------
    font_templates : ``dict`` mapping each character → uint8 2D template
                      bitmap at GUI scale 1. Use
                      ``vision.mcfont.ensure_font_cache`` to produce.
    ui_scale       : Minecraft GUI scale.
    """

    def __init__(self,
                 font_templates: Dict[str, np.ndarray],
                 *,
                 ui_scale: int = 2):
        self._scale = max(1, int(ui_scale))
        # Lazy import to avoid a dep cycle.
        from vision.glyph_ocr import GlyphOCR, GlyphOCRConfig
        # Tooltip text comes in several colours — display-name white,
        # id-line dark grey, durability bright green, enchant purple.
        # Use a low threshold so even grey text binarises into ink; we
        # filter false-positives by requiring each glyph match a
        # template, not by colour.
        self._ocr = GlyphOCR(
            templates=font_templates,
            config=GlyphOCRConfig(
                ui_scale=self._scale,
                text_threshold=60,
                match_threshold=0.85,
            ),
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def find_tooltip(self,
                     frame_rgb: np.ndarray,
                     *,
                     near_xy: Optional[Tuple[int, int]] = None,
                     ) -> Optional[Tuple[int, int, int, int]]:
        """
        Locate the tooltip's bounding box in ``frame_rgb``, or ``None``
        if no tooltip is visible.

        If ``near_xy`` is given and multiple tooltip-like regions are
        found, the one closest to that point wins.
        """
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        lo = np.array(_TIP_HSV_LOW,  dtype=np.uint8)
        hi = np.array(_TIP_HSV_HIGH, dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        if not mask.any():
            return None

        # Dilate slightly so a 1-px border break doesn't split the
        # tooltip into two components.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=1)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        candidates: List[Tuple[int, int, int, int]] = []
        for i in range(1, n_labels):
            x, y, w, h, area = stats[i]
            if (area >= _MIN_TIP_AREA_PX
                    and w >= _MIN_TIP_W_PX and h >= _MIN_TIP_H_PX):
                candidates.append((int(x), int(y), int(w), int(h)))
        if not candidates:
            return None

        if near_xy is None:
            # Largest area wins — there's almost always only one.
            candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
            return candidates[0]

        nx, ny = near_xy
        def _dist(box):
            x, y, w, h = box
            cx, cy = x + w // 2, y + h // 2
            return (cx - nx) ** 2 + (cy - ny) ** 2
        candidates.sort(key=_dist)
        return candidates[0]

    def read(self,
             frame_rgb: np.ndarray,
             *,
             near_xy: Optional[Tuple[int, int]] = None,
             ) -> Optional[TooltipInfo]:
        """
        Detect + OCR a tooltip from ``frame_rgb``. Returns ``None`` when
        no tooltip is visible; otherwise a populated ``TooltipInfo``.
        """
        bbox = self.find_tooltip(frame_rgb, near_xy=near_xy)
        if bbox is None:
            return None

        x, y, w, h = bbox
        s = self._scale
        # Trim 1 GUI px of border on each side so OCR sees just the
        # interior text area.
        inner_x0 = x + 1 * s
        inner_y0 = y + _TIP_TOP_PAD_GUI * s
        inner_x1 = x + w - 1 * s
        inner_y1 = y + h - _TIP_TOP_PAD_GUI * s
        inner = frame_rgb[inner_y0:inner_y1, inner_x0:inner_x1]
        if inner.size == 0:
            return TooltipInfo(bbox=bbox)

        info = TooltipInfo(bbox=bbox)
        info.raw_lines = self._ocr_lines(frame_rgb, bbox)

        # Populate parsed fields from the raw lines.
        for line in info.raw_lines:
            t = line.text.strip()
            if not t:
                continue
            if info.display_name is None:
                info.display_name = t
            if info.item_id is None and t.startswith("minecraft:"):
                info.item_id = t.split()[0]      # strip trailing junk
            if info.durability is None and t.lower().startswith("durability"):
                info.durability = _parse_durability(t)
            if info.component_count is None and "component" in t.lower():
                info.component_count = _parse_component_count(t)

        return info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ocr_lines(self,
                   frame_rgb: np.ndarray,
                   bbox: Tuple[int, int, int, int]) -> List[TooltipLine]:
        """
        Walk down the tooltip in ``_LINE_STRIDE_GUI``-px increments,
        OCRing each candidate row.
        """
        x, y, w, h = bbox
        s = self._scale
        stride = _LINE_STRIDE_GUI * s
        band_h = _LINE_GLYPH_H_GUI * s + 4   # generous margin
        text_x0 = x + _TIP_LEFT_PAD_GUI * s
        text_x1 = x + w - _TIP_LEFT_PAD_GUI * s
        text_y0 = y + _TIP_TOP_PAD_GUI * s
        text_y1 = y + h - _TIP_TOP_PAD_GUI * s

        out: List[TooltipLine] = []
        cur_y = text_y0
        while cur_y + band_h <= text_y1 + s:
            row = frame_rgb[cur_y:cur_y + band_h, text_x0:text_x1]
            if row.size == 0:
                break
            try:
                text = self._ocr.recognize_line(row)
            except Exception:
                text = ""
            text = text.rstrip()
            if text:
                out.append(TooltipLine(
                    text=text,
                    y_top_px=cur_y,
                    bbox=(text_x0, cur_y, text_x1 - text_x0, band_h),
                ))
            cur_y += stride
        return out


# ---------------------------------------------------------------------------
# Tiny parsers
# ---------------------------------------------------------------------------

def _parse_durability(line: str) -> Optional[Tuple[int, int]]:
    """
    Parse "Durability: 328 / 336" → (328, 336). Returns None on any
    other layout.

    Rejects ``mx <= 0`` outright so a malformed OCR (digits missing
    on the max side) doesn't return ``(cur, 0)`` and trip a
    ZeroDivisionError when a caller computes ``cur / mx`` for a
    durability bar.
    """
    parts = line.replace("Durability:", "").strip().split("/")
    if len(parts) != 2:
        return None
    try:
        cur = int("".join(c for c in parts[0] if c.isdigit()))
        mx  = int("".join(c for c in parts[1] if c.isdigit()))
    except ValueError:
        return None
    if mx <= 0 or cur < 0 or cur > mx:
        return None
    return cur, mx


def _parse_component_count(line: str) -> Optional[int]:
    """Parse "18 component(s)" → 18."""
    digits = "".join(c for c in line if c.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_tooltip_reader(settings: Optional[Dict] = None,
                         *,
                         font_templates=None,
                         assets=None) -> TooltipReader:
    """
    Build a ``TooltipReader`` wired to project settings + the cached
    MC font templates.
    """
    cap = (settings or {}).get("capture", {}) or {}
    ui_scale = int(cap.get("ui_scale", 2))
    if font_templates is None:
        from vision.mcfont import ensure_font_cache
        if assets is not None:
            cache_path = (Path(assets.root).resolve().parent.parent.parent
                          / "data" / "calibration" / "mc_font.npz")
        else:
            cache_path = (Path(__file__).resolve().parent.parent
                          / "data" / "calibration" / "mc_font.npz")
        font_templates = ensure_font_cache(str(cache_path))
    return TooltipReader(font_templates, ui_scale=ui_scale)


__all__ = [
    "TooltipReader", "TooltipInfo", "TooltipLine",
    "build_tooltip_reader",
]
