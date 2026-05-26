# vision/glyph_ocr.py
"""
Pixel-perfect OCR for Minecraft's default bitmap font.

Strategy
--------
Tesseract's LSTM was trained on smooth fonts and routinely confuses
Minecraft's 5×7 bitmap glyphs (``0`` ↔ ``B``/``D``/``e``, ``1`` ↔ ``i``/``L``).
Since the font is fixed, deterministic, and we can extract its glyph
bitmaps directly from the game jar (see ``vision/mcfont.py``), we can do
exact bitmap matching instead and get 100 % accuracy on clean captures.

Algorithm per line
------------------
1. Greyscale + threshold the captured line crop so glyph pixels are 1
   and everything else is 0. MC's text is bright (alpha-1) on a dark
   panel, so a simple ``> 180`` threshold isolates the glyphs cleanly.
2. Find the vertical band that contains text (the line's baseline can
   sit at slightly different y depending on which F3 lines are visible).
3. Within that band, walk left-to-right. At each column either:
   * find an inter-glyph gap → emit ``' '`` if the gap is wide enough,
   * or try every template at this x and keep the one whose pixels
     match the captured region best. Advance by the glyph width + 1
     (MC's default 1-px inter-character spacing).
4. Continue until the line is consumed.

The matcher tolerates ±1 px vertical drift in case the line crop's y
isn't perfectly aligned with the glyph row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from vision.mcfont import upscale_template


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GlyphOCRConfig:
    ui_scale: int = 2
    text_threshold: int = 180     # pixels >= this are treated as glyph ink
    match_threshold: float = 0.90 # min pixel-match ratio for a template to win
    space_min_gap_gui_px: int = 3 # 3 base px of empty columns = ' '
    inter_char_spacing_gui_px: int = 1   # MC default font's 1-px right padding
    y_drift_px: int = 1           # search ± this many GUI px vertically
    line_glyph_h_gui_px: int = 8  # MC default font cell height (GUI px)

    # ── Background-robust binarisation (only kicks in on noisy crops) ──
    # We discriminate by the MEDIAN brightness of the line crop. Text
    # pixels are < 10 % of any crop, so the median is dominated by
    # the BACKGROUND. A dark-panel F3 sits at median ≈ 20; a dark-
    # forest background at median ≈ 60–80; a daytime / leafy / snow
    # background at median ≥ 120. We use the shadow-gated path only
    # for the bright-background case, where the simple bright-
    # threshold binary would let too much foliage / sky through.
    #
    # Why not ink-density: a line that's mostly text characters can
    # have a high ink density even on a dark background, which would
    # mis-route XYZ-style lines through shadow gating and break the
    # thin diagonal glyphs (``/``) those rely on.
    background_robust:           bool  = True
    bright_background_median:    int   = 110
    shadow_drop_min_grey: int = 90
    shadow_bright_min:    int = 200
    shadow_region_gui_px: int = 8


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GlyphOCR:
    """
    Pixel template OCR for MC's default font.

    Build once per process (templates get upscaled to ``cfg.ui_scale`` and
    cached internally). Call ``recognize_line(rgb_or_gray_line)`` per F3 line.
    """

    def __init__(self,
                 templates: Dict[str, np.ndarray],
                 config: Optional[GlyphOCRConfig] = None):
        self.cfg = config or GlyphOCRConfig()
        if not templates:
            raise ValueError("GlyphOCR requires a non-empty template dict")

        scale = max(1, int(self.cfg.ui_scale))
        # Sort templates by width descending — when two templates overlap
        # at the same x, the wider one usually wins (e.g. prefer "10" digits
        # over partial matches). Tied widths are broken by char value for
        # determinism.
        scaled: List[Tuple[str, np.ndarray]] = []
        for ch, tpl in templates.items():
            if tpl.size == 0:
                continue
            scaled.append((ch, upscale_template(tpl, scale)))
        scaled.sort(key=lambda t: (-t[1].shape[1], t[0]))

        self._chars     = [ch for ch, _ in scaled]
        self._templates = [tpl for _, tpl in scaled]

        # Pre-binarize templates so matching is a fast equality test.
        self._tpl_bin = [(t > 0) for t in self._templates]

        self._glyph_h_screen = self.cfg.line_glyph_h_gui_px * scale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recognize_line(self, line_img: np.ndarray) -> str:
        """Decode a single F3 line. Returns "" if no text was found."""
        binary = self._binarize(line_img)
        if not binary.any():
            return ""

        y_top = self._find_y_top(binary)
        if y_top is None:
            return ""

        band = binary[y_top : y_top + self._glyph_h_screen, :]
        if band.shape[0] < self._glyph_h_screen:
            pad = self._glyph_h_screen - band.shape[0]
            band = np.pad(band, ((0, pad), (0, 0)), mode="constant")

        return self._scan_band(band)

    def recognize_frame_lines(
        self,
        frame: np.ndarray,
        line_crops: List[Tuple[int, int, int, int]],
    ) -> List[str]:
        """
        Decode each (x, y, w, h) line crop from a full frame and return a
        list of decoded strings — one per crop.
        """
        out = []
        for (x, y, w, h) in line_crops:
            crop = frame[y:y + h, x:x + w]
            out.append(self.recognize_line(crop))
        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def known_chars(self) -> str:
        return "".join(self._chars)

    def debug_binary(self, line_img: np.ndarray) -> np.ndarray:
        """Return the binarised line as 0/255 uint8 for saving as a PNG."""
        return self._binarize(line_img) * 255

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _binarize(self, line_img: np.ndarray) -> np.ndarray:
        """
        Binarise a line crop to {0, 1} glyph ink. Two-path:

          1. Simple bright-threshold (cheap; what we want on dark
             backgrounds where text is uniquely bright).
          2. Shadow-gated: keep only bright pixels with a dark pixel
             one row down + one column right (MC's text drop-shadow
             signature). Used when the simple path lights up too many
             pixels — a sign of bright/noisy background contaminating
             the binary.

        Real-world impact: night-time + dark forest crops stay on the
        fast path; daytime sky / leaves / snow crops automatically
        switch to shadow gating so background noise is filtered out
        without breaking the existing pipeline.
        """
        if line_img.ndim == 3:
            gray = cv2.cvtColor(line_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = line_img
        bright = (gray >= self.cfg.text_threshold)
        if not self.cfg.background_robust:
            return bright.astype(np.uint8)
        # Background median picks the binariser: dark backgrounds
        # (panel, dark forest) use the simple path; bright backgrounds
        # (sky, leaves, snow) trigger shadow gating.
        median_bg = (int(np.median(gray)) if gray.size else 0)
        if median_bg < self.cfg.bright_background_median:
            return bright.astype(np.uint8)
        return self._shadow_binarize(gray, bright)

    def _shadow_binarize(self,
                          gray: np.ndarray,
                          bright_mask: np.ndarray,
                          ) -> np.ndarray:
        """
        Shadow-gated binarisation. Drop-shadow signature is a structural
        invariant of MC's text renderer; using it as a *region detector*
        (then keeping the original bright pattern inside those regions)
        preserves glyph shape so the existing templates still match.
        """
        cfg = self.cfg
        gi = gray.astype(np.int16)
        is_bright_here = (gi >= cfg.shadow_bright_min)
        is_dark_drop = np.zeros_like(gi, dtype=bool)
        is_dark_drop[:-1, :-1] = (
            (gi[:-1, :-1] - gi[1:, 1:]) >= cfg.shadow_drop_min_grey
        )
        has_shadow = is_bright_here & is_dark_drop
        if not has_shadow.any():
            # No text-like signature found; nothing to keep.
            return np.zeros_like(bright_mask, dtype=np.uint8)

        # Generously expand each shadow pixel UP+LEFT to cover one
        # glyph's worth of neighbourhood (the glyph BODY lies above-
        # left of its bottom-right shadow). Anchor at bottom-right.
        scale = max(1, int(cfg.ui_scale))
        k = max(1, cfg.shadow_region_gui_px * scale)
        kernel = np.ones((k + 1, k + 1), dtype=np.uint8)
        anchor = (k, k)
        text_region = cv2.dilate(has_shadow.astype(np.uint8), kernel,
                                  anchor=anchor, iterations=1).astype(bool)
        return (bright_mask & text_region).astype(np.uint8)

    def _find_y_top(self, binary: np.ndarray) -> Optional[int]:
        """Pick the y-row where the densest band of glyph pixels begins."""
        # Sum per row, then find the contiguous tallest run with text. The
        # first row of that run is our band's top.
        rows_with_text = binary.any(axis=1)
        if not rows_with_text.any():
            return None
        # Find the longest run of True; the top of that run is our anchor.
        idx = np.where(rows_with_text)[0]
        # Group consecutive indices
        gaps = np.diff(idx)
        run_starts = [int(idx[0])]
        run_ends   = []
        for i, g in enumerate(gaps):
            if g > 1:
                run_ends.append(int(idx[i]))
                run_starts.append(int(idx[i + 1]))
        run_ends.append(int(idx[-1]))
        best = max(range(len(run_starts)),
                   key=lambda k: run_ends[k] - run_starts[k])
        return run_starts[best]

    def _scan_band(self, band: np.ndarray) -> str:
        h, w = band.shape
        out: List[str] = []
        x = 0
        scale = max(1, int(self.cfg.ui_scale))
        space_thresh = self.cfg.space_min_gap_gui_px * scale
        inter_pad    = self.cfg.inter_char_spacing_gui_px * scale

        # Skip leading blank columns silently.
        while x < w and not band[:, x].any():
            x += 1

        while x < w:
            # Gap = potential space
            if not band[:, x].any():
                gap_start = x
                while x < w and not band[:, x].any():
                    x += 1
                if out and (x - gap_start) >= space_thresh:
                    out.append(" ")
                continue

            best = self._best_template(band, x)
            if best is None:
                # Advance past this unmatched column to avoid infinite loop;
                # emit ``?`` so the caller can see something failed.
                out.append("?")
                while x < w and band[:, x].any():
                    x += 1
                continue

            ch, advance = best
            out.append(ch)
            x += advance + inter_pad

        return "".join(out).rstrip()

    def _best_template(self,
                       band: np.ndarray,
                       x: int) -> Optional[Tuple[str, int]]:
        h, w = band.shape
        best_ch:    Optional[str]   = None
        best_score: float           = self.cfg.match_threshold
        best_w:     int             = 0

        for ch, tpl_bin in zip(self._chars, self._tpl_bin):
            th, tw = tpl_bin.shape
            if th > h or x + tw > w + 0:
                continue
            if x + tw > w:
                continue
            region = band[:th, x:x + tw].astype(bool)
            # Fraction of pixels that agree (both set or both unset)
            score = float(np.mean(region == tpl_bin))
            if score > best_score:
                best_score = score
                best_ch    = ch
                best_w     = tw

        if best_ch is None:
            return None
        return best_ch, best_w


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_glyph_ocr(settings: Dict, templates: Dict[str, np.ndarray]) -> GlyphOCR:
    """Wire a GlyphOCR from project settings + already-loaded templates."""
    cap_cfg = (settings or {}).get("capture", {}) or {}
    ocr_cfg = (settings or {}).get("vision", {}).get("ocr", {}) or {}
    cfg = GlyphOCRConfig(
        ui_scale=int(cap_cfg.get("ui_scale", 2)),
        text_threshold=int(ocr_cfg.get("text_threshold", 180)),
        match_threshold=float(ocr_cfg.get("match_threshold", 0.90)),
    )
    return GlyphOCR(templates=templates, config=cfg)
