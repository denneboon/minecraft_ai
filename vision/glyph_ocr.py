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

import time
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
    # background at median ≥ 120. The bright-background path (see
    # ``bright_method``) handles the case where the simple bright-
    # threshold binary would let too much foliage / sky through.
    #
    # Why not ink-density: a line that's mostly text characters can
    # have a high ink density even on a dark background, which would
    # mis-route XYZ-style lines through the bright path and break the
    # thin diagonal glyphs (``/``) those rely on.
    background_robust:           bool  = True
    bright_background_median:    int   = 110

    # Which binariser to use on a bright background.
    #   "adaptive" — local-mean adaptive threshold (a pixel is ink iff
    #                it's brighter than its local neighbourhood mean by
    #                ``adaptive_bias``). Invariant to the absolute
    #                background brightness, so it reads MC text on a
    #                desert / snow biome where the translucent debug
    #                panel is actually DARKER than the surrounding bright
    #                terrain (global thresholds fail there because the
    #                bare terrain exceeds the threshold while the boxed
    #                glyphs sit below it). This is the F3 reader default.
    #   "shadow"   — keep only bright pixels carrying MC's drop-shadow
    #                signature (bright pixel with a darker down-right
    #                neighbour). Cheaper, and the historical default for
    #                the inventory / tooltip / menu readers whose crops
    #                are small and uniformly lit; kept for back-compat.
    bright_method:        str   = "shadow"
    # Adaptive-threshold params (used when bright_method == "adaptive").
    adaptive_block_gui_px: int  = 5    # local window, GUI px (→ odd screen px)
    adaptive_bias:        int   = 10   # ink iff pixel > local_mean + bias
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

        # ── Width-grouped, pre-stacked templates for vectorised matching ──
        # ``_best_template`` used to call ``np.mean`` once per template
        # (100+ Python-level reductions per scanned column). cProfile
        # showed that as ~0.5 s per F3 read — the single biggest live-
        # loop cost. Instead we group templates of identical (height,
        # width), stack each group into one ``(k, th, tw)`` bool array,
        # and score the whole group against the band region in ONE
        # numpy op. Groups are processed widest-first and, within a
        # group, chars stay in ascending order — so ``argmax`` + a
        # strict ``>`` running-best reproduces the old tie-break exactly
        # (widest wins, then lowest char value).
        from collections import OrderedDict
        groups: "OrderedDict[Tuple[int, int], Tuple[List[str], List[np.ndarray]]]" = \
            OrderedDict()
        for ch, tb in zip(self._chars, self._tpl_bin, strict=True):
            if tb.ndim != 2 or tb.size == 0:
                continue
            key = (tb.shape[0], tb.shape[1])
            chars_list, tpls_list = groups.setdefault(key, ([], []))
            chars_list.append(ch)
            tpls_list.append(tb)
        # Width-descending so a wider template wins ties (matches the old
        # ``sort(key=(-width, char))`` + strict ``>`` semantics).
        self._width_groups: List[Tuple[int, int, List[str], np.ndarray]] = []
        for (th, tw), (chars_list, tpls_list) in sorted(
            groups.items(), key=lambda kv: -kv[0][1]
        ):
            self._width_groups.append(
                (th, tw, chars_list, np.stack(tpls_list).astype(bool))
            )

        self._glyph_h_screen = self.cfg.line_glyph_h_gui_px * scale

        # Optional wall-clock deadline for a multi-line read. The retry
        # cascades (alternative binarisations + anchored sub-crops) are what
        # make a GARBLED busy scene explode: every line fails _is_garbled and
        # pays the full ~10-decode cascade, so a forest F3 read measured ~3.7 s
        # (vs ~86 ms on a clean panel). The primary decode of each line ALWAYS
        # runs (cheap, gets pose/xyz even on busy scenes); the retries only run
        # while under this deadline. None = unlimited (the default for the
        # inventory/tooltip/menu readers, whose crops are tiny).
        self._deadline: Optional[float] = None

    def begin_read(self, budget_s: float) -> None:
        """Start a budgeted multi-line read: the retry cascades in
        ``recognize_line`` bail once ``budget_s`` of wall-clock elapses, so a
        busy/garbled scene can't blow the per-read time up to seconds. Call
        once before a batch of ``recognize_line`` calls. ``budget_s <= 0``
        clears the budget (unlimited)."""
        self._deadline = (time.perf_counter() + budget_s) if budget_s and budget_s > 0 else None

    def _past_deadline(self) -> bool:
        return self._deadline is not None and time.perf_counter() > self._deadline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recognize_line(self, line_img: np.ndarray) -> str:
        """Decode a single F3 line. Returns "" if no text was found.

        Robust multi-binarisation: the bright-background binariser has a
        single tuning (adaptive window / drop threshold) that reads MOST
        views, but a busy local background (e.g. a coloured block bleeding
        into the line over bright terrain) can defeat any single setting.
        Rather than gamble on one, we decode with the primary binariser
        and, ONLY IF that came out garbled (unrecognised ``?`` glyphs or
        empty), retry with a few alternative binarisations and keep the
        cleanest decode. The downstream regex parsers are the final
        arbiter; this just maximises the chance a readable line is read.
        The fast/dark path (uniform background) parses on the first try
        and never pays for the alternatives."""
        best = self._decode_binary(self._binarize(line_img))
        if not self._is_garbled(best):
            return best
        # Fast mode: skip the (expensive) retries. Used during the bridge
        # click loop where a ~150-200 ms multi-binarisation read would
        # stall the tight loop; a quick best-effort first pass is enough
        # to track the player's block position for jump timing.
        if getattr(self, "fast_mode", False):
            return best
        # Budget guard: on a busy scene EVERY line is garbled and would run the
        # full retry cascade below — ~3.7 s for a whole F3 panel. Once the
        # per-read deadline passes, skip the (expensive) retries and return the
        # primary decode. Pose/xyz already parse from the primary pass, so the
        # bot keeps navigating fast instead of "thinking" once every few sec.
        if self._past_deadline():
            return best
        # Retry 1: alternative binarisations of the FULL crop.
        for alt in self._alternative_binaries(line_img):
            cand = self._decode_binary(alt)
            if self._decode_quality(cand) > self._decode_quality(best):
                best = cand
                if not self._is_garbled(best):
                    return best
            if self._past_deadline():
                return best
        # Retry 2: horizontally-anchored SUB-CROPS. F3 columns are left-
        # or right-aligned, so the empty side of a wide line crop is bare
        # terrain. Over bright sand that empty side produces adaptive
        # noise that fools the line-band finder (it locks onto the sand's
        # top edge instead of the text). Decoding a crop anchored to the
        # text side excludes that noise and recovers the line. We try a
        # few widths on each side and keep the cleanest, fullest decode.
        for sub in self._anchored_subcrops(line_img):
            if self._past_deadline():
                return best
            cand = self._decode_binary(self._binarize(sub))
            if self._decode_quality(cand) > self._decode_quality(best):
                best = cand
                if not self._is_garbled(best):
                    return best
        return best

    def _anchored_subcrops(self, line_img: np.ndarray):
        """Yield right-anchored then left-anchored horizontal sub-crops of
        a line, widest-first, to isolate left-/right-aligned F3 text from
        bare-terrain noise on the empty side (see :meth:`recognize_line`)."""
        w = line_img.shape[1]
        if w < 80:
            return
        for frac in (0.62, 0.48, 0.36):          # right-anchored
            x0 = int(w * (1.0 - frac))
            if w - x0 >= 40:
                yield line_img[:, x0:]
        for frac in (0.62, 0.48):                # left-anchored
            x1 = int(w * frac)
            if x1 >= 40:
                yield line_img[:, :x1]

    def _decode_binary(self, binary: np.ndarray) -> str:
        """Run band-finding + glyph scan on an already-binarised line."""
        if binary is None or not binary.any():
            return ""
        y_top = self._find_y_top(binary)
        if y_top is None:
            return ""
        band = binary[y_top : y_top + self._glyph_h_screen, :]
        if band.shape[0] < self._glyph_h_screen:
            pad = self._glyph_h_screen - band.shape[0]
            band = np.pad(band, ((0, pad), (0, 0)), mode="constant")
        return self._scan_band(band)

    @staticmethod
    def _is_garbled(text: str) -> bool:
        """A decode is 'garbled' if it's empty or carries any unmatched
        ``?`` marker — either is a signal worth a second binarisation
        attempt. (A clean F3 line decodes with zero ``?``.)"""
        return (not text) or ("?" in text)

    @staticmethod
    def _decode_quality(text: str) -> float:
        """Higher is better. Rewards recognised characters and penalises
        unmatched ``?`` markers, so the alternative with the cleanest
        decode wins. Length breaks ties (a fuller line is usually the
        better read)."""
        if not text:
            return -1e9
        q = text.count("?")
        recognised = sum(1 for c in text if c not in "? ")
        return recognised - 3.0 * q + 0.01 * len(text)

    def _alternative_binaries(self, line_img: np.ndarray):
        """Yield alternative binarisations to try when the primary decode
        is garbled. Only meaningful on the bright path (uniform/dark
        backgrounds already parse first try); we vary the adaptive window
        size — a SMALLER window isolates glyphs from a busy local
        background, a LARGER one rides through fine texture — and fall
        back to the structural drop-shadow gate. Cheap: each is one
        threshold pass, and we only get here on a line that already
        failed."""
        if line_img.ndim == 3:
            gray = cv2.cvtColor(line_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = line_img
        gray_u8 = gray if gray.dtype == np.uint8 else gray.astype(np.uint8)
        scale = max(1, int(self.cfg.ui_scale))
        bias = int(self.cfg.adaptive_bias)
        # A SMALLER local window than the default — isolates glyphs from a
        # busy local background (a coloured block bleeding into the line).
        block = max(3, 3 * scale)
        if block % 2 == 0:
            block += 1
        yield cv2.adaptiveThreshold(
            gray_u8, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            block, -bias).astype(np.uint8)
        # Structural drop-shadow gate as a last resort (works where local
        # contrast is ambiguous but the text still carries its shadow).
        yield self._shadow_binarize(gray_u8.astype(np.int16),
                                    gray_u8 >= self.cfg.text_threshold)

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
          2. Bright-background path (``bright_method``): "adaptive"
             local-mean thresholding (robust to the absolute terrain
             brightness) or "shadow" drop-shadow gating. Used when the
             simple path lights up too many pixels — a sign of
             bright/noisy background contaminating the binary.

        Real-world impact: night-time + dark forest crops stay on the
        fast path; daytime sky / leaves / snow / desert crops
        automatically switch to the bright path so background noise is
        filtered out without breaking the existing pipeline.
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
        # (sky, leaves, snow, sand) trigger the bright path.
        median_bg = (int(np.median(gray)) if gray.size else 0)
        if median_bg < self.cfg.bright_background_median:
            return bright.astype(np.uint8)
        if self.cfg.bright_method == "adaptive":
            return self._adaptive_binarize(gray)
        return self._shadow_binarize(gray, bright)

    def _adaptive_binarize(self, gray: np.ndarray) -> np.ndarray:
        """
        Local-mean adaptive threshold: a pixel is glyph ink iff it is
        brighter than the mean of its local neighbourhood by
        ``adaptive_bias``. Because the threshold tracks the local
        background, this reads MC text regardless of the absolute
        terrain brightness — including the desert/snow case where the
        translucent debug panel is DARKER than the surrounding bright
        terrain, so the bare terrain exceeds any global bright-threshold
        while the boxed glyphs sit below it. The bright glyph bodies
        still stand out from their immediate (panel-box) background, so
        a local comparison recovers them cleanly.
        """
        scale = max(1, int(self.cfg.ui_scale))
        block = max(3, self.cfg.adaptive_block_gui_px * scale)
        if block % 2 == 0:
            block += 1                      # cv2 requires an odd block size
        gray_u8 = gray if gray.dtype == np.uint8 else gray.astype(np.uint8)
        binary = cv2.adaptiveThreshold(
            gray_u8, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            block, -int(self.cfg.adaptive_bias),
        )
        return binary.astype(np.uint8)

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
        """
        Pick the y-row where the densest band of glyph pixels begins.

        Two pitfalls we explicitly defend against:

        1. **Noise above the text** — bright biome pixels that survived
           the shadow-gated binariser (a few foliage leaves, a sky
           gradient pixel, a glint highlight) put scattered ink at the
           top of the crop. The previous version of this method
           looked for the LONGEST contiguous run of "any-ink rows",
           which a few scattered noise pixels would inflate to span
           the entire crop, pulling ``y_top`` to row 0. The matcher
           band was then 3-4 px misaligned vs. the templates and every
           glyph score collapsed below the match threshold — that's
           what made the `minecraft:<id>` line of the F3 panel render
           as a single ``?`` while the surrounding lines parsed fine.

        2. **Drop-shadow tail** — every MC glyph has a 1-GUI-px
           shadow one row down-right of the glyph body. That row
           IS real text, so we don't want to discard it; we want
           the y_top to land on the top of the glyph BODY, with the
           shadow row trailing inside the band.

        Strategy: classify rows by their ink-DENSITY (column count),
        find the densest contiguous block, return its top. A row
        with one stray pixel (1 / 700 = 0.1 %) does not count as a
        text row; a row of glyph pixels (typically 15-30 % density)
        does. The threshold scales with the actual densest row so
        it adapts to short lines (`minecraft:dirt`, ~20 chars) as
        well as long ones (~50 chars).
        """
        if binary.size == 0:
            return None
        col_counts = binary.sum(axis=1).astype(np.int32)
        if col_counts.max() == 0:
            return None

        # Distinguish text rows from foliage-noise rows by ABSOLUTE ink
        # WIDTH per row, not by relative density:
        #
        # * A real glyph row spans most of the line's character width
        #   (think the middle bar of ``m``, ``n``, ``e``) — typically
        #   100+ inked pixels in a 700-px-wide crop.
        # * Foliage / sky / glint noise that survives the shadow gate
        #   is sparse — a few pixels scattered through the row.
        #
        # We use ``min_ink_width`` = 2 % of the crop width (with a hard
        # floor of 10 px so very narrow debug crops still work). The
        # noise from one or two stray leaves comes in below this; the
        # descender row of ``p`` / ``y`` / ``g`` STAYS above it (a
        # descender is one or two columns wide × the descender length,
        # producing ~5-10 inked pixels per character — and there's
        # usually more than one descender in a 30-char line). And even
        # if a particular descender row falls below the threshold, the
        # band is anchored at the TOP of the text so the descender
        # rows are still INSIDE the 16-px-tall band regardless.
        W = binary.shape[1]
        min_ink_width = max(10, W // 50)
        text_rows = col_counts >= min_ink_width
        if not text_rows.any():
            return None

        # Find the longest run of dense-text rows.
        idx = np.where(text_rows)[0]
        gaps = np.diff(idx)
        run_starts = [int(idx[0])]
        run_ends:   List[int] = []
        for i, g in enumerate(gaps):
            if g > 1:
                run_ends.append(int(idx[i]))
                run_starts.append(int(idx[i + 1]))
        run_ends.append(int(idx[-1]))
        best = max(range(len(run_starts)),
                   key=lambda k: run_ends[k] - run_starts[k])
        visible_top = run_starts[best]
        visible_bot = run_ends[best]

        # Lowercase-only lines (no ascender, no capital, no descender)
        # have visible text spanning ONLY the x-height portion of the
        # cell — typically 9-10 rows at GUI scale 2. Cap-and-body lines
        # span 14-15 rows. The templates encode the FULL cell (16 rows)
        # with empty top rows reserved for ascenders, so a band aligned
        # to "visible text top" of a lowercase-only line places the
        # text at band-row 0 — but the templates expect that text at
        # band-row ``cap_offset``. Detect this case and shift the
        # returned y_top UP by the cap offset so the band-row 0 always
        # corresponds to the CELL top regardless of which glyph
        # heights actually got rendered on this line.
        scale = max(1, int(self.cfg.ui_scale))
        full_cell_span = 13 * scale // 2     # ~13 rows at scale 2; 6 at scale 1
        # Calibrated for GUI scale 2 (what this project runs). The ``* scale //
        # 2`` form mis-rounds the ascender reserve at scale 1 (4*1//2 = 2 vs the
        # true ~2-3), so lowercase-only lines can sit a row off there — no live
        # impact at scale 2. If scale-1 is ever used, derive cap_offset from the
        # actual font ascender reserve instead of this ratio.
        cap_offset     = 4 * scale // 2      # ~4 rows at scale 2; 2 at scale 1
        span = visible_bot - visible_top + 1
        if span < full_cell_span:
            shifted = max(0, visible_top - cap_offset)
            return shifted
        return visible_top

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
        """Find the best-matching glyph template starting at column ``x``.

        Vectorised: for each width-group that fits, the band region is
        compared against ALL stacked templates of that width in a single
        numpy reduction. Equivalent to (but ~5-10× faster than) the old
        per-template ``np.mean`` loop; the widest-first group order plus
        the strict ``>`` running-best reproduce the original tie-break
        (widest wins, then lowest char value).
        """
        h, w = band.shape
        best_ch:    Optional[str] = None
        best_score: float         = self.cfg.match_threshold
        best_w:     int           = 0

        for th, tw, chars_list, stack in self._width_groups:
            if th > h or x + tw > w:
                continue
            region = band[:th, x:x + tw].astype(bool)
            # Agreement fraction per template (both set or both unset),
            # computed for the whole group at once. ``stack`` is
            # (k, th, tw) bool; broadcasting compares the single region
            # against every template, then we mean over the glyph pixels.
            agree = (stack == region).reshape(stack.shape[0], -1)
            scores = agree.mean(axis=1)
            ki = int(scores.argmax())   # first max => lowest char on ties
            s = float(scores[ki])
            if s > best_score:
                best_score = s
                best_ch    = chars_list[ki]
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
