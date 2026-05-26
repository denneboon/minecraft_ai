# vision/ocr.py
"""
F3 debug overlay reader.

This module owns the runtime API used by the agent (`F3Reader`, `F3Info`,
`build_f3_reader`). The actual character recognition is delegated to
``vision.glyph_ocr`` (pixel-perfect template matching against Minecraft's
extracted default font — see ``vision/mcfont.py``) which is far more
reliable on bitmap fonts than running Tesseract LSTM.

Tesseract is kept as a graceful fallback: if the MC font cache hasn't
been built yet (e.g. the user runs the agent on a machine where Java
Edition isn't installed), F3Reader transparently switches to the older
Tesseract path so the AI can still get *something* from the overlay.

Design goals
------------
* Same public API as before (``F3Info``, ``F3Reader``, ``build_f3_reader``)
  so callers (main.py, pipeline_test.py) don't have to change.
* Robust to a varying number of F3 lines — players can toggle individual
  debug options between Always / In Overlay / Off, so any line may
  appear or disappear from frame to frame. The reader scans up to
  ``OCRConfig.max_lines`` (default 12) and dispatches each line to
  whichever parser recognises it.
* Zero hard-coded line indices — every parser keys off line *content*,
  not position.
"""

from __future__ import annotations

import os
import platform
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from vision.glyph_ocr import GlyphOCR, GlyphOCRConfig
from vision.mcfont import ensure_font_cache


# ---------------------------------------------------------------------------
# Tesseract fallback (used only if the MC font cache cannot be built)
# ---------------------------------------------------------------------------

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None   # type: ignore
    _TESSERACT_AVAILABLE = False


def _autoconfigure_tesseract_path() -> Optional[str]:
    if not _TESSERACT_AVAILABLE:
        return None
    try:
        pytesseract.get_tesseract_version()
        return getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
    except Exception:
        pass
    if platform.system().lower() != "windows":
        return None
    for path in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ):
        if path and os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            try:
                pytesseract.get_tesseract_version()
                return path
            except Exception:
                continue
    return None


_autoconfigure_tesseract_path()


# ---------------------------------------------------------------------------
# F3Info — the structured result returned by F3Reader.read()
# ---------------------------------------------------------------------------

@dataclass
class F3Info:
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    yaw:         Optional[float] = None
    pitch:       Optional[float] = None
    facing_name: Optional[str]   = None
    dimension:   Optional[str]   = None
    fps:         Optional[int]   = None
    block_x:     Optional[int]   = None
    block_y:     Optional[int]   = None
    block_z:     Optional[int]   = None
    section_rel: Optional[Tuple[int, int, int]] = None
    timestamp:   float = field(default_factory=time.perf_counter)
    raw_text:    str   = ""
    backend:     str   = "glyph"   # "glyph" or "tesseract"

    def is_valid(self) -> bool:
        return self.x is not None

    def position(self) -> Optional[Tuple[float, float, float]]:
        if None not in (self.x, self.y, self.z):
            return (self.x, self.y, self.z)  # type: ignore
        return None

    def block_position(self) -> Optional[Tuple[int, int, int]]:
        if None not in (self.block_x, self.block_y, self.block_z):
            return (self.block_x, self.block_y, self.block_z)  # type: ignore
        return None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class OCRConfig:
    # Font geometry, derived from capture.ui_scale by build_f3_reader.
    # At ui_scale 2 (Minecraft default): 16-px glyph height, 2-px line gap.
    line_height_px: int = 16
    line_gap_px:    int = 2
    text_start_y:   int = 2
    text_start_x:   int = 2
    line_margin_px: int = 1
    # Stride between successive line crops. Defaults to MC's natural
    # line stride (~20 px at ui_scale 2 = 10 base px). build_f3_reader
    # rewrites this from ui_scale.
    line_step_px:   int = 20

    # Read at most this many lines from each F3 column. Players can
    # toggle individual debug options to Always, so the actual line
    # count varies — scanning generously costs us nothing (the OCR's
    # column-walker is cheap on blank crops). For F3-only commit
    # mode (the current default), we only need the FIRST 4-5 lines
    # of each column: LEFT = "Targeted Block:" + block id, RIGHT =
    # XYZ / Block / Chunk / Facing. Drop ``max_lines`` low to cut
    # OCR time by ~3-5x.
    max_lines:     int = 5
    line_width_px: int = 700

    # ── Multi-column layout (MC 1.20+) ─────────────────────────────
    # Modern MC renders F3 text in two columns:
    #   * LEFT  : looking-at, biome, FPS, system info
    #   * RIGHT : XYZ, Block, Chunk, Facing, dimension, section
    # Both can be "Always" on, so the reader must crop and OCR both.
    # The RIGHT column is right-aligned to the window edge; the crop
    # below grabs a generous slab ending at the right edge so the
    # widest expected line still fits.
    enable_right_column: bool = True
    right_column_width_px: int = 620   # crop width from right edge
    right_column_margin_px: int = 2    # blank pixels at the right edge

    # Glyph OCR settings (forwarded to GlyphOCRConfig).
    ui_scale:        int   = 2
    text_threshold:  int   = 180
    match_threshold: float = 0.90

    # Tesseract fallback toggle.
    enable_tesseract_fallback: bool = True


# ---------------------------------------------------------------------------
# Cardinal direction set (used by facing parser)
# ---------------------------------------------------------------------------

CARDINALS = ("north", "south", "east", "west")


# ---------------------------------------------------------------------------
# Per-line parsers — each tries to extract a piece of F3Info from any line.
# ---------------------------------------------------------------------------

# Three signed decimal numbers separated by *something* that isn't a
# digit. Tolerant separator handles OCR degrade modes where "/" reads
# as ".?" or just "?" — common on bright/busy backgrounds where the
# diagonal stroke gets eaten by the shadow filter.
#
# At least ONE of the three captures must have a decimal point. Without
# this guard the regex also matches the Block line ("Block: -110 70 -200")
# which is three integers — and if the actual XYZ line failed to OCR
# this tick, the block ints would be silently committed as the player's
# float coordinates.
_RE_XYZ_DEC   = re.compile(
    r'(-?\d+(?:\.\d+)?)[^\d-]{1,6}(-?\d+(?:\.\d+)?)[^\d-]{1,6}(-?\d+(?:\.\d+)?)'
)
_RE_HAS_DECIMAL = re.compile(r'\d\.\d')
_RE_BLOCK_INT = re.compile(r'(-?\d+)\s+(-?\d+)\s+(-?\d+)')
# Dimension id is one of a small fixed set of vanilla worlds. Restricting
# the match prevents block ids / tags (``minecraft:grass_block``,
# ``#minecraft:sniffer_diggable_block``) from being read as dimensions
# when the LEFT-column F3 lines (Targeted Block + tags) bleed into the
# OCR output alongside the RIGHT-column dimension line.
_RE_DIM       = re.compile(
    r'\bminecraft:(overworld|the_nether|the_end|nether|end)\b',
    re.IGNORECASE,
)
_RE_FPS       = re.compile(r'(\d+)\s*fps', re.IGNORECASE)
_RE_FACING    = re.compile(r'facing[:\s]+(north|south|east|west)\b', re.IGNORECASE)
# yaw / pitch on the Facing line. Same tolerant separator as XYZ.
_RE_ANGLES    = re.compile(
    r'(-?\d+\.\d+)[^\d-]{1,6}(-?\d+\.\d+)'
)

_Y_MIN, _Y_MAX = -64, 320
_XZ_LIMIT      = 30_000_000


def _coords_in_bounds(x: float, y: float, z: float) -> bool:
    return abs(x) < _XZ_LIMIT and _Y_MIN <= y <= _Y_MAX and abs(z) < _XZ_LIMIT


def _ocr_repair_digits(s: str) -> str:
    """
    Repair common glyph-OCR digit confusions in MC's 5×7 bitmap font.

    The font has several near-identical glyph pairs that the
    template matcher routinely confuses on noisy / busy backgrounds:
      * ``1`` ↔ ``L`` ↔ ``l`` ↔ ``|`` ↔ ``I``
      * ``0`` ↔ ``O`` ↔ ``o`` ↔ ``D`` (less common but happens)
      * ``5`` ↔ ``S`` ↔ ``s``
      * ``8`` ↔ ``B``
      * ``2`` ↔ ``Z`` ↔ ``z``

    Applied to numeric-only substrings: we walk the string and any
    character that looks letter-like sitting between two digits (or
    next to a digit / decimal point / sign) gets coerced to its
    digit twin. Pure-text substrings are untouched.
    """
    if not s:
        return s
    NEIGHBOURS = set("0123456789.-+/")
    # Repair table (letter → digit).
    REPAIR = {
        "L": "1", "l": "1", "|": "1", "I": "1",
        "O": "0", "o": "0", "D": "0",
        "S": "5", "s": "5",
        "B": "8",
        "Z": "2", "z": "2",
    }
    out_chars: List[str] = []
    chars = list(s)
    for i, ch in enumerate(chars):
        if ch in REPAIR:
            prev = chars[i - 1] if i > 0 else ""
            nxt  = chars[i + 1] if i + 1 < len(chars) else ""
            if prev in NEIGHBOURS or nxt in NEIGHBOURS:
                out_chars.append(REPAIR[ch])
                continue
        out_chars.append(ch)
    return "".join(out_chars)


def _try_xyz_dec(line: str) -> Optional[Tuple[float, float, float]]:
    """
    Attempt the XYZ regex on ``line``. Try the repaired copy FIRST so
    that letter-corrupted digits become numbers BEFORE the regex
    matches — otherwise a partial like ``-L10.500`` parses as the
    sub-string ``10.500`` (wrong sign and magnitude) instead of
    ``-110.500``.
    """
    for candidate in (_ocr_repair_digits(line), line):
        m = _RE_XYZ_DEC.search(candidate)
        if not m:
            continue
        # Require at least one decimal point in the matched span — a
        # line of three integers is the Block line, not XYZ. Guards
        # against integer coords ever populating ``info.x/y/z``.
        if not _RE_HAS_DECIMAL.search(m.group(0)):
            continue
        try:
            x, y, z = float(m.group(1)), float(m.group(2)), float(m.group(3))
        except ValueError:
            continue
        if _coords_in_bounds(x, y, z):
            return x, y, z
    return None


def _parse_lines(lines: List[str]) -> F3Info:
    """Walk every line and accumulate whatever fields can be recognised."""
    info = F3Info()

    for line in lines:
        if not line:
            continue
        ll = line.lower()

        # XYZ position (decimals separated by /). Uses a repair pass
        # to recover from common OCR digit/letter confusions before
        # giving up on the line.
        if info.x is None:
            triple = _try_xyz_dec(line)
            if triple is not None:
                info.x, info.y, info.z = triple

        # Block coords (whole numbers, follows "Block:")
        if info.block_x is None and ll.startswith("block"):
            m = _RE_BLOCK_INT.search(line)
            if m:
                try:
                    info.block_x = int(m.group(1))
                    info.block_y = int(m.group(2))
                    info.block_z = int(m.group(3))
                except ValueError:
                    pass

        # Section-relative coords
        if info.section_rel is None and "section-relative" in ll:
            m = _RE_BLOCK_INT.search(line)
            if m:
                try:
                    info.section_rel = (int(m.group(1)),
                                        int(m.group(2)),
                                        int(m.group(3)))
                except ValueError:
                    pass

        # Facing + yaw/pitch
        if info.facing_name is None:
            mf = _RE_FACING.search(line)
            if mf:
                info.facing_name = mf.group(1).lower()
                ma = _RE_ANGLES.search(line)
                if ma:
                    try:
                        info.yaw   = float(ma.group(1))
                        info.pitch = float(ma.group(2))
                    except ValueError:
                        pass
            elif info.facing_name is None:
                # Cardinal name without explicit "Facing:" prefix
                for card in CARDINALS:
                    if card in ll:
                        info.facing_name = card
                        ma = _RE_ANGLES.search(line)
                        if ma:
                            try:
                                info.yaw   = float(ma.group(1))
                                info.pitch = float(ma.group(2))
                            except ValueError:
                                pass
                        break

        # Dimension. Match the full ``minecraft:<dim>`` string with
        # canonical casing/aliases (so OCR'd ``nether`` is normalised
        # to ``minecraft:the_nether`` like Mojang's tag).
        if info.dimension is None:
            md = _RE_DIM.search(line)
            if md:
                dim = md.group(1).lower()
                alias = {"nether": "the_nether", "end": "the_end"}
                info.dimension = f"minecraft:{alias.get(dim, dim)}"

        # FPS
        if info.fps is None:
            mfps = _RE_FPS.search(line)
            if mfps:
                try:
                    info.fps = int(mfps.group(1))
                except ValueError:
                    pass

    info.raw_text = "\n".join(lines)
    return info


# ---------------------------------------------------------------------------
# F3Reader
# ---------------------------------------------------------------------------

class F3Reader:
    """
    Reads the F3 debug overlay via glyph-template OCR.

    Construct via ``build_f3_reader(settings)`` so font extraction and
    config wiring happen in one place.
    """

    def __init__(self,
                 config: OCRConfig,
                 templates: Optional[Dict[str, np.ndarray]] = None):
        self.cfg = config
        self._tesseract_used = False
        self._glyph_ocr: Optional[GlyphOCR] = None

        if templates:
            self._glyph_ocr = GlyphOCR(
                templates=templates,
                config=GlyphOCRConfig(
                    ui_scale=int(self.cfg.ui_scale),
                    text_threshold=int(self.cfg.text_threshold),
                    match_threshold=float(self.cfg.match_threshold),
                ),
            )

        if self._glyph_ocr is None and not (
            self.cfg.enable_tesseract_fallback and _TESSERACT_AVAILABLE
        ):
            raise RuntimeError(
                "F3Reader has no working backend: glyph templates are unavailable "
                "and Tesseract fallback is disabled or pytesseract is not installed."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self, frame: np.ndarray) -> F3Info:
        """
        Read the F3 panel from a full-resolution RGB frame.

        Cheap pre-check: a captured frame WITHOUT the F3 panel runs the
        same line-scan + per-column template matching as one with it,
        and on a bright sky/landscape the OCR can take 300–600 ms
        because half the columns trigger template attempts. The F3
        panel has a very distinctive top-left signature — a dark
        translucent rectangle with bright text on top — so we check
        that signature in <1 ms and short-circuit the read when it's
        absent. Anywhere F3 is genuinely on, this falls straight
        through to the normal OCR path.
        """
        if not self._f3_panel_visible(frame):
            return F3Info(
                backend=("glyph" if self._glyph_ocr is not None else "tesseract"),
                raw_text="",
            )

        crops = self._crop_lines(frame)
        if self._glyph_ocr is not None:
            lines = [self._glyph_ocr.recognize_line(c) for c in crops]
            backend = "glyph"
        else:
            lines = [self._tesseract_line(c) for c in crops]
            backend = "tesseract"

        # Strip empty/short trailing lines (the F3 panel ends; the rest is
        # whatever shows through from gameplay, which we don't want).
        while lines and not lines[-1].strip():
            lines.pop()

        # De-duplicate (overlapping crops produce the same line twice).
        # We preserve the FIRST occurrence so positional context for the
        # multi-line "Targeted Block" parser is preserved.
        seen: set = set()
        deduped: List[str] = []
        for ln in lines:
            stripped = ln.strip()
            # Keep blank entries as separators so the line-order-aware
            # parsers (looking_at_block) can still see grouping breaks.
            if not stripped:
                if deduped and deduped[-1] != "":
                    deduped.append("")
                continue
            if stripped in seen:
                continue
            seen.add(stripped)
            deduped.append(ln)

        info = _parse_lines(deduped)
        info.backend = backend
        return info

    def read_position_only(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        return self.read(frame).position()

    def get_debug_strip(self, frame: np.ndarray) -> np.ndarray:
        """Return all line crops binarised and stacked vertically (for debug PNGs)."""
        crops = self._crop_lines(frame)
        if not crops:
            return np.zeros((1, 1), dtype=np.uint8)
        if self._glyph_ocr is not None:
            bins = [self._glyph_ocr.debug_binary(c) for c in crops]
        else:
            bins = [self._binarize_for_tesseract(c) for c in crops]
        max_w = max(b.shape[1] for b in bins)
        padded = []
        for b in bins:
            if b.shape[1] < max_w:
                b = np.concatenate(
                    [b, np.zeros((b.shape[0], max_w - b.shape[1]), dtype=np.uint8)],
                    axis=1,
                )
            padded.append(b)
        return np.concatenate(padded, axis=0)

    def get_debug_image(self, frame: np.ndarray) -> np.ndarray:
        """Backwards-compatible alias for the binarised first line."""
        crops = self._crop_lines(frame)
        if not crops:
            return np.zeros((1, 1), dtype=np.uint8)
        first = crops[0]
        if self._glyph_ocr is not None:
            return self._glyph_ocr.debug_binary(first)
        return self._binarize_for_tesseract(first)

    @property
    def backend(self) -> str:
        return "glyph" if self._glyph_ocr is not None else "tesseract"

    # ------------------------------------------------------------------
    # Line cropping
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # F3-visibility pre-check
    # ------------------------------------------------------------------

    # Sample window for the pre-check, in screen pixels. The F3 panel
    # always starts at the very top-left and is wider than this region,
    # so 60×500 is comfortably inside it whenever F3 is on.
    _PANEL_SAMPLE_H = 60
    _PANEL_SAMPLE_W = 500
    # Empirically measured on vanilla 1.21.x captures (see git history
    # for the test that produced these numbers):
    #   F3 visible:    mean ≈ 49,  std ≈ 64
    #   F3 not visible: mean ≈ 178, std ≈ 27
    # The thresholds below sit comfortably inside that gap.
    _PANEL_MEAN_MAX = 110
    _PANEL_STD_MIN  = 40

    # Drop-shadow text-detection thresholds. The same structural rule
    # the glyph OCR uses (bright pixel with significantly darker
    # below-right neighbour) is the most reliable pre-check: it fires
    # on real MC text in every lighting/biome condition and ignores
    # sky / snow / leaves / sand. Threshold tuned generously so that
    # even a single-line always-on overlay (5-10 shadow pixels per
    # glyph × ~20 glyphs = ~100 signatures) reliably triggers, while
    # a leafy gameplay scene (≤ 5 spurious matches per 30 K-pixel
    # sample window) does not.
    _SHADOW_BRIGHT_MIN  = 200   # this pixel ≥ here
    _SHADOW_DROP_MIN    = 90    # below-right pixel must be that much darker
    _SHADOW_MIN_COUNT   = 6     # >= this many shadow signatures = "text here"

    def _f3_panel_visible(self, frame: np.ndarray) -> bool:
        """
        Return True if the captured frame appears to contain F3 text
        anywhere in the top region — left OR right column. Uses the
        drop-shadow signature so it works in any background condition
        (dark panel, bright sky, snow biome, deep cave).
        """
        h, w = frame.shape[:2]
        if h < 30 or w < 100:
            return False
        # Sample a top strip wide enough to cover BOTH columns.
        # Using the same 60-px height as before keeps the cost low.
        sample = frame[2:min(self._PANEL_SAMPLE_H, h),
                        2:w - 2]
        if sample.ndim == 3:
            gray = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY).astype(np.int16)
        else:
            gray = sample.astype(np.int16)
        bright_here  = gray >= self._SHADOW_BRIGHT_MIN
        drop_to_br   = np.zeros_like(gray, dtype=bool)
        drop_to_br[:-1, :-1] = (gray[:-1, :-1] - gray[1:, 1:]) >= self._SHADOW_DROP_MIN
        return int((bright_here & drop_to_br).sum()) >= self._SHADOW_MIN_COUNT

    def _crop_lines(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Return one crop per visible F3 line. Modern MC renders two
        columns of debug text — we crop and return crops from both
        the LEFT and RIGHT slabs (interleaved by line index) so a
        single ``read()`` recovers fields from either side.

        The downstream :func:`_parse_lines` function walks all
        returned lines and accumulates fields wherever they appear,
        so the LEFT/RIGHT ordering does not matter.
        """
        cfg = self.cfg
        fh, fw = frame.shape[:2]
        crop_h = cfg.line_height_px + 2 * cfg.line_margin_px
        step   = max(1, cfg.line_step_px)

        crops: List[np.ndarray] = []
        # Right-column slab x-range (computed once).
        right_x1 = fw - cfg.right_column_margin_px
        right_x0 = max(0, right_x1 - cfg.right_column_width_px)

        for i in range(cfg.max_lines):
            y0 = cfg.text_start_y + i * step - cfg.line_margin_px
            y1 = y0 + crop_h
            y0 = max(0, y0); y1 = min(fh, y1)
            if y1 <= y0 or (y1 - y0) < cfg.line_height_px // 2:
                break

            # LEFT column.
            lx0 = cfg.text_start_x
            lx1 = min(fw, lx0 + cfg.line_width_px)
            if lx1 > lx0:
                crops.append(frame[y0:y1, lx0:lx1])

            # RIGHT column (only when enabled and there is room).
            if cfg.enable_right_column and right_x1 > right_x0:
                if right_x0 >= lx1:
                    crops.append(frame[y0:y1, right_x0:right_x1])

        return crops

    # ------------------------------------------------------------------
    # Tesseract fallback (kept simple — glyph OCR is the primary path)
    # ------------------------------------------------------------------

    def _binarize_for_tesseract(self, line_img: np.ndarray) -> np.ndarray:
        if line_img.ndim == 3:
            gray = cv2.cvtColor(line_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = line_img
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        scale = max(1, int(self.cfg.ui_scale)) * 2
        h, w = binary.shape
        return cv2.resize(binary, (w * scale, h * scale),
                          interpolation=cv2.INTER_NEAREST)

    def _tesseract_line(self, line_img: np.ndarray) -> str:
        if pytesseract is None:
            return ""
        binary = self._binarize_for_tesseract(line_img)
        try:
            return pytesseract.image_to_string(
                binary,
                config="--psm 7 --oem 3 -l eng",
            ).strip()
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# Builder used by main.py + pipeline_test.py
# ---------------------------------------------------------------------------

def build_f3_reader(settings: Dict[str, Any],
                    *,
                    font_cache_path: Optional[str] = None) -> F3Reader:
    """
    Build an F3Reader from a settings dict.

    The first call extracts and caches Minecraft's default font into
    ``data/calibration/mc_font.npz``; subsequent calls reuse the cache.
    If no Minecraft JAR can be found, the reader falls back to Tesseract
    (with a clearly tagged ``info.backend == "tesseract"``).

    Settings keys read:
        capture.ui_scale            — int, default 2
        vision.ocr.text_threshold   — int, default 180
        vision.ocr.match_threshold  — float, default 0.90
        vision.ocr.max_lines        — int, default 12
        vision.ocr.line_width_px    — int, default 700
        vision.ocr.enable_tesseract_fallback — bool, default True
    """
    cap_cfg = (settings or {}).get("capture", {}) or {}
    ocr_cfg = (settings or {}).get("vision", {}).get("ocr", {}) or {}

    ui_scale = max(1, int(cap_cfg.get("ui_scale", 2)))

    cfg = OCRConfig(
        ui_scale       = ui_scale,
        line_height_px = 8 * ui_scale,         # 16 at scale 2 (one glyph row)
        line_gap_px    = max(1, ui_scale),
        text_start_y   = max(1, ui_scale),
        text_start_x   = max(1, ui_scale),
        line_margin_px = 1,
        line_step_px   = 9 * ui_scale,         # 18 at scale 2 = MC F3 stride
    )
    if "text_threshold" in ocr_cfg:
        cfg.text_threshold = int(ocr_cfg["text_threshold"])
    if "match_threshold" in ocr_cfg:
        cfg.match_threshold = float(ocr_cfg["match_threshold"])
    if "max_lines" in ocr_cfg:
        cfg.max_lines = int(ocr_cfg["max_lines"])
    if "line_width_px" in ocr_cfg:
        cfg.line_width_px = int(ocr_cfg["line_width_px"])
    if "line_step_px" in ocr_cfg:
        cfg.line_step_px = int(ocr_cfg["line_step_px"])
    if "enable_right_column" in ocr_cfg:
        cfg.enable_right_column = bool(ocr_cfg["enable_right_column"])
    if "right_column_width_px" in ocr_cfg:
        cfg.right_column_width_px = int(ocr_cfg["right_column_width_px"])
    if "right_column_margin_px" in ocr_cfg:
        cfg.right_column_margin_px = int(ocr_cfg["right_column_margin_px"])
    if "enable_tesseract_fallback" in ocr_cfg:
        cfg.enable_tesseract_fallback = bool(ocr_cfg["enable_tesseract_fallback"])

    if font_cache_path is None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        font_cache_path = os.path.join(root, "data", "calibration", "mc_font.npz")

    templates: Optional[Dict[str, np.ndarray]] = None
    try:
        templates = ensure_font_cache(font_cache_path)
    except Exception as e:
        if not (cfg.enable_tesseract_fallback and _TESSERACT_AVAILABLE):
            raise RuntimeError(
                "Could not load MC font templates and Tesseract fallback is "
                f"unavailable. Underlying error: {e}"
            ) from e
        # else: continue with Tesseract fallback.

    return F3Reader(cfg, templates=templates)
