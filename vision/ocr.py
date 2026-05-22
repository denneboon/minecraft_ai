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

    # Read at most this many lines from the top of the F3 panel.
    # Players can toggle individual debug options to Always, so the
    # actual line count varies — scanning generously costs us nothing.
    max_lines: int = 12
    line_width_px: int = 700

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

_RE_XYZ_DEC   = re.compile(r'(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)')
_RE_BLOCK_INT = re.compile(r'(-?\d+)\s+(-?\d+)\s+(-?\d+)')
_RE_DIM       = re.compile(r'minecraft:[a-z_]+', re.IGNORECASE)
_RE_FPS       = re.compile(r'(\d+)\s*fps', re.IGNORECASE)
_RE_FACING    = re.compile(r'facing[:\s]+(north|south|east|west)\b', re.IGNORECASE)
_RE_ANGLES    = re.compile(r'(-?\d+\.\d+)\s*/\s*(-?\d+\.\d+)')

_Y_MIN, _Y_MAX = -64, 320
_XZ_LIMIT      = 30_000_000


def _coords_in_bounds(x: float, y: float, z: float) -> bool:
    return abs(x) < _XZ_LIMIT and _Y_MIN <= y <= _Y_MAX and abs(z) < _XZ_LIMIT


def _parse_lines(lines: List[str]) -> F3Info:
    """Walk every line and accumulate whatever fields can be recognised."""
    info = F3Info()

    for line in lines:
        if not line:
            continue
        ll = line.lower()

        # XYZ position (decimals separated by /)
        if info.x is None:
            m = _RE_XYZ_DEC.search(line)
            if m:
                try:
                    x, y, z = float(m.group(1)), float(m.group(2)), float(m.group(3))
                    if _coords_in_bounds(x, y, z):
                        info.x, info.y, info.z = x, y, z
                except ValueError:
                    pass

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

        # Dimension
        if info.dimension is None:
            md = _RE_DIM.search(line)
            if md:
                info.dimension = md.group(0).lower()

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

        info = _parse_lines(lines)
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

    def _f3_panel_visible(self, frame: np.ndarray) -> bool:
        h, w = frame.shape[:2]
        if h < 30 or w < 100:
            return False
        sample = frame[2:min(self._PANEL_SAMPLE_H, h),
                       2:min(self._PANEL_SAMPLE_W, w)]
        if sample.ndim == 3:
            gray = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
        else:
            gray = sample
        return bool(
            gray.mean() < self._PANEL_MEAN_MAX
            and gray.std()  > self._PANEL_STD_MIN
        )

    def _crop_lines(self, frame: np.ndarray) -> List[np.ndarray]:
        cfg = self.cfg
        fh, fw = frame.shape[:2]
        lh = cfg.line_height_px
        lg = cfg.line_gap_px
        m  = cfg.line_margin_px

        crops: List[np.ndarray] = []
        for i in range(cfg.max_lines):
            y0 = cfg.text_start_y + i * (lh + lg) - m
            y1 = y0 + lh + 2 * m
            x0 = cfg.text_start_x
            x1 = x0 + cfg.line_width_px

            y0 = max(0, y0); y1 = min(fh, y1)
            x0 = max(0, x0); x1 = min(fw, x1)
            if y1 <= y0 or x1 <= x0:
                break
            crops.append(frame[y0:y1, x0:x1])
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
        line_height_px = 8 * ui_scale,
        line_gap_px    = max(1, ui_scale),
        text_start_y   = max(1, ui_scale),
        text_start_x   = max(1, ui_scale),
        line_margin_px = 1,
    )
    if "text_threshold" in ocr_cfg:
        cfg.text_threshold = int(ocr_cfg["text_threshold"])
    if "match_threshold" in ocr_cfg:
        cfg.match_threshold = float(ocr_cfg["match_threshold"])
    if "max_lines" in ocr_cfg:
        cfg.max_lines = int(ocr_cfg["max_lines"])
    if "line_width_px" in ocr_cfg:
        cfg.line_width_px = int(ocr_cfg["line_width_px"])
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
