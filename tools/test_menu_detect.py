#!/usr/bin/env python3
"""Offline self-test for vision/menu_detect.py — the centre-band crop that keeps
the F3 debug overlay (left/right edges) from starving the menu OCR. Regression
for the live bug where a clearly-open 'Game Menu' was missed, so the bot thought
it was playing on a paused/frozen frame and never recovered."""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

import numpy as np
from vision.menu_detect import MenuDetector, MenuDetectorConfig

_fails = 0
def ok(m): print(f"  [OK]  {m}")
def bad(m):
    global _fails; _fails += 1; print(f"  [FAIL] {m}")


class _RecordingOCR:
    """Stand-in for GlyphOCR: records the width of every crop it's handed, and
    returns menu text ONLY for crops drawn from the centre (simulating that the
    pause-menu buttons live in the centre while the F3 overlay sits at the
    edges)."""
    def __init__(self, frame_w):
        self.frame_w = frame_w
        self.crop_widths = []
    def begin_read(self, _budget): pass
    def recognize_line(self, crop):
        self.crop_widths.append(crop.shape[1])
        return "Back to Game"     # the row's text (would be edge-garble at full width)


def main() -> int:
    print("=" * 56); print(" menu detector centre-crop — offline self-test"); print("=" * 56)

    w, h = 1920, 1094
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Build a detector without the heavy glyph templates, then inject the mock.
    md = MenuDetector.__new__(MenuDetector)
    md.cfg = MenuDetectorConfig()
    rec = _RecordingOCR(w)
    md._ocr = rec

    print("\n[1] detect() scans only the CENTRE band, not the full width")
    d = md.detect(frame)
    band = max(rec.crop_widths) if rec.crop_widths else 0
    frac = band / w
    (ok if 0.0 < frac <= 0.85 else bad)(
        f"crop width {band}px is the centre band (~{frac:.0%} of {w}), not full width")
    (ok if band < w else bad)("never OCRs the full width (would include the F3 overlay)")

    print("\n[2] a centred 'Back to Game' is detected as the pause menu")
    (ok if d.open and d.menu == "pause" and d.matched_keyword == "back to game" else bad)(
        f"pause detected (open={d.open}, menu={d.menu}, kw={d.matched_keyword!r})")

    print("\n[3] the centre band matches config x_start/x_end")
    exp = int(w * md.cfg.x_end_frac) - int(w * md.cfg.x_start_frac)
    (ok if band == exp else bad)(f"crop width {band} == configured band {exp}")

    print("\n" + ("ALL MENU-DETECT TESTS PASSED" if not _fails
                  else f"{_fails} CHECK(S) FAILED"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
