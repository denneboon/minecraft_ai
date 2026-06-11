#!/usr/bin/env python3
"""Offline self-test for the stack-count reader — builds synthetic slots
with known counts drawn from the MC font glyphs (white digits, right-
aligned at the bottom, on a slot with a light bevel) and checks the reader
recovers the number. Guards the digit-matcher against regressions."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

from vision.inventory import build_inventory_reader

_fails = 0
def ok(m): print(f"  [OK]  {m}")
def bad(m):
    global _fails; _fails += 1; print(f"  [FAIL] {m}")


def _make_slot(cr, count: int) -> np.ndarray:
    """A 32x32 slot: dark grey well, light bottom/right bevel (the real
    noise source), with ``count`` drawn white + right-aligned at the bottom
    using the reader's own font glyphs."""
    sz = 16 * cr._ui_scale
    slot = np.full((sz, sz, 3), 55, np.uint8)            # dark well
    slot[:, -2:] = 200                                    # right bevel highlight
    slot[-2:, :] = 200                                    # bottom bevel highlight
    def _ink(tb):
        b = np.asarray(tb, bool)
        ys, xs = np.where(b)
        return b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    glyphs = {ch: _ink(tb) for ch, tb in
              zip(cr._ocr._chars, cr._ocr._tpl_bin) if ch in "0123456789"}
    digits = str(count)
    x_right = sz - 2                                      # ~1px from right edge
    y_bot = sz - 2                                        # ~1px from bottom
    for ch in reversed(digits):                          # right-to-left
        g = glyphs[ch]; gh, gw = g.shape
        x0 = x_right - gw
        if x0 < 0:
            break
        slot[y_bot - gh:y_bot, x0:x_right][g] = 255
        x_right = x0 - 1                                  # 1px gap
    return slot


def main() -> int:
    print("=" * 56); print(" stack-count reader — offline self-test"); print("=" * 56)
    reader = build_inventory_reader()
    cr = reader.count
    (ok if cr._digit_norm else bad)(f"digit templates loaded ({len(cr._digit_norm)})")

    print("\n[1] reads single + multi-digit counts off a bevelled slot")
    for n in (2, 5, 8, 9, 12, 16, 23, 47, 56, 64):
        got = cr.read(_make_slot(cr, n))
        (ok if got == n else bad)(f"count {n:>2} -> read {got}")

    print("\n[2] empty / single-item slot reads 0 (no number drawn)")
    sz = 16 * cr._ui_scale
    bare = np.full((sz, sz, 3), 55, np.uint8)
    bare[:, -2:] = 200; bare[-2:, :] = 200               # bevel only, no digits
    (ok if cr.read(bare) == 0 else bad)(f"bevel-only slot -> {cr.read(bare)} (want 0)")

    print("\n" + ("ALL STACK-COUNT TESTS PASSED" if not _fails
                  else f"{_fails} CHECK(S) FAILED"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
