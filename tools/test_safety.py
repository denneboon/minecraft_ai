#!/usr/bin/env python3
"""
Offline safety tests — focused on the emergency-stop hotkey matching, which
is a CRITICAL operator-safety path: if it silently never fires, there is no
way to stop the bot.

Regression: pynput reports the LEFT/RIGHT variant of a modifier (pressing
Ctrl yields ``Key.ctrl_l`` -> ``<ctrl_l>``), so a configured ``<ctrl>`` in
the combo never matched and Ctrl+Shift+F12 did nothing. ``_norm_key_name``
collapses the variants so the combo matches what pynput actually emits.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from control.safety import _norm_key_name

_fail = 0


def ok(msg):
    print(f"  [OK]  {msg}")


def bad(msg):
    global _fail
    _fail += 1
    print(f"  [BAD] {msg}")


def _combo_fires(configured, pressed_raw):
    """Mirror Safety._hotkey_loop matching: normalise both the configured
    combo and the keys pynput reports, then check subset."""
    combo = set(_norm_key_name(k) for k in configured)
    pressed = set(_norm_key_name(n) for n in pressed_raw)
    return combo.issubset(pressed)


def test_norm():
    print("[1] _norm_key_name collapses modifier variants")
    (ok if _norm_key_name("<ctrl_l>") == "<ctrl>" else bad)("ctrl_l -> <ctrl>")
    (ok if _norm_key_name("<ctrl_r>") == "<ctrl>" else bad)("ctrl_r -> <ctrl>")
    (ok if _norm_key_name("<shift_l>") == "<shift>" else bad)("shift_l -> <shift>")
    (ok if _norm_key_name("<shift>") == "<shift>" else bad)("shift stays <shift>")
    (ok if _norm_key_name("<alt_gr>") == "<alt>" else bad)("alt_gr -> <alt>")
    (ok if _norm_key_name("<f12>") == "<f12>" else bad)("f12 unchanged")
    (ok if _norm_key_name("x") == "x" else bad)("char key unchanged")


def test_panic_combo():
    print("[2] Ctrl+Shift+F12 panic combo actually fires")
    cfg = ["<ctrl>", "<shift>", "<f12>"]
    (ok if _combo_fires(cfg, ["<ctrl_l>", "<shift>", "<f12>"]) else bad)(
        "left-ctrl press fires (the real-world regression)")
    (ok if _combo_fires(cfg, ["<ctrl_r>", "<shift_r>", "<f12>"]) else bad)(
        "right-side modifiers fire")
    (ok if not _combo_fires(cfg, ["<ctrl_l>", "<f12>"]) else bad)(
        "incomplete combo (no shift) does NOT fire")
    (ok if not _combo_fires(cfg, ["<ctrl_l>", "<shift>"]) else bad)(
        "incomplete combo (no f12) does NOT fire")


def main():
    test_norm()
    test_panic_combo()
    print(f"\n[test_safety] {'OK' if _fail == 0 else f'{_fail} FAILED'}")
    return 1 if _fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
