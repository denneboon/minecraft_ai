#!/usr/bin/env python3
"""
Offline self-test for F3 overlay OCR on hard backgrounds (no Minecraft).

Locks in the bright-/busy-background robustness work so it can't silently
regress:

  * BRIGHT terrain (desert sand, snow, daytime): the translucent F3 box
    is DARKER than its surroundings, so a global bright-threshold drops
    the glyphs — the reader must use the local-mean "adaptive" binariser
    (GlyphOCRConfig.bright_method="adaptive", the build_f3_reader default).
  * The visibility pre-check (F3Reader._f3_panel_visible) must not
    false-negative on bright terrain (its drop-shadow threshold once sat
    ABOVE the text's actual contrast there and skipped OCR entirely).
  * A BUSY local background — a coloured block bleeding into one debug
    line — can defeat any single adaptive window, so recognize_line
    retries garbled lines with alternative binarisations and keeps the
    cleanest decode.

Fixture ``tests/test_vision/fixtures/f3_desert_busy_bg.png`` is a real
1920x1094 desert god-bridge capture: bright sand behind the right-column
debug text with green concrete bleeding into the Facing line. Before this
work it read ``yaw=None pitch=None`` (XYZ parsed, Facing was garbage).

Run: python tools/test_ocr_f3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

import main as M
from vision.ocr import build_f3_reader

FIXTURES = ROOT / "tests" / "test_vision" / "fixtures"

_fail = 0


def ok(msg):
    print(f"  [ok] {msg}")


def bad(msg):
    global _fail
    _fail += 1
    print(f"  [FAIL] {msg}")


def _load_rgb(name):
    bgr = cv2.imread(str(FIXTURES / name))
    if bgr is None:
        bad(f"fixture missing/unreadable: {name}")
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def test_desert_busy_background():
    frame = _load_rgb("f3_desert_busy_bg.png")
    if frame is None:
        return
    reader = build_f3_reader(M._load_yaml(M.SETTINGS_PATH))

    (ok if reader._f3_panel_visible(frame) else bad)(
        "panel visible on bright sand (pre-check not false-negative)")

    info = reader.read(frame)
    (ok if info.position() is not None else bad)(
        f"XYZ parsed -> {info.position()}")
    (ok if info.yaw is not None else bad)(
        f"Facing-line yaw parsed (busy bg) -> {info.yaw}")
    (ok if info.pitch is not None else bad)(
        f"Facing-line pitch parsed (busy bg) -> {info.pitch}")

    pos = info.position()
    if pos is not None:
        x, y, z = pos
        (ok if (round(x), round(y), round(z)) == (-186, -8, 549) else bad)(
            f"XYZ value sane -> ({x}, {y}, {z})")
    (ok if info.facing_name == "south" else bad)(
        f"facing direction -> {info.facing_name}")


def test_high_altitude_faint_right_column():
    """A high-altitude view where the right-column F3 text is very faint
    over bright distant sand, with bare sand to the LEFT of the right-
    aligned text. The bare-sand noise fooled the line-band finder on the
    full-width crop; the horizontally-anchored sub-crop retry recovers
    it. Every field must parse."""
    frame = _load_rgb("f3_high_altitude_faint.png")
    if frame is None:
        return
    reader = build_f3_reader(M._load_yaml(M.SETTINGS_PATH))
    info = reader.read(frame)
    (ok if info.position() is not None else bad)(
        f"faint XYZ recovered via sub-crop -> {info.position()}")
    (ok if info.yaw is not None and info.pitch is not None else bad)(
        f"faint Facing recovered -> yaw={info.yaw} pitch={info.pitch}")
    pos = info.position()
    if pos is not None:
        x, y, z = pos
        (ok if (round(x), round(y), round(z)) == (210, 8, 516) else bad)(
            f"high-altitude XYZ value sane -> ({x}, {y}, {z})")


def test_sign_flip_rejected():
    """A dropped minus sign in the XYZ line (float disagrees grossly with
    the integer Block line) must be rejected, not handed back as a
    teleported pose."""
    from vision.ocr import _parse_lines

    # y read as +8.0 (sign dropped) while Block correctly says -8.
    flipped = _parse_lines([
        "XYZ: -186.211 / 8.00000 / 548.700",
        "Block: -186 -8 548",
    ])
    (ok if flipped.position() is None else bad)(
        f"sign-flipped XYZ rejected -> {flipped.position()}")

    # Consistent float + block: keep full float precision.
    clean = _parse_lines([
        "XYZ: -186.211 / -8.00000 / 548.700",
        "Block: -186 -8 548",
    ])
    good = clean.position() is not None and round(clean.position()[1]) == -8
    (ok if good else bad)(
        f"consistent XYZ kept at full precision -> {clean.position()}")

    # XYZ line absent but Block present: fall back to block ints.
    blockonly = _parse_lines(["Block: -186 -8 548"])
    (ok if blockonly.position() == (-186.0, -8.0, 548.0) else bad)(
        f"block-only fallback -> {blockonly.position()}")


def test_facing_angles_survive_garbled_cardinal():
    """Yaw/pitch must parse from the Facing line even when the cardinal
    WORD is OCR-garbled but the angle pair is clean (the bright-terrain
    failure that stalled the god-bridge precise-yaw step)."""
    from vision.ocr import _parse_lines

    # "south" garbled to "sout|:?", angles clean.
    garbled = _parse_lines([
        "?Facing: sout|:? (Towards positive Z) (45.0 / 45.0)",
    ])
    (ok if garbled.yaw == 45.0 and garbled.pitch == 45.0 else bad)(
        f"angles parsed despite garbled cardinal -> yaw={garbled.yaw} "
        f"pitch={garbled.pitch}")

    # A block-state property line with a cardinal but NO angle pair must
    # NOT be mistaken for the Facing line.
    prop = _parse_lines(["    east: true"])
    (ok if prop.yaw is None and prop.pitch is None else bad)(
        f"property line not mistaken for Facing -> yaw={prop.yaw}")


def test_deterministic():
    frame = _load_rgb("f3_desert_busy_bg.png")
    if frame is None:
        return
    reader = build_f3_reader(M._load_yaml(M.SETTINGS_PATH))
    a = reader.read(frame)
    b = reader.read(frame)
    same = (a.yaw, a.pitch, a.position()) == (b.yaw, b.pitch, b.position())
    (ok if same else bad)("read is deterministic (same frame -> same pose)")


def test_read_budget_caps_busy_read():
    """The read budget (OCRConfig.read_budget_ms) bounds the glyph-OCR retry
    cascade so a busy/garbled scene can't blow a read up to seconds (it hit
    ~3.7s live, vs ~86ms clean). Guarantees: a tight budget is never SLOWER
    than unlimited, and the pose still parses (the primary per-line decode
    always runs, so xyz survives budgeting)."""
    import time
    frame = _load_rgb("f3_desert_busy_bg.png")
    if frame is None:
        return
    reader = build_f3_reader(M._load_yaml(M.SETTINGS_PATH))

    reader.cfg.read_budget_ms = 0          # unlimited -> full cascade
    reader.read(frame)                     # warm
    t = time.perf_counter()
    for _ in range(3):
        reader.read(frame)
    slow = (time.perf_counter() - t) / 3

    reader.cfg.read_budget_ms = 50         # tight -> cascade bails early
    t = time.perf_counter()
    for _ in range(3):
        info = reader.read(frame)
    fast = (time.perf_counter() - t) / 3

    (ok if fast <= slow + 0.010 else bad)(
        f"tight budget never slower than unlimited ({fast*1000:.0f}ms "
        f"<= {slow*1000:.0f}ms)")
    (ok if info.position() is not None else bad)(
        f"pose still parses under a tight budget -> {info.position()}")


def main():
    print("=" * 60)
    print(" vision.ocr F3 hard-background self-test")
    print("=" * 60)
    test_desert_busy_background()
    test_high_altitude_faint_right_column()
    test_sign_flip_rejected()
    test_facing_angles_survive_garbled_cardinal()
    test_deterministic()
    test_read_budget_caps_busy_read()
    print("=" * 60)
    if _fail:
        print(f" {_fail} CHECK(S) FAILED")
        return 1
    print(" ALL F3 OCR TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
