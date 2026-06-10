# tools/pipeline_test.py
"""
End-to-end pipeline smoke test.

Run this with Minecraft open and in-game (not paused, not in a menu).

Usage:
    python tools/pipeline_test.py
"""

from __future__ import annotations

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Force UTF-8 stdout/stderr so the box-drawing characters in section banners
# render correctly on Windows consoles that default to cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):    print(f"  {GREEN}PASS{RESET}  {msg}")
def fail(msg):  print(f"  {RED}FAIL{RESET}  {msg}")
def warn(msg):  print(f"  {YELLOW}WARN{RESET}  {msg}")
def section(t): print(f"\n{BOLD}{'─'*60}\n  {t}\n{'─'*60}{RESET}")

passed = []
failed = []

def check(name, fn):
    try:
        result = fn()
        ok(name)
        passed.append(name)
        return result
    except Exception as e:
        fail(f"{name}\n         {RED}{type(e).__name__}: {e}{RESET}")
        failed.append(name)
        return None


# ═════════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ═════════════════════════════════════════════════════════════════════════════
section("1 · Configuration")

def _test_yaml_load():
    import yaml
    path = os.path.join(ROOT, "config", "settings.yaml")
    assert os.path.isfile(path), f"settings.yaml not found at {path}"
    with open(path, "r", encoding="utf-8") as f:
        s = yaml.safe_load(f)
    assert isinstance(s, dict)
    assert s.get("capture", {}).get("ui_scale") == 2, \
        f"ui_scale should be 2, got {s.get('capture',{}).get('ui_scale')}"
    return s

settings = check("settings.yaml loads + ui_scale==2", _test_yaml_load)

def _test_keymap_load():
    import json
    path = os.path.join(ROOT, "config", "keymap.json")
    with open(path, "r", encoding="utf-8") as f:
        k = json.load(f)
    assert "movement" in k
    return k

check("keymap.json loads", _test_keymap_load)

def _test_hud_regions():
    assert settings is not None
    hud = settings.get("vision", {}).get("hud_regions", {})
    for key in ("health_bar", "hunger_bar", "hotbar_region", "xp_bar",
                "armor_bar", "crosshair_center"):
        assert key in hud, f"missing: {key}"
        assert all(v > 0 for v in hud[key]), f"{key} has zero values: {hud[key]}"

check("HUD regions present and non-zero", _test_hud_regions)

def _test_slot_rects():
    assert settings is not None
    slots = (settings.get("vision", {})
                     .get("hud_extended", {})
                     .get("hotbar_slot_rects", []))
    assert len(slots) == 9, f"expected 9 slot rects, got {len(slots)}"
    xs = [s[0] for s in slots]
    assert xs == sorted(xs), "slots not left-to-right"
    for i in range(len(slots) - 1):
        gap = slots[i+1][0] - (slots[i][0] + slots[i][2])
        assert gap == 0, f"gap between slots {i+1} and {i+2}: {gap}px"

check("hotbar_slot_rects: 9 slots, contiguous", _test_slot_rects)


# ═════════════════════════════════════════════════════════════════════════════
# 2. WINDOW / FOCUS
# ═════════════════════════════════════════════════════════════════════════════
section("2 · Window detection & focus")

hwnd = None

def _test_find_hwnd():
    global hwnd
    from utils.focus import _find_minecraft_hwnd
    wins = _find_minecraft_hwnd()
    assert wins, "No Minecraft window found (is javaw.exe running?)"
    hwnd = wins[0][0]
    rect = wins[0][1]
    w, h = rect[2] - rect[0], rect[3] - rect[1]
    assert w >= 800 and h >= 600, f"Window too small: {w}x{h}"
    print(f"         hwnd={hwnd}  size={w}x{h}")
    return hwnd

check("Minecraft window found", _test_find_hwnd)

def _test_focus_and_maximize():
    """Focus and maximize Minecraft so captures are full-res."""
    assert hwnd is not None, "no hwnd"
    import win32gui, win32con, win32process, win32api, ctypes
    user32 = ctypes.windll.user32

    win32gui.ShowWindow(hwnd, win32con.SW_SHOWMAXIMIZED)
    fg = win32gui.GetForegroundWindow()
    fg_tid, _ = win32process.GetWindowThreadProcessId(fg) if fg else (0, 0)
    tgt_tid, _ = win32process.GetWindowThreadProcessId(hwnd)
    cur = win32api.GetCurrentThreadId()
    user32.AttachThreadInput(cur, fg_tid, True)
    user32.AttachThreadInput(cur, tgt_tid, True)
    win32gui.BringWindowToTop(hwnd)
    win32gui.SetForegroundWindow(hwnd)
    win32gui.SetActiveWindow(hwnd)
    win32gui.SetFocus(hwnd)
    user32.AttachThreadInput(cur, fg_tid, False)
    user32.AttachThreadInput(cur, tgt_tid, False)
    time.sleep(0.5)  # wait for maximize animation

    fg_after = win32gui.GetForegroundWindow()
    assert fg_after == hwnd, \
        f"Minecraft is not foreground after focus attempt (fg={fg_after}, mc={hwnd})"

check("Focus + maximize Minecraft", _test_focus_and_maximize)


# ═════════════════════════════════════════════════════════════════════════════
# 3. CAPTURE
# ═════════════════════════════════════════════════════════════════════════════
section("3 · Screen capture")

frame = None

def _test_capture():
    global frame
    from vision.capture import Capture, CaptureConfig
    cap = Capture(CaptureConfig(
        hwnd=hwnd,
        use_client_area=True,
        max_fps=30,
        track_window_each_frame=True,
    ))
    cap.start()
    f = cap.get_frame()
    cap.stop()
    assert f is not None and f.size > 0
    assert f.dtype == np.uint8
    assert f.ndim == 3 and f.shape[2] == 3
    frame = f
    print(f"         frame shape: {f.shape}  mean brightness: {f.mean():.1f}")
    return f

check("Capture returns valid RGB frame", _test_capture)

def _test_frame_not_black():
    assert frame is not None
    mean = float(frame.mean())
    assert mean > 20,  f"Frame looks black (mean={mean:.1f})"
    assert mean < 240, f"Frame looks blown-out (mean={mean:.1f})"

check("Frame brightness looks like gameplay", _test_frame_not_black)

def _test_frame_resolution():
    assert frame is not None and settings is not None
    h, w = frame.shape[:2]
    calib_res = (settings.get("vision_calibration", {}) or {}) \
                    .get("last", {}).get("resolution")
    if calib_res and len(calib_res) == 2:
        cw, ch = int(calib_res[0]), int(calib_res[1])
        if w == cw and h == ch:
            print(f"         {w}x{h} matches last calibration ({cw}x{ch})")
            return
        if w >= 1280 and h >= 720:
            warn(
                f"Frame is {w}x{h}, but last calibration was {cw}x{ch}.\n"
                f"         HUD pixel coordinates in settings.yaml may be off.\n"
                f"         Re-run calibration:\n"
                f"           python vision/calibration.py"
            )
            return
    else:
        if w >= 1280 and h >= 720:
            warn(
                f"No calibration recorded. Frame is {w}x{h}. "
                f"Run: python vision/calibration.py"
            )
            return
    raise AssertionError(f"Frame too small: {w}x{h}")

check("Frame resolution", _test_frame_resolution)

def _save_capture_debug():
    import cv2
    assert frame is not None
    out = os.path.join(ROOT, "data", "calibration", "pipeline_test_capture.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"         saved → {out}")

check("Save capture to data/calibration/", _save_capture_debug)


# ═════════════════════════════════════════════════════════════════════════════
# 4. FRAME PROCESSING
# ═════════════════════════════════════════════════════════════════════════════
section("4 · Frame processing → GameState")

state = None

def _test_processor_builds():
    from vision.processing import build_processor
    assert settings is not None
    return build_processor(settings)

processor = check("FrameProcessor builds", _test_processor_builds)

def _test_process_frame():
    global state
    assert processor is not None and frame is not None
    s = processor.process(frame)
    assert s is not None
    assert s.frame is not None and s.frame.size > 0
    assert s.screen_state in ("playing", "paused", "menu", "loading", "unknown")
    state = s
    print(f"         screen_state : {s.screen_state}")
    print(f"         health={s.health:.2f}  hunger={s.hunger:.2f}  "
          f"armor={s.armor:.2f}  xp={s.xp_bar:.2f}")
    print(f"         hotbar_slot  : {s.hotbar_slot}")
    print(f"         model frame  : {s.frame.shape}")
    return s

check("process(frame) → GameState", _test_process_frame)

def _test_screen_state_playing():
    assert state is not None
    if state.screen_state != "playing":
        raise AssertionError(
            f"screen_state='{state.screen_state}'. "
            f"Make sure Minecraft is in-game, not paused. "
            f"Frame mean brightness was {frame.mean():.1f}."
        )

check("screen_state == 'playing'", _test_screen_state_playing)

def _test_hud_values_sane():
    assert state is not None
    for name, val in [("health", state.health), ("hunger", state.hunger),
                      ("armor", state.armor), ("xp_bar", state.xp_bar)]:
        assert 0.0 <= val <= 1.0, f"{name}={val} out of range"
    assert 1 <= state.hotbar_slot <= 9, f"hotbar_slot={state.hotbar_slot}"

check("HUD values in valid ranges", _test_hud_values_sane)


# ═════════════════════════════════════════════════════════════════════════════
# 5. OCR
# ═════════════════════════════════════════════════════════════════════════════
section("5 · OCR — F3 overlay reader")

f3_info  = None
f3_reader = None

def _test_tesseract_available():
    import pytesseract
    ver = pytesseract.get_tesseract_version()
    print(f"         Tesseract version: {ver}")

check("Tesseract binary reachable", _test_tesseract_available)

def _test_f3_reader_builds():
    global f3_reader
    from vision.ocr import build_f3_reader
    assert settings is not None
    f3_reader = build_f3_reader(settings)
    print(f"         backend      : {f3_reader.backend}")
    return f3_reader

check("F3Reader builds", _test_f3_reader_builds)

def _test_f3_read():
    global f3_info
    assert f3_reader is not None and frame is not None
    info = f3_reader.read(frame)
    f3_info = info
    preview = repr(info.raw_text[:300]) if info.raw_text else "(empty)"
    print(f"         OCR output preview:\n           {preview}")
    return info

check("F3Reader.read() runs", _test_f3_read)

def _test_f3_position():
    assert f3_info is not None
    assert f3_info.x is not None, (
        "XYZ not parsed. Make sure 'player_position' is set to Always "
        "in F3 debug options (F3+F6 menu).\n"
        f"         OCR saw: {repr(f3_info.raw_text[:200])}"
    )
    print(f"         x={f3_info.x:.2f}  y={f3_info.y:.2f}  z={f3_info.z:.2f}")
    if f3_info.block_position():
        bx, by, bz = f3_info.block_position()
        print(f"         block        : {bx} {by} {bz}")
    if f3_info.dimension:
        print(f"         dimension    : {f3_info.dimension}")

check("F3: XYZ parsed", _test_f3_position)

def _test_f3_facing():
    assert f3_info is not None
    assert f3_info.facing_name is not None, (
        "Facing direction not parsed.\n"
        f"         OCR saw: {repr(f3_info.raw_text[:200])}"
    )
    yaw_str = f"{f3_info.yaw:.1f}" if f3_info.yaw is not None else "n/a (best-effort)"
    pitch_str = f"{f3_info.pitch:.1f}" if f3_info.pitch is not None else "n/a"
    print(f"         facing={f3_info.facing_name}  yaw={yaw_str}  pitch={pitch_str}")

check("F3: facing + yaw/pitch parsed", _test_f3_facing)

def _save_ocr_debug():
    import cv2
    assert f3_reader is not None and frame is not None
    out_dir = os.path.join(ROOT, "data", "calibration")
    os.makedirs(out_dir, exist_ok=True)
    binary = f3_reader.get_debug_image(frame)
    cv2.imwrite(os.path.join(out_dir, "pipeline_test_ocr_input.png"), binary)
    # Save the full strip too — much more useful when XYZ parsing fails.
    if hasattr(f3_reader, "get_debug_strip"):
        strip = f3_reader.get_debug_strip(frame)
        cv2.imwrite(os.path.join(out_dir, "pipeline_test_ocr_strip.png"), strip)
    print(f"         saved → {out_dir}\\pipeline_test_ocr_*.png")

check("Save OCR debug image", _save_ocr_debug)


# ═════════════════════════════════════════════════════════════════════════════
# 6. CONTROL  (read-only)
# ═════════════════════════════════════════════════════════════════════════════
section("6 · Control subsystems  (no input sent)")

def _test_input_gate():
    from control.input_gate import InputGate
    g = InputGate()
    assert not g.allow()
    g.set_allowed(True);  assert g.allow()
    g.set_allowed(False); assert not g.allow()

check("InputGate", _test_input_gate)

def _test_keyboard_builds():
    import json
    from main import _flatten_keymap_for_keyboard, build_keyboard
    with open(os.path.join(ROOT, "config", "keymap.json")) as f:
        raw = json.load(f)
    return build_keyboard(settings, _flatten_keymap_for_keyboard(raw))

check("Keyboard builds", _test_keyboard_builds)

def _test_mouse_builds():
    from main import build_mouse
    return build_mouse(settings)

check("Mouse builds", _test_mouse_builds)

def _test_action_wrapper_builds():
    import json
    from main import build_keyboard, build_mouse, _flatten_keymap_for_keyboard
    from control.action_wrapper import ActionWrapper
    with open(os.path.join(ROOT, "config", "keymap.json")) as f:
        raw = json.load(f)
    flat = _flatten_keymap_for_keyboard(raw)
    aw = ActionWrapper(keyboard=build_keyboard(settings, flat),
                       mouse=build_mouse(settings))
    return aw

check("ActionWrapper builds", _test_action_wrapper_builds)


# ═════════════════════════════════════════════════════════════════════════════
# 7. LIVE INPUT TEST
# ═════════════════════════════════════════════════════════════════════════════
section("7 · Live input test  (OPTIONAL — moves mouse + hotbar)")

print("\n  Minecraft is already focused from step 2.")
print("  This will: move mouse right+back, then select hotbar slots 1-2-3-1.")
answer = input("  Run live input test? [y/N] ").strip().lower()

if answer == "y":
    def _test_live_input():
        import json
        from control.input_gate import InputGate
        from main import build_mouse, build_keyboard, _flatten_keymap_for_keyboard

        # Re-focus Minecraft before sending any input
        import win32gui, win32con, win32process, win32api, ctypes
        user32 = ctypes.windll.user32
        win32gui.ShowWindow(hwnd, win32con.SW_SHOWMAXIMIZED)
        fg = win32gui.GetForegroundWindow()
        fg_tid, _ = win32process.GetWindowThreadProcessId(fg) if fg else (0, 0)
        tgt_tid, _ = win32process.GetWindowThreadProcessId(hwnd)
        cur = win32api.GetCurrentThreadId()
        user32.AttachThreadInput(cur, fg_tid, True)
        user32.AttachThreadInput(cur, tgt_tid, True)
        win32gui.SetForegroundWindow(hwnd)
        win32gui.SetFocus(hwnd)
        user32.AttachThreadInput(cur, fg_tid, False)
        user32.AttachThreadInput(cur, tgt_tid, False)
        time.sleep(0.4)

        gate = InputGate()
        gate.set_allowed(True)

        # try/finally around start...stop so a mid-test crash doesn't
        # leak the velocity worker. Without this, an exception in
        # the motion or hotbar sequence would skip ``ms.stop()`` and
        # the daemon thread would keep emitting motion until process
        # exit — the "mouse still moves after the program says it's
        # done" symptom. Same for the keyboard.
        ms = build_mouse(settings, gate=gate)
        ms.start()
        try:
            ms.track_target(40, 0)
            time.sleep(0.3)
            ms.track_target(-40, 0)
            time.sleep(0.2)
        finally:
            ms.stop()

        with open(os.path.join(ROOT, "config", "keymap.json")) as f:
            raw = json.load(f)
        flat = _flatten_keymap_for_keyboard(raw)
        kb = build_keyboard(settings, flat, gate=gate)
        kb.start()
        try:
            for slot in [1, 2, 3, 1]:
                kb.select_hotbar_slot(slot)
        finally:
            kb.stop()

    check("Live mouse + hotbar (Minecraft focused)", _test_live_input)
else:
    warn("Live input test skipped.")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
section("Summary")

total = len(passed) + len(failed)
print(f"\n  {GREEN}{len(passed)}{RESET} passed   {RED}{len(failed)}{RESET} failed   {total} total\n")

if failed:
    print(f"  {RED}Failed tests:{RESET}")
    for f in failed:
        print(f"    • {f}")
    print()
    sys.exit(1)
else:
    print(f"  {GREEN}{BOLD}All tests passed. Pipeline is ready.{RESET}\n")
    sys.exit(0)
