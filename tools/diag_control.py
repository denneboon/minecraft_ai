#!/usr/bin/env python3
"""
Isolate the in-world control pipeline. With MC focused + in gameplay, this:
  1. checks the capture is LIVE (turn camera -> does the frame change?),
  2. checks CAMERA look reaches the game (does F3 yaw change?),
  3. checks MOVEMENT reaches the game (hold W -> does F3 xyz change?),
  4. checks ATTACK reaches the game (hold left -> does anything change?).

    python tools/diag_control.py     # KEEP MINECRAFT FOCUSED for ~8s

Tells us whether the bot can actually drive the world, or is reading a stale
frame / has gated input.
"""
from __future__ import annotations

import os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

import numpy as np
import main as M
from utils.focus import _find_minecraft_hwnd
from control.input_gate import InputGate
from vision.capture import Capture, CaptureConfig
from vision.ocr import build_f3_reader


def _pose(f3, cap):
    fr = cap.get_frame()
    r = f3.read(fr)
    txt = getattr(r, "raw_text", "") or ""
    return fr, txt


def _xyz_yaw(txt):
    import re
    m = re.search(r"XYZ:\s*([-0-9.]+)\s*/\s*([-0-9.]+)\s*/\s*([-0-9.]+)", txt)
    y = re.search(r"\(([-0-9.]+)\s*/\s*[-0-9.]+\)\s*$", txt.strip().splitlines()[-1]) if txt else None
    yaw = re.search(r"\(([-0-9.]+)\s*/\s*[-0-9.]+\)", txt)
    xyz = (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else None
    return xyz, (float(yaw.group(1)) if yaw else None)


def main() -> int:
    wins = _find_minecraft_hwnd()
    if not wins:
        print("[ctl] MC not found"); return 2
    hwnd = wins[0][0]
    settings = M._load_yaml(M.SETTINGS_PATH)
    keymap = M._flatten_keymap_for_keyboard(M._load_json(M.KEYMAP_PATH))
    gate = InputGate()
    safety = M.build_safety(settings, gate=gate)
    mouse = M.build_mouse(settings, gate=gate)
    kb = M.build_keyboard(settings, keymap, gate=gate)
    capture = Capture(CaptureConfig(hwnd=hwnd, use_client_area=True, threaded=True))
    safety.start(); mouse.start(); kb.start(); capture.start(); time.sleep(0.4)
    f3 = build_f3_reader(settings)
    menu = M.build_menu_detector_default(settings)
    try:
        ok, reason = M.ensure_controllable(capture, menu, kb, gate)
        print(f"[ctl] controllable={ok} ({reason}); gate.allow()={gate.allow()}")
        if not ok:
            return 1

        # 1+2. CAMERA: turn right ~30deg, check frame + yaw change.
        frA, tA = _pose(f3, capture); xA, yawA = _xyz_yaw(tA)
        for _ in range(6):
            mouse.move(60, 0); time.sleep(0.08)
        time.sleep(0.3)
        frB, tB = _pose(f3, capture); xB, yawB = _xyz_yaw(tB)
        fdiff = float(np.mean(np.abs(frA.astype(int) - frB.astype(int)))) if frA is not None and frB is not None else -1
        print(f"[ctl] CAMERA: frame-diff={fdiff:.2f} (>1 = live capture), "
              f"yaw {yawA} -> {yawB} ({'MOVED' if yawA is not None and yawB is not None and abs(yawB-yawA)>2 else 'NO CHANGE'})")

        # 3. MOVEMENT: hold forward ~1.5s, check xyz change.
        from control.action_wrapper import ActionWrapper
        act = ActionWrapper(kb, mouse, gate=gate)
        frC, tC = _pose(f3, capture); xC, _ = _xyz_yaw(tC)
        act.set_movement_state(forward=True, backward=False, left=False, right=False,
                               jump=False, sprint=False, sneak=False)
        time.sleep(1.6)
        act.release_all_movement()
        time.sleep(0.3)
        frD, tD = _pose(f3, capture); xD, _ = _xyz_yaw(tD)
        moved = (xC and xD and (abs(xD[0]-xC[0]) + abs(xD[2]-xC[2])) > 0.3)
        print(f"[ctl] MOVEMENT(hold W): xyz {xC} -> {xD} ({'WALKED' if moved else 'NO CHANGE'})")

        # 4. ATTACK: hold left ~1.2s (does the frame change? swing/break).
        frE, _ = _pose(f3, capture)
        mouse.left_press(); time.sleep(1.2); mouse.left_release()
        time.sleep(0.2)
        frF, _ = _pose(f3, capture)
        adiff = float(np.mean(np.abs(frE.astype(int) - frF.astype(int)))) if frE is not None and frF is not None else -1
        print(f"[ctl] ATTACK(hold left): frame-diff={adiff:.2f} (>1 = something happened)")
        return 0
    finally:
        try: act.release_all_movement()
        except Exception: pass
        try: kb.stop()
        except Exception: pass
        capture.stop()
        try: safety.stop()
        except Exception: pass


if __name__ == "__main__":
    raise SystemExit(main())
