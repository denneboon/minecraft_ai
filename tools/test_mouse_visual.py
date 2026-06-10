# tools/test_mouse_visual.py
"""
Visual mouse-camera test. Captures BEFORE + AFTER screenshots,
moves the camera in between, saves them so we can inspect that the
camera actually rotated.
"""

from __future__ import annotations

import os
import sys
import time

import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> int:
    from utils.focus import _find_minecraft_hwnd, activate_minecraft
    from vision.capture import Capture, CaptureConfig
    from control.mouse import Mouse, MouseConfig
    from control.input_gate import InputGate

    wins = _find_minecraft_hwnd()
    if not wins:
        print("[ERR] Minecraft not running")
        return 2
    hwnd, _ = wins[0]
    activate_minecraft()
    time.sleep(0.5)

    cap = Capture(config=CaptureConfig(
        hwnd=hwnd, max_fps=30.0, use_client_area=True,
        clamp_to_monitor=True, name="mouse_visual",
    ))
    cap.start(); time.sleep(0.4)

    gate = InputGate(); gate.set_allowed(True)
    mouse = Mouse(config=MouseConfig(), gate=gate); mouse.start()

    out_dir = os.path.join(ROOT, "data", "calibration")

    # try/finally — without it a crash mid-script leaks the velocity
    # worker thread, which can still emit motion until the daemon
    # thread is killed at process exit. That's the "mouse moves a
    # bit after the program says it's done" bug.
    try:
        print("[1] capture BEFORE")
        f0 = cap.get_frame()
        cv2.imwrite(os.path.join(out_dir, "mouse_before.png"),
                    cv2.cvtColor(f0, cv2.COLOR_RGB2BGR))

        print("[2] move mouse: yaw +400 px")
        mouse.track_target(400, 0)
        time.sleep(0.4)

        f1 = cap.get_frame()
        cv2.imwrite(os.path.join(out_dir, "mouse_after_yaw.png"),
                    cv2.cvtColor(f1, cv2.COLOR_RGB2BGR))

        print("[3] move mouse: pitch +120 px (look down)")
        mouse.track_target(0, 120)
        time.sleep(0.4)

        f2 = cap.get_frame()
        cv2.imwrite(os.path.join(out_dir, "mouse_after_pitch.png"),
                    cv2.cvtColor(f2, cv2.COLOR_RGB2BGR))

        # Reverse
        print("[4] restoring (yaw -400, pitch -120)")
        mouse.track_target(-400, -120)
        time.sleep(0.5)

        # Quantitative measure: pixel difference between frames.
        import numpy as np
        def diff_pct(a, b):
            return np.abs(a.astype(int) - b.astype(int)).mean()
        print(f"diff before vs after_yaw   = {diff_pct(f0, f1):.2f}")
        print(f"diff after_yaw vs after_p  = {diff_pct(f1, f2):.2f}")
        print(f"diff before vs after_p     = {diff_pct(f0, f2):.2f}")

        print(f"images saved in {out_dir}/")
    finally:
        try:
            mouse.stop()
        finally:
            cap.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
