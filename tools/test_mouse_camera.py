# tools/test_mouse_camera.py
"""
Live verification: does the WorldExplorer-style mouse pipeline
actually rotate the in-game camera?

What it does
------------
1. Captures a frame, reads F3 yaw + pitch (BEFORE).
2. Tells the mouse subsystem to apply a defined dx via
   ``mouse.track_target(dx, 0)``.
3. Waits a beat, captures another frame, reads F3 yaw + pitch (AFTER).
4. Reports the delta and whether the camera actually rotated.

This isolates the mouse-control path from the agent's decision logic
— if the camera doesn't move here, the agent can't possibly work.
"""

from __future__ import annotations

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load_settings():
    import yaml
    with open(os.path.join(ROOT, "config", "settings.yaml")) as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    settings = _load_settings()
    from utils.focus import _find_minecraft_hwnd, activate_minecraft
    from vision.capture import Capture, CaptureConfig
    from vision.ocr import build_f3_reader
    from control.mouse import Mouse, MouseConfig
    from control.input_gate import InputGate

    wins = _find_minecraft_hwnd()
    if not wins:
        print("[ERR] Minecraft not running")
        return 2
    hwnd, _rect = wins[0]
    print(f"[ok] hwnd={hwnd}")

    print("[..] focus Minecraft")
    activate_minecraft()
    time.sleep(0.4)

    cap = Capture(config=CaptureConfig(
        hwnd=hwnd, window_title_query="Minecraft",
        max_fps=30.0, track_window_each_frame=True,
        use_client_area=True, clamp_to_monitor=True,
        name="mouse_test_capture",
    ))
    cap.start()
    time.sleep(0.25)

    gate = InputGate()
    gate.set_allowed(True)  # always allow for this isolated test
    mouse = Mouse(config=MouseConfig(), gate=gate)
    mouse.start()

    f3 = build_f3_reader(settings)

    # Read pose BEFORE
    frame = cap.get_frame()
    info0 = f3.read(frame)
    y0 = info0.yaw if info0.yaw is not None else None
    p0 = info0.pitch if info0.pitch is not None else None
    print(f"[BEFORE] yaw={y0} pitch={p0}")

    if y0 is None or p0 is None:
        print("[WARN] F3 OCR didn't return pose. Make sure F3 always-on text is on.")

    # Apply some mouse motion
    print("[..] moving mouse: yaw +600 px (should turn right ~90°)")
    mouse.track_target(600, 0)
    time.sleep(0.5)

    frame = cap.get_frame()
    info1 = f3.read(frame)
    y1 = info1.yaw if info1.yaw is not None else None
    p1 = info1.pitch if info1.pitch is not None else None
    print(f"[AFTER 1] yaw={y1} pitch={p1}")
    if y0 is not None and y1 is not None:
        d = y1 - y0
        while d >  180.0: d -= 360.0
        while d < -180.0: d += 360.0
        print(f"   delta yaw = {d:+.1f} deg")

    print("[..] moving mouse: pitch +100 px (should look down ~15°)")
    mouse.track_target(0, 100)
    time.sleep(0.5)
    frame = cap.get_frame()
    info2 = f3.read(frame)
    y2 = info2.yaw if info2.yaw is not None else None
    p2 = info2.pitch if info2.pitch is not None else None
    print(f"[AFTER 2] yaw={y2} pitch={p2}")
    if p1 is not None and p2 is not None:
        print(f"   delta pitch = {p2 - p1:+.1f} deg")

    # Move back to roughly where we started
    print("[..] restoring camera")
    if y0 is not None and y2 is not None and p0 is not None and p2 is not None:
        # Reverse motion
        # mouse_per_degree ~6.5, so:
        dyaw = y0 - y2
        while dyaw > 180: dyaw -= 360
        while dyaw <= -180: dyaw += 360
        dpitch = p0 - p2
        mouse.track_target(int(dyaw * 6.5), int(dpitch * 6.5))
    else:
        mouse.track_target(-600, -100)

    time.sleep(0.5)
    mouse.stop()
    cap.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
