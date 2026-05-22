# tools/capture_smoketest.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import cv2

from utils.focus import _find_minecraft_hwnd, activate_minecraft
from vision.capture import Capture, CaptureConfig

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    wins = _find_minecraft_hwnd()
    assert wins, "No Minecraft HWND found."
    hwnd, rect = wins[0]
    print("Using HWND:", hwnd, "rect:", rect)

    activate_minecraft()
    time.sleep(0.25)

    cap = Capture(CaptureConfig(
        hwnd=hwnd,
        use_client_area=True,
        max_fps=30,
        track_window_each_frame=True,
        strict_window_find=False,
    ))
    cap.start()

    frame = cap.get_frame()
    cap.stop()

    print("Frame shape:", frame.shape, "(H, W, C)")

    out_path = os.path.join(_SCRIPT_DIR, "capture_debug.png")
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
