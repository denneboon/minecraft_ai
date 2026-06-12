#!/usr/bin/env python3
"""
Diagnose the capture/focus foundation: which window the bot targets, whether
it's minimised / foreground, and whether the CAPTURE IS LIVE or returning a
stale (frozen) frame. Run while you're looking at Minecraft.

    python tools/diag_capture.py
"""
from __future__ import annotations

import os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

import numpy as np
import win32gui, win32process, psutil
from utils.focus import _find_minecraft_hwnd, _is_foreground, activate_minecraft
from vision.capture import Capture, CaptureConfig


def _proc(hwnd):
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        return psutil.Process(pid).name()
    except Exception:
        return "?"


def main() -> int:
    fg = win32gui.GetForegroundWindow()
    print(f"[diag] foreground window: hwnd={fg} '{win32gui.GetWindowText(fg)}' "
          f"proc={_proc(fg)}")

    wins = _find_minecraft_hwnd()
    print(f"[diag] _find_minecraft_hwnd found {len(wins)} javaw window(s) "
          f"(largest-area first):")
    for i, (hwnd, rect) in enumerate(wins):
        w, h = rect[2]-rect[0], rect[3]-rect[1]
        print(f"    [{i}] hwnd={hwnd} '{win32gui.GetWindowText(hwnd)}' "
              f"rect={rect} {w}x{h} "
              f"iconic(min)={win32gui.IsIconic(hwnd)} "
              f"foreground={_is_foreground(hwnd)}")
    if not wins:
        print("[diag] NO javaw window >=200x200 found — is the game (not just "
              "the launcher) actually open?")
        return 2
    hwnd = wins[0][0]
    print(f"[diag] bot would target wins[0] = hwnd={hwnd}")

    print("[diag] calling activate_minecraft()...")
    ok = activate_minecraft()
    time.sleep(0.4)
    print(f"[diag] activate_minecraft returned {ok}; "
          f"now foreground={_is_foreground(hwnd)} "
          f"(fg hwnd={win32gui.GetForegroundWindow()})")

    # Is the capture LIVE? Grab frames over ~2s and check they change.
    cap = Capture(CaptureConfig(hwnd=hwnd, use_client_area=True, threaded=True))
    cap.start(); time.sleep(0.4)
    frames = []
    for k in range(5):
        f = cap.get_frame()
        frames.append(f)
        m = float(np.mean(f)) if f is not None else -1
        print(f"[diag] frame {k}: shape={None if f is None else f.shape} "
              f"mean={m:.1f}")
        time.sleep(0.5)
    # compare consecutive frames
    diffs = []
    for a, b in zip(frames, frames[1:]):
        if a is None or b is None:
            diffs.append(-1); continue
        diffs.append(float(np.mean(np.abs(a.astype(int) - b.astype(int)))))
    print(f"[diag] consecutive-frame mean-abs-diffs: {[round(d,2) for d in diffs]}")
    if all(0 <= d < 0.5 for d in diffs):
        print("[diag] >>> FROZEN: the capture is returning a STALE frame "
              "(window minimised/occluded/off-screen, or grab failing). The bot "
              "would be acting on a dead screenshot.")
    else:
        print("[diag] >>> LIVE: the capture is updating.")
    try:
        import cv2
        if frames[-1] is not None:
            cv2.imwrite(os.path.join(ROOT, "data", "_capture_now.png"),
                        frames[-1][:, :, ::-1])
            print("[diag] saved data/_capture_now.png (what the bot sees)")
    except Exception as e:
        print(f"[diag] save failed: {e}")
    cap.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
