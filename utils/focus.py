# utils/focus.py
from __future__ import annotations
import time
import platform
from typing import List, Optional

try:
    import pygetwindow as gw
except Exception:
    gw = None

def _win32():
    import win32gui, win32process, win32con, win32api
    return win32gui, win32process, win32con, win32api
if platform.system().lower() == "windows":
    win32gui, win32process, win32con, win32api = _win32()
else:
    win32gui = win32process = win32con = win32api = None
import psutil
import ctypes

if platform.system().lower() == "windows":
    user32 = ctypes.windll.user32
else:
    user32 = None



def _find_minecraft_hwnd():
    result = []

    def cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return True

        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            pname = psutil.Process(pid).name().lower()
        except:
            return True

        if pname != "javaw.exe":
            return True

        rect = win32gui.GetWindowRect(hwnd)
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]

        if w < 200 or h < 200:
            return True

        result.append((hwnd, rect))
        return True

    win32gui.EnumWindows(cb, None)
    result.sort(key=lambda e: (e[1][2] - e[1][0]) * (e[1][3]-e[1][1]), reverse=True)
    return result


def activate_minecraft(maximize: bool = True) -> bool:
    """
    Bring the Minecraft window to the foreground.

    By default the window is also maximized — the project's calibration
    (vision/calibration.py) runs in maximized state, and the HUD region
    coordinates in settings.yaml are only valid at that exact client-area
    size. Pass ``maximize=False`` if you've calibrated at a custom window
    size and want the bot to operate at that size instead.
    """
    wins = _find_minecraft_hwnd()
    if not wins:
        return False

    hwnd, _ = wins[0]
    fg = None

    try:
        # Maximize (or merely restore if maximize=False) so the window
        # ends up at the same client-area size that calibration captured.
        # SW_SHOWMAXIMIZED also restores from minimized.
        win32gui.ShowWindow(
            hwnd,
            win32con.SW_SHOWMAXIMIZED if maximize else win32con.SW_RESTORE,
        )

        fg = win32gui.GetForegroundWindow()
        fg_tid, _ = win32process.GetWindowThreadProcessId(fg) if fg else (0, 0)
        target_tid, _ = win32process.GetWindowThreadProcessId(hwnd)
        cur_tid = win32api.GetCurrentThreadId()

        user32.AttachThreadInput(cur_tid, fg_tid, True)
        user32.AttachThreadInput(cur_tid, target_tid, True)

        win32gui.BringWindowToTop(hwnd)
        win32gui.SetForegroundWindow(hwnd)
        win32gui.SetActiveWindow(hwnd)
        win32gui.SetFocus(hwnd)

        time.sleep(0.05)
        return True

    except Exception as e:
        print("[activate_minecraft] error:", e)
        return False

    finally:
        try:
            if fg:
                user32.AttachThreadInput(win32api.GetCurrentThreadId(), fg_tid, False)
        except:
            pass
        try:
            user32.AttachThreadInput(win32api.GetCurrentThreadId(), target_tid, False)
        except:
            pass