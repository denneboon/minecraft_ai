# utils/focus.py
from __future__ import annotations
import time
import platform

try:
    import pygetwindow as gw
except ImportError:
    # pygetwindow not installed (Linux/macOS without it). The Win32
    # helpers below are the primary path on Windows; pygetwindow is
    # only consulted as a cross-platform fallback elsewhere.
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
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            # The process died between window enumeration and the name
            # lookup, or it belongs to a user we can't query (rare).
            # Either way, treat the window as "not Minecraft".
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


def _disable_foreground_lock() -> None:
    """Zero the system foreground-lock timeout so ``SetForegroundWindow``
    is permitted. Windows otherwise blocks a process that hasn't received
    user input recently from stealing foreground (focus-stealing
    prevention) — which is exactly why ``SetForegroundWindow`` returns
    "Access is denied" when the bot is launched from a terminal. Best-
    effort; harmless if it fails."""
    if user32 is None:
        return
    try:
        SPI_SETFOREGROUNDLOCKTIMEOUT = 0x2001
        SPIF_SENDCHANGE = 0x0002
        # pvParam holds the new timeout (0) passed by value as the void*.
        user32.SystemParametersInfoW(SPI_SETFOREGROUNDLOCKTIMEOUT, 0, 0,
                                     SPIF_SENDCHANGE)
    except Exception:
        pass


def _alt_tap() -> None:
    """Synthesise a harmless ALT press+release. This registers input for
    our process, which Windows uses to grant the right to call
    ``SetForegroundWindow`` — a well-known unlock for the focus-stealing
    block."""
    if user32 is None:
        return
    try:
        VK_MENU = 0x12
        KEYEVENTF_KEYUP = 0x0002
        user32.keybd_event(VK_MENU, 0, 0, 0)
        user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)
    except Exception:
        pass


def _is_foreground(hwnd) -> bool:
    try:
        return win32gui.GetForegroundWindow() == hwnd
    except Exception:
        return False


def activate_minecraft(maximize: bool = True) -> bool:
    """
    Bring the Minecraft window to the foreground.

    Robust against Windows' focus-stealing block (the cause of
    "SetForegroundWindow: Access is denied"): we zero the foreground-lock
    timeout, attach thread input, then try each focus call independently
    so one denial can't skip the rest, verify the result, and fall back
    to an ALT-tap and finally a minimize/restore cycle if needed.

    By default the window is also maximized — calibration runs maximized
    and the HUD region coords in settings.yaml are only valid at that
    client-area size. Pass ``maximize=False`` to operate at the current
    window size. Returns True only if MC actually ended up foreground.
    """
    wins = _find_minecraft_hwnd()
    if not wins or win32gui is None or user32 is None:
        return False

    hwnd, _ = wins[0]
    show = win32con.SW_SHOWMAXIMIZED if maximize else win32con.SW_RESTORE
    _disable_foreground_lock()

    fg_tid = target_tid = cur_tid = 0
    attached_fg = attached_target = False

    def _try_focus_calls() -> None:
        for fn in (lambda: win32gui.BringWindowToTop(hwnd),
                   lambda: win32gui.SetForegroundWindow(hwnd),
                   lambda: win32gui.SetActiveWindow(hwnd),
                   lambda: win32gui.SetFocus(hwnd)):
            try:
                fn()
            except Exception:
                pass   # a single denied call must not abort the others

    try:
        try:
            win32gui.ShowWindow(hwnd, show)
        except Exception:
            pass

        # Attach our input queue to the current-foreground + target so
        # SetForegroundWindow is honoured.
        try:
            fg = win32gui.GetForegroundWindow()
            fg_tid = win32process.GetWindowThreadProcessId(fg)[0] if fg else 0
            target_tid = win32process.GetWindowThreadProcessId(hwnd)[0]
            cur_tid = win32api.GetCurrentThreadId()
            if fg_tid and fg_tid != cur_tid:
                user32.AttachThreadInput(cur_tid, fg_tid, True)
                attached_fg = True
            if target_tid and target_tid != cur_tid:
                user32.AttachThreadInput(cur_tid, target_tid, True)
                attached_target = True
        except Exception:
            pass

        _try_focus_calls()
        time.sleep(0.05)
        if _is_foreground(hwnd):
            return True

        # Fallback 1: register input for our process, then retry.
        _alt_tap()
        _try_focus_calls()
        time.sleep(0.05)
        if _is_foreground(hwnd):
            return True

        # Fallback 2 (last resort): a minimize→restore cycle, which
        # reliably grants the foreground right at the cost of a flicker.
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
            win32gui.ShowWindow(hwnd, show)
        except Exception:
            pass
        _try_focus_calls()
        time.sleep(0.05)
        if not _is_foreground(hwnd):
            print("[activate_minecraft][WARN] could not bring Minecraft to "
                  "the foreground (Windows denied focus). Click the MC "
                  "window once; the bot will start when it has focus.")
        return _is_foreground(hwnd)
    finally:
        # Detach input queues (best-effort).
        try:
            if attached_fg:
                user32.AttachThreadInput(cur_tid, fg_tid, False)
        except Exception:
            pass
        try:
            if attached_target:
                user32.AttachThreadInput(cur_tid, target_tid, False)
        except Exception:
            pass
