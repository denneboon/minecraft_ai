# tools/focus_probe.py
import time, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import platform
import psutil

if platform.system().lower() == "windows":
    def _win32():
        import win32gui, win32process, win32con, win32api
        return win32gui, win32process, win32con, win32api
if platform.system().lower() == "windows":
    win32gui, win32process, win32con, win32api = _win32()
else:
    win32gui = win32process = win32con = win32api = None

def active_exe():
    try:
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return "(none)"
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        pname = psutil.Process(pid).name().lower()
        return f"{pname} (pid={pid}, hwnd={hwnd})"
    except Exception as e:
        return f"(error: {e})"

def main():
    print("Printing foreground process every 100ms for 6 seconds...")
    t0 = time.time()
    while time.time() - t0 < 6:
        print(active_exe())
        time.sleep(0.1)

if __name__ == "__main__":
    main()