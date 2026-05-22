import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.focus import activate_minecraft, _find_minecraft_hwnd
import time
def _win32():
    import win32gui, win32process, win32con, win32api
    return win32gui, win32process, win32con, win32api

wins = _find_minecraft_hwnd()
assert wins, "No Minecraft HWND found."
hwnd, rect = wins[0]
print("Found Minecraft HWND:", hwnd, "Rect:", rect)

print("Trying to activate...")
ok = activate_minecraft()
print("activate_minecraft() ->", ok)

time.sleep(0.2)
win32gui, win32process, win32con, win32api = _win32()
fg = win32gui.GetForegroundWindow()
print("Foreground HWND:", fg, "(expected:", hwnd, ")")
print("FOCUSED:", fg == hwnd)