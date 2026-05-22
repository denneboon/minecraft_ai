import psutil
def _win32():
    import win32gui, win32process, win32con, win32api
    return win32gui, win32process, win32con, win32api
win32gui, win32process, win32con, win32api = _win32()

def enum_handler(hwnd, data):
    if not win32gui.IsWindowVisible(hwnd):
        return

    cls = win32gui.GetClassName(hwnd)
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        pname = psutil.Process(pid).name()
    except:
        pname = "?"

    rect = win32gui.GetWindowRect(hwnd)

    print(f"HWND={hwnd}  CLS={cls}  EXE={pname}  RECT={rect}")

win32gui.EnumWindows(enum_handler, None)