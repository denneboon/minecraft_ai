def _win32():
    import win32gui, win32process, win32con, win32api
    return win32gui, win32process, win32con, win32api

win32gui, win32process, win32con, win32api = _win32()
import psutil

MINECRAFT_CLASSES = (
    "LWJGL", "Lwjgl", "LWJGL2", "LWJGL3",
    "GLFW", "GLFW3", "GLFW30", "GLFW32", "GLFW40",
)

def enum_handler(hwnd, data):
    if not win32gui.IsWindowVisible(hwnd):
        return

    cls = win32gui.GetClassName(hwnd)
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        pname = psutil.Process(pid).name().lower()
    except:
        pname = "?"

    rect = win32gui.GetWindowRect(hwnd)

    # ALWAYS print window:
    print(f"HWND={hwnd}  CLS={cls}  EXE={pname}  RECT={rect}")

    # NOW check match logic:
    if any(cls.startswith(pref) for pref in MINECRAFT_CLASSES) and pname == "javaw.exe":
        print(">>> MATCHED AS MINECRAFT <<<")

win32gui.EnumWindows(enum_handler, None)