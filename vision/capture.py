# vision/capture.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import mss
import platform
import ctypes
from ctypes import wintypes
import threading

try:
    import pygetwindow as gw
except Exception:
    gw = None

from vision.capture_backends import ICaptureBackend

# Windows-only (guarded use)
if platform.system().lower() == "windows":
    def _win32():
        import win32gui, win32process, win32con, win32api
        return win32gui, win32process, win32con, win32api

    win32gui, win32process, win32con, win32api = _win32()
    user32 = ctypes.windll.user32
else:
    win32gui = win32process = win32con = win32api = None


def _windows_get_client_rect(hwnd) -> Optional[Tuple[int, int, int, int]]:
    if hwnd is None:
        return None
    try:
        rect = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return None
        pt = wintypes.POINT(0, 0)
        if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
            return None
        left, top = pt.x, pt.y
        right, bottom = left + (rect.right - rect.left), top + (rect.bottom - rect.top)
        return (left, top, right, bottom)
    except Exception:
        return None


def _find_window_rect_client_or_window(query: str, prefer_client: bool = True) -> Optional[Tuple[int,int,int,int]]:
    if gw is None:
        return None
    q = query.lower()
    titles = [t for t in gw.getAllTitles() if t and q in t.lower()]
    if not titles:
        return None
    try:
        wins = gw.getWindowsWithTitle(titles[0])
        win = wins[0]
        if win.isMinimized or win.width <= 0 or win.height <= 0:
            return None

        if prefer_client and platform.system().lower() == "windows" and hasattr(win, "_hWnd"):
            client = _windows_get_client_rect(win._hWnd)
            if client:
                return client

        left, top = win.left, win.top
        return (left, top, left + win.width, top + win.height)
    except Exception:
        return None


@dataclass
class CaptureConfig:
    hwnd: Optional[int] = None
    window_title_query: str = "Minecraft"
    max_fps: Optional[float] = 60.0
    downscale: float = 1.0
    track_window_each_frame: bool = True
    strict_window_find: bool = False
    clamp_to_monitor: bool = True
    use_client_area: bool = True
    name: str = "minecraft_capture"


def _rect_to_region(rect: Tuple[int, int, int, int]) -> Dict[str, int]:
    l, t, r, b = rect
    return {
        "left": int(l),
        "top": int(t),
        "width": int(max(1, r - l)),
        "height": int(max(1, b - t)),
    }


def _bgra_to_rgb(arr_bgra: np.ndarray) -> np.ndarray:
    return arr_bgra[:, :, :3][:, :, ::-1].copy()


def _downscale(img: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    nh, nw = int(h * scale), int(w * scale)
    if nh < 1 or nw < 1:
        return img
    yy = (np.linspace(0, h - 1, nh)).astype(np.int32)
    xx = (np.linspace(0, w - 1, nw)).astype(np.int32)
    return img[np.ix_(yy, xx)].copy()


class _MSSBackend(ICaptureBackend):
    def __init__(self, region: Dict[str, int], max_fps: Optional[float]):
        self.region = region
        self.max_fps = max_fps
        self.sct = mss.mss()
        self._last_t = 0.0

    def _limit_fps(self):
        if self.max_fps and self.max_fps > 0:
            now = time.perf_counter()
            min_dt = 1.0 / self.max_fps
            dt = now - self._last_t
            if dt < min_dt:
                time.sleep(min_dt - dt)
            self._last_t = time.perf_counter()

    def grab(self, region: Dict[str, int]) -> np.ndarray:
        self.region = region
        self._limit_fps()
        shot = self.sct.grab(self.region)
        return np.frombuffer(shot.raw, dtype=np.uint8).reshape(shot.height, shot.width, 4)

    def stop(self) -> None:
        try:
            self.sct.close()
        except Exception:
            pass

class Capture:
    def __init__(self, config: CaptureConfig, backend=None):
        self.cfg = config
        self._backend = None
        self._window_rect = None
        self._lock = threading.RLock()
        self._owner_thread = None

    def start(self):
        with self._lock:
            if self._backend is not None:
                return
            rect = self._ensure_rect()
            self._backend = _MSSBackend(_rect_to_region(rect), self.cfg.max_fps)

    def stop(self):
        with self._lock:
            if self._backend:
                try:
                    self._backend.stop()
                finally:
                    self._backend = None

    def get_frame(self) -> np.ndarray:
        # Lazy start + owner-thread registration, both under the lock
        with self._lock:
            tid = threading.get_ident()
            if self._owner_thread is None:
                self._owner_thread = tid
            elif self._owner_thread != tid:
                raise RuntimeError("Capture.get_frame called from multiple threads")
            if self._backend is None:
                self.start()
            rect = self._ensure_rect()
            raw = self._backend.grab(_rect_to_region(rect))
        rgb = _bgra_to_rgb(raw)
        if self.cfg.downscale != 1.0:
            rgb = _downscale(rgb, self.cfg.downscale)
        return rgb
    
    def window_origin(self) -> Tuple[int, int]:
        """
        Return the (x, y) desktop coordinates of the top-left corner of
        the captured area. Useful for converting frame-pixel coordinates
        (used by HUD readers, slot rects, etc.) into desktop-pixel
        coordinates (used by absolute cursor moves).

        Triggers a fresh rect lookup if the cache is empty.
        """
        with self._lock:
            rect = self._ensure_rect()
        return int(rect[0]), int(rect[1])

    def _ensure_rect(self):
        if self.cfg.hwnd:
            rect = _windows_get_client_rect(self.cfg.hwnd)
        else:
            rect = _find_window_rect_client_or_window(
                self.cfg.window_title_query,
                prefer_client=self.cfg.use_client_area
            )

        if rect is not None:
            self._window_rect = rect
            return rect

        if self._window_rect is not None:
            return self._window_rect

        raise RuntimeError("Could not determine capture window")