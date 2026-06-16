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
except ImportError:
    # pygetwindow isn't installed (Linux/macOS users without it). All
    # window-finding helpers below already gracefully no-op when gw is
    # None, but we narrow the except so genuine import-time failures
    # (e.g. a corrupted install raising RuntimeError) still surface.
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
    except (OSError, AttributeError, ValueError):
        # OSError covers Windows API failures (invalid HWND, wrong proc);
        # AttributeError covers cases where user32 isn't loaded (non-Win
        # platform); ValueError covers ctypes type-coercion failures.
        # Anything else (KeyboardInterrupt, MemoryError) should propagate.
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
    except (IndexError, AttributeError, OSError):
        # IndexError: no windows matched the title after all.
        # AttributeError: pygetwindow Win object missing expected fields
        # (varies across the lib's minor versions).
        # OSError: the underlying GetWindowRect call failed.
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
    # When True, a background thread grabs frames continuously and
    # ``get_frame()`` returns the most-recent one instantly instead of
    # blocking on the ~16-30 ms screen grab. This takes capture latency
    # off the agent loop's critical path so the control rate is bounded
    # by decision/perception cost, not by the grab. Off by default so
    # act-then-capture tools (inventory hover→read, calibration) keep
    # their synchronous semantics; main.py enables it for the agent loop
    # via ``capture.threaded`` in settings.yaml.
    threaded: bool = False


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
        except (OSError, AttributeError):
            # Best-effort close — mss's MSS.close() can raise OSError
            # when the display handle has already been released or
            # AttributeError after a partially-failed __init__.
            pass

class Capture:
    def __init__(self, config: CaptureConfig, backend=None):
        self.cfg = config
        # An injected backend (used by tests / alternative grabbers) is
        # honoured. ``None`` means we lazily construct the default
        # ``_MSSBackend`` — in threaded mode that construction happens
        # ON the grab thread, because mss binds GDI handles to the
        # creating thread and is not safe to share across threads.
        self._backend = backend
        self._external_backend = backend is not None
        self._window_rect = None
        self._lock = threading.RLock()
        self._owner_thread = None

        # ── Threaded-grabber state ───────────────────────────────────
        self._threaded = bool(getattr(config, "threaded", False))
        self._grab_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # Permanent shutdown latch (distinct from the restartable _stop).
        # Once ``stop()`` runs, a late ``get_frame()`` — e.g. from the OCR
        # worker still mid-read while teardown proceeds — must NOT
        # re-spawn the grab thread (which would leak a thread + a fresh
        # mss/GDI backend). ``start()`` clears it for a deliberate restart.
        self._shutdown = False
        self._frame_lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._first_frame = threading.Event()
        self._grab_err: Optional[BaseException] = None

    # ── Synchronous helpers (also used by the grab thread) ───────────

    def _grab_once(self) -> np.ndarray:
        """Grab one RGB frame using the current backend. Assumes the
        backend exists and is being used from its owning thread."""
        rect = self._ensure_rect()
        raw = self._backend.grab(_rect_to_region(rect))
        rgb = _bgra_to_rgb(raw)
        if self.cfg.downscale != 1.0:
            rgb = _downscale(rgb, self.cfg.downscale)
        return rgb

    def _grab_loop(self) -> None:
        """Background grabber: builds the backend on THIS thread (mss
        affinity), then publishes the latest RGB frame continuously.
        The backend's own FPS limiter paces the loop, so this costs no
        more CPU than the configured ``max_fps``.

        Resilient by design: a transient failure to resolve the window
        rect or build the backend (common during the startup
        maximize/focus race) is RETRIED inside the loop rather than
        killing the thread. The previous version exited on the first
        such failure, which — once ``get_frame`` restarted it and it
        failed again — left the agent capturing nothing for a whole run
        (observed as a 0-tick session, ~3 s wasted per ``get_frame``
        first-frame wait)."""
        try:
            while not self._stop.is_set():
                # (Re)acquire the backend if we don't have one yet.
                if self._backend is None:
                    try:
                        rect = self._ensure_rect()
                        self._backend = _MSSBackend(
                            _rect_to_region(rect), self.cfg.max_fps)
                    except Exception as e:
                        self._grab_err = e
                        self._stop.wait(0.1)   # window not ready — retry soon
                        continue
                try:
                    rgb = self._grab_once()
                except Exception as e:
                    # Transient window-move / focus-change / mss hiccup.
                    # Drop the backend so we rebuild it (handles a window
                    # that closed + reopened with a new handle), back off,
                    # retry — never kill the grabber permanently.
                    self._grab_err = e
                    if not self._external_backend:
                        try:
                            self._backend.stop()
                        except Exception:
                            pass
                        self._backend = None
                    self._stop.wait(0.1)
                    continue
                with self._frame_lock:
                    self._latest = rgb
                    self._latest_ts = time.perf_counter()
                self._first_frame.set()
        finally:
            # Close the backend on the SAME thread that created it.
            if self._backend is not None and not self._external_backend:
                try:
                    self._backend.stop()
                except Exception:
                    pass

    def start(self):
        with self._lock:
            if self._threaded:
                self._shutdown = False   # a deliberate (re)start
                if self._grab_thread is not None and self._grab_thread.is_alive():
                    return
                self._stop.clear()
                self._first_frame.clear()
                self._grab_err = None
                self._grab_thread = threading.Thread(
                    target=self._grab_loop,
                    name=f"{self.cfg.name}-grab",
                    daemon=True,
                )
                self._grab_thread.start()
                return
            if self._backend is not None:
                return
            rect = self._ensure_rect()
            self._backend = _MSSBackend(_rect_to_region(rect), self.cfg.max_fps)

    def stop(self):
        if self._threaded:
            self._shutdown = True       # latch: no auto-restart after this
            self._stop.set()
            th = self._grab_thread
            if th is not None:
                th.join(timeout=1.0)
            self._grab_thread = None
            # The grab loop closes the backend on exit. We deliberately do
            # NOT null ``_latest`` here: a concurrent get_frame() could
            # then observe None and raise spuriously during shutdown.
            # Keeping the last frame ref is harmless (callers are tearing
            # down too); ``_shutdown`` is what prevents re-spawning.
            return
        with self._lock:
            if self._backend and not self._external_backend:
                try:
                    self._backend.stop()
                finally:
                    self._backend = None

    def get_frame(self) -> np.ndarray:
        if self._threaded:
            if self._shutdown:
                # Teardown in progress — don't resurrect the grab thread.
                with self._frame_lock:
                    frame = self._latest
                if frame is not None:
                    return frame
                raise RuntimeError("Capture: stopped")
            if self._grab_thread is None or not self._grab_thread.is_alive():
                self.start()
            # Read _latest ONLY under the lock (the grab thread writes it under
            # the same lock). Reading it unsynchronised could observe a stale
            # None right after the writer published a frame, spuriously raising
            # "grab thread produced no frame" and aborting a live run.
            with self._frame_lock:
                frame = self._latest
            if frame is None:
                # Block only until the very first frame is published.
                if not self._first_frame.wait(timeout=3.0):
                    raise RuntimeError("Capture: no frame within 3 s")
                with self._frame_lock:
                    frame = self._latest
                if frame is None:
                    raise RuntimeError(
                        f"Capture: grab thread produced no frame "
                        f"({self._grab_err!r})")
            return frame

        # Synchronous path — lazy start + single-owner-thread guard so a
        # stray cross-thread call can't corrupt mss's GDI state.
        with self._lock:
            tid = threading.get_ident()
            if self._owner_thread is None:
                self._owner_thread = tid
            elif self._owner_thread != tid:
                raise RuntimeError("Capture.get_frame called from multiple threads")
            if self._backend is None:
                self.start()
            return self._grab_once()

    def latest_frame_age(self) -> Optional[float]:
        """Seconds since the most recent grabbed frame (threaded mode),
        or ``None`` if no frame yet / not threaded. Lets latency-
        sensitive callers detect a stalled grabber."""
        if not self._threaded or self._latest_ts == 0.0:
            return None
        return time.perf_counter() - self._latest_ts

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

    def window_bounds(self) -> Optional[Tuple[int, int, int, int]]:
        """The captured window's (left, top, right, bottom) in DESKTOP pixels,
        or ``None`` if it can't be determined. Unlike ``window_origin`` +
        frame-shape, this is the TRUE window rect straight from the OS, so it
        stays correct even if the capture ever falls back to a different-sized
        grab. Used to fence absolute cursor moves inside the MC window so a
        misread slot can never click the desktop/taskbar."""
        try:
            with self._lock:
                rect = self._ensure_rect()
            return (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
        except Exception:
            return None

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
