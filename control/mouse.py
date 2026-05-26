"""
High-performance mouse controller for Minecraft with human-like camera movement.

Features:
- Smooth, eased rotation instead of snapping to target.
- Precise delta-based mouse movement (game-friendly).
- Left click, right click, simultaneous click, hold & drag.
- Scroll wheel support (for menus or optional in-game behavior).
- Thread-safe, stateful (tracks pressed mouse buttons).
- Motion curve system that never confuses the AI about where it is looking.
- Backend abstraction (currently pynput, easily swappable for Win32 SendInput).

Designed for Minecraft:
- Uses relative mouse movement (dx, dy) rather than setting absolute positions.
- Movement smoothing never hides "true" deltas from your AI: the bot requests
  a movement, the easing executes it over exactly-known increments.
- Perfect for PvP, building, parkour, smooth rotations, flicks, etc.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable, Dict

try:
    from pynput.mouse import Button, Controller
except Exception:
    raise RuntimeError("pynput is required for mouse control. Install via pip install pynput")


# -------------------------------
# Config
# -------------------------------

@dataclass
class MouseConfig:
    # ================= MOVEMENT PARAMETERS =================
    # Movement smoothing: how many ms a typical movement should be eased over.
    # Larger = smoother but slower; smaller = sharper.
    move_duration_ms: int = 35

    # If your AI wants a "flick" (fast look), set this multiplier lower.
    flick_multiplier: float = 0.35

    # Number of subdivision steps for easing. Higher = smoother path.
    curve_steps: int = 10

    # Easing function: lambda t in [0,1] -> eased progress.
    # Default = ease-out cubic ``1 - (1-t)^3``: fast at the start,
    # gently decelerating to zero at the end. This produces a more
    # humanlike "smooth glance" than the symmetric smoothstep, which
    # has a noticeable acceleration in the middle that reads as a
    # snappy hand twitch when the per-tick delta is small.
    easing_fn: Callable[[float], float] = field(default_factory=lambda: (
        lambda t: 1.0 - (1.0 - t) ** 3
    ))

    # ================= CLICK PARAMETERS =================
    default_click_duration: float = 0.05  # seconds held for a tap
    allow_simultaneous: bool = True

    # ================= THREAD / SAFETY =================
    max_events_per_sec: int = 240  # Rate limiter

    # ================= SCROLL =================
    enable_scroll: bool = True


# -------------------------------
# Backend Interface
# -------------------------------

class _IMouseBackend:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def move(self, dx: int, dy: int) -> None: ...
    def press_left(self) -> None: ...
    def release_left(self) -> None: ...
    def press_right(self) -> None: ...
    def release_right(self) -> None: ...
    def scroll(self, dx: int, dy: int) -> None: ...


# -------------------------------
# Pynput Backend
# -------------------------------

class _PynputBackend(_IMouseBackend):
    def __init__(self):
        self._controller = Controller()

    def start(self): pass
    def stop(self): pass

    def move(self, dx: int, dy: int) -> None:
        # Move relative to current position
        self._controller.move(dx, dy)

    def press_left(self): self._controller.press(Button.left)
    def release_left(self): self._controller.release(Button.left)
    def press_right(self): self._controller.press(Button.right)
    def release_right(self): self._controller.release(Button.right)

    def scroll(self, dx: int, dy: int) -> None:
        self._controller.scroll(dx, dy)


# -------------------------------
# Win32 SendInput Backend (Minecraft-compatible)
# -------------------------------
# Minecraft locks the cursor during gameplay and reads RELATIVE mouse
# motion via the Raw Input API (WM_INPUT). pynput's controller uses
# SetCursorPos, which moves the OS cursor but is INVISIBLE to a Raw
# Input listener — so MC sees no camera input. The Win32 SendInput
# API with MOUSEEVENTF_MOVE generates a real WM_MOUSEMOVE event that
# DOES feed the Raw Input stream, which is what MC's mouse-look reads.
#
# This backend is the right choice whenever the bot is driving an
# active 3-D game window on Windows. The pynput backend remains for
# Linux / macOS / non-game GUIs.

try:
    import ctypes
    from ctypes import wintypes
    _WIN32_OK = True
except Exception:
    _WIN32_OK = False


if _WIN32_OK:
    # SendInput input struct layout (see MSDN INPUT / MOUSEINPUT).
    _INPUT_MOUSE = 0
    _MOUSEEVENTF_MOVE       = 0x0001
    _MOUSEEVENTF_LEFTDOWN   = 0x0002
    _MOUSEEVENTF_LEFTUP     = 0x0004
    _MOUSEEVENTF_RIGHTDOWN  = 0x0008
    _MOUSEEVENTF_RIGHTUP    = 0x0010
    _MOUSEEVENTF_WHEEL      = 0x0800
    _WHEEL_DELTA            = 120

    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx",          wintypes.LONG),
            ("dy",          wintypes.LONG),
            ("mouseData",   wintypes.DWORD),
            ("dwFlags",     wintypes.DWORD),
            ("time",        wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
        ]

    class _INPUT_UNION(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT)]

    class _INPUT(ctypes.Structure):
        _fields_ = [
            ("type",  wintypes.DWORD),
            ("union", _INPUT_UNION),
        ]


class _Win32SendInputBackend(_IMouseBackend):
    """Mouse backend using Win32 SendInput. Required for Minecraft."""

    def __init__(self):
        if not _WIN32_OK:
            raise RuntimeError("Win32 SendInput backend is Windows-only.")
        # Use a FRESH WinDLL handle, not the cached ``ctypes.windll.user32``
        # singleton — otherwise setting ``argtypes`` on its SendInput
        # pollutes every other library in the process that also calls
        # SendInput (e.g. pynput's keyboard, which uses a DIFFERENT INPUT
        # struct layout and then crashes with "expected LP__INPUT").
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._send_input = self._user32.SendInput
        self._send_input.argtypes = (
            wintypes.UINT,
            ctypes.POINTER(_INPUT),
            ctypes.c_int,
        )
        self._send_input.restype = wintypes.UINT

    def start(self): pass
    def stop(self): pass

    def _send(self, dx: int, dy: int, flags: int, data: int = 0) -> None:
        inp = _INPUT(
            type=_INPUT_MOUSE,
            union=_INPUT_UNION(mi=_MOUSEINPUT(
                dx=int(dx), dy=int(dy),
                mouseData=int(data),
                dwFlags=int(flags),
                time=0,
                dwExtraInfo=None,
            )),
        )
        self._send_input(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))

    def move(self, dx: int, dy: int) -> None:
        # MOUSEEVENTF_MOVE with the ABSOLUTE flag UNSET = relative motion,
        # which generates Raw Input events that Minecraft listens for.
        self._send(dx, dy, _MOUSEEVENTF_MOVE)

    def press_left(self):    self._send(0, 0, _MOUSEEVENTF_LEFTDOWN)
    def release_left(self):  self._send(0, 0, _MOUSEEVENTF_LEFTUP)
    def press_right(self):   self._send(0, 0, _MOUSEEVENTF_RIGHTDOWN)
    def release_right(self): self._send(0, 0, _MOUSEEVENTF_RIGHTUP)

    def scroll(self, dx: int, dy: int) -> None:
        if dy:
            self._send(0, 0, _MOUSEEVENTF_WHEEL, data=int(dy) * _WHEEL_DELTA)


# -------------------------------
# Main Mouse Controller
# -------------------------------

class Mouse:
    """
    High-level, thread-safe mouse controller with easing and smoothing.

    Public API:

        mouse.move(dx, dy)
        mouse.move_smooth(dx, dy, duration_ms=None)

        mouse.left_click()
        mouse.right_click()
        mouse.left_press()
        mouse.left_release()
        mouse.right_press()
        mouse.right_release()

        mouse.scroll(up/down)
        mouse.move_to_target(dx, dy)

        mouse.flick(dx, dy)              # fast PvP flick
        mouse.track_target(dx, dy)       # intelligent smooth turn
    """

    def __init__(self, config: Optional[MouseConfig] = None, backend: Optional[_IMouseBackend] = None, gate=None):
        self.cfg = config or MouseConfig()
        # Default to the Win32 SendInput backend on Windows — that's
        # the only one that reaches Minecraft's Raw Input listener.
        # Fall back to pynput everywhere else (and if SendInput somehow
        # fails to construct).
        if backend is None:
            try:
                if _WIN32_OK:
                    backend = _Win32SendInputBackend()
                else:
                    backend = _PynputBackend()
            except Exception:
                backend = _PynputBackend()
        self._backend: _IMouseBackend = backend

        self._lock = threading.RLock()
        self._running = False

        self._pressed = {
            "left": False,
            "right": False,
        }

        self._last_event_ts = 0.0
        self._gate = gate

        # ── Velocity-mode worker state ───────────────────────────
        # When the agent calls ``set_velocity(vx, vy)`` we run a
        # background thread that emits tiny relative-motion events
        # at a steady cadence. This produces TRULY continuous
        # camera motion — no per-tick gaps — which is what makes a
        # 360° pan feel like a smooth turn instead of 20 chunked
        # nudges. Velocity is in pixels per second.
        self._velocity_vx: float = 0.0
        self._velocity_vy: float = 0.0
        self._velocity_tick_hz: int = 240   # cadence of micro-motions
        self._velocity_thread: Optional[threading.Thread] = None
        self._velocity_stop = threading.Event()
        self._velocity_residual_x: float = 0.0
        self._velocity_residual_y: float = 0.0


    def start(self):
        with self._lock:
            if self._running:
                return
            self._backend.start()
            self._running = True
            # Start the velocity worker. It's idle (no-op) until the
            # agent calls set_velocity with a non-zero value.
            self._velocity_stop.clear()
            self._velocity_thread = threading.Thread(
                target=self._velocity_loop,
                name="mouse-velocity",
                daemon=True,
            )
            self._velocity_thread.start()

    def stop(self):
        with self._lock:
            if not self._running:
                return
            if self._pressed["left"]:
                self._backend.release_left()
                self._pressed["left"] = False
            if self._pressed["right"]:
                self._backend.release_right()
                self._pressed["right"] = False
            self._running = False
            self._velocity_stop.set()
        # Wait outside the lock so the worker can exit cleanly.
        try:
            if self._velocity_thread is not None:
                self._velocity_thread.join(timeout=0.25)
        except Exception:
            pass
        self._velocity_thread = None
        self._backend.stop()

    # ── Continuous-velocity motion API ─────────────────────────────

    def set_velocity(self, vx_per_sec: float, vy_per_sec: float) -> None:
        """
        Drive the mouse at a continuous angular velocity (pixels per
        second). Replaces any previous velocity command immediately.
        ``set_velocity(0, 0)`` halts motion. The mouse keeps moving
        until you change the velocity — there are no per-tick gaps.

        This is the canonical humanlike camera-motion API: a real
        person turns at a roughly-constant rate during a glance,
        not in discrete jumps. The agent's tick rate (20 Hz) is too
        coarse to express smooth motion on its own, but here the
        velocity changes every tick while the *motion itself* is
        emitted at 240 Hz inside the worker.
        """
        if self._gate and not self._gate.allow():
            self._velocity_vx = 0.0
            self._velocity_vy = 0.0
            return
        # Cap velocity at a sane upper bound so a bug can never spin
        # the camera at 10 000 px/sec.
        cap = 3000.0
        self._velocity_vx = max(-cap, min(cap, float(vx_per_sec)))
        self._velocity_vy = max(-cap, min(cap, float(vy_per_sec)))

    def _velocity_loop(self) -> None:
        """Background thread that converts the current velocity into
        a stream of single-pixel mouse-move events. Sleeps when
        velocity is zero so it costs nothing when idle."""
        period = 1.0 / max(30, self._velocity_tick_hz)
        last_t = time.perf_counter()
        while not self._velocity_stop.is_set():
            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            vx = self._velocity_vx
            vy = self._velocity_vy
            if vx == 0.0 and vy == 0.0:
                self._velocity_residual_x = 0.0
                self._velocity_residual_y = 0.0
                # Idle — sleep longer to save CPU.
                self._velocity_stop.wait(timeout=0.01)
                last_t = time.perf_counter()
                continue
            if self._gate and not self._gate.allow():
                self._velocity_stop.wait(timeout=0.02)
                last_t = time.perf_counter()
                continue
            self._velocity_residual_x += vx * dt
            self._velocity_residual_y += vy * dt
            ix = int(self._velocity_residual_x)
            iy = int(self._velocity_residual_y)
            if ix or iy:
                self._velocity_residual_x -= ix
                self._velocity_residual_y -= iy
                try:
                    self._backend.move(ix, iy)
                except Exception:
                    pass
            self._velocity_stop.wait(timeout=period)



    def _rate_limit(self) -> None:
        max_eps = self.cfg.max_events_per_sec
        if not max_eps:
            return
        min_dt = 1.0 / max_eps
        now = time.perf_counter()
        dt = now - self._last_event_ts
        if dt < min_dt:
            time.sleep(min_dt - dt)
        self._last_event_ts = time.perf_counter()


    def move(self, dx: int, dy: int) -> None:
        if self._gate and not self._gate.allow():
            return
        self._rate_limit()
        with self._lock:
            if self._gate and not self._gate.allow():
                return
            self._backend.move(int(dx), int(dy))

    def move_smooth(self, dx: int, dy: int, duration_ms: Optional[int] = None) -> None:
        duration = duration_ms if duration_ms is not None else self.cfg.move_duration_ms
        steps = self.cfg.curve_steps

        if steps <= 1 or duration <= 0:
            self.move(dx, dy)
            return

        sent_x = 0
        sent_y = 0
        step_sleep = duration / 1000.0 / steps

        def eased_at(i: int) -> float:
            t = i / steps
            return self.cfg.easing_fn(t)

        for i in range(1, steps + 1):
            # Abort early if the gate closes mid-movement (e.g. focus lost)
            if self._gate and not self._gate.allow():
                return

            curr = eased_at(i)
            target_x = int(round(dx * curr))
            target_y = int(round(dy * curr))

            sub_dx = target_x - sent_x
            sub_dy = target_y - sent_y

            if sub_dx or sub_dy:
                self.move(sub_dx, sub_dy)
                sent_x += sub_dx
                sent_y += sub_dy

            time.sleep(step_sleep)



    def flick(self, dx: int, dy: int) -> None:
        scaled_dx = int(dx * self.cfg.flick_multiplier)
        scaled_dy = int(dy * self.cfg.flick_multiplier)
        self.move_smooth(scaled_dx, scaled_dy, duration_ms=int(self.cfg.move_duration_ms * 0.35))

    def track_target(self, dx: int, dy: int) -> None:
        self.move_smooth(dx, dy, duration_ms=self.cfg.move_duration_ms)

    def left_press(self) -> None:
        if self._gate and not self._gate.allow():
            return
        if not self._pressed["left"]:
            self._backend.press_left()
            self._pressed["left"] = True

    def left_release(self) -> None:
        # Releases are NEVER gated. If the gate closed while a button
        # was held (focus loss), we still need the up-event to fire so
        # the game doesn't see the click as still-pressed when focus
        # returns. The gate is a "no new input" rule, not a "freeze the
        # current world" rule.
        if self._pressed["left"]:
            self._backend.release_left()
            self._pressed["left"] = False

    def right_press(self) -> None:
        if self._gate and not self._gate.allow():
            return
        if not self._pressed["right"]:
            self._backend.press_right()
            self._pressed["right"] = True

    def right_release(self) -> None:
        # See left_release — releases are never gated.
        if self._pressed["right"]:
            self._backend.release_right()
            self._pressed["right"] = False

    def left_click(self, duration: Optional[float] = None) -> None:
        if self._gate and not self._gate.allow():
            return
        dur = duration or self.cfg.default_click_duration
        self.left_press()
        try:
            time.sleep(dur)
        finally:
            # try/finally so a gate flip mid-click still releases.
            self.left_release()

    def right_click(self, duration: Optional[float] = None) -> None:
        if self._gate and not self._gate.allow():
            return
        dur = duration or self.cfg.default_click_duration
        self.right_press()
        try:
            time.sleep(dur)
        finally:
            self.right_release()

    def simultaneous_click(self, duration: Optional[float] = None) -> None:
        if self._gate and not self._gate.allow():
            return
        dur = duration or self.cfg.default_click_duration
        self.left_press()
        self.right_press()
        try:
            time.sleep(dur)
        finally:
            self.right_release()
            self.left_release()

    def scroll_up(self, amount: int = 1):
        if self._gate and not self._gate.allow():
            return
        if self.cfg.enable_scroll:
            self._backend.scroll(0, amount)

    def scroll_down(self, amount: int = 1):
        if self._gate and not self._gate.allow():
            return
        if self.cfg.enable_scroll:
            self._backend.scroll(0, -amount)

    def scroll_horizontal(self, amount: int = 1):
        if self._gate and not self._gate.allow():
            return
        if self.cfg.enable_scroll:
            self._backend.scroll(amount, 0)

    def emergency_stop(self) -> None:
        """Force-release every mouse button and halt velocity-mode motion
        regardless of the gate state. Mirror of ``Keyboard.emergency_stop``;
        Safety calls this on the panic hotkey so a left-click that was
        already in flight when focus was lost still releases."""
        with self._lock:
            self._velocity_vx = 0.0
            self._velocity_vy = 0.0
            self._velocity_residual_x = 0.0
            self._velocity_residual_y = 0.0
            if self._pressed["left"]:
                try:
                    self._backend.release_left()
                except Exception:
                    pass
                self._pressed["left"] = False
            if self._pressed["right"]:
                try:
                    self._backend.release_right()
                except Exception:
                    pass
                self._pressed["right"] = False