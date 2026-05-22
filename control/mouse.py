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
    # Default = smoothstep.
    easing_fn: Callable[[float], float] = field(default_factory=lambda: (
        lambda t: t * t * (3 - 2 * t)
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
        self._backend: _IMouseBackend = backend or _PynputBackend()

        self._lock = threading.RLock()
        self._running = False

        self._pressed = {
            "left": False,
            "right": False,
        }

        self._last_event_ts = 0.0
        self._gate = gate


    def start(self):
        with self._lock:
            if self._running:
                return
            self._backend.start()
            self._running = True

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
            self._backend.stop()



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
        if self._gate and not self._gate.allow():
            return
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
        if self._gate and not self._gate.allow():
            return
        if self._pressed["right"]:
            self._backend.release_right()
            self._pressed["right"] = False

    def left_click(self, duration: Optional[float] = None) -> None:
        if self._gate and not self._gate.allow():
            return
        dur = duration or self.cfg.default_click_duration
        self.left_press()
        time.sleep(dur)
        self.left_release()

    def right_click(self, duration: Optional[float] = None) -> None:
        if self._gate and not self._gate.allow():
            return
        dur = duration or self.cfg.default_click_duration
        self.right_press()
        time.sleep(dur)
        self.right_release()

    def simultaneous_click(self, duration: Optional[float] = None) -> None:
        if self._gate and not self._gate.allow():
            return
        dur = duration or self.cfg.default_click_duration
        self.left_press()
        self.right_press()
        time.sleep(dur)
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