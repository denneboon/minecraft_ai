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
from typing import Optional, Tuple, Callable

try:
    from pynput.mouse import Button, Controller
except ImportError as e:
    # ``raise ... from e`` preserves the original ImportError context so
    # a missing-pynput failure shows both messages (the friendly hint
    # AND the underlying "no module named pynput") instead of hiding
    # the cause behind the bare RuntimeError.
    raise RuntimeError(
        "pynput is required for mouse control. Install via pip install pynput"
    ) from e


# ---------------------------------------------------------------------------
# Win32 detection — hoisted above the absolute-cursor helpers so they
# can branch on it. The SendInput-based backend below uses the same
# flag plus ``wintypes`` / the ``ctypes`` symbol imported here.
# ---------------------------------------------------------------------------
try:
    import ctypes
    from ctypes import wintypes
    _WIN32_OK = True
except ImportError:
    _WIN32_OK = False


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

    # ================= ABSOLUTE-POSITION HUMANLIKE REACH =================
    # Used in MENUS where MC's cursor is unlocked and movement is
    # ABSOLUTE (inventory hover, GUI navigation). Gameplay camera uses
    # the relative-delta APIs above, never this one.
    #
    # Motion follows a minimum-jerk velocity profile (zero velocity +
    # zero acceleration at both endpoints) over a distance-scaled
    # duration. A small perpendicular bow and per-step jitter keep
    # the path from looking mechanically straight — real arms don't
    # reach in a perfect line.
    screen_move_min_ms:   int   = 80        # short hops still take this long
    screen_move_max_ms:   int   = 450       # cap for big diagonal traversals
    screen_move_ms_per_px: float = 0.6      # base distance scaling
    screen_move_step_hz:  int   = 240       # micro-step cadence
    # Perpendicular bow as a fraction of the move's straight-line
    # distance. 0.0 = perfectly straight; 0.06 ≈ what real hand
    # tracking looks like for a deliberate UI hover.
    screen_move_curvature: float = 0.06
    # Per-step random offset in pixels. Adds the high-frequency wobble
    # that no smooth easing curve produces. Set to 0 for deterministic
    # paths (testing).
    screen_move_jitter_px: float = 0.4

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
# Absolute-cursor helpers (Win32) — used by the eased screen-reach API
# -------------------------------
# The OS-level cursor APIs live in user32. They're cheap (single syscall
# each) but the easing path calls them ~240 times per move, so we read
# the handle once and stash it. On non-Windows hosts both functions
# silently no-op — callers gate behaviour on whether ``Mouse`` (or
# ``eased_screen_move``) is supported via runtime checks elsewhere.

if _WIN32_OK:
    try:
        _USER32_CURSOR = ctypes.windll.user32
    except (OSError, AttributeError):
        _USER32_CURSOR = None
else:
    _USER32_CURSOR = None


class _CURSOR_POINT(ctypes.Structure if _WIN32_OK else object):
    if _WIN32_OK:
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _get_screen_xy_raw() -> Tuple[int, int]:
    """Instant read of the OS cursor position. ``(0, 0)`` if unavailable."""
    if _USER32_CURSOR is None:
        return (0, 0)
    pt = _CURSOR_POINT()
    try:
        _USER32_CURSOR.GetCursorPos(ctypes.byref(pt))
    except (OSError, AttributeError):
        return (0, 0)
    return (int(pt.x), int(pt.y))


def _set_screen_xy_raw(x: int, y: int) -> None:
    """Instant teleport of the OS cursor. The :func:`eased_screen_move`
    function calls this many times along a smooth path; external callers
    should prefer the eased version."""
    if _USER32_CURSOR is None:
        return
    try:
        _USER32_CURSOR.SetCursorPos(int(x), int(y))
    except (OSError, AttributeError):
        pass


def _smoothstep_min_jerk(t: float) -> float:
    """Minimum-jerk position curve ``10t³ − 15t⁴ + 6t⁵``. Velocity AND
    acceleration are zero at both endpoints, which matches biological
    reach motion (Flash & Hogan, 1985)."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def eased_screen_move(target_x: int,
                      target_y: int,
                      *,
                      min_ms:    int   = 80,
                      max_ms:    int   = 450,
                      ms_per_px: float = 0.6,
                      step_hz:   int   = 240,
                      curvature: float = 0.06,
                      jitter_px: float = 0.4,
                      duration_ms: Optional[int] = None,
                      gate=None,
                      interrupt_event: Optional[threading.Event] = None) -> None:
    """
    Standalone humanlike absolute move of the OS cursor.

    Pulled out of :class:`Mouse` so unit tests and standalone scripts
    can ease the cursor without spinning up the full runtime — and
    so the ``Mouse`` method is just a thin wrapper that also zeros
    velocity-mode commands before driving the path.

    Parameters
    ----------
    target_x, target_y : int
        Desktop pixel coordinates.
    duration_ms : Optional[int]
        Override auto-scaling. ``None`` means
        ``clamp(min_ms, distance × ms_per_px, max_ms)``.
    curvature, jitter_px : float
        Path realism. Curvature is a perpendicular bow as a fraction
        of straight-line distance; jitter is a per-step random offset
        in pixels.
    gate : Optional[InputGate-like]
        Anything with ``.allow() -> bool``. Aborts the move when False.
    interrupt_event : Optional[threading.Event]
        Sleeps use ``event.wait(timeout)`` so setting the event mid-
        move cuts the in-flight reach short — used by ``Mouse.stop``
        to drop animation on shutdown.
    """
    if _USER32_CURSOR is None:
        return
    if gate is not None and not gate.allow():
        return

    sx, sy = _get_screen_xy_raw()
    dx = float(target_x) - sx
    dy = float(target_y) - sy
    dist = (dx * dx + dy * dy) ** 0.5
    if dist < 1.0:
        _set_screen_xy_raw(target_x, target_y)
        return

    if duration_ms is None:
        duration_ms = int(max(min_ms, min(max_ms, dist * ms_per_px)))

    hz = max(60, step_hz)
    n_steps = max(2, int(duration_ms * hz / 1000))
    step_dt = (duration_ms / 1000.0) / n_steps

    # Perpendicular bow direction; sign deterministic per destination
    # so consecutive hovers over neighbouring slots don't oscillate
    # between left- and right-bowing paths.
    perp_x = -dy / dist
    perp_y =  dx / dist
    bow_sign = 1.0 if ((target_x * 1103515245 + target_y) & 1) else -1.0
    bow_amplitude = curvature * dist * bow_sign

    # Deterministic 32-bit LCG seeded by the endpoints — the same hover
    # always replays identically, which is useful for bug reports.
    jitter_state = (sx * 2654435761
                    + sy * 40503
                    + target_x * 7
                    + target_y) & 0xFFFFFFFF

    for i in range(1, n_steps + 1):
        if gate is not None and not gate.allow():
            return
        u = i / n_steps
        s = _smoothstep_min_jerk(u)
        # Bow vanishes at the endpoints, max at the midpoint.
        # 4u(1−u) peaks at 1.0 when u=0.5, equals 0 at the ends.
        bow = bow_amplitude * (4.0 * u * (1.0 - u))

        # LCG: x_{n+1} = (1664525 * x_n + 1013904223) mod 2^32
        jitter_state = (1664525 * jitter_state + 1013904223) & 0xFFFFFFFF
        jx = ((jitter_state & 0xFFFF) / 0xFFFF - 0.5) * 2.0 * jitter_px
        jitter_state = (1664525 * jitter_state + 1013904223) & 0xFFFFFFFF
        jy = ((jitter_state & 0xFFFF) / 0xFFFF - 0.5) * 2.0 * jitter_px

        x = sx + dx * s + perp_x * bow + jx
        y = sy + dy * s + perp_y * bow + jy
        _set_screen_xy_raw(int(round(x)), int(round(y)))

        if interrupt_event is not None:
            # Interruptible sleep — ``event.set()`` from another thread
            # cuts the wait short. The event being set doesn't end the
            # reach (the gate / loop counter does that), it only avoids
            # blocking shutdown on a step that's still snoozing.
            if interrupt_event.wait(timeout=step_dt):
                break
        else:
            time.sleep(step_dt)

    if gate is None or gate.allow():
        # Land exactly on target — float math + jitter on the last
        # step can otherwise leave the cursor a pixel off.
        _set_screen_xy_raw(int(target_x), int(target_y))


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
#
# (Win32 / ctypes detection is hoisted to the top of the module so the
# absolute-cursor helpers above can also branch on ``_WIN32_OK``.)

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
        mouse.move_to_screen_xy(x, y)    # cursor to an absolute screen point

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
            except Exception as e:
                # Falling back to pynput is a correctness DEGRADATION
                # on Windows — pynput uses SetCursorPos, which MC's
                # Raw Input listener ignores during in-game play. The
                # camera will NOT respond to bot input in this fallback.
                # Surface it so the user knows why the AI looks broken.
                print(f"[mouse][WARN] Win32 SendInput backend init failed "
                      f"({e!r}); falling back to pynput. MC's raw-input "
                      f"camera will NOT respond to bot input in-game.")
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

        # Optional "play area" — the MC window rect (left, top, right, bottom)
        # in desktop pixels. When set, every ABSOLUTE cursor move
        # (``move_to_screen_xy``, used for menu/inventory slot clicks) is
        # clamped inside it. A misread inventory slot can otherwise send the
        # cursor onto the desktop/taskbar and right-click it — which opens a
        # context menu, steals focus from MC, and derails the run. Clamping
        # keeps a stray click on a (wrong) IN-WINDOW slot instead of off-app.
        # ``None`` = no fence (tests / standalone). Relative gameplay moves
        # (``move``/``set_velocity``) are unaffected — MC's locked cursor
        # ignores absolute position there.
        self._play_area: Optional[Tuple[int, int, int, int]] = None

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
            # CRITICAL: zero out velocity state BEFORE signalling stop.
            # The worker thread reads ``_velocity_vx/vy`` once per loop
            # iteration at the TOP of the loop, then emits motion at
            # the bottom. If we set ``_velocity_stop`` first WITHOUT
            # zeroing velocity, a worker iteration already past the
            # ``while not _velocity_stop.is_set()`` check will still
            # emit one more motion event with the old non-zero
            # velocity — that's the "mouse moves a bit AFTER the
            # program says it's done" symptom. Zeroing first means
            # even if one stray iteration races through, it emits
            # nothing.
            self._velocity_vx = 0.0
            self._velocity_vy = 0.0
            self._velocity_residual_x = 0.0
            self._velocity_residual_y = 0.0
            self._running = False
            self._velocity_stop.set()
        # Wait outside the lock so the worker can exit cleanly. Bumped
        # the join timeout to 0.5 s (was 0.25) so a worker mid-sleep
        # on a slow Windows scheduler has time to wake and exit
        # before we tear the backend down.
        try:
            if self._velocity_thread is not None:
                self._velocity_thread.join(timeout=0.5)
        except Exception:
            pass
        self._velocity_thread = None
        self._backend.stop()

    def set_play_area(self, bounds: Optional[Tuple[int, int, int, int]]) -> None:
        """Fence absolute cursor moves to ``(left, top, right, bottom)`` desktop
        pixels — the MC window. Pass ``None`` to remove the fence. Set it once
        the capture window is known; refresh it if the window moves/resizes."""
        if bounds is None:
            self._play_area = None
            return
        l, t, r, b = (int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3]))
        if r < l: l, r = r, l
        if b < t: t, b = b, t
        self._play_area = (l, t, r, b)

    def _clamp_play_area(self, x: int, y: int) -> Tuple[int, int]:
        """Clamp an absolute target inside the play area (a small inset keeps
        it off the very edge). No-op when no play area is set."""
        pa = self._play_area
        if pa is None:
            return int(x), int(y)
        l, t, r, b = pa
        m = 2                                     # inset so we never sit on the frame
        cx = min(max(int(x), l + m), r - m)
        cy = min(max(int(y), t + m), b - m)
        if (cx, cy) != (int(x), int(y)):
            print(f"[mouse][WARN] absolute move ({int(x)},{int(y)}) is OUTSIDE the "
                  f"MC window {pa} — clamped to ({cx},{cy}). A container slot was "
                  f"likely misread (GUI not open?); refusing to click off-app.")
        return cx, cy

    def _cursor_off_window(self) -> bool:
        """True if a play area is set and the OS cursor is OUTSIDE it right now.
        A button-press is refused in that case so the bot can never click the
        desktop/taskbar — e.g. when a relative camera nudge moved the UNLOCKED
        GUI cursor off-window. During gameplay MC grabs the cursor and keeps it
        at the window centre, so this never blocks a real in-game click."""
        pa = self._play_area
        if pa is None:
            return False
        x, y = _get_screen_xy_raw()
        l, t, r, b = pa
        return not (l <= x <= r and t <= y <= b)

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
        velocity is zero so it costs nothing when idle.

        Three places we check the stop event:

        1. Loop-top ``while not _velocity_stop.is_set()`` — usual exit.
        2. After computing the integer step, RIGHT BEFORE emitting —
           ``stop()`` zeros velocity before signalling, so this is
           the last-line defence against emitting motion after the
           shutdown sequence began.
        3. After the gate-closed path — same belt-and-suspenders.
        """
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
                # Two final race-guards — both eliminate the
                # symptom "mouse moves a bit after the program
                # says it's done":
                #
                # 1. stop() was signalled while this iteration was
                #    in flight — don't emit a stale residual on the
                #    way out.
                # 2. set_velocity(0, 0) was called mid-iteration —
                #    the vx/vy we cached at the top of the loop are
                #    now stale and the user's intent is "no more
                #    motion". Honour that even though our residuals
                #    still hold the leftover from the previous
                #    accumulation step.
                if self._velocity_stop.is_set():
                    break
                if (self._velocity_vx == 0.0
                        and self._velocity_vy == 0.0):
                    self._velocity_residual_x = 0.0
                    self._velocity_residual_y = 0.0
                else:
                    try:
                        self._backend.move(ix, iy)
                    except Exception as e:
                        # First-failure warning: a recurring move-error
                        # means MC is no longer reachable (window closed,
                        # off-screen, permission revoked). Without
                        # surfacing this, the agent silently fails to
                        # rotate forever. After the first warn we stay
                        # silent so the 240 Hz worker can't spam the
                        # console.
                        if not getattr(self, "_move_warn_emitted", False):
                            self._move_warn_emitted = True
                            print(f"[mouse][WARN] backend.move({ix}, "
                                  f"{iy}) raised {e!r} — further "
                                  f"errors silenced")
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
        if self._cursor_off_window():
            print("[mouse][WARN] refusing LEFT click — cursor is OUTSIDE the MC "
                  "window (would click the desktop/taskbar).")
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
        if self._cursor_off_window():
            print("[mouse][WARN] refusing RIGHT click — cursor is OUTSIDE the MC "
                  "window (would click the desktop/taskbar).")
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

    # -----------------------------------------------------------------
    # Absolute (screen-coord) movement — used in MENUS only.
    #
    # Gameplay camera control uses ``track_target`` / ``set_velocity``
    # which emit RELATIVE deltas to MC's locked-cursor raw-input
    # listener. When the cursor is UNLOCKED (inventory, paused menu,
    # crafting screen) MC reads the OS cursor position directly — for
    # those screens we need absolute moves, and a teleport via raw
    # SetCursorPos reads as a robotic snap. ``move_to_screen_xy``
    # produces a minimum-jerk reach that feels like a real hand
    # arriving at the target.
    # -----------------------------------------------------------------

    @staticmethod
    def get_screen_xy() -> Tuple[int, int]:
        """Read the OS cursor position (desktop pixels). Windows only —
        on other platforms returns ``(0, 0)``. Used by callers that want
        to remember the original cursor before a hover sequence so they
        can restore it later via :meth:`move_to_screen_xy`."""
        return _get_screen_xy_raw()

    def move_to_screen_xy(self,
                          target_x: int,
                          target_y: int,
                          *,
                          duration_ms: Optional[int] = None,
                          curvature: Optional[float] = None,
                          jitter_px: Optional[float] = None) -> None:
        """
        Eased absolute move of the OS cursor to ``(target_x, target_y)``.

        Motion follows the minimum-jerk profile
        ``s(t) = 10t³ − 15t⁴ + 6t⁵`` so velocity AND acceleration are
        zero at both endpoints — the same accel-then-decel pattern a
        human arm produces when reaching for a UI target. A perpendicular
        bow scaled by ``curvature`` keeps the path from being perfectly
        straight, and a small per-step random offset (``jitter_px``)
        adds the high-frequency wobble no smooth curve has.

        Duration auto-scales with distance unless explicitly overridden:
        ``clamp(min_ms, distance × ms_per_px, max_ms)``. Short hops
        stay snappy, long traversals don't take forever.

        Side effects:
        * Zeros any in-flight velocity command so the background worker
          doesn't fight the absolute path.
        * Respects the input gate — aborts mid-move if the gate closes.
        """
        if self._gate and not self._gate.allow():
            return
        # SAFETY FENCE: never let an absolute move leave the MC window. A
        # misread inventory slot would otherwise reach the desktop/taskbar and
        # the follow-up click steals focus / opens a context menu.
        target_x, target_y = self._clamp_play_area(target_x, target_y)
        # Cancel any active velocity command — otherwise the worker
        # keeps emitting relative motion while we're trying to position
        # absolutely, and the cursor judders along a compound path.
        self._velocity_vx = 0.0
        self._velocity_vy = 0.0

        cfg = self.cfg
        eased_screen_move(
            target_x, target_y,
            min_ms=cfg.screen_move_min_ms,
            max_ms=cfg.screen_move_max_ms,
            ms_per_px=cfg.screen_move_ms_per_px,
            step_hz=cfg.screen_move_step_hz,
            curvature=cfg.screen_move_curvature if curvature is None else curvature,
            jitter_px=cfg.screen_move_jitter_px if jitter_px is None else jitter_px,
            duration_ms=duration_ms,
            gate=self._gate,
            interrupt_event=self._velocity_stop,
        )

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
