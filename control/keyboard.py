# control/keyboard.py
"""
Robust keyboard control for Minecraft (and similar games).

Features:
- Press, release, tap
- Chords, macros
- State tracking to avoid stuck keys
- Debounce & rate-limiting
- Hotbar management with Minecraft tick-safe cooldown (50 ms default)
- Backend abstraction (pynput today; SendInput can be added later)

Notes:
- Run Minecraft in windowed/borderless mode for highest reliability.
"""

from __future__ import annotations

import threading
import time
import platform
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# --------------------------
# Configuration
# --------------------------

@dataclass
class KeyboardConfig:
    # Timing
    tap_default: float = 0.06          # seconds key is held for a tap
    max_actions_per_sec: int = 120     # soft cap for keyboard actions
    debounce_ms: int = 10              # ignore duplicate press/release within X ms

    # Behavior
    strict_state: bool = True          # track pressed state; avoid duplicates
    auto_release_on_stop: bool = True  # release all held keys on stop
    allow_repeat_press: bool = False   # allow press on already-pressed keys

    # Sprint/Sneak styles
    sprint_key: str = "control"
    sneak_key: str = "shift"
    sprint_mode: str = "hold"          # "hold" or "toggle"
    sneak_mode: str = "hold"           # "hold" or "toggle"

    # Autorun feature
    autorun_key: str = "w"             # movement key to hold for autorun

    # Context (future)
    context: str = "gameplay"          # "gameplay", "inventory", "chat"
    context_blocklists: Dict[str, Set[str]] = field(default_factory=lambda: {
        # Block common movement/modifier keys and hotbar numbers while chat is open
        "chat": {"w", "a", "s", "d", "space", "shift", "control", "tab",
                 "1","2","3","4","5","6","7","8","9","0","q"},
        "inventory": set(),
        "gameplay": set(),
    })

    # Key aliases to normalize incoming names (lowercase)
    key_aliases: Dict[str, str] = field(default_factory=lambda: {
        "left_shift": "shift",
        "right_shift": "shift_r",
        "left_control": "control",
        "right_control": "control_r",
        "left_alt": "alt",
        "right_alt": "alt_r",
        "return": "enter",
        "esc": "escape",
        "spacebar": "space",
        "caps": "caps_lock",
        "cmd": "cmd",
        "win": "cmd",
        "super": "cmd",
        "lshift": "shift",
        "rshift": "shift_r",
        "lctrl": "control",
        "rctrl": "control_r",
        "lalt": "alt",
        "ralt": "alt_r",
    })

    # --------- Minecraft tick-safe hotbar cooldown ----------
    # Enforced *only* for number-key hotbar slot selection.
    # 55 ms > 50 ms (1 MC tick) so each press lands on a separate tick
    # even with OS scheduler jitter. Safe default everywhere.
    hotbar_cooldown_ms: int = 55

    # Verbose log of every press/release/tap call. Off by default;
    # toggled via the debug.verbose_inputs settings flag.
    verbose: bool = False


# --------------------------
# Backend Interface
# --------------------------

class _IKeyboardBackend:
    """Interface a backend must implement."""
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def press(self, key: str) -> None: ...
    def release(self, key: str) -> None: ...


# --------------------------
# Pynput Backend
# --------------------------

class _PynputBackend(_IKeyboardBackend):
    """Backend using pynput (cross-platform)."""
    def __init__(self):
        from pynput.keyboard import Controller, Key
        self._controller = Controller()
        self._Key = Key
        self._special_map = self._build_special_map()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def _build_special_map(self) -> Dict[str, object]:
        from pynput.keyboard import Key
        m = {
            "shift": Key.shift,
            "shift_r": getattr(Key, "shift_r", Key.shift),
            "control": Key.ctrl,
            "control_r": getattr(Key, "ctrl_r", Key.ctrl),
            "alt": Key.alt,
            "alt_r": getattr(Key, "alt_r", Key.alt),
            "cmd": getattr(Key, "cmd", getattr(Key, "cmd_r", None)) or getattr(Key, "cmd_l", None) or Key.cmd,
            "tab": Key.tab,
            "caps_lock": Key.caps_lock,
            "enter": Key.enter,
            "space": Key.space,
            "backspace": Key.backspace,
            "delete": Key.delete,
            "escape": Key.esc,
            "up": Key.up,
            "down": Key.down,
            "left": Key.left,
            "right": Key.right,
            "home": Key.home,
            "end": Key.end,
            "page_up": Key.page_up,
            "page_down": Key.page_down,
            "insert": Key.insert,
            "print_screen": getattr(Key, "print_screen", None) or getattr(Key, "print_screen", None),
            "menu": getattr(Key, "menu", None),
        }
        # Function keys
        for i in range(1, 25):
            name = f"f{i}"
            key_attr = getattr(Key, name, None)
            if key_attr is not None:
                m[name] = key_attr
        return {k: v for k, v in m.items() if v is not None}

    def _translate(self, key: str) -> object:
        """Return pynput Key or a literal character."""
        k = key.lower()
        if k in self._special_map:
            return self._special_map[k]
        if len(k) == 1:
            return k  # literal character
        if k.startswith("numpad_") and len(k) == len("numpad_x"):
            # pynput doesn't reliably expose distinct numpad keys; fall back to char best-effort.
            return k[-1] if k[-1].isdigit() else k[-1]
        raise ValueError(f"Unsupported key for pynput backend: {key!r}")

    def press(self, key: str) -> None:
        self._controller.press(self._translate(key))

    def release(self, key: str) -> None:
        self._controller.release(self._translate(key))


# --------------------------
# Keyboard High-Level API
# --------------------------

class Keyboard:
    def __init__(
        self,
        config: Optional[KeyboardConfig] = None,
        keymap: Optional[Dict[str, str]] = None,
        backend: Optional[_IKeyboardBackend] = None,
        gate = None
    ):
        self._gate = gate
        self.cfg = config or KeyboardConfig()
        self._os = platform.system().lower()

        # Action → key mapping (e.g., {"move_forward": "w"})
        self.keymap: Dict[str, str] = keymap or {}
        self._aliases = self.cfg.key_aliases

        # Hotbar keys 1..9
        hb: List[str] = []
        if keymap and "hotbar_slots" in keymap and isinstance(keymap["hotbar_slots"], (list, tuple)):
            hb = [self._normalize(k) for k in keymap["hotbar_slots"]]
        if not hb:
            hb = [str(i) for i in range(1, 10)]
        self._hotbar_keys: List[str] = hb
        self._current_slot: Optional[int] = None  # 1..len(hb)
        self._last_slot: Optional[int] = None

        # Backend
        self._backend: _IKeyboardBackend = backend or _PynputBackend()

        # State & concurrency
        self._pressed: Set[str] = set()
        self._last_event_ms: Dict[Tuple[str, str], float] = {}
        self._lock = threading.RLock()
        self._running = False

        # Rate limiting
        self._max_aps = self.cfg.max_actions_per_sec
        self._last_action_ts = 0.0

        # Autorun state
        self._autorun = False

        # --------- NEW: hotbar cooldown tracker ---------
        self._last_hotbar_switch: float = 0.0

    # --------------------------
    # Lifecycle
    # --------------------------

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._backend.start()
            self._running = True

    def stop(self) -> None:
        with self._lock:
            if self.cfg.auto_release_on_stop:
                self._release_all_locked()
            self._backend.stop()
            self._running = False

    # --------------------------
    # Public Methods (single key)
    # --------------------------

    def press(self, key_or_action: str) -> None:
        if self._gate and not self._gate.allow():
            if self.cfg.verbose:
                self._vlog(f"PRESS  {key_or_action!r:>18}  skip(gate-closed)")
            return
        key = self._resolve(key_or_action)
        if self._is_blocked_in_context(key):
            if self.cfg.verbose:
                self._vlog(f"PRESS  {key!r:>18}  skip(context-blocked)")
            return
        # FAST PATH — if the key is already pressed and we don't allow
        # repeat presses, skip BEFORE paying the rate-limit sleep in
        # _should_send. Without this, holding W across 20 Hz ticks burns
        # 8.3 ms per tick on a sleep that exists only so we can no-op.
        if (self.cfg.strict_state
                and not self.cfg.allow_repeat_press
                and key in self._pressed):
            if self.cfg.verbose:
                self._vlog(f"PRESS  {key!r:>18}  skip(already-pressed)")
            return
        with self._lock:
            if (self.cfg.strict_state
                    and not self.cfg.allow_repeat_press
                    and key in self._pressed):
                if self.cfg.verbose:
                    self._vlog(f"PRESS  {key!r:>18}  skip(race-lost)")
                return
            if not self._should_send("press", key):
                if self.cfg.verbose:
                    self._vlog(f"PRESS  {key!r:>18}  skip(debounce)")
                return
            self._backend.press(key)
            self._pressed.add(key)
            self._stamp("press", key)
            if self.cfg.verbose:
                self._vlog(f"PRESS  {key!r:>18}  SENT")

    def release(self, key_or_action: str) -> None:
        if self._gate and not self._gate.allow():
            if self.cfg.verbose:
                self._vlog(f"REL    {key_or_action!r:>18}  skip(gate-closed)")
            return
        key = self._resolve(key_or_action)
        # FAST PATH — same idea: a release on a key that isn't held is a
        # no-op under strict_state, so skip before paying the sleep.
        if self.cfg.strict_state and key not in self._pressed:
            return  # silent — this is the common case for unheld keys
        with self._lock:
            if self.cfg.strict_state and key not in self._pressed:
                return
            if not self._should_send("release", key):
                if self.cfg.verbose:
                    self._vlog(f"REL    {key!r:>18}  skip(debounce)")
                return
            try:
                self._backend.release(key)
            finally:
                self._pressed.discard(key)
                self._stamp("release", key)
                if self.cfg.verbose:
                    import traceback as _tb
                    # Capture caller — knowing WHO releases W is the
                    # whole point of this debug mode.
                    caller = "  ".join(
                        f"{f.filename.split(chr(92))[-1]}:{f.lineno}"
                        for f in _tb.extract_stack(limit=8)[:-1]
                    )
                    self._vlog(f"REL    {key!r:>18}  SENT  via {caller}")

    def type_text(self, text: str, char_delay: float = 0.006) -> None:
        """
        Type a literal string into whatever window is foreground.

        Used by main.py to send chat messages in Minecraft (open chat
        with ``t``, then call ``type_text("hello world")``, then ``tap("enter")``).
        Goes through the pynput backend's Controller.type() because that
        handles the keyboard-layout translation correctly for symbols
        and capital letters; the rest of our keyboard API only deals
        with named keys.
        """
        if self._gate and not self._gate.allow():
            return
        backend = getattr(self, "_backend", None)
        controller = getattr(backend, "_controller", None)
        if controller is None:
            # Backend doesn't expose a controller; fall back to per-char tap
            # which handles ASCII letters/digits but not symbols.
            for ch in text:
                if ch == " ":
                    self.tap("space", 0.02)
                elif ch.isprintable():
                    self.tap(ch, 0.02)
                time.sleep(char_delay)
            return
        for ch in text:
            try:
                controller.type(ch)
            except Exception:
                # Skip characters the backend can't synthesise (e.g.
                # non-ASCII on a US layout).
                pass
            if char_delay > 0:
                time.sleep(char_delay)

    def tap(self, key_or_action: str, duration: Optional[float] = None) -> None:
        if self._gate and not self._gate.allow():
            return
        key = self._resolve(key_or_action)
        if self._is_blocked_in_context(key):
            return
        hold_s = self.cfg.tap_default if duration is None else float(duration)
        self.press(key)
        _sleep_safe(hold_s)
        self.release(key)

    # --------------------------
    # Public Methods (multi-key)
    # --------------------------

    def chord(self, keys_or_actions: Sequence[str], hold_s: float) -> None:
        if self._gate and not self._gate.allow():
            return
        keys = [self._resolve(k) for k in keys_or_actions if not self._is_blocked_in_context(self._resolve(k))]
        with self._lock:
            for k in keys:
                if self._should_send("press", k) and (not self.cfg.strict_state or k not in self._pressed or self.cfg.allow_repeat_press):
                    self._backend.press(k)
                    self._pressed.add(k)
                    self._stamp("press", k)
        _sleep_safe(hold_s)
        with self._lock:
            for k in reversed(keys):  # slightly more human-like release order
                if self._should_send("release", k) and (not self.cfg.strict_state or k in self._pressed):
                    try:
                        self._backend.release(k)
                    finally:
                        self._pressed.discard(k)
                        self._stamp("release", k)

    def macro(self, steps: Sequence[Tuple[Sequence[str], float]]) -> None:
        if self._gate and not self._gate.allow():
            return
        for keys, hold_s in steps:
            self.chord(keys, hold_s)

    # --------------------------
    # Action-centric helpers
    # --------------------------

    def hold(self, action_or_key: str) -> None:
        if self._gate and not self._gate.allow():
            return
        self.press(action_or_key)

    def release_action(self, action_or_key: str) -> None:
        if self._gate and not self._gate.allow():
            return
        self.release(action_or_key)

    def tap_action(self, action_or_key: str, duration: Optional[float] = None) -> None:
        if self._gate and not self._gate.allow():
            return
        self.tap(action_or_key, duration)

    # --------------------------
    # Sprint / Sneak / Autorun
    # --------------------------

    def sprint_on(self) -> None:
        k = self._normalize(self.cfg.sprint_key)
        if self.cfg.sprint_mode == "toggle":
            self.tap(k, self.cfg.tap_default)
        else:
            self.press(k)

    def sprint_off(self) -> None:
        if self.cfg.sprint_mode == "hold":
            self.release(self.cfg.sprint_key)

    def sneak_on(self) -> None:
        if self.cfg.sneak_mode == "toggle":
            self.tap(self.cfg.sneak_key, self.cfg.tap_default)
        else:
            self.press(self.cfg.sneak_key)

    def sneak_off(self) -> None:
        if self.cfg.sneak_mode == "hold":
            self.release(self.cfg.sneak_key)

    def autorun_on(self) -> None:
        key = self._normalize(self.cfg.autorun_key)
        if not self._autorun:
            self.press(key)
            self._autorun = True

    def autorun_off(self) -> None:
        key = self._normalize(self.cfg.autorun_key)
        if self._autorun:
            self.release(key)
            self._autorun = False

    # --------------------------
    # Hotbar Management (1..9) with tick-safe cooldown
    # --------------------------

    def set_hotbar_keys(self, keys: Sequence[str]) -> None:
        """Replace hotbar keys with a new sequence (len 9 recommended)."""
        norm = [self._normalize(k) for k in keys]
        if len(norm) < 2:
            raise ValueError("Hotbar keys must contain at least 2 entries; recommended 9 (slots 1..9).")
        self._hotbar_keys = norm

    def get_hotbar_keys(self) -> List[str]:
        return list(self._hotbar_keys)

    def select_hotbar_slot(self, slot: int, tap_duration: Optional[float] = None, remember: bool = True) -> None:
        """
        Select a hotbar slot via number hotkey (1-based index).
        Enforces a pre-press cooldown so each press hits a separate MC tick.
        """
        idx = self._validate_slot(slot)
        key = self._hotbar_keys[idx - 1]
        dur = self.cfg.tap_default if tap_duration is None else float(tap_duration)

        # Respect contexts that block number keys (e.g., chat)
        if self._is_blocked_in_context(key):
            return

        # ---- Minecraft tick-safe gap BEFORE pressing ----
        # Gap is measured from the END of the last tap so the previous key is
        # fully released before we begin the cooldown window.
        gap = max(0.0, float(self.cfg.hotbar_cooldown_ms) / 1000.0)
        now = time.perf_counter()
        since = now - self._last_hotbar_switch
        if since < gap:
            _sleep_safe(gap - since)

        # Send the key (press + release) for the desired slot
        self.tap(key, dur)

        # Stamp AFTER the tap so the cooldown starts from when the key was
        # actually released, not from when we started waiting.
        self._last_hotbar_switch = time.perf_counter()

        if remember:
            self._last_slot = self._current_slot
            self._current_slot = idx

    def next_hotbar_slot(self, step: int = 1, remember: bool = True) -> int:
        total = len(self._hotbar_keys)
        curr = self._current_slot if (self._current_slot and 1 <= self._current_slot <= total) else 1
        target = ((curr - 1 + step) % total) + 1
        self.select_hotbar_slot(target, remember=remember)
        return target

    def previous_hotbar_slot(self, remember: bool = True) -> int:
        return self.next_hotbar_slot(step=-1, remember=remember)

    def swap_to(self, slot: int) -> None:
        self.select_hotbar_slot(slot, remember=True)

    def swap_back(self) -> None:
        if self._last_slot is not None:
            self.select_hotbar_slot(self._last_slot, remember=True)

    # --------------------------
    # Context & Safety
    # --------------------------

    def set_context(self, context: str) -> None:
        self.cfg.context = context

    def emergency_stop(self) -> None:
        """Release ALL keys we believe are pressed."""
        with self._lock:
            self._release_all_locked()

    def _vlog(self, msg: str) -> None:
        """Verbose log helper. Off unless cfg.verbose is True."""
        if not self.cfg.verbose:
            return
        t = (time.perf_counter() - getattr(self, "_t0", time.perf_counter()))
        if not hasattr(self, "_t0"):
            self._t0 = time.perf_counter()
        print(f"[KB {t:7.3f}] {msg}")

    def resync_pressed_keys(self) -> None:
        """
        Re-emit a key-down event for every key we currently believe is held.

        Why this exists: Windows synthesises ``WM_KEYUP`` events for the
        previously-foreground window whenever focus changes (so games
        don't end up with "stuck" keys after Alt-Tab). pynput knows
        nothing about this and our ``_pressed`` set still says W is
        held — so the per-tick fast-path skips re-emitting the key,
        and MC silently never sees W go back down. The result looks
        exactly like the bot is tap-tap-tapping the movement keys.

        Calling this on every gate transition False→True, and once per
        second as a belt-and-suspenders heartbeat, keeps MC's notion of
        held keys in sync with ours without breaking the fast-path's
        no-op behaviour for true held-key ticks.
        """
        if self._gate and not self._gate.allow():
            if self.cfg.verbose:
                self._vlog("RESYNC skip(gate-closed)")
            return
        with self._lock:
            held = list(self._pressed)
            for key in held:
                try:
                    self._backend.press(key)
                except Exception:
                    pass
            self._last_action_ts = time.perf_counter()
            if self.cfg.verbose:
                self._vlog(f"RESYNC re-emitted {held}")

    # --------------------------
    # Internals
    # --------------------------

    def _resolve(self, key_or_action: str) -> str:
        """Map an action or a raw key to a normalized key string."""
        k = key_or_action.strip()
        if k in self.keymap:
            k = self.keymap[k]
        return self._normalize(k)

    def _normalize(self, key: str) -> str:
        k = key.strip().lower()
        if k in self._aliases:
            k = self._aliases[k]
        # f-keys already lowercase; digits & letters unchanged
        return k

    def _validate_slot(self, slot: int) -> int:
        if not isinstance(slot, int):
            raise TypeError("slot must be an integer (1-based index).")

        total = len(self._hotbar_keys)
        if not (1 <= slot <= total):
            raise ValueError(f"slot must be between 1 and {total}, got {slot}.")

        return slot

    def _is_blocked_in_context(self, key: str) -> bool:
        blocked = self.cfg.context_blocklists.get(self.cfg.context, set())
        return key in blocked

    def _should_send(self, kind: str, key: str) -> bool:
        """
        Rate limit and debounce logic. Returns True if we should send the event.
        """
        now = time.perf_counter()
        # Rate limit (global for keyboard actions)
        if self._max_aps and self._max_aps > 0:
            min_dt = 1.0 / float(self._max_aps)
            elapsed = now - self._last_action_ts
            if elapsed < min_dt:
                _sleep_safe(min_dt - elapsed)
            self._last_action_ts = time.perf_counter()
            now = self._last_action_ts  # refresh after potential sleep

        # Debounce (per key + event kind) — uses post-sleep timestamp
        last = self._last_event_ms.get((kind, key), 0.0)
        if (now - last) * 1000.0 < self.cfg.debounce_ms:
            return False
        return True

    def _stamp(self, kind: str, key: str) -> None:
        self._last_event_ms[(kind, key)] = time.perf_counter()

    def _release_all_locked(self) -> None:
        # Assumes lock is held
        # Release in reverse (modifiers first) to avoid sticky states
        def sort_key(k: str) -> int:
            if "shift" in k or "control" in k or "alt" in k or "cmd" in k:
                return 0
            return 1
        for k in sorted(list(self._pressed), key=sort_key):
            try:
                self._backend.release(k)
            except Exception:
                pass
        self._pressed.clear()


# --------------------------
# Helpers
# --------------------------

def _sleep_safe(sec: float) -> None:
    """Sleep utility with sub-millisecond tail spin for short sleeps."""
    if sec <= 0:
        return
    t_end = time.perf_counter() + sec
    while True:
        now = time.perf_counter()
        dt = t_end - now
        if dt <= 0:
            break
        if dt > 0.002:
            time.sleep(dt - 0.0015)
        else:
            # spin the last ~1.5 ms to reduce scheduler jitter on very short waits
            pass