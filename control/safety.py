
from __future__ import annotations
import time
import threading
from dataclasses import dataclass
from typing import Optional, Callable
import platform

try:
    import pygetwindow as gw
except Exception:
    gw = None

from pynput import keyboard as pynput_keyboard
from rich.console import Console
from rich.table import Table
from rich import box

try:
    import psutil
except Exception:
    psutil = None

if platform.system().lower() == "windows":
    try:
        import win32process
    except Exception:
        win32process = None
else:
    win32process = None

from utils.focus import _find_minecraft_hwnd


# Collapse pynput's left/right modifier variants to a generic name so a
# configured combo like ``<ctrl>+<shift>+<f12>`` matches what pynput actually
# reports when those keys are pressed (``<ctrl_l>``, ``<shift_l>``, …).
# Without this the emergency-stop hotkey silently never fires.
_MOD_BASES = ("ctrl", "shift", "alt", "cmd")


def _norm_key_name(name: str) -> str:
    s = str(name)
    inner = s[1:-1] if s.startswith("<") and s.endswith(">") else s
    for base in _MOD_BASES:
        if inner in (base, f"{base}_l", f"{base}_r", f"{base}_gr"):
            return f"<{base}>"
    return s




@dataclass
class SafetyConfig:
    minecraft_title_query: str = "minecraft"
    check_focus_interval: float = 0.075
    emergency_hotkey: tuple = ("<ctrl>", "<shift>", "<f12>")
    log_actions: bool = True
    log_focus_events: bool = True
    allow_run_without_focus: bool = False
    auto_stop_on_focus_loss: bool = True
    print_status_table: bool = True
    assume_focused_when_unknown: bool = False


    focus_loss_debounce_ms: int = 250
    startup_focus_grace_ms: int = 800


    debug_focus_trace: bool = False






class SafetyLogger:
    def __init__(self):
        self.console = Console()
        self._lock = threading.Lock()

    def log(self, msg: str):
        with self._lock:
            self.console.print(f"[bold cyan][SAFETY][/bold cyan] {msg}")

    def warn(self, msg: str):
        with self._lock:
            self.console.print(f"[bold yellow][WARN][/bold yellow] {msg}")

    def error(self, msg: str):
        with self._lock:
            self.console.print(f"[bold red][ERROR][/bold red] {msg}")

    def table(self, title: str, data: dict):
        table = Table(title=title, box=box.ROUNDED, style="bold")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="magenta")
        for k, v in data.items():
            table.add_row(k, str(v))
        with self._lock:
            self.console.print(table)






class Safety:
    def __init__(self, config: Optional[SafetyConfig] = None, gate=None):
        self.cfg = config or SafetyConfig()
        self.gate = gate
        self.logger = SafetyLogger()

        self._running = False
        self._focus_thread = None
        self._status_table_thread = None
        self._hotkey_thread = None
        self._hotkey_listener = None

        self._last_focus_true_ts = 0.0
        self._last_focus_false_ts = 0.0

        self._action_count = 0
        self._last_action = None

        self._on_emergency_stop = None

        self._focused_state: bool = False
        self._regain_ts: float = 0.0
        self._loss_start_ts: float = 0.0


        self._target_hwnd = None
        try:
            wins = _find_minecraft_hwnd()
            if wins:
                self._target_hwnd = wins[0][0]
        except Exception:
            pass





    def set_emergency_callback(self, fn: Callable) -> None:
        self._on_emergency_stop = fn

    def notify_action(self, action_name: str):
        if self.cfg.log_actions:
            self.logger.log(f"ACTION: {action_name}")
        self._last_action = action_name
        self._action_count += 1





    def _window_has_focus(self) -> bool:
        """
        Return True when Minecraft is the foreground window.

        Caches the MC window handle in ``self._target_hwnd`` so we
        don't enumerate all top-level windows on every check. Previous
        bug: this method used to NULL the cached handle whenever the
        foreground window wasn't MC, causing a full window-enumeration
        on every single 75 ms tick when the user had any other app
        focused — wasteful and a source of jitter. We now keep the
        cached handle and only re-look-up when it goes stale (the
        window no longer exists).
        """
        if self.cfg.allow_run_without_focus:
            return True

        try:
            import win32gui
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return False

            # Fast path: cached MC hwnd is the foreground window.
            if self._target_hwnd and hwnd == self._target_hwnd:
                return True

            # Cache invalidation: the cached MC hwnd is no longer a
            # valid window (game closed and relaunched, etc.).
            if self._target_hwnd and not win32gui.IsWindow(self._target_hwnd):
                self._target_hwnd = None

            # If we don't have a cached MC hwnd, find it once.
            if self._target_hwnd is None:
                try:
                    wins = _find_minecraft_hwnd()
                    if wins:
                        self._target_hwnd = wins[0][0]
                except Exception:
                    pass

            if self._target_hwnd and hwnd == self._target_hwnd:
                return True

            # Fallback: foreground is a Java process but not the one
            # we cached. Trust the process-name match.
            import win32process
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            name = psutil.Process(pid).name().lower() if psutil else ""
            return name in ("javaw.exe", "java.exe")
        except Exception:
            return bool(self.cfg.assume_focused_when_unknown)

    def _focus_loop(self):
        self.logger.log("Focus monitoring thread started.")
        grace_s = float(self.cfg.startup_focus_grace_ms) / 1000.0
        debounce_s = float(self.cfg.focus_loss_debounce_ms) / 1000.0

        while self._running:
            try:
                now = time.perf_counter()
                has_focus = self._window_has_focus()

                if has_focus and not self._focused_state:
                    self._focused_state = True
                    self._regain_ts = now
                    self._loss_start_ts = 0.0

                    if self.gate:
                        self.gate.set_allowed(True)
                    if self.cfg.log_focus_events:
                        self.logger.log("[green]Minecraft regained focus[/green]")

                elif (not has_focus) and self._focused_state:
                    self._focused_state = False
                    self._loss_start_ts = now

                if not has_focus and self.gate and self.gate.allow():
                    in_grace    = self._regain_ts     and (now - self._regain_ts)     < grace_s
                    in_debounce = self._loss_start_ts and (now - self._loss_start_ts) < debounce_s
                    if not (in_grace or in_debounce):
                        if self.gate:
                            self.gate.set_allowed(False)
                        if self.cfg.log_focus_events:
                            self.logger.warn("Minecraft lost focus")
                        if self.cfg.auto_stop_on_focus_loss:
                            self._trigger_emergency_stop("Focus lost")
            except Exception as e:
                if self.cfg.debug_focus_trace:
                    self.logger.error(f"Focus loop error: {e!r}")

            time.sleep(self.cfg.check_focus_interval)





    def _hotkey_loop(self):
        combo = set(_norm_key_name(k) for k in self.cfg.emergency_hotkey)
        pressed = set()

        def _name(key):
            # ``key.char`` exists for character keys but is missing on
            # ``Key.shift`` / ``Key.f12`` / etc., which raise
            # AttributeError. We use the named form for those so the
            # emergency-hotkey combo (e.g. ``<ctrl>+<shift>+<f12>``)
            # matches the registered combo string. CRITICAL: pynput
            # reports the LEFT/RIGHT variant of modifiers (Ctrl ->
            # ``Key.ctrl_l`` -> ``<ctrl_l>``), which would NEVER match a
            # configured ``<ctrl>`` — so the panic combo silently never
            # fired. ``_norm_key_name`` collapses ctrl_l/ctrl_r/shift_l/…
            # to the generic ``<ctrl>``/``<shift>`` the config uses.
            try:
                return key.char.lower()
            except AttributeError:
                return _norm_key_name(f"<{str(key).replace('Key.', '')}>")

        def on_press(key):
            pressed.add(_name(key))
            if combo.issubset(pressed):
                self._trigger_emergency_stop("Emergency hotkey")

        def on_release(key):
            pressed.discard(_name(key))

        listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
        self._hotkey_listener = listener
        listener.start()
        while self._running:
            time.sleep(0.1)





    def _trigger_emergency_stop(self, reason: str):
        self.logger.error(f"EMERGENCY STOP: {reason}")

        if self.gate:
            self.gate.set_allowed(False)

        if getattr(self, "_on_emergency_stop", None):
            try:
                self._on_emergency_stop()
            except Exception as e:
                self.logger.error(f"Error in emergency callback: {e}")

    def start(self):
        if self._running:
            return
        self._running = True
        self.logger.log("Safety controller starting…")


        if self.gate and self._window_has_focus():
            self.gate.set_allowed(True)

        self._focus_thread = threading.Thread(target=self._focus_loop, daemon=True)
        self._focus_thread.start()

        self._status_table_thread = threading.Thread(
            target=self._status_table_loop, daemon=True
        )
        self._status_table_thread.start()

        self._hotkey_thread = threading.Thread(target=self._hotkey_loop, daemon=True)
        self._hotkey_thread.start()

    def stop(self):
        self._running = False
        if self.gate:
            self.gate.set_allowed(False)

        # Join ALL three daemon threads (focus monitor, status-table
        # renderer, hotkey listener) so they don't print anything
        # AFTER the caller has logged its shutdown line. Without
        # joining, the focus loop's ~75 ms sleep interval can let it
        # fire one more "Minecraft lost focus" message after the user
        # sees "[MAIN] Done." — exactly the kind of trailing noise
        # that makes "is the agent really done?" hard to answer.
        # 1.0 s is a generous timeout; threads exit promptly when
        # ``_running`` flips False.
        for attr in ("_focus_thread", "_status_table_thread",
                      "_hotkey_thread"):
            th = getattr(self, attr, None)
            if th is not None and th.is_alive():
                th.join(timeout=1.0)
            setattr(self, attr, None)

        try:
            if self._hotkey_listener:
                self._hotkey_listener.stop()
                self._hotkey_listener = None
        except Exception:
            # pynput's Listener.stop() can raise RuntimeError if the
            # internal listener thread has already exited or its
            # backing OS hook is gone — both are fine during shutdown.
            # Anything else is also best-effort cleanup; don't block
            # the rest of safety.stop() on it.
            pass

        self.logger.warn("Safety controller stopped.")





    def _status_table_loop(self):
        last_snapshot: dict = {}
        while self._running and self.cfg.print_status_table:
            data = {
                "Focused":      self.gate.allow() if self.gate else False,
                "Last Action":  self._last_action or "(none)",
                "Action Count": self._action_count,
            }
            # Only re-print when something actually changed
            if data != last_snapshot:
                self.logger.table("Bot Status", data)
                last_snapshot = dict(data)
            time.sleep(0.5)





    def allow_input(self) -> bool:
        if self.cfg.allow_run_without_focus:
            return True
        return self.gate.allow() if self.gate else False
