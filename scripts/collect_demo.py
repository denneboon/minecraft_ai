#!/usr/bin/env python3
# scripts/collect_demo.py
"""
Record human Minecraft demonstrations as (frame, input-state) pairs.

What it produces
----------------
For each session::

    data/raw/<session-id>/
        frames/000000001.png   — captured frames (downscaled by default)
        frames/000000002.png
        ...
        events.jsonl           — one JSON object per recorded frame
        meta.json              — session metadata

``events.jsonl`` line format (one per frame, ordered by frame index)::

    {
        "frame":     1,
        "ts":        0.05,                  # seconds since session start
        "keys":      ["w", "shift"],        # keys held during this frame
        "buttons":   {"left": false,
                      "right": true,
                      "middle": false},
        "scroll_dy": 0,                     # accumulated scroll since
                                            # last frame (sign = direction)
        "mc_focused": true                  # was MC the foreground window?
    }

``meta.json`` records the resolutions, FPS, MC version, etc. — everything
a future training script needs to load the session.

Design notes
------------
* Frames are saved on a **background writer thread** so the capture
  loop never blocks on disk I/O.
* Keyboard and mouse events are captured via ``pynput`` listeners that
  run in their own threads. We aggregate them into "currently held"
  state and snapshot that state per-frame — VPT-style.
* Raw mouse movement is NOT recorded (when MC has cursor capture,
  Windows doesn't deliver raw deltas to user-space listeners reliably).
  When we want to record where the player looks, we'll derive it from
  the F3 yaw/pitch deltas via the existing OCR.
* When MC is not the foreground window, we still record the frame and
  events but tag the event ``mc_focused=false`` so training can filter
  out garbage (user typing in another app, etc.).

Usage
-----
::

    # default: 5 minutes, 20 Hz, downscaled to 0.5×, into data/raw/<ts>
    python scripts/collect_demo.py

    # explicit
    python scripts/collect_demo.py --duration 120 --fps 20 --downscale 0.5
    python scripts/collect_demo.py --output data/raw/my-session

Press **Ctrl+Shift+F12** to stop early (same hotkey as the agent's
emergency stop).
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# Ensure we can import from the project root regardless of cwd.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from vision.capture import Capture, CaptureConfig  # noqa: E402
from vision.ocr import build_f3_reader, F3Reader  # noqa: E402
from utils.focus import _find_minecraft_hwnd  # noqa: E402

try:
    import yaml  # noqa: E402
except ImportError:
    yaml = None  # type: ignore


# ─── Configuration defaults ──────────────────────────────────────────

DEFAULT_FPS         = 20
DEFAULT_DURATION    = 300         # 5 minutes
DEFAULT_DOWNSCALE   = 0.5         # 1920×1129 → 960×564
WRITER_QUEUE_SIZE   = 128         # frames buffered before back-pressure
PROGRESS_EVERY      = 100         # print a status line every N frames

# F3 OCR is ~30 ms per call. Running it at 5 Hz adds ~150 ms/sec of CPU,
# which is fine. The bot can interpolate yaw/pitch between samples;
# the agent loop already does this via state.f3 carry-over.
F3_OCR_INTERVAL_SEC = 0.20


# ─── Key normalisation ───────────────────────────────────────────────
#
# Goals:
#   1. Use Windows virtual-key codes when available — they give the
#      actual physical key regardless of held modifiers. (pynput's
#      KeyCode.char produces "\x04" for Ctrl+D, "\x17" for Ctrl+W,
#      etc., which is useless for training — we want "d" and "w".)
#   2. Collapse left/right modifier variants. For Minecraft, holding
#      left vs right Shift is the same intent, so we record both as
#      "shift". Matches what control/keyboard.py emits at runtime.
#   3. Return None for keys we can't usefully identify.

# Pynput Key.* enum names that should collapse to a canonical form.
# Used as a fallback when key.vk isn't populated (which happens for
# some modifier events depending on Windows driver / pynput version).
_PYNPUT_NAMES: Dict[str, str] = {
    "ctrl":   "control", "ctrl_l":  "control", "ctrl_r":  "control",
    "shift":  "shift",   "shift_l": "shift",   "shift_r": "shift",
    "alt":    "alt",     "alt_l":   "alt",     "alt_r":   "alt", "alt_gr": "alt",
    "cmd":    "cmd",     "cmd_l":   "cmd",     "cmd_r":   "cmd",
    "esc":    "escape",
}


_VK_NAMES: Dict[int, str] = {
    # Modifiers (left/right collapsed)
    0xA0: "shift",   0xA1: "shift",
    0xA2: "control", 0xA3: "control",
    0xA4: "alt",     0xA5: "alt",
    0x10: "shift",   # virtual non-distinguished modifiers
    0x11: "control",
    0x12: "alt",
    # Whitespace / control
    0x20: "space", 0x0D: "enter", 0x09: "tab",
    0x08: "backspace", 0x1B: "escape", 0x14: "caps_lock",
    # Arrow + nav
    0x25: "left", 0x26: "up", 0x27: "right", 0x28: "down",
    0x2D: "insert", 0x2E: "delete",
    0x21: "page_up", 0x22: "page_down",
    0x24: "home", 0x23: "end",
    # Windows / menu
    0x5B: "cmd", 0x5C: "cmd", 0x5D: "menu",
}
# F1..F24
for _i in range(1, 25):
    _VK_NAMES[0x70 + _i - 1] = f"f{_i}"


def _normalize_key(key) -> Optional[str]:
    """
    Convert a pynput key event to a stable lower-case string.

    Prefers ``key.vk`` (the Windows virtual-key code) because pynput's
    ``key.char`` is modifier-affected on Windows — pressing D while Ctrl
    is held yields ``char='\\x04'`` even though the player meant ``"d"``.
    """
    vk = getattr(key, "vk", None)
    if vk is not None:
        # Letters A-Z → lowercase ascii
        if 0x41 <= vk <= 0x5A:
            return chr(vk + 32)
        # Top-row digits 0-9
        if 0x30 <= vk <= 0x39:
            return chr(vk)
        # Numpad digits → same key as top-row digit
        if 0x60 <= vk <= 0x69:
            return chr(0x30 + (vk - 0x60))
        if vk in _VK_NAMES:
            return _VK_NAMES[vk]

    # Fallback to a printable .char (no modifier garbage handled here).
    try:
        ch = key.char
    except AttributeError:
        ch = None
    if ch is not None and len(ch) == 1 and ch.isprintable():
        return ch.lower()

    # Last resort — stringify the special-key name pynput gives us
    # (e.g. "Key.ctrl_l" → "ctrl_l") and canonicalise modifiers.
    name = str(key).replace("Key.", "").lower()
    if name in _PYNPUT_NAMES:
        return _PYNPUT_NAMES[name]
    return name or None


# ─── Input recorder ──────────────────────────────────────────────────

@dataclass
class InputSnapshot:
    keys:      List[str]
    buttons:   Dict[str, bool]
    scroll_dy: int


class InputRecorder:
    """
    Tracks held keys + mouse-button state via pynput listeners. Call
    ``snapshot()`` once per frame to grab the current state and reset
    the scroll accumulator.

    Thread-safe: listener callbacks run on pynput's background threads
    while the capture loop reads via ``snapshot()``.
    """

    def __init__(self) -> None:
        self._held: Set[str] = set()
        self._buttons: Dict[str, bool] = {
            "left":   False,
            "right":  False,
            "middle": False,
        }
        self._scroll_dy: int = 0
        self._lock = threading.Lock()
        self._kb_listener = None
        self._mouse_listener = None

    def start(self) -> None:
        from pynput import keyboard as _kb, mouse as _ms

        def on_press(key):
            n = _normalize_key(key)
            if n is None:
                return
            with self._lock:
                self._held.add(n)

        def on_release(key):
            n = _normalize_key(key)
            if n is None:
                return
            with self._lock:
                self._held.discard(n)

        def on_click(_x, _y, button, pressed):
            with self._lock:
                self._buttons[button.name] = bool(pressed)

        def on_scroll(_x, _y, _dx, dy):
            with self._lock:
                self._scroll_dy += int(dy)

        self._kb_listener = _kb.Listener(on_press=on_press, on_release=on_release)
        self._mouse_listener = _ms.Listener(on_click=on_click, on_scroll=on_scroll)
        self._kb_listener.start()
        self._mouse_listener.start()

    def stop(self) -> None:
        for lis in (self._kb_listener, self._mouse_listener):
            try:
                if lis is not None:
                    lis.stop()
            except Exception:
                pass

    def snapshot(self) -> InputSnapshot:
        with self._lock:
            snap = InputSnapshot(
                keys=sorted(self._held),
                buttons=dict(self._buttons),
                scroll_dy=self._scroll_dy,
            )
            self._scroll_dy = 0
        return snap


# ─── Background frame writer ─────────────────────────────────────────

class FrameWriter(threading.Thread):
    """Pulls (idx, RGB-array) off a queue and writes PNGs to disk."""

    def __init__(self, frames_dir: Path,
                 compression: int = 3,
                 q_size: int = WRITER_QUEUE_SIZE) -> None:
        super().__init__(daemon=True, name="demo-frame-writer")
        self._dir = frames_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._q: "queue.Queue" = queue.Queue(maxsize=q_size)
        self._stop_evt = threading.Event()
        self._compression = compression
        self._dropped = 0
        self._written = 0

    def enqueue(self, idx: int, frame_rgb: np.ndarray) -> bool:
        """Returns False if the queue is full (frame was dropped)."""
        try:
            self._q.put_nowait((idx, frame_rgb))
            return True
        except queue.Full:
            self._dropped += 1
            return False

    def stop(self) -> None:
        self._stop_evt.set()
        try:
            self._q.put_nowait((None, None))  # wake up the consumer
        except queue.Full:
            pass

    @property
    def written(self) -> int:
        return self._written

    @property
    def dropped(self) -> int:
        return self._dropped

    def run(self) -> None:
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), int(self._compression)]
        while not self._stop_evt.is_set() or not self._q.empty():
            try:
                idx, frame = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            if idx is None:
                break
            path = self._dir / f"{idx:09d}.png"
            try:
                cv2.imwrite(str(path),
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                            params)
                self._written += 1
            except Exception as e:
                print(f"[WRITER] failed to write {path}: {e}")


# ─── Stop hotkey ─────────────────────────────────────────────────────

class StopHotkey:
    """Watch for Ctrl+Shift+F12 (matches the agent's emergency stop)."""

    HOTKEY = {"<ctrl>", "<shift>", "<f12>"}

    def __init__(self) -> None:
        self._pressed: Set[str] = set()
        self._event = threading.Event()
        self._listener = None

    def start(self) -> None:
        from pynput import keyboard as _kb

        def on_press(key):
            try:
                ch = key.char
            except AttributeError:
                ch = None
            if ch is not None:
                self._pressed.add(ch.lower())
            else:
                self._pressed.add(
                    "<" + str(key).replace("Key.", "") + ">"
                )
            if self.HOTKEY.issubset(self._pressed):
                self._event.set()

        def on_release(key):
            try:
                ch = key.char
            except AttributeError:
                ch = None
            if ch is not None:
                self._pressed.discard(ch.lower())
            else:
                self._pressed.discard(
                    "<" + str(key).replace("Key.", "") + ">"
                )

        self._listener = _kb.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def stop(self) -> None:
        try:
            if self._listener is not None:
                self._listener.stop()
        except Exception:
            pass

    @property
    def event(self) -> threading.Event:
        return self._event


# ─── MC focus helper (best effort) ───────────────────────────────────

def _mc_foreground(target_hwnd: Optional[int]) -> bool:
    if target_hwnd is None:
        return True
    try:
        import win32gui
        return win32gui.GetForegroundWindow() == target_hwnd
    except Exception:
        return True


# ─── Settings loader ─────────────────────────────────────────────────

def _load_settings() -> dict:
    if yaml is None:
        return {}
    path = ROOT / "config" / "settings.yaml"
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ─── CLI ─────────────────────────────────────────────────────────────

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[1] if __doc__ else "",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                   help="Max seconds to record. Ctrl+Shift+F12 stops early.")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS,
                   help="Target frame rate.")
    p.add_argument("--downscale", type=float, default=DEFAULT_DOWNSCALE,
                   help="Frame downscale factor (1.0 = native resolution).")
    p.add_argument("--output", type=str, default=None,
                   help="Output directory. Default: data/raw/<timestamp>.")
    p.add_argument("--no-frames", action="store_true",
                   help="Don't write frames to disk — useful to dry-run "
                        "the event capture path.")
    p.add_argument("--countdown", type=int, default=3,
                   help="Seconds to wait before starting (gives you time "
                        "to tab to Minecraft). 0 to start immediately.")
    return p


# ─── Main loop ───────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> int:
    args = _build_cli().parse_args(argv)

    wins = _find_minecraft_hwnd()
    if not wins:
        print("[ERROR] Minecraft (javaw.exe) is not running. "
              "Start your instance and try again.")
        return 2
    target_hwnd = wins[0][0]
    print(f"[demo] Found Minecraft hwnd={target_hwnd}")

    settings = _load_settings()
    cap_cfg = (settings.get("capture") or {})
    ui_scale = int(cap_cfg.get("ui_scale", 2))

    # Output directory.
    session_id = (args.output and Path(args.output).name) \
        or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.output) if args.output else ROOT / "data" / "raw" / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    events_path = out_dir / "events.jsonl"
    meta_path   = out_dir / "meta.json"
    print(f"[demo] Output → {out_dir}")

    # Build capture using the same logic as main.py.
    capture = Capture(CaptureConfig(
        hwnd=target_hwnd,
        use_client_area=bool(cap_cfg.get("use_client_area", True)),
        max_fps=float(args.fps) * 2,   # let mss grab faster than we use
        track_window_each_frame=True,
        name="demo_capture",
    ))
    capture.start()

    # Warm up — first capture call sets the rect and is sometimes slow.
    try:
        first = capture.get_frame()
        full_h, full_w = first.shape[:2]
        print(f"[demo] Capture: {full_w}×{full_h}")
    except Exception as e:
        print(f"[ERROR] Initial capture failed: {e}")
        capture.stop()
        return 3

    # Decide frame resolution after downscale.
    if args.downscale != 1.0:
        out_w = max(1, int(round(full_w * args.downscale)))
        out_h = max(1, int(round(full_h * args.downscale)))
    else:
        out_w, out_h = full_w, full_h
    print(f"[demo] Saved frame size: {out_w}×{out_h}")

    recorder = InputRecorder();   recorder.start()
    writer:   Optional[FrameWriter] = None
    if not args.no_frames:
        writer = FrameWriter(frames_dir);   writer.start()
    stopper = StopHotkey();        stopper.start()

    # Build the F3 reader if possible — we use it at low frequency
    # (5 Hz) to label each recorded frame with current yaw/pitch and
    # x/y/z. Without these, the recording can't tell us "where the
    # player was looking" — which is half the action signal a vision
    # model needs to learn.
    f3_reader: Optional[F3Reader] = None
    try:
        f3_reader = build_f3_reader(settings)
        print(f"[demo] F3 reader: backend={f3_reader.backend}")
    except Exception as e:
        print(f"[demo][WARN] F3 reader unavailable ({e}); "
              "events will not include yaw/pitch.")
    last_f3_ts = 0.0
    last_yaw:   Optional[float] = None
    last_pitch: Optional[float] = None
    last_xyz:   Optional[tuple] = None

    # One-shot check: does the user have F3 on right now? If not, the
    # recording will still work but yaw/pitch/xyz will all be null —
    # which makes the demo much less useful for training. Warn loudly
    # so they can enable it before the countdown finishes.
    if f3_reader is not None:
        try:
            probe = capture.get_frame()
            if not f3_reader._f3_panel_visible(probe):
                print("[demo][HINT] F3 debug overlay isn't on in Minecraft. "
                      "Press F3 NOW to capture yaw/pitch/xyz in this session "
                      "— otherwise those fields will be null for every frame.")
        except Exception:
            pass

    print("[demo] Recording — press Ctrl+Shift+F12 to stop.")
    print(f"[demo] Will run up to {args.duration:.0f} s at {args.fps:.0f} Hz.")

    if args.countdown > 0:
        from utils.focus import activate_minecraft
        try:
            activate_minecraft(maximize=True)
        except Exception:
            pass
        for n in range(args.countdown, 0, -1):
            print(f"[demo] Starting in {n}...")
            time.sleep(1.0)

    tick_period = 1.0 / float(args.fps)
    start_wall  = time.time()
    start_perf  = time.perf_counter()

    frame_idx   = 0
    abort       = "unknown"
    events_file = open(events_path, "w", encoding="utf-8", buffering=1)

    try:
        while True:
            tick_start = time.perf_counter()
            elapsed_total = tick_start - start_perf
            if stopper.event.is_set():
                abort = "user_hotkey"
                break
            if elapsed_total >= args.duration:
                abort = "duration_reached"
                break

            try:
                full_frame = capture.get_frame()
            except Exception as e:
                print(f"[demo][WARN] capture failed: {e}")
                time.sleep(0.05)
                continue

            # F3 OCR at 5 Hz on the full-res frame. We update the
            # cached yaw/pitch/xyz and stamp the latest values onto
            # every event, so training can interpolate. Stale ≤200 ms.
            if (f3_reader is not None
                    and (tick_start - last_f3_ts) >= F3_OCR_INTERVAL_SEC):
                try:
                    info = f3_reader.read(full_frame)
                    if info.yaw is not None:
                        last_yaw = float(info.yaw)
                    if info.pitch is not None:
                        last_pitch = float(info.pitch)
                    if info.position() is not None:
                        last_xyz = info.position()
                except Exception:
                    pass
                last_f3_ts = tick_start

            # Downscale for storage.
            if (out_w, out_h) != (full_w, full_h):
                frame = cv2.resize(full_frame, (out_w, out_h),
                                   interpolation=cv2.INTER_AREA)
            else:
                frame = full_frame

            frame_idx += 1
            if writer is not None:
                writer.enqueue(frame_idx, frame)

            snap = recorder.snapshot()
            focused = _mc_foreground(target_hwnd)
            event = {
                "frame":       frame_idx,
                "ts":          round(elapsed_total, 4),
                "keys":        snap.keys,
                "buttons":     snap.buttons,
                "scroll_dy":   snap.scroll_dy,
                "mc_focused":  focused,
                "yaw":         (round(last_yaw, 2)   if last_yaw   is not None else None),
                "pitch":       (round(last_pitch, 2) if last_pitch is not None else None),
                "xyz":         ([round(c, 3) for c in last_xyz] if last_xyz else None),
            }
            events_file.write(json.dumps(event) + "\n")

            if frame_idx % PROGRESS_EVERY == 0:
                realised = frame_idx / max(elapsed_total, 1e-6)
                drops = writer.dropped if writer is not None else 0
                print(f"  frame {frame_idx:6d}  t={elapsed_total:5.1f}s  "
                      f"realised={realised:4.1f} Hz  "
                      f"focused={focused}  drops={drops}")

            # Pace.
            elapsed_tick = time.perf_counter() - tick_start
            if elapsed_tick < tick_period:
                time.sleep(tick_period - elapsed_tick)

    except KeyboardInterrupt:
        abort = "ctrl_c"
    except Exception as e:
        abort = f"error: {type(e).__name__}: {e}"
        raise
    finally:
        end_perf = time.perf_counter()
        duration = end_perf - start_perf
        try:
            events_file.flush(); events_file.close()
        except Exception:
            pass
        stopper.stop()
        recorder.stop()
        if writer is not None:
            writer.stop()
            writer.join(timeout=5.0)
        capture.stop()

        meta = {
            "session_id":         session_id,
            "start_ts_unix":      start_wall,
            "duration_sec":       round(duration, 3),
            "fps_target":         args.fps,
            "fps_realised":       round(frame_idx / max(duration, 1e-6), 2),
            "capture_resolution": [full_w, full_h],
            "frame_resolution":   [out_w, out_h],
            "downscale":          args.downscale,
            "ui_scale":           ui_scale,
            "frame_count":        frame_idx,
            "frames_written":     (writer.written if writer is not None else 0),
            "frames_dropped":     (writer.dropped if writer is not None else 0),
            "no_frames_mode":     bool(args.no_frames),
            "abort_reason":       abort,
            "mc_hwnd":            target_hwnd,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Size summary.
        events_size = events_path.stat().st_size if events_path.exists() else 0
        if writer is not None and frames_dir.exists():
            frame_bytes = sum(
                p.stat().st_size for p in frames_dir.iterdir() if p.is_file()
            )
        else:
            frame_bytes = 0

        print()
        print(f"[demo] Done. {abort}")
        print(f"        duration:  {duration:.1f} s")
        print(f"        frames:    {frame_idx} "
              f"({meta['fps_realised']:.1f} Hz realised)")
        print(f"        events:    {events_size/1024:.1f} KiB")
        print(f"        frame I/O: {frame_bytes/1024/1024:.1f} MiB"
              + (f"  ({writer.dropped} dropped)"
                 if (writer and writer.dropped) else ""))
        print(f"        meta:      {meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
