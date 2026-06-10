#!/usr/bin/env python3
"""
Record live keyboard / mouse input into a replayable macro.

Perform a technique once (e.g. bridging) and this captures the exact
key / click / scroll sequence + timing into a ``.json`` macro that
``control.script_runner`` (``python main.py --script <file>``) plays
back. Closes the loop: you never have to hand-author or "teach" a
repetitive technique — record it, replay it.

Usage
-----
    python tools/record_macro.py scripts/macros/mybridge.json
    python tools/record_macro.py out.json --stop f9 --moves
    python tools/record_macro.py out.mcs            # also writes the line DSL

Notes
-----
* Recording starts after a short countdown so you can alt-tab into MC.
* Press the STOP key (default F8) to finish. It is NOT recorded.
* Camera mouse-MOVES are OFF by default: Minecraft locks the cursor
  during gameplay, so move events there are recentre-noise, not real
  look motion. Pass ``--moves`` to capture them (useful for menu
  macros where the cursor is free).
* By default only input while Minecraft is the foreground window is
  recorded (``--no-mc-only`` to record everything).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pynput import keyboard as _kb
from pynput import mouse as _ms


def _key_name(key) -> str:
    """pynput key -> our script key name."""
    char = getattr(key, "char", None)
    if char:
        return char
    # Key.shift / Key.space / Key.ctrl_l / Key.f3 …
    name = str(key).replace("Key.", "")
    # pynput uses ctrl_l/ctrl_r/shift/shift_r/alt_l … the runner
    # normalises ctrl->control etc., so just hand the raw name over.
    return name


def _button_name(button) -> str:
    return getattr(button, "name", str(button)).lower()


def _mc_foreground_checker():
    """Return a callable() -> bool: True when Minecraft is foreground.
    Always-True fallback off Windows / when the handle can't be found."""
    try:
        import win32gui
        from utils.focus import _find_minecraft_hwnd
        wins = _find_minecraft_hwnd()
        hwnd = wins[0][0] if wins else None
        if hwnd is None:
            return lambda: True
        def _check() -> bool:
            try:
                return win32gui.GetForegroundWindow() == hwnd
            except Exception:
                return True
        return _check
    except Exception:
        return lambda: True


class _Recorder:
    def __init__(self, *, stop_key: str, record_moves: bool,
                 mc_only: bool, min_sleep_ms: int):
        self._stop_key = stop_key.lower()
        self._record_moves = record_moves
        self._focused = _mc_foreground_checker() if mc_only else (lambda: True)
        self._min_sleep_ms = max(0, min_sleep_ms)
        # Raw timeline: list of (t, op_dict). Moves stored as deltas.
        self._events: List[Tuple[float, Dict[str, Any]]] = []
        self._t0 = time.perf_counter()
        self._last_xy: Optional[Tuple[int, int]] = None
        self._stop = False

    # ── pynput callbacks ──────────────────────────────────────────

    def on_press(self, key):
        if _key_name(key).lower() == self._stop_key:
            self._stop = True
            return False   # stops the keyboard listener
        if self._focused():
            self._add({"op": "key_down", "key": _key_name(key)})

    def on_release(self, key):
        if _key_name(key).lower() == self._stop_key:
            self._stop = True
            return False
        if self._focused():
            self._add({"op": "key_up", "key": _key_name(key)})

    def on_click(self, x, y, button, pressed):
        if self._stop:
            return False
        if self._focused():
            b = _button_name(button)
            self._add({"op": "mouse_down" if pressed else "mouse_up", "button": b})

    def on_scroll(self, x, y, dx, dy):
        if self._stop:
            return False
        if self._focused() and dy:
            self._add({"op": "scroll", "amount": int(dy)})

    def on_move(self, x, y):
        if self._stop:
            return False
        if not self._record_moves:
            self._last_xy = (x, y)
            return
        if self._last_xy is not None and self._focused():
            dx, dy = x - self._last_xy[0], y - self._last_xy[1]
            if dx or dy:
                self._add({"op": "move", "dx": int(dx), "dy": int(dy)})
        self._last_xy = (x, y)

    def _add(self, op: Dict[str, Any]) -> None:
        self._events.append((time.perf_counter(), op))

    # ── Build the op list (insert sleeps, merge consecutive moves) ─

    def to_ops(self) -> List[Dict[str, Any]]:
        ops: List[Dict[str, Any]] = []
        prev_t: Optional[float] = None
        pending_move: Optional[List[int]] = None   # [dx, dy]

        def flush_move():
            nonlocal pending_move
            if pending_move and (pending_move[0] or pending_move[1]):
                ops.append({"op": "move", "dx": pending_move[0], "dy": pending_move[1]})
            pending_move = None

        for t, op in self._events:
            if op["op"] == "move":
                # Accumulate consecutive moves into one op (don't emit a
                # sleep mid-burst; the burst is effectively instantaneous).
                if pending_move is None:
                    pending_move = [0, 0]
                pending_move[0] += op["dx"]
                pending_move[1] += op["dy"]
                prev_t = t
                continue
            flush_move()
            if prev_t is not None:
                gap_ms = int(round((t - prev_t) * 1000.0))
                if gap_ms >= self._min_sleep_ms and gap_ms > 0:
                    ops.append({"op": "sleep", "ms": gap_ms})
            ops.append(op)
            prev_t = t
        flush_move()
        return ops


def _ops_to_mcs(name: str, ops: List[Dict[str, Any]]) -> str:
    """Serialise ops to the simple line DSL (best-effort, flat)."""
    lines = [f"name {name}"]
    for op in ops:
        k = op["op"]
        if k == "key_down":
            lines.append(f"down {op['key']}")
        elif k == "key_up":
            lines.append(f"up {op['key']}")
        elif k == "key_tap":
            lines.append(f"tap {op['key']} {op.get('ms', '')}".strip())
        elif k == "mouse_down":
            lines.append(f"press {op['button']}")
        elif k == "mouse_up":
            lines.append(f"release {op['button']}")
        elif k == "mouse_click":
            lines.append(f"click {op['button']} {op.get('ms','')}".strip())
        elif k == "move":
            lines.append(f"move {op['dx']} {op['dy']}")
        elif k == "scroll":
            lines.append(f"scroll {op['amount']}")
        elif k == "sleep":
            lines.append(f"sleep {op['ms']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="Output macro path (.json or .mcs).")
    ap.add_argument("--stop", default="f8", help="Key that ends recording.")
    ap.add_argument("--moves", action="store_true",
                    help="Also record mouse-move deltas (menu macros).")
    ap.add_argument("--no-mc-only", action="store_true",
                    help="Record input even when Minecraft isn't focused.")
    ap.add_argument("--countdown", type=int, default=3,
                    help="Seconds before recording starts.")
    ap.add_argument("--min-sleep-ms", type=int, default=10,
                    help="Drop sleeps shorter than this (jitter filter).")
    ap.add_argument("--name", default=None, help="Macro name (default: file stem).")
    args = ap.parse_args(argv)

    out = Path(args.out)
    name = args.name or out.stem

    rec = _Recorder(stop_key=args.stop, record_moves=args.moves,
                    mc_only=not args.no_mc_only, min_sleep_ms=args.min_sleep_ms)

    for n in range(max(0, args.countdown), 0, -1):
        print(f"[record] starting in {n}…")
        time.sleep(1.0)
    print(f"[record] RECORDING — perform the technique, then press {args.stop.upper()} to stop.")

    kl = _kb.Listener(on_press=rec.on_press, on_release=rec.on_release)
    mlst = _ms.Listener(on_click=rec.on_click, on_scroll=rec.on_scroll,
                        on_move=rec.on_move)
    kl.start()
    mlst.start()
    try:
        kl.join()           # blocks until the stop key returns False
    except KeyboardInterrupt:
        rec._stop = True
    finally:
        mlst.stop()
        kl.stop()

    ops = rec.to_ops()
    if not ops:
        print("[record] no input captured (was Minecraft focused?). Nothing written.")
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in (".mcs", ".macro"):
        out.write_text(_ops_to_mcs(name, ops), encoding="utf-8")
    else:
        out.write_text(json.dumps({"name": name, "ops": ops}, indent=2),
                       encoding="utf-8")
    n_sleep = sum(1 for o in ops if o["op"] == "sleep")
    print(f"[record] wrote {len(ops)} ops ({len(ops)-n_sleep} actions) -> {out}")
    print(f"[record] replay with:  python main.py --script {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
