#!/usr/bin/env python3
"""
Run (or inspect) a Minecraft macro/script file.

Two modes:

* ``--dry-run`` — parse the file and print the normalised op tree. No
  Minecraft needed. Use this to validate a macro before letting it drive
  real input.
* live (default) — focus Minecraft and play the script through the gated
  keyboard + mouse, with focus-loss auto-stop and the Ctrl+Shift+F12
  emergency hotkey active (same Safety controller the agents use).

Examples
--------
    python tools/run_script.py scripts/macros/bridge.ahk --dry-run
    python tools/run_script.py scripts/macros/bridge.mcs
    python tools/run_script.py my_macro.txt --max-seconds 60
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.script_runner import (
    ScriptRunner, ScriptRunnerConfig, load_script,
)


def _print_ops(ops, indent=2):
    pad = " " * indent
    for op in ops:
        kind = op.get("op")
        if kind == "loop":
            n = op.get("count", 1)
            label = "forever" if n <= 0 else f"x{n}"
            print(f"{pad}loop {label}:")
            _print_ops(op.get("ops", []), indent + 4)
        else:
            extra = {k: v for k, v in op.items() if k != "op"}
            print(f"{pad}{kind} {extra}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("script", help="Path to a macro file (.ahk/.txt/.json/.mcs).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse + print the ops; do not touch input or MC.")
    ap.add_argument("--max-seconds", type=float, default=120.0,
                    help="Wall-clock safety cap on playback.")
    args = ap.parse_args(argv)

    try:
        name, ops = load_script(args.script)
    except Exception as e:
        print(f"[run_script][ERROR] could not load {args.script!r}: {e}")
        return 2

    if args.dry_run:
        print(f"[run_script] {name!r} parsed from {args.script}:")
        _print_ops(ops)
        return 0

    import main as M
    from utils.focus import _find_minecraft_hwnd, activate_minecraft
    from control.input_gate import InputGate

    if not _find_minecraft_hwnd():
        print("[run_script][ERROR] Minecraft (javaw.exe) is not running.")
        return 2

    settings = M._load_yaml(M.SETTINGS_PATH)
    keymap_flat = M._flatten_keymap_for_keyboard(M._load_json(M.KEYMAP_PATH))

    gate = InputGate()
    safety = M.build_safety(settings, gate=gate)
    keyboard = M.build_keyboard(settings, keymap_flat, gate=gate)
    mouse = M.build_mouse(settings, gate=gate)

    runner = ScriptRunner(keyboard, mouse, gate=gate,
                          config=ScriptRunnerConfig(max_runtime_sec=args.max_seconds))

    def emergency():
        for sub in (keyboard, mouse):
            try:
                sub.emergency_stop()
            except Exception:
                pass
            try:
                sub.stop()
            except Exception:
                pass
    safety.set_emergency_callback(emergency)

    activate_minecraft()
    safety.start()
    keyboard.start()
    mouse.start()
    time.sleep(0.3)
    try:
        if not safety.allow_input():
            print("[run_script][WARN] Minecraft not focused — gate closed. "
                  "Focus the window; aborting.")
            return 1
        runner.run(ops, name=name)
    finally:
        for sub in (keyboard, mouse):
            try:
                sub.stop()
            except Exception:
                pass
        safety.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
