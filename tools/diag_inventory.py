#!/usr/bin/env python3
"""
Diagnostic: open the inventory, STATIC read (no hover), and dump every
non-empty slot with full recognition detail — item, source, confidence,
score, runner-up — to see exactly how confusable items (coal, mushroom,
log-vs-table) get (mis)labelled. Foundation for hardening the recogniser.

    python tools/diag_inventory.py

Needs MC running + focused.
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import main as M
from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

from utils.focus import _find_minecraft_hwnd, activate_minecraft
from control.input_gate import InputGate
from vision.capture import Capture, CaptureConfig
from vision.inventory import build_inventory_reader
from vision.mc_assets import MCAssets


def main(argv=None) -> int:
    wins = _find_minecraft_hwnd()
    if not wins:
        print("[diag] Minecraft not found"); return 2
    hwnd = wins[0][0]; activate_minecraft(); time.sleep(0.4)
    settings = M._load_yaml(M.SETTINGS_PATH)
    keymap = M._flatten_keymap_for_keyboard(M._load_json(M.KEYMAP_PATH))
    gate = InputGate()
    safety = M.build_safety(settings, gate=gate)
    kb = M.build_keyboard(settings, keymap, gate=gate)
    capture = Capture(CaptureConfig(hwnd=hwnd, use_client_area=True, threaded=True))
    safety.start(); kb.start(); capture.start(); time.sleep(0.3)
    reader = build_inventory_reader(settings, assets=MCAssets.load())
    menu_detector = M.build_menu_detector_default(settings)
    try:
        if not M.ensure_playing(capture, menu_detector, kb):
            print("[diag] paused — click into MC"); return 1
        kb.tap("e"); time.sleep(0.4)
        frame = capture.get_frame()
        snap = reader.read(frame, container="player_inventory")
        kb.tap("e"); time.sleep(0.2)
        print(f"[diag] frame {frame.shape}; non-empty slots (static read):")
        print(f"    {'slot':12} {'item':18} {'src':11} {'conf':>5} {'score':>7} 2nd")
        for name, sc in sorted((snap.slots or {}).items()):
            it = getattr(sc, "item", None)
            if it is None and getattr(sc, "source", "") in ("placeholder", "empty"):
                continue
            if it is None and getattr(sc, "is_empty", True):
                continue
            print(f"    {name:12} {str(it).split(':')[-1] if it else '(unknown)':18} "
                  f"{getattr(sc,'source',''):11} {getattr(sc,'confidence',0):>5.2f} "
                  f"{getattr(sc,'score',0):>7.1f} "
                  f"{str(getattr(sc,'second',None)).split(':')[-1]}")
        return 0
    finally:
        try: kb.stop()
        except Exception: pass
        capture.stop()
        try: safety.stop()
        except Exception: pass


if __name__ == "__main__":
    raise SystemExit(main())
