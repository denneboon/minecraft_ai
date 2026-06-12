#!/usr/bin/env python3
"""
Demonstrate reading the GAMEPLAY HUD hotbar (inventory CLOSED) — the bot
learns its hotbar counts WITHOUT opening anything, by relabelling the HUD
slots from the known layout (one prior open) + the reliable count OCR.

    python tools/read_hotbar.py

Flow: open the inventory ONCE (seed the per-slot layout), close it, then read
the HUD live and print item + live count per slot. Needs MC running + focused.
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
from control.hotbar import build_hotbar_manager
from control.inventory_control import InventoryController
from agents.inventory_memory import InventoryMemory
from knowledge.catalog import Catalog
from vision.mc_assets import MCAssets


def main(argv=None) -> int:
    wins = _find_minecraft_hwnd()
    if not wins:
        print("[hotbar] Minecraft not found"); return 2
    hwnd = wins[0][0]; activate_minecraft(); time.sleep(0.4)

    settings = M._load_yaml(M.SETTINGS_PATH)
    keymap = M._flatten_keymap_for_keyboard(M._load_json(M.KEYMAP_PATH))
    ui_scale = int((settings.get("capture") or {}).get("ui_scale", 2))
    gate = InputGate()
    safety = M.build_safety(settings, gate=gate)
    mouse = M.build_mouse(settings, gate=gate)
    kb = M.build_keyboard(settings, keymap, gate=gate)
    capture = Capture(CaptureConfig(hwnd=hwnd, use_client_area=True, threaded=True))
    safety.start(); mouse.start(); kb.start(); capture.start(); time.sleep(0.3)

    a = MCAssets.load(); cat = Catalog.load(a)
    reader = build_inventory_reader(settings, assets=a)
    hotbar = build_hotbar_manager(settings, catalog=cat)
    menu_detector = M.build_menu_detector_default(settings)
    memory = InventoryMemory()
    try:
        origin = capture.window_origin()
    except Exception:
        origin = (0, 0)
    ctl = InventoryController(mouse, kb, reader, hotbar, capture,
                              ui_scale=ui_scale, window_origin=origin, memory=memory)
    try:
        if not M.ensure_playing(capture, menu_detector, kb):
            print("[hotbar] game paused — click into MC"); return 1
        # 1. Seed the per-slot layout with ONE open + read.
        print("[hotbar] seeding hotbar layout (one open)…")
        ctl.open_inventory(); ctl.read(stop_when=lambda s: True); ctl.close()
        time.sleep(0.4)
        print("[hotbar] hotbar layout from the read:")
        for i in range(9):
            it = hotbar.item_in_slot(i + 1)
            print(f"    slot {i+1}: {it.split(':')[-1] if it else '(empty)'}")
        # 2. Now read the HUD live (inventory CLOSED) — counts without opening.
        print("[hotbar] reading HUD live (inventory CLOSED):")
        snap = ctl.read_hotbar()
        for i in range(9):
            sc = snap.slots.get(f"hotbar_{i}")
            it = getattr(sc, "item", None)
            if it:
                print(f"    slot {i+1}: {it.split(':')[-1]:16} x{getattr(sc,'count',0)}")
            else:
                print(f"    slot {i+1}: (empty)")
        print(f"[hotbar] ledger hotbar totals: "
              f"{ {k.split(':')[-1]: v for k, v in memory._hotbar.items()} }")
        return 0
    finally:
        try: kb.stop()
        except Exception: pass
        capture.stop()
        try: safety.stop()
        except Exception: pass


if __name__ == "__main__":
    raise SystemExit(main())
