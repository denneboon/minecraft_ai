#!/usr/bin/env python3
"""
Wear the BEST armour the bot is carrying.

    python tools/equip_armor.py

Equips the best helmet / chestplate / leggings / boots from the inventory into
the armour slots (by protection tier: netherite > diamond > iron >
chainmail/turtle > gold > leather). Only upgrades — never swaps in a worse or
equal piece. Needs MC running + focused. Panic: Ctrl+Shift+F12.
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

import main as M
from utils.focus import _find_minecraft_hwnd, activate_minecraft
from control.input_gate import InputGate
from vision.capture import Capture, CaptureConfig
from vision.inventory import build_inventory_reader
from vision.tooltip import build_tooltip_reader
from agents.inventory_inspector import InventoryInspector, InspectorConfig
from control.hotbar import build_hotbar_manager
from control.inventory_control import InventoryController
from agents.inventory_memory import InventoryMemory
from agents.armor_equip import equip_best_armor
from knowledge.catalog import Catalog
from vision.mc_assets import MCAssets


def main(argv=None) -> int:
    wins = _find_minecraft_hwnd()
    if not wins:
        print("[armor] Minecraft not found"); return 2
    hwnd = wins[0][0]; activate_minecraft(); time.sleep(0.5)

    settings = M._load_yaml(M.SETTINGS_PATH)
    keymap = M._flatten_keymap_for_keyboard(M._load_json(M.KEYMAP_PATH))
    ui_scale = int((settings.get("capture") or {}).get("ui_scale", 2))

    gate = InputGate()
    safety = M.build_safety(settings, gate=gate)
    mouse = M.build_mouse(settings, gate=gate)
    kb = M.build_keyboard(settings, keymap, gate=gate)
    capture = Capture(CaptureConfig(hwnd=hwnd, use_client_area=True, threaded=True))
    safety.start(); mouse.start(); kb.start(); capture.start(); time.sleep(0.3)

    menu_detector = M.build_menu_detector_default(settings)
    a = MCAssets.load(); cat = Catalog.load(a)
    reader = build_inventory_reader(settings, assets=a)
    hotbar = build_hotbar_manager(settings, catalog=cat)
    tooltip = build_tooltip_reader(settings, assets=a)
    try:
        origin = capture.window_origin()
    except Exception:
        origin = (0, 0)
    inspector = InventoryInspector(
        tooltip, capture, mouse=mouse, gate=gate,
        sample_store=getattr(reader, "sample_store", None),
        config=InspectorConfig(max_resolutions_per_call=24))
    memory = InventoryMemory()
    ctl = InventoryController(mouse, kb, reader, hotbar, capture,
                              ui_scale=ui_scale, window_origin=origin,
                              inspector=inspector, memory=memory)

    try:
        ctrl_ok, reason = M.ensure_controllable(capture, menu_detector, kb, gate)
        if not ctrl_ok:
            M.bot_cannot_start_banner(reason); return 1
        eq = equip_best_armor(ctl, catalog=cat, log=print)
        if eq:
            print("[armor] equipped: " + ", ".join(
                f"{s}={i.split(':')[-1]}" for s, i in eq.items()))
        else:
            print("[armor] nothing to upgrade (already wearing the best, or none carried)")
        return 0
    finally:
        try:
            if hasattr(mouse, "release_all"):
                mouse.release_all()
        except Exception:
            pass
        try:
            kb.stop()
        except Exception:
            pass
        capture.stop()
        try:
            safety.stop()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
