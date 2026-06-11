#!/usr/bin/env python3
"""
Live crafting test — open the inventory and craft an item with the
hotkey-first controller (number-key swaps, no drag).

    python tools/craft.py oak_planks
    python tools/craft.py crafting_table

Needs Minecraft running + focused. Camera is untouched; this only operates
the inventory screen. Panic: Ctrl+Shift+F12 (the usual emergency stop).
"""
from __future__ import annotations

import sys
import time

import main as M
from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

from utils.focus import _find_minecraft_hwnd, activate_minecraft
from control.input_gate import InputGate
from vision.capture import Capture, CaptureConfig
from vision.inventory import build_inventory_reader
from control.hotbar import build_hotbar_manager
from control.inventory_control import InventoryController
from agents.crafting import Crafter, inventory_counts
from knowledge.catalog import Catalog
from vision.mc_assets import MCAssets


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    target = (argv[0] if argv else "oak_planks")
    if ":" not in target:
        target = "minecraft:" + target

    wins = _find_minecraft_hwnd()
    if not wins:
        print("[craft] Minecraft window not found — is it running?")
        return 2
    hwnd = wins[0][0]
    activate_minecraft(); time.sleep(0.5)

    settings = M._load_yaml(M.SETTINGS_PATH)
    keymap_flat = M._flatten_keymap_for_keyboard(M._load_json(M.KEYMAP_PATH))
    # Same source the inventory reader uses, so slot rects line up.
    ui_scale = int(((settings.get("capture") or {}).get("ui_scale", 2)))

    gate = InputGate()
    safety = M.build_safety(settings, gate=gate)
    mouse = M.build_mouse(settings, gate=gate)
    kb = M.build_keyboard(settings, keymap_flat, gate=gate)
    capture = Capture(CaptureConfig(hwnd=hwnd, threaded=True))
    safety.start(); mouse.start(); kb.start(); capture.start()
    time.sleep(0.3)

    a = MCAssets.load()
    cat = Catalog.load(a)
    reader = build_inventory_reader(settings)
    hotbar = build_hotbar_manager(settings, catalog=cat)
    ctl = InventoryController(mouse, kb, reader, hotbar, capture, ui_scale=ui_scale)
    crafter = Crafter(ctl, a, cat)

    try:
        print(f"[craft] opening inventory (gui_scale={ui_scale}) — target {target}")
        ctl.open_inventory()
        snap = ctl.read()
        have = inventory_counts(snap)
        print(f"[craft] inventory: "
              + ", ".join(f"{k.split(':')[-1]}={v}" for k, v in sorted(have.items())[:12]))
        okc, msg = crafter.craft(target, snap=snap)
        print(f"[craft] {'OK' if okc else 'FAIL'}: {msg}")
        time.sleep(0.4)
        # show the result
        after = inventory_counts(ctl.read())
        made = after.get(target, 0)
        print(f"[craft] now have {target.split(':')[-1]}={made}")
        ctl.close()
        return 0 if okc else 1
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
