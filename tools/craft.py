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
from vision.tooltip import build_tooltip_reader
from agents.inventory_inspector import InventoryInspector, InspectorConfig
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
    capture = Capture(CaptureConfig(hwnd=hwnd, use_client_area=True, threaded=True))
    safety.start(); mouse.start(); kb.start(); capture.start()
    time.sleep(0.3)

    a = MCAssets.load()
    cat = Catalog.load(a)
    reader = build_inventory_reader(settings, assets=a)
    hotbar = build_hotbar_manager(settings, catalog=cat)
    # Hover-to-learn: identify items the static recogniser is unsure about
    # by hovering the slot + OCRing the tooltip id.
    tooltip = build_tooltip_reader(settings, assets=a)
    # sample_store=reader.sample_store: each hover TEACHES the recogniser, so
    # learned items are read statically next time (no more hovering them).
    inspector = InventoryInspector(
        tooltip, capture, mouse=mouse, gate=gate,
        sample_store=getattr(reader, "sample_store", None),
        config=InspectorConfig(max_resolutions_per_call=20))
    try:
        origin = capture.window_origin()
    except Exception:
        origin = (0, 0)
    print(f"[craft] window origin (desktop top-left): {origin}")
    ctl = InventoryController(mouse, kb, reader, hotbar, capture,
                              ui_scale=ui_scale, window_origin=origin,
                              inspector=inspector)
    crafter = Crafter(ctl, a, cat)

    try:
        print(f"[craft] opening inventory (gui_scale={ui_scale}) — target {target}")
        ctl.open_inventory()
        if "--debug" in argv:
            try:
                import cv2
                dbg = capture.get_frame()
                cv2.imwrite(os.path.join(ROOT, "data", "_craft_debug.png"),
                            dbg[:, :, ::-1])
                print(f"[craft] frame {dbg.shape} saved to data/_craft_debug.png")
            except Exception as e:
                print(f"[craft] frame dump failed: {e}")
        # Crafter does a SURGICAL read internally (hovers only until the
        # recipe is plannable). We don't pre-read here, which would hover the
        # whole inventory and defeat the point.
        _static = lambda s: True                  # stop_when=True -> no hovers
        if "--debug" in argv:
            snap = ctl.read(stop_when=_static)     # static-only peek for debug
            for nm, sc in sorted((getattr(snap, "slots", {}) or {}).items()):
                if getattr(sc, "item", None):
                    print(f"    {nm:12} {sc.item.split(':')[-1]:18} "
                          f"conf={getattr(sc,'confidence',0):.2f}")
        okc, msg = crafter.craft(target)
        print(f"[craft] {'OK' if okc else 'FAIL'}: {msg}")
        time.sleep(0.4)
        after = inventory_counts(ctl.read(stop_when=_static))   # no-hover check
        print(f"[craft] now have {target.split(':')[-1]}={after.get(target, 0)}")
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
