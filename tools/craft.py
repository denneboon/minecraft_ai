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
from agents.crafting import Crafter
from agents.inventory_memory import InventoryMemory
from knowledge.catalog import Catalog
from vision.mc_assets import MCAssets


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    pos = [x for x in argv if not x.startswith("-")]
    target = (pos[0] if pos else "oak_planks")
    if ":" not in target:
        target = "minecraft:" + target
    # Optional AMOUNT: "craft oak_planks 8" -> ensure we hold >= 8, crafting
    # only the shortfall (and not even opening if we already have enough).
    want = 1
    if len(pos) > 1:
        try:
            want = max(1, int(pos[1]))
        except ValueError:
            want = 1

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
    # Shared inventory ledger: every read updates it, so the bot can tell it
    # already has enough WITHOUT re-opening (and won't craft a pile for no
    # reason). Lives in the controller so all reads here feed it.
    memory = InventoryMemory()
    ctl = InventoryController(mouse, kb, reader, hotbar, capture,
                              ui_scale=ui_scale, window_origin=origin,
                              inspector=inspector, memory=memory)
    crafter = Crafter(ctl, a, cat)
    menu_detector = M.build_menu_detector_default(settings)

    try:
        # Resume if the game is paused (singleplayer pauses on focus loss);
        # otherwise every inventory read/action would hit a frozen frame.
        if not M.ensure_playing(capture, menu_detector, kb):
            print("[craft] game is paused and won't resume — click into MC"); return 1
        short = target.split(":")[-1]
        print(f"[craft] ensure >={want} {short} (gui_scale={ui_scale})")
        # Count-aware: skip entirely if we already hold enough, else open and
        # craft ONLY the shortfall. ensure manages the inventory screen.
        okc, msg = crafter.ensure(target, want, memory=memory)
        print(f"[craft] {'OK' if okc else 'FAIL'}: {msg}")
        print(f"[craft] now have {short}={memory.count(target)} (wanted {want})")
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
