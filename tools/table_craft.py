#!/usr/bin/env python3
"""
Live crafting-table flow: PLACE a crafting table from the hotbar, OPEN it,
craft a 3x3 recipe in it, then BREAK the table back so nothing's left behind.

    python tools/table_craft.py wooden_pickaxe
    python tools/table_craft.py stick            # (works in 2x2 too, but fine)

Needs Minecraft running + focused. Uses the camera (place/break) + the
inventory screen (craft). Panic: Ctrl+Shift+F12.
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
from control.action_wrapper import ActionWrapper
from vision.capture import Capture, CaptureConfig
from vision.ocr import build_f3_reader
from vision.world import build_world_perception
from vision.inventory import build_inventory_reader
from vision.tooltip import build_tooltip_reader
from agents.inventory_inspector import InventoryInspector, InspectorConfig
from control.hotbar import build_hotbar_manager
from control.inventory_control import InventoryController
from agents.crafting import Crafter, find_item_slot
from agents.skills import (PlaceBlock, BreakLookedAt, LookAtVoxel,
                           SkillContext, SkillStatus)
from vision.world.f3_target import targeted_block_pos
from agents.treechop import _full_movement
from knowledge.catalog import Catalog
from vision.mc_assets import MCAssets

TABLE = "minecraft:crafting_table"


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    target = (argv[0] if argv and not argv[0].startswith("-") else "wooden_pickaxe")
    if ":" not in target:
        target = "minecraft:" + target
    debug = "--debug" in argv

    wins = _find_minecraft_hwnd()
    if not wins:
        print("[table] Minecraft not found"); return 2
    hwnd = wins[0][0]; activate_minecraft(); time.sleep(0.5)

    settings = M._load_yaml(M.SETTINGS_PATH)
    keymap_flat = M._flatten_keymap_for_keyboard(M._load_json(M.KEYMAP_PATH))
    ui_scale = int((settings.get("capture") or {}).get("ui_scale", 2))
    px_per_deg = float((settings.get("agent") or {}).get("mouse_per_degree", 6.5) or 6.5)

    gate = InputGate()
    safety = M.build_safety(settings, gate=gate)
    mouse = M.build_mouse(settings, gate=gate)
    kb = M.build_keyboard(settings, keymap_flat, gate=gate)
    capture = Capture(CaptureConfig(hwnd=hwnd, use_client_area=True, threaded=True))
    safety.start(); mouse.start(); kb.start(); capture.start(); time.sleep(0.3)

    f3 = build_f3_reader(settings)
    wp = build_world_perception(settings)
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
    actions = ActionWrapper(kb, mouse, gate=gate)

    def _dispatch(action):
        if action.hotbar:
            kb.tap(str(action.hotbar))
        if action.look_dx or action.look_dy:
            try:
                # track_target = relative camera motion MC's raw input sees;
                # mouse.move() is absolute and invisible to the game.
                mouse.track_target(int(action.look_dx), int(action.look_dy))
            except Exception:
                pass
        if action.interact == "use_item":
            try:
                mouse.right_click()
            except Exception:
                pass
        actions.set_attack(action.interact == "attack")
        try:
            actions.set_movement_state(**_full_movement(action.movement))
        except Exception:
            pass

    def _stop():
        actions.set_attack(False)
        try:
            actions.release_all_movement()
        except Exception:
            pass

    def drive(skill, label, max_secs=14.0, debug=False):
        t0 = time.time(); n = 0
        while time.time() - t0 < max_secs:
            frame = capture.get_frame()
            # Don't act on a frozen, paused frame — resume first.
            if menu_detector is not None and menu_detector.is_pause_menu(frame):
                M.ensure_playing(capture, menu_detector, kb)
                time.sleep(0.2); continue
            reading = f3.read(frame)
            wf = wp.update(frame, reading)
            pose = wf.pose
            # Raw targeted-block coords (survive an unreadable block id — a
            # freshly-placed table OCRs to '?' but its coords stay clean), so
            # PlaceBlock can confirm a placement by position.
            try:
                tpos = targeted_block_pos(getattr(reading, "raw_text", "").splitlines())
            except Exception:
                tpos = None
            ctx = SkillContext(pose=pose, world_map=wp.world_map,
                               looking_at=wf.looking_at, targeted_pos=tpos,
                               hotbar=hotbar, px_per_deg=px_per_deg,
                               dimension=getattr(pose, "dimension", None) if pose else None)
            r = skill.tick(ctx)
            _dispatch(r.action)
            n += 1
            if debug and n % 8 == 0:
                la = wf.looking_at
                print(f"[table]  .{label} t={n} pitch={getattr(pose,'pitch',None)} "
                      f"look=({r.action.look_dx},{r.action.look_dy}) "
                      f"at={getattr(la,'block_id',None)} face={getattr(la,'face',None)} "
                      f"| {r.status.value}: {r.info}")
            if r.status in (SkillStatus.DONE, SkillStatus.FAILED, SkillStatus.BLOCKED):
                _stop()
                print(f"[table] {label}: {r.status.value} ({r.info})")
                return r.status
            time.sleep(0.05)
        _stop()
        print(f"[table] {label}: TIMEOUT")
        return SkillStatus.FAILED

    try:
        # 0. Make sure the game isn't paused (singleplayer pauses on focus
        # loss; a paused frame is frozen and every read/action fails).
        if not M.ensure_playing(capture, menu_detector, kb):
            print("[table] game is paused and won't resume — click into MC"); return 1

        # 1. Find the crafting table in the hotbar (must be there to place it).
        ctl = InventoryController(mouse, kb, reader, hotbar, capture,
                                  ui_scale=ui_scale, window_origin=origin,
                                  inspector=inspector)
        ctl.open_inventory()
        snap = ctl.read(stop_when=lambda s: find_item_slot(s, TABLE) is not None)
        tname = find_item_slot(snap, TABLE)
        if tname is None:
            ctl.close()
            print("[table] no crafting_table in inventory — craft one first "
                  "(python tools/craft.py crafting_table)"); return 1
        # The table must be in the HOTBAR to place it. If it's in the main
        # inventory (e.g. just crafted), number-key swap it into a hotbar slot.
        if not tname.startswith("hotbar_"):
            table_slot = ctl.to_hotbar(tname, snap)
            print(f"[table] moved crafting_table {tname} -> hotbar slot {table_slot}")
        else:
            table_slot = int(tname.split("_")[1]) + 1
            print(f"[table] crafting_table in hotbar slot {table_slot}")
        ctl.close(); time.sleep(0.3)

        def _looking():
            fr = capture.get_frame()
            return wp.update(fr, f3.read(fr)).looking_at

        # 2. Place the table. PlaceBlock now SCANS look directions for a
        # placeable surface, places, and self-VERIFIES the block appeared under
        # the crosshair (retrying other views if MC rejected it), so one drive
        # call is enough — DONE means it's confirmed on the ground.
        pb = PlaceBlock(slot=table_slot)
        if drive(pb, "place", max_secs=30.0, debug=debug) != SkillStatus.DONE \
                or pb.placed_at is None:
            print("[table] couldn't place the table"); return 1
        table_pos = pb.placed_at
        print(f"[table] table placed + confirmed at {table_pos}")

        # 3. Open it (crosshair is on it).
        print("[table] opening the table")
        mouse.right_click(); time.sleep(0.7)

        # 4. Craft the 3x3 recipe in the open table.
        tctl = InventoryController(mouse, kb, reader, hotbar, capture,
                                   ui_scale=ui_scale, window_origin=origin,
                                   inspector=inspector, container="crafting_table")
        okc, msg = Crafter(tctl, a, cat).craft(target)
        print(f"[table] craft: {'OK' if okc else 'FAIL'}: {msg}")
        tctl.close(); time.sleep(0.5)

        # 5. Break the table back. After the screen closes the crosshair is
        # still on it; if not, BreakLookedAt safely refuses anything else.
        drive(BreakLookedAt(avoid=lambda b: "crafting_table" not in str(b)),
              "break", max_secs=12.0)
        return 0 if okc else 1
    finally:
        _stop()
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
