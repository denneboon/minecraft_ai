#!/usr/bin/env python3
"""
Live crafting-table flow: PLACE a crafting table from the hotbar, OPEN it,
craft a 3x3 recipe in it, then BREAK the table back so nothing's left behind.

    python tools/table_craft.py wooden_pickaxe
    python tools/table_craft.py stick            # (works in 2x2 too, but fine)

Needs Minecraft running + focused. Uses the camera (place/break) + the
inventory screen (craft). Panic: Ctrl+Shift+F12.

The flow is exposed as ``run_table_craft(target, *, <components>)`` so other
orchestrators (e.g. tools/make.py) can reuse it with their own shared
components; ``main`` just builds the components and calls it.
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
                           SkillContext, SkillStatus, norm_angle)
from vision.world.f3_target import targeted_block_pos
from agents.treechop import _full_movement
from knowledge.catalog import Catalog
from vision.mc_assets import MCAssets

TABLE = "minecraft:crafting_table"


def run_table_craft(target, *, capture, mouse, kb, f3, wp, menu_detector,
                    reader, hotbar, inspector, actions, gate, cat, assets,
                    ui_scale=2, origin=(0, 0), px_per_deg=6.5, memory=None,
                    debug=False, f3_worker=None):
    """Place a crafting table, open it, craft ``target`` (3x3), break the table
    back. Components are provided + owned by the CALLER (not started/stopped
    here). Returns ``(ok, message)``."""
    a = assets
    if ":" not in target:
        target = "minecraft:" + target

    def _dispatch(action, apply_look=True):
        if action.hotbar:
            kb.tap(str(action.hotbar))
        if apply_look and (action.look_dx or action.look_dy):
            try:
                # INSTANT relative move (like the chop FSM in make.py), applied
                # ONLY on a FRESH pose. The eased track_target applied every
                # ~20Hz tick re-issued camera moves on stale poses between the
                # ~11Hz F3 updates and over-rotated -> the place-aim oscillated
                # back and forth and never settled.
                mouse.move(int(action.look_dx), int(action.look_dy))
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

    # Set by drive() when a GUI opens during the PLACE step: that means the
    # table got placed and the bot's click opened it — proof of placement.
    place_gui = {"open": False, "pos": None}

    def drive(skill, label, max_secs=14.0, debug=False):
        t0 = time.time(); n = 0; _last_pause = time.time()
        _last_cam = None; _frozen = 0      # frozen-camera (opened-GUI) detector
        _last_tpos = None                  # last targeted block (the table we hit)
        _last_ts = None                    # pose freshness (gate camera on fresh)
        while time.time() - t0 < max_secs:
            n += 1
            now = time.time()
            frame = capture.get_frame()
            # is_pause_menu is a ~2.4s full-frame OCR; TIME-throttle it (the
            # preflight already put us in gameplay and a LAN world doesn't pause
            # on focus loss) so it never crawls the loop.
            if menu_detector is not None and now - _last_pause > 4.0:
                _last_pause = now
                if menu_detector.is_pause_menu(frame):
                    M.ensure_playing(capture, menu_detector, kb)
                    time.sleep(0.2); continue
            # Prefer the background OCR worker (O(1) latest pose) over a ~86 ms
            # inline read; fall back to inline when no worker was supplied.
            reading = (f3_worker.latest() if f3_worker is not None
                       else f3.read(frame))
            wf = wp.update(frame, reading)
            pose = wf.pose
            # Raw targeted-block coords (survive an unreadable block id — a
            # freshly-placed table OCRs to '?' but its coords stay clean), so
            # PlaceBlock can confirm a placement by position.
            try:
                tpos = targeted_block_pos(getattr(reading, "raw_text", "").splitlines())
            except Exception:
                tpos = None
            if tpos is not None:
                _last_tpos = tpos
            ctx = SkillContext(pose=pose, world_map=wp.world_map,
                               looking_at=wf.looking_at, targeted_pos=tpos,
                               hotbar=hotbar, px_per_deg=px_per_deg,
                               dimension=getattr(pose, "dimension", None) if pose else None)
            # Only apply camera moves on a FRESH pose (a new F3 read), so the
            # aimer's per-update correction lands once and converges instead of
            # being re-applied on stale poses and overshooting.
            ts = getattr(reading, "timestamp", None) if reading is not None else None
            fresh = ts is not None and ts != _last_ts
            _last_ts = ts
            r = skill.tick(ctx)
            _dispatch(r.action, apply_look=fresh)
            # Frozen-camera guard: if we keep commanding a turn but the view
            # won't move, SOMETHING froze the camera. It could be (a) a crafting
            # GUI we opened by clicking an existing/just-placed table, (b) the
            # PAUSE menu, or (c) a genuinely stuck aim (terrain/yaw limit/stale
            # pose). We MUST tell these apart: the old code assumed "frozen during
            # place == table placed+open", which mistook the PAUSE menu for a
            # placed table (and blind-escaping with nothing open OPENS the pause
            # menu). Classify with the menu detector before acting.
            if pose is not None and (r.action.look_dx or r.action.look_dy):
                cam = (round(float(getattr(pose, "yaw", 0.0)), 1),
                       round(float(getattr(pose, "pitch", 0.0)), 1))
                if cam == _last_cam:
                    _frozen += 1
                    if _frozen >= 8:
                        _frozen = 0
                        det = (menu_detector.detect(frame)
                               if menu_detector is not None else None)
                        is_pause = bool(det) and det.menu == "pause"
                        gui_open = bool(det) and det.open and not is_pause
                        if is_pause:
                            # Stray pause (an earlier mis-aimed escape, or a focus
                            # blip). RESUME — it is NOT a placed table.
                            if debug:
                                print(f"[table]  .{label}: PAUSE menu detected -> "
                                      f"resume (not a placed table)")
                            M.ensure_playing(capture, menu_detector, kb)
                            time.sleep(0.2)
                        elif gui_open and label == "place":
                            # A real crafting GUI is open during placement — the
                            # click landed on the (just-placed or pre-existing)
                            # table and opened it. THAT is the success signal
                            # (a fresh table's id OCRs to garble so PlaceBlock's
                            # visual verify often can't read it).
                            place_gui["open"] = True
                            place_gui["pos"] = _last_tpos
                            if debug:
                                print(f"[table]  .place: crafting GUI open -> "
                                      f"table placed + open at {_last_tpos}")
                            _stop()
                            return SkillStatus.DONE
                        elif gui_open:
                            # A GUI is genuinely open and we're NOT placing ->
                            # safe to escape it (we confirmed one is open).
                            if debug:
                                print(f"[table]  .{label}: GUI open — closing (escape)")
                            kb.tap("escape"); time.sleep(0.35)
                        elif debug:
                            # No menu open: the aim is stuck (terrain/limit/stale
                            # pose). Do NOT escape — that would OPEN the pause
                            # menu. Let the skill keep retrying / time out.
                            print(f"[table]  .{label}: camera stuck, no GUI — "
                                  f"NOT escaping (would open pause)")
                else:
                    _frozen = 0
                _last_cam = cam
            if debug and n % 5 == 0:
                la = wf.looking_at
                print(f"[table]  .{label} t={n} "
                      f"yaw={getattr(pose,'yaw',None)} pitch={getattr(pose,'pitch',None)} "
                      f"look=({r.action.look_dx},{r.action.look_dy}) "
                      f"at={getattr(la,'block_id',None)} tpos={tpos} "
                      f"| {r.status.value}: {r.info}")
            if r.status in (SkillStatus.DONE, SkillStatus.FAILED, SkillStatus.BLOCKED):
                _stop()
                if debug:
                    print(f"[table] {label}: {r.status.value} ({r.info})")
                return r.status
            time.sleep(0.05)
        _stop()
        print(f"[table] {label}: TIMEOUT")
        return SkillStatus.FAILED

    try:
        # 0. Make sure the game isn't paused (a paused frame is frozen).
        if not M.ensure_playing(capture, menu_detector, kb):
            return False, "game is paused and won't resume"

        # 1. Find the crafting table (must be in the HOTBAR to place it).
        ctl = InventoryController(mouse, kb, reader, hotbar, capture,
                                  ui_scale=ui_scale, window_origin=origin,
                                  inspector=inspector, memory=memory)
        ctl.open_inventory()
        _has_table = lambda s: find_item_slot(s, TABLE) is not None
        # The static hotbar read is flaky, so retry; the fallback hover-scan
        # lifts the gate AND hovers "empty" slots (the iso table icon
        # false-empties), rescuing + teaching the table.
        snap = None; tname = None
        for attempt in range(3):
            snap = ctl.read(stop_when=_has_table)
            tname = find_item_slot(snap, TABLE)
            if tname is not None:
                break
            print(f"[table] table not found (attempt {attempt+1}/3) — hover-scanning")
            prev_gate = inspector.cfg.skip_above_confidence
            inspector.cfg.skip_above_confidence = 1.01
            try:
                snap = ctl.read(stop_when=_has_table, include_empty=True)
            finally:
                inspector.cfg.skip_above_confidence = prev_gate
            tname = find_item_slot(snap, TABLE)
            if tname is not None:
                break
            time.sleep(0.4)
        if tname is None:
            ctl.close()
            return False, "no crafting_table in inventory"
        if not tname.startswith("hotbar_"):
            table_slot = ctl.to_hotbar(tname, snap)
            print(f"[table] moved crafting_table {tname} -> hotbar slot {table_slot}")
        else:
            table_slot = int(tname.split("_")[1]) + 1
            print(f"[table] crafting_table in hotbar slot {table_slot}")
        ctl.close(); time.sleep(0.3)

        def _pose_now():
            fr = capture.get_frame()
            return wp.update(fr, f3.read(fr)).pose

        def _ensure_camera_live(tries=4):
            """After an inventory screen MC sometimes hasn't re-grabbed the
            mouse (the view is FROZEN, track_target can't aim). Nudge and
            confirm the pose moves; if a menu is actually OPEN, close it (or
            resume from pause). NEVER blind-escape — escape with nothing open
            opens the PAUSE menu (the live bug)."""
            for _ in range(tries):
                p0 = _pose_now(); y0 = getattr(p0, "yaw", None)
                mouse.track_target(90, 0); time.sleep(0.30)
                p1 = _pose_now(); y1 = getattr(p1, "yaw", None)
                if y0 is not None and y1 is not None \
                        and abs(norm_angle(float(y1) - float(y0))) > 1.0:
                    mouse.track_target(-90, 0); time.sleep(0.15)
                    return True
                # View didn't move. Only escape if a GUI is genuinely open.
                det = (menu_detector.detect(capture.get_frame())
                       if menu_detector is not None else None)
                if det is not None and det.menu == "pause":
                    M.ensure_playing(capture, menu_detector, kb); time.sleep(0.2)
                elif det is not None and det.open:
                    kb.tap("escape"); time.sleep(0.35)
                else:
                    time.sleep(0.25)         # no menu — wait for mouse re-grab
            return False

        if not _ensure_camera_live():
            return False, "camera won't respond (a menu may be stuck open)"

        # 2. Place the table (scans look-views, places, self-verifies). Generous
        # budget: on cratered post-chop ground the placer may step back several
        # times to reach flat ground before it can place.
        pb = PlaceBlock(slot=table_slot)
        st = drive(pb, "place", max_secs=140.0, debug=debug)
        # Success is EITHER PlaceBlock's visual confirm OR a GUI opening during
        # the scan (the click landed on the just-placed table and opened it — a
        # fresh table OCRs to garble so the visual verify often can't read it).
        gui_already_open = place_gui["open"]
        table_pos = pb.placed_at or place_gui["pos"]
        if st != SkillStatus.DONE or table_pos is None:
            return False, "couldn't place the table"
        print(f"[table] table placed at {table_pos}"
              f"{' (already open via place-click)' if gui_already_open else ''}")

        # 3. Open it — UNLESS a click during placement already opened it. Either
        # way, SETTLE before reading: the table GUI needs a moment to fully open
        # and render its slots, or the first inventory read sees a half-open /
        # empty grid and the craft fails "can't craft … from inventory".
        if not gui_already_open:
            mouse.right_click()
        time.sleep(0.9)

        # 4. Craft the 3x3 recipe in the open table.
        tctl = InventoryController(mouse, kb, reader, hotbar, capture,
                                   ui_scale=ui_scale, window_origin=origin,
                                   inspector=inspector, container="crafting_table",
                                   memory=memory)
        okc, msg = Crafter(tctl, a, cat).craft(target)
        print(f"[table] craft: {'OK' if okc else 'FAIL'}: {msg}")
        tctl.close(); time.sleep(0.5)

        # 5. Break the table back (by POSITION — its id won't OCR).
        bk = BreakLookedAt(expect_pos=table_pos)
        drive(bk, "break", max_secs=14.0, debug=debug)
        print(f"[table] table reclaimed: {bk.broke}")
        return okc, msg
    finally:
        _stop()


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
    result = ("FAILED", "did not start")
    try:
        ctrl_ok, reason = M.ensure_controllable(capture, menu_detector, kb, gate)
        if not ctrl_ok:
            M.bot_cannot_start_banner(reason); return 1
        M.bot_running_banner(f"table-craft {target.split(':')[-1]}")
        ok, msg = run_table_craft(
            target, capture=capture, mouse=mouse, kb=kb, f3=f3, wp=wp,
            menu_detector=menu_detector, reader=reader, hotbar=hotbar,
            inspector=inspector, actions=actions, gate=gate, cat=cat, assets=a,
            ui_scale=ui_scale, origin=origin, px_per_deg=px_per_deg, debug=debug)
        result = ("SUCCESS" if ok else "FAILED", msg)
        return 0 if ok else 1
    finally:
        M.bot_stopped_banner(*result)
        try:
            actions.set_attack(False); actions.release_all_movement()
        except Exception:
            pass
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
