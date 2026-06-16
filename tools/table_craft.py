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
import math
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
from agents.skills import (PlaceBlock, BreakLookedAt, LookAtVoxel, PillarUp,
                           SkillContext, SkillStatus, norm_angle)
from vision.world.f3_target import targeted_block_pos
from agents.treechop import _full_movement
from knowledge.catalog import Catalog
from vision.mc_assets import MCAssets

TABLE = "minecraft:crafting_table"


def run_table_craft(target, *, capture, mouse, kb, f3, wp, menu_detector,
                    reader, hotbar, inspector, actions, gate, cat, assets,
                    ui_scale=2, origin=(0, 0), px_per_deg=6.5, memory=None,
                    debug=False, f3_worker=None, pillar_place=False):
    """Place a crafting table, open it, craft ``target`` (3x3), break the table
    back. Components are provided + owned by the CALLER (not started/stopped
    here). Returns ``(ok, message)``.

    ``pillar_place`` places the table UNDER THE FEET (jump + place, the pillar
    mechanic) and opens it by looking straight down — so it works in a tight
    1-wide spot (e.g. the bottom of a dig shaft) where there's no ground in
    front to place it on. The bot stands on the table to use it."""
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
            # Gate on FRESH: the F3 worker returns the SAME cached pose for
            # several drive ticks between its ~11 Hz updates while this loop
            # spins at ~20 Hz, so a stale repeat looks identical even when the
            # camera is really turning. Counting stale repeats as "frozen"
            # spuriously triggers the (expensive) menu probe mid-swing and, worse,
            # could mis-read a real-but-irrelevant GUI as "table placed". Only
            # judge freeze across genuinely NEW poses.
            if fresh and pose is not None and (r.action.look_dx or r.action.look_dy):
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
            # Prefer the background F3 worker — it applies the pose filter and
            # reads continuously, so it resolves a yaw where a single inline
            # read garbles (the live "camera won't respond" was actually an
            # UNREADABLE pose, not a frozen view). Its reading carries .yaw, so
            # the camera-live nudge check can use it directly.
            if f3_worker is not None:
                r = f3_worker.latest()
                if r is not None and getattr(r, "yaw", None) is not None:
                    return r
            fr = capture.get_frame()
            return wp.update(fr, f3.read(fr)).pose

        def _ensure_camera_live(tries=4):
            """After an inventory screen MC sometimes hasn't re-grabbed the
            mouse (the view is FROZEN, track_target can't aim). Nudge and
            confirm the pose moves; if a menu is actually OPEN, close it (or
            resume from pause). NEVER blind-escape — escape with nothing open
            opens the PAUSE menu (the live bug)."""
            saw_real_freeze = False
            for i in range(tries):
                p0 = _pose_now(); y0 = getattr(p0, "yaw", None)
                mouse.track_target(90, 0); time.sleep(0.30)
                p1 = _pose_now(); y1 = getattr(p1, "yaw", None)
                readable = (y0 is not None and y1 is not None)
                dy = abs(norm_angle(float(y1) - float(y0))) if readable else None
                if dy is not None and dy > 1.0:
                    mouse.track_target(-90, 0); time.sleep(0.15)
                    if debug:
                        print(f"[table] camera-live OK (try {i+1}, dyaw={dy:.1f})")
                    return True
                det = (menu_detector.detect(capture.get_frame())
                       if menu_detector is not None else None)
                if debug:
                    print(f"[table] camera-live try {i+1}: "
                          f"{'FROZEN' if readable else 'UNREADABLE'} (y0={y0} y1={y1} "
                          f"dyaw={dy}) menu={getattr(det,'menu',None)} "
                          f"open={getattr(det,'open',None)}")
                if det is not None and det.menu == "pause":
                    saw_real_freeze = True
                    M.ensure_playing(capture, menu_detector, kb); time.sleep(0.2)
                elif det is not None and det.open:
                    saw_real_freeze = True
                    kb.tap("escape"); time.sleep(0.35)
                elif readable:
                    # Readable pose that DIDN'T move + no menu -> a genuine stuck
                    # view (rare). Undo our nudge and keep probing.
                    saw_real_freeze = True
                    mouse.track_target(-90, 0); time.sleep(0.2)
                else:
                    # UNREADABLE pose — F3 just couldn't OCR this frame's yaw.
                    # That is NOT evidence of a frozen camera, so don't treat it
                    # as one. Undo the nudge and wait for a readable frame.
                    mouse.track_target(-90, 0); time.sleep(0.2)
            # Never confirmed the view turned. If every failure was an UNREADABLE
            # pose (no real freeze, no menu), the camera is almost surely fine —
            # F3 just couldn't read it — so PROCEED rather than abort a good
            # craft (the live "camera won't respond" was exactly this). Only give
            # up when we actually observed a real freeze / stuck menu.
            if debug:
                print(f"[table] camera-live done: "
                      f"{'GAVE UP (real freeze)' if saw_real_freeze else 'proceeding (only unreadable poses)'}")
            return not saw_real_freeze

        if not _ensure_camera_live():
            return False, "camera won't respond (a menu may be stuck open)"

        # 2. Place the table. For a normal craft, try the GROUND placement
        # first; if it can't find a clear spot (cluttered / uneven / edge
        # ground — the live "couldn't place the table"), FALL BACK to the pillar
        # place (table UNDER the feet — works anywhere there's air above), so a
        # bad stance never strands the craft. ``pillar_place`` (shaft) goes
        # straight to pillar.
        used_pillar = bool(pillar_place)
        gui_already_open = False
        table_pos = None
        if not pillar_place:
            # Place on the ground in front (scans look-views, self-verifies).
            pb = PlaceBlock(slot=table_slot)
            st = drive(pb, "place", max_secs=140.0, debug=debug)
            # Success is EITHER PlaceBlock's visual confirm OR a GUI opening
            # during the scan (the click landed on the just-placed table and
            # opened it — a fresh table OCRs to garble so the visual verify
            # often can't read it).
            gui_already_open = place_gui["open"]
            table_pos = pb.placed_at or place_gui["pos"]
            if st == SkillStatus.DONE and table_pos is not None:
                print(f"[table] table placed at {table_pos}"
                      f"{' (already open via place-click)' if gui_already_open else ''}")
            else:
                print("[table] ground placement failed — falling back to "
                      "pillar-place (table under the feet)")
                used_pillar = True
        if used_pillar:
            # Pillar-place it UNDER the feet (jump + place). The bot ends up
            # standing ON the table; opened by looking straight down.
            pu = PillarUp(height=1, slot=table_slot, place_pitch=85.0,
                          per_block_budget=60)
            st = drive(pu, "place-table", max_secs=22.0, debug=debug)
            reading = (f3_worker.latest() if f3_worker is not None
                       else f3.read(capture.get_frame()))
            wf = wp.update(capture.get_frame(), reading)
            pose = wf.pose
            if st != SkillStatus.DONE or pose is None:
                return False, "couldn't place the table (ground + pillar both failed)"
            table_pos = (int(math.floor(pose.x)), int(math.floor(pose.y)) - 1,
                         int(math.floor(pose.z)))
            gui_already_open = False
            print(f"[table] pillar-placed table under feet at {table_pos}")

        # 3. Open it — UNLESS a click during placement already opened it.
        def _probe_camera():
            """One look nudge: True if it TURNED the view, False if not, None if
            the pose was unreadable."""
            p0 = _pose_now(); y0 = getattr(p0, "yaw", None)
            mouse.move(70, 0); time.sleep(0.28)
            p1 = _pose_now(); y1 = getattr(p1, "yaw", None)
            if y0 is None or y1 is None:
                mouse.move(-70, 0); time.sleep(0.05)
                return None
            moved = abs(norm_angle(float(y1) - float(y0))) > 1.5
            if moved:
                mouse.move(-70, 0); time.sleep(0.05)   # undo the probe turn
            return moved

        def _camera_frozen():
            """True if look nudges do NOT turn the view — i.e. a GUI is open (MC
            unlocks the cursor, so look input no longer rotates the camera). The
            menu detector only knows the PAUSE menu, NOT a crafting GUI, so the
            camera response is what distinguishes 'table open' from 'gameplay'.

            We re-probe before concluding FROZEN: right after a GUI CLOSES, MC
            briefly hasn't re-grabbed the mouse, so a single probe reads as
            'frozen' even though we're back in gameplay — that false positive
            made the verified-close press Escape and pop the PAUSE menu (the
            'keeps pausing' symptom). If EITHER probe turns the view it's
            gameplay; only a view that stays put across both is really frozen."""
            r = _probe_camera()
            if r is None:
                return False            # unreadable -> don't claim a GUI is open
            if r:
                return False            # turned -> gameplay
            time.sleep(0.2)             # let a just-closed GUI finish re-grabbing
            r2 = _probe_camera()
            return r2 is False          # frozen only if it STAYS put (not None)

        def _open_table(straight_down, tries=5):
            """Open the just-placed table, VERIFIED by camera freeze, retrying.
            A single unverified right-click (the old ground path) silently
            misses on messy/forest ground — the craft then fails, the table is
            lost, and the maker remakes ANOTHER table and loops (the live
            'places, never opens, keeps crafting tables' bug). Here we AIM at the
            table, right-click, and confirm the view FROZE (= a GUI opened);
            on a miss we re-aim and retry.

            ``straight_down`` (pillar): the table is under the feet, and F3
            under-reads pitch at steep angles, so we can't trust an aim skill —
            FORCE the look straight down with relative nudges. Otherwise (ground)
            the table is in front: aim onto it with LookAtVoxel. A non-placeable
            slot is selected first so a stray right-click can only OPEN the
            table, never place a block onto it."""
            safe = hotbar.best_slot_for("pickaxe") if hotbar is not None else None
            if safe is not None:
                kb.tap(str(int(safe))); time.sleep(0.2)
            for k in range(tries):
                # If a GUI is ALREADY open (a prior right-click opened the table
                # but the freeze-check missed it last loop), STOP — do NOT run the
                # relative straight-down nudges below: with the cursor unlocked
                # they walk the OS cursor off the bottom of the window onto the
                # TASKBAR, and the next right-click hits it (the live bug). A
                # frozen view = a GUI is open = the table is open.
                if _camera_frozen():
                    det = (menu_detector.detect(capture.get_frame())
                           if menu_detector is not None else None)
                    if det is not None and det.menu == "pause":
                        M.ensure_playing(capture, menu_detector, kb); time.sleep(0.2)
                        continue
                    if debug:
                        print(f"[table] table already open (view frozen, try {k+1})")
                    return True
                if straight_down:
                    # Relative nudges past 90° clamp at straight-down regardless
                    # of what F3 reads. Land directly (no fresh-pose gating).
                    # Reached only in GAMEPLAY (the freeze-check above guards it),
                    # so they move the CAMERA, not the unlocked GUI cursor.
                    for _ in range(8):
                        mouse.move(0, 40); time.sleep(0.02)
                    time.sleep(0.15)
                else:
                    # Table is in front: turn the crosshair onto it.
                    drive(LookAtVoxel(table_pos, tol_deg=6.0), "aim-open",
                          max_secs=3.0, debug=debug)
                mouse.right_click()
                time.sleep(0.6)
                if _camera_frozen():
                    # A GUI is open. Make sure it isn't the PAUSE menu (also
                    # freezes the view) — if it is, resume and retry.
                    det = (menu_detector.detect(capture.get_frame())
                           if menu_detector is not None else None)
                    if det is not None and det.menu == "pause":
                        M.ensure_playing(capture, menu_detector, kb); time.sleep(0.2)
                        continue
                    if debug:
                        print(f"[table] table opened + verified (view froze, "
                              f"try {k+1}/{tries})")
                    return True
                elif debug:
                    print(f"[table] open try {k+1}/{tries}: view still live "
                          f"(not open) — retrying")
            return False

        def _reclaim_table():
            """Break the placed table back into the inventory. For a PILLAR
            table (under the feet) FORCE the look straight down — LookAtVoxel
            can't confirm the steep angle (F3 under-reads pitch), so the
            aim-break TIMES OUT and the table is LOST, which then makes the
            maker re-gather wood to remake it. The forced nudges put the
            crosshair on the table directly. A ground table is aimed normally."""
            if used_pillar:
                # Only nudge straight down in GAMEPLAY — if a GUI were somehow
                # still open these relative moves would walk the unlocked cursor
                # onto the taskbar. The freeze-check guards it.
                if not _camera_frozen():
                    for _ in range(8):
                        mouse.move(0, 40); time.sleep(0.02)
                    time.sleep(0.2)
            else:
                drive(LookAtVoxel(table_pos, tol_deg=4.0), "aim-break",
                      max_secs=3.0, debug=debug)
            bk = BreakLookedAt(expect_pos=table_pos)
            drive(bk, "break", max_secs=14.0, debug=debug)
            return bk.broke

        # Trust a GUI that a place-click already opened ONLY if the view is in
        # fact frozen now; otherwise open it ourselves (verified, retrying).
        if not (gui_already_open and _camera_frozen()):
            if not _open_table(straight_down=used_pillar):
                # Couldn't open it — do NOT blind-click slots (that's how the
                # cursor ends up off-window) and do NOT leave the table behind
                # for the maker to remake. Reclaim it and fail clean.
                print("[table] could not open the placed table — reclaiming")
                got_back = _reclaim_table()
                print(f"[table] table reclaimed: {got_back}")
                return False, "couldn't open the placed table"
        time.sleep(0.9)

        # 4. Craft the 3x3 recipe in the open table.
        tctl = InventoryController(mouse, kb, reader, hotbar, capture,
                                   ui_scale=ui_scale, window_origin=origin,
                                   inspector=inspector, container="crafting_table",
                                   memory=memory)
        okc, msg = Crafter(tctl, a, cat).craft(target)
        print(f"[table] craft: {'OK' if okc else 'FAIL'}: {msg}")
        tctl.close(); time.sleep(0.5)
        # Make sure the GUI REALLY closed before trying to break the table. A
        # single Escape can miss, and the menu detector can't see a crafting
        # GUI — so a still-open table leaves the camera frozen, the aim-break
        # can't turn onto the table, and the reclaim fails (live: lost table ->
        # the maker has to re-gather wood to remake it). Verify by camera
        # response and re-press Escape until the view turns again.
        for _ in range(3):
            if not _camera_frozen():
                break                       # gameplay (camera responds) -> closed
            # Still frozen. It could be the table GUI (Escape missed) OR — if a
            # prior Escape over-shot — the PAUSE menu. Check before pressing
            # Escape again, so we never TOGGLE pause on/off (the "keeps pausing"
            # symptom); if pause is up, resume instead of Escaping it back open.
            det = (menu_detector.detect(capture.get_frame())
                   if menu_detector is not None else None)
            if det is not None and det.menu == "pause":
                M.ensure_playing(capture, menu_detector, kb); time.sleep(0.2)
                break
            kb.tap("escape"); time.sleep(0.3)

        # 5. Break the table back (by POSITION — its id won't OCR). _reclaim_table
        # re-aims onto it first (force straight-down for a pillar table, where
        # LookAtVoxel can't confirm the steep angle and would time out, losing
        # the table) then breaks it.
        print(f"[table] table reclaimed: {_reclaim_table()}")
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
    try:
        mouse.set_play_area(capture.window_bounds())   # fence slot clicks to MC
    except Exception:
        pass

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
