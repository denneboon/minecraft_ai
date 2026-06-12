#!/usr/bin/env python3
"""
make X from scratch — the autonomous goal that ties everything together.

    python tools/make.py wooden_pickaxe
    python tools/make.py oak_planks 16

Works out the plan (gather raw + craft chain), GATHERS logs it's short on,
CRAFTS the 2x2 intermediates (planks/sticks/table), and TABLE-CRAFTS the final
3x3 recipe — count-aware (only the shortfall). Needs MC running + focused.
Panic: Ctrl+Shift+F12.
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
from vision.ocr import build_f3_reader, F3ReaderWorker
from vision.pose_filter import PoseFilter
from vision.world import build_world_perception
from vision.inventory import build_inventory_reader
from vision.tooltip import build_tooltip_reader
from agents.inventory_inspector import InventoryInspector, InspectorConfig
from control.hotbar import build_hotbar_manager
from control.inventory_control import InventoryController
from agents.crafting import Crafter
from agents.inventory_memory import InventoryMemory
from agents.maker import Maker
from agents.treechop import FindAndChopLogs, _full_movement
from agents.skills import SkillContext, SkillStatus
from vision.world.f3_target import targeted_block_pos
from knowledge.catalog import Catalog
from vision.mc_assets import MCAssets
from tools.table_craft import run_table_craft

_LOGSUF = ("_log", "_wood", "_stem", "_hyphae")


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    pos = [x for x in argv if not x.startswith("-")]
    target = (pos[0] if pos else "wooden_pickaxe")
    if ":" not in target:
        target = "minecraft:" + target
    count = int(pos[1]) if len(pos) > 1 and pos[1].isdigit() else 1
    debug = "--debug" in argv

    wins = _find_minecraft_hwnd()
    if not wins:
        print("[make] Minecraft not found"); return 2
    hwnd = wins[0][0]; activate_minecraft(); time.sleep(0.5)

    settings = M._load_yaml(M.SETTINGS_PATH)
    keymap = M._flatten_keymap_for_keyboard(M._load_json(M.KEYMAP_PATH))
    ui_scale = int((settings.get("capture") or {}).get("ui_scale", 2))
    px_per_deg = float((settings.get("agent") or {}).get("mouse_per_degree", 6.5) or 6.5)

    gate = InputGate()
    safety = M.build_safety(settings, gate=gate)
    mouse = M.build_mouse(settings, gate=gate)
    kb = M.build_keyboard(settings, keymap, gate=gate)
    capture = Capture(CaptureConfig(hwnd=hwnd, use_client_area=True, threaded=True))
    safety.start(); mouse.start(); kb.start(); capture.start(); time.sleep(0.3)

    f3 = build_f3_reader(settings)
    # F3 OCR is ~86 ms/read — far too slow to run inline every control tick
    # (it capped the loop at ~5 Hz). Run it on a background worker thread (as
    # main.py does) so the control loop just grabs the latest good pose in O(1)
    # and ticks at the perception's natural ~11 Hz instead. interval_sec=0 lets
    # the worker re-read back-to-back; the ~86 ms read time paces it to ~11 Hz.
    wp = build_world_perception(settings)
    f3w = F3ReaderWorker(f3, capture, pose_filter=PoseFilter(),
                         interval_sec=float((settings.get("vision", {})
                                             .get("ocr", {})
                                             .get("read_interval_sec", 0.0)) or 0.0))
    f3w.start()
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
    memory = InventoryMemory()
    ctl = InventoryController(mouse, kb, reader, hotbar, capture,
                              ui_scale=ui_scale, window_origin=origin,
                              inspector=inspector, memory=memory)
    crafter = Crafter(ctl, a, cat)

    log_ids = {b.id for b in cat.blocks_in_tag("logs")}
    leaf_ids = {b.id for b in cat.blocks_in_tag("leaves")}

    def is_log(bid):
        return bool(bid) and (bid in log_ids or str(bid).endswith(_LOGSUF))

    def is_breakable(bid):
        return bool(bid) and (is_log(bid) or bid in leaf_ids
                              or str(bid).endswith("_leaves"))

    def _dispatch(action, apply_look=True):
        if action.hotbar:
            kb.tap(str(action.hotbar))
        if apply_look and (action.look_dx or action.look_dy):
            try:
                # INSTANT relative move (not eased track_target): the treechop
                # FSM expects its per-tick look correction to land immediately,
                # else it lags and ends up aimed at a distant log it can't reach.
                # apply_look is gated to FRESH poses — re-applying the same
                # relative delta on a stale (unchanged) pose would over-rotate
                # the camera N times for one real perception update.
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

    def _gather(item_id, qty):
        """Gather ``qty`` of a raw material (logs only) by chopping trees."""
        if not is_log(item_id):
            print(f"[make] can't gather {item_id.split(':')[-1]} (only logs)")
            return False
        fsm = FindAndChopLogs(is_log=is_log, reach=3.5, max_logs=max(3, qty + 2),
                              goal_blocks=qty, tool_role="axe",
                              is_breakable=is_breakable)
        budget = 90.0 + 90.0 * qty            # generous: walk to + chop each log
        t0 = time.time(); last = None; ended = "timeout"; _lost = None
        last_ts = None; last_tick = 0.0; last_pause = time.time()
        try:
            while time.time() - t0 < budget:
                now = time.time()
                if not gate.allow():              # focus lost: don't act on a
                    _stop()                       # gated/stale frame. Try to GRAB
                    if _lost is None: _lost = now          # focus back; abort only
                    try: activate_minecraft()              # if it won't hold.
                    except Exception: pass
                    if now - _lost > 10.0:
                        ended = "focus lost (couldn't hold Minecraft foreground)"; break
                    time.sleep(0.3); t0 += 0.3; continue
                _lost = None
                # is_pause_menu is a ~2.4s full-frame OCR — TIME-throttle it (not
                # tick-throttle: ticks are now ~11 Hz) to once every few seconds.
                # The gate handles focus loss and a LAN world doesn't pause on it.
                if menu_detector is not None and now - last_pause > 4.0:
                    last_pause = now
                    if menu_detector.is_pause_menu(capture.get_frame()):
                        p0 = time.time()
                        M.ensure_playing(capture, menu_detector, kb)
                        time.sleep(0.2); t0 += time.time() - p0; continue

                # Grab the latest background-OCR'd pose (O(1), no inline OCR).
                # Tick the FSM once per FRESH pose (~11 Hz) so its look-deltas and
                # tick-based timeouts run at perception cadence; fall back to a
                # ~6 Hz forced tick during an F3-garble streak (no fresh pose) so
                # scan/recovery + timeouts keep advancing.
                f3info = f3w.latest()
                ts = getattr(f3info, "timestamp", None) if f3info is not None else None
                fresh = ts is not None and ts != last_ts
                if not fresh and (now - last_tick) < 0.15:
                    time.sleep(0.005); continue
                last_ts = ts; last_tick = now

                frame = capture.get_frame()
                wf = wp.update(frame, f3info)
                pose = wf.pose
                # Raw targeted-block coords — survive a garbled F3 id ('?' on
                # busy forest scenes) so MineBlock can confirm it's aimed on the
                # target log by POSITION even when the id is unreadable.
                tpos = None
                if f3info is not None and getattr(f3info, "raw_text", None):
                    try:
                        tpos = targeted_block_pos(f3info.raw_text.splitlines())
                    except Exception:
                        tpos = None
                ctx = SkillContext(pose=pose, world_map=wp.world_map,
                                   looking_at=wf.looking_at, targeted_pos=tpos,
                                   hotbar=hotbar, px_per_deg=px_per_deg,
                                   dimension=getattr(pose, "dimension", None) if pose else None)
                r = fsm.tick(ctx)
                _dispatch(r.action, apply_look=fresh)
                if debug and (fsm._state != last):
                    print(f"[make]  gather: {fsm._state} chopped={fsm.chopped} "
                          f"logs={fsm.logs}/{qty} pos={getattr(pose,'x',None)},{getattr(pose,'z',None)} | {r.info}")
                    last = fsm._state
                # NB: do NOT stop the instant the block breaks — the FSM still
                # has to WALK OVER the dropped item to collect it (its 'collect'
                # state). Let it run to DONE (which is after collection), so the
                # log actually lands in the inventory before we craft.
                if r.status in (SkillStatus.DONE, SkillStatus.FAILED):
                    ended = r.status.value; break
            _stop(); time.sleep(1.2)          # let auto-pickup settle
        finally:
            _stop()
        got = fsm.logs >= qty
        print(f"[make] gather {item_id.split(':')[-1]}: chopped {fsm.logs}/{qty} "
              f"({'enough' if got else 'short'}; ended={ended})")
        return got

    def _table_craft(tgt):
        return run_table_craft(
            tgt, capture=capture, mouse=mouse, kb=kb, f3=f3, wp=wp,
            menu_detector=menu_detector, reader=reader, hotbar=hotbar,
            inspector=inspector, actions=actions, gate=gate, cat=cat, assets=a,
            ui_scale=ui_scale, origin=origin, px_per_deg=px_per_deg,
            memory=memory, debug=debug, f3_worker=f3w)

    maker = Maker(ctl, crafter, memory, a, cat,
                  gather_fn=_gather, table_craft_fn=_table_craft, log=print)
    result = ("FAILED", "did not start")
    try:
        ctrl_ok, reason = M.ensure_controllable(capture, menu_detector, kb, gate)
        if not ctrl_ok:
            M.bot_cannot_start_banner(reason)
            return 1
        M.bot_running_banner(f"making {count}x {target.split(':')[-1]}")
        ok, msg = maker.make(target, count)
        result = ("SUCCESS" if ok else "FAILED", msg)
        return 0 if ok else 1
    finally:
        _stop()
        M.bot_stopped_banner(*result)
        try:
            f3w.stop()
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
