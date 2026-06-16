#!/usr/bin/env python3
"""
make X from scratch — the autonomous goal that ties everything together.

    python tools/make.py wooden_pickaxe
    python tools/make.py oak_planks 16
    python tools/make.py stone_pickaxe stone_sword        # several, in order
    python tools/make.py stone_tools                      # the whole stone set

Works out the plan (gather raw + craft chain), GATHERS logs it's short on,
CRAFTS the 2x2 intermediates (planks/sticks/table), and TABLE-CRAFTS the final
3x3 recipe — count-aware (only the shortfall). Needs MC running + focused.

Multiple targets are made in sequence, each RE-READING the inventory first, so
leftover materials carry over (e.g. after the stone pickaxe the bot still has
sticks + the reclaimed table, so the sword/axe/shovel only re-dig the
cobblestone they're short on). ``stone_tools`` expands to the full set with the
PICKAXE first (the priority tool), then sword, axe, shovel.

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
from agents.hotbar_arranger import arrange_hotbar
from agents.armor_equip import equip_best_armor
from agents.treechop import FindAndChopLogs, _full_movement, log_predicates
from agents.mining_descent import DescendToStone
from agents.skills import SkillContext, SkillStatus, MineBlock
from vision.world.f3_target import targeted_block_pos
from knowledge.catalog import Catalog
from knowledge.mining import gather_source_for
from vision.mc_assets import MCAssets
from tools.table_craft import run_table_craft

# Source blocks acquired by DIGGING DOWN (a safe staircase) rather than by
# finding an exposed face — the stone family, which sits a few blocks under the
# surface everywhere.
_DIG_DOWN_SOURCES = {"stone", "deepslate", "andesite", "diorite", "granite",
                     "tuff"}

# Convenience goals that expand to a SEQUENCE of targets, made in order (reusing
# leftover materials between them). The stone toolset makes the PICKAXE first —
# the priority tool — then sword, axe, shovel.
_GOAL_ALIASES = {
    "stone_tools":  ["stone_pickaxe", "stone_sword", "stone_axe", "stone_shovel",
                     "stone_hoe"],
    "wooden_tools": ["wooden_pickaxe", "wooden_sword", "wooden_axe", "wooden_shovel",
                     "wooden_hoe"],
    # The full demo: a wooden pickaxe (so we can mine stone), the whole stone
    # toolset, then the utility items. Each is made in order, reusing leftovers.
    "kit": ["wooden_pickaxe",
            "stone_pickaxe", "stone_sword", "stone_axe", "stone_shovel", "stone_hoe",
            "furnace", "chest", "oak_boat"],
}


def _parse_goals(pos):
    """Ordered list of (target_id, count) from the positional args.

      make oak_planks 16              -> [(oak_planks, 16)]
      make stone_pickaxe stone_sword  -> [(stone_pickaxe, 1), (stone_sword, 1)]
      make stone_tools                -> the stone toolset (pickaxe first)
    A bare number sets the count of the target just before it."""
    goals = []
    for tok in pos:
        if tok.isdigit():
            if goals:
                goals[-1] = (goals[-1][0], int(tok))
            continue
        for name in _GOAL_ALIASES.get(tok, [tok]):
            goals.append((name if ":" in name else "minecraft:" + name, 1))
    return goals or [("minecraft:wooden_pickaxe", 1)]


def _gprog(fsm):
    """How many target blocks a gather FSM has secured — works for both the
    tree/visible-block FSM (``.logs``) and the dig-down miner (``.gathered``)."""
    v = getattr(fsm, "logs", None)
    return v if v is not None else getattr(fsm, "gathered", 0)


def _gstate(fsm):
    return getattr(fsm, "_state", None) or getattr(fsm, "_phase", "")


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    pos = [x for x in argv if not x.startswith("-")]
    goals = _parse_goals(pos)
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
    # Fence absolute cursor moves (inventory/craft slot clicks) to the MC
    # window so a misread slot can never click the desktop/taskbar.
    try:
        mouse.set_play_area(capture.window_bounds())
    except Exception:
        pass

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

    # Species-agnostic log/leaf predicates (shared with the treechop agent so
    # the classification can't drift between the gatherer and the FSM).
    is_log, is_breakable = log_predicates(cat)

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

    dug = {"v": False}   # set when a gather dug a shaft -> craft via pillar-place

    def _gather(item_id, qty):
        """Gather ``qty`` of a raw material from the world.

        Logs are felled with the tree FSM; any other MINEABLE block is acquired
        with the SAME find->walk->mine FSM but a single-block predicate and the
        right tool — e.g. cobblestone by mining stone with a pickaxe (the planner
        already knows stone tools need cobblestone). Returns False up front for a
        material we can't mine, or one whose tool isn't on the hotbar (mining
        stone bare-handed drops nothing, so that would loop fruitlessly)."""
        if is_log(item_id):
            fsm = FindAndChopLogs(is_log=is_log, reach=3.5, max_logs=max(3, qty + 2),
                                  goal_blocks=qty, tool_role="axe",
                                  is_breakable=is_breakable)
        else:
            source, role = gather_source_for(item_id, cat)
            short = item_id.split(":")[-1]
            if source is None or role is None:
                print(f"[make] can't gather {short} (no known mine source)")
                return False
            if hotbar.best_slot_for(role) is None:
                # No pickaxe/shovel/… in the hotbar → the block won't drop its
                # item (stone needs a pickaxe). Fail honestly instead of mining air.
                print(f"[make] can't gather {short} — need a {role} in the hotbar "
                      f"first (mine {source.split(':')[-1]} for {short})")
                return False
            src_stem = source.split(":")[-1]
            if src_stem in _DIG_DOWN_SOURCES:
                # Stone is everywhere a few blocks DOWN — don't depend on an
                # exposed face. Cut a safe descending staircase to it (never
                # digs straight down / into lava — see agents.mining_descent).
                # Dig ONE extra: the descent counts a block as cobblestone from
                # an F3 stone-read, which can over-count by one at the dirt/stone
                # boundary (a stale/garbled read) or miss a drop — leaving the
                # craft a single cobble short, which (live) loses the just-placed
                # table on the failed craft and snowballs into a wood re-gather.
                # The spare carries over to the next tool, so it's not wasted.
                dig_qty = qty + 1
                print(f"[make] gather {short}: digging straight down to "
                      f"{src_stem} with a {role} (target {qty}, +1 buffer)")
                fsm = DescendToStone(count=dig_qty, tool_role=role,
                                     max_depth=max(8, dig_qty + 6))
                dug["v"] = True   # we'll be in a shaft -> craft via pillar-place
            else:
                pred = (lambda b, s=source, st=src_stem:
                        bool(b) and (b == s or str(b).split(":")[-1] == st))
                mine = (lambda v, r=role, p=pred:
                        MineBlock(v, tool_role=r, is_target=p, is_passthrough=p))
                print(f"[make] gather {short}: mining {src_stem} with a {role}")
                fsm = FindAndChopLogs(is_log=pred, is_breakable=pred, reach=3.5,
                                      max_logs=max(3, qty + 2), goal_blocks=qty,
                                      tool_role=role, mine_action=mine)
        budget = 90.0 + 90.0 * qty            # generous: walk to + chop each log
        t0 = time.time(); last = None; ended = "timeout"; _lost = None
        last_ts = None; last_tick = 0.0; last_pause = time.time(); garble = 0
        try:
            while time.time() - t0 < budget:
                now = time.time()
                if not gate.allow():              # MC not foreground: input is
                    _stop()                       # gated. RESPECT the user — do
                    if _lost is None:             # NOT steal focus back (that
                        _lost = now               # "kept tabbing me to MC"). Pause
                        print("[make] Minecraft lost focus — pausing. Click "
                              "Minecraft to resume, or Ctrl+Shift+F12 to stop.")
                    if now - _lost > 60.0:        # gave up waiting -> end cleanly
                        ended = "focus lost (Minecraft not refocused within 60s)"; break
                    time.sleep(0.2); t0 += 0.2; continue   # don't burn budget
                if _lost is not None:
                    print("[make] Minecraft refocused — resuming.")
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
                # Camera moves apply on FRESH poses so the aimer doesn't
                # over-rotate on stale ones. BUT if F3 garbles persistently (the
                # crosshair on empty sky / a cleared area gives no pose), the
                # look would never apply and a SCAN would deadlock — unable to
                # rotate AWAY from the empty view that's causing the garble. So
                # after a garble streak, force the look through occasionally
                # (slow, throttled) to rotate out of it.
                garble = 0 if fresh else garble + 1
                force_look = (not fresh) and garble >= 8 and (garble % 4 == 0)
                _dispatch(r.action, apply_look=fresh or force_look)
                if debug and (_gstate(fsm) != last):
                    print(f"[make]  gather: {_gstate(fsm)} got={_gprog(fsm)}/{qty} "
                          f"pos={getattr(pose,'x',None)},{getattr(pose,'z',None)} | {r.info}")
                    last = _gstate(fsm)
                # NB: do NOT stop the instant the block breaks — the FSM still
                # has to WALK OVER the dropped item to collect it (its 'collect'
                # state). Let it run to DONE (which is after collection), so the
                # log actually lands in the inventory before we craft.
                if r.status in (SkillStatus.DONE, SkillStatus.FAILED):
                    ended = r.status.value; break
            _stop(); time.sleep(1.2)          # let auto-pickup settle
        finally:
            _stop()
        got = _gprog(fsm) >= qty
        print(f"[make] gather {item_id.split(':')[-1]}: got {_gprog(fsm)}/{qty} "
              f"({'enough' if got else 'short'}; ended={ended})")
        # PROGRESS, not all-or-nothing: a partial gather (chopped some, but the
        # spot ran dry before the full qty) is NOT a failure — the Maker re-reads
        # the inventory and re-plans the remaining deficit next round, exploring
        # further. Returning False only when we got NOTHING lets the Maker's
        # gather-fails / no-progress guards stop a truly barren area, while a
        # forest edge that yields 1-2 logs per pass still completes over rounds.
        return _gprog(fsm) >= 1

    def _table_craft(tgt):
        return run_table_craft(
            tgt, capture=capture, mouse=mouse, kb=kb, f3=f3, wp=wp,
            menu_detector=menu_detector, reader=reader, hotbar=hotbar,
            inspector=inspector, actions=actions, gate=gate, cat=cat, assets=a,
            ui_scale=ui_scale, origin=origin, px_per_deg=px_per_deg,
            memory=memory, debug=debug, f3_worker=f3w, pillar_place=dug["v"])

    maker = Maker(ctl, crafter, memory, a, cat,
                  gather_fn=_gather, table_craft_fn=_table_craft, log=print)
    result = ("FAILED", "did not start")
    try:
        ctrl_ok, reason = M.ensure_controllable(capture, menu_detector, kb, gate)
        if not ctrl_ok:
            M.bot_cannot_start_banner(reason)
            return 1
        # Tidy the hotbar first: put the BEST of each role (best sword/pickaxe/
        # axe/shovel/hoe by material tier, best food, biggest block stack) into
        # its reserved slot, so the tool-selection behaviours grab the right
        # item. Best-effort — a hiccup here never blocks the make.
        try:
            roles = {int(k): str(v) for k, v in
                     ((settings.get("hotbar") or {}).get("slot_roles") or {}).items()}
            xfood = tuple((settings.get("hotbar") or {}).get("extra_food") or ())
            arr = arrange_hotbar(ctl, slot_roles=roles or None, catalog=cat,
                                 extra_food=xfood, log=print)
            if arr:
                print("[make] hotbar: " + ", ".join(
                    f"{r}={i.split(':')[-1]}" for r, i in arr.items()))
        except Exception as e:
            print(f"[make] hotbar arrange skipped: {e}")

        # Wear the best armour we're carrying (survival upkeep; best-effort).
        try:
            eq = equip_best_armor(ctl, catalog=cat, log=print)
            if eq:
                print("[make] armor: " + ", ".join(
                    f"{s}={i.split(':')[-1]}" for s, i in eq.items()))
        except Exception as e:
            print(f"[make] armor equip skipped: {e}")

        # Make each goal in order. Each maker.make() re-reads the inventory, so
        # a later tool reuses whatever the earlier ones left (sticks, planks,
        # the reclaimed table) and only re-gathers its shortfall. One tool
        # failing doesn't abort the rest — they're independent attempts.
        results = []
        for tgt, cnt in goals:
            short = tgt.split(':')[-1]
            M.bot_running_banner(f"making {cnt}x {short}")
            ok, msg = maker.make(tgt, cnt)
            results.append((short, ok, msg))
            print(f"[make] {'DONE' if ok else 'FAILED'}: {short} — {msg}")
        n_ok = sum(1 for _, ok, _ in results if ok)
        all_ok = n_ok == len(results)
        summary = ", ".join(f"{s}={'OK' if ok else 'FAIL'}" for s, ok, _ in results)
        # Tidy the hotbar after making, so crafted items land on their reserved
        # slots (e.g. a boat -> slot 4). Best-effort; never fails the run.
        if n_ok:
            try:
                arr2 = arrange_hotbar(ctl, slot_roles=roles or None, catalog=cat,
                                      log=print)
                if arr2:
                    print("[make] hotbar tidied: " + ", ".join(
                        f"{s}={i.split(':')[-1]}" for s, i in sorted(arr2.items())))
            except Exception as e:
                print(f"[make] post-make hotbar arrange skipped: {e}")
        if len(results) == 1:
            result = ("SUCCESS" if all_ok else "FAILED", results[0][2])
        else:
            result = ("SUCCESS" if all_ok else "FAILED",
                      f"{n_ok}/{len(results)} made [{summary}]")
        return 0 if all_ok else 1
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
