#!/usr/bin/env python3
"""
Live run of the WHOLE tree-chopping behaviour: FindAndChopLogs scans for a
log, walks to it, chops the trunk, and moves on — up to --logs trees.

Only mines logs (safe for builds). It walks, so stand somewhere with a
tree or two around. Panic: Ctrl+Shift+X / End / Pause.
    python tools/test_treechop_live.py --logs 2
"""
from __future__ import annotations

import argparse
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
from agents.treechop import FindAndChopLogs
from agents.skills import SkillContext, SkillStatus

try:
    import ctypes
    _U = ctypes.windll.user32
except Exception:
    _U = None
def _panic():
    if _U is None: return False
    g = _U.GetAsyncKeyState; d = lambda v: (g(v) & 0x8000) != 0
    return (d(0x11) and d(0x10) and d(0x58)) or d(0x23) or d(0x13)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=int, default=2, help="max logs to chop")
    ap.add_argument("--max-steps", type=int, default=700)
    args = ap.parse_args()

    wins = _find_minecraft_hwnd()
    if not wins:
        print("[chop] Minecraft not found."); return 2
    hwnd = wins[0][0]
    activate_minecraft(); time.sleep(0.4)
    settings = M._load_yaml(M.SETTINGS_PATH)
    keymap_flat = M._flatten_keymap_for_keyboard(M._load_json(M.KEYMAP_PATH))
    gate = InputGate()
    safety = M.build_safety(settings, gate=gate)
    keyboard = M.build_keyboard(settings, keymap_flat, gate=gate)
    mouse = M.build_mouse(settings, gate=gate)
    capture = Capture(CaptureConfig(hwnd=hwnd, threaded=True))
    safety.start(); mouse.start(); capture.start()
    actions = ActionWrapper(keyboard=keyboard, mouse=mouse, gate=gate)
    f3_reader = build_f3_reader(settings)
    wp = build_world_perception(settings)
    px_per_deg = float(((settings.get("agent", {}) or {}).get("mouse_per_degree", 6.5)) or 6.5)
    from vision.mc_assets import MCAssets
    from knowledge.catalog import Catalog
    from control.hotbar import build_hotbar_manager
    _cat = Catalog.load(MCAssets.load())
    log_ids = {b.id for b in _cat.blocks_in_tag("logs")}
    leaf_ids = {b.id for b in _cat.blocks_in_tag("leaves")}
    def is_breakable(bid):   # tree-chopping breaks ONLY logs + leaves
        return bool(bid) and (bid in log_ids or bid in leaf_ids
                              or str(bid).endswith("_log") or str(bid).endswith("_leaves"))
    # Config-trusting hotbar (no inventory read needed): mining selects the
    # reserved axe slot from settings.yaml (hotbar.slot_roles).
    hotbar = build_hotbar_manager(settings, catalog=_cat)
    print(f"[chop] hotbar: axe slot {hotbar.assigned_slot('axe')}, "
          f"blocks slot {hotbar.assigned_slot('blocks')}")
    def is_log(bid):
        return bool(bid) and (bid in log_ids or str(bid).endswith("_log"))
    time.sleep(0.3)

    attack = [False]
    def release_all():
        try:
            actions.set_movement_state(forward=False, backward=False, left=False,
                                       right=False, jump=False, sprint=False, sneak=False)
        except Exception: pass
        if attack[0]:
            try: mouse.left_release()
            except Exception: pass
            attack[0] = False

    def dispatch(a):
        if a.movement:
            actions.set_movement_state(**a.movement)
        if a.hotbar is not None:
            try: keyboard.select_hotbar_slot(int(a.hotbar))
            except Exception: pass
        if a.look_dx or a.look_dy:
            try: mouse.move(int(a.look_dx), int(a.look_dy))
            except Exception: pass
        want = (a.interact == "attack")
        if want and not attack[0]:
            mouse.left_press(); attack[0] = True
        elif not want and attack[0]:
            mouse.left_release(); attack[0] = False

    fsm = FindAndChopLogs(is_log=is_log, reach=3.5, max_logs=args.logs,
                          tool_role="axe", is_breakable=is_breakable)
    print(f"[chop] FindAndChopLogs — target {args.logs} log(s). Walks + mines "
          f"logs only. Panic: Ctrl+Shift+X.")
    status = SkillStatus.RUNNING
    aborted = False
    last_state = None
    try:
        for step in range(args.max_steps):
            if _panic(): print("[chop] PANIC."); aborted = True; break
            if not safety.allow_input():
                print("[chop] gate closed (focus MC)."); release_all(); time.sleep(0.3); continue
            frame = capture.get_frame()
            f3 = f3_reader.read(frame)
            try:
                wf = wp.update(frame, f3)
            except Exception as e:
                print(f"[chop] perception error: {e!r}"); continue
            ctx = SkillContext(pose=wf.pose, world_map=wp.world_map,
                               looking_at=wf.looking_at, hotbar=hotbar,
                               px_per_deg=px_per_deg,
                               tick=step, dimension=wf.pose.dimension if wf.pose else None)
            res = fsm.tick(ctx)
            status = res.status
            dispatch(res.action)
            if fsm._state != last_state or status != SkillStatus.RUNNING:
                print(f"  [{step:3d}] {fsm._state:9} chopped={fsm.chopped} | {res.info}")
                last_state = fsm._state
            if status in (SkillStatus.DONE, SkillStatus.FAILED):
                break
            time.sleep(0.13)
    finally:
        release_all()
        try: mouse.release_all() if hasattr(mouse, "release_all") else None
        except Exception: pass
        capture.stop()
        try: safety.stop()
        except Exception: pass

    print("\n" + "=" * 56)
    print(f"[chop] {'ABORTED' if aborted else status.value.upper()} — "
          f"chopped {fsm.chopped} log column(s)")
    if fsm.chopped >= 1:
        print("[chop] PASS — found, approached, and chopped a tree autonomously.")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
