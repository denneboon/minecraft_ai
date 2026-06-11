#!/usr/bin/env python3
"""
Live test: chop a whole tree TRUNK in place (no walking). Mines the log
you're looking at, then works up the trunk, mining each log above until
there are no more. Only ever mines LOGS (safe for builds).

Prereq: Minecraft running, F3 on, LOOKING AT the base of an oak trunk
within reach. Panic: Ctrl+Shift+X / End / Pause.
    python tools/test_chop_trunk_live.py
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
from vision.ocr import build_f3_reader
from vision.world import build_world_perception
from agents.skills import ChopTrunk, SkillContext, SkillStatus

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
    wins = _find_minecraft_hwnd()
    if not wins:
        print("[chop] Minecraft not found."); return 2
    hwnd = wins[0][0]
    activate_minecraft(); time.sleep(0.4)
    settings = M._load_yaml(M.SETTINGS_PATH)
    gate = InputGate()
    safety = M.build_safety(settings, gate=gate)
    mouse = M.build_mouse(settings, gate=gate)
    capture = Capture(CaptureConfig(hwnd=hwnd, threaded=True))
    safety.start(); mouse.start(); capture.start()
    f3_reader = build_f3_reader(settings)
    wp = build_world_perception(settings)
    px_per_deg = float(((settings.get("agent", {}) or {}).get("mouse_per_degree", 6.5)) or 6.5)
    from vision.mc_assets import MCAssets
    from knowledge.catalog import Catalog
    log_ids = {b.id for b in Catalog.load(MCAssets.load()).blocks_in_tag("logs")}
    def is_log(bid):
        return bool(bid) and (bid in log_ids or str(bid).endswith("_log"))

    print("[chop] LOOK AT THE BASE OF AN OAK TRUNK. Mining logs only. "
          "Waiting up to 20s…  Panic: Ctrl+Shift+X.")
    time.sleep(0.3)

    skill = None
    attack_held = False
    status = SkillStatus.RUNNING
    aborted = False
    t0 = time.time()

    try:
        for step in range(800):
            if _panic(): print("[chop] PANIC."); aborted = True; break
            if not safety.allow_input():
                print("[chop] gate closed (focus MC)."); time.sleep(0.3); continue
            frame = capture.get_frame()
            f3 = f3_reader.read(frame)
            try:
                wf = wp.update(frame, f3)
            except Exception as e:
                print(f"[chop] perception error: {e!r}"); continue
            la = wf.looking_at

            if skill is None:
                if la is not None and is_log(la.block_id):
                    skill = ChopTrunk(tuple(la.pos), is_log=is_log,
                                      max_height=10, tool_role=None)
                    print(f"[chop] base log {tuple(la.pos)} ({la.block_id}) — chopping trunk…")
                else:
                    if time.time() - t0 > 20.0:
                        print("[chop] no log targeted within 20s — aim at a trunk "
                              "base and re-run. (safe no-op)")
                        break
                    time.sleep(0.15); continue

            ctx = SkillContext(pose=wf.pose, world_map=wp.world_map, looking_at=la,
                               px_per_deg=px_per_deg, tick=step,
                               dimension=wf.pose.dimension if wf.pose else None)
            res = skill.tick(ctx)
            status = res.status
            if res.action.look_dx or res.action.look_dy:
                try: mouse.move(int(res.action.look_dx), int(res.action.look_dy))
                except Exception: pass
            want_attack = (res.action.interact == "attack")
            if want_attack and not attack_held:
                mouse.left_press(); attack_held = True
            elif not want_attack and attack_held:
                mouse.left_release(); attack_held = False
            if step % 6 == 0 or status != SkillStatus.RUNNING:
                print(f"  [{step:3d}] {res.status.value:7} {res.info}")
            if status in (SkillStatus.DONE, SkillStatus.FAILED):
                break
            time.sleep(0.15)
    finally:
        if attack_held:
            try: mouse.left_release()
            except Exception: pass
        try: mouse.release_all() if hasattr(mouse, "release_all") else None
        except Exception: pass
        capture.stop()
        try: safety.stop()
        except Exception: pass

    mined = getattr(skill, "_mined", 0) if skill else 0
    print("\n" + "=" * 56)
    print(f"[chop] {'ABORTED' if aborted else status.value.upper() if skill else 'NO TARGET'} "
          f"— mined {mined} log(s) up the trunk")
    if skill and status == SkillStatus.DONE and mined >= 1:
        print("[chop] PASS — chopped the reachable trunk in place.")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
