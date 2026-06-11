#!/usr/bin/env python3
"""
Live verification of the MineBlock skill on a real LOG (and only a log —
it refuses to mine anything else, so it can't damage builds/chests).

It runs the real perception each tick, and when F3's targeted block is a
log it runs the actual MineBlock skill: aim, then hold attack until the
log breaks. It reports which completion signal fired (F3 target moved off
vs WorldMap carved air) and how long it took — the data that confirms the
mining primitive works in-game.

Prereq: Minecraft running, F3 on, and you LOOKING AT (or near) an oak log
within reach. Panic: Ctrl+Shift+X / End / Pause.
    python tools/test_mine_live.py
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
from agents.skills import MineBlock, SkillContext, SkillStatus

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
        print("[mine-live] Minecraft not found."); return 2
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

    # Log id set (authoritative) + suffix fallback.
    try:
        log_ids = {b.id for b in wp_catalog_logs()}
    except Exception:
        log_ids = set()
    def is_log(bid):
        return bool(bid) and (bid in log_ids or bid.endswith("_log")
                              or bid.endswith("_stem") or "stripped" in bid)

    print("[mine-live] LOOK AT AN OAK LOG (within reach). I will only mine "
          "logs. Waiting up to 20s for a log target…  Panic: Ctrl+Shift+X.")
    time.sleep(0.3)

    target = None
    skill = None
    attack_held = False
    f3_signal_tick = wm_signal_tick = None
    status = SkillStatus.RUNNING
    aborted = False
    t0 = time.time()
    mine_start_tick = None

    try:
        for step in range(400):
            if _panic(): print("[mine-live] PANIC."); aborted = True; break
            if not safety.allow_input():
                print("[mine-live] gate closed (focus MC)."); time.sleep(0.3); continue
            frame = capture.get_frame()
            f3 = f3_reader.read(frame)
            try:
                wf = wp.update(frame, f3)
            except Exception as e:
                print(f"[mine-live] perception error: {e!r}"); continue
            la = wf.looking_at

            # Acquire a log target.
            if target is None:
                if la is not None and is_log(la.block_id):
                    target = tuple(la.pos)
                    skill = MineBlock(target, tool_role=None, max_ticks=300)
                    mine_start_tick = step
                    print(f"[mine-live] LOG target {target} ({la.block_id}) — mining…")
                else:
                    if time.time() - t0 > 20.0:
                        print("[mine-live] no log target within 20s — aim at an "
                              "oak log and re-run. (safe no-op)")
                        break
                    time.sleep(0.15)
                    continue

            # Independent break-signal monitoring (diagnostic).
            if f3_signal_tick is None and la is not None and tuple(la.pos) != target:
                f3_signal_tick = step
            try:
                obs = wp.world_map.get_block(target, dimension=wf.pose.dimension if wf.pose else None)
            except Exception:
                obs = None
            if wm_signal_tick is None and obs is not None and obs.block_id == "minecraft:air":
                wm_signal_tick = step

            ctx = SkillContext(pose=wf.pose, world_map=wp.world_map,
                               looking_at=la, px_per_deg=px_per_deg, tick=step,
                               dimension=wf.pose.dimension if wf.pose else None)
            res = skill.tick(ctx)
            status = res.status

            # Dispatch the skill's action (look + hold-attack only).
            if res.action.look_dx or res.action.look_dy:
                try: mouse.move(int(res.action.look_dx), int(res.action.look_dy))
                except Exception: pass
            want_attack = (res.action.interact == "attack")
            if want_attack and not attack_held:
                mouse.left_press(); attack_held = True
            elif not want_attack and attack_held:
                mouse.left_release(); attack_held = False

            if step % 5 == 0 or status != SkillStatus.RUNNING:
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

    print("\n" + "=" * 56)
    if target is None:
        print("[mine-live] no log mined (none targeted).")
    else:
        dur = (step - mine_start_tick) * 0.15 if mine_start_tick is not None else 0
        print(f"[mine-live] {'ABORTED' if aborted else status.value.upper()} "
              f"on {target} after ~{dur:.1f}s")
        print(f"[mine-live] break signals — F3-target-moved: "
              f"{'tick '+str(f3_signal_tick) if f3_signal_tick is not None else 'never'} | "
              f"WorldMap-air: {'tick '+str(wm_signal_tick) if wm_signal_tick is not None else 'never'}")
        if status == SkillStatus.DONE:
            print("[mine-live] PASS — MineBlock broke the log in-game.")
        elif status == SkillStatus.FAILED:
            print("[mine-live] FAIL — timed out; check the break-signal data above.")
    print("=" * 56)
    return 0


def wp_catalog_logs():
    from vision.mc_assets import MCAssets
    from knowledge.catalog import Catalog
    return Catalog.load(MCAssets.load()).blocks_in_tag("logs")


if __name__ == "__main__":
    raise SystemExit(main())
