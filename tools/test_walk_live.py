#!/usr/bin/env python3
"""
Live test of the WalkToward skill: face a target ~N blocks ahead and walk
to it on foot. Reports arrival / stuck / edge. Reuses main._dispatch_action
so movement+look go through the real ActionWrapper.

It walks toward where you're FACING, so point yourself at open-ish ground
first. It stops itself on arrival, on getting stuck (e.g. into a tree), or
at a known drop. Panic: Ctrl+Shift+X / End / Pause.
    python tools/test_walk_live.py --blocks 4
"""
from __future__ import annotations

import argparse
import math
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
from agents.skills import WalkToward, SkillContext, SkillStatus

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
    ap.add_argument("--blocks", type=int, default=4, help="distance ahead to target")
    args = ap.parse_args()

    wins = _find_minecraft_hwnd()
    if not wins:
        print("[walk] Minecraft not found."); return 2
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
    time.sleep(0.3)

    def release_all():
        try:
            actions.set_movement_state(forward=False, backward=False, left=False,
                                       right=False, jump=False, sprint=False, sneak=False)
        except Exception: pass
        try: mouse.set_velocity(0.0, 0.0)
        except Exception: pass

    # Read a starting pose, compute a target N blocks in the facing direction.
    pose = None
    for _ in range(12):
        f3 = f3_reader.read(capture.get_frame())
        wf = wp.update(capture.get_frame(), f3)
        pose = wf.pose
        if pose is not None:
            break
        time.sleep(0.2)
    if pose is None:
        print("[walk] no F3 pose (is F3 on?)."); safety.stop(); capture.stop(); return 2

    yaw_r = math.radians(pose.yaw)
    fwd = (-math.sin(yaw_r), math.cos(yaw_r))
    target = (int(math.floor(pose.x + args.blocks * fwd[0])),
              int(math.floor(pose.y)),
              int(math.floor(pose.z + args.blocks * fwd[1])))
    start = (pose.x, pose.z)
    d0 = math.hypot(target[0] + 0.5 - pose.x, target[2] + 0.5 - pose.z)
    print(f"[walk] start ({pose.x:.1f},{pose.z:.1f}) yaw={pose.yaw:.0f} -> "
          f"target {target} (d={d0:.1f}). Walking…  Panic: Ctrl+Shift+X.")

    sk = WalkToward(target, arrive_dist=1.6, stuck_window=20)
    status = SkillStatus.RUNNING
    aborted = False
    try:
        for step in range(200):
            if _panic(): print("[walk] PANIC."); aborted = True; break
            if not safety.allow_input():
                print("[walk] gate closed (focus MC)."); release_all(); time.sleep(0.3); continue
            frame = capture.get_frame()
            f3 = f3_reader.read(frame)
            try:
                wf = wp.update(frame, f3)
            except Exception as e:
                print(f"[walk] perception error: {e!r}"); continue
            if wf.pose is not None:
                pose = wf.pose
            ctx = SkillContext(pose=pose, world_map=wp.world_map,
                               px_per_deg=px_per_deg, tick=step,
                               dimension=pose.dimension if pose else None)
            res = sk.tick(ctx)
            status = res.status
            M._dispatch_action(res.action, actions, mouse, keyboard)
            if step % 5 == 0 or status != SkillStatus.RUNNING:
                print(f"  [{step:3d}] {res.status.value:7} {res.info}")
            if status in (SkillStatus.DONE, SkillStatus.FAILED):
                break
            time.sleep(0.12)
    finally:
        release_all()
        try: mouse.release_all() if hasattr(mouse, "release_all") else None
        except Exception: pass
        capture.stop()
        try: safety.stop()
        except Exception: pass

    dend = math.hypot(target[0] + 0.5 - pose.x, target[2] + 0.5 - pose.z)
    moved = math.hypot(pose.x - start[0], pose.z - start[1])
    print("\n" + "=" * 56)
    print(f"[walk] {'ABORTED' if aborted else status.value.upper()} | "
          f"moved {moved:.1f} blocks | dist {d0:.1f} -> {dend:.1f}")
    if status == SkillStatus.DONE:
        print("[walk] PASS — WalkToward reached the target on foot.")
    elif status == SkillStatus.FAILED and moved > 1.0:
        print("[walk] partial — moved but stopped (stuck/edge); reactive safety worked.")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
