#!/usr/bin/env python3
"""
Live verification of camera-only skills (SAFE — never moves, never places,
never breaks; only turns the view). Confirms the look-sign and that
LookAtVoxel actually converges in-game.

It reads the F3 pose, computes a target voxel offset ~40deg to the side and
slightly down from where the player currently looks, then runs LookAtVoxel
and prints the yaw/pitch error each tick. If the error SHRINKS to ~0 ->
the aim + sign are correct. If it GROWS -> a look sign is flipped.

Prereq: Minecraft running with F3 (pose) visible. Panic: Ctrl+Shift+X /
End / Pause.
    python tools/test_skills_live.py
"""
from __future__ import annotations

import math
import os
import sys
import time
from types import SimpleNamespace

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
from agents.skills import LookAtVoxel, SkillContext, SkillStatus, aim_angles, norm_angle

try:
    import ctypes
    _U = ctypes.windll.user32
except Exception:
    _U = None
def _panic():
    if _U is None: return False
    g = _U.GetAsyncKeyState; d = lambda v: (g(v) & 0x8000) != 0
    return (d(0x11) and d(0x10) and d(0x58)) or d(0x23) or d(0x13)


def _pose_from_f3(f3):
    if f3 is None or f3.x is None or f3.yaw is None or f3.pitch is None:
        return None
    return SimpleNamespace(x=float(f3.x), y=float(f3.y), z=float(f3.z),
                           yaw=float(f3.yaw), pitch=float(f3.pitch))


def main() -> int:
    wins = _find_minecraft_hwnd()
    if not wins:
        print("[skills-live] Minecraft not found."); return 2
    hwnd = wins[0][0]
    activate_minecraft(); time.sleep(0.4)
    settings = M._load_yaml(M.SETTINGS_PATH)
    gate = InputGate()
    safety = M.build_safety(settings, gate=gate)
    mouse = M.build_mouse(settings, gate=gate)
    capture = Capture(CaptureConfig(hwnd=hwnd, threaded=True))
    safety.start(); mouse.start(); capture.start()
    f3_reader = build_f3_reader(settings)
    px_per_deg = float(((settings.get("agent", {}) or {}).get("mouse_per_degree", 6.5)) or 6.5)
    time.sleep(0.3)

    # Read a starting pose.
    pose = None
    for _ in range(12):
        f3 = f3_reader.read(capture.get_frame())
        pose = _pose_from_f3(f3)
        if pose is not None:
            break
        time.sleep(0.2)
    if pose is None:
        print("[skills-live] couldn't read F3 pose (is F3 on?)."); safety.stop(); capture.stop(); return 2

    # Target ~ +35deg yaw, ~ +20deg pitch (down), 4 blocks out, from current view.
    Y = math.radians(pose.yaw + 35.0)
    eye = (pose.x, pose.y + 1.62, pose.z)
    tx = eye[0] + 4.0 * (-math.sin(Y))
    tz = eye[2] + 4.0 * (math.cos(Y))
    ty = eye[1] - 1.6                       # a bit below eye -> look down
    target_voxel = (int(math.floor(tx)), int(math.floor(ty)), int(math.floor(tz)))
    wy, wp = aim_angles(eye, (target_voxel[0]+0.5, target_voxel[1]+0.5, target_voxel[2]+0.5))
    print(f"[skills-live] start yaw={pose.yaw:.1f} pitch={pose.pitch:.1f}; "
          f"target voxel {target_voxel} wants yaw={wy:.1f} pitch={wp:.1f} "
          f"(d_yaw={norm_angle(wy-pose.yaw):+.1f} d_pitch={wp-pose.pitch:+.1f})")
    print("[skills-live] running LookAtVoxel (camera only)…")

    skill = LookAtVoxel(target_voxel, tol_deg=2.5)
    first_err = None
    status = SkillStatus.RUNNING
    aborted = False
    for step in range(40):
        if _panic(): print("[skills-live] PANIC."); aborted = True; break
        if not safety.allow_input():
            print("[skills-live] gate closed (focus MC)."); time.sleep(0.3); continue
        f3 = f3_reader.read(capture.get_frame())
        p = _pose_from_f3(f3)
        if p is not None:
            pose = p
        ctx = SkillContext(pose=pose, px_per_deg=px_per_deg, tick=step)
        res = skill.tick(ctx)
        ye = norm_angle(wy - pose.yaw); pe = wp - pose.pitch
        err = math.hypot(ye, pe)
        if first_err is None:
            first_err = err
        print(f"  [{step:2d}] yaw={pose.yaw:7.1f} pitch={pose.pitch:6.1f} | "
              f"err yaw={ye:+6.1f} pitch={pe:+6.1f} | dx={res.action.look_dx:+4d} "
              f"dy={res.action.look_dy:+4d} | {res.status.value}")
        status = res.status
        if status in (SkillStatus.DONE, SkillStatus.FAILED, SkillStatus.BLOCKED):
            break
        if res.action.look_dx or res.action.look_dy:
            try:
                mouse.move(int(res.action.look_dx), int(res.action.look_dy))
            except Exception as e:
                print(f"   mouse.move failed: {e!r}")
        time.sleep(0.18)

    final_err = math.hypot(norm_angle(wy - pose.yaw), wp - pose.pitch)
    mouse_release = getattr(mouse, "release_all", None)
    if callable(mouse_release):
        try: mouse_release()
        except Exception: pass
    capture.stop()
    try: safety.stop()
    except Exception: pass

    print("\n" + "=" * 56)
    verdict = "OK"
    if aborted:
        verdict = "ABORTED"
    elif status == SkillStatus.DONE and final_err < 4.0:
        verdict = "PASS — LookAtVoxel converged, look-sign CORRECT"
    elif first_err is not None and final_err > first_err + 5.0:
        verdict = "FAIL — error GREW: a look sign is FLIPPED"
    else:
        verdict = f"INCONCLUSIVE (status={status.value}, err {first_err:.1f}->{final_err:.1f})"
    print(f"[skills-live] {verdict}")
    print(f"[skills-live] aim error {first_err:.1f}deg -> {final_err:.1f}deg, "
          f"status={status.value}")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
