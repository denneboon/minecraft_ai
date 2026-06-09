#!/usr/bin/env python3
"""
Autonomous self-teaching live test for the world block recogniser.

Drives ONLY the camera (relative mouse look — never moves, never clicks,
so the world is never modified) to sweep the view across many blocks.
At each step it runs the full perception pipeline: the F3 "Targeted
Block" line gives a free ground-truth label, the crosshair patch is
auto-collected into the sample store, and the CNN retrains itself in the
background. It also scores the recogniser's guess for the crosshair block
against the F3 truth, reporting per-block and overall accuracy — so you
can watch the recogniser teach itself across a real, varied scene.

Prereqs: Minecraft running with the F3 debug overlay ON (so the Targeted
Block + pose lines are visible). Stop any time with Ctrl+Shift+X / End /
Pause (queue-independent panic poll) — the tool also stops on focus loss.

Usage:
    python tools/learn_world_live.py                 # 80 steps
    python tools/learn_world_live.py --steps 200 --pan 55
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

import numpy as np

import main as M
from utils.focus import _find_minecraft_hwnd, activate_minecraft
from control.input_gate import InputGate
from vision.capture import Capture, CaptureConfig
from vision.ocr import build_f3_reader
from vision.world import build_world_perception

# ── Queue-independent panic poll (Ctrl+Shift+X / End / Pause) ──────────
try:
    import ctypes
    _USER32 = ctypes.windll.user32
except Exception:
    _USER32 = None
_VK_CONTROL, _VK_SHIFT, _VK_X, _VK_END, _VK_PAUSE = 0x11, 0x10, 0x58, 0x23, 0x13


def _panic() -> bool:
    if _USER32 is None:
        return False
    try:
        g = _USER32.GetAsyncKeyState
        d = lambda vk: (g(vk) & 0x8000) != 0
        return (d(_VK_CONTROL) and d(_VK_SHIFT) and d(_VK_X)) or d(_VK_END) or d(_VK_PAUSE)
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=80,
                    help="camera-sweep steps (each looks at a new view)")
    ap.add_argument("--pan", type=int, default=55,
                    help="relative yaw mouse-move per step (px); the sweep "
                         "rotates a full circle and oscillates pitch")
    ap.add_argument("--settle", type=float, default=0.28,
                    help="seconds to let the view settle after each pan")
    args = ap.parse_args()

    wins = _find_minecraft_hwnd()
    if not wins:
        print("[learn] Minecraft window not found — is it running?")
        return 2
    hwnd = wins[0][0]
    activate_minecraft()
    time.sleep(0.5)

    settings = M._load_yaml(M.SETTINGS_PATH)
    gate = InputGate()
    safety = M.build_safety(settings, gate=gate)
    mouse = M.build_mouse(settings, gate=gate)
    capture = Capture(CaptureConfig(hwnd=hwnd, threaded=True))
    # start() the subsystems: safety opens the input gate while MC is the
    # foreground window (and closes it on focus loss / panic), and the
    # mouse backend needs starting before move() emits anything. Without
    # these the gate stays shut and every camera move is silently dropped.
    safety.start()
    mouse.start()
    capture.start()
    f3_reader = build_f3_reader(settings)
    wp = build_world_perception(settings)
    cnn = getattr(wp.block_classifier, "cnn", None)
    start_samples = cnn.sample_count() if cnn is not None else 0
    time.sleep(0.3)
    if not safety.allow_input():
        print("[learn] input gate is CLOSED — Minecraft must be the focused "
              "foreground window for camera moves to land. Click the MC "
              "window and re-run (keep it focused).")
    print(f"[learn] start: {wp.block_classifier.template_count()} signatures; "
          f"sample store = {start_samples}; gate={'open' if safety.allow_input() else 'closed'}")
    print("[learn] sweeping camera (look-only, world untouched). "
          "Panic: Ctrl+Shift+X / End / Pause.")

    seen = hits = abstain = no_target = 0
    per_total: Counter = Counter()
    per_hit: Counter = Counter()
    discovered = set()
    aborted = False

    try:
        for step in range(1, args.steps + 1):
            if _panic():
                print("[learn] PANIC — stopping.")
                aborted = True
                break
            # Sweep: rotate yaw steadily, oscillate pitch to scan ground↔eye.
            dx = args.pan
            dy = int(18 * np.sin(step * 0.5))     # gentle up/down scan
            try:
                mouse.move(dx, dy)
            except Exception:
                pass
            time.sleep(args.settle)

            try:
                frame = capture.get_frame()
            except Exception:
                continue
            f3 = f3_reader.read(frame)
            try:
                wf = wp.update(frame, f3)
            except Exception as e:
                print(f"[learn] step {step}: perception error: {e!r}")
                continue

            if wf.pose is None:
                continue
            if wf.looking_at is None:
                no_target += 1
                continue

            truth = wf.looking_at.block_id
            discovered.add(truth)
            h, w = frame.shape[:2]
            # Crop EXACTLY like the auto-sampler trains on: sample_capture_px
            # (NOT patch_size_px) + crosshair inpaint, or the patch is at a
            # different scale than the model ever saw. The crosshair is at
            # screen centre.
            patch = wp._crop_patch(frame, w // 2, h // 2,
                                   wp.cfg.sample_capture_px)
            if patch is not None and wp.cfg.mask_crosshair_in_samples:
                try:
                    patch = wp._mask_crosshair(patch)
                except Exception:
                    pass
            guess, conf = (None, 0.0)
            if patch is not None:
                guess, conf = wp.block_classifier.classify(patch)
            seen += 1
            per_total[truth] += 1
            if guess is None:
                abstain += 1
                mark = "abstain"
            elif guess == truth:
                hits += 1
                per_hit[truth] += 1
                mark = "HIT"
            else:
                mark = f"miss→{guess}"
            acc = hits / seen if seen else 0.0
            ns = cnn.sample_count() if cnn is not None else 0
            tr = " train" if (cnn is not None and cnn._training) else ""
            print(f"[{step:3d}] {truth:24s} guess={str(guess):24s} "
                  f"{conf:.2f} {mark:14s} | acc={acc:.0%} samples={ns}{tr}")
    finally:
        try:
            mouse.release_all() if hasattr(mouse, "release_all") else None
        except Exception:
            pass
        capture.stop()
        try:
            safety.stop()
        except Exception:
            pass

    # ── Summary ──
    print("\n" + "=" * 60)
    end_samples = cnn.sample_count() if cnn is not None else 0
    acc = hits / seen if seen else 0.0
    print(f"[learn] {'ABORTED — ' if aborted else ''}steps with a target: {seen} "
          f"({no_target} steps saw no block), abstained {abstain}")
    print(f"[learn] overall accuracy vs F3 truth: {acc:.0%} ({hits}/{seen})")
    print(f"[learn] sample store: {start_samples} → {end_samples} "
          f"(+{end_samples - start_samples}); blocks seen this run: {len(discovered)}")
    if cnn is not None:
        print(f"[learn] recogniser: {cnn.status()}")
    if per_total:
        print("[learn] per-block (correct/seen):")
        for b in sorted(per_total):
            print(f"    {b:26s} {per_hit[b]}/{per_total[b]}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
