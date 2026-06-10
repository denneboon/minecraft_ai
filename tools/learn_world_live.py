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
from vision.world.metrics import SessionMetrics, default_metrics_root

# ── Queue-independent panic poll (Ctrl+Shift+X / End / Pause) ──────────
try:
    import ctypes
    _USER32 = ctypes.windll.user32
except Exception:
    _USER32 = None
_VK_CONTROL, _VK_SHIFT, _VK_X, _VK_END, _VK_PAUSE = 0x11, 0x10, 0x58, 0x23, 0x13


def _save_annotated(patch_dir, step, frame_rgb, patch_rgb, cap_px,
                    truth, guess, conf, mark):
    """Write two artefacts for one classification so a human can verify it:

      * ``stepNNN_<mark>_<truth>.png`` — the EXACT crop the recogniser
        classified, upscaled, with a guess-vs-truth banner and a border
        colour-coded HIT(green)/miss(red)/abstain(yellow).
      * ``stepNNN_ctx.png`` — the full frame (downscaled) with the crop
        box drawn at the crosshair, so you can confirm the crop is
        centred on the right block and sized to ~one block face.
    """
    import cv2
    GREEN, RED, YELLOW = (0, 200, 0), (0, 0, 220), (0, 200, 220)
    col = GREEN if mark == "HIT" else (YELLOW if mark == "abstain" else RED)
    short_t = truth.split(":")[-1]
    short_g = str(guess).split(":")[-1]
    # Filename-safe tag (mark may be "miss→oak_leaves" with non-ASCII).
    tag = "HIT" if mark == "HIT" else ("abstain" if mark == "abstain" else "miss")

    # --- annotated crop ---
    if patch_rgb is not None and patch_rgb.size:
        bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
        big = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_NEAREST)
        canvas = cv2.copyMakeBorder(big, 4, 78, 4, 4,
                                    cv2.BORDER_CONSTANT, value=col)
        y0 = 256 + 4
        cv2.putText(canvas, f"#{step} {mark}", (8, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"F3 : {short_t}", (8, y0 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"AI : {short_g} {conf:.2f}", (8, y0 + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1, cv2.LINE_AA)
        fn = f"step{step:03d}_{tag}_{short_t}.png"
        cv2.imwrite(os.path.join(patch_dir, fn), canvas)

    # --- boxed context frame ---
    if frame_rgb is not None and frame_rgb.size:
        h, w = frame_rgb.shape[:2]
        scale = 480.0 / w
        ctx = cv2.resize(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                         (480, int(h * scale)), interpolation=cv2.INTER_AREA)
        cx, cy = 240, int(h * scale / 2)
        half = max(2, int(cap_px * scale / 2))
        cv2.rectangle(ctx, (cx - half, cy - half), (cx + half, cy + half), col, 2)
        cv2.drawMarker(ctx, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
        cv2.imwrite(os.path.join(patch_dir, f"step{step:03d}_ctx.png"), ctx)


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
    ap.add_argument("--save-patches", action="store_true",
                    help="save an annotated screenshot of every classified "
                         "patch (the crop the recogniser saw + its guess vs "
                         "the F3 truth + a boxed full-frame context) under "
                         "data/debug/self_teach_<ts>/ so guesses can be "
                         "visually double-checked.")
    ap.add_argument("--no-metrics", action="store_true",
                    help="don't record session metrics (default: records "
                         "graph-ready accuracy/coverage/per-block history to "
                         "data/metrics/).")
    args = ap.parse_args()

    session_ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    session_id = f"self_teach_{session_ts}"
    start_ts_unix = time.time()

    patch_dir = None
    if args.save_patches:
        patch_dir = os.path.join(ROOT, "data", "debug", session_id)
        os.makedirs(patch_dir, exist_ok=True)
        print(f"[learn] saving annotated patches -> {patch_dir}")

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
    metrics = (None if args.no_metrics
               else SessionMetrics("learn_world_live", session_id, start_ts_unix))

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
            # Crop EXACTLY like the auto-sampler trains on: distance-
            # normalised size at the target's distance + crosshair inpaint.
            lp = wf.looking_at.pos
            dist = (((lp[0] + 0.5 - wf.pose.x) ** 2
                     + (lp[1] + 0.5 - wf.pose.eye_y) ** 2
                     + (lp[2] + 0.5 - wf.pose.z) ** 2) ** 0.5)
            sr = getattr(wp, "_screen_ray", None)
            intr = sr.intrinsics if sr is not None else None
            cap_px = wp._apparent_crop_px(intr, dist)
            patch = wp._crop_patch(frame, w // 2, h // 2, cap_px)
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
            if metrics is not None:
                metrics.record(step=step, truth=truth, guess=guess,
                               conf=conf, samples=ns)
            print(f"[{step:3d}] {truth:24s} guess={str(guess):24s} "
                  f"{conf:.2f} {mark:14s} | acc={acc:.0%} samples={ns}{tr}")
            if patch_dir is not None:
                try:
                    _save_annotated(patch_dir, step, frame, patch, cap_px,
                                    truth, guess, conf, mark)
                except Exception as e:
                    print(f"[learn] patch-save failed at step {step}: {e!r}")
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
    if metrics is not None:
        summ = metrics.finalize(
            no_target=no_target,
            samples_before=start_samples, samples_after=end_samples,
            recognizer_status=(cnn.status() if cnn is not None else ""),
            end_ts_unix=time.time(), aborted=aborted)
        print(f"[learn] metrics -> {default_metrics_root() / 'sessions.jsonl'} "
              f"(session {session_id}); decided-acc={summ['accuracy']:.0%} "
              f"coverage={summ['coverage']:.0%}")
        print(f"[learn] graph it: python tools/plot_metrics.py")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
