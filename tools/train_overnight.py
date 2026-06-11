#!/usr/bin/env python3
"""
Overnight self-teaching trainer for the world block recogniser.

Runs the self-teaching loop CONTINUOUSLY for hours. It sweeps the view and
the CNN retrains itself in the background as samples land.

Two collection modes:
  * default (camera-only): relative mouse look, never moves/clicks, world
    untouched. Day/weather cycles vary the lighting on the SAME blocks.
  * ``--walk`` (recommended): between sweeps it WALKS to a new spot
    (edge-safe WalkToward, never clicks → world still untouched) so it
    gathers blocks from MANY positions / distances / angles, not just one
    standing point. POSITION diversity is what fixes per-block confusion
    (e.g. oak_log vs birch_log/dirt) that lighting-only sweeps can't —
    a stationary spin sees the same few trees forever.

The CNN retrains itself in the background as samples land.

Resilience (it runs unattended):
  * Focus loss  → pauses and waits (safety gate closed); resumes when MC
    is foreground again. Never burns the CPU spinning.
  * Any per-tick error → logged and skipped; the loop never dies.
  * Panic stop  → Ctrl+Shift+X / End / Pause, any time.
  * Checkpoints every --checkpoint-min: writes a metrics line (graphable
    via tools/plot_metrics.py) + a heartbeat status file
    (data/metrics/overnight_status.json) so progress survives a kill.

Usage:
    python tools/train_overnight.py                       # 8 hours
    python tools/train_overnight.py --minutes 60 --checkpoint-min 10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

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
from vision.world.sample_store import (
    WorldSampleStore, WorldSampleStoreConfig, default_world_sample_root,
)

# ── Panic poll (queue-independent) ─────────────────────────────────────
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


def _now() -> float:
    return time.time()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=480.0,
                    help="total run time in minutes (default 8 hours)")
    ap.add_argument("--checkpoint-min", type=float, default=15.0,
                    help="minutes between metric/heartbeat checkpoints")
    ap.add_argument("--pan", type=int, default=34,
                    help="relative yaw mouse-move per step (px)")
    ap.add_argument("--settle", type=float, default=0.22,
                    help="seconds to let the view settle after each pan")
    ap.add_argument("--per-block-cap", type=int, default=220,
                    help="max samples kept per block (raised from the default "
                         "80 so a full day/weather cycle of conditions is "
                         "retained for training)")
    ap.add_argument("--save-patches-per-checkpoint", type=int, default=6,
                    help="annotated screenshots to save each checkpoint (audit)")
    ap.add_argument("--walk", action="store_true",
                    help="WALK between sweeps to collect from many positions "
                         "(edge-safe, never clicks). The fix for per-block "
                         "confusion a stationary spin can't break.")
    ap.add_argument("--relocate-every", type=int, default=24,
                    help="[--walk] camera-sweep steps between relocations")
    ap.add_argument("--walk-ticks", type=int, default=40,
                    help="[--walk] control ticks to walk per relocation")
    ap.add_argument("--walk-dist", type=float, default=10.0,
                    help="[--walk] waypoint distance per relocation (blocks)")
    args = ap.parse_args(argv)

    wins = _find_minecraft_hwnd()
    if not wins:
        print("[overnight] Minecraft window not found — is it running?")
        return 2
    hwnd = wins[0][0]
    activate_minecraft()
    time.sleep(0.5)

    settings = M._load_yaml(M.SETTINGS_PATH)
    gate = InputGate()
    safety = M.build_safety(settings, gate=gate)
    mouse = M.build_mouse(settings, gate=gate)
    capture = Capture(CaptureConfig(hwnd=hwnd, threaded=True))
    safety.start(); mouse.start(); capture.start()
    f3_reader = build_f3_reader(settings)
    wp = build_world_perception(settings)

    # Raise the per-block sample cap so a whole day/weather cycle of
    # conditions is retained (the default 80 would evict last-few-minutes).
    big_store = WorldSampleStore(
        default_world_sample_root(),
        config=WorldSampleStoreConfig(max_samples_per_block=args.per_block_cap))
    wp.sample_store = big_store
    cnn = getattr(wp.block_classifier, "cnn", None)
    sample_nn = getattr(wp.block_classifier, "sample", None)
    if cnn is not None:
        cnn._store = big_store
    if sample_nn is not None:
        sample_nn._store = big_store

    # ── WALK mode: edge-safe relocation between sweeps (position diversity) ──
    keyboard = actions = None
    px_per_deg = float(M._get(settings, "agent.mouse_per_degree", 6.5) or 6.5)
    if args.walk:
        from control.action_wrapper import ActionWrapper
        keymap_flat = M._flatten_keymap_for_keyboard(M._load_json(M.KEYMAP_PATH))
        keyboard = M.build_keyboard(settings, keymap_flat, gate=gate)
        try:
            keyboard.start()
        except Exception:
            pass
        actions = ActionWrapper(keyboard, mouse, gate=gate)

    metrics_root = default_metrics_root()
    metrics_root.mkdir(parents=True, exist_ok=True)
    status_path = metrics_root / "overnight_status.json"
    log_path = metrics_root / f"overnight_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logf = open(log_path, "a", encoding="utf-8")

    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        try:
            logf.write(line + "\n"); logf.flush()
        except Exception:
            pass

    start = _now()
    deadline = start + args.minutes * 60.0
    ckpt_every = args.checkpoint_min * 60.0
    next_ckpt = start + ckpt_every
    start_samples_total = cnn.sample_count() if cnn is not None else big_store.total_samples()

    log(f"=== OVERNIGHT TRAIN start: {args.minutes:.0f} min, checkpoint "
        f"every {args.checkpoint_min:.0f} min, per-block cap {args.per_block_cap} ===")
    log(f"store={big_store.total_samples()} samples/{big_store.block_count()} blocks; "
        f"gate={'open' if safety.allow_input() else 'CLOSED (focus MC)'}")

    ckpt_idx = 0
    last_reactivate = 0.0
    chunk = SessionMetrics("train_overnight",
                           f"overnight_{time.strftime('%Y-%m-%d_%H-%M-%S')}_c{ckpt_idx}",
                           start, root=metrics_root)
    step = 0
    saved_this_ckpt = 0
    focus_paused = False
    errors = 0
    patch_dir = None
    steps_since_relocate = 0
    relocate_idx = 0

    def _relocate():
        """Walk (edge-safe, never clicks → world untouched) to a fanned-out
        new spot, sampling the whole way, so the recogniser sees blocks from
        many positions/distances/angles — the diversity an in-place spin
        can't get."""
        from agents.skills import WalkToward, SkillContext, SkillStatus
        from agents.treechop import _full_movement
        nonlocal relocate_idx
        relocate_idx += 1
        frame = capture.get_frame(); f3 = f3_reader.read(frame)
        wf = wp.update(frame, f3)
        if wf.pose is None:
            return 0
        p = wf.pose
        hdg = math.radians(float(p.yaw) + relocate_idx * 73.0)   # fan headings
        goal = (int(math.floor(p.x + args.walk_dist * (-math.sin(hdg)))),
                int(math.floor(p.y)),
                int(math.floor(p.z + args.walk_dist * math.cos(hdg))))
        walk = WalkToward(goal, arrive_dist=1.5)
        moved = 0
        reason = "ran out of ticks"
        for _ in range(args.walk_ticks):
            if _panic() or not safety.allow_input():
                reason = "panic/focus"; break
            frame = capture.get_frame(); f3 = f3_reader.read(frame)
            wf = wp.update(frame, f3)                # SAMPLE while moving
            if wf.pose is None:
                time.sleep(0.05); continue
            ctx = SkillContext(pose=wf.pose, world_map=wp.world_map,
                               looking_at=wf.looking_at, px_per_deg=px_per_deg,
                               dimension=getattr(wf.pose, "dimension", None))
            r = walk.tick(ctx)
            try:
                actions.set_movement_state(**_full_movement(r.action.movement))
            except Exception:
                pass
            if r.action.look_dx or r.action.look_dy:
                try:
                    mouse.move(int(r.action.look_dx), int(r.action.look_dy))
                except Exception:
                    pass
            moved += 1
            if r.status in (SkillStatus.DONE, SkillStatus.FAILED, SkillStatus.BLOCKED):
                reason = r.info; break
            time.sleep(0.05)
        try:
            actions.release_all_movement()
        except Exception:
            pass
        return moved, reason

    def write_status(extra=None):
        try:
            st = {
                "alive_ts_unix": round(_now(), 1),
                "elapsed_min": round((_now() - start) / 60.0, 1),
                "remaining_min": round(max(0.0, deadline - _now()) / 60.0, 1),
                "steps": step, "errors": errors,
                "checkpoints": ckpt_idx,
                "samples_total": big_store.total_samples(),
                "blocks": big_store.block_count(),
                "recognizer": cnn.status() if cnn is not None else "",
                "gate_open": bool(safety.allow_input()),
            }
            if extra:
                st.update(extra)
            status_path.write_text(json.dumps(st, indent=2), encoding="utf-8")
        except Exception:
            pass

    try:
        while _now() < deadline:
            if _panic():
                log("PANIC — stopping."); break

            # Pause cleanly if MC isn't focused (gate closed) — don't spin.
            if not safety.allow_input():
                if not focus_paused:
                    log("input gate CLOSED (MC not focused) — pausing; will "
                        "resume when MC is foreground.")
                    focus_paused = True
                # Unattended runs: gently try to bring MC back to the
                # foreground every ~20s so a transient focus loss (a popup,
                # an alt-tab) doesn't stall training for the whole night.
                if _now() - last_reactivate > 20.0:
                    last_reactivate = _now()
                    try:
                        activate_minecraft()
                    except Exception:
                        pass
                write_status({"paused": True})
                time.sleep(3.0)
                continue
            if focus_paused:
                log("MC focused again — resuming.")
                focus_paused = False

            try:
                dx = args.pan
                # Oscillate pitch through floor↔eye↔sky to see all visible
                # blocks; slow phase so the sweep covers the hemisphere.
                dy = int(20 * np.sin(step * 0.4))
                try:
                    mouse.move(dx, dy)
                except Exception:
                    pass
                time.sleep(args.settle)

                frame = capture.get_frame()
                f3 = f3_reader.read(frame)
                wf = wp.update(frame, f3)     # collects samples + retrains
                step += 1
                steps_since_relocate += 1

                # WALK mode: every N sweeps, relocate to a fresh spot so the
                # next sweep sees DIFFERENT blocks/distances (position
                # diversity is what an in-place spin can't get).
                if (args.walk and actions is not None
                        and steps_since_relocate >= args.relocate_every):
                    steps_since_relocate = 0
                    n, why = _relocate()
                    log(f"relocated: walked {n}/{args.walk_ticks} ticks "
                        f"(heading #{relocate_idx}, stop: {why})")

                if wf.pose is not None and wf.looking_at is not None:
                    truth = wf.looking_at.block_id
                    h, w = frame.shape[:2]
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
                    ns = cnn.sample_count() if cnn is not None else big_store.total_samples()
                    chunk.record(step=step, truth=truth, guess=guess,
                                 conf=conf, samples=ns)
                    # Save a few audit screenshots per checkpoint.
                    if (patch_dir is not None and patch is not None
                            and saved_this_ckpt < args.save_patches_per_checkpoint):
                        try:
                            from tools.learn_world_live import _save_annotated
                            mark = ("HIT" if guess == truth
                                    else "abstain" if guess is None else "miss")
                            _save_annotated(patch_dir, step, frame, patch,
                                            cap_px, truth, guess, conf, mark)
                            saved_this_ckpt += 1
                        except Exception:
                            pass
            except Exception as e:
                errors += 1
                if errors <= 20 or errors % 100 == 0:
                    log(f"step {step}: error (#{errors}): {e!r}")
                    if errors <= 3:
                        traceback.print_exc()
                time.sleep(0.2)

            # ── Checkpoint ──
            if _now() >= next_ckpt:
                summ = chunk.finalize(
                    no_target=0,
                    samples_before=start_samples_total if ckpt_idx == 0 else 0,
                    samples_after=big_store.total_samples(),
                    recognizer_status=cnn.status() if cnn is not None else "",
                    end_ts_unix=_now())
                comp = big_store.manifest()
                top = ", ".join(f"{k.split(':')[-1]}={v}"
                                for k, v in sorted(comp.items(), key=lambda x: -x[1])[:8])
                log(f"--- checkpoint {ckpt_idx} | {summ['steps_with_target']} guesses, "
                    f"acc={summ['accuracy']:.0%} cov={summ['coverage']:.0%} | "
                    f"store={big_store.total_samples()}/{big_store.block_count()} | "
                    f"elapsed={(_now()-start)/60:.0f}m | {cnn.status() if cnn else ''}")
                log(f"    blocks: {top}")
                write_status({"last_checkpoint_acc": summ["accuracy"],
                              "last_checkpoint_coverage": summ["coverage"]})
                ckpt_idx += 1
                saved_this_ckpt = 0
                patch_dir = os.path.join(
                    ROOT, "data", "debug", f"overnight_c{ckpt_idx}")
                try:
                    os.makedirs(patch_dir, exist_ok=True)
                except Exception:
                    patch_dir = None
                chunk = SessionMetrics(
                    "train_overnight",
                    f"overnight_{time.strftime('%Y-%m-%d_%H-%M-%S')}_c{ckpt_idx}",
                    _now(), root=metrics_root)
                next_ckpt += ckpt_every
            else:
                if step % 25 == 0:
                    write_status()
    finally:
        # Final checkpoint flush.
        try:
            chunk.finalize(no_target=0, samples_before=0,
                           samples_after=big_store.total_samples(),
                           recognizer_status=cnn.status() if cnn is not None else "",
                           end_ts_unix=_now())
        except Exception:
            pass
        try:
            if actions is not None:
                actions.release_all_movement()
        except Exception:
            pass
        try:
            mouse.release_all() if hasattr(mouse, "release_all") else None
        except Exception:
            pass
        try:
            if keyboard is not None:
                keyboard.stop()
        except Exception:
            pass
        capture.stop()
        try:
            safety.stop()
        except Exception:
            pass
        end_total = big_store.total_samples()
        log(f"=== DONE: {step} steps over {(_now()-start)/60:.1f} min, "
            f"{ckpt_idx} checkpoints, {errors} errors. "
            f"samples {start_samples_total} -> {end_total} "
            f"(+{end_total - start_samples_total}). ===")
        write_status({"finished": True})
        try:
            logf.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
