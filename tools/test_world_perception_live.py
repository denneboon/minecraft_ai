# tools/test_world_perception_live.py
"""
Live smoke test for ``vision/world/`` — captures real frames from the
running Minecraft window and runs the full perception pipeline.

What it does
------------
1. Finds the Minecraft window (must already be running) and grabs a
   single screenshot via :class:`vision.capture.Capture`.
2. Runs :class:`vision.ocr.F3Reader` to extract the player's pose.
3. Runs :class:`vision.world.WorldPerception` for N ticks (default 10,
   one per second so the perception layer sees several different views
   if the player happens to pan the camera in between).
4. Saves diagnostics:
   * ``data/calibration/world_live_capture.png``   — first captured frame
   * ``data/calibration/world_live_pose.json``     — F3 pose dump
   * ``data/calibration/world_live_map.json``      — full WorldMap snapshot
   * ``data/calibration/world_live_summary.txt``   — human-readable summary

Does NOT drive any inputs. The script is read-only with respect to the
game — it never moves the mouse, presses a key, or sends chat. Safe to
run while you're playing.

Usage
-----
    python tools/test_world_perception_live.py
    python tools/test_world_perception_live.py --ticks 20 --interval 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

import cv2
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


OUT_DIR = os.path.join(ROOT, "data", "calibration")
os.makedirs(OUT_DIR, exist_ok=True)
CAPTURE_PATH = os.path.join(OUT_DIR, "world_live_capture.png")
POSE_PATH    = os.path.join(OUT_DIR, "world_live_pose.json")
MAP_PATH     = os.path.join(OUT_DIR, "world_live_map.json")
SUMMARY_PATH = os.path.join(OUT_DIR, "world_live_summary.txt")


def _print(msg: str) -> None:
    # cp1252-safe printing
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def _ensure_minecraft_running() -> bool:
    from utils.focus import _find_minecraft_hwnd
    try:
        wins = _find_minecraft_hwnd()
    except Exception as e:
        _print(f"[ERR] Could not enumerate windows: {e}")
        return False
    if not wins:
        _print("[ERR] Minecraft (javaw.exe) is not running.")
        _print("      Start the 'Minecraft AI' instance and try again.")
        return False
    hwnd, rect = wins[0]
    w, h = rect[2] - rect[0], rect[3] - rect[1]
    _print(f"[ok] Found Minecraft hwnd={hwnd} size={w}x{h}")
    return True


def _load_config() -> Dict[str, Any]:
    """Load settings.yaml; force-enable world perception for this run."""
    try:
        import yaml
    except Exception:
        _print("[WARN] PyYAML missing — using defaults")
        return {"vision": {"world": {"enabled": True}}}
    settings_path = os.path.join(ROOT, "config", "settings.yaml")
    if not os.path.isfile(settings_path):
        return {"vision": {"world": {"enabled": True}}}
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f) or {}
    settings.setdefault("vision", {}).setdefault("world", {})["enabled"] = True
    return settings


_OVERLAY_PATH = os.path.join(OUT_DIR, "world_live_overlay.png")


def _render_overlay(frame: np.ndarray, patches_info: list) -> None:
    """
    Draw rectangles + labels on the captured frame so the user can see
    where each patch landed and what the classifier called it.

    ``patches_info``: list of (x, y, w, h, guess, confidence).
    """
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
    for x, y, w, h, guess, conf in patches_info:
        ok = guess is not None
        color = (0, 200, 0) if ok else (40, 40, 200)   # BGR
        cv2.rectangle(bgr, (x, y), (x + w, y + h), color, 2)
        label = (f"{(guess or 'unk').split(':')[-1]} "
                 f"({conf:.2f})")
        cv2.putText(bgr, label, (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                    lineType=cv2.LINE_AA)
    cv2.imwrite(_OVERLAY_PATH, bgr)


def _debug_patch_dump(frame: np.ndarray, settings: Dict[str, Any],
                       f3: Any, W: int, H: int) -> None:
    """
    Sample the same patch grid WorldPerception sweeps, classify every
    patch ignoring the confidence floor, and print the top guess plus
    the patch's mean RGB. Diagnostic only — used to figure out why a
    classifier is rejecting in-world surfaces.
    """
    from vision.world.block_classifier import build_block_classifier
    cls = build_block_classifier(settings)
    cfg = ((settings or {}).get("vision", {}).get("world", {})) or {}
    grid = cfg.get("patch_grid_size", [6, 4])
    n_cols, n_rows = int(grid[0]), int(grid[1])
    psize  = int(cfg.get("patch_size_px", 24))
    ml = int(cfg.get("margin_left_px",   80))
    mr = int(cfg.get("margin_right_px",  80))
    mt = int(cfg.get("margin_top_px",    120))
    mb = int(cfg.get("margin_bottom_px", 180))
    x0, x1 = ml, W - mr
    y0, y1 = mt, H - mb
    _print(f"[debug] patch grid {n_cols}x{n_rows}, area "
           f"({x0},{y0}) -> ({x1},{y1}), patch={psize}px")
    half = psize // 2
    patches_info = []
    for j in range(n_rows):
        for i in range(n_cols):
            px = int(x0 + (i + 0.5) * (x1 - x0) / n_cols)
            py = int(y0 + (j + 0.5) * (y1 - y0) / n_rows)
            px0, py0 = max(0, px - half), max(0, py - half)
            px1, py1 = min(W, px0 + psize), min(H, py0 + psize)
            patch = frame[py0:py1, px0:px1]
            mean = patch.reshape(-1, 3).mean(axis=0)
            guess, conf = cls.classify(patch)
            patches_info.append((px0, py0, px1 - px0, py1 - py0, guess, conf))
            _print(f"  ({i},{j}) px=({px:4d},{py:4d})  "
                   f"meanRGB=({mean[0]:5.1f},{mean[1]:5.1f},{mean[2]:5.1f})  "
                   f"-> {guess or '(unknown)':30s} conf={conf:.2f}")
    _render_overlay(frame, patches_info)
    _print(f"     -> {_OVERLAY_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks",    type=int,   default=10,
                        help="number of perception updates to run")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="seconds between ticks")
    parser.add_argument("--debug-patches", action="store_true",
                        help="dump per-patch top guesses on the first frame "
                             "regardless of confidence (helps diagnose why "
                             "the classifier is rejecting in-world surfaces)")
    parser.add_argument("--self-teach", action="store_true",
                        help="SELF-TEACHING MONITOR: each tick, compare the "
                             "recogniser's guess for the block under the "
                             "crosshair against the F3 'Targeted Block' truth, "
                             "and report running accuracy as the model learns. "
                             "Point the crosshair at different blocks (F3 on) "
                             "and watch accuracy climb. Read-only / safe; the "
                             "perception layer auto-collects samples & retrains "
                             "in the background as it always does.")
    args = parser.parse_args()

    if not _ensure_minecraft_running():
        return 2

    settings = _load_config()

    from vision.capture       import Capture, CaptureConfig
    from vision.ocr           import build_f3_reader
    from vision.world         import build_world_perception
    from utils.focus          import _find_minecraft_hwnd, activate_minecraft

    # --- Focus Minecraft -----------------------------------------
    # Capture uses screen-region screenshotting, so MC must be the
    # foreground window or we'll grab whatever sits in front of it.
    _print("[..] Bringing Minecraft to the foreground (read-only — won't drive inputs)...")
    if activate_minecraft():
        _print("[ok] Minecraft focused")
        time.sleep(0.4)  # let the window settle to maximised
    else:
        _print("[WARN] Could not focus Minecraft programmatically. "
               "Manually click on it within 3 s and the capture will proceed.")
        time.sleep(3.0)

    # --- Capture --------------------------------------------------
    wins = _find_minecraft_hwnd()
    hwnd = wins[0][0] if wins else None
    cap = Capture(config=CaptureConfig(
        hwnd=hwnd,
        window_title_query="Minecraft",
        max_fps=10.0,
        track_window_each_frame=True,
        strict_window_find=False,
        clamp_to_monitor=True,
        use_client_area=True,
        name="world_live_capture",
    ))
    cap.start()
    time.sleep(0.25)

    try:
        frame = cap.get_frame()
    except Exception as e:
        _print(f"[ERR] First frame capture failed: {e}")
        cap.stop()
        return 2
    h, w = frame.shape[:2]
    _print(f"[ok] First capture: {w}x{h}")
    cv2.imwrite(CAPTURE_PATH, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    _print(f"     -> {CAPTURE_PATH}")

    # --- F3 OCR ---------------------------------------------------
    f3_reader = build_f3_reader(settings)
    _print(f"[ok] F3 reader backend = {f3_reader.backend}")
    f3 = f3_reader.read(frame)
    if f3.x is None:
        _print("[WARN] F3 OCR returned no position from the first frame.")
        _print("       Your debug-screen-text settings have player_position=ALWAYS,")
        _print("       which renders the XYZ line OVER the gameplay (no dark panel).")
        _print("       The F3Reader's panel pre-check is tuned for the F3-on dark panel.")
        _print("       Try this: press F3 once in-game so the full debug panel is visible,")
        _print("       leave it on, then re-run this script.")
    else:
        yaw = f3.yaw if f3.yaw is not None else float('nan')
        pitch = f3.pitch if f3.pitch is not None else float('nan')
        _print(f"[ok] Pose from F3: X={f3.x:.2f}  Y={f3.y:.2f}  Z={f3.z:.2f}  "
               f"yaw={yaw:.1f}  pitch={pitch:.1f}  facing={f3.facing_name}")
    with open(POSE_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "backend":     f3.backend,
            "x":           f3.x,
            "y":           f3.y,
            "z":           f3.z,
            "yaw":         f3.yaw,
            "pitch":       f3.pitch,
            "facing":      f3.facing_name,
            "dimension":   f3.dimension,
            "block":       [f3.block_x, f3.block_y, f3.block_z],
            "raw_text":    f3.raw_text.splitlines(),
        }, f, indent=2)
    _print(f"     -> {POSE_PATH}")

    # --- Debug per-patch dump (optional) -------------------------
    if args.debug_patches:
        _debug_patch_dump(frame, settings, f3, w, h)

    # --- Build perception ----------------------------------------
    try:
        wp = build_world_perception(settings)
    except Exception as e:
        _print(f"[ERR] Could not build WorldPerception: {e}")
        cap.stop()
        return 2
    _print(f"[ok] WorldPerception built with "
           f"{wp.block_classifier.template_count()} block signatures")

    # --- Self-teaching monitor state -----------------------------
    st_seen = st_hits = st_abstain = 0          # guess-vs-F3-truth tally
    st_recent: list = []                        # last-20 hit/miss (sliding acc)
    st_start_samples = None

    # --- Tick the perception loop --------------------------------
    n_with_pose = 0
    n_blocks_added = 0
    last_dt_ms = 0.0
    for tick in range(1, args.ticks + 1):
        try:
            frame = cap.get_frame()
        except Exception as e:
            _print(f"[WARN] tick {tick}: capture failed: {e}")
            continue
        f3 = f3_reader.read(frame)
        prev_n = wp.world_map.block_count()
        t0 = time.perf_counter()
        try:
            wf = wp.update(frame, f3)
        except Exception as e:
            _print(f"[WARN] tick {tick}: perception failed: {e}")
            continue
        last_dt_ms = (time.perf_counter() - t0) * 1000.0
        n_now = wp.world_map.block_count()
        n_blocks_added += max(0, n_now - prev_n)
        if wf.pose is not None:
            n_with_pose += 1
            looking = ""
            if wf.looking_at is not None:
                looking = (f"  looking_at={wf.looking_at.block_id}"
                           f" @ {wf.looking_at.pos}"
                           f"(conf={wf.looking_at.confidence:.2f})")
            _print(f"[tick {tick:2d}] pose=({wf.pose.x:.1f},{wf.pose.y:.1f},"
                   f"{wf.pose.z:.1f})  yaw={wf.pose.yaw:.0f} "
                   f"pitch={wf.pose.pitch:.0f}  "
                   f"map={n_now}  ({last_dt_ms:.1f} ms){looking}")
        else:
            _print(f"[tick {tick:2d}] no pose (F3 not visible in panel)  "
                   f"map={n_now}  ({last_dt_ms:.1f} ms)")

        # --- Self-teaching monitor: guess vs F3 truth -------------
        if args.self_teach and wf.looking_at is not None:
            truth = wf.looking_at.block_id
            patch = wp._crop_patch(frame, w // 2, h // 2,
                                   wp.cfg.patch_size_px)
            guess, conf = (None, 0.0)
            if patch is not None:
                guess, conf = wp.block_classifier.classify(patch)
            st_seen += 1
            if guess is None:
                st_abstain += 1
                mark = "abstain"
            elif guess == truth:
                st_hits += 1
                st_recent.append(1)
                mark = "HIT"
            else:
                st_recent.append(0)
                mark = f"miss (guessed {guess})"
            st_recent = st_recent[-20:]
            acc = st_hits / st_seen if st_seen else 0.0
            racc = (sum(st_recent) / len(st_recent)) if st_recent else 0.0
            cnn_status = ""
            cnn = getattr(wp.block_classifier, "cnn", None)
            if cnn is not None:
                if st_start_samples is None:
                    st_start_samples = cnn.sample_count()
                cnn_status = (f"  | samples={cnn.sample_count()} "
                              f"{'(training…)' if cnn._training else ''}")
            _print(f"    └─ SELF-TEACH: truth={truth} guess={guess} "
                   f"conf={conf:.2f} → {mark}  | acc={acc:.0%} "
                   f"recent20={racc:.0%}{cnn_status}")

        if tick < args.ticks:
            time.sleep(args.interval)

    cap.stop()

    # --- Self-teaching summary -----------------------------------
    if args.self_teach:
        acc = st_hits / st_seen if st_seen else 0.0
        cnn = getattr(wp.block_classifier, "cnn", None)
        grew = ""
        if cnn is not None and st_start_samples is not None:
            grew = (f"; sample store {st_start_samples} -> "
                    f"{cnn.sample_count()} (+{cnn.sample_count() - st_start_samples})")
        _print("")
        _print(f"[self-teach] {st_hits}/{st_seen} correct ({acc:.0%}), "
               f"{st_abstain} abstained{grew}")
        if cnn is not None:
            _print(f"[self-teach] recogniser: {cnn.status()}")
        _print("[self-teach] tip: run with more --ticks while panning F3 over "
               "many blocks; accuracy climbs as samples accrue & it retrains.")

    # --- Dump WorldMap snapshot ----------------------------------
    try:
        wp.world_map.dump_json(MAP_PATH)
        _print(f"[ok] WorldMap dumped -> {MAP_PATH}")
    except Exception as e:
        _print(f"[WARN] Could not dump WorldMap: {e}")

    # --- Summary -------------------------------------------------
    stats = wp.world_map.stats()
    lines = [
        "World perception live test summary",
        f"capture          : {CAPTURE_PATH}",
        f"frame size       : {w}x{h}",
        f"F3 backend       : {f3_reader.backend}",
        f"ticks ran        : {args.ticks}",
        f"ticks with pose  : {n_with_pose}",
        f"blocks added     : {n_blocks_added}",
        f"last update ms   : {last_dt_ms:.1f}",
        f"map stats        : {json.dumps(stats)}",
    ]
    for ln in lines:
        _print(ln)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    _print(f"     -> {SUMMARY_PATH}")

    # Soft success criterion: pipeline ran end-to-end without crashing.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
