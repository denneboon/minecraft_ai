# tools/world_map_view.py
"""
Live top-down view of the AI's world map.

Captures Minecraft frames, runs the WorldPerception pipeline tick-by-
tick, and renders the resulting :class:`WorldMap` to an OpenCV window
that updates in real time. While the window is open you can also see:

  * the player's position + yaw arrow,
  * the crosshair-targeted voxel (red box) when F3 "Looking at block"
    is visible,
  * stats: blocks observed, sample-store size, FPS, corrections.

Self-improvement loop
---------------------
Whenever the F3 overlay shows "Targeted Block" (i.e., player is
holding F3, or has Debug Options → Looking at block set to Always),
the perception layer auto-samples the crosshair patch and saves it to
``data/training/world_samples/<block_id>/``. Over time, the
sample-NN recogniser learns *this specific Minecraft install's* block
appearance — including biome tint, night-time darkness, weather, and
shader effects — without anyone writing training code.

Read-only with respect to the game: this tool does NOT move the mouse
or press keys. Safe to run while you play.

Keys (inside the OpenCV window)
-------------------------------
  q / ESC  — quit
  + / -    — zoom in / out
  s        — save a PNG snapshot to data/calibration/world_map_view.png
  c        — clear the in-memory WorldMap (samples on disk stay)
  r        — reload the sample recogniser (picks up disk samples)

Usage
-----
    python tools/world_map_view.py
    python tools/world_map_view.py --fps 5 --no-window   # headless PNG-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import cv2
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


OUT_DIR = os.path.join(ROOT, "data", "calibration")
os.makedirs(OUT_DIR, exist_ok=True)
SNAPSHOT_PATH = os.path.join(OUT_DIR, "world_map_view.png")


def _print(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def _load_settings() -> Dict[str, Any]:
    try:
        import yaml
    except Exception:
        return {"vision": {"world": {"enabled": True}}}
    settings_path = os.path.join(ROOT, "config", "settings.yaml")
    if not os.path.isfile(settings_path):
        return {"vision": {"world": {"enabled": True}}}
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f) or {}
    settings.setdefault("vision", {}).setdefault("world", {})["enabled"] = True
    return settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=float, default=8.0,
                        help="target perception FPS (default 8)")
    parser.add_argument("--zoom", type=float, default=1.0,
                        help="initial map zoom (1.0 = ~64 blocks across)")
    parser.add_argument("--no-window", action="store_true",
                        help="skip the OpenCV window; only write PNG snapshots")
    parser.add_argument("--snapshot-every", type=float, default=2.0,
                        help="seconds between PNG snapshots (0 = never)")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="auto-stop after N seconds (0 = run forever)")
    args = parser.parse_args()

    settings = _load_settings()

    # --- Imports kept inside main so import errors surface nicely. ---
    from utils.focus       import _find_minecraft_hwnd, activate_minecraft
    from vision.capture    import Capture, CaptureConfig
    from vision.ocr        import build_f3_reader
    from vision.world      import (
        WorldMapRenderer, MapRenderConfig,
        IsoWorldRenderer, IsoRenderConfig,
        build_world_perception,
    )

    # --- Ensure Minecraft is running and focused. -----------------
    try:
        wins = _find_minecraft_hwnd()
    except Exception as e:
        _print(f"[ERR] could not enumerate windows: {e}")
        return 2
    if not wins:
        _print("[ERR] Minecraft is not running. Start the 'Minecraft AI' "
               "instance first.")
        return 2
    hwnd, rect = wins[0]
    w, h = rect[2] - rect[0], rect[3] - rect[1]
    _print(f"[ok] Minecraft hwnd={hwnd} window={w}x{h}")

    _print("[..] Focusing Minecraft (read-only; we don't drive inputs)...")
    if activate_minecraft():
        time.sleep(0.4)
    else:
        _print("[WARN] Could not focus Minecraft. Click it within 3 s...")
        time.sleep(3.0)

    # --- Build perception ----------------------------------------
    cap = Capture(config=CaptureConfig(
        hwnd=hwnd,
        window_title_query="Minecraft",
        max_fps=max(2.0, args.fps),
        track_window_each_frame=True,
        strict_window_find=False,
        clamp_to_monitor=True,
        use_client_area=True,
        name="world_map_view",
    ))
    cap.start()
    time.sleep(0.25)

    f3_reader = build_f3_reader(settings)
    _print(f"[ok] F3 reader backend={f3_reader.backend}")

    wp = build_world_perception(settings)
    _print(f"[ok] WorldPerception built with "
           f"{wp.block_classifier.template_count()} block signatures")
    if wp.sample_store is not None:
        n_blocks = wp.sample_store.block_count()
        n_total  = wp.sample_store.total_samples()
        _print(f"[ok] Sample store at {wp.sample_store.root}")
        _print(f"     {n_total} samples covering {n_blocks} block ids")
    else:
        _print("[..] Sample store disabled in settings")

    # --- Renderers + window --------------------------------------
    top_renderer = WorldMapRenderer(MapRenderConfig(zoom=args.zoom))
    iso_renderer = IsoWorldRenderer(IsoRenderConfig(zoom=args.zoom))
    window_name = "Minecraft AI – World Map (iso 3D + top-down)"
    if not args.no_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 600)

    # --- Main loop ----------------------------------------------
    period = 1.0 / max(1e-3, args.fps)
    last_snapshot = 0.0
    t_start = time.perf_counter()
    n_ticks = 0
    fps_smoothed = 0.0
    last_tick_t  = time.perf_counter()

    _print("[..] Live view running. Press q in the window to quit. "
           "Hold F3 in-game to feed labels to the sample store.")

    try:
        while True:
            t_loop_start = time.perf_counter()

            try:
                frame = cap.get_frame()
            except Exception as e:
                _print(f"[WARN] frame capture failed: {e}")
                time.sleep(period)
                continue

            f3 = f3_reader.read(frame)
            try:
                wf = wp.update(frame, f3)
            except Exception as e:
                _print(f"[WARN] perception update failed: {e}")
                time.sleep(period)
                continue
            n_ticks += 1

            # Smoothed FPS for the header.
            now = time.perf_counter()
            inst_fps = 1.0 / max(1e-3, now - last_tick_t)
            last_tick_t = now
            fps_smoothed = (0.85 * fps_smoothed + 0.15 * inst_fps
                            if fps_smoothed > 0 else inst_fps)

            target_voxel = (wf.looking_at.pos
                            if wf.looking_at is not None else None)

            # Stats line under the pose.
            stats = wp.stats()
            ss = stats.get("sample_store", {})
            cur_dim = wp.world_map.current_dimension()
            dim_stats = stats['map']['dimensions'].get(cur_dim, {})
            n_solid = sum(
                1 for _ in wp.world_map.iter_solid_blocks(dimension=cur_dim)
            )
            n_air = dim_stats.get('blocks', 0) - n_solid
            extra: List[str] = [
                f"ticks={n_ticks}",
                f"fps={fps_smoothed:4.1f}",
                f"solid={n_solid}",
                f"air={n_air}",
                f"curi={stats.get('curiosity_size', 0)}",
                f"conf'd={stats.get('confirmed_count', 0)}",
                f"samp={ss.get('total_samples', 0)}/{ss.get('blocks_known', 0)}b",
                f"fix={stats['corrections_seen']}",
            ]
            if wf.looking_at is not None and wf.looking_at.block_id:
                extra.append(f"->{wf.looking_at.block_id.split(':')[-1]}")

            iso_img = iso_renderer.render(wp.world_map, wf.pose,
                                           target_voxel=target_voxel,
                                           extra_lines=extra)
            top_img = top_renderer.render(wp.world_map, wf.pose,
                                           target_voxel=target_voxel,
                                           extra_lines=extra)

            # Side-by-side: iso 3D (wider) on the left, top-down on the right.
            H_iso = iso_img.shape[0]
            H_top = top_img.shape[0]
            H_out = max(H_iso, H_top)
            def _pad(arr, target_h):
                if arr.shape[0] == target_h:
                    return arr
                pad = np.full((target_h - arr.shape[0], arr.shape[1], 3),
                              iso_renderer.cfg.background, dtype=np.uint8)
                return np.vstack([arr, pad])
            iso_p = _pad(iso_img, H_out)
            top_p = _pad(top_img, H_out)
            img = np.hstack([iso_p, top_p])

            # Show + handle key input.
            if not args.no_window:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow(window_name, bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
                elif key == ord('+') or key == ord('='):
                    top_renderer.cfg.zoom = min(8.0, top_renderer.cfg.zoom * 1.25)
                    iso_renderer.cfg.zoom = min(8.0, iso_renderer.cfg.zoom * 1.25)
                    _print(f"[..] zoom -> {top_renderer.cfg.zoom:.2f}")
                elif key == ord('-') or key == ord('_'):
                    top_renderer.cfg.zoom = max(0.25, top_renderer.cfg.zoom / 1.25)
                    iso_renderer.cfg.zoom = max(0.25, iso_renderer.cfg.zoom / 1.25)
                    _print(f"[..] zoom -> {top_renderer.cfg.zoom:.2f}")
                elif key == ord('s'):
                    cv2.imwrite(SNAPSHOT_PATH, bgr)
                    _print(f"[..] snapshot -> {SNAPSHOT_PATH}")
                elif key == ord('c'):
                    wp.world_map.clear()
                    _print("[..] world map cleared")
                elif key == ord('r'):
                    cls = wp.block_classifier
                    rfn = getattr(cls, "reload_samples", None)
                    if callable(rfn):
                        rfn()
                        _print("[..] sample recogniser reloaded")
                elif key == ord('3'):
                    # Toggle: show only iso 3D (no top-down).
                    # Implemented by zeroing the top renderer canvas next
                    # tick — set a flag the renderer respects. Simplest
                    # path is to flip the zoom of top to 0 → renderer
                    # produces a tiny strip.
                    top_renderer.cfg.canvas_size_px = 80
                    _print("[..] minimised top-down panel")

            # Periodic snapshot to disk (also works in --no-window mode).
            if args.snapshot_every > 0:
                if (now - last_snapshot) >= args.snapshot_every:
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(SNAPSHOT_PATH, bgr)
                    last_snapshot = now

            # Auto-stop after duration.
            if args.duration > 0 and (now - t_start) >= args.duration:
                break

            # Sleep what's left of this tick's budget.
            elapsed = time.perf_counter() - t_loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

    finally:
        cap.stop()
        if not args.no_window:
            cv2.destroyAllWindows()

    # --- Final summary -------------------------------------------
    stats = wp.stats()
    _print("")
    _print(f"--- session over after {n_ticks} ticks "
           f"({time.perf_counter() - t_start:.1f}s) ---")
    _print(json.dumps(stats, indent=2))
    if os.path.isfile(SNAPSHOT_PATH):
        _print(f"final snapshot -> {SNAPSHOT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
