#!/usr/bin/env python3
# tools/test_inventory.py
"""
Capture (or load) a Minecraft frame, run the full inventory pipeline on
it, and write a labelled debug image + a structured JSON snapshot.

What it does
------------
1. Find the Minecraft window (or load a frame from ``--from-file``).
2. Detect which menu is currently open (player inventory, chest, …) so
   we don't blindly try to read the player-inventory layout when the
   user actually has a chest open.
3. Build the full ``InventoryReader`` pipeline (item recognizer + block
   isometric icons + stack-count OCR + durability + glint).
4. Read the snapshot, print a nicely formatted slot table, and save:
     * ``data/calibration/inventory_test.png``  — labelled overlay
     * ``data/calibration/inventory_test.json`` — structured snapshot

This tool is read-only — no clicks, no keypresses.

Examples
--------
    # Live, with a 4-second countdown so you can press E:
    python tools/test_inventory.py

    # Re-analyse the last saved frame (handy after tweaking the
    # recognizer without having to re-grab from the live game):
    python tools/test_inventory.py --from-file data/calibration/pipeline_test_capture.png

    # Force the chest layout (useful when menu auto-detect fails):
    python tools/test_inventory.py --container chest_single
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import yaml

from vision.capture import Capture, CaptureConfig
from vision.inventory import build_inventory_reader
from vision.inventory_layout import slot_rects, available_layouts
from vision.menu_detect import build_menu_detector
from vision.mc_assets import MCAssets
from vision.mcfont import ensure_font_cache
from vision.block_icons import ensure_block_icon_cache
from vision.tooltip import build_tooltip_reader
from agents.inventory_inspector import InventoryInspector, InspectorConfig
from control.mouse import Mouse, MouseConfig
from utils.focus import _find_minecraft_hwnd, activate_minecraft


# Map the menu detector's labels onto our layout keys. The detector
# tells us "inventory" / "chest" / "furnace" / "crafting_table" — we
# need the exact layout name from ``inventory_layout.available_layouts``.
_MENU_TO_LAYOUT = {
    "inventory":           "player_inventory",
    "crafting_table":      "crafting_table",
    "chest":               "chest_single",
    "furnace":             "furnace",
    "creative_inventory":  "player_inventory",   # close enough for now
}


def _menu_to_layout(menu_name: str) -> str:
    return _MENU_TO_LAYOUT.get(menu_name, "player_inventory")


# ---------------------------------------------------------------------------

def _open_capture(countdown: int) -> Tuple[Capture, int]:
    """
    Find Minecraft, focus it, count down so the user can open their
    inventory, and return a STARTED Capture instance + the hwnd. The
    caller is responsible for ``cap.stop()`` when done.
    """
    wins = _find_minecraft_hwnd()
    if not wins:
        raise SystemExit("[invtest] Minecraft is not running.")
    hwnd = wins[0][0]
    print(f"[invtest] Using Minecraft hwnd={hwnd}")
    activate_minecraft(maximize=True)
    time.sleep(0.3)

    if countdown > 0:
        print(f"[invtest] Open the container you want to read in Minecraft. "
              f"Capturing in {countdown} seconds…")
        for n in range(countdown, 0, -1):
            print(f"  {n}…")
            time.sleep(1.0)

    cap = Capture(CaptureConfig(hwnd=hwnd, use_client_area=True,
                                track_window_each_frame=True,
                                max_fps=30, name="invtest"))
    cap.start()
    return cap, hwnd


def _grab_frame(use_file: str | None, countdown: int
                ) -> Tuple[np.ndarray, Optional[Capture], Optional[int]]:
    """
    Return ``(frame_rgb, capture_or_None, hwnd_or_None)``. When
    ``use_file`` is given we just load the saved PNG and return None
    for the capture + hwnd. Otherwise we open a live capture and
    return it so the caller can grab additional frames (used by the
    Phase-2 hover loop).
    """
    if use_file:
        bgr = cv2.imread(use_file, cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"[invtest] Could not read frame: {use_file}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None, None

    cap, hwnd = _open_capture(countdown)
    frame = cap.get_frame()
    return frame, cap, hwnd


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Capture an inventory frame and run the full pipeline.")
    p.add_argument("--countdown", type=int, default=4,
                   help="Seconds before capture (for switching to MC).")
    p.add_argument("--from-file", default=None,
                   help="Path to a saved PNG to analyse instead of live capture.")
    p.add_argument("--container", default=None,
                   choices=available_layouts(),
                   help="Override the auto-detected container layout.")
    p.add_argument("--no-menu-check", action="store_true",
                   help="Skip the menu-detector pre-check.")
    p.add_argument("--save-crops", action="store_true",
                   help="Save every per-slot crop as a 6×-upscaled PNG "
                        "under data/calibration/slot_crops/ to debug "
                        "alignment + matching problems.")
    p.add_argument("--hover-unknowns", action="store_true",
                   help="Phase 2: after vision recognition, hover over "
                        "each unknown slot and OCR the tooltip to "
                        "extract its minecraft:<id>. Moves the cursor — "
                        "don't run while you're trying to do something else.")
    args = p.parse_args()

    settings_path = ROOT / "config" / "settings.yaml"
    with open(settings_path, encoding="utf-8") as f:
        settings = yaml.safe_load(f) or {}
    ui_scale = int((settings.get("capture") or {}).get("ui_scale", 2))

    frame, live_capture, live_hwnd = _grab_frame(args.from_file, args.countdown)
    try:
        return _run_pipeline(args, settings, ui_scale, frame,
                             live_capture, live_hwnd)
    finally:
        if live_capture is not None:
            try:
                live_capture.stop()
            except Exception:
                pass


def _run_pipeline(args, settings, ui_scale, frame, live_capture, live_hwnd):
    """Everything after the initial grab — split out so main() can put
    the live capture into a try/finally without indenting the world."""
    h, w = frame.shape[:2]
    print(f"[invtest] Frame: {w}×{h}  ui_scale={ui_scale}")

    # The Phase-2 hover step needs the live capture and the window's
    # desktop-pixel origin. From-file mode can't hover (no live MC) and
    # silently downgrades to vision-only.
    if args.hover_unknowns and live_capture is None:
        print("[invtest][WARN] --hover-unknowns ignored with --from-file "
              "(no live game to hover over).")
        args.hover_unknowns = False

    # ── Menu detection ────────────────────────────────────────────────
    layout_name: str
    if args.container:
        layout_name = args.container
        print(f"[invtest] Container override: {layout_name}")
    elif args.no_menu_check:
        layout_name = "player_inventory"
        print(f"[invtest] --no-menu-check → assuming {layout_name}")
    else:
        font_templates = ensure_font_cache(
            str(ROOT / "data" / "calibration" / "mc_font.npz"))
        menu_det = build_menu_detector(settings, font_templates)
        det = menu_det.detect(frame)
        if not det.open:
            print("[invtest][WARN] No menu detected. Defaulting to "
                  "player_inventory — open a container in MC and re-run if "
                  "the readings look wrong.")
            layout_name = "player_inventory"
        else:
            layout_name = _menu_to_layout(det.menu)
            print(f"[invtest] Menu detector: {det.menu!r} "
                  f"(matched '{det.matched_keyword}') → layout={layout_name}")

    # ── Asset / template loading ──────────────────────────────────────
    print("[invtest] Loading Minecraft assets…")
    assets = MCAssets.load()
    print(f"[invtest] Assets version: {assets.version}")

    print("[invtest] Building isometric block-icon cache (first run is slow)…")
    icon_cache = ensure_block_icon_cache(assets,
                                         progress=lambda i, n:
                                         print(f"  rendered {i}/{n}") if i % 200 == 0 else None)
    manifest = icon_cache.load_manifest() or {}
    print(f"[invtest]   rendered={len(manifest.get('rendered') or [])} "
          f"skipped={len(manifest.get('skipped') or [])}")

    print("[invtest] Building inventory reader (loading templates)…")
    t0 = time.perf_counter()
    reader = build_inventory_reader(settings, assets=assets)
    dt = time.perf_counter() - t0
    print(f"[invtest]   templates={reader.recognizer.template_count()} "
          f"kinds={reader.recognizer.template_kinds()} (built in {dt:.2f}s)")
    n_samples = reader.sample_recog.sample_count()
    n_items   = reader.sample_recog.item_count()
    print(f"[invtest]   sample store: {n_samples} samples across "
          f"{n_items} distinct items")

    # ── Read snapshot (Phase 1: pure vision) ──────────────────────────
    print(f"[invtest] Reading container {layout_name!r}…")
    t0 = time.perf_counter()
    snap = reader.read(frame, container=layout_name)
    dt = time.perf_counter() - t0
    n_phase1_unknown = sum(1 for s in snap.slots.values() if s.is_unknown)
    print(f"[invtest]   read in {dt * 1000:.1f} ms "
          f"({n_phase1_unknown} unknown of {len(snap.slots)} slots)")

    # ── Phase 2: hover over unknowns and OCR their tooltips ───────────
    inspection_results = {}
    hover_mouse: Optional[Mouse] = None
    if args.hover_unknowns and live_capture is not None:
        print(f"[invtest] Phase 2: hovering {n_phase1_unknown} unknown slots…")
        tooltip_reader = build_tooltip_reader(settings, assets=assets)
        # Spin up a Mouse instance just for the hover phase so reaches
        # use the eased minimum-jerk path instead of teleporting. No
        # gate is wired in here — this is a standalone tool, not the
        # agent loop — and the easing curve is the same one the live
        # agent uses.
        hover_mouse = Mouse(config=MouseConfig())
        hover_mouse.start()
        try:
            # Share the InventoryReader's sample store so hovers feed
            # the Phase-3 NN matcher for the next run.
            inspector = InventoryInspector(
                tooltip_reader=tooltip_reader,
                capture=live_capture,
                mouse=hover_mouse,
                sample_store=reader.sample_store,
                config=InspectorConfig(hover_settle_ms=220),
            )
            window_origin = live_capture.window_origin()
            t0 = time.perf_counter()
            inspection_results = inspector.resolve_unknowns(
                snap,
                window_origin=window_origin,
                container=layout_name,
                restore_cursor=True,
                pre_hover_frame=frame,
            )
            dt = time.perf_counter() - t0
            print(f"[invtest]   resolved {len(inspection_results)} slots "
                  f"via tooltip OCR in {dt:.2f}s")
            for slot_name, result in inspection_results.items():
                short = (result.item_id or "?").replace("minecraft:", "")
                # Use the canonical display name from the lang cache —
                # the OCR'd display-name from the tooltip is often
                # garbled because MC colour-codes it (italics for
                # renamed items, rarity colours for special ones). The
                # minecraft:<id> line is the reliable source of truth.
                canonical = (assets.display_name(result.item_id)
                             if result.item_id else None)
                dn = f" ({canonical})" if canonical else ""
                print(f"             {slot_name:<14} → {short}{dn}")
        finally:
            # Always tear the hover mouse down so the velocity worker
            # thread exits even if the inspection raises mid-loop.
            try:
                hover_mouse.stop()
            except Exception:
                pass

    # ── Pretty print ──────────────────────────────────────────────────
    # SRC column shows which stage identified the slot:
    #   sample = NN match against the Phase-3 sample store
    #   vision = synthetic template match
    #   hover  = Phase-2 tooltip OCR ground truth
    print()
    print(f"  {'SLOT':<14}  {'SRC':<6}  {'ITEM':<32}  {'CT':>3}  "
          f"{'CONF':>5}  {'DUR':>5}  {'GLINT':<5}  {'2ND-GUESS':<28}")
    print(f"  {'-' * 14}  {'-' * 6}  {'-' * 32}  {'-' * 3}  "
          f"{'-' * 5}  {'-' * 5}  {'-' * 5}  {'-' * 28}")
    n_filled = n_unknown = 0
    for slot_name, content in snap.slots.items():
        if content.is_empty:
            continue
        dur = "-" if content.durability is None else f"{content.durability:.2f}"
        glint = "yes" if content.enchanted else ""
        src = content.source or "?"
        if content.is_unknown:
            n_unknown += 1
            short = "?"
            guess = (content.second or "").replace("minecraft:", "") if content.second else ""
            guess = f"(closest: {guess} @ {content.score:.1f})" if guess else f"(score {content.score:.1f})"
            print(f"  {slot_name:<14}  {src:<6}  {short:<32}  {content.count:>3}  "
                  f"{content.confidence:>5.2f}  {dur:>5}  {glint:<5}  {guess:<28}")
        else:
            n_filled += 1
            short = content.item.replace("minecraft:", "")
            second = (content.second or "").replace("minecraft:", "")
            print(f"  {slot_name:<14}  {src:<6}  {short:<32}  {content.count:>3}  "
                  f"{content.confidence:>5.2f}  {dur:>5}  {glint:<5}  {second:<28}")
    print()
    summary = snap.items_summary()
    if summary:
        print(f"[invtest] Items summary ({len(summary)} types, "
              f"{n_filled} identified / {n_unknown} unknown / "
              f"{len(snap.slots) - n_filled - n_unknown} empty of "
              f"{len(snap.slots)} total):")
        for item, total in sorted(summary.items(), key=lambda kv: -kv[1]):
            short = item.replace("minecraft:", "")
            name = assets.display_name(item) or short
            print(f"          {total:>4}×  {short:<32}  ({name})")
    else:
        print("[invtest] Inventory is empty.")

    # ── Save overlay PNG ──────────────────────────────────────────────
    canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for name, slot in slot_rects(frame.shape, layout=layout_name,
                                 ui_scale=ui_scale).items():
        content = snap.slots.get(name)
        if content is None or content.is_empty:
            colour = (90, 90, 90)
            label  = None
        elif content.is_unknown:
            colour = (0, 0, 220)             # red for unknown
            label  = "?"
        elif content.confidence > 0.4:
            colour = (0, 200, 0)             # green for confident
            label  = content.item.replace("minecraft:", "")
        else:
            colour = (0, 165, 255)           # amber for low-confidence
            label  = content.item.replace("minecraft:", "")
        cv2.rectangle(canvas, (slot.x, slot.y),
                      (slot.x + slot.w, slot.y + slot.h), colour, 1)
        if label is not None:
            if len(label) > 12:
                label = label[:12] + "…"
            cv2.putText(canvas, label, (slot.x, slot.y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA)
            if content.count > 1:
                cv2.putText(canvas, f"x{content.count}",
                            (slot.x, slot.y + slot.h + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 220, 220),
                            1, cv2.LINE_AA)

    out_dir = ROOT / "data" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "inventory_test.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"\n[invtest] Wrote labelled overlay → {out_path}")

    if args.save_crops:
        # Save each slot's crop side-by-side with the *template* the
        # recognizer matched it against, so misalignment / mis-matches
        # are obvious at a glance.
        crops_dir = out_dir / "slot_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        # Wipe stale crops from previous runs first.
        for old in crops_dir.glob("*.png"):
            old.unlink()
        import numpy as np
        for name, slot in slot_rects(frame.shape, layout=layout_name,
                                     ui_scale=ui_scale).items():
            crop = frame[slot.y:slot.y + slot.h, slot.x:slot.x + slot.w]
            if crop.size == 0:
                continue
            # 6× upscale for visibility.
            big = cv2.resize(crop, (slot.w * 6, slot.h * 6),
                             interpolation=cv2.INTER_NEAREST)
            content = snap.slots.get(name)
            # Right pane: template the recognizer's best match used.
            tmpl_pane = np.full_like(big, 64)
            template_id = (content.item if content and content.item
                           else content.second if content and content.second
                           else None)
            if template_id is not None:
                # Find this template in the recognizer's library.
                if template_id in reader.recognizer._names:
                    idx = reader.recognizer._names.index(template_id)
                    tpl = reader.recognizer._tpl_rgb[idx]
                    tpl_big = cv2.resize(tpl, (slot.w * 6, slot.h * 6),
                                         interpolation=cv2.INTER_NEAREST)
                    tmpl_pane = tpl_big
            side_by_side = np.concatenate([big, tmpl_pane], axis=1)
            # Label
            tag = (content.item.replace("minecraft:", "") if content and content.item
                   else (f"?{content.second.replace('minecraft:','')}"
                         if content and content.second else "(empty)"))
            cv2.putText(side_by_side, f"{name}: {tag}",
                        (4, big.shape[0] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.imwrite(str(crops_dir / f"{name}.png"),
                        cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
        print(f"[invtest] Wrote {len(snap.slots)} per-slot crops → {crops_dir}")

    json_path = out_dir / "inventory_test.json"
    json_path.write_text(json.dumps(snap.to_dict(), indent=2), encoding="utf-8")
    print(f"[invtest] Wrote structured snapshot → {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
