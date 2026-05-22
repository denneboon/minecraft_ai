#!/usr/bin/env python3
# tools/test_inventory.py
"""
Capture the current MC frame, draw inventory slot rectangles on it,
identify every slot's item, and save a labelled debug image.

Usage
-----
1. Open Minecraft, press E to open your inventory.
2. Run ``python tools/test_inventory.py``
3. Inspect ``data/calibration/inventory_test.png`` —
   each slot will be outlined and labelled with the recognised item
   (and its confidence).

This script does NOT modify game state; it only reads.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import yaml

from vision.capture import Capture, CaptureConfig
from vision.inventory import slot_rects, build_item_recognizer
from vision.menu_detect import build_menu_detector
from vision.mc_assets import MCAssets
from vision.mcfont import ensure_font_cache
from utils.focus import _find_minecraft_hwnd, activate_minecraft


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Capture an inventory frame and identify every slot."
    )
    p.add_argument("--countdown", type=int, default=4,
                   help="Seconds to wait after focusing MC so you can press E.")
    args = p.parse_args()

    wins = _find_minecraft_hwnd()
    if not wins:
        print("[ERROR] Minecraft is not running.")
        return 2
    hwnd = wins[0][0]
    print(f"[invtest] Using Minecraft hwnd={hwnd}")

    activate_minecraft(maximize=True)
    import time
    time.sleep(0.3)   # let the maximize animation settle

    if args.countdown > 0:
        print(f"[invtest] Press E in Minecraft to open your inventory. "
              f"Capturing in {args.countdown} seconds…")
        for n in range(args.countdown, 0, -1):
            print(f"  {n}…")
            time.sleep(1.0)

    with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    ui_scale = int((settings.get("capture") or {}).get("ui_scale", 2))

    cap = Capture(CaptureConfig(hwnd=hwnd, use_client_area=True,
                                track_window_each_frame=True,
                                max_fps=30, name="invtest"))
    cap.start()
    frame = cap.get_frame()
    cap.stop()
    h, w = frame.shape[:2]
    print(f"[invtest] Captured frame: {w}×{h}  ui_scale={ui_scale}")

    # Sanity: is the inventory actually open?
    templates = ensure_font_cache(str(ROOT / "data" / "calibration" / "mc_font.npz"))
    menu_det  = build_menu_detector(settings, templates)
    det = menu_det.detect(frame)
    if not det.open:
        print("[invtest][WARN] Menu detector says no menu is open. "
              "Press E in Minecraft to open the inventory, then re-run.")
    else:
        print(f"[invtest] Menu detector: {det.menu!r} "
              f"(matched '{det.matched_keyword}')")

    # Build the recogniser and read every slot.
    print("[invtest] Loading item-texture database…")
    assets = MCAssets.load()
    recog = build_item_recognizer(assets=assets)
    print(f"[invtest] Templates loaded: {recog.template_count()}")

    print("[invtest] Recognising slots…")
    results = recog.recognize_all(frame, ui_scale=ui_scale)

    # Print the non-empty slots.
    print()
    print(f"  {'SLOT':<14}  {'ITEM':<35}  {'CONF':>5}  {'SCORE':>6}  {'NAME':<25}")
    print(f"  {'-'*14}  {'-'*35}  {'-'*5}  {'-'*6}  {'-'*25}")
    for name, match in results.items():
        if match.item is None:
            continue
        display = assets.display_name(match.item) or "?"
        item_short = match.item.replace("minecraft:", "")
        print(f"  {name:<14}  {item_short:<35}  {match.confidence:>5.2f}  "
              f"{match.score:>6.2f}  {display:<25}")

    # Build a labelled debug image.
    canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for name, slot in slot_rects(frame.shape, ui_scale).items():
        match = results[name]
        # Outline colour: green for confident, yellow for low-confidence,
        # grey for empty.
        if match.item is None:
            colour = (90, 90, 90)
        elif match.confidence > 0.4:
            colour = (0, 200, 0)
        else:
            colour = (0, 165, 255)
        cv2.rectangle(canvas, (slot.x, slot.y),
                      (slot.x + slot.w, slot.y + slot.h), colour, 1)
        if match.item is not None:
            label = match.item.replace("minecraft:", "")
            label = label if len(label) <= 12 else label[:12] + "…"
            cv2.putText(canvas, label,
                        (slot.x, slot.y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA)

    out_dir = ROOT / "data" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "inventory_test.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"\n[invtest] Wrote labelled overlay → {out_path}")

    # Also dump structured JSON for inspection / future diffing.
    json_path = out_dir / "inventory_test.json"
    serial = {
        name: {
            "item":       m.item,
            "confidence": round(m.confidence, 3),
            "score":      round(m.score,      3),
        }
        for name, m in results.items()
    }
    json_path.write_text(json.dumps(serial, indent=2), encoding="utf-8")
    print(f"[invtest] Wrote structured JSON → {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
