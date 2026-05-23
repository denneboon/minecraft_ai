#!/usr/bin/env python3
"""
Synthetic inventory test — paints known items into known slots on a
fake frame, then verifies the InventoryReader identifies each correctly.

Useful for regression testing the recognizer without needing Minecraft
to be open. Real-game capture is still gated on the user opening the
inventory live; this test exercises the pure-pixel path end-to-end.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from vision.inventory import build_inventory_reader
from vision.inventory_layout import slot_rects
from vision.mc_assets import MCAssets
from vision.block_icons import BlockIconRenderer


W, H = 1920, 1080
UI_SCALE = 2

# Slot → item id (no "minecraft:" prefix). Mix of items, blocks, tools,
# and armour so we exercise every template kind.
TESTS = [
    ("armor_head",   "diamond_helmet"),
    ("armor_chest",  "iron_chestplate"),
    ("armor_legs",   "leather_leggings"),
    ("armor_feet",   "netherite_boots"),
    ("offhand",      "shield"),
    ("craft_in_0",   "oak_planks"),
    ("craft_in_1",   "oak_planks"),
    ("craft_in_2",   "oak_planks"),
    ("craft_in_3",   "oak_planks"),
    ("craft_result", "crafting_table"),
    ("inv_0",        "cobblestone"),
    ("inv_1",        "dirt"),
    ("inv_2",        "sand"),
    ("inv_3",        "oak_log"),
    ("inv_4",        "apple"),
    ("inv_5",        "bread"),
    ("inv_6",        "diamond"),
    ("inv_7",        "iron_ingot"),
    ("inv_8",        "gold_ingot"),
    ("hotbar_0",     "diamond_sword"),
    ("hotbar_1",     "diamond_pickaxe"),
    ("hotbar_2",     "diamond_axe"),
    ("hotbar_3",     "diamond_shovel"),
    ("hotbar_4",     "bow"),
    ("hotbar_5",     "arrow"),
    ("hotbar_6",     "bucket"),
    ("hotbar_7",     "torch"),
    ("hotbar_8",     "cooked_beef"),
]


def resolve_template(assets, renderer, item: str):
    """
    Pick the template the GAME would draw for this item:
      * if there's a flat item PNG, that's what gets drawn into the slot
      * else fall back to the isometric block icon
    Returns (rgba uint8 16x16) or None.
    """
    tex = assets.item_texture(item)
    if tex is not None:
        # Normalise to RGBA 16x16 — item textures may be grayscale,
        # RGB, or RGBA, and animated ones come back as N×16 strips.
        if tex.ndim == 2:
            tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2RGBA)
        elif tex.ndim == 3 and tex.shape[2] == 3:
            tex = cv2.cvtColor(tex, cv2.COLOR_RGB2RGBA)
        elif tex.ndim != 3 or tex.shape[2] != 4:
            tex = None
        if tex is not None:
            h, w = tex.shape[:2]
            if h != w and w > 0 and h % w == 0:
                tex = tex[:w]
            if tex.shape[0] != 16:
                tex = cv2.resize(tex, (16, 16), interpolation=cv2.INTER_AREA)
            return tex
    icon = renderer.render(item)
    return icon


def paint_slot(frame: np.ndarray, rect, rgba16: np.ndarray) -> None:
    """Alpha-composite a 16×16 RGBA icon into the slot rect."""
    s = rect.w
    big = cv2.resize(rgba16, (s, s), interpolation=cv2.INTER_NEAREST)
    rgb = big[..., :3].astype(np.float32)
    a   = (big[..., 3].astype(np.float32) / 255.0)[..., None]
    bg  = frame[rect.y:rect.y + s, rect.x:rect.x + s].astype(np.float32)
    out = rgb * a + bg * (1.0 - a)
    frame[rect.y:rect.y + s, rect.x:rect.x + s] = out.astype(np.uint8)


def main() -> int:
    # Build a frame: MC's GUI slot background colour fills the whole
    # frame so the slot pixels behind transparent icons are correct.
    frame = np.full((H, W, 3), (139, 139, 139), dtype=np.uint8)

    rects = slot_rects(frame.shape, layout="player_inventory",
                       ui_scale=UI_SCALE)
    print(f"[synth] built {len(rects)} slot rects")

    assets = MCAssets.load()
    renderer = BlockIconRenderer(assets)

    placed = []
    for slot_name, item in TESTS:
        rect = rects.get(slot_name)
        if rect is None:
            continue
        rgba = resolve_template(assets, renderer, item)
        if rgba is None:
            print(f"  SKIP {slot_name}={item}: no template")
            continue
        paint_slot(frame, rect, rgba)
        placed.append((slot_name, item))

    out_path = ROOT / "data" / "calibration" / "synthetic_inventory.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"[synth] wrote {out_path} ({len(placed)} items)")

    reader = build_inventory_reader(assets=assets)
    print(f"[synth] reader templates={reader.recognizer.template_count()} "
          f"kinds={reader.recognizer.template_kinds()}")

    t0 = time.perf_counter()
    snap = reader.read(frame, container="player_inventory")
    dt  = time.perf_counter() - t0
    print(f"[synth] read in {dt*1000:.1f} ms "
          f"({len(snap.filled_slots())}/{len(snap.slots)} non-empty)")

    print()
    print(f"{'SLOT':<14} {'EXPECTED':<24} {'GOT':<24} {'CONF':>5} OK")
    print("-" * 80)
    correct = total = 0
    for slot_name, expected in TESTS:
        content = snap.slots.get(slot_name)
        if content is None:
            got, conf, ok = "MISSING", 0.0, False
        elif content.is_empty:
            got, conf, ok = "(empty)", 0.0, False
        elif content.item is None:
            # Recognizer saw something but wasn't confident enough.
            guess = (content.second or "?").replace("minecraft:", "")
            got = f"?(guess:{guess})"
            conf = content.confidence
            ok = guess == expected     # count it OK if the best guess is right
        else:
            got = content.item.replace("minecraft:", "")
            conf = content.confidence
            ok = got == expected
        total += 1
        if ok:
            correct += 1
        flag = "OK" if ok else "FAIL"
        print(f"{slot_name:<14} {expected:<24} {got:<24} {conf:>5.2f} {flag}")

    print()
    print(f"[synth] Accuracy: {correct}/{total} "
          f"= {100.0 * correct / max(1, total):.1f}%")
    return 0 if correct == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
