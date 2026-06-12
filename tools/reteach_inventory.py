#!/usr/bin/env python3
"""
Re-teach / de-poison the inventory sample store. Opens the inventory, and for
each non-empty slot hovers it, reads the TOOLTIP (ground truth), and compares
that to what the sample-NN recogniser currently guesses. Where they disagree,
a hover-to-learn mistake poisoned the store (e.g. a coal icon saved under
``mud``), so the NN reports the wrong id forever.

    python tools/reteach_inventory.py            # DRY RUN — just report
    python tools/reteach_inventory.py --apply    # fix: relabel + purge poison

``--apply`` calls SampleStore.relabel(tooltip_id, crop): saves the crop under
the tooltip-OCR'd id AND removes the same crop from any wrong-id folder.
Tooltip ids are validated against the asset catalog so a garbled OCR can't
re-poison. Needs MC running + focused.
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import main as M
from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

from utils.focus import _find_minecraft_hwnd, activate_minecraft
from control.input_gate import InputGate
from vision.capture import Capture, CaptureConfig
from vision.inventory import build_inventory_reader
from vision.inventory_layout import slot_rects
from vision.tooltip import build_tooltip_reader
from agents.inventory_inspector import InventoryInspector, InspectorConfig
from knowledge.catalog import Catalog
from vision.mc_assets import MCAssets


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    apply = "--apply" in argv

    wins = _find_minecraft_hwnd()
    if not wins:
        print("[reteach] Minecraft not found"); return 2
    hwnd = wins[0][0]; activate_minecraft(); time.sleep(0.4)
    settings = M._load_yaml(M.SETTINGS_PATH)
    keymap = M._flatten_keymap_for_keyboard(M._load_json(M.KEYMAP_PATH))
    ui_scale = int((settings.get("capture") or {}).get("ui_scale", 2))
    gate = InputGate()
    safety = M.build_safety(settings, gate=gate)
    mouse = M.build_mouse(settings, gate=gate)
    kb = M.build_keyboard(settings, keymap, gate=gate)
    capture = Capture(CaptureConfig(hwnd=hwnd, use_client_area=True, threaded=True))
    safety.start(); mouse.start(); kb.start(); capture.start(); time.sleep(0.3)
    a = MCAssets.load(); cat = Catalog.load(a)
    reader = build_inventory_reader(settings, assets=a)
    tooltip = build_tooltip_reader(settings, assets=a)
    store = getattr(reader, "sample_store", None)
    menu_detector = M.build_menu_detector_default(settings)
    try:
        origin = capture.window_origin()
    except Exception:
        origin = (0, 0)
    inspector = InventoryInspector(tooltip, capture, mouse=mouse, gate=gate,
                                   config=InspectorConfig())

    def _valid(item_id):
        if not item_id:
            return False
        try:
            return (cat.item(item_id) is not None) or (cat.block(item_id) is not None)
        except Exception:
            return False

    try:
        if not M.ensure_playing(capture, menu_detector, kb):
            print("[reteach] paused — click into MC"); return 1
        kb.tap("e"); time.sleep(0.45)
        pre = capture.get_frame()
        snap = reader.read(pre, container="player_inventory")
        rects = slot_rects(pre.shape, layout="player_inventory", ui_scale=ui_scale)

        print(f"[reteach] {'APPLY' if apply else 'DRY RUN'} — hover each slot, "
              f"compare tooltip vs NN guess:")
        print(f"    {'slot':12} {'NN guess':18} {'tooltip':18} verdict")
        conflicts = reinforced = bad = 0
        for name in sorted(rects):
            if not name.startswith(("inv_", "hotbar_")):
                continue
            sc = snap.slots.get(name)
            if sc is None or getattr(sc, "is_empty", True):
                continue
            guess = getattr(sc, "item", None)
            res = inspector.resolve_slot(rects[name], window_origin=origin)
            tip = getattr(res, "item_id", None) if res else None
            slot = rects[name]
            crop = pre[slot.y:slot.y + slot.h, slot.x:slot.x + slot.w]
            g = str(guess).split(":")[-1] if guess else "-"
            t = str(tip).split(":")[-1] if tip else "(no tooltip)"
            if not _valid(tip):
                verdict = "skip (bad/garbled tooltip)"; bad += 1
            elif guess == tip:
                verdict = "ok"; reinforced += 1
            else:
                verdict = f"POISON: {g} -> {t}"; conflicts += 1
                if apply and store is not None:
                    n = store.relabel(tip, crop)
                    verdict += f"  (relabelled, purged {n})"
            print(f"    {name:12} {g:18} {t:18} {verdict}")
        kb.tap("e"); time.sleep(0.2)
        print(f"[reteach] {conflicts} poisoned, {reinforced} correct, {bad} unreadable.")
        if conflicts and not apply:
            print("[reteach] re-run with --apply to fix the poisoned samples.")
        if apply and store is not None and hasattr(reader, "sample_recog"):
            reader.sample_recog.reload()
            print("[reteach] recogniser reloaded.")
        return 0
    finally:
        try: kb.stop()
        except Exception: pass
        try: mouse.release_all() if hasattr(mouse, "release_all") else None
        except Exception: pass
        capture.stop()
        try: safety.stop()
        except Exception: pass


if __name__ == "__main__":
    raise SystemExit(main())
