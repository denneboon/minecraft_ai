#!/usr/bin/env python3
"""Offline self-test for agents/crafting.py — the Crafter picks the right
recipe + emits the right controller calls (hotkey move vs split) from a
mock inventory, without a live game."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

from knowledge.catalog import Catalog
from vision.mc_assets import MCAssets
from agents.crafting import Crafter, inventory_counts, find_item_slot

_fails = 0
def ok(m): print(f"  [OK]  {m}")
def bad(m):
    global _fails; _fails += 1; print(f"  [FAIL] {m}")


class _Ctl:
    """Records the high-level controller calls the Crafter makes."""
    def __init__(self, snap):
        self._snap = snap
        self.calls = []
    def read(self): return self._snap
    def move_stack(self, src, dst, snap): self.calls.append(("move_stack", src, dst))
    def distribute_one(self, src, cells): self.calls.append(("distribute_one", src, tuple(cells)))
    def take_result(self): self.calls.append(("take_result",))
    def shift_left_click(self, slot): self.calls.append(("shift_clear", slot))


def _snap(items):
    """items: {slot_name: (item_id, count)}."""
    slots = {}
    for name, (it, ct) in items.items():
        slots[name] = SimpleNamespace(item=it, count=ct)
    return SimpleNamespace(slots=slots)


def main() -> int:
    print("=" * 56); print(" crafting — offline self-test"); print("=" * 56)
    a = MCAssets.load()
    cat = Catalog.load(a)

    # helpers
    print("\n[1] inventory_counts + find_item_slot")
    snap = _snap({"inv_0": ("minecraft:oak_log", 5), "hotbar_2": ("minecraft:iron_axe", 1),
                  "inv_7": ("minecraft:oak_log", 3)})
    cnt = inventory_counts(snap)
    (ok if cnt.get("minecraft:oak_log") == 8 else bad)(f"counts logs across slots ({cnt.get('minecraft:oak_log')})")
    (ok if find_item_slot(snap, "minecraft:oak_log") == "inv_0" else bad)(
        "find_item_slot picks the fullest slot")

    # planks: single ingredient -> whole-stack number-key move + take
    print("\n[2] craft planks (single cell -> hotkey move)")
    ctl = _Ctl(_snap({"inv_0": ("minecraft:oak_log", 5)}))
    okc, msg = Crafter(ctl, a, cat).craft("minecraft:oak_planks")
    mv = [c for c in ctl.calls if c[0] == "move_stack"]
    (ok if okc and mv and mv[0][1] == "inv_0" and mv[0][2] == "craft_in_0" else bad)(
        f"moves log stack into craft_in_0 ({msg})")
    (ok if any(c[0] == "take_result" for c in ctl.calls) else bad)("takes the result")
    (ok if not any(c[0] == "distribute_one" for c in ctl.calls) else bad)(
        "no split needed (hotkey-only)")

    # crafting_table: 4 planks, one per cell -> distribute_one (split)
    print("\n[3] craft crafting_table (4 cells -> split one per cell)")
    ctl = _Ctl(_snap({"inv_0": ("minecraft:oak_planks", 8)}))
    okc, msg = Crafter(ctl, a, cat).craft("minecraft:crafting_table")
    dist = [c for c in ctl.calls if c[0] == "distribute_one"]
    (ok if okc and dist and len(dist[0][2]) == 4 else bad)(
        f"distributes 1 plank into 4 cells ({msg})")
    (ok if dist and dist[0][2] == ("craft_in_0", "craft_in_1", "craft_in_2", "craft_in_3")
     else bad)("the four 2x2 cells, row-major")

    # wooden_pickaxe: 3x3 -> needs table, refuse cleanly
    print("\n[4] wooden_pickaxe -> needs table (refused)")
    ctl = _Ctl(_snap({"inv_0": ("minecraft:oak_planks", 9), "inv_1": ("minecraft:stick", 4)}))
    okc, msg = Crafter(ctl, a, cat).craft("minecraft:wooden_pickaxe")
    (ok if not okc and "table" in msg else bad)(f"refuses 3x3 in inventory ({msg})")

    # can't craft without ingredients
    print("\n[5] missing ingredients -> clean failure")
    ctl = _Ctl(_snap({"inv_0": ("minecraft:cobblestone", 9)}))
    okc, msg = Crafter(ctl, a, cat).craft("minecraft:oak_planks")
    (ok if not okc and not any(c[0] in ("move_stack", "distribute_one") for c in ctl.calls)
     else bad)(f"no moves when uncraftable ({msg})")

    print("\n" + ("ALL CRAFTING TESTS PASSED" if not _fails
                  else f"{_fails} CHECK(S) FAILED"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
