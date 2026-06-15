#!/usr/bin/env python3
"""Offline self-test for hotbar auto-arrange: best-item-per-role ranking
(material tier / food quality / block count) and the inventory-swap arranger."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

from knowledge.item_roles import best_item_for_role, material_rank
from agents.hotbar_arranger import arrange_hotbar, DEFAULT_ARRANGE

_fails = 0
def ok(m): print(f"  [OK]  {m}")
def bad(m):
    global _fails; _fails += 1; print(f"  [FAIL] {m}")


class _MockCat:
    """Minimal Catalog stand-in: only is_block_item matters here."""
    _BLOCKS = {"cobblestone", "dirt", "oak_planks", "stone", "crafting_table"}
    def item(self, iid):
        stem = iid.split(":")[-1]
        return SimpleNamespace(tags=set(), equipment_slot=None,
                               is_block_item=(stem in self._BLOCKS))


class _MockCtl:
    """Simulates the inventory: read() returns the current slots; number_swap
    actually swaps two slots' contents so re-reads see the move. Slot values are
    ``(item_id, count)`` (a bare id is taken as count 1)."""
    def __init__(self, slots):
        self.slots = {n: (v if isinstance(v, tuple) else (v, 1 if v else 0))
                      for n, v in slots.items()}
        self.swaps = []
        self.closed = 0
    def item(self, name): return self.slots.get(name, (None, 0))[0]
    def open_inventory(self): pass
    def close(self): self.closed += 1
    def read(self, stop_when=None):
        sl = {n: SimpleNamespace(item=it, count=c)
              for n, (it, c) in self.slots.items()}
        return SimpleNamespace(slots=sl)
    def number_swap(self, src, n):
        tgt = f"hotbar_{n - 1}"
        self.swaps.append((src, tgt))
        self.slots[src], self.slots[tgt] = (self.slots.get(tgt, (None, 0)),
                                            self.slots.get(src, (None, 0)))


def main() -> int:
    print("=" * 56); print(" hotbar auto-arrange — offline self-test"); print("=" * 56)
    cat = _MockCat()

    # 1. material tier ordering.
    print("\n[1] best tool by material tier (netherite>diamond>iron>copper>gold>stone>wood)")
    inv = {"minecraft:wooden_sword": 1, "minecraft:iron_sword": 3,
           "minecraft:diamond_sword": 1, "minecraft:stone_sword": 1}
    (ok if best_item_for_role(inv, "sword") == "minecraft:diamond_sword" else bad)(
        f"sword -> diamond ({best_item_for_role(inv, 'sword')})")
    inv2 = {"minecraft:netherite_pickaxe": 1, "minecraft:diamond_pickaxe": 1}
    (ok if best_item_for_role(inv2, "pickaxe") == "minecraft:netherite_pickaxe" else bad)(
        "pickaxe -> netherite over diamond")
    inv3 = {"minecraft:golden_axe": 5, "minecraft:iron_axe": 1}
    (ok if best_item_for_role(inv3, "axe") == "minecraft:iron_axe" else bad)(
        "axe -> iron over gold even with fewer (tier beats count)")
    (ok if material_rank("minecraft:copper_shovel") == 4
        and material_rank("minecraft:stone_hoe") == 2 else bad)(
        "copper=4, stone=2 ranks")

    # 2. food quality + blocks-by-count.
    print("\n[2] best food + best block")
    food = {"minecraft:rotten_flesh": 30, "minecraft:bread": 4,
            "minecraft:cooked_beef": 2, "minecraft:golden_carrot": 1}
    (ok if best_item_for_role(food, "food") == "minecraft:golden_carrot" else bad)(
        f"food -> golden_carrot ({best_item_for_role(food, 'food')})")
    blocks = {"minecraft:dirt": 12, "minecraft:cobblestone": 64}
    (ok if best_item_for_role(blocks, "blocks", cat) == "minecraft:cobblestone" else bad)(
        "blocks -> the biggest stack (cobblestone x64)")
    (ok if best_item_for_role({"minecraft:stick": 9}, "sword") is None else bad)(
        "no sword present -> None")

    # 3. the arranger moves the best of each role into its slot, re-reading
    #    between moves so a displaced item is re-found.
    print("\n[3] arrange_hotbar swaps the best of each role into its slot")
    slots = {f"hotbar_{i}": None for i in range(9)}
    slots["hotbar_0"] = ("minecraft:dirt", 5)            # wrong item in sword slot
    slots["hotbar_5"] = ("minecraft:diamond_pickaxe", 1)  # best pickaxe, wrong slot
    slots.update({
        "inv_0": ("minecraft:wooden_sword", 1),
        "inv_1": ("minecraft:netherite_sword", 1),        # best sword, buried
        "inv_2": ("minecraft:iron_axe", 1),
        "inv_3": ("minecraft:cooked_beef", 3),
        "inv_4": ("minecraft:cobblestone", 64),           # biggest block stack
    })
    ctl = _MockCtl(slots)
    placed = arrange_hotbar(ctl, catalog=cat, settle=0.0)
    checks = {
        "hotbar_0": "minecraft:netherite_sword",   # slot 1 sword
        "hotbar_1": "minecraft:diamond_pickaxe",   # slot 2 pickaxe
        "hotbar_2": "minecraft:iron_axe",          # slot 3 axe
        "hotbar_5": "minecraft:cobblestone",       # slot 6 blocks
        "hotbar_6": "minecraft:cooked_beef",       # slot 7 food
    }
    for slot, want in checks.items():
        got = ctl.item(slot)
        (ok if got == want else bad)(f"{slot} = {want.split(':')[-1]} (got {got})")
    (ok if placed.get("sword") == "minecraft:netherite_sword" else bad)(
        "returns what it placed")
    (ok if ctl.closed == 1 else bad)("closes the inventory exactly once")

    # 4. a slot_roles map that REPEATS a role must not pull the item back out
    #    of the slot it was just placed in (would corrupt the earlier slot).
    print("\n[4] duplicate-role layout keeps the first placement intact")
    slots2 = {f"hotbar_{i}": None for i in range(9)}
    slots2.update({
        "hotbar_7": ("minecraft:dirt", 50),    # in a non-role slot
        "inv_0": ("minecraft:cobblestone", 64),  # the single best block stack
    })
    ctl2 = _MockCtl(slots2)
    arrange_hotbar(ctl2, slot_roles={5: "blocks", 6: "blocks"},
                   catalog=cat, settle=0.0)
    (ok if ctl2.item("hotbar_4") == "minecraft:cobblestone" else bad)(
        f"slot 5 keeps cobblestone (got {ctl2.item('hotbar_4')})")
    (ok if ctl2.item("hotbar_5") != "minecraft:cobblestone" else bad)(
        "slot 6 did NOT steal it back out of slot 5")

    print("\n" + ("ALL HOTBAR-ARRANGE TESTS PASSED" if not _fails
                  else f"{_fails} CHECK(S) FAILED"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
