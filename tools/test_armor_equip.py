#!/usr/bin/env python3
"""Offline self-test for armour auto-equip: protection-tier ranking and the
equip behaviour (shift-click into an empty slot, 3-click upgrade swap, and
no-op when the worn piece is already as good)."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

from knowledge.item_roles import (best_armor_for_slot, armor_material_rank,
                                   armor_slot_of)
from agents.armor_equip import equip_best_armor

_fails = 0
def ok(m): print(f"  [OK]  {m}")
def bad(m):
    global _fails; _fails += 1; print(f"  [FAIL] {m}")


class _MockCat:
    """Catalog stand-in: maps a few ids to their equipment_slot."""
    _SLOT = {
        "diamond_helmet": "head", "iron_helmet": "head", "turtle_helmet": "head",
        "leather_helmet": "head", "carved_pumpkin": "head",
        "netherite_chestplate": "chest", "iron_chestplate": "chest",
        "elytra": "chest",
        "diamond_leggings": "legs", "chainmail_leggings": "legs",
        "gold_boots": "feet", "leather_boots": "feet",
    }
    def item(self, iid):
        return SimpleNamespace(equipment_slot=self._SLOT.get(iid.split(":")[-1]),
                               tags=set(), is_block_item=False)


class _MockCtl:
    """Faithful inventory sim: a cursor + slots; left_click swaps cursor<->slot;
    shift_left_click auto-equips a loose armour piece into its EMPTY body slot
    (vanilla behaviour). Slot values are ``(item_id, count)``."""
    def __init__(self, slots, cat):
        self.slots = {n: (v if isinstance(v, tuple) else (v, 1 if v else 0))
                      for n, v in slots.items()}
        self.cat = cat
        self.cursor = (None, 0)
        self.calls = []
        self.closed = 0

    def item(self, name): return self.slots.get(name, (None, 0))[0]
    def open_inventory(self): pass
    def close(self): self.closed += 1

    def read(self, stop_when=None):
        sl = {n: SimpleNamespace(item=it, count=c)
              for n, (it, c) in self.slots.items()}
        armor = {k: sl[k] for k in sl if k.startswith("armor_")}
        return SimpleNamespace(slots=sl, armor=armor)

    def hover(self, name): pass

    def left_click(self, name):
        self.calls.append(("left", name))
        cur = self.cursor
        self.cursor = self.slots.get(name, (None, 0))
        self.slots[name] = cur

    def shift_left_click(self, name):
        self.calls.append(("shift", name))
        it, c = self.slots.get(name, (None, 0))
        if not it:
            return
        slot = armor_slot_of(it, self.cat)
        key = f"armor_{slot}" if slot else None
        if key and self.slots.get(key, (None, 0))[0] is None:   # only if empty
            self.slots[key] = (it, c)
            self.slots[name] = (None, 0)


def main() -> int:
    print("=" * 56); print(" armour auto-equip — offline self-test"); print("=" * 56)
    cat = _MockCat()

    # 1. ranking.
    print("\n[1] best piece by protection tier")
    inv = {"minecraft:leather_helmet": 1, "minecraft:diamond_helmet": 1,
           "minecraft:iron_helmet": 3}
    (ok if best_armor_for_slot(inv, "head", cat) == "minecraft:diamond_helmet" else bad)(
        f"head -> diamond ({best_armor_for_slot(inv, 'head', cat)})")
    (ok if armor_material_rank("minecraft:turtle_helmet") == 4 else bad)(
        "turtle_helmet ranks at iron tier (4)")
    (ok if best_armor_for_slot({"minecraft:carved_pumpkin": 1, "minecraft:elytra": 1},
                               "head", cat) is None else bad)(
        "carved_pumpkin/elytra are NOT auto-equipped (rank 0)")
    (ok if armor_slot_of("minecraft:netherite_chestplate", cat) == "chest" else bad)(
        "netherite_chestplate -> chest slot")

    # 2. equip into EMPTY slots (shift-click auto-equip).
    print("\n[2] empty slots: shift-click auto-equip the best of each")
    slots = {f"armor_{s}": None for s in ("head", "chest", "legs", "feet")}
    slots.update({
        "inv_0": ("minecraft:iron_helmet", 1),
        "inv_1": ("minecraft:diamond_helmet", 1),     # better head, buried
        "inv_2": ("minecraft:iron_chestplate", 1),
        "inv_3": ("minecraft:gold_boots", 1),
    })
    ctl = _MockCtl(slots, cat)
    worn = equip_best_armor(ctl, catalog=cat, settle=0.0)
    (ok if ctl.item("armor_head") == "minecraft:diamond_helmet" else bad)(
        f"head = diamond_helmet (got {ctl.item('armor_head')})")
    (ok if ctl.item("armor_chest") == "minecraft:iron_chestplate" else bad)(
        f"chest = iron_chestplate (got {ctl.item('armor_chest')})")
    (ok if ctl.item("armor_feet") == "minecraft:gold_boots" else bad)(
        f"feet = gold_boots (got {ctl.item('armor_feet')})")
    (ok if ctl.item("armor_legs") is None else bad)("legs stays empty (none carried)")
    (ok if worn.get("head") == "minecraft:diamond_helmet" else bad)("returns what it wore")

    # 3. UPGRADE swap: a better piece replaces a worn worse one (3-click), and
    #    the displaced old piece ends up back in the source slot.
    print("\n[3] upgrade an already-worn worse piece")
    slots2 = {f"armor_{s}": None for s in ("head", "chest", "legs", "feet")}
    slots2["armor_head"] = ("minecraft:iron_helmet", 1)        # already worn
    slots2["inv_0"] = ("minecraft:diamond_helmet", 1)          # better, loose
    ctl2 = _MockCtl(slots2, cat)
    equip_best_armor(ctl2, catalog=cat, settle=0.0)
    (ok if ctl2.item("armor_head") == "minecraft:diamond_helmet" else bad)(
        f"head upgraded to diamond (got {ctl2.item('armor_head')})")
    (ok if ctl2.item("inv_0") == "minecraft:iron_helmet" else bad)(
        f"old iron_helmet displaced back into inv_0 (got {ctl2.item('inv_0')})")

    # 4. no-op when the worn piece is already as good or better.
    print("\n[4] already wearing better -> no change")
    slots3 = {f"armor_{s}": None for s in ("head", "chest", "legs", "feet")}
    slots3["armor_head"] = ("minecraft:diamond_helmet", 1)
    slots3["inv_0"] = ("minecraft:iron_helmet", 1)
    ctl3 = _MockCtl(slots3, cat)
    worn3 = equip_best_armor(ctl3, catalog=cat, settle=0.0)
    (ok if ctl3.item("armor_head") == "minecraft:diamond_helmet" else bad)(
        "keeps the diamond helmet")
    (ok if not any(c for c in ctl3.calls if c[1] == "armor_head") else bad)(
        "no click touched the head slot")
    (ok if "head" not in worn3 else bad)("reports no head change")
    (ok if ctl3.closed == 1 else bad)("closes the inventory exactly once")

    print("\n" + ("ALL ARMOUR-EQUIP TESTS PASSED" if not _fails
                  else f"{_fails} CHECK(S) FAILED"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
