#!/usr/bin/env python3
"""Offline self-test for the inventory ledger + count-aware crafting:
InventoryMemory (remember, assess, deficit) and Crafter.ensure (skip when we
already have enough; otherwise craft ONLY the shortfall)."""
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
from agents.inventory_memory import InventoryMemory, inventory_counts
from agents.crafting import Crafter

_fails = 0
def ok(m): print(f"  [OK]  {m}")
def bad(m):
    global _fails; _fails += 1; print(f"  [FAIL] {m}")


def _snap(items):
    """items: {slot_name: (item_id, count)}."""
    slots = {n: SimpleNamespace(item=it, count=ct) for n, (it, ct) in items.items()}
    return SimpleNamespace(slots=slots)


def _snap_from_counts(counts):
    """One inv_ slot per item id (count). Hotbar items use hotbar_ slots."""
    slots, i, h = {}, 0, 0
    for it, ct in counts.items():
        if ct <= 0:
            continue
        slots[f"inv_{i}"] = SimpleNamespace(item=it, count=ct); i += 1
    return SimpleNamespace(slots=slots)


class _EnsureCtl:
    """Mock controller: read() reflects evolving counts; a craft (take_result)
    bumps the target by per_craft. Records opens/reads/crafts."""
    def __init__(self, counts, target, per_craft):
        self.counts = dict(counts)
        self.target = target
        self.per_craft = per_craft
        self.opens = 0; self.reads = 0; self.crafts = 0
        self.container = ""
    def open_inventory(self): self.opens += 1
    def close(self): pass
    def read(self, stop_when=None):
        self.reads += 1
        return _snap_from_counts(self.counts)
    def move_stack(self, src, dst, snap): pass
    def distribute_one(self, src, cells): pass
    def take_result(self):
        self.counts[self.target] = self.counts.get(self.target, 0) + self.per_craft
        self.crafts += 1
    def shift_left_click(self, slot): pass


def main() -> int:
    print("=" * 56); print(" inventory memory — offline self-test"); print("=" * 56)
    TABLE = "minecraft:crafting_table"
    PLANK = "minecraft:oak_planks"

    # 1. observe + counts + hotbar split.
    print("\n[1] observe / count / hotbar_count / deficit")
    m = InventoryMemory()
    m.observe(_snap({"inv_0": (PLANK, 8), "hotbar_3": (TABLE, 2), "inv_5": (TABLE, 1)}))
    (ok if m.count(TABLE) == 3 else bad)(f"counts a table across slots ({m.count(TABLE)})")
    (ok if m.hotbar_count(TABLE) == 2 else bad)(f"hotbar portion ({m.hotbar_count(TABLE)})")
    (ok if m.deficit(TABLE, 5) == 2 and m.deficit(TABLE, 3) == 0 else bad)("deficit math")

    # 2. assess: have / hotbar-have / check.
    print("\n[2] assess (have / hotbar / check)")
    (ok if m.assess(TABLE, 3)[0] == "have" else bad)("remembered >= wanted -> have")
    (ok if m.assess(TABLE, 2)[0] == "have" else bad)("hotbar alone covers it -> have")
    (ok if m.assess(TABLE, 4)[0] == "check" else bad)("short of wanted -> check")
    (ok if m.assess("minecraft:stick", 1)[0] == "check" else bad)("unknown item -> check")
    fresh = InventoryMemory()
    (ok if fresh.assess(TABLE, 1)[0] == "check" else bad)("never read -> check")

    # 3. note_delta keeps the ledger in sync without a re-read.
    print("\n[3] note_delta")
    m.note_delta(TABLE, +1)
    (ok if m.count(TABLE) == 4 else bad)(f"crafted +1 -> {m.count(TABLE)}")
    m.note_delta(TABLE, -2)
    (ok if m.count(TABLE) == 2 else bad)(f"used -2 -> {m.count(TABLE)}")

    a = MCAssets.load(); cat = Catalog.load(a)

    # 4. ensure: memory already has enough -> DON'T open, DON'T craft.
    print("\n[4] ensure: already have enough -> no open, no craft")
    mem = InventoryMemory()
    mem.observe(_snap({"hotbar_3": (TABLE, 2)}))
    ctl = _EnsureCtl({PLANK: 64, TABLE: 2}, TABLE, per_craft=1)
    okc, msg = Crafter(ctl, a, cat).ensure(TABLE, 2, memory=mem)
    (ok if okc and ctl.opens == 0 and ctl.crafts == 0 else bad)(
        f"have 2, want 2 -> skip ({msg}; opens={ctl.opens} crafts={ctl.crafts})")

    # 5. ensure: short -> open + craft ONLY the deficit.
    print("\n[5] ensure: craft only the shortfall")
    mem = InventoryMemory()                       # empty memory -> must check
    ctl = _EnsureCtl({PLANK: 64, TABLE: 1}, TABLE, per_craft=1)
    okc, msg = Crafter(ctl, a, cat).ensure(TABLE, 3, memory=mem)
    (ok if okc and ctl.opens == 1 and ctl.crafts == 2 else bad)(
        f"have 1, want 3 -> craft 2 ({msg}; crafts={ctl.crafts})")
    (ok if mem.count(TABLE) == 3 else bad)(f"memory updated to 3 ({mem.count(TABLE)})")

    # 6. ensure: inventory turns out to already have enough (memory was stale).
    print("\n[6] ensure: open finds enough -> no craft")
    mem = InventoryMemory()                       # nothing remembered
    ctl = _EnsureCtl({PLANK: 64, TABLE: 5}, TABLE, per_craft=1)
    okc, msg = Crafter(ctl, a, cat).ensure(TABLE, 4, memory=mem)
    (ok if okc and ctl.opens == 1 and ctl.crafts == 0 else bad)(
        f"opened, found 5 >= 4 -> no craft ({msg}; crafts={ctl.crafts})")

    # 7. ensure: count<=0 is a no-op.
    print("\n[7] ensure: want 0 -> no-op")
    ctl = _EnsureCtl({}, TABLE, per_craft=1)
    okc, _ = Crafter(ctl, a, cat).ensure(TABLE, 0)
    (ok if okc and ctl.opens == 0 else bad)("want 0 -> nothing happens")

    print("\n" + ("ALL INVENTORY-MEMORY TESTS PASSED" if not _fails
                  else f"{_fails} CHECK(S) FAILED"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
