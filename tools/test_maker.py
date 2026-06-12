#!/usr/bin/env python3
"""Offline self-test for agents/maker.py — the Maker orchestrates
plan -> gather raw -> craft 2x2 chain -> table-craft the 3x3 target, count-aware,
using mock controller/crafter/callbacks (no live game)."""
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
from knowledge.recipes import recipe_for
from vision.mc_assets import MCAssets
from agents.inventory_memory import InventoryMemory
from agents.maker import Maker

_fails = 0
def ok(m): print(f"  [OK]  {m}")
def bad(m):
    global _fails; _fails += 1; print(f"  [FAIL] {m}")


def _snap(counts):
    slots = {}
    for i, (it, ct) in enumerate(counts.items()):
        if ct > 0:
            slots[f"inv_{i}"] = SimpleNamespace(item=it, count=ct)
    return SimpleNamespace(slots=slots)


class _World:
    """Shared mock state: item -> count, plus an ordered event log."""
    def __init__(self, counts=None):
        self.counts = dict(counts or {})
        self.events = []


class _Ctl:
    def __init__(self, world):
        self.w = world
    def open_inventory(self): self.w.events.append("open")
    def close(self): self.w.events.append("close")
    def read(self, stop_when=None): return _snap(self.w.counts)
    def read_hotbar(self): return _snap({})


class _Crafter:
    """Mock 2x2 crafter: a craft 'succeeds' and produces the recipe's output."""
    def __init__(self, world, assets, cat):
        self.w = world; self.a = assets; self.cat = cat
    def craft(self, result_id, *, snap=None):
        rec = recipe_for(result_id, self.a, self.cat)
        n = rec.result_count if rec else 1
        self.w.counts[result_id] = self.w.counts.get(result_id, 0) + n
        self.w.events.append(f"craft:{result_id.split(':')[-1]}")
        return True, f"crafted {result_id.split(':')[-1]} x{n}"


def _gather_fn(world):
    def g(item, qty):
        world.counts[item] = world.counts.get(item, 0) + qty
        world.events.append(f"gather:{item.split(':')[-1]}:{qty}")
        return True
    return g


def _table_fn(world):
    def t(target):
        world.counts[target] = world.counts.get(target, 0) + 1
        world.events.append(f"table:{target.split(':')[-1]}")
        return True, "table-crafted"
    return t


def _make(world, assets, cat, *, gather=True, table=True):
    mem = InventoryMemory()
    ctl = _Ctl(world)
    return Maker(ctl, _Crafter(world, assets, cat), mem, assets, cat,
                 gather_fn=_gather_fn(world) if gather else None,
                 table_craft_fn=_table_fn(world) if table else None,
                 log=lambda m: None)


def main() -> int:
    print("=" * 56); print(" maker — offline self-test"); print("=" * 56)
    a = MCAssets.load(); cat = Catalog.load(a)
    PICK = "minecraft:wooden_pickaxe"

    # 1. Make a wooden_pickaxe from NOTHING -> gather logs, craft chain, table-craft.
    print("\n[1] make wooden_pickaxe from scratch")
    w = _World()
    okc, msg = _make(w, a, cat).make(PICK, 1)
    (ok if okc else bad)(f"reports success ({msg})")
    (ok if w.counts.get(PICK, 0) >= 1 else bad)("a pickaxe now exists")
    (ok if any(e.startswith("gather:") and "log" in e for e in w.events) else bad)(
        "gathered logs")
    (ok if f"table:{PICK.split(':')[-1]}" in w.events else bad)("table-crafted the 3x3 pickaxe")
    # gather happened before the table-craft
    gi = next(i for i, e in enumerate(w.events) if e.startswith("gather:"))
    ti = next(i for i, e in enumerate(w.events) if e.startswith("table:"))
    (ok if gi < ti else bad)("gathered raw materials BEFORE table-crafting")

    # 2. Already have it -> no gather, no craft.
    print("\n[2] already have the target -> no work")
    w = _World({PICK: 1})
    okc, msg = _make(w, a, cat).make(PICK, 1)
    (ok if okc and not any(e.startswith(("gather", "craft", "table")) for e in w.events)
        else bad)(f"skips entirely ({msg}; events={w.events})")

    # 3. No gather capability + missing raw -> clean failure.
    print("\n[3] missing raw material, no gather -> fail")
    w = _World()
    okc, msg = _make(w, a, cat, gather=False).make(PICK, 1)
    (ok if not okc and "gather" in msg else bad)(f"fails clearly ({msg})")

    # 4. 2x2-only target (planks) needs neither gather (have logs) nor table.
    print("\n[4] 2x2 target with materials on hand")
    w = _World({"minecraft:oak_log": 2})
    okc, msg = _make(w, a, cat, table=False).make("minecraft:oak_planks", 4)
    (ok if okc and not any(e.startswith("table:") for e in w.events) else bad)(
        f"crafts planks, no table needed ({msg})")

    # 5. Transient craft hiccup -> the round loop re-reads + re-plans and
    #    still completes (resilience: a one-off failure must not abort the make).
    print("\n[5] transient craft failure -> retries and succeeds")
    class _FlakyCrafter(_Crafter):
        def __init__(self, world, assets, cat, fail_on, times):
            super().__init__(world, assets, cat)
            self._fail_on = fail_on; self._left = times
        def craft(self, result_id, *, snap=None):
            if result_id == self._fail_on and self._left > 0:
                self._left -= 1
                self.w.events.append(f"craftFAIL:{result_id.split(':')[-1]}")
                return False, "transient glitch"
            return super().craft(result_id, snap=snap)
    w = _World()
    mem = InventoryMemory(); ctl = _Ctl(w)
    # Fail the stick craft once; the partial chain (planks/table made) changes
    # inventory, so the no-progress guard lets it retry and finish.
    flaky = _FlakyCrafter(w, a, cat, fail_on="minecraft:stick", times=1)
    mk = Maker(ctl, flaky, mem, a, cat, gather_fn=_gather_fn(w),
               table_craft_fn=_table_fn(w), log=lambda m: None)
    okc, msg = mk.make(PICK, 1)
    (ok if okc and w.counts.get(PICK, 0) >= 1 else bad)(
        f"recovers from a transient craft failure ({msg})")
    (ok if any(e.startswith("craftFAIL:") for e in w.events) else bad)(
        "a craft did fail at least once (the retry path was exercised)")

    # 6. Persistent blocker (a step that ALWAYS fails without consuming) ->
    #    aborts in bounded rounds via the no-progress guard, never hangs.
    print("\n[6] persistent craft failure -> bounded abort, no hang")
    class _StuckCrafter(_Crafter):
        def craft(self, result_id, *, snap=None):
            if result_id == "minecraft:oak_planks":
                self.w.events.append("craftFAIL:oak_planks")
                return False, "always fails"      # never consumes/produces
            return super().craft(result_id, snap=snap)
    w = _World({"minecraft:oak_log": 5})       # has logs, but planks never craft
    mem = InventoryMemory(); ctl = _Ctl(w)
    mk = Maker(ctl, _StuckCrafter(w, a, cat), mem, a, cat,
               gather_fn=_gather_fn(w), table_craft_fn=_table_fn(w),
               log=lambda m: None)
    okc, msg = mk.make(PICK, 1)
    (ok if (not okc) and w.counts.get(PICK, 0) == 0 else bad)(
        f"fails cleanly without hanging ({msg})")
    (ok if "stuck" in msg or "round" in msg.lower() else bad)(
        f"reports a bounded give-up ({msg})")

    print("\n" + ("ALL MAKER TESTS PASSED" if not _fails
                  else f"{_fails} CHECK(S) FAILED"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
