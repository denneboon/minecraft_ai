"""
Maker — autonomous "make X from scratch".

Ties the pieces together into one goal: given a target item, work out the full
plan (knowledge.recipes.plan_make), GATHER the raw materials it needs, CRAFT the
2x2 intermediates (planks, sticks, a crafting table), and TABLE-CRAFT the final
3x3 recipe — all count-aware (only the shortfall) via the inventory ledger.

The heavyweight, world-touching steps are injected as callbacks so this stays a
thin, offline-testable orchestrator:

  * ``gather_fn(item_id, qty) -> bool`` — acquire ``qty`` of a raw material from
    the world (e.g. chop logs). Returns whether it got enough.
  * ``table_craft_fn(target_id) -> (ok, msg)`` — place a crafting table, open
    it, craft the 3x3 recipe, break the table back.

2x2 crafting uses the injected :class:`Crafter` on the inventory grid; the
ledger (:class:`InventoryMemory`) lets it skip work it doesn't need.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

from knowledge.recipes import plan_make
from agents.inventory_memory import inventory_counts

_TABLE = "minecraft:crafting_table"


class Maker:
    def __init__(self, controller, crafter, memory, assets, cat, *,
                 gather_fn: Optional[Callable[[str, int], bool]] = None,
                 table_craft_fn: Optional[Callable[[str], Tuple[bool, str]]] = None,
                 log: Callable[[str], None] = print):
        self.ctl = controller
        self.crafter = crafter
        self.memory = memory
        self.assets = assets
        self.cat = cat
        self.gather_fn = gather_fn
        self.table_craft_fn = table_craft_fn
        self.log = log

    # ------------------------------------------------------------------
    def _read_counts(self) -> dict:
        """Open + full-read the inventory (updates the ledger), return counts."""
        self.ctl.open_inventory()
        try:
            try:
                snap = self.ctl.read(stop_when=lambda s: False)
            except TypeError:
                snap = self.ctl.read()
        finally:
            self.ctl.close()
        return inventory_counts(snap)

    # ------------------------------------------------------------------
    def make(self, target_id: str, count: int = 1) -> Tuple[bool, str]:
        """Make ``count`` of ``target_id`` from scratch. Returns (ok, message)."""
        short = target_id.split(":")[-1]
        if count <= 0:
            return True, f"need 0 {short}"

        # 0. Already have enough? (cheap HUD glance + ledger, no full open.)
        try:
            if hasattr(self.ctl, "read_hotbar"):
                self.ctl.read_hotbar()
        except Exception:
            pass
        if self.memory.assess(target_id, count)[0] == "have":
            return True, f"already have >={count} {short}"

        # 1. Accurate read + plan from what's truly on hand.
        avail = self._read_counts()
        if avail.get(target_id, 0) >= count:
            return True, f"already have {avail.get(target_id, 0)} {short}"
        plan = plan_make(target_id, count, avail, self.assets, self.cat)
        if plan is None:
            return False, f"no crafting recipe for {short}"
        raw, steps = plan
        raw_summary = ", ".join(f"{k.split(':')[-1]}:{v}" for k, v in raw.items()) or "-"
        self.log(f"[make] {short}: gather [{raw_summary}], {len(steps)} craft step(s)")

        # 2. Gather raw materials (logs, …) — only the shortfall.
        for item, qty in raw.items():
            if self.gather_fn is None:
                return False, f"need {qty} {item.split(':')[-1]} but no gather capability"
            self.log(f"[make] gather {qty} {item.split(':')[-1]}")
            if not self.gather_fn(item, qty):
                return False, f"couldn't gather {qty} {item.split(':')[-1]}"

        # 3. Run the craft steps in order: batch consecutive 2x2 crafts inside
        # one open inventory; hand each 3x3 step to the table-craft callback.
        i = 0
        while i < len(steps):
            if steps[i].needs_table:
                if self.table_craft_fn is None:
                    return False, "a 3x3 recipe needs a table but no table-craft capability"
                res = steps[i].result_id
                self.log(f"[make] table-craft {res.split(':')[-1]}")
                ok, msg = self.table_craft_fn(res)
                if not ok:
                    return False, f"table-craft {res.split(':')[-1]}: {msg}"
                i += 1
            else:
                self.ctl.open_inventory()
                try:
                    while i < len(steps) and not steps[i].needs_table:
                        res = steps[i].result_id
                        ok, msg = self.crafter.craft(res)
                        self.log(f"[make] craft {res.split(':')[-1]}: {'OK' if ok else 'FAIL'} ({msg})")
                        if not ok:
                            return False, f"craft {res.split(':')[-1]}: {msg}"
                        i += 1
                finally:
                    self.ctl.close()

        # 4. Verify.
        avail = self._read_counts()
        have = avail.get(target_id, 0)
        return (have >= count), f"have {have}/{count} {short}"
