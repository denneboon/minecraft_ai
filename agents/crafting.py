"""
Crafter — turns "make me X" into hovers + hotkey swaps.

Composes the recipe knowledge (knowledge/recipes.py: what item goes in
which grid cell) with the inventory controller (control/inventory_control.py:
how to move it, hotkey-first, no drag). Reads the open inventory, plans the
craft from what's actually on hand, places the ingredients into the 2x2
grid, and takes the result.

Scope today: the 2x2 INVENTORY grid (planks, sticks, crafting_table). 3x3
recipes (tools) are detected and reported as "needs a table" — executing
them needs an open crafting-table screen, the next increment.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from knowledge.recipes import plan_step
from control.inventory_control import grid_slot
# inventory_counts lives in inventory_memory now (single source of truth for
# "what am I carrying"); re-exported here so existing importers keep working.
from agents.inventory_memory import inventory_counts

_STORAGE_PREFIXES = ("inv_", "hotbar_")


def find_item_slot(snap, item_id: str) -> Optional[str]:
    """The storage slot holding the most of ``item_id`` (or None)."""
    best, best_n = None, -1
    for name, sc in (getattr(snap, "slots", {}) or {}).items():
        if not name.startswith(_STORAGE_PREFIXES):
            continue
        if getattr(sc, "item", None) == item_id:
            n = max(1, int(getattr(sc, "count", 1) or 1))
            if n > best_n:
                best, best_n = name, n
    return best


class Crafter:
    def __init__(self, controller, assets, cat):
        self.ctl = controller
        self.assets = assets
        self.cat = cat

    def _clear_grid(self, width: int) -> None:
        """Return anything left in the craft grid to the inventory (so
        nothing scatters/drops when the screen closes)."""
        for i in range(width * width):
            self.ctl.shift_left_click(f"craft_in_{i}")

    def _read_now(self, *, quick: bool = False):
        """Fresh inventory snapshot, best-effort. ``quick`` skips the
        hover-to-learn pass (presence/positions only — used for the post-craft
        result-slot check). Returns None on failure or a controller without
        ``read()``."""
        stop = (lambda _s: True) if quick else (lambda _s: False)
        try:
            return self.ctl.read(stop_when=stop)
        except TypeError:
            try:
                return self.ctl.read()
            except Exception:
                return None
        except Exception:
            return None

    def _read_settled(self, stop_when):
        """Read the open inventory, but first let MC finish RENDERING the
        current state. A just-completed craft moved stacks; the threaded screen
        capture can otherwise hand us a PRE-render frame where the ingredients
        look missing -> a bogus 'can't craft … from inventory' (the live
        mid-chain failure). Settle, read (hovering until plannable), and if it's
        still not plannable, settle + retry once before trusting the result."""
        snap = None
        for attempt in range(2):
            time.sleep(0.45 if attempt == 0 else 0.4)
            try:
                snap = self.ctl.read(stop_when=stop_when)
            except TypeError:
                snap = self.ctl.read()        # controllers without stop_when
            try:
                if snap is not None and (stop_when is None or stop_when(snap)):
                    return snap
            except Exception:
                return snap
        return snap

    def craft(self, target_id: str, *, snap=None) -> Tuple[bool, str]:
        """Craft ``target_id`` from the current inventory using the 2x2 grid.
        Single-ingredient recipes (planks) move the whole stack in and
        shift-take ALL output; multi-cell recipes place one item per cell.
        Returns (ok, message)."""
        if snap is None:
            # Surgical hover-to-learn: only identify slots until the recipe
            # is plannable (and not at all if the ingredients are already
            # recognised) — instead of hovering the whole inventory.
            def _plannable(s):
                return plan_step(target_id, inventory_counts(s),
                                 self.assets, self.cat) is not None
            snap = self._read_settled(_plannable)
        avail = inventory_counts(snap)
        step = plan_step(target_id, avail, self.assets, self.cat)
        if step is None:
            return False, f"can't craft {target_id.split(':')[-1]} from inventory"
        # 3x3 grid when operating an open crafting table; 2x2 in the
        # inventory. A table recipe in the inventory grid is refused.
        width = 3 if getattr(self.ctl, "container", "") == "crafting_table" else 2
        if step.needs_table and width < 3:
            return False, f"{target_id.split(':')[-1]} needs a 3x3 crafting table"

        # Group the grid cells by the concrete item each needs.
        by_item: Dict[str, List[str]] = {}
        for (rc, item) in step.cell_items.items():
            by_item.setdefault(item, []).append(grid_slot(rc[0], rc[1], width))

        cur = snap
        for idx, (item, cells) in enumerate(by_item.items()):
            if idx > 0:
                # A prior ingredient's move_stack displaces a hotbar item into
                # an inventory slot, so a source slot resolved from the stale
                # pre-move snapshot may now hold something else. Re-read so
                # find_item_slot resolves against the CURRENT inventory (matters
                # for recipes mixing single-cell + multi-cell ingredients).
                cur = self._read_now() or cur
            src = find_item_slot(cur, item)
            if src is None:
                return False, f"no inventory slot holds {item.split(':')[-1]}"
            if len(cells) == 1:
                # whole-stack into the one cell via number-key swap (preferred)
                self.ctl.move_stack(src, cells[0], cur)
            else:
                # several cells of the same item -> split one into each
                self.ctl.distribute_one(src, cells)

        # Verify the recipe actually PRODUCED output before claiming success: a
        # cell under-fill (e.g. a misread source slot) leaves the result slot
        # empty, and returning True there makes Maker advance on a no-op. We
        # need only PRESENCE in the result slot, not its identity (so skip the
        # hover-resolve). If we can't read it, fall back to optimistic success —
        # don't regress the happy path or offline mocks without a result slot.
        after = self._read_now(quick=True)
        result_slot = (after.slots.get("craft_result")
                       if after is not None and getattr(after, "slots", None)
                       else None)
        if result_slot is not None and getattr(result_slot, "is_empty", False):
            self._clear_grid(width)
            return False, (f"{step.result_id.split(':')[-1]}: nothing produced "
                           f"(grid under-filled)")

        self.ctl.take_result()
        self._clear_grid(width)
        return True, f"crafted {step.result_id.split(':')[-1]} x{step.result_count}"

    def ensure(self, target_id: str, count: int = 1, *, memory=None,
               manage_screen: bool = True) -> Tuple[bool, str]:
        """Make sure the inventory holds at least ``count`` of ``target_id``,
        crafting ONLY the shortfall — never a pile for no reason.

        Cost-aware via ``memory`` (an :class:`InventoryMemory`):
          1. If memory already shows >= ``count`` (incl. the hotbar), return
             immediately WITHOUT opening the inventory.
          2. Otherwise open + read the TRUE count (which also refreshes memory),
             compute the deficit, and craft exactly that many — re-reading after
             each craft so the count reflects what the game actually produced.

        ``manage_screen`` opens/closes the inventory itself; set False if the
        caller already has it open. Returns ``(ok, message)``."""
        short = target_id.split(":")[-1]
        if count <= 0:
            return True, f"need 0 {short}"
        # 1. Believe the memory if it's confident we already have enough.
        if memory is not None:
            if memory.assess(target_id, count)[0] == "have":
                return True, (f"already have >={count} {short} "
                              f"(remembered {memory.count(target_id)}; didn't open)")
            # A cheap HUD-hotbar glance (no inventory open) may already cover it.
            try:
                if hasattr(self.ctl, "read_hotbar"):
                    self.ctl.read_hotbar()
            except Exception:
                pass
            if memory.assess(target_id, count)[0] == "have":
                return True, (f"already have >={count} {short} "
                              f"(hotbar has {memory.hotbar_count(target_id)}; didn't open)")
        # 2. Open + verify the real count before crafting anything.
        opened = False
        try:
            if manage_screen:
                self.ctl.open_inventory(); opened = True

            def _read():
                try:
                    s = self.ctl.read(stop_when=lambda _s: False)  # full read
                except TypeError:
                    s = self.ctl.read()
                if memory is not None:
                    memory.observe(s)                # keep the ledger fresh
                return s

            snap = _read()
            have = inventory_counts(snap).get(target_id, 0)
            if have >= count:
                return True, f"already have {have} {short}"
            # 3. Craft the deficit, re-reading the true count after each craft.
            safety = (count - have) + 6            # guard against a stuck loop
            while have < count and safety > 0:
                ok, msg = self.craft(target_id, snap=snap)
                if not ok:
                    return False, f"have {have}/{count} {short}: {msg}"
                snap = _read()
                new = inventory_counts(snap).get(target_id, 0)
                if new <= have:                    # made no progress -> bail
                    return False, (f"have {have}/{count} {short}: craft didn't "
                                   f"increase the count ({msg})")
                have = new
                safety -= 1
            return (have >= count), f"have {have}/{count} {short}"
        finally:
            if opened:
                try:
                    self.ctl.close()
                except Exception:
                    pass
