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

from typing import Dict, List, Optional, Tuple

from knowledge.recipes import plan_step
from control.inventory_control import grid_slot

_STORAGE_PREFIXES = ("inv_", "hotbar_")


def inventory_counts(snap) -> Dict[str, int]:
    """item id -> total count across the main inventory + hotbar."""
    counts: Dict[str, int] = {}
    for name, sc in (getattr(snap, "slots", {}) or {}).items():
        if not name.startswith(_STORAGE_PREFIXES):
            continue
        item = getattr(sc, "item", None)
        if not item:
            continue
        counts[item] = counts.get(item, 0) + max(1, int(getattr(sc, "count", 1) or 1))
    return counts


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

    def craft(self, target_id: str, *, snap=None) -> Tuple[bool, str]:
        """Craft ``target_id`` from the current inventory using the 2x2 grid.
        Single-ingredient recipes (planks) move the whole stack in and
        shift-take ALL output; multi-cell recipes place one item per cell.
        Returns (ok, message)."""
        if snap is None:
            snap = self.ctl.read()
        avail = inventory_counts(snap)
        step = plan_step(target_id, avail, self.assets, self.cat)
        if step is None:
            return False, f"can't craft {target_id.split(':')[-1]} from inventory"
        if step.needs_table:
            return False, f"{target_id.split(':')[-1]} needs a 3x3 crafting table"

        width = 2
        # Group the grid cells by the concrete item each needs.
        by_item: Dict[str, List[str]] = {}
        for (rc, item) in step.cell_items.items():
            by_item.setdefault(item, []).append(grid_slot(rc[0], rc[1], width))

        for item, cells in by_item.items():
            src = find_item_slot(snap, item)
            if src is None:
                return False, f"no inventory slot holds {item.split(':')[-1]}"
            if len(cells) == 1:
                # whole-stack into the one cell via number-key swap (preferred)
                self.ctl.move_stack(src, cells[0], snap)
            else:
                # several cells of the same item -> split one into each
                self.ctl.distribute_one(src, cells)

        self.ctl.take_result()
        self._clear_grid(width)
        return True, f"crafted {step.result_id.split(':')[-1]} x{step.result_count}"
