"""
Arrange the hotbar: put the BEST item of each reserved role into its slot.

"Best" = highest material tier for tools (netherite > diamond > iron > copper
> gold > stone > wood), best food quality for food, and most-count for blocks
(see :func:`knowledge.item_roles.best_item_for_role`). So if the inventory has
a wooden and a diamond sword, the diamond one ends up in the sword slot; the
best pickaxe/axe/shovel/hoe in theirs; the best food and the biggest block
stack in theirs.

Pure-policy lives in ``knowledge.item_roles``; this module is the thin
inventory-manipulation layer that READS the open inventory and SWAPS the best
item of each role into its hotbar slot via the controller's ``number_swap``
(hover slot + press the hotbar number — no drag). Re-reads between moves so a
displaced item is re-found at its new location.
"""
from __future__ import annotations

import time
from typing import Dict, Optional

from knowledge.item_roles import best_item_for_role
from agents.inventory_memory import inventory_counts
from agents.crafting import find_item_slot

# Default layout when the caller doesn't supply one — covers every category the
# bot uses (slot 1-9 -> role). Match this to hotbar.slot_roles in settings so
# the tool-selection behaviours look in the same slots we fill.
DEFAULT_ARRANGE: Dict[int, str] = {
    1: "sword", 2: "pickaxe", 3: "axe", 4: "shovel", 5: "hoe",
    6: "blocks", 7: "food",
}


def arrange_hotbar(ctl, slot_roles: Optional[Dict[int, str]] = None,
                   catalog=None, *, extra_food=(), log=None,
                   settle: float = 0.3) -> Dict[str, str]:
    """Move the BEST item of each reserved role into its hotbar slot.

    ``ctl``        an open-able InventoryController (open_inventory/read/
                   number_swap/close).
    ``slot_roles`` ``{slot(1-9): role}``; defaults to :data:`DEFAULT_ARRANGE`.
    ``catalog``    optional Catalog (needed to classify blocks/armour).
    Returns ``{role: item_id}`` for the items it placed (or found already in
    place). Best-effort: a read/move hiccup on one role just skips it.
    """
    layout = dict(slot_roles or DEFAULT_ARRANGE)
    placed: Dict[str, str] = {}
    ctl.open_inventory()
    try:
        time.sleep(settle)
        for slot in sorted(layout):
            role = layout[slot]
            try:
                snap = ctl.read(stop_when=lambda s: False)
            except TypeError:
                snap = ctl.read()
            if snap is None:
                continue
            best = best_item_for_role(inventory_counts(snap), role, catalog,
                                      extra_food=extra_food)
            if not best:
                continue
            src = find_item_slot(snap, best)
            target = f"hotbar_{slot - 1}"
            if src is None:
                continue
            if src == target:                      # already in the right slot
                placed[role] = best
                continue
            try:
                ctl.number_swap(src, slot)         # best -> this hotbar slot
            except Exception:
                continue
            placed[role] = best
            if log:
                log(f"[hotbar] slot {slot} {role} <- {best.split(':')[-1]}")
            time.sleep(0.12)
    finally:
        try:
            ctl.close()
        except Exception:
            pass
    return placed


__all__ = ["arrange_hotbar", "DEFAULT_ARRANGE"]
