"""
Equip the BEST armour the bot is carrying into each body slot.

The companion to :func:`agents.hotbar_arranger.arrange_hotbar`: that puts the
best tools/food/blocks in the hotbar; this WEARS the best helmet, chestplate,
leggings and boots (by protection tier — netherite > diamond > iron >
chainmail/turtle > gold > leather). Only upgrades — it never swaps in a worse
or equal piece, and never equips a non-armour head item (carved_pumpkin, mob
head, elytra) since those rank 0.

Pure ranking lives in ``knowledge.item_roles``; this is the thin
inventory-manipulation layer that READS the open inventory and moves pieces:

  * an EMPTY armour slot -> a single shift-click auto-equips the piece;
  * an occupied slot holding something WORSE -> a 3-click swap (pick the better
    piece, drop it on the armour slot, put the displaced old piece back in the
    now-empty source slot).

Re-reads between slots so a move is seen by the next one.
"""
from __future__ import annotations

import time
from typing import Dict, Optional

from knowledge.item_roles import best_armor_for_slot, armor_material_rank
from agents.inventory_memory import inventory_counts, read_open_inventory
from agents.crafting import find_item_slot

_BODY_SLOTS = ("head", "chest", "legs", "feet")


def equip_best_armor(ctl, catalog=None, *, log=None,
                     settle: float = 0.3) -> Dict[str, str]:
    """Wear the best carried armour in each body slot. ``ctl`` is an open-able
    InventoryController (open_inventory/read/shift_left_click/left_click/close).
    Returns ``{body_slot: item_id}`` for the pieces it equipped (only the ones
    it actually changed). Best-effort: a read/move hiccup on one slot is skipped.
    """
    equipped: Dict[str, str] = {}
    ctl.open_inventory()
    try:
        time.sleep(settle)
        for slot in _BODY_SLOTS:
            snap = read_open_inventory(ctl)
            if snap is None:
                continue
            # inventory_counts excludes the armour slots, so this is only LOOSE
            # pieces — exactly the upgrade candidates (not the one already worn).
            best = best_armor_for_slot(inventory_counts(snap), slot, catalog)
            if not best:
                continue
            armor_key = f"armor_{slot}"
            worn = (snap.armor or {}).get(armor_key)
            worn_id = getattr(worn, "item", None) if worn is not None else None
            if worn_id is not None and \
                    armor_material_rank(worn_id) >= armor_material_rank(best):
                continue                       # already as good or better
            src = find_item_slot(snap, best)
            if src is None:
                continue
            try:
                if worn_id is None:
                    ctl.shift_left_click(src)          # auto-equip into empty slot
                else:
                    ctl.left_click(src)                # pick the better piece
                    ctl.left_click(armor_key)          # wear it; old piece -> cursor
                    ctl.left_click(src)                # drop old in the freed slot
            except Exception:
                continue
            equipped[slot] = best
            if log:
                log(f"[armor] {slot} <- {best.split(':')[-1]}")
            time.sleep(0.12)
    finally:
        try:
            ctl.close()
        except Exception:
            pass
    return equipped


__all__ = ["equip_best_armor"]
