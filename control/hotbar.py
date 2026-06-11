# control/hotbar.py
"""
Hotbar slot-role management.

The player reserves hotbar slots for roles — e.g. slot 1 = sword, 2 = axe,
3 = pickaxe, 5 = blocks, 9 = food. Behaviours then ask "which slot do I
select to eat / to mine this log / to bridge?" without caring where the
item physically sits, and the bot can flag when a slot holds the wrong
thing.

This manager is PURE policy over a hotbar READING (it takes the 9 slot
contents as input from the inventory recogniser) plus the role config and
the item-role classifier. It does NOT touch input — selecting the slot is
``keyboard.select_hotbar_slot`` or ``AgentAction.hotbar``. Keeping it pure
makes it unit-testable and lets a smarter (e.g. learned) slot policy drop
in behind the same interface later.

Slot reorganisation ("keep ONLY the assigned item in each slot" by moving
mis-placed items) is a separate inventory-manipulation skill; this module
provides the KNOWLEDGE for it via :meth:`mis_stocked`, and meanwhile
:meth:`slot_for_role` still finds the item wherever it actually is so the
bot keeps working even when the hotbar isn't tidy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from knowledge.item_roles import ROLES, item_role


@dataclass
class HotbarConfig:
    # slot number (1-9) -> reserved role (one of knowledge.item_roles.ROLES)
    slot_roles: Dict[int, str] = field(default_factory=dict)
    # extra item stems to treat as food (player-food has no vanilla tag).
    extra_food: Tuple[str, ...] = ()


class HotbarManager:
    """Maps reserved roles to hotbar slots given the current hotbar read."""

    def __init__(self, config: Optional[HotbarConfig] = None, catalog=None):
        self.cfg = config or HotbarConfig()
        self._catalog = catalog
        # role -> assigned slot (inverse of slot_roles). If the user assigns
        # the same role to two slots, the lower slot wins (deterministic).
        self._role_slot: Dict[str, int] = {}
        for slot in sorted(self.cfg.slot_roles):
            role = self.cfg.slot_roles[slot]
            self._role_slot.setdefault(role, slot)
        self._items: List[Optional[str]] = [None] * 9

    # ── Ingest a hotbar reading ────────────────────────────────────
    def update(self, hotbar) -> None:
        """Feed the current hotbar contents. Accepts a list of item-id
        strings, ``None``s, or objects with an ``.item`` attribute (the
        inventory recogniser's ``SlotContent``). Index 0 = slot 1."""
        items: List[Optional[str]] = []
        for x in (hotbar or []):
            if x is None:
                items.append(None)
            elif isinstance(x, str):
                items.append(x or None)
            else:
                items.append(getattr(x, "item", None) or None)
        self._items = (items + [None] * 9)[:9]

    # ── Queries ────────────────────────────────────────────────────
    def item_in_slot(self, slot: int) -> Optional[str]:
        return self._items[slot - 1] if 1 <= slot <= 9 else None

    def role_in_slot(self, slot: int) -> Optional[str]:
        return item_role(self.item_in_slot(slot), self._catalog,
                         extra_food=self.cfg.extra_food)

    def assigned_slot(self, role: str) -> Optional[int]:
        """The slot the CONFIG reserves for ``role`` (ignores contents)."""
        return self._role_slot.get(role)

    def slot_for_role(self, role: str) -> Optional[int]:
        """The slot to SELECT to use ``role`` right now: the reserved slot
        if it actually holds that role, else any slot that does, else
        ``None`` (no such item in the hotbar). This keeps the bot working
        even when the hotbar is untidy."""
        a = self._role_slot.get(role)
        if a is not None and self.role_in_slot(a) == role:
            return a
        for s in range(1, 10):
            if self.role_in_slot(s) == role:
                return s
        return None

    def has_role(self, role: str) -> bool:
        return self.slot_for_role(role) is not None

    def mis_stocked(self) -> List[Tuple[int, str, Optional[str]]]:
        """``[(slot, expected_role, actual_item), …]`` for reserved slots
        whose current item doesn't fill the reserved role (empty counts as
        mis-stocked). Drives 'keep only the assigned item here' tidying."""
        out: List[Tuple[int, str, Optional[str]]] = []
        for slot in sorted(self.cfg.slot_roles):
            role = self.cfg.slot_roles[slot]
            if self.role_in_slot(slot) != role:
                out.append((slot, role, self.item_in_slot(slot)))
        return out

    # ── Convenience accessors (the common roles) ───────────────────
    def sword_slot(self) -> Optional[int]:   return self.slot_for_role("sword")
    def axe_slot(self) -> Optional[int]:      return self.slot_for_role("axe")
    def pickaxe_slot(self) -> Optional[int]:  return self.slot_for_role("pickaxe")
    def blocks_slot(self) -> Optional[int]:   return self.slot_for_role("blocks")
    def food_slot(self) -> Optional[int]:     return self.slot_for_role("food")


def build_hotbar_manager(settings: Optional[dict] = None, catalog=None) -> HotbarManager:
    """Build a HotbarManager from the ``hotbar`` settings section::

        hotbar:
          slot_roles: {1: sword, 2: axe, 3: pickaxe, 5: blocks, 9: food}
          extra_food: []
    """
    h = ((settings or {}).get("hotbar", {}) or {})
    raw = h.get("slot_roles", {}) or {}
    slot_roles: Dict[int, str] = {}
    for k, v in raw.items():
        try:
            slot = int(k)
        except (TypeError, ValueError):
            continue
        if 1 <= slot <= 9 and isinstance(v, str) and v in ROLES:
            slot_roles[slot] = v
    extra_food = tuple(h.get("extra_food", []) or ())
    return HotbarManager(HotbarConfig(slot_roles=slot_roles, extra_food=extra_food),
                         catalog=catalog)


__all__ = ["HotbarManager", "HotbarConfig", "build_hotbar_manager"]
