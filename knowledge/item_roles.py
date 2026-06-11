# knowledge/item_roles.py
"""
Classify a Minecraft item id into a functional ROLE.

The hotbar manager (and any tool-selection logic) needs to answer "is this
a sword / axe / pickaxe / food / placeable block?" without hard-coding
every item. We derive the role from the asset Catalog where possible
(authoritative for the installed version, so new items are handled
automatically) and fall back to id-suffix heuristics so it still works
when the catalog is unavailable.

Roles
-----
``sword`` ``axe`` ``pickaxe`` ``shovel`` ``hoe`` — the tool tags
(``swords``/``axes``/… in the item tag data), with an ``_sword``/``_axe``/…
suffix fallback.
``food``  — player-edible items. Minecraft has NO universal player-food
tag (the ``*_food`` tags are all animal-feeding), so this is a curated,
config-extensible set.
``armor`` — anything with an equipment slot (head/chest/legs/feet).
``blocks`` — placeable block items (``is_block_item``), the bridging /
building material role.
``None`` — anything else (sticks, ingots, misc).

Keep this PURE (catalog + a string) so it unit-tests without a game.
"""

from __future__ import annotations

from typing import Iterable, Optional, Set


# Functional roles a hotbar slot can be assigned.
ROLES = ("sword", "axe", "pickaxe", "shovel", "hoe", "food", "armor", "blocks")

# Tool tag -> role (the item-tag names Mojang ships).
_TOOL_TAG_ROLE = {
    "swords": "sword", "axes": "axe", "pickaxes": "pickaxe",
    "shovels": "shovel", "hoes": "hoe",
}
# Id-suffix -> role (fallback when tags are missing).
_TOOL_SUFFIX_ROLE = {
    "_sword": "sword", "_axe": "axe", "_pickaxe": "pickaxe",
    "_shovel": "shovel", "_hoe": "hoe",
}

# Curated player-edible items (vanilla 1.21). Extensible via config so a
# new food or a modded one can be added without code changes. Stored as
# bare stems (no ``minecraft:``).
DEFAULT_FOOD: Set[str] = {
    "apple", "golden_apple", "enchanted_golden_apple",
    "carrot", "golden_carrot", "potato", "baked_potato", "poisonous_potato",
    "beetroot", "beetroot_soup", "bread", "cookie", "pumpkin_pie",
    "melon_slice", "sweet_berries", "glow_berries", "chorus_fruit",
    "dried_kelp", "honey_bottle",
    "beef", "cooked_beef", "porkchop", "cooked_porkchop",
    "chicken", "cooked_chicken", "mutton", "cooked_mutton",
    "rabbit", "cooked_rabbit", "rabbit_stew",
    "cod", "cooked_cod", "salmon", "cooked_salmon",
    "tropical_fish", "pufferfish",
    "mushroom_stew", "suspicious_stew", "spider_eye", "rotten_flesh",
}


def _stem(item_id: str) -> str:
    return item_id.split(":", 1)[-1] if ":" in item_id else item_id


def item_role(item_id: Optional[str],
              catalog=None,
              *,
              extra_food: Optional[Iterable[str]] = None) -> Optional[str]:
    """Return the functional role of ``item_id`` (see module docstring),
    or ``None`` for empty / unrecognised / role-less items.

    ``catalog`` (optional) is a :class:`knowledge.catalog.Catalog`; when
    given we use its tags / ``is_block_item`` / ``equipment_slot`` first.
    Suffix + curated-food heuristics back it up so this never hard-fails.
    """
    if not item_id:
        return None
    stem = _stem(item_id)

    info = None
    if catalog is not None:
        try:
            info = catalog.item(item_id)
        except Exception:
            info = None

    # 1. Tools — by tag (authoritative) then by suffix.
    if info is not None and info.tags:
        for tag, role in _TOOL_TAG_ROLE.items():
            if tag in info.tags:
                return role
    for suf, role in _TOOL_SUFFIX_ROLE.items():
        if stem.endswith(suf):
            return role

    # 2. Food — curated set (+ caller extension). Checked before blocks so a
    # food never gets mistaken for something else.
    food = DEFAULT_FOOD if extra_food is None else (DEFAULT_FOOD | set(extra_food))
    if stem in food:
        return "food"

    # 3. Armor — anything with an equipment slot (head/chest/legs/feet).
    if info is not None and info.equipment_slot in ("head", "chest", "legs", "feet"):
        return "armor"
    if any(stem.endswith(s) for s in ("_helmet", "_chestplate", "_leggings", "_boots")):
        return "armor"

    # 4. Placeable block (bridging / building material).
    if info is not None and info.is_block_item:
        return "blocks"

    return None


def matches_role(item_id: Optional[str], role: str,
                 catalog=None, *, extra_food: Optional[Iterable[str]] = None) -> bool:
    """True iff ``item_id`` fills ``role``."""
    return item_role(item_id, catalog, extra_food=extra_food) == role


__all__ = ["ROLES", "DEFAULT_FOOD", "item_role", "matches_role"]
