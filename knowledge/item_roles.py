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


# Tool material tiers, BEST first — the operator's preferred order
# (netherite > diamond > iron > copper > gold > stone > wood).
_MATERIAL_RANK = {
    "netherite": 7, "diamond": 6, "iron": 5, "copper": 4,
    "golden": 3, "gold": 3, "stone": 2, "wooden": 1, "wood": 1,
}

# Food preference (roughly by hunger+saturation restored), best first.
_FOOD_RANK = {
    "enchanted_golden_apple": 30, "golden_apple": 26, "golden_carrot": 24,
    "cooked_beef": 22, "cooked_porkchop": 22, "cooked_mutton": 20,
    "cooked_salmon": 18, "cooked_chicken": 16, "cooked_cod": 16,
    "cooked_rabbit": 15, "rabbit_stew": 15, "mushroom_stew": 14,
    "beetroot_soup": 14, "suspicious_stew": 14, "bread": 12, "baked_potato": 11,
    "pumpkin_pie": 10, "carrot": 8, "apple": 7, "beetroot": 6,
    "melon_slice": 5, "sweet_berries": 4, "glow_berries": 4, "cookie": 3,
    "dried_kelp": 3, "honey_bottle": 3, "chorus_fruit": 3,
    "beef": 2, "porkchop": 2, "chicken": 2, "mutton": 2, "rabbit": 2,
    "cod": 2, "salmon": 2, "potato": 2, "tropical_fish": 1,
    "pufferfish": 0, "spider_eye": 0, "rotten_flesh": 0, "poisonous_potato": 0,
}


# Armor material tiers, BEST first. Distinct from tool tiers: armour has
# leather / chainmail / turtle and NO wood/stone/copper. turtle_helmet has no
# tiered prefix but protects ~iron-tier, so rank it there.
_ARMOR_RANK = {
    "netherite": 6, "diamond": 5, "iron": 4, "chainmail": 3, "turtle": 4,
    "golden": 2, "gold": 2, "leather": 1,
}
# Which body slot a piece occupies, by id suffix (fallback for the catalog's
# equipment_slot). turtle_helmet is a head piece despite the odd name.
_ARMOR_SLOT_SUFFIX = {
    "head": ("_helmet",), "chest": ("_chestplate",),
    "legs": ("_leggings",), "feet": ("_boots",),
}


def armor_material_rank(item_id: Optional[str]) -> int:
    """Protection tier for ranking 'which helmet/chestplate/… is best' —
    netherite=6 … leather=1, turtle=4, and 0 for a non-tiered head item
    (carved_pumpkin, mob head, elytra) so those never beat real armour."""
    if not item_id:
        return 0
    prefix = _stem(item_id).split("_", 1)[0]
    return _ARMOR_RANK.get(prefix, 0)


def armor_slot_of(item_id: Optional[str], catalog=None) -> Optional[str]:
    """The body slot ('head'/'chest'/'legs'/'feet') a piece equips into, or
    None. Uses the catalog's equipment_slot first, id-suffix as a fallback."""
    if not item_id:
        return None
    info = None
    if catalog is not None:
        try:
            info = catalog.item(item_id)
        except Exception:
            info = None
    if info is not None and getattr(info, "equipment_slot", None) in (
            "head", "chest", "legs", "feet"):
        return info.equipment_slot
    stem = _stem(item_id)
    for slot, sufs in _ARMOR_SLOT_SUFFIX.items():
        if any(stem.endswith(s) for s in sufs) or (
                slot == "head" and stem == "turtle_helmet"):
            return slot
    return None


def best_armor_for_slot(items, slot: str, catalog=None) -> Optional[str]:
    """The BEST real armour piece for body ``slot`` ('head'/'chest'/'legs'/
    'feet') from ``items`` — highest protection tier, ties broken by count.

    Only pieces with a real armour material (tier > 0) are considered, so a
    carved_pumpkin / mob head / elytra is never auto-equipped over nothing.
    ``items`` is a ``{item_id: count}`` map or an iterable of ids; returns the
    item id or None when nothing fits the slot."""
    if isinstance(items, dict):
        pairs = list(items.items())
    else:
        counts: dict = {}
        for it in (items or []):
            if it:
                counts[it] = counts.get(it, 0) + 1
        pairs = list(counts.items())
    best, best_key = None, None
    for item_id, count in pairs:
        rank = armor_material_rank(item_id)
        if rank <= 0 or armor_slot_of(item_id, catalog) != slot:
            continue
        key = (rank, int(count or 0))
        if best_key is None or key > best_key:
            best, best_key = item_id, key
    return best


def material_rank(item_id: Optional[str]) -> int:
    """Tool material tier for ranking 'which sword/pickaxe is best' —
    netherite=7 … wood=1, and 0 for an item with no tiered material prefix."""
    if not item_id:
        return 0
    prefix = _stem(item_id).split("_", 1)[0]
    return _MATERIAL_RANK.get(prefix, 0)


def _role_quality(item_id: str, role: str) -> int:
    """Sort key for 'best of a role': material tier for tools, curated quality
    for food, 0 otherwise (blocks/armour fall back to count in the caller)."""
    if role in ("sword", "axe", "pickaxe", "shovel", "hoe"):
        return material_rank(item_id)
    if role == "food":
        return _FOOD_RANK.get(_stem(item_id), 1)
    return 0


def best_item_for_role(items, role: str, catalog=None,
                       *, extra_food: Optional[Iterable[str]] = None
                       ) -> Optional[str]:
    """The BEST item filling ``role`` from ``items`` — highest material tier
    (tools) or food quality, ties (and blocks/armour) broken by greatest count.

    ``items`` is either a ``{item_id: count}`` mapping or an iterable of item
    ids. Returns the item id, or ``None`` when nothing fills the role."""
    if isinstance(items, dict):
        pairs = list(items.items())
    else:
        counts: dict = {}
        for it in (items or []):
            if it:
                counts[it] = counts.get(it, 0) + 1
        pairs = list(counts.items())
    best, best_key = None, None
    for item_id, count in pairs:
        if not matches_role(item_id, role, catalog, extra_food=extra_food):
            continue
        key = (_role_quality(item_id, role), int(count or 0))
        if best_key is None or key > best_key:
            best, best_key = item_id, key
    return best


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


__all__ = ["ROLES", "DEFAULT_FOOD", "item_role", "matches_role",
           "material_rank", "best_item_for_role",
           "armor_material_rank", "armor_slot_of", "best_armor_for_slot"]
