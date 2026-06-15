"""
Mining knowledge: which TOOL mines a block fastest, and which BLOCK you mine to
obtain a given raw item.

The wooden tech tree needs the bot to gather more than logs — stone tools need
cobblestone, which you get by mining stone with a pickaxe. Two pure lookups make
that generic:

  * :func:`tool_role_for_block` — the hotbar tool ROLE (pickaxe / axe / shovel /
    hoe) that mines a block, from the catalog's ``mineable/*`` tags (authoritative
    for the installed version) with id-suffix fallbacks.
  * :func:`gather_source_for` — given a raw ITEM you want (cobblestone, dirt, …),
    the block you MINE to get it and the tool role to do it with. Most blocks
    drop themselves; the early-game exceptions (stone → cobblestone, deepslate →
    cobbled_deepslate, grass_block → dirt) are curated.

Pure data logic (catalog + strings) — fully offline-testable, no screen/mouse.
"""
from __future__ import annotations

from typing import Optional, Tuple

# mineable/* tag -> hotbar tool role. Order matters only if a block carried
# several (it shouldn't); pickaxe is the most common so it's listed first.
_MINEABLE_TAG_ROLE = (
    ("mineable/pickaxe", "pickaxe"),
    ("mineable/axe", "axe"),
    ("mineable/shovel", "shovel"),
    ("mineable/hoe", "hoe"),
)

# Id-suffix fallbacks when the catalog has no tag (or no catalog supplied).
_AXE_SUFFIX = ("_log", "_wood", "_stem", "_hyphae", "_planks", "_fence",
               "_door", "_trapdoor", "_sign", "_slab", "_stairs")
_SHOVEL_STEM = {"dirt", "grass_block", "sand", "red_sand", "gravel", "clay",
                "soul_sand", "soul_soil", "mud", "snow", "snow_block",
                "podzol", "coarse_dirt", "rooted_dirt", "farmland", "mycelium"}
_PICKAXE_HINT = ("_ore", "stone", "deepslate", "cobble", "_bricks", "_brick",
                 "andesite", "diorite", "granite", "tuff", "basalt", "_block")


def _stem(block_id: str) -> str:
    return block_id.split(":", 1)[-1] if ":" in block_id else block_id


def tool_role_for_block(block_id: Optional[str], catalog=None) -> Optional[str]:
    """The hotbar tool role that mines ``block_id`` fastest ('pickaxe'/'axe'/
    'shovel'/'hoe'), or None when no tool helps (e.g. a torch / instamine block,
    or unknown). Catalog ``mineable/*`` tags first, id-suffix heuristics after."""
    if not block_id:
        return None
    info = None
    if catalog is not None:
        try:
            info = catalog.block(block_id)
        except Exception:
            info = None
    tags = getattr(info, "tags", None) or set()
    for tag, role in _MINEABLE_TAG_ROLE:
        if tag in tags:
            return role
    stem = _stem(block_id)
    if stem.endswith("_leaves"):
        return "hoe"
    if stem.endswith(_AXE_SUFFIX):
        return "axe"
    if stem in _SHOVEL_STEM or stem.endswith("_concrete_powder"):
        return "shovel"
    if any(h in stem for h in _PICKAXE_HINT):
        return "pickaxe"
    return None


# Raw item you WANT -> block you MINE to drop it (when they differ). Most blocks
# drop themselves; these early-game ones don't. Curated (no loot-table data ships
# with the assets), extend as the tech tree grows.
_MINE_SOURCE = {
    "minecraft:cobblestone": "minecraft:stone",
    "minecraft:cobbled_deepslate": "minecraft:deepslate",
    "minecraft:dirt": "minecraft:dirt",            # grass_block also drops dirt
}


def gather_source_for(raw_item: Optional[str], catalog=None
                      ) -> Tuple[Optional[str], Optional[str]]:
    """``(source_block_id, tool_role)`` to obtain ``raw_item`` by mining — e.g.
    ``cobblestone -> (stone, pickaxe)``. Defaults to mining the block of the same
    id (cobblestone-from-stone is the notable exception). Returns ``(None, None)``
    for a falsy item. ``tool_role`` may be None if no tool is needed/known."""
    if not raw_item:
        return None, None
    source = _MINE_SOURCE.get(raw_item, raw_item)
    return source, tool_role_for_block(source, catalog)


__all__ = ["tool_role_for_block", "gather_source_for"]
