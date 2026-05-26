# knowledge/__init__.py
"""
Knowledge layer — read-only registries and reference data the AI uses
to make decisions independent of pixel input.

The two most-used entry points:

  * ``Catalog`` — every vanilla block / item / entity, built from the
    cached MC assets.  Use this when the AI needs to reason about what
    something *is* (mob category, item stack size, block tags …).
  * ``load_minecraft_settings`` (re-exported from ``config``) — the
    player's in-game options as YAML.

Embeddings, recipe graphs, biome statistics, and other future
data sources will plug in here without touching the rest of the code.
"""

from __future__ import annotations

from knowledge.catalog import BlockInfo, Catalog, EntityInfo, ItemInfo

__all__ = ["BlockInfo", "ItemInfo", "EntityInfo", "Catalog"]
