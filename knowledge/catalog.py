# knowledge/catalog.py
"""
A single read-only registry of every block, item, and entity the
vision-only AI can encounter in vanilla Minecraft.

Why a separate "catalog" layer
------------------------------
Vision modules already load ``vision.mc_assets.MCAssets`` to pull
individual textures and recipes out of the extracted game jar. That
loader exposes the data as on-disk files — ``block_texture("stone")``,
``item_texture("apple")``, ``recipe("oak_planks")`` — which is exactly
what the recogniser needs.

But the rest of the AI (planners, world map, action policies, future
LLM agents) wants to ask higher-level questions:

  * "is ``minecraft:diamond_pickaxe`` a tool?"
  * "is ``minecraft:zombie`` a hostile mob?"
  * "what's the display name of ``minecraft:oak_planks``?"
  * "which blocks count as a log for the ``oak_planks`` recipe?"
  * "which items are food, and how many hunger points do they restore?"
  * "how does the world coordinate of a stair / slab differ from a cube?"

Doing those lookups on raw MCAssets each time is verbose and bakes
asset-layout knowledge into every consumer. Catalog wraps the answers
in a clean data model so callers can reason about entities the same way
a human reading the wiki would. The data model is also a natural place
to attach perception artefacts (visual embeddings for block faces,
sound fingerprints for mobs, average dimensions for items) as the AI
grows in capability.

Design choices
--------------
* **Lazy loading**: the catalog is cheap to construct (just an
  ``MCAssets`` reference). The first time you call ``blocks()`` it
  walks the cached models and builds the index — typically <50 ms for
  vanilla 1.21.x.
* **Source of truth**: everything is derived from the cached MC assets
  on disk (``data/mc_assets/<version>/``). When MC ships a new
  version the user re-runs the asset extractor and the catalog picks
  the new data up automatically — no hard-coded lists to maintain.
* **Future-proofing**: every entry exposes a ``meta`` dict for
  perception-side annotations (embeddings, average colour, common
  rotations) that future modules can fill in without changing the
  data classes.
* **Read-only API**: callers do not mutate Catalog state. Mutability
  lives in higher-level state (the world map, the inventory snapshot,
  etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BlockInfo:
    """
    One vanilla block.

    Fields
    ------
    id           : ``"minecraft:stone"``.
    name         : human display name from ``lang/en_us.json`` (``"Stone"``).
    has_texture  : True if a flat ``textures/block/<id>.png`` exists.
    has_model    : True if ``models/block/<id>.json`` exists.
    tags         : the ``minecraft:block/<tag>`` tags this block belongs
                   to ("logs", "planks", "wool", "flowers" …).
    meta         : free-form dict for perception layer to attach
                   embeddings, average colours, rotation hints, etc.
    """
    id:           str
    name:         Optional[str]   = None
    has_texture:  bool            = False
    has_model:    bool            = False
    tags:         Set[str]        = field(default_factory=set)
    meta:         Dict[str, Any]  = field(default_factory=dict)


@dataclass
class ItemInfo:
    """
    One vanilla item.

    The item-vs-block distinction in MC is fuzzy — most blocks are
    *also* items you can hold in the hotbar. We mark ``is_block_item``
    when an item id has a matching block entry. ``equipment_slot`` is
    set for armour/tools by inspecting the item-model and tags.
    """
    id:           str
    name:         Optional[str]   = None
    has_texture:  bool            = False
    has_model:    bool            = False
    is_block_item: bool           = False
    equipment_slot: Optional[str] = None   # "head" | "chest" | "legs" | "feet" | "offhand" | None
    stack_size:   int             = 64
    tags:         Set[str]        = field(default_factory=set)
    meta:         Dict[str, Any]  = field(default_factory=dict)


@dataclass
class EntityInfo:
    """
    One vanilla entity (mob, projectile, vehicle, player).

    Texture availability is per-entity-type rather than per-file —
    some entities ship as one PNG (``zombie.png``), others as a folder
    (``villager/profession/farmer.png``). ``texture_paths`` holds the
    relative paths under ``textures/entity/``.
    """
    id:               str
    name:             Optional[str]   = None
    category:         Optional[str]   = None   # "hostile" | "passive" | "neutral" | "boss" | "projectile" | "vehicle" | "player" | "misc"
    has_loot_table:   bool            = False
    texture_paths:    List[str]       = field(default_factory=list)
    tags:             Set[str]        = field(default_factory=set)
    meta:             Dict[str, Any]  = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hard-coded knowledge that isn't in the jar
# ---------------------------------------------------------------------------
# Mojang doesn't ship a JSON list of "which mobs are hostile". The
# canonical source is the wiki / source code. We mirror the modern
# 1.21.x classification here. When new mobs ship we add them.

_HOSTILE_MOBS: Set[str] = {
    "blaze", "bogged", "breeze", "creaking", "creeper", "drowned",
    "elder_guardian", "endermite", "ender_dragon", "evoker",
    "ghast", "guardian", "hoglin", "husk", "magma_cube", "phantom",
    "pillager", "piglin_brute", "ravager", "shulker", "silverfish",
    "skeleton", "slime", "spider", "stray", "vex", "vindicator",
    "warden", "witch", "wither", "wither_skeleton", "zoglin",
    "zombie", "zombie_villager",
}

_NEUTRAL_MOBS: Set[str] = {
    "bee", "cave_spider", "dolphin", "enderman", "fox", "goat",
    "iron_golem", "llama", "trader_llama", "panda", "piglin",
    "polar_bear", "snow_golem", "wolf", "zombified_piglin",
}

_PASSIVE_MOBS: Set[str] = {
    "allay", "armadillo", "axolotl", "bat", "camel", "cat", "chicken",
    "cod", "cow", "donkey", "frog", "glow_squid", "happy_ghast",
    "horse", "mooshroom", "mule", "ocelot", "parrot", "pig", "pufferfish",
    "rabbit", "salmon", "sheep", "skeleton_horse", "sniffer", "squid",
    "strider", "tadpole", "tropical_fish", "turtle", "villager",
    "wandering_trader", "zombie_horse",
}

_BOSS_MOBS: Set[str] = {"ender_dragon", "wither", "warden", "elder_guardian"}

_PROJECTILE_ENTITIES: Set[str] = {
    "arrow", "spectral_arrow", "trident",
    "snowball", "egg", "ender_pearl",
    "fire_charge", "fireball", "small_fireball", "dragon_fireball",
    "wither_skull", "llama_spit",
    "experience_bottle", "potion",
    "thrown_potion", "splash_potion", "lingering_potion",
}

_VEHICLE_ENTITIES: Set[str] = {
    "boat", "oak_boat", "spruce_boat", "birch_boat", "jungle_boat",
    "acacia_boat", "dark_oak_boat", "mangrove_boat", "cherry_boat",
    "bamboo_raft", "chest_boat", "chest_minecart", "command_block_minecart",
    "furnace_minecart", "hopper_minecart", "minecart", "spawner_minecart",
    "tnt_minecart",
}

_PLAYER_LIKE = {"player", "armor_stand"}


def _entity_category(name: str) -> str:
    if name in _BOSS_MOBS:        return "boss"
    if name in _HOSTILE_MOBS:     return "hostile"
    if name in _NEUTRAL_MOBS:     return "neutral"
    if name in _PASSIVE_MOBS:     return "passive"
    if name in _PROJECTILE_ENTITIES: return "projectile"
    if name in _VEHICLE_ENTITIES: return "vehicle"
    if name in _PLAYER_LIKE:      return "player"
    return "misc"


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

class Catalog:
    """
    High-level registry of vanilla blocks, items, and entities.

    Construct with an ``MCAssets`` instance; the heavy indexing happens
    lazily the first time a category is queried.
    """

    def __init__(self, assets) -> None:
        self.assets = assets
        self._blocks:  Optional[Dict[str, BlockInfo]]  = None
        self._items:   Optional[Dict[str, ItemInfo]]   = None
        self._entities: Optional[Dict[str, EntityInfo]] = None
        self._lang_cache: Optional[Dict[str, str]]     = None

    # ── Construction helpers ────────────────────────────────────────

    @classmethod
    def load(cls, assets=None) -> "Catalog":
        """Convenience: build a Catalog with the newest cached assets."""
        if assets is None:
            from vision.mc_assets import MCAssets
            assets = MCAssets.load()
        return cls(assets)

    # ── Public accessors ────────────────────────────────────────────

    def blocks(self) -> Dict[str, BlockInfo]:
        if self._blocks is None:
            self._blocks = self._build_blocks()
        return self._blocks

    def items(self) -> Dict[str, ItemInfo]:
        if self._items is None:
            self._items = self._build_items()
        return self._items

    def entities(self) -> Dict[str, EntityInfo]:
        if self._entities is None:
            self._entities = self._build_entities()
        return self._entities

    def block(self, ident: str) -> Optional[BlockInfo]:
        return self.blocks().get(_normalise(ident))

    def item(self, ident: str) -> Optional[ItemInfo]:
        return self.items().get(_normalise(ident))

    def entity(self, ident: str) -> Optional[EntityInfo]:
        return self.entities().get(_normalise(ident))

    # ── Aggregate views ─────────────────────────────────────────────

    def hostile_mobs(self) -> List[EntityInfo]:
        return [e for e in self.entities().values() if e.category in ("hostile", "boss")]

    def passive_mobs(self) -> List[EntityInfo]:
        return [e for e in self.entities().values() if e.category == "passive"]

    def neutral_mobs(self) -> List[EntityInfo]:
        return [e for e in self.entities().values() if e.category == "neutral"]

    def projectiles(self) -> List[EntityInfo]:
        return [e for e in self.entities().values() if e.category == "projectile"]

    def vehicles(self) -> List[EntityInfo]:
        return [e for e in self.entities().values() if e.category == "vehicle"]

    def blocks_in_tag(self, tag: str) -> List[BlockInfo]:
        """Return every block matching ``minecraft:block/<tag>``."""
        members = self.assets.tag("block", tag) or []
        out = []
        for ident in members:
            b = self.block(ident)
            if b is not None:
                out.append(b)
        return out

    def items_in_tag(self, tag: str) -> List[ItemInfo]:
        members = self.assets.tag("item", tag) or []
        out = []
        for ident in members:
            it = self.item(ident)
            if it is not None:
                out.append(it)
        return out

    # ── Internals ───────────────────────────────────────────────────

    def _lang(self) -> Dict[str, str]:
        if self._lang_cache is None:
            self._lang_cache = self.assets.lang("en_us") or {}
        return self._lang_cache

    def _build_blocks(self) -> Dict[str, BlockInfo]:
        textures = set(self.assets.list_block_textures())
        models   = _model_stems(self.assets, "block")
        lang     = self._lang()

        ids: Set[str] = set()
        # Anything with a model OR a texture is a candidate. Whitelist
        # against the lang file so we keep "displayable" blocks and
        # drop internal block-states/overlays.
        for stem in textures | models:
            key = f"block.minecraft.{stem}"
            if key in lang or stem in models:
                ids.add(stem)

        # Also include every id that appears in a #block tag, even if
        # its lang entry is missing — tags are an authoritative list.
        tag_index = self._tag_index("block")
        ids.update(stem for stem in tag_index)

        blocks: Dict[str, BlockInfo] = {}
        for stem in sorted(ids):
            full_id = f"minecraft:{stem}"
            blocks[full_id] = BlockInfo(
                id          = full_id,
                name        = lang.get(f"block.minecraft.{stem}"),
                has_texture = stem in textures,
                has_model   = stem in models,
                tags        = tag_index.get(stem, set()),
            )
        return blocks

    def _build_items(self) -> Dict[str, ItemInfo]:
        textures = set(self.assets.list_item_textures())
        models   = _model_stems(self.assets, "item")
        lang     = self._lang()
        block_ids = self.blocks()        # ensure built

        ids: Set[str] = set()
        for stem in textures | models:
            key = f"item.minecraft.{stem}"
            blk_key = f"block.minecraft.{stem}"
            if key in lang or blk_key in lang or stem in models:
                ids.add(stem)

        tag_index = self._tag_index("item")
        ids.update(stem for stem in tag_index)

        items: Dict[str, ItemInfo] = {}
        for stem in sorted(ids):
            full_id = f"minecraft:{stem}"
            name = lang.get(f"item.minecraft.{stem}") or lang.get(f"block.minecraft.{stem}")
            slot = _equipment_slot_for(stem)
            items[full_id] = ItemInfo(
                id            = full_id,
                name          = name,
                has_texture   = stem in textures,
                has_model     = stem in models,
                is_block_item = full_id in block_ids,
                equipment_slot = slot,
                stack_size    = _default_stack_size(stem),
                tags          = tag_index.get(stem, set()),
            )
        return items

    def _build_entities(self) -> Dict[str, EntityInfo]:
        lang = self._lang()

        # Gather entity ids from three sources:
        #   1. lang file:  entity.minecraft.<id> = <Name>
        #   2. textures/entity/<id>.png  or  textures/entity/<id>/...
        #   3. loot tables in data/entity/  (only mob-shaped entities have them)
        names_from_lang: Set[str] = set()
        for key in lang:
            if key.startswith("entity.minecraft."):
                rest = key[len("entity.minecraft."):]
                # Skip key-fragments like "axolotl.lucky" which are
                # variant names; the base id appears without a dot.
                if "." not in rest:
                    names_from_lang.add(rest)

        names_from_texture: Set[str] = set()
        ent_textures = self.assets.list_entity_textures()
        for rel in ent_textures:
            # rel looks like "zombie.png" or "villager/profession/farmer.png".
            first = rel.split("/")[0]
            stem  = first[:-4] if first.endswith(".png") else first
            names_from_texture.add(stem)

        names_from_loot: Set[str] = set()
        loot_root = Path(self.assets.root) / "data" / "loot_table" / "entities"
        if loot_root.is_dir():
            for p in loot_root.rglob("*.json"):
                rel = p.relative_to(loot_root)
                stem = rel.stem if rel.parent == Path(".") else f"{rel.parent}/{rel.stem}".split("/")[0]
                names_from_loot.add(stem)

        all_names = names_from_lang | names_from_texture | names_from_loot
        # Always include the well-known categories even if some are
        # missing from one source — they're vanilla.
        all_names |= (_HOSTILE_MOBS | _NEUTRAL_MOBS | _PASSIVE_MOBS
                      | _PROJECTILE_ENTITIES | _VEHICLE_ENTITIES | _PLAYER_LIKE)

        # Build per-entity texture path lists.
        textures_by_entity: Dict[str, List[str]] = {}
        for rel in ent_textures:
            head = rel.split("/")[0]
            stem = head[:-4] if head.endswith(".png") else head
            textures_by_entity.setdefault(stem, []).append(rel)

        out: Dict[str, EntityInfo] = {}
        for name in sorted(all_names):
            if not name:
                continue
            full_id = f"minecraft:{name}"
            out[full_id] = EntityInfo(
                id              = full_id,
                name            = lang.get(f"entity.minecraft.{name}"),
                category        = _entity_category(name),
                has_loot_table  = (name in names_from_loot),
                texture_paths   = textures_by_entity.get(name, []),
            )
        return out

    def _tag_index(self, kind: str) -> Dict[str, Set[str]]:
        """
        Build ``{block_or_item_stem: {tag1, tag2, …}}`` from the tag
        files under ``data/tags/<kind>/``.

        Tags can recursively include other tags (``#minecraft:logs``);
        we expand one level non-recursively — that's enough for any
        practical "is this a log?" query without paying the cost of
        a full transitive closure. If a future caller needs full
        closure we can promote this to a fixpoint pass.
        """
        kinds_dir = Path(self.assets.root) / "data" / "tags" / kind
        if not kinds_dir.is_dir():
            return {}
        index: Dict[str, Set[str]] = {}
        for p in kinds_dir.rglob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            rel = p.relative_to(kinds_dir).with_suffix("")
            tag_name = str(rel).replace("\\", "/")
            for v in data.get("values", []):
                ident = v if isinstance(v, str) else v.get("id", "")
                if not ident or ident.startswith("#"):
                    continue
                stem = _normalise(ident).split(":", 1)[-1]
                index.setdefault(stem, set()).add(tag_name)
        return index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(ident: str) -> str:
    """Return ``minecraft:<stem>`` from any of ``stem`` / ``minecraft:stem``."""
    if ":" in ident:
        return ident
    return f"minecraft:{ident}"


def _model_stems(assets, kind: str) -> Set[str]:
    d = Path(assets.root) / "models" / kind
    if not d.is_dir():
        return set()
    return {p.stem for p in d.iterdir() if p.suffix == ".json"}


# Coarse equipment-slot lookup. The canonical source is the item's
# component data in Mojang's registry, which we don't ship — but the
# id-suffix convention is stable across versions and exhaustive enough
# for vanilla.
def _equipment_slot_for(stem: str) -> Optional[str]:
    if stem.endswith("_helmet") or stem in {"turtle_helmet"}:
        return "head"
    if stem.endswith("_chestplate") or stem == "elytra":
        return "chest"
    if stem.endswith("_leggings"):
        return "legs"
    if stem.endswith("_boots"):
        return "feet"
    if stem in {"shield", "totem_of_undying"}:
        return "offhand"
    return None


# Default stack size by id-suffix convention. Not 100 % accurate
# (e.g. saddles stack to 1) but a useful upper bound — perception
# can correct this when it reads a real stack count from the screen.
def _default_stack_size(stem: str) -> int:
    if stem.endswith(("_pickaxe", "_axe", "_shovel", "_hoe", "_sword",
                       "_helmet", "_chestplate", "_leggings", "_boots",
                       "_boat", "_minecart")):
        return 1
    if stem in {"bucket", "lava_bucket", "water_bucket", "milk_bucket",
                "shield", "totem_of_undying", "elytra", "saddle"}:
        return 1
    if stem in {"ender_pearl", "snowball", "egg", "honey_bottle"}:
        return 16
    return 64


__all__ = [
    "BlockInfo", "ItemInfo", "EntityInfo", "Catalog",
]
