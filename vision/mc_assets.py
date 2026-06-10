# vision/mc_assets.py
"""
Minecraft asset extractor + loader.

Pulls every type of asset the vision-based AI might need out of the
installed game jar and caches it on disk so subsequent runs are
instant. Sister module to ``vision/mcfont.py`` (which extracts just
the ASCII font atlas); both share ``find_mc_jars`` for jar discovery.

What we extract — and why
-------------------------
**Textures**
    * ``textures/block/``       — every block face, for identifying blocks
                                  the AI is looking at in the world.
    * ``textures/item/``        — item icons, for identifying inventory
                                  contents and ingredients.
    * ``textures/entity/``      — mob skins (zombie, skeleton, …) for
                                  identifying creatures in the world.
    * ``textures/gui/``         — inventory background, hotbar widget,
                                  button textures, etc. — for detecting
                                  GUI screens.
    * ``textures/painting/``    — paintings (in-world art).
    * ``textures/particle/``    — particle sprites; transient but
                                  occasionally diagnostic.
    * ``textures/mob_effect/``  — status-effect icons (Poison, Strength).
    * ``textures/environment/`` — sun, moon, clouds, rain. Useful for
                                  knowing time of day / weather visually.
    * ``textures/colormap/``    — grass/foliage tint maps. Biome colours
                                  shift block textures slightly; this
                                  data lets future code compensate.
    * ``textures/map/``         — minimap & banner icons.
    * ``textures/misc/``        — vignette, enchantment glint, etc.

**Models**
    * ``models/block/``         — block geometry (faces, sizes,
                                  rotation). Future code can render a
                                  reference view for matching.
    * ``models/item/``          — flat item layered models.
    * ``blockstates/``          — block-state → model lookup
                                  (e.g. oak_log: axis=y → oak_log.json).

**Data**
    * ``data/recipe/``          — every vanilla crafting recipe.
    * ``data/loot_table/``      — what mobs and blocks drop.
    * ``data/advancement/``     — advancement criteria (useful for
                                  goal-conditioned agents later).
    * ``data/tags/``            — block/item taxonomic groupings
                                  (#logs, #planks, #wool, etc.).

**Language**
    * ``lang/en_us.json``       — display-name map for items, blocks,
                                  entities, biomes, advancements.

What we DON'T extract
---------------------
* ``sounds/`` — audio isn't part of the vision pipeline.
* ``shaders/`` — runtime rendering programs, not useful as observation.
* ``texts/`` — credits / end poem.
* ``assets/realms/`` — Realms-only marketing assets.

Output layout
-------------
::

    data/mc_assets/<version>/
        textures/...
        models/...
        blockstates/...
        lang/en_us.json
        data/recipe/, loot_table/, advancement/, tags/
        meta.json                         # provenance + counts

Loader
------
``MCAssets.load()`` opens a cached directory and offers
``get_block_texture("stone")``, ``display_name("minecraft:diamond")``,
``recipes_for("oak_planks")`` and friends. JSON files (recipes, loot
tables, models) are read lazily — we have hundreds of them but a typical
query touches one.
"""

from __future__ import annotations

import json
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

# Reuse the jar-discovery logic from mcfont.
from vision.mcfont import find_mc_jars, _version_key

# Sentinel for the texture cache so a legitimately-cached ``None`` (a
# texture that doesn't exist on disk) is distinguished from a cache miss
# — without it a missing texture would re-hit the filesystem every call.
_MISSING = object()


# ---------------------------------------------------------------------------
# Asset category definitions
# ---------------------------------------------------------------------------
# Each entry: (jar_prefix, output_subdir, allowed_extensions).
# Order is descriptive only — extraction iterates the dict.

@dataclass(frozen=True)
class _Category:
    jar_prefix: str
    out_subdir: str
    extensions: Tuple[str, ...]
    description: str


ASSET_CATEGORIES: Dict[str, _Category] = {
    # ── Textures ────────────────────────────────────────────────────
    "block_textures": _Category(
        "assets/minecraft/textures/block/", "textures/block",
        (".png", ".mcmeta"),
        "Block face textures (16×16 typically; some animated).",
    ),
    "item_textures": _Category(
        "assets/minecraft/textures/item/",  "textures/item",
        (".png", ".mcmeta"),
        "Item icons (16×16, RGBA — many have transparency).",
    ),
    "entity_textures": _Category(
        "assets/minecraft/textures/entity/", "textures/entity",
        (".png", ".mcmeta"),
        "Mob and entity skins (variable size).",
    ),
    "gui_textures": _Category(
        "assets/minecraft/textures/gui/",   "textures/gui",
        (".png", ".mcmeta"),
        "Inventory background, widgets, container backgrounds.",
    ),
    "painting_textures": _Category(
        "assets/minecraft/textures/painting/", "textures/painting",
        (".png",),
        "In-world painting art.",
    ),
    "particle_textures": _Category(
        "assets/minecraft/textures/particle/", "textures/particle",
        (".png", ".mcmeta"),
        "Particle sprites.",
    ),
    "effect_textures": _Category(
        "assets/minecraft/textures/mob_effect/", "textures/mob_effect",
        (".png",),
        "Status-effect icons.",
    ),
    "environment_textures": _Category(
        "assets/minecraft/textures/environment/", "textures/environment",
        (".png", ".mcmeta"),
        "Sun, moon, clouds, rain, snow.",
    ),
    "colormap_textures": _Category(
        "assets/minecraft/textures/colormap/", "textures/colormap",
        (".png",),
        "Grass / foliage biome-tint colormaps.",
    ),
    "map_textures": _Category(
        "assets/minecraft/textures/map/", "textures/map",
        (".png",),
        "Map icons (banner markers, etc.).",
    ),
    "misc_textures": _Category(
        "assets/minecraft/textures/misc/", "textures/misc",
        (".png",),
        "Vignette, enchantment glint, pumpkin overlay.",
    ),

    # ── Models ──────────────────────────────────────────────────────
    "block_models": _Category(
        "assets/minecraft/models/block/", "models/block", (".json",),
        "Block geometry (faces, sizes, rotations).",
    ),
    "item_models": _Category(
        "assets/minecraft/models/item/",  "models/item",  (".json",),
        "Item layered models (which textures stack to form the icon).",
    ),
    "blockstates": _Category(
        "assets/minecraft/blockstates/",  "blockstates",  (".json",),
        "Block-state → model lookups (oak_log axis=y → which model).",
    ),
    "atlases": _Category(
        "assets/minecraft/atlases/", "atlases", (".json",),
        "Texture-atlas definitions (which sprites belong to which sheet).",
    ),

    # ── Language ────────────────────────────────────────────────────
    "lang": _Category(
        "assets/minecraft/lang/", "lang", (".json",),
        "Display-name dictionaries (en_us.json + translations).",
    ),

    # ── Data (recipes, loot, tags, advancements) ────────────────────
    "recipes": _Category(
        "data/minecraft/recipe/", "data/recipe", (".json",),
        "Every vanilla crafting recipe.",
    ),
    "loot_tables": _Category(
        "data/minecraft/loot_table/", "data/loot_table", (".json",),
        "Drop tables for mobs, blocks, chests.",
    ),
    "tags": _Category(
        "data/minecraft/tags/", "data/tags", (".json",),
        "Block/item tag groupings (#logs, #planks, #wool, etc.).",
    ),
    "advancements": _Category(
        "data/minecraft/advancement/", "data/advancement", (".json",),
        "Advancement (achievement) criteria.",
    ),
}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _version_from_jar_path(jar_path: str) -> str:
    name = Path(jar_path).stem
    # Both "minecraft-1.21.11-client" (Prism layout) and plain "1.21.11"
    # (official launcher) collapse to "1.21.11".
    name = name.replace("minecraft-", "").replace("-client", "")
    return name


_zip_warn_state = {"emitted": False}


def extract_assets(
    jar_path: str,
    output_dir: str,
    *,
    categories: Optional[Iterable[str]] = None,
    overwrite: bool = False,
) -> Dict[str, Dict[str, int]]:
    """
    Extract selected categories from a Minecraft jar into ``output_dir``.

    Parameters
    ----------
    jar_path    : path to a Minecraft client jar.
    output_dir  : root for the cached files. Created if missing.
    categories  : subset of ``ASSET_CATEGORIES`` keys. ``None`` = all.
    overwrite   : if True, re-extract files that already exist.

    Returns a per-category ``{"count": N, "bytes": M}`` summary.
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    cats = list(categories) if categories else list(ASSET_CATEGORIES)
    stats: Dict[str, Dict[str, int]] = {c: {"count": 0, "bytes": 0} for c in cats}

    with zipfile.ZipFile(jar_path) as z:
        names = z.namelist()
        for cat in cats:
            spec = ASSET_CATEGORIES[cat]
            out_sub = out_root / spec.out_subdir
            out_sub.mkdir(parents=True, exist_ok=True)
            for name in names:
                if not name.startswith(spec.jar_prefix):
                    continue
                if not any(name.endswith(ext) for ext in spec.extensions):
                    continue
                # name might be a directory record — skip
                if name.endswith("/"):
                    continue
                rel = name[len(spec.jar_prefix):]
                dest = out_sub / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.is_file() and not overwrite:
                    stats[cat]["count"] += 1
                    stats[cat]["bytes"] += dest.stat().st_size
                    continue
                try:
                    data = z.read(name)
                except Exception as e:
                    # A genuinely-corrupted entry inside the jar
                    # silently dropped here would leave downstream
                    # code with a None texture and degrade recogniser
                    # quality without explanation. Surface the FIRST
                    # corruption so the user knows the jar may be
                    # incomplete; subsequent failures cached but
                    # silent so a thoroughly-broken jar doesn't
                    # produce thousands of identical log lines.
                    if not _zip_warn_state["emitted"]:
                        _zip_warn_state["emitted"] = True
                        print(f"[mc_assets][WARN] corrupted jar entry "
                              f"{name!r}: {e!r}. The asset will be "
                              f"missing from the cache; further "
                              f"corruption warnings silenced.")
                    continue
                dest.write_bytes(data)
                stats[cat]["count"] += 1
                stats[cat]["bytes"] += len(data)
    return stats


def _write_meta(out_root: Path, jar_path: str, version: str,
                stats: Dict[str, Dict[str, int]]) -> None:
    total_files = sum(s["count"] for s in stats.values())
    total_bytes = sum(s["bytes"] for s in stats.values())
    (out_root / "meta.json").write_text(
        json.dumps({
            "version":     version,
            "source_jar":  str(jar_path),
            "extracted_at": int(time.time()),
            "total_files": total_files,
            "total_bytes": total_bytes,
            "categories":  stats,
        }, indent=2),
        encoding="utf-8",
    )


def ensure_assets(
    cache_root: str,
    *,
    jar_path: Optional[str] = None,
    categories: Optional[Iterable[str]] = None,
    force: bool = False,
) -> str:
    """
    Make sure assets are extracted somewhere under ``cache_root``.

    Returns the directory holding the extracted assets (one level deep:
    ``<cache_root>/<version>/``). If a usable cache already exists it is
    returned unchanged; otherwise the newest installed jar (or
    ``jar_path`` if given) is extracted.
    """
    cache = Path(cache_root)
    if jar_path is None:
        jars = find_mc_jars()
        if not jars:
            raise RuntimeError("No Minecraft jar found on this system.")
        jar_path = jars[0]
    version = _version_from_jar_path(jar_path)
    out_root = cache / version
    if not force and out_root.is_dir() and (out_root / "meta.json").is_file():
        return str(out_root)
    stats = extract_assets(jar_path, str(out_root),
                           categories=categories, overwrite=force)
    _write_meta(out_root, jar_path, version, stats)
    return str(out_root)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

@dataclass
class MCAssets:
    """
    Loader for a cached asset directory.

    Construct via ``MCAssets.load()`` (auto-finds newest cached version)
    or ``MCAssets.load(version="1.21.11")``. JSON-heavy data (recipes,
    loot tables, etc.) is loaded on first use, not at construction.
    """
    root:    Path
    version: str
    _lang_cache:    Dict[str, dict] = field(default_factory=dict, repr=False)
    _recipe_cache:  Dict[str, dict] = field(default_factory=dict, repr=False)
    _model_cache:   Dict[str, dict] = field(default_factory=dict, repr=False)
    # Decoded-texture cache. Texture PNGs are immutable for the life of
    # a run, but the inverse renderer + block classifier re-request the
    # same handful every tick — without this they hit cv2.imread (disk +
    # decode) ~28×/tick (~4 ms). Keyed by (path, rgb-flag). Cached arrays
    # are treated read-only by all callers (resize/cvtColor/astype all
    # return new arrays), so sharing the reference is safe.
    _png_cache:     Dict[Tuple[str, bool], Optional[np.ndarray]] = field(
        default_factory=dict, repr=False)

    # ─── Constructors ───────────────────────────────────────────────

    @classmethod
    def load(cls,
             version: Optional[str] = None,
             cache_root: Optional[str] = None) -> "MCAssets":
        if cache_root is None:
            cache_root = str(Path(__file__).resolve().parent.parent
                             / "data" / "mc_assets")
        base = Path(cache_root)
        if not base.is_dir():
            raise RuntimeError(
                f"No assets cache at {base}. Run "
                f"`python -m vision.mc_assets --extract` to create it."
            )
        if version is None:
            versions = [d.name for d in base.iterdir()
                        if d.is_dir() and (d / "meta.json").is_file()]
            if not versions:
                raise RuntimeError(
                    f"No extracted versions in {base}. Run "
                    f"`python -m vision.mc_assets --extract`."
                )
            version = sorted(versions, key=_version_key, reverse=True)[0]
        root = base / version
        if not root.is_dir():
            raise RuntimeError(f"No assets for version {version} at {root}")
        return cls(root=root, version=version)

    # ─── Texture access ─────────────────────────────────────────────

    def block_texture(self, name: str) -> Optional[np.ndarray]:
        """Return a block texture as RGB uint8, or None if missing."""
        return self._read_png(self.root / "textures" / "block" / f"{name}.png", rgb=True)

    def item_texture(self, name: str) -> Optional[np.ndarray]:
        """Return an item texture as RGBA uint8 (alpha preserved), or None."""
        return self._read_png(self.root / "textures" / "item" / f"{name}.png", rgb=False)

    def entity_texture(self, relpath: str) -> Optional[np.ndarray]:
        """
        Return an entity skin texture. ``relpath`` is the path under
        ``textures/entity/`` (e.g. ``"zombie/zombie.png"``).
        """
        return self._read_png(self.root / "textures" / "entity" / relpath, rgb=False)

    def gui_texture(self, name: str) -> Optional[np.ndarray]:
        return self._read_png(self.root / "textures" / "gui" / f"{name}.png", rgb=False)

    def _read_png(self, path: Path, *, rgb: bool) -> Optional[np.ndarray]:
        # Memoised on (path, rgb): immutable assets, hot-path re-reads.
        key = (str(path), rgb)
        cached = self._png_cache.get(key, _MISSING)
        if cached is not _MISSING:
            return cached
        out = self._decode_png(path)
        self._png_cache[key] = out
        return out

    @staticmethod
    def _decode_png(path: Path) -> Optional[np.ndarray]:
        if not path.is_file():
            return None
        bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            return None
        if bgr.ndim == 2:
            return bgr  # already grayscale
        if bgr.shape[2] == 4:
            # BGRA → RGBA
            return cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGBA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ─── Listing helpers ────────────────────────────────────────────

    def list_block_textures(self) -> List[str]:
        return self._list_pngs(self.root / "textures" / "block")

    def list_item_textures(self) -> List[str]:
        return self._list_pngs(self.root / "textures" / "item")

    def list_entity_textures(self) -> List[str]:
        d = self.root / "textures" / "entity"
        if not d.is_dir():
            return []
        return sorted(str(p.relative_to(d)).replace("\\", "/")
                      for p in d.rglob("*.png"))

    def list_gui_textures(self) -> List[str]:
        return self._list_pngs(self.root / "textures" / "gui")

    def _list_pngs(self, d: Path) -> List[str]:
        if not d.is_dir():
            return []
        return sorted(p.stem for p in d.iterdir() if p.suffix == ".png")

    # ─── Language / display names ───────────────────────────────────

    def lang(self, locale: str = "en_us") -> Dict[str, str]:
        if locale in self._lang_cache:
            return self._lang_cache[locale]
        p = self.root / "lang" / f"{locale}.json"
        if not p.is_file():
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
        self._lang_cache[locale] = data
        return data

    def display_name(self, identifier: str, *, locale: str = "en_us"
                    ) -> Optional[str]:
        """
        Resolve an entity/block/item ID to its localised display name.

        ``identifier`` may be either ``"minecraft:diamond"`` or just
        ``"diamond"``. Looks up the lang keys in order of likelihood:
        ``item.minecraft.diamond``, ``block.minecraft.diamond``,
        ``entity.minecraft.diamond``.
        """
        if ":" in identifier:
            ns, name = identifier.split(":", 1)
        else:
            ns, name = "minecraft", identifier
        L = self.lang(locale)
        for prefix in ("item", "block", "entity", "biome"):
            key = f"{prefix}.{ns}.{name}"
            if key in L:
                return L[key]
        return None

    # ─── Recipes ────────────────────────────────────────────────────

    def recipe(self, name: str) -> Optional[dict]:
        if ":" in name:
            _, name = name.split(":", 1)
        if name in self._recipe_cache:
            return self._recipe_cache[name]
        p = self.root / "data" / "recipe" / f"{name}.json"
        if not p.is_file():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        self._recipe_cache[name] = data
        return data

    def list_recipes(self) -> List[str]:
        d = self.root / "data" / "recipe"
        if not d.is_dir():
            return []
        return sorted(p.stem for p in d.iterdir() if p.suffix == ".json")

    def recipes_producing(self, item_id: str) -> List[str]:
        """
        Return recipe IDs whose ``result`` produces ``item_id``.

        ``item_id`` may be ``"minecraft:oak_planks"`` or ``"oak_planks"``.
        Scans every recipe lazily (cached). Acceptably fast for the ~700
        vanilla recipes; if it ever isn't, build an index once.
        """
        if ":" not in item_id:
            item_id = "minecraft:" + item_id
        out = []
        for name in self.list_recipes():
            rec = self.recipe(name)
            if not rec:
                continue
            result = rec.get("result") or {}
            # result can be a string id (new format) or {"id": ..., "count": ...}
            if isinstance(result, str):
                if result == item_id:
                    out.append(name)
            elif isinstance(result, dict):
                if result.get("id") == item_id or result.get("item") == item_id:
                    out.append(name)
        return sorted(out)

    # ─── Models / blockstates ──────────────────────────────────────

    def block_model(self, name: str) -> Optional[dict]:
        return self._read_json_cached(
            self.root / "models" / "block" / f"{name}.json",
            cache_key=("block_model", name),
        )

    def item_model(self, name: str) -> Optional[dict]:
        return self._read_json_cached(
            self.root / "models" / "item" / f"{name}.json",
            cache_key=("item_model", name),
        )

    def blockstate(self, name: str) -> Optional[dict]:
        return self._read_json_cached(
            self.root / "blockstates" / f"{name}.json",
            cache_key=("blockstate", name),
        )

    def _read_json_cached(self, path: Path, *,
                          cache_key: Tuple[str, str]) -> Optional[dict]:
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        self._model_cache[cache_key] = data
        return data

    # ─── Tags ───────────────────────────────────────────────────────

    def tag(self, kind: str, name: str) -> Optional[List[str]]:
        """
        Return the flat list of values in tag ``minecraft:<kind>/<name>``.

        Example: ``assets.tag("item", "logs")`` → all log items.

        Does not recursively resolve nested ``#tag`` references.
        """
        if ":" in name:
            _, name = name.split(":", 1)
        p = self.root / "data" / "tags" / kind / f"{name}.json"
        if not p.is_file():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        vals = data.get("values") or []
        return [v if isinstance(v, str) else v.get("id", "") for v in vals]

    def list_tags(self, kind: str) -> List[str]:
        d = self.root / "data" / "tags" / kind
        if not d.is_dir():
            return []
        return sorted(p.stem for p in d.iterdir() if p.suffix == ".json")

    # ─── Meta ───────────────────────────────────────────────────────

    def meta(self) -> dict:
        p = self.root / "meta.json"
        if not p.is_file():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_cache_root() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "mc_assets"


def _cli() -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Extract Minecraft assets (textures, models, "
                    "recipes, lang, tags) from an installed jar.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--jar", default=None,
                   help="Specific jar path. Default: newest installed.")
    p.add_argument("--output", default=None,
                   help="Output dir. Default: data/mc_assets/<version>/.")
    p.add_argument("--force", action="store_true",
                   help="Re-extract even if files exist.")
    p.add_argument("--list-categories", action="store_true",
                   help="List available asset categories and exit.")
    p.add_argument("--categories", nargs="+", default=None,
                   metavar="CAT",
                   help="Specific categories to extract. Default: all.")
    p.add_argument("--list-jars", action="store_true",
                   help="List candidate Minecraft jars and exit.")
    args = p.parse_args()

    if args.list_categories:
        for k, c in ASSET_CATEGORIES.items():
            print(f"  {k:20s}  {c.jar_prefix:<45s} → {c.out_subdir}")
            print(f"  {' ':20s}  {c.description}")
        return 0

    if args.list_jars:
        for j in find_mc_jars():
            print(j)
        return 0

    jar = args.jar
    if jar is None:
        jars = find_mc_jars()
        if not jars:
            print("[mc_assets][ERROR] No Minecraft jar found.")
            return 2
        jar = jars[0]
    if not Path(jar).is_file():
        print(f"[mc_assets][ERROR] Jar not found: {jar}")
        return 2

    version = _version_from_jar_path(jar)
    out_root = Path(args.output) if args.output else _default_cache_root() / version
    print(f"[mc_assets] Source: {jar}")
    print(f"[mc_assets] Output: {out_root}")
    print(f"[mc_assets] Force:  {args.force}")

    t0 = time.perf_counter()
    stats = extract_assets(jar, str(out_root),
                           categories=args.categories,
                           overwrite=args.force)
    _write_meta(out_root, jar, version, stats)
    dt = time.perf_counter() - t0

    print()
    print(f"  {'CATEGORY':<22s} {'FILES':>8s} {'SIZE (MiB)':>12s}")
    print(f"  {'-'*22:<22s} {'-'*8:>8s} {'-'*12:>12s}")
    total_files = total_bytes = 0
    for cat, s in stats.items():
        print(f"  {cat:<22s} {s['count']:>8d} {s['bytes']/1024/1024:>12.2f}")
        total_files += s["count"]
        total_bytes += s["bytes"]
    print(f"  {'-'*22:<22s} {'-'*8:>8s} {'-'*12:>12s}")
    print(f"  {'TOTAL':<22s} {total_files:>8d} {total_bytes/1024/1024:>12.2f}")
    print(f"\n[mc_assets] Extracted in {dt:.1f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
