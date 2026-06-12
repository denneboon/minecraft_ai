"""
Crafting-recipe knowledge for the AI.

The vanilla recipes are already extracted by MCAssets (data/recipe/*.json,
1470 of them) and reachable via ``assets.recipe(name)``. This module turns
that raw JSON into something the bot can ACT on:

  * resolve tag ingredients (``#minecraft:planks`` -> the concrete plank
    ids, recursively) against the Catalog,
  * normalise shaped + shapeless recipes into a grid of (row,col)->
    acceptable-item placements,
  * know whether a recipe fits the 2x2 inventory grid or needs a 3x3
    crafting table,
  * given what the bot HAS, decide if a target is craftable now and which
    concrete item goes in each grid cell — and chain simple dependencies
    (logs -> planks -> sticks -> table -> wooden tools).

It is pure data logic — no screen, no mouse — so it is fully offline
testable. The execution layer (hover slots, hotkey-swap items into the
grid) consumes the ``CraftPlan`` this produces.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def _ingredient_tokens(spec) -> List[str]:
    """Normalise a recipe ingredient spec to token strings ('minecraft:x' or
    '#minecraft:tag'). Handles the 1.21 string form, the older
    {"item":…}/{"tag":…} dict form, and lists of either."""
    if spec is None:
        return []
    if isinstance(spec, str):
        return [spec]
    if isinstance(spec, list):
        out: List[str] = []
        for s in spec:
            out.extend(_ingredient_tokens(s))
        return out
    if isinstance(spec, dict):
        if spec.get("item"):
            return [spec["item"]]
        if spec.get("tag"):
            return ["#" + spec["tag"]]
    return []


def _resolve_spec(spec, cat) -> List[str]:
    """Ingredient spec -> concrete acceptable item ids (tags expanded)."""
    opts: List[str] = []
    for tk in _ingredient_tokens(spec):
        opts.extend(_resolve_token(tk, cat))
    seen, uniq = set(), []
    for v in opts:
        if v not in seen:
            seen.add(v); uniq.append(v)
    return uniq


def _resolve_token(token: str, cat, _seen=None) -> List[str]:
    """A recipe ingredient is either a literal item id ('minecraft:stick')
    or a tag ('#minecraft:planks'). Resolve to a list of concrete item ids,
    recursing through nested tags (#logs -> #logs_that_burn -> oak_log…)."""
    if _seen is None:
        _seen = set()
    if not token:
        return []
    if not token.startswith("#"):
        return [token]
    tag = token[1:]
    if tag in _seen:
        return []
    _seen.add(tag)
    out: List[str] = []
    try:
        values = cat.items_in_tag(tag) or []
    except Exception:
        values = []
    for v in values:
        v = v if isinstance(v, str) else getattr(v, "id", str(v))
        if v.startswith("#"):
            out.extend(_resolve_token(v, cat, _seen))
        else:
            out.append(v)
    # de-dup, keep order
    seen, uniq = set(), []
    for v in out:
        if v not in seen:
            seen.add(v); uniq.append(v)
    return uniq


@dataclass
class CraftRecipe:
    """A normalised crafting recipe (table/inventory grid only — smelting,
    smithing etc. are skipped)."""
    result_id: str
    result_count: int
    shaped: bool
    width: int                       # bounding grid width  (1-3)
    height: int                      # bounding grid height (1-3)
    # (row, col) -> list of acceptable concrete item ids for that cell.
    # For shapeless, ingredients are laid out row-major into the smallest
    # grid that holds them (so a 2-ingredient shapeless is width=2,height=1).
    cells: Dict[Tuple[int, int], List[str]] = field(default_factory=dict)

    @property
    def fits_2x2(self) -> bool:
        return self.width <= 2 and self.height <= 2

    def placements(self) -> List[Tuple[Tuple[int, int], List[str]]]:
        """[( (row,col), [acceptable item ids] ), …] — every non-empty cell."""
        return sorted(self.cells.items(), key=lambda kv: kv[0])


def recipe_for(item_id: str, assets, cat) -> Optional[CraftRecipe]:
    """The crafting-grid recipe that PRODUCES ``item_id`` (oak_planks,
    stick, crafting_table, wooden_pickaxe, …), or None if it isn't made on
    a crafting grid."""
    name = item_id.split(":")[-1]
    data = assets.recipe(name)
    if not data:
        return None
    rtype = str(data.get("type", ""))
    res = data.get("result") or {}
    result_id = res.get("id") or res.get("item")
    if not result_id:
        return None
    count = int(res.get("count", 1))

    if rtype == "minecraft:crafting_shaped":
        pattern: List[str] = data.get("pattern") or []
        key: Dict[str, object] = data.get("key") or {}
        height = len(pattern)
        width = max((len(r) for r in pattern), default=0)
        cells: Dict[Tuple[int, int], List[str]] = {}
        for r, row in enumerate(pattern):
            for c, ch in enumerate(row):
                if ch == " ":
                    continue
                opts = _resolve_spec(key.get(ch), cat)
                if opts:
                    cells[(r, c)] = opts
        return CraftRecipe(result_id, count, True, width, height, cells)

    if rtype == "minecraft:crafting_shapeless":
        ings = data.get("ingredients") or []
        resolved: List[List[str]] = [_resolve_spec(ing, cat) for ing in ings]
        n = len([r for r in resolved if r])
        width = min(2, n) if n <= 4 else 3
        cells = {}
        idx = 0
        for opts in resolved:
            if not opts:
                continue
            r, c = divmod(idx, max(1, width))
            cells[(r, c)] = opts
            idx += 1
        height = (max((rc[0] for rc in cells), default=0) + 1) if cells else 0
        return CraftRecipe(result_id, count, False, width, height, cells)

    return None


@dataclass
class CraftStep:
    """One crafting action: put ``cell_items`` (concrete id per grid cell)
    into the grid and take ``count`` of ``result_id``."""
    result_id: str
    result_count: int
    needs_table: bool
    # (row, col) -> the concrete item id to place there (chosen from options
    # by what the bot actually has).
    cell_items: Dict[Tuple[int, int], str]
    times: int = 1                   # how many times to run this craft


_TABLE = "minecraft:crafting_table"


def _have(available: Dict[str, int], item: str) -> int:
    return int(available.get(item, 0))


def plan_step(target_id: str, available: Dict[str, int], assets, cat
              ) -> Optional[CraftStep]:
    """Can ``target_id`` be crafted in ONE step from ``available`` (item ->
    count)? Returns the concrete cell placement, or None.

    PRESENCE-based, not count-based: a cell is satisfiable if the bot HAS at
    least one of an acceptable item — the same stack can fill several cells.
    Stack COUNTS from the inventory OCR are unreliable (and a 64-stack
    legitimately fills all 9 cells), so we don't decrement per cell; the game
    enforces the real amount (placing simply does nothing if short, which the
    executor handles harmlessly)."""
    rec = recipe_for(target_id, assets, cat)
    if rec is None:
        return None
    cell_items: Dict[Tuple[int, int], str] = {}
    for (rc, opts) in rec.placements():
        choice = next((opt for opt in sorted(opts, key=lambda o: -_have(available, o))
                       if _have(available, opt) > 0), None)
        if choice is None:
            return None                       # don't have ANY acceptable item
        cell_items[rc] = choice
    return CraftStep(rec.result_id, rec.result_count, not rec.fits_2x2,
                     cell_items)


def plan_craft(target_id: str, available: Dict[str, int], assets, cat,
               max_depth: int = 4) -> Optional[List[CraftStep]]:
    """An ordered list of craft steps that yields at least one ``target_id``
    from ``available``, crafting intermediates as needed (logs -> planks ->
    sticks …). None if it can't be reached within ``max_depth``. Greedy +
    depth-bounded — enough for the wooden tech tree; not a general solver."""
    avail = dict(available)

    def _ensure(item: str, qty: int, depth: int, steps: List[CraftStep]) -> bool:
        if _have(avail, item) >= qty:
            return True
        if depth <= 0:
            return False
        rec = recipe_for(item, assets, cat)
        if rec is None:
            return False
        # how many crafts to cover the shortfall
        per = max(1, rec.result_count)
        runs = -(-(qty - _have(avail, item)) // per)     # ceil div
        # make sure each cell's ingredient exists (craft it first if needed)
        placements = rec.placements()
        for _ in range(runs):
            cell_items: Dict[Tuple[int, int], str] = {}
            for (rc, opts) in placements:
                got = None
                for opt in sorted(opts, key=lambda o: -_have(avail, o)):
                    if _have(avail, opt) > 0:
                        got = opt
                        break
                if got is None:
                    # try to craft the first option
                    for opt in opts:
                        if _ensure(opt, 1, depth - 1, steps) and _have(avail, opt) > 0:
                            got = opt
                            break
                if got is None or _have(avail, got) <= 0:
                    return False
                cell_items[rc] = got
                avail[got] = _have(avail, got) - 1
            steps.append(CraftStep(rec.result_id, rec.result_count,
                                   not rec.fits_2x2, cell_items))
            avail[item] = _have(avail, item) + rec.result_count
        return _have(avail, item) >= qty

    steps: List[CraftStep] = []
    if _ensure(target_id, 1, max_depth, steps):
        return steps
    return None


def plan_make(target_id: str, count: int, available: Dict[str, int],
              assets, cat, max_depth: int = 6
              ) -> Optional[Tuple[Dict[str, int], List[CraftStep]]]:
    """Plan to MAKE ``count`` of ``target_id`` from scratch, separating what
    must be GATHERED from what must be CRAFTED.

    Like :func:`plan_craft`, but when an ingredient has no crafting recipe (a
    raw material — logs, cobblestone, …) and isn't on hand, it records the
    shortfall as something to GATHER instead of failing. Returns
    ``(raw_to_gather, steps)``:

      * ``raw_to_gather``: ``{raw_item_id: qty}`` to acquire in the world first,
      * ``steps``: the ordered crafts (deps first; ``needs_table`` flags the
        3x3 ones) to run once the raw materials + intermediates are present.

    Returns ``None`` only if the target itself has no crafting recipe."""
    target_rec = recipe_for(target_id, assets, cat)
    if target_rec is None:
        return None
    avail = dict(available)
    raw: Dict[str, int] = {}
    steps: List[CraftStep] = []

    def _ensure(item: str, qty: int, depth: int) -> bool:
        if _have(avail, item) >= qty:
            return True
        short = qty - _have(avail, item)
        rec = recipe_for(item, assets, cat)
        if rec is None or depth <= 0:
            # raw material (or recursion bottomed out) -> gather it
            raw[item] = raw.get(item, 0) + short
            avail[item] = _have(avail, item) + short        # assume acquired
            return True
        per = max(1, rec.result_count)
        runs = -(-short // per)                              # ceil div
        placements = rec.placements()
        for _ in range(runs):
            cell_items: Dict[Tuple[int, int], str] = {}
            for (rc, opts) in placements:
                got = next((o for o in sorted(opts, key=lambda o: -_have(avail, o))
                            if _have(avail, o) > 0), None)
                if got is None:
                    # make/gather the first acceptable option, then use it
                    opt = opts[0]
                    if not _ensure(opt, 1, depth - 1) or _have(avail, opt) <= 0:
                        return False
                    got = opt
                cell_items[rc] = got
                avail[got] = _have(avail, got) - 1
            steps.append(CraftStep(rec.result_id, rec.result_count,
                                   not rec.fits_2x2, cell_items))
            avail[item] = _have(avail, item) + rec.result_count
        return _have(avail, item) >= qty

    # A 3x3 target is crafted on a placed crafting table (which is reclaimed
    # afterwards), so we need ONE table on hand — plan it first if absent, so
    # its planks/logs are part of the gather/craft plan. The table itself is
    # 2x2-craftable, so this doesn't loop.
    if count > 0 and not target_rec.fits_2x2 and _have(avail, _TABLE) < 1:
        _ensure(_TABLE, 1, max_depth)

    if _ensure(target_id, count, max_depth):
        return raw, steps
    return None
