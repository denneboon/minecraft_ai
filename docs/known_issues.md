# Known issues — needs live validation before fixing

Findings from a code audit that are **real but timing/state-sensitive on the
live crafting + item-movement path**, so they should be reproduced and fixed
with Minecraft attached (not blind), to avoid regressing the working
`make.py wooden_pickaxe` flow. The core crafting/recipe/slot-indexing logic was
audited and verified correct (2×2 and 3×3 slot maps, wooden_pickaxe placement,
plank/stick/table count math).

## Crafting / inventory control

- **`agents/crafting.py` `Crafter.craft` — stale-snapshot source lookup (MED).**
  `find_item_slot(snap, item)` is re-queried for each ingredient from the SAME
  `snap` taken before any item moved. `move_stack`'s two `number_swap`s displace
  a hotbar item into the inventory source slot; if two ingredients' source/carry
  slots alias, the second lookup reads a slot whose contents changed and places
  the wrong item with no error. Latent on the wooden_pickaxe path (both
  ingredients use `distribute_one`, which restores `src`); a landmine for
  recipes mixing single-cell + multi-cell ingredients. Fix: re-read (or
  re-resolve `find_item_slot`) after each placement, or reserve carry slots so
  they can't alias an unplaced ingredient. **Validate live** with a recipe that
  mixes a 1-cell and a multi-cell ingredient.

- **`agents/crafting.py` `craft()` over-reports success (MED).** Returns
  `(True, "crafted …")` even when cells were under-filled and nothing was
  produced; only the `ensure()` re-read catches it, while `Maker`'s 2×2 batch
  loop advances on the optimistic `True` and detects the no-op a round later.
  Fix: have `craft()` verify the result slot actually produced output before
  returning success. **Validate live** (depends on post-craft re-read timing).

- **`control/inventory_control.py` `read_hud_hotbar` — stale relabel (LOW).**
  Overwrites a fresh HUD item read with the remembered layout id while keeping
  the fresh count, so a slot whose contents changed since the last full read
  yields a confidently-wrong ledger entry. Fix: only relabel when the HUD count
  is consistent with the remembered item; else mark "unknown".

## Knowledge / catalog (latent, not on the craft path)

- **`knowledge/catalog.py` `_tag_index` is one-level but `items_in_tag`
  recurses (LOW).** Reading `ItemInfo.tags` gives a non-transitive answer while
  `items_in_tag` is transitive — two tag APIs disagree. Document or make
  `_tag_index` a fixpoint pass.
- **`knowledge/recipes.py` `CraftStep.times` is always 1 (LOW).** Batched counts
  are handled by callers; a consumer reading `times` expecting a batch would be
  wrong. Dead field — document or wire up.
- **`vision/weather.py` `precip_edge_min` is dead config (LOW).** Defined +
  wired through the builder but never used in `_classify_heuristic`, so the
  documented edge-density gate on rain/snow does nothing. The heuristic is
  deprecated (weather labels now come from biome + subtitles, `trust_weather`
  off), so impact is nil — remove the dead threshold or wire it in if the
  heuristic is ever revived.
- **`vision/glyph_ocr.py` lowercase cap-offset at `ui_scale==1` (LOW).** The
  `13/4`-row heuristic mis-aligns lowercase-only lines at GUI scale 1; the
  project runs at scale 2, so no live impact. Derive the offset from the actual
  template ascender reserve if scale-1 is ever used.
