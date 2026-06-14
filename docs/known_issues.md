# Known issues — needs live validation before fixing

## make.py wooden_pickaxe — table PLACE → OPEN is the last blocker (FOCUSED FOLLOW-UP)

As of this session the pipeline works **except** placing the crafting table and
opening it. Validated live and SOLID: gather (find→walk→break→collect logs),
inventory read, 2×2 crafts (planks/sticks/table), and the 3×3 pickaxe craft in a
*manually-opened* table. The flaky last 10% is `tools/table_craft.py` +
`agents/skills.py:PlaceBlock`, which fails differently each run:

- **Aimer oscillation on place look-views** — the `_Aimer` overshoots (±, look
  =±140px) on a place view and never settles `aimed`, so `can_place_block` is
  never evaluated for that view and it times out without placing. Intermittent
  (pose/spot dependent). Likely needs a gentler gain / lower `max_px` for the
  fine place-aim, or accept "close enough" yaw.
- **Can't VISUALLY verify a freshly-placed table** — the recogniser garbles a
  fresh table's id and the open GUI blocks F3 targeting, so `PlaceBlock`'s
  verify never confirms → it keeps scanning → a later click opens the table →
  loop. PARTIALLY FIXED (`table_craft` now treats a GUI opening during the place
  step as proof-of-placement and crafts in the open table — commit d5b9137), but
  it only helps once it actually places; the oscillation above can stop it
  first.
- **Pre-existing tables ignored** — tables left in the world from prior runs
  aren't recognised/used; the bot places its own next to them. Test worlds get
  cluttered with tables, which confounds runs (break/clear them between tests).
- **Finding a placeable spot** — improved (replaceable-plant placement + a
  step-back to fresh ground + a shallower pitch), but cluttered chop-spots still
  make it scan many views.

**Recommended fix (focused session):** redesign table-craft to be DETERMINISTIC
instead of scan→visually-verify: (1) if a crafting_table is already within reach
/ mapped, walk to it and open it; else (2) clear/step to open ground, place once,
`wait`, right-click the placed voxel, and CONFIRM via the GUI-open (camera-frozen)
signal rather than the recogniser. Tune the place-aimer to not oscillate. Test
in a CLEAN area (no stray tables).

---

# (audit findings below) — needs live validation before fixing

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

## Live recogniser / perception (touch the running self-teach — validate live)

- **`vision/world/cnn_recognizer.py` incremental embeds can mix embedding
  spaces (HIGH, low-frequency).** `reload_incremental` → `_embed_into_index`
  embeds new samples with the *current* model and appends to `self._emb`; if a
  background retrain publishes a new model + rebuilds in between, appended
  old-space vectors mix with new-space ones, making cosine sims meaningless for
  those rows until the next full `_rebuild_index`. Self-healing but transiently
  noisy. Fix: capture model identity at embed time and, if it changed, mark the
  index dirty for a rebuild instead of appending. **Needs care** — it's the
  live self-teaching path; validate that the rebuild cadence isn't thrashed.
- **`vision/world/cnn_recognizer.py` texture-proto retrain thrash (MED).** If a
  real train bails (fewer than `min_blocks_to_train` trainable classes) it
  doesn't clear `_texture_proto`, so every `reload`/`reload_incremental`
  re-triggers a background train thread that bails again — CPU thrash in the
  cold-start (few-blocks) case. Fix: gate the texture-proto trigger in
  `_maybe_kickoff_training` on having ≥ `min_blocks_to_train` real classes
  (already computed nearby). Low impact once the store has enough blocks.
- **`vision/pose_filter.py` force-accept after `max_hold_seconds` (MED).** After
  a sustained reject streak the next read is accepted with only the hard
  y/pitch caps, no velocity check — an in-range-but-wrong XYZ jump can become
  the new ground truth ("teleport the eye"). The naive fix (velocity-check it)
  BREAKS the intended recovery after a long legitimate blind gap (the player
  really did move far). Correct fix: require N consecutive *mutually
  consistent* reads before re-trusting, rather than force-accepting one.
  Validate against busy/garbled scenes.

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
