# Bugs to fix — prioritized

Living list of known bugs. Detailed diagnoses for the crafting/inventory items
live in `docs/known_issues.md`; this file is the actionable, prioritized index.
Order is roughly "what unblocks the headline goal (wooden_pickaxe) first, then
inventory-read efficiency, then latent/audit findings".

## STATUS (2026-06-15)

All items below ADDRESSED in code (26+1 offline suites green). The P0/P1
crafting + inventory fixes still want a LIVE end-to-end `make.py wooden_pickaxe`
run to confirm on the real game (offline mocks can't reproduce capture timing):

- P0 #1 crafter stale-frame mid-chain — FIXED (settle + read-until-plannable retry)
- P0 #2 craft() over-reports — already had result-slot verification; confirmed
- P1 #3 cursor-restore-before-close — REMOVED
- P1 #4 stop checking crafting squares — inspector now skips `craft_*` hovers
- P1 #5 empty-slot caching — `__empty__` sentinel sample + session skip + consume
- P1 #6 branchy-tree collection — post-chop dwell to vacuum drops
- P2 cnn embed-space mixing — FIXED (model-version guard); texture-proto thrash
  + hud stale-relabel were ALREADY fixed; pose_filter force-accept — FIXED
  (re-acquisition streak, with new `test_pose_filter`); tag/ times / precip /
  glyph items — documented (no-/nil-impact).

---

## P0 — blocks `make.py wooden_pickaxe` (the headline goal)

### 1. 2×2 crafter fails mid-chain on a stale snapshot  ⟵ THE pickaxe blocker
- **Symptom (live):** in a full run the chain does `craft acacia_planks OK` →
  `craft crafting_table OK` → **`craft acacia_planks FAIL (can't craft from
  inventory)` while logs are still on hand**, which loses the wood margin so the
  final `craft wooden_pickaxe` fails too.
- **Cause:** `agents/crafting.py` `Crafter.craft` resolves `find_item_slot`
  against a snapshot taken *before* the previous craft moved stacks. After the
  table craft displaces items, the next craft reads stale slot state and bails
  (`plan_step` returns None on the pre-move snap).
- **Fix:** re-read the inventory after each craft/placement (or reserve carry
  slots so they can't alias an unplaced ingredient). Validate live — the wood
  budget (3 logs → 12 planks vs. 9 needed) is sufficient once no craft is wasted.

### 2. `craft()` over-reports success
- **Symptom:** returns `(True, "crafted …")` even when cells were under-filled
  and nothing was produced; only a later re-read notices the no-op.
- **Fix:** verify the result slot actually produced output before returning
  success. `agents/crafting.py`.

---

## P1 — inventory-read correctness + efficiency (user-reported)

### 3. Remove the useless "restore cursor before closing" step
- **Symptom:** before closing an inventory, the bot eases the OS cursor back to
  where it was before it opened — pointless (the screen is about to close and a
  synthetic bot doesn't care where the cursor "looks like" it was). Wastes time.
- **Location:** `agents/inventory_inspector.py:253` (`original_cursor =
  get_cursor_xy() …`) and the `finally` restore at lines ~308–312
  (`self._move_cursor_to(*original_cursor)`).
- **Fix:** drop the save/restore entirely (remove `restore_cursor` plumbing, or
  hard-default it off). Leave the cursor wherever the last hover left it.

### 4. Track the CRAFTING-grid slots in the ledger; never re-read them
- **Symptom:** the bot frequently re-checks the 2×2/3×3 crafting squares even
  right after opening the inventory or shift-crafting, instead of remembering
  what it put there.
- **Fix:** treat craft-grid cells like any other slot in the ledger
  (`agents/inventory_memory.py` — `count_regions` currently counts only
  `hotbar_*` / `inv_*`, ignoring `craft_in_*`). When the bot CRAFTS, subtract 1
  from every occupied craft cell (and add the known result) via `note_delta`, so
  it always knows the craft-slot contents without a re-read. This also feeds
  fix #1 (the crafter can trust the ledger instead of a stale re-read).

### 5. Cache "this slot is empty" so false-occupied slots stop being re-checked
- **Symptom:** slots that *look* non-empty but aren't (the armor + shield slots
  have a faint placeholder icon/background) repeatedly trip the empty-detector,
  get hovered to confirm they're empty, and are re-checked again next read.
- **Cause:** `vision/inventory.py` `EmptySlotDetector` (`_EMPTY_STD_THRESHOLD`)
  false-positives on the armor/shield placeholder art → slot reads "occupied" →
  hover → found empty → but nothing is learned, so it repeats.
- **Fix:** when a hovered slot resolves to EMPTY, save a screenshot/signature of
  that slot crop labelled "empty" (a sentinel class in the SampleStore /
  SampleRecognizer), so future reads match it and skip the slot. Mirrors the
  existing Phase-3 "record pre-hover pixels with the OCR'd label" path
  (`inventory_inspector.py` `_save_sample_from_frame`) — extend it to the empty
  case.

### 6. Drop collection is partial on branchy trees
- **Symptom:** `gather` counts BREAKS, not pickups; tall branchy acacia (sparse
  savanna) drops logs out of the ~1.5-block pickup radius, so ~1 of 3 is
  collected per pass (the maker re-gathers to make up, but it's slow).
- **Fix:** after chopping, do a short "vacuum" walk over the trunk base to sweep
  drops, and/or count COLLECTED (inventory delta) rather than blocks broken.
  Reliable already in straight-trunk oak/birch forest (drops fall at the feet).

---

## P2 — latent / audit findings (not on the hot path; see known_issues.md)

- **`control/inventory_control.py` `read_hud_hotbar` stale relabel** — overwrites
  a fresh HUD read with the remembered id while keeping the fresh count → a
  changed slot yields a confidently-wrong entry. Relabel only when consistent;
  else mark unknown.
- **`vision/world/cnn_recognizer.py` incremental embeds can mix embedding
  spaces** — `reload_incremental` appends current-model vectors to an index a
  background retrain may have rebuilt in a new space. Capture model identity at
  embed time; mark the index dirty on change instead of appending.
- **`vision/world/cnn_recognizer.py` texture-proto retrain thrash** — a train
  that bails (too few classes) doesn't clear `_texture_proto`, so every reload
  re-triggers a thread that bails again. Gate the trigger on having ≥
  `min_blocks_to_train` real classes.
- **`vision/pose_filter.py` force-accept after `max_hold_seconds`** — after a
  reject streak the next read is accepted on hard caps only, so an in-range wrong
  XYZ can become ground truth. Require N consecutive *mutually consistent* reads
  before re-trusting (don't just velocity-gate — that breaks legit long blind gaps).
- **`knowledge/catalog.py` `_tag_index` one-level but `items_in_tag` recurses** —
  `ItemInfo.tags` is non-transitive while `items_in_tag` is transitive. Make
  `_tag_index` a fixpoint, or document the asymmetry.
- **`knowledge/recipes.py` `CraftStep.times` always 1** — dead field; batched
  counts handled by callers. Wire up or document.
- **`vision/weather.py` `precip_edge_min` dead config** — defined + wired but
  never used in `_classify_heuristic`. Remove or wire in (heuristic is
  deprecated anyway — labels come from biome + subtitles).
- **`vision/glyph_ocr.py` lowercase cap-offset at `ui_scale==1`** — `13/4`-row
  heuristic mis-aligns lowercase-only lines at GUI scale 1 (project runs at
  scale 2, so no live impact). Derive the offset from the template ascender if
  scale-1 is ever used.
