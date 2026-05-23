# vision/inventory.py
"""
End-to-end inventory perception for the vision-only Minecraft AI.

This module is the AI's eyes on its bags. Given a screenshot taken while
*any* container screen is open (player inventory, chest, crafting
table, furnace, …) it returns an ``InventorySnapshot`` describing every
slot the player can see:

  * which item is in the slot (``minecraft:<id>`` or ``None`` if empty)
  * how many of it (stack count read from the slot's bottom-right number)
  * how worn it is (durability fraction read from the green→red bar at
    the bottom of damageable items, ``None`` for non-tools)
  * whether it's enchanted (glint detection via HSV)
  * confidence in the recognition (template-matching score margin)

It also reports:

  * which armor pieces are equipped, by slot
  * the off-hand item
  * any item being held by the mouse cursor (the one you're dragging)
  * the currently selected hotbar slot

Pipeline
--------
1. :class:`SlotLayout` (in ``vision/inventory_layout.py``) — compute
   slot rects for the active container.
2. :class:`ItemRecognizer` — alpha-mask template match against:
     * every flat 16×16 item texture (``textures/item/*.png``)
     * every pre-rendered isometric block icon
       (see ``vision/block_icons.py``)
3. :class:`StackCountReader` — glyph-OCR the digit string at the bottom
   right of each slot using the MC default-font templates.
4. :class:`DurabilityReader` — scan the 1-px bar at y=13 (GUI) of each
   slot; bar colour interpolates green→red with fill width = fraction.
5. :class:`GlintDetector` — bright violet ripple in HSV space.
6. :class:`CursorReader` — recognise the icon attached to the mouse
   cursor (no slot is "empty when held").
7. :class:`InventoryReader` — wires the above and surfaces a clean
   ``InventorySnapshot``.

Designed to be:

* **Long-term stable** — every layout / asset path is read from the
  cached game assets, so a Minecraft version bump only needs the assets
  cache to be re-extracted (``python -m vision.mc_assets --extract``).
* **Pure read** — nothing in here moves the mouse or presses a key.
  All state changes happen via the control/ stack.
* **Cheap** — the matching tensor is built once and lives in numpy; a
  full 41-slot scan takes a few ms on CPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from vision.inventory_layout import (
    SlotRect,
    slot_rects,
    background_rect,
    available_layouts,
    get_layout,
    ARMOR_SLOTS, HOTBAR_SLOTS, MAIN_SLOTS,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SlotContent:
    """
    What the recognizer found in a single slot.

    Fields
    ------
    item        : ``"minecraft:diamond_sword"`` or ``None`` if empty.
    count       : Stack count, OCR'd from the bottom-right number.
                  ``0`` when empty, ``1`` for unstacked single items
                  (MC omits the "1" digit on screen for non-stacks; we
                  imply it from a confident non-empty recognition).
    durability  : ``None`` if no bar shown; else fraction in ``[0, 1]``
                  (1.0 = brand-new, ~0.0 = about to break).
    enchanted   : True if the rainbow glint overlay was detected.
    confidence  : 0..1 — how confident we are in the ``item`` label.
                  Lower numbers mean "could be one of several visually
                  similar items"; callers can ignore matches below a
                  threshold.
    score       : Raw template-matching score (lower = better fit).
                  Exposed mostly for debugging.
    second      : The runner-up item id (useful when confidence is low).
    """
    item:       Optional[str]    = None
    count:      int              = 0
    durability: Optional[float]  = None
    enchanted:  bool             = False
    confidence: float            = 0.0
    score:      float            = float("inf")
    second:     Optional[str]    = None
    # Which stage identified this slot. Lets downstream code (and
    # operators) tell apart "I'm sure because this exactly matches a
    # real sample I recorded earlier" from "I'm sure because a synthetic
    # template matched" from "I OCR'd the tooltip and read it
    # literally". Values: "empty" | "placeholder" | "sample" | "vision"
    # | "hover" | "unknown".
    source:     str              = "unknown"

    @property
    def is_empty(self) -> bool:
        """True when the slot is truly empty (no item drawn).

        Distinguished from :attr:`is_unknown` by the score: an empty
        slot's std-dev was below the empty-detector threshold so we
        never even ran template matching, leaving score at +inf. An
        "unknown" slot has visible pixels but no template matched closely
        enough to trust the result.
        """
        return self.item is None and self.score == float("inf")

    @property
    def is_unknown(self) -> bool:
        """True when there's something in the slot but we couldn't
        confidently identify it — the slot is NOT empty, but the best
        template was too far away in pixel space to report."""
        return self.item is None and self.score != float("inf")

    def display_name(self, assets) -> Optional[str]:
        if self.item is None:
            return None
        return assets.display_name(self.item) or self.item


@dataclass
class InventorySnapshot:
    """
    A complete read of the open inventory screen.

    Slot keys are the names from :mod:`vision.inventory_layout` (e.g.
    ``"hotbar_3"``, ``"armor_chest"``, ``"chest_5"``, ``"craft_in_2"``).
    """
    container: str                              # layout name
    slots: Dict[str, SlotContent]               = field(default_factory=dict)

    # Item currently attached to the mouse cursor (None when nothing is held).
    cursor: Optional[SlotContent]               = None

    # Heuristic: 1..9. Filled in by the InventoryReader if a hotbar
    # selection-highlight is visible (it isn't when the inventory is
    # open and the hotbar is overlaid by it).
    selected_hotbar_slot: Optional[int]         = None

    # The raw frame's (H, W) and the ui_scale used. Useful for callers
    # that want to convert a slot's name back to a rect for clicking.
    frame_shape: Optional[Tuple[int, int]]      = None
    ui_scale:    int                            = 2

    # Free-form diagnostics (per-slot scores, dropped-empty heuristics …).
    diagnostics: Dict[str, Any]                 = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Slot-group views
    # ------------------------------------------------------------------

    @property
    def armor(self) -> Dict[str, SlotContent]:
        return {k: self.slots[k] for k in ARMOR_SLOTS if k in self.slots}

    @property
    def hotbar(self) -> List[SlotContent]:
        return [self.slots[k] for k in HOTBAR_SLOTS if k in self.slots]

    @property
    def main(self) -> List[SlotContent]:
        return [self.slots[k] for k in MAIN_SLOTS if k in self.slots]

    @property
    def offhand(self) -> Optional[SlotContent]:
        return self.slots.get("offhand")

    @property
    def crafting_grid(self) -> List[SlotContent]:
        out = []
        for i in range(9):                   # 0..8 for 3×3, first 4 used for 2×2
            k = f"craft_in_{i}"
            if k in self.slots:
                out.append(self.slots[k])
        return out

    @property
    def crafting_result(self) -> Optional[SlotContent]:
        return self.slots.get("craft_result")

    @property
    def container_slots(self) -> List[SlotContent]:
        """Return chest/furnace/etc. slots — everything that's not part of
        the standard player inventory."""
        excluded = (set(ARMOR_SLOTS) | set(HOTBAR_SLOTS) | set(MAIN_SLOTS)
                    | {"offhand", "craft_result"}
                    | {f"craft_in_{i}" for i in range(9)})
        return [v for k, v in self.slots.items() if k not in excluded]

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    def filled_slots(self) -> List[SlotContent]:
        return [s for s in self.slots.values() if not s.is_empty]

    def empty_slot_count(self) -> int:
        return sum(1 for s in self.slots.values() if s.is_empty)

    def total_count(self, item: str) -> int:
        """How many of ``item`` total across all slots."""
        if ":" not in item:
            item = "minecraft:" + item
        return sum(s.count for s in self.slots.values() if s.item == item)

    def has_item(self, item: str, *, at_least: int = 1) -> bool:
        return self.total_count(item) >= at_least

    def items_summary(self) -> Dict[str, int]:
        """Aggregate ``{item_id: total_count_across_all_slots}``.

        Skips both empty AND unknown slots — an unknown slot has no
        item id we could safely use as a key.
        """
        out: Dict[str, int] = {}
        for s in self.slots.values():
            if s.item is None:                      # empty OR unknown
                continue
            out[s.item] = out.get(s.item, 0) + s.count
        return out

    def first_slot_of(self, item: str) -> Optional[str]:
        """Return the slot key holding ``item``, or ``None``."""
        if ":" not in item:
            item = "minecraft:" + item
        for k, s in self.slots.items():
            if s.item == item:
                return k
        return None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "container": self.container,
            "ui_scale":  self.ui_scale,
            "frame_shape": list(self.frame_shape or []),
            "selected_hotbar_slot": self.selected_hotbar_slot,
            "cursor": _slot_to_jsonable(self.cursor),
            "slots":  {k: _slot_to_jsonable(v) for k, v in self.slots.items()},
            "items_summary": self.items_summary(),
        }


def _slot_to_jsonable(s: Optional[SlotContent]) -> Optional[dict]:
    if s is None:
        return None
    # JSON has no Infinity literal; encode +inf as null so json.dumps
    # produces strictly-spec-compliant output.
    score = (None if (s.score == float("inf") or s.score != s.score)
             else round(float(s.score), 3))
    return {
        "item":       s.item,
        "count":      s.count,
        "durability": (None if s.durability is None
                       else round(float(s.durability), 3)),
        "enchanted":  s.enchanted,
        "confidence": round(float(s.confidence), 3),
        "score":      score,
        "second":     s.second,
        "status":     ("empty" if s.is_empty
                       else "unknown" if s.is_unknown
                       else "identified"),
    }


# ---------------------------------------------------------------------------
# Item recognizer
# ---------------------------------------------------------------------------

@dataclass
class _RecogTemplate:
    """One entry in the recognizer's library."""
    item_id: str           # "minecraft:diamond"
    kind:    str           # "item" | "block_iso" | "block_face"
    rgba:    np.ndarray    # 16×16×4 uint8
    mask:    np.ndarray    # 16×16 bool — alpha > 16 pixels
    n_pixels: int          # mask.sum(); for normalisation


class ItemRecognizer:
    """
    Identify the item in a slot crop via alpha-masked template matching.

    Build the library once at startup (loads ~1000 16×16 PNGs into a
    single uint8 tensor); calling ``recognize(crop)`` is then a single
    vectorised distance computation across all templates and runs in
    a few milliseconds.

    Template sources, in priority order
    -----------------------------------
    1. Pre-rendered isometric block icons from
       :mod:`vision.block_icons`. These render Mojang's actual inventory
       view for block items (cobblestone, oak_log, …) which the flat
       block-face texture cannot.
    2. Flat item textures (``textures/item/<id>.png``). Apples, tools,
       ingots — these are drawn straight from PNG into the slot.
    3. (Fallback) Flat block face textures (``textures/block/<id>.png``).
       Only registered for blocks that didn't get an iso icon — gives the
       recognizer something to fall back on for blocks with non-cube
       geometry (saplings, fences, …).
    """

    # When a slot crop's std dev (over RGB) is below this, we declare
    # the slot empty rather than trusting whichever template scored best.
    # Empty slots are uniformly the dark grey slot background (~139,139,
    # 139), giving std ≈ 0–3 across the 16×16 area.
    _EMPTY_STD_THRESHOLD = 6.0

    # Pixels with mask=False contribute zero to the per-pixel diff. To
    # avoid a tiny-mask template winning by sheer luck, we discount its
    # score against the largest mask in the library (penalising matches
    # that only "see" a sliver of the slot).
    _MIN_MASK_FRACTION = 0.10

    # Slot background RGB (the grey panel MC draws inside each empty
    # slot, sampled from a 1.21.11 GUI-scale-2 capture). Where a
    # template's alpha is zero, the captured pixel SHOULD be this
    # colour — if it isn't, the actual item is bigger than the template
    # and we owe the score a penalty (see negative-evidence below).
    _SLOT_BG_RGB = (139, 139, 139)

    # Negative-evidence weight: how much we punish a template for being
    # "too small" to cover the actual item. A weight of 1.0 means a
    # transparent-where-item-actually-is pixel costs as much as an
    # opaque-but-wrong-colour pixel; 0.0 disables. 0.7 was tuned to keep
    # small-but-correct items like nuggets/arrows winning their slots
    # while killing sparse-mask templates (lightning_rod, tripwire_hook)
    # that previously "ghost-matched" to furnaces and concrete blocks.
    _NEG_EVIDENCE_WEIGHT = 0.7
    # If the captured pixel is within this MAE of the slot background,
    # we treat it as "background" — i.e. no negative evidence is
    # generated, no matter what the template says.
    _BG_TOLERANCE_MAE = 22.0

    # Two-stage "unknown" gate. The recognizer's score is per-pixel MAE
    # over a slot's opaque pixels; it splits naturally into two regimes:
    #
    #  * **Flat items** (apple, ingot, sword): the template IS what MC
    #    draws into the slot. Correct matches score 0–8, runner-ups
    #    score 15+, giving huge margins.
    #  * **Iso blocks** (cobblestone, oak_log, wool): our isometric
    #    renderer is a close-but-not-pixel-perfect approximation of
    #    Mojang's 3D inventory render. Correct matches score 15–30,
    #    runner-ups score 25–35, giving margins of 2–8.
    #
    # So a single absolute threshold either kills good iso matches
    # (low cap) or admits clearly-wrong flat matches (high cap). We use
    # both filters:
    #   * score must be below ``_MAX_RECOGNISABLE_SCORE`` (truly absurd
    #     fits are rejected unconditionally)
    #   * AND confidence — which is the score margin to the runner-up,
    #     normalised — must be at least ``_MIN_RECOGNISABLE_CONFIDENCE``
    #     (no clear winner = unknown, regardless of absolute score)
    _MAX_RECOGNISABLE_SCORE      = 18.0
    _MIN_RECOGNISABLE_CONFIDENCE = 0.35   # ≈ 8.75 MAE points of margin
    # Empirical thresholds (2026-05-23, real 1.21.11 captures at GUI=2):
    #   * Pixel-accurate flat-item matches (apple, ingot, sword): score 0–6,
    #     confidence 0.6–1.0 — pass easily.
    #   * Pixel-accurate iso-block matches (cobblestone, dirt, planks):
    #     score 5–14, confidence 0.4–0.9 — pass if the block model's
    #     iso render is close to Mojang's.
    #   * Borderline matches (look-alikes — black_concrete vs coal_block,
    #     pink_stained_glass vs shulker_box top): score 12–22,
    #     confidence 0.05–0.25 — REJECTED. These need the Phase-2
    #     hover-tooltip-OCR backstop to disambiguate.
    # The user's explicit preference: prefer "unknown" over a confident-
    # but-wrong id. Hover-OCR (see [[feedback-inventory-phases]]) will
    # confirm the unknowns later.

    def __init__(self,
                 assets,
                 *,
                 block_icon_cache=None,
                 include_items: bool = True,
                 include_block_face_fallback: bool = True):
        """
        Parameters
        ----------
        assets             : ``MCAssets`` instance.
        block_icon_cache   : ``BlockIconCache`` populated via
                             ``ensure_block_icon_cache(assets)``. If
                             ``None`` we build one here.
        include_items      : include all flat item textures (default on).
        include_block_face_fallback :
                             include flat block-face textures for blocks
                             that didn't render an iso icon. Cheap noise
                             insurance — won't override an iso match.
        """
        self.assets = assets

        # Lazy import to avoid a hard dep cycle.
        if block_icon_cache is None:
            from vision.block_icons import ensure_block_icon_cache
            block_icon_cache = ensure_block_icon_cache(assets)
        self._icon_cache = block_icon_cache

        # ── Whitelists: only register templates that are real items ────
        # Mojang ships ~500 block textures that are sub-components of
        # other blocks (oak_log_top, sculk_sensor_tendril_inactive,
        # bundle_open_back, leather_helmet_overlay, ...) and a handful
        # of "internal" blocks the player can never hold (tripwire,
        # exposed_lightning_rod weathering states, pitcher_crop growth
        # stages). They exist as model + texture files but are never
        # drawn standalone in an inventory slot. If we register them as
        # templates the recognizer will happily pick them for any blob
        # it can't otherwise identify.
        #
        # Filter passes:
        #   1. There must be a matching ``models/<kind>/<name>.json``
        #      (kills animation frames, overlays, texture layers).
        #   2. The block / item must have a localised name in en_us
        #      (kills internal block states the player can't hold).
        item_models  = self._list_model_stems(assets, "item")
        block_models = self._list_model_stems(assets, "block")
        named_items, named_blocks = self._lang_names(assets)

        templates: Dict[str, _RecogTemplate] = {}

        # ── 1. Block isometric icons (priority) ─────────────────────────
        block_names = _list_rendered_icons(block_icon_cache)
        for name in block_names:
            # Internal block states (e.g. ``exposed_lightning_rod``,
            # ``weathered_copper_bulb``) have a model and texture but
            # are never in a player's inventory — they're the in-world
            # representation. Skip them at the template level.
            if name not in named_blocks and name not in named_items:
                continue
            rgba = block_icon_cache.get(name)
            if rgba is None:
                continue
            norm = _to_template(rgba)
            if norm is None:
                continue
            arr, mask = norm
            templates[f"minecraft:{name}"] = _RecogTemplate(
                item_id=f"minecraft:{name}",
                kind="block_iso",
                rgba=arr, mask=mask, n_pixels=int(mask.sum()),
            )

        # ── 2. Flat item textures ──────────────────────────────────────
        if include_items:
            for name in assets.list_item_textures():
                if name not in item_models:
                    continue
                # Drop item models whose texture isn't actually shown
                # to the player (bundle open-state layers, pitcher crop
                # growth stages, etc.). The lang file is the
                # authoritative "is this an obtainable item" gate.
                if name not in named_items and name not in named_blocks:
                    continue
                tex = assets.item_texture(name)
                norm = _to_template(tex)
                if norm is None:
                    continue
                arr, mask = norm
                key = f"minecraft:{name}"
                # Items beat blocks in the library — a flat icon of, say,
                # "bone_meal" should win over the bone_block iso even if
                # both look similar at 16×16. (Items get drawn EXACTLY
                # from their png; blocks are rendered, so the iso is
                # only an approximation.)
                templates[key] = _RecogTemplate(
                    item_id=key, kind="item",
                    rgba=arr, mask=mask, n_pixels=int(mask.sum()),
                )

        # ── 3. Block face fallback ─────────────────────────────────────
        if include_block_face_fallback:
            for name in assets.list_block_textures():
                if name not in block_models:
                    continue
                if name not in named_blocks and name not in named_items:
                    continue
                key = f"minecraft:{name}"
                if key in templates:
                    continue
                tex = assets.block_texture(name)
                norm = _to_template(tex)
                if norm is None:
                    continue
                arr, mask = norm
                templates[key] = _RecogTemplate(
                    item_id=key, kind="block_face",
                    rgba=arr, mask=mask, n_pixels=int(mask.sum()),
                )

        # Pack the library as numpy tensors for vectorised matching.
        self._templates = list(templates.values())
        self._names = [t.item_id for t in self._templates]
        if not self._templates:
            raise RuntimeError("ItemRecognizer: no templates loaded — check that "
                               "the assets cache exists "
                               "(python -m vision.mc_assets --extract).")
        self._tpl_rgb  = np.stack([t.rgba[..., :3] for t in self._templates], axis=0)
        self._tpl_mask = np.stack([t.mask          for t in self._templates], axis=0)
        self._tpl_filled = np.maximum(
            np.array([t.n_pixels for t in self._templates], dtype=np.float32),
            1.0,
        )
        self._max_filled = float(self._tpl_filled.max())

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def template_count(self) -> int:
        return len(self._templates)

    def template_kinds(self) -> Dict[str, int]:
        """How many templates of each kind we have."""
        out: Dict[str, int] = {}
        for t in self._templates:
            out[t.kind] = out.get(t.kind, 0) + 1
        return out

    @staticmethod
    def _list_model_stems(assets, kind: str) -> set:
        """
        Return the set of model file stems under
        ``data/mc_assets/<version>/models/<kind>/`` (e.g. ``"apple"`` if
        ``models/item/apple.json`` exists). Used to whitelist which
        textures may be registered as recognition templates.
        """
        d = Path(assets.root) / "models" / kind
        if not d.is_dir():
            return set()
        return {p.stem for p in d.iterdir() if p.suffix == ".json"}

    @staticmethod
    def _lang_names(assets) -> Tuple[set, set]:
        """
        Pull the names that vanilla MC exposes to the player from
        ``lang/en_us.json``. Returns ``(item_names, block_names)`` where
        each set holds the bare id (no namespace, no prefix).

        These two sets together are the universe of things a player can
        see named in a tooltip — they're the right whitelist for the
        recognizer's templates.
        """
        L = assets.lang("en_us") or {}
        items, blocks = set(), set()
        for k in L:
            if k.startswith("item.minecraft."):
                items.add(k[len("item.minecraft."):])
            elif k.startswith("block.minecraft."):
                blocks.add(k[len("block.minecraft."):])
        return items, blocks

    def recognize(self, crop_rgb: np.ndarray) -> SlotContent:
        """
        Identify the item in a single slot crop. The crop is resized to
        16×16 internally so any input size is fine.
        """
        if crop_rgb is None or crop_rgb.size == 0:
            return SlotContent()

        if crop_rgb.ndim == 3 and crop_rgb.shape[2] == 4:
            crop_rgb = crop_rgb[..., :3]
        small = cv2.resize(crop_rgb, (16, 16), interpolation=cv2.INTER_AREA)

        # Empty-slot heuristic — the slot background is uniformly dark
        # grey, so an empty slot's std is near-zero.
        if float(small.std()) < self._EMPTY_STD_THRESHOLD:
            return SlotContent()

        # Per-pixel MAE between every template and the slot crop.
        diff = np.abs(self._tpl_rgb.astype(np.int32)
                      - small[None, ...].astype(np.int32))
        per_pixel_mae = diff.mean(axis=3)            # (N, 16, 16) in [0, 255]

        # ── Positive evidence ─────────────────────────────────────────
        # Where the template is opaque, the slot pixel must match the
        # template colour. Average MAE over those pixels.
        pos_sum = (per_pixel_mae * self._tpl_mask
                  ).reshape(len(self._templates), -1).sum(axis=1)
        pos_score = pos_sum / self._tpl_filled

        # ── Negative evidence ─────────────────────────────────────────
        # Where the template is transparent, the slot pixel SHOULD be
        # the slot background grey. If it's not, the actual item is
        # bigger than the template — proof the template is too small.
        #
        # This is the fix for "lightning_rod won hotbar_7 furnace":
        # lightning_rod's opaque pixels (~30 px) happened to align with
        # furnace pixels of similar grey, giving a low positive score —
        # but the FURNACE'S ~170 OTHER opaque pixels fell on the
        # template's transparent area. Negative evidence catches that:
        # those pixels are far from slot-grey, so we add the gap to the
        # score. The furnace template (which covers the same ~200 px)
        # adds zero negative penalty and wins outright.
        bg = np.array(self._SLOT_BG_RGB, dtype=np.int32)
        bg_diff = np.abs(small.astype(np.int32) - bg[None, None, :]
                        ).mean(axis=2)              # (16, 16)
        # Pixels close to background = no negative evidence.
        bg_violation = np.maximum(0.0, bg_diff - self._BG_TOLERANCE_MAE)  # (16, 16)
        neg_mask = (~self._tpl_mask).astype(np.float32)  # (N, 16, 16)
        n_neg_pixels = neg_mask.reshape(len(self._templates), -1).sum(axis=1)
        n_neg_pixels = np.maximum(n_neg_pixels, 1.0)
        neg_sum = (bg_violation[None, ...] * neg_mask
                  ).reshape(len(self._templates), -1).sum(axis=1)
        neg_score = neg_sum / n_neg_pixels

        scores = pos_score + self._NEG_EVIDENCE_WEIGHT * neg_score

        # Residual small-mask penalty for templates with so few pixels
        # that even the positive score isn't very meaningful.
        mask_frac = self._tpl_filled / self._max_filled
        penalty   = (1.0 - np.minimum(1.0, mask_frac / self._MIN_MASK_FRACTION)
                    ) ** 2 * 10.0
        scores = scores + penalty

        order = np.argsort(scores)
        best, second = int(order[0]), int(order[1])
        best_score   = float(scores[best])
        second_score = float(scores[second])

        # Confidence: margin between top two, mapped to 0..1.
        # A 25-point MAE gap is "obviously different"; 0 → totally
        # ambiguous. Clipped to [0, 1].
        margin = max(0.0, second_score - best_score)
        confidence = float(min(1.0, margin / 25.0))

        # Two-stage unknown gate (see _MAX_RECOGNISABLE_SCORE comment).
        if (best_score >= self._MAX_RECOGNISABLE_SCORE
                or confidence < self._MIN_RECOGNISABLE_CONFIDENCE):
            return SlotContent(
                item       = None,
                count      = 0,
                confidence = confidence,
                score      = best_score,
                second     = self._names[best],   # best guess for debug
            )

        return SlotContent(
            item       = self._names[best],
            count      = 0,                          # filled by StackCountReader
            confidence = confidence,
            score      = best_score,
            second     = self._names[second],
        )


# ---------------------------------------------------------------------------
# Stack count OCR
# ---------------------------------------------------------------------------

class StackCountReader:
    """
    Read the stack-count number that Minecraft draws in the lower-right
    of each slot when there's more than one item.

    Method
    ------
    Crop the bottom-right ``count_box_w × count_box_h`` GUI px of the
    slot (default 16 × 8), then run :class:`GlyphOCR` over it with the
    cached MC default-font templates. We pre-build a digit-only ``GlyphOCR``
    instance so we don't try to read 'B' or 'D' off a 2-digit stack.

    Drop-shadow handling
    --------------------
    MC draws the count in white with a 1-px black drop shadow at (+1, +1).
    Our binariser thresholds bright pixels, so the shadow falls below
    threshold and is ignored — no special handling needed.
    """

    # GUI px box, measured from the slot's bottom-right corner up/left.
    # MC draws the count using its default font (5 px wide × 7 px tall
    # for digits with a 1-px drop shadow at +1,+1). For a 2-digit count
    # the rendered string is 12 GUI px wide and 8 GUI px tall, starting
    # ~1 GUI px in from each edge. Box of 16×8 covers comfortably and
    # leaves a small margin so we don't slice off the leading digit.
    COUNT_BOX_W_GUI = 16
    COUNT_BOX_H_GUI = 8

    # A pixel counts as part of the count digit only when ALL THREE rgb
    # channels are above this brightness. MC draws the count in pure
    # (255, 255, 255); we accept down to ~230 to allow for any tiny
    # capture-side colour drift. Item icon pixels — even bright ones —
    # are almost always tinted (one channel lower than the other two)
    # and so don't pass this filter. Eliminates the "bell's yellow body
    # bleeds into the OCR" failure mode.
    _WHITE_CHANNEL_MIN = 230

    def __init__(self,
                 templates: Dict[str, np.ndarray],
                 *,
                 ui_scale: int = 2,
                 match_threshold: float = 0.78):
        # Restrict the glyph set to the digits — anything else is noise.
        digit_templates = {ch: tpl for ch, tpl in templates.items()
                           if ch in "0123456789"}
        if not digit_templates:
            raise ValueError("StackCountReader: no digit glyphs in templates")

        # Lazy import to keep the dependency clear.
        from vision.glyph_ocr import GlyphOCR, GlyphOCRConfig
        self._ocr = GlyphOCR(
            templates=digit_templates,
            config=GlyphOCRConfig(
                ui_scale=int(ui_scale),
                # The slot crop fed to the OCR is already pre-masked to
                # only "very white" pixels — see read() — so we can keep
                # the OCR's own brightness threshold low to admit every
                # masked pixel as glyph ink.
                text_threshold=64,
                match_threshold=float(match_threshold),
                space_min_gap_gui_px=99,   # disable space emission
            ),
        )
        self._ui_scale = max(1, int(ui_scale))

    def read(self, slot_crop_rgb: np.ndarray) -> int:
        """
        Return the stack count in ``slot_crop_rgb``. Returns 0 if no
        number is visible (which conventionally means "1 item" for items
        that stack and "1 item" for non-stackables too).
        """
        if slot_crop_rgb is None or slot_crop_rgb.size == 0:
            return 0
        h, w = slot_crop_rgb.shape[:2]
        box_w = self.COUNT_BOX_W_GUI * self._ui_scale
        box_h = self.COUNT_BOX_H_GUI * self._ui_scale
        x0 = max(0, w - box_w)
        y0 = max(0, h - box_h)
        crop = slot_crop_rgb[y0:, x0:]

        # Pre-mask: keep only pixels where every channel is > the white
        # threshold (count digits are pure white; item icon pixels are
        # tinted, so they get zeroed). Pass the masked greyscale image
        # to the OCR, which will then binarize at its much lower
        # text_threshold and pick up exactly the digit shapes.
        if crop.ndim == 3 and crop.shape[2] >= 3:
            r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
            white_mask = ((r >= self._WHITE_CHANNEL_MIN)
                          & (g >= self._WHITE_CHANNEL_MIN)
                          & (b >= self._WHITE_CHANNEL_MIN)).astype(np.uint8)
            masked = white_mask * 255          # 0 / 255 single-channel
        else:
            masked = crop

        text = self._ocr.recognize_line(masked)
        if not text:
            return 0
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            return 0
        try:
            return int(digits)
        except ValueError:
            return 0


# ---------------------------------------------------------------------------
# Durability bar reader
# ---------------------------------------------------------------------------

class DurabilityReader:
    """
    Detect the durability bar Minecraft draws across the bottom of a
    damageable item's slot, and report its fill fraction.

    Bar geometry (at GUI scale 1) — from Mojang's ItemRenderer
    -----------------------------------------------------------
    * Lives at ``y = 13`` (i.e. 13 GUI px from the slot's top edge).
    * Two rows tall:
       - row 13 (top): solid black background bar, 13 px wide,
                       starting at x=2.
       - row 14 (bottom): coloured fill, ``round(durability * 13)`` px
                       wide, in green→yellow→red.
    * The two-row signature is what distinguishes a durability bar from
      "any item that happens to have green pixels near its bottom edge".

    The previous implementation only looked at the coloured row and
    triggered on rails / minecarts / wool whose icons happen to have
    bright pixels near y=14. We now require BOTH:

      1. The fill row (y=14) is saturated colour.
      2. The background row (y=13) is dark (V < 50).

    Which together are >99 % specific to a real bar.
    """

    BAR_Y_GUI         = 13
    BAR_X_OFFSET_GUI  = 2
    BAR_WIDTH_GUI     = 13

    # HSV ranges
    _FILL_HSV_LOW   = (0,   100, 100)
    _FILL_HSV_HIGH  = (180, 255, 255)
    _BG_V_MAX       = 50          # background row must be near-black

    # Mask coverage thresholds (fraction of pixels matching).
    _FILL_MIN_COVERAGE = 0.30     # at least 30 % of fill row coloured
    _BG_MIN_COVERAGE   = 0.60     # at least 60 % of bg row dark

    def __init__(self, ui_scale: int = 2):
        self._scale = max(1, int(ui_scale))

    def read(self, slot_crop_rgb: np.ndarray) -> Optional[float]:
        """
        Return the durability fraction in ``[0, 1]``, or ``None`` if no
        bar appears to be present in this slot.
        """
        if slot_crop_rgb is None or slot_crop_rgb.size == 0:
            return None
        h, w = slot_crop_rgb.shape[:2]
        s = self._scale

        x0 = self.BAR_X_OFFSET_GUI * s
        y_bg   = self.BAR_Y_GUI * s
        y_fill = y_bg + s
        bw = self.BAR_WIDTH_GUI * s
        if y_fill + s > h or x0 + bw > w:
            return None

        bg_row   = slot_crop_rgb[y_bg   : y_bg + s,   x0 : x0 + bw]
        fill_row = slot_crop_rgb[y_fill : y_fill + s, x0 : x0 + bw]
        if bg_row.size == 0 or fill_row.size == 0:
            return None

        # Background-row check: must be mostly dark.
        bg_hsv = cv2.cvtColor(bg_row, cv2.COLOR_RGB2HSV)
        bg_dark = (bg_hsv[..., 2] < self._BG_V_MAX).astype(np.float32)
        if float(bg_dark.mean()) < self._BG_MIN_COVERAGE:
            return None

        # Fill-row check: at least some saturated coloured pixels.
        fill_hsv = cv2.cvtColor(fill_row, cv2.COLOR_RGB2HSV)
        lo = np.array(self._FILL_HSV_LOW,  dtype=np.uint8)
        hi = np.array(self._FILL_HSV_HIGH, dtype=np.uint8)
        fill_mask = cv2.inRange(fill_hsv, lo, hi)
        if float(fill_mask.mean()) / 255.0 < self._FILL_MIN_COVERAGE:
            return None

        # Now read the fill fraction: rightmost coloured column / bar_w.
        col_filled = fill_mask.any(axis=0)
        if not col_filled.any():
            return None
        last_filled_col = int(np.where(col_filled)[0][-1])
        gui_filled = (last_filled_col + 1) / s
        frac = min(1.0, gui_filled / self.BAR_WIDTH_GUI)
        return float(frac)


# ---------------------------------------------------------------------------
# Empty placeholder-slot detector (armour, off-hand, smithing inputs, …)
# ---------------------------------------------------------------------------

class EmptySlotDetector:
    """
    Recognise slots that are showing one of Minecraft's "ghost"
    placeholder icons — the faint outline of a helmet / chestplate /
    leggings / boots / shield drawn into an empty armour or off-hand
    slot, and the brewing-fuel / smithing-template hints drawn in
    similar wells on other containers.

    Why a dedicated detector
    ------------------------
    The general ``ItemRecognizer`` would happily match those placeholder
    icons against full item templates and report nonsense (``dead_horn_
    coral`` in the head slot, ``brown_mushroom_block`` in the off-hand)
    because the placeholders are sparse grey shapes that vaguely
    resemble a lot of items at 16×16. By comparing the slot crop
    against the EXACT placeholder texture FIRST and short-circuiting on
    a close match, we get a clean "this slot is empty" decision in
    constant time before any general matching runs.

    Sources
    -------
    Placeholder PNGs live in the cached game assets at
    ``textures/gui/sprites/container/slot/<name>.png``. The set the
    detector cares about is wired up per-container by
    :data:`_PLACEHOLDER_PER_SLOT`. Adding a new container kind (eg
    smithing table) is one line: add the slot key → PNG-stem mapping.
    """

    # Per slot-name → placeholder PNG stem under sprites/container/slot/.
    # The same slot key may appear in multiple container layouts; the
    # placeholder is the same texture in each case.
    _PLACEHOLDER_PER_SLOT: Dict[str, str] = {
        # Player inventory
        "armor_head":  "helmet",
        "armor_chest": "chestplate",
        "armor_legs":  "leggings",
        "armor_feet":  "boots",
        "offhand":     "shield",
        # Brewing stand
        "brew_fuel":       "brewing_fuel",
        "brew_ingredient": "potion",
        "brew_potion_0":   "potion",
        "brew_potion_1":   "potion",
        "brew_potion_2":   "potion",
        # Smithing table (slot names defined when we add that layout)
        # "smithing_template": "smithing_template_armor_trim",
    }

    # If the slot crop matches its placeholder to within this masked-MAE
    # threshold, we declare the slot empty. The placeholder is drawn at
    # ~50 % alpha so even an empty slot has slight pixel variation from
    # the bare slot background; a real item drawn over it shifts every
    # pixel substantially. 6 MAE is well above placeholder noise and
    # well below any "I think there's an item here" reading.
    _MATCH_MAX_MAE = 6.0

    def __init__(self, assets, *, ui_scale: int = 2):
        self._scale = max(1, int(ui_scale))
        self._templates: Dict[str, np.ndarray] = {}    # slot_name → 16×16 RGBA
        slot_root = (Path(assets.root) / "textures" / "gui"
                     / "sprites" / "container" / "slot")
        if not slot_root.is_dir():
            return
        # Same slot bg colour used by the negative-evidence matcher.
        bg = np.array(ItemRecognizer._SLOT_BG_RGB, dtype=np.uint8)
        for slot_name, stem in self._PLACEHOLDER_PER_SLOT.items():
            p = slot_root / f"{stem}.png"
            if not p.is_file():
                continue
            raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if raw is None:
                continue
            # Some placeholder PNGs ship as single-channel alpha (notably
            # helmet.png) rather than full RGBA. Promote everything to
            # 16×16 RGBA so the comparison code can treat all entries
            # uniformly.
            if raw.ndim == 2:
                # Grayscale: treat as ALPHA only; the visible colour
                # becomes the same dark grey MC actually uses for the
                # placeholder outline (sampled: ~55,55,55).
                alpha = raw
                rgb = np.full(alpha.shape + (3,), 55, dtype=np.uint8)
                rgba = np.dstack([rgb, alpha])
            elif raw.ndim == 3 and raw.shape[2] == 3:
                # Plain BGR → RGBA, fully opaque.
                rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                alpha = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
                rgba = np.concatenate([rgb, alpha], axis=2)
            elif raw.ndim == 3 and raw.shape[2] == 4:
                rgba = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA)
            else:
                continue
            # Resize to 16×16 if needed. Most are already 16×16.
            if rgba.shape[:2] != (16, 16):
                rgba = cv2.resize(rgba, (16, 16),
                                  interpolation=cv2.INTER_AREA)
            # Composite the placeholder onto the slot's grey background
            # so we get the EXPECTED appearance of an empty slot — that's
            # what we compare against later. Placeholder alpha is
            # typically ~120/255.
            alpha = rgba[..., 3:4].astype(np.float32) / 255.0
            expected = (rgba[..., :3].astype(np.float32) * alpha
                        + bg.astype(np.float32) * (1.0 - alpha))
            self._templates[slot_name] = expected.clip(0, 255).astype(np.uint8)

    def slots_with_placeholders(self) -> List[str]:
        return list(self._templates.keys())

    def is_empty(self, slot_name: str, crop_rgb: np.ndarray) -> bool:
        """
        Return True if ``crop_rgb`` looks like the empty-placeholder
        appearance for ``slot_name``. False for any unknown slot name
        or any slot whose contents differ from the placeholder.
        """
        tpl = self._templates.get(slot_name)
        if tpl is None or crop_rgb is None or crop_rgb.size == 0:
            return False
        if crop_rgb.ndim == 3 and crop_rgb.shape[2] == 4:
            crop_rgb = crop_rgb[..., :3]
        small = cv2.resize(crop_rgb, (16, 16), interpolation=cv2.INTER_AREA)
        mae = float(np.abs(small.astype(np.int32)
                           - tpl.astype(np.int32)).mean())
        return mae < self._MATCH_MAX_MAE


# ---------------------------------------------------------------------------
# Enchantment glint detector
# ---------------------------------------------------------------------------

class GlintDetector:
    """
    Detect the rainbow / violet "enchantment glint" overlay MC draws on
    enchanted items. The overlay scrolls across the icon, so a single
    frame catches only a slice — we use a forgiving threshold.

    Method
    ------
    Convert to HSV, count pixels in the violet hue band that are also
    bright (high V). An unenchanted item has near-zero such pixels;
    enchanted items show a 10-25 % fraction depending on timing.
    """

    # Violet/purple in OpenCV HSV (hue 0..180): roughly 125..160.
    _GLINT_HSV_LOW  = (125,  80, 160)
    _GLINT_HSV_HIGH = (170, 255, 255)

    _MIN_GLINT_FRACTION = 0.05   # 5 % of icon pixels = "enchanted"

    def detect(self, slot_crop_rgb: np.ndarray) -> bool:
        if slot_crop_rgb is None or slot_crop_rgb.size == 0:
            return False
        hsv = cv2.cvtColor(slot_crop_rgb, cv2.COLOR_RGB2HSV)
        lo = np.array(self._GLINT_HSV_LOW,  dtype=np.uint8)
        hi = np.array(self._GLINT_HSV_HIGH, dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        return float(mask.mean() / 255.0) >= self._MIN_GLINT_FRACTION


# ---------------------------------------------------------------------------
# High-level InventoryReader
# ---------------------------------------------------------------------------

class InventoryReader:
    """
    Wire :class:`ItemRecognizer`, :class:`StackCountReader`,
    :class:`DurabilityReader`, and :class:`GlintDetector` together into
    a single ``read(frame, container=...)`` call returning an
    ``InventorySnapshot``.

    Construct once per run; call ``read`` each time you need a fresh
    snapshot. The heavy lifting (loading ~1500 templates, building the
    block-icon cache) happens in the constructor.
    """

    def __init__(self,
                 assets,
                 *,
                 ui_scale: int = 2,
                 font_templates: Optional[Dict[str, np.ndarray]] = None,
                 block_icon_cache=None):
        self.assets = assets
        self.ui_scale = max(1, int(ui_scale))

        self.recognizer = ItemRecognizer(assets,
                                         block_icon_cache=block_icon_cache)
        self.durability = DurabilityReader(ui_scale=self.ui_scale)
        self.glint      = GlintDetector()
        # Empty placeholder slots (armour, off-hand, brewing wells …)
        # are matched against the exact MC placeholder texture BEFORE
        # falling into the general recogniser. See EmptySlotDetector
        # docstring for why this short-circuit matters.
        self.empty_slot = EmptySlotDetector(assets, ui_scale=self.ui_scale)
        # NN matcher against the on-disk store of confirmed real
        # captures. Empty on first run; grows every time the Phase-2
        # inspector OCRs a tooltip. Tried BEFORE synthetic template
        # matching because real captures beat synthetic templates for
        # any item MC renders through code we don't simulate (shulker
        # boxes, banners, chests, chains, potions, decorated pots…).
        from vision.sample_store import build_sample_store
        from vision.nn_recognizer import build_sample_recognizer
        self.sample_store = build_sample_store()
        self.sample_recog = build_sample_recognizer(self.sample_store)

        # Stack count is the only piece that needs the MC font; the
        # caller can pass the cached templates in to avoid re-extracting.
        if font_templates is None:
            from vision.mcfont import ensure_font_cache
            cache_path = (Path(assets.root).resolve().parent.parent.parent
                          / "data" / "calibration" / "mc_font.npz")
            font_templates = ensure_font_cache(str(cache_path))
        self.count = StackCountReader(font_templates, ui_scale=self.ui_scale)

    # ------------------------------------------------------------------

    def read(self, frame_rgb: np.ndarray,
             *,
             container: str = "player_inventory",
             mouse_xy: Optional[Tuple[int, int]] = None,
             ) -> InventorySnapshot:
        """
        Read every visible slot for ``container``.

        Parameters
        ----------
        frame_rgb   : full captured RGB frame (any size).
        container   : layout key, see ``inventory_layout.available_layouts()``.
        mouse_xy    : optional cursor (x, y) in screen pixels — if given,
                      we also recognise the icon held by the cursor.
        """
        snap = InventorySnapshot(
            container=container,
            frame_shape=tuple(frame_rgb.shape[:2]),
            ui_scale=self.ui_scale,
        )
        rects = slot_rects(frame_rgb.shape, layout=container,
                           ui_scale=self.ui_scale)

        for name, slot in rects.items():
            crop = frame_rgb[slot.y:slot.y + slot.h,
                             slot.x:slot.x + slot.w]
            # 1. Short-circuit on placeholder slots (armour, off-hand,
            # brewing fuel/ingredient).
            if self.empty_slot.is_empty(name, crop):
                snap.slots[name] = SlotContent(source="placeholder")
                continue

            # 2. NN match against real captured samples — these
            # describe Mojang's actual renderer output, so they beat
            # any synthetic template for items whose icon doesn't come
            # from a simple flat PNG or full-cube projection.
            content = self.sample_recog.recognize(crop)
            if content is not None:
                content.source = "sample"
            else:
                # 3. Synthetic templates — works for flat items + cube
                # blocks; everything else falls through to "unknown".
                content = self.recognizer.recognize(crop)
                if content.is_empty:
                    content.source = "empty"
                elif content.item is None:
                    content.source = "unknown"
                else:
                    content.source = "vision"

            # Count / durability / glint applies whenever the slot has
            # ANY pixels (identified by either matcher OR marked
            # unknown). True empty slots skip these as a perf win.
            if not content.is_empty:
                content.count      = max(1, self.count.read(crop))
                content.durability = self.durability.read(crop)
                content.enchanted  = self.glint.detect(crop)
            snap.slots[name] = content

        if mouse_xy is not None:
            snap.cursor = self._read_cursor(frame_rgb, mouse_xy)

        return snap

    # ------------------------------------------------------------------

    def _read_cursor(self, frame_rgb: np.ndarray,
                     mouse_xy: Tuple[int, int]) -> Optional[SlotContent]:
        """
        MC draws the cursor-held item icon centred on the mouse position.
        Crop a 16×ui_scale square centred there and recognise it.
        """
        mx, my = int(mouse_xy[0]), int(mouse_xy[1])
        s = 16 * self.ui_scale
        H, W = frame_rgb.shape[:2]
        x0 = max(0, mx - s // 2)
        y0 = max(0, my - s // 2)
        x1 = min(W, x0 + s)
        y1 = min(H, y0 + s)
        if x1 <= x0 or y1 <= y0:
            return None
        crop = frame_rgb[y0:y1, x0:x1]
        content = self.recognizer.recognize(crop)
        if content.is_empty:
            return None
        content.count      = max(1, self.count.read(crop))
        content.durability = self.durability.read(crop)
        content.enchanted  = self.glint.detect(crop)
        return content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_template(arr: Optional[np.ndarray]
                ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Normalise a loaded texture / icon into ``(rgba 16x16, mask 16x16 bool)``.

    Rejects:
      * malformed inputs
      * uniformly-coloured "tint overlays" (leather_chestplate_overlay,
        redstone_dust_overlay) — those textures are tinted by MC at
        render time and otherwise match every slot with score 0.
    """
    if arr is None or arr.size == 0:
        return None

    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGBA)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2RGBA)
    elif arr.ndim != 3 or arr.shape[2] != 4:
        return None

    h, w = arr.shape[:2]
    if h != w:
        if w == 0 or h % w != 0:
            return None
        arr = arr[:w]
    if arr.shape[0] != 16:
        arr = cv2.resize(arr, (16, 16), interpolation=cv2.INTER_AREA)
    arr = arr.astype(np.uint8)

    mask_bool = arr[..., 3] > 16
    if int(mask_bool.sum()) < 24:
        return None

    # Reject uniform-colour tint masks.
    opaque = arr[..., :3][mask_bool]
    if opaque.std() < 4.0:
        return None

    return arr, mask_bool


def _list_rendered_icons(cache) -> List[str]:
    """Wrapper for ``vision.block_icons.list_rendered_icons``."""
    from vision.block_icons import list_rendered_icons
    return list_rendered_icons(cache)


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------

def build_item_recognizer(settings: Optional[Dict] = None,
                          *,
                          assets=None) -> "ItemRecognizer":
    """
    Build an ``ItemRecognizer`` wired to the cached assets and the
    block-icon cache. ``settings`` is accepted but currently unused —
    the recognizer doesn't read configuration today.
    """
    if assets is None:
        from vision.mc_assets import MCAssets
        assets = MCAssets.load()
    return ItemRecognizer(assets)


def build_inventory_reader(settings: Optional[Dict] = None,
                           *,
                           assets=None,
                           font_templates: Optional[Dict[str, np.ndarray]] = None,
                           ) -> InventoryReader:
    """Build a fully-wired ``InventoryReader`` from project settings."""
    if assets is None:
        from vision.mc_assets import MCAssets
        assets = MCAssets.load()
    cap = (settings or {}).get("capture", {}) or {}
    ui_scale = int(cap.get("ui_scale", 2))
    return InventoryReader(assets,
                           ui_scale=ui_scale,
                           font_templates=font_templates)


__all__ = [
    "SlotContent", "InventorySnapshot",
    "ItemRecognizer", "StackCountReader",
    "DurabilityReader", "GlintDetector",
    "InventoryReader",
    "build_item_recognizer", "build_inventory_reader",
    # Re-exports from the layout module — kept here for backwards
    # compatibility with callers that used to import slot_rects from
    # vision.inventory.
    "slot_rects", "SlotRect", "ARMOR_SLOTS", "HOTBAR_SLOTS", "MAIN_SLOTS",
]
