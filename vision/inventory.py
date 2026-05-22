# vision/inventory.py
"""
Inventory perception — slot layout + item recognition.

Two pieces of machinery, glued together.

1. ``slot_rects(frame_shape, ui_scale)`` — Mojang's inventory GUI has a
   fixed pixel layout at GUI scale 1. The container background is
   176×166, drawn centred on screen, and every slot sits at a known
   offset within it. Multiplying through by the current ``ui_scale``
   gives the exact screen rect of every slot, for any resolution.

2. ``ItemRecognizer`` — given a slot crop, find the most likely
   ``minecraft:<item>`` identifier by template-matching against the
   item-texture library extracted by ``vision/mc_assets.py``. Compared
   against per-pixel alpha-masked MAE; cheap and very accurate for
   flat-textured items (tools, food, ingots, etc.).

Caveats
-------
* For *block items* (cobblestone, oak_log, dirt …) Minecraft renders an
  isometric 3D view of the block in the inventory, not the flat texture.
  We don't render isometric views here, so block-items return either
  the wrong match or a low-confidence hit. The recognizer reports a
  confidence so callers can ignore low-confidence results.
* Items with durability bars or custom NBT (enchantments, custom names)
  may look different from the bare texture; we match on the icon
  region only, so the durability bar at the bottom edge is masked out.

Future improvements (not yet built):
* Pre-render isometric block icons once and add them to the template
  database. Closes the cobblestone/dirt gap.
* OCR the small white stack-count number in the bottom-right of each
  slot — the existing glyph OCR can handle it with a tighter crop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Layout constants — all at GUI scale 1, multiplied by ui_scale at runtime.
# ---------------------------------------------------------------------------
#
# These positions are baked into Mojang's
# ``net.minecraft.client.gui.screens.inventory.InventoryScreen`` /
# ``Slot`` declarations and have been stable for many years. If they
# ever change in a future MC update, only this table needs updating.

# Background sprite size.
_BG_W_GUI = 176
_BG_H_GUI = 166

# Slot inner size (the actual item-icon area, before the 1 px border).
_SLOT_W_GUI = 16
_SLOT_H_GUI = 16
_SLOT_PITCH_GUI = 18    # 16 + 1 + 1 px border

# 4 armour slots stacked vertically on the top-left.
_ARMOR_OFFS = [
    ("armor_head",  ( 8,  8)),
    ("armor_chest", ( 8, 26)),
    ("armor_legs",  ( 8, 44)),
    ("armor_feet",  ( 8, 62)),
]

# Off-hand sits to the right of the legs.
_OFFHAND_OFF = ("offhand", (77, 62))

# 2×2 crafting grid (input) + result slot.
_CRAFT_OFFS = [
    ("craft_in_0",   ( 98, 18)),
    ("craft_in_1",   (116, 18)),
    ("craft_in_2",   ( 98, 36)),
    ("craft_in_3",   (116, 36)),
    ("craft_result", (154, 28)),
]

# Main inventory: 3 rows × 9 cols, starting at (8, 84).
_MAIN_TOP_OFF = (8, 84)
# Hotbar: single row of 9 at y=142.
_HOTBAR_OFF   = (8, 142)


SLOT_GROUP_INVENTORY: Tuple[str, ...] = (
    *("armor_head", "armor_chest", "armor_legs", "armor_feet", "offhand"),
    *("craft_in_0", "craft_in_1", "craft_in_2", "craft_in_3", "craft_result"),
    *tuple(f"inv_{i}"    for i in range(27)),
    *tuple(f"hotbar_{i}" for i in range( 9)),
)


def _all_slot_offsets() -> Dict[str, Tuple[int, int]]:
    """Return the GUI-scale-1 (x, y) offsets of every inventory slot."""
    out: Dict[str, Tuple[int, int]] = {}
    for name, off in _ARMOR_OFFS:
        out[name] = off
    out[_OFFHAND_OFF[0]] = _OFFHAND_OFF[1]
    for name, off in _CRAFT_OFFS:
        out[name] = off
    base_x, base_y = _MAIN_TOP_OFF
    for r in range(3):
        for c in range(9):
            out[f"inv_{r*9 + c}"] = (
                base_x + c * _SLOT_PITCH_GUI,
                base_y + r * _SLOT_PITCH_GUI,
            )
    hx, hy = _HOTBAR_OFF
    for i in range(9):
        out[f"hotbar_{i}"] = (hx + i * _SLOT_PITCH_GUI, hy)
    return out


_SLOT_OFFSETS = _all_slot_offsets()


@dataclass(frozen=True)
class SlotRect:
    name: str
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


def slot_rects(frame_shape: Tuple[int, int],
               ui_scale: int = 2) -> Dict[str, SlotRect]:
    """
    Compute the screen-pixel rect of every inventory slot for a given
    frame size and ``ui_scale``.

    ``frame_shape`` is the ``(H, W)`` (or ``(H, W, C)``) of the
    captured frame. We assume the inventory background is centred —
    which it is in all default MC GUI modes.
    """
    H, W = frame_shape[:2]
    bg_w = _BG_W_GUI * ui_scale
    bg_h = _BG_H_GUI * ui_scale
    bg_x = (W - bg_w) // 2
    bg_y = (H - bg_h) // 2
    sw   = _SLOT_W_GUI * ui_scale
    sh   = _SLOT_H_GUI * ui_scale

    out: Dict[str, SlotRect] = {}
    for name, (ox, oy) in _SLOT_OFFSETS.items():
        out[name] = SlotRect(
            name=name,
            x=bg_x + ox * ui_scale,
            y=bg_y + oy * ui_scale,
            w=sw,
            h=sh,
        )
    return out


# ---------------------------------------------------------------------------
# ItemRecognizer
# ---------------------------------------------------------------------------

@dataclass
class ItemMatch:
    item:       Optional[str]      # "minecraft:diamond" or None for empty
    confidence: float              # 0..1 (higher = better)
    score:      float              # raw matcher score (lower = better)
    second:     Optional[str] = None
    second_score: float = float("inf")


class ItemRecognizer:
    """
    Identify items by alpha-masked template matching.

    Build once at startup — loading 800+ 16×16 PNGs and packing them as
    a uniform tensor takes a few hundred ms. ``recognize(crop)`` is then
    a single masked-MAE pass over all templates and runs in a few ms.
    """

    # Empty-slot detection: when the slot crop has very few non-background
    # pixels (i.e. the slot background shows through), we declare it empty
    # rather than reporting whichever template happened to score lowest.
    _EMPTY_FILLED_PIXEL_RATIO = 0.05

    def __init__(self,
                 assets,                            # MCAssets
                 *,
                 include_items: bool = True,
                 include_block_topfaces: bool = True):
        """
        Parameters
        ----------
        assets : MCAssets
            From ``vision.mc_assets``.
        include_items : if True, load every 16×16 RGBA item texture.
        include_block_topfaces : if True, also include block top-face
            textures as templates (helps weakly identify block items
            even though they're rendered isometric in inventory).
        """
        self.assets = assets

        # Names → (rgba_uint8 16×16×4, alpha_mask_bool 16×16)
        templates: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        if include_items:
            for name in assets.list_item_textures():
                tex = assets.item_texture(name)
                rgba = _to_16x16_rgba(tex)
                if rgba is None:
                    continue
                templates[f"minecraft:{name}"] = rgba

        if include_block_topfaces:
            for name in assets.list_block_textures():
                tex = assets.block_texture(name)
                rgba = _to_16x16_rgba(tex)
                if rgba is None:
                    continue
                # Don't overwrite an existing item-texture entry with the
                # same name (an item-style icon beats a block-face top).
                key = f"minecraft:{name}"
                if key not in templates:
                    templates[key] = rgba

        self._names: List[str] = list(templates.keys())
        # Pack as N×16×16×4 tensor for vectorised distance computation.
        self._tpl_rgba = np.stack([templates[n][0] for n in self._names], axis=0)
        self._tpl_mask = np.stack([templates[n][1] for n in self._names], axis=0)
        # Per-template pixel count for normalisation.
        self._tpl_filled = self._tpl_mask.reshape(len(self._names), -1).sum(axis=1)
        # Avoid div-by-zero on a fully-transparent template (shouldn't happen).
        self._tpl_filled = np.maximum(self._tpl_filled, 1)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def template_count(self) -> int:
        return len(self._names)

    def recognize(self, crop_rgb: np.ndarray) -> ItemMatch:
        """
        Recognise a single inventory-slot crop.

        ``crop_rgb`` may be any size; we resize to 16×16 to match the
        templates' base resolution. We treat the input as opaque (no
        alpha) since the slot crop comes from the game's framebuffer.
        """
        if crop_rgb is None or crop_rgb.size == 0:
            return ItemMatch(item=None, confidence=0.0, score=float("inf"))

        # Drop alpha if present and resize to 16×16.
        if crop_rgb.ndim == 3 and crop_rgb.shape[2] == 4:
            crop_rgb = crop_rgb[:, :, :3]
        small = cv2.resize(crop_rgb, (16, 16), interpolation=cv2.INTER_AREA)

        # Empty-slot heuristic: compare against the dark slot background.
        # Slot backgrounds are uniformly dark; an empty crop has very low
        # variance and a mean RGB near (139,139,139)±something. Use std
        # to decide if there's "real content" in this slot.
        if small.std() < 5.0:
            return ItemMatch(item=None, confidence=0.95, score=0.0)

        # Compute per-template MAE on alpha-masked pixels.
        diff = np.abs(self._tpl_rgba[:, :, :, :3].astype(np.int32)
                      - small[None, :, :, :].astype(np.int32))
        per_pixel = diff.mean(axis=3)        # (N, 16, 16)
        masked    = per_pixel * self._tpl_mask  # zero where template alpha==0
        scores    = masked.reshape(len(self._names), -1).sum(axis=1) \
                  / self._tpl_filled                 # average over kept pixels

        order = np.argsort(scores)
        best, second = order[0], order[1]
        best_score = float(scores[best])
        second_score = float(scores[second])

        # Confidence: how much better the best is than the runner-up,
        # mapped into 0..1. A best-score of 0 against a runner-up of
        # 100 gives confidence 1.0; same score → 0.0.
        margin = max(0.0, second_score - best_score)
        confidence = float(min(1.0, margin / 30.0))

        return ItemMatch(
            item=self._names[best],
            confidence=confidence,
            score=best_score,
            second=self._names[second],
            second_score=second_score,
        )

    # ------------------------------------------------------------------
    # Bulk: read every slot in one go
    # ------------------------------------------------------------------

    def recognize_all(self,
                      frame_rgb: np.ndarray,
                      ui_scale: int = 2,
                      ) -> Dict[str, ItemMatch]:
        """
        Crop every inventory slot from ``frame_rgb`` and recognise each.

        Returns ``{slot_name: ItemMatch}``. Empty slots are reported as
        ``ItemMatch(item=None, ...)``.
        """
        out: Dict[str, ItemMatch] = {}
        for slot in slot_rects(frame_rgb.shape, ui_scale).values():
            x0, y0, w, h = slot.as_tuple()
            crop = frame_rgb[y0:y0 + h, x0:x0 + w]
            out[slot.name] = self.recognize(crop)
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_16x16_rgba(arr: Optional[np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Normalise a loaded texture to a (rgba, mask) pair at 16×16, or
    return None when the texture is not a useful identification template.

    A template is rejected if:
      * the array is malformed or zero-sized,
      * it's an animated strip we can't trivially de-strip,
      * after threshold its alpha mask covers fewer than 30 pixels
        (this filters the leather/redstone/etc "_overlay" tint masks
        that MC blends at render time — they look like a few opaque
        pixels and otherwise match every slot with zero score), or
      * the opaque region has near-zero colour variance (a uniform
        tint mask is useless for discrimination).
    """
    if arr is None or arr.size == 0:
        return None

    # Greyscale → RGBA.
    if arr.ndim == 2:
        rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        alpha = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
        arr = np.concatenate([rgb, alpha], axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        # RGB → add opaque alpha
        alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
        arr = np.concatenate([arr, alpha], axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        pass
    else:
        return None

    h, w = arr.shape[:2]
    # Skip animated textures (NxN tall strips: H >> W) — we'd want only
    # the first frame, but it's safer to just exclude them.
    if h != w:
        if h % w != 0 or h // w < 2:
            return None
        arr = arr[:w]   # take the first frame square
    if arr.shape[0] != 16:
        arr = cv2.resize(arr, (16, 16), interpolation=cv2.INTER_AREA)

    mask = (arr[:, :, 3] > 16).astype(np.uint8)
    opaque_count = int(mask.sum())
    if opaque_count < 30:
        return None

    # Reject low-variance tint masks (overlay/empty_armor_slot textures).
    opaque_pixels = arr[:, :, :3][mask.astype(bool)]
    if opaque_pixels.std() < 4.0:
        return None

    return arr.astype(np.uint8), mask


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_item_recognizer(settings: Optional[Dict] = None,
                          *,
                          assets=None) -> "ItemRecognizer":
    """
    Build a recogniser wired to the cached assets.

    ``settings`` may be None — only ``capture.ui_scale`` is read, and the
    recogniser itself doesn't need it (the caller passes ui_scale per
    ``recognize_all`` call).
    """
    if assets is None:
        from vision.mc_assets import MCAssets
        assets = MCAssets.load()
    return ItemRecognizer(assets=assets)
