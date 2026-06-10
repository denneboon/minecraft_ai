# vision/world/biome_tint.py
"""
Single source of truth for Minecraft's per-biome block tints.

Minecraft stores grass / foliage / water textures as grayscale-ish atlas
tiles and multiplies them by a per-biome colour at render time. So the
raw texture for ``grass_block_top`` is grey (~145,145,135) while the
in-world block is green — and which green depends on the biome.

Two consumers need this exact same knowledge and must not drift apart:

  * :mod:`vision.world.block_classifier` — adds one colour-signature
    variant per (tintable block, representative biome) so the baseline
    matcher recognises tinted surfaces.
  * the CNN texture pre-training (``tools/pretrain_block_cnn.py`` via
    :mod:`vision.world.cnn_recognizer`) — without tinting, the synthetic
    grass/foliage patches are grey and the resulting prototypes are
    useless against real green foliage (the documented "all green
    foliage looks alike" failure). Tinting the synthetic patches with
    these same biome colours closes that gap at the warm-start.

Tint values are the wiki's GAME-CANONICAL biome colours:
https://minecraft.wiki/w/Color#Biome_colors

When the learned recogniser fully supersedes the colour baseline this
table stays relevant for pre-training — it's render knowledge, not a
classifier hack.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# Blocks Minecraft tints with the per-biome GRASS colour.
GRASS_TINTED_BLOCKS: List[str] = [
    "grass_block",
    "short_grass",
    "tall_grass",
    "fern",
    "large_fern",
    "potted_fern",
    "sugar_cane",
]

# Blocks Minecraft tints with the per-biome FOLIAGE colour.
FOLIAGE_TINTED_BLOCKS: List[str] = [
    "oak_leaves",
    "jungle_leaves",
    "acacia_leaves",
    "dark_oak_leaves",
    "mangrove_leaves",
    "vine",
    "lily_pad",
]

# Birch + spruce leaves use FIXED tints, not biome colours.
FIXED_TINT_BLOCKS: List[Tuple[str, Tuple[int, int, int]]] = [
    ("birch_leaves",  (128, 167, 85)),
    ("spruce_leaves", ( 97, 153, 97)),
]

# Representative biome GRASS tints — covers the most common overworld
# biomes the AI is likely to play in.
BIOME_GRASS_TINTS: List[Tuple[str, Tuple[int, int, int]]] = [
    ("plains",         (145, 189, 89)),
    ("forest",         ( 89, 168, 67)),
    ("birch_forest",   (107, 165, 86)),
    ("taiga",          (134, 184, 127)),
    ("savanna",        (191, 183, 85)),
    ("jungle",         ( 89, 197, 30)),
    ("desert",         (191, 183, 85)),
    ("swamp",          (106, 112, 57)),
    ("dark_forest",    ( 80, 122, 50)),   # dimmer woods
]

# Representative biome FOLIAGE tints.
BIOME_FOLIAGE_TINTS: List[Tuple[str, Tuple[int, int, int]]] = [
    ("plains",         (119, 171, 47)),
    ("forest",         ( 89, 174, 50)),
    ("taiga",          (104, 158, 78)),
    ("jungle",         ( 48, 187, 10)),
    ("savanna",        (174, 164, 42)),
    ("swamp",          (106, 112, 57)),
    ("dark_forest",    ( 76, 142, 38)),
]


def apply_tint(rgba: np.ndarray, tint: Tuple[int, int, int]) -> np.ndarray:
    """Multiply ``rgba``'s RGB channels by ``tint / 255`` (Mojang's
    in-shader tint formula). Alpha, if present, is preserved."""
    if rgba is None:
        return rgba
    arr = rgba.astype(np.float32)
    if arr.ndim != 3:
        return rgba
    tint_arr = np.array(tint, dtype=np.float32) / 255.0
    arr[..., 0] *= tint_arr[0]
    arr[..., 1] *= tint_arr[1]
    arr[..., 2] *= tint_arr[2]
    return np.clip(arr, 0, 255).astype(np.uint8)


def tints_for_block(stem: str
                    ) -> List[Tuple[str, Tuple[int, int, int]]]:
    """Return ``[(label, (r,g,b)), …]`` of the tints that apply to this
    block stem, or ``[]`` if the block is not tinted. A grass-tinted
    block yields the biome-grass palette, a foliage-tinted block the
    biome-foliage palette, and birch/spruce a single fixed tint."""
    if stem in GRASS_TINTED_BLOCKS:
        return list(BIOME_GRASS_TINTS)
    if stem in FOLIAGE_TINTED_BLOCKS:
        return list(BIOME_FOLIAGE_TINTS)
    for fixed_stem, tint in FIXED_TINT_BLOCKS:
        if stem == fixed_stem:
            return [(fixed_stem, tint)]
    return []


def is_tintable(stem: str) -> bool:
    """True iff Minecraft applies a biome / fixed tint to this block."""
    return bool(tints_for_block(stem))


__all__ = [
    "GRASS_TINTED_BLOCKS", "FOLIAGE_TINTED_BLOCKS", "FIXED_TINT_BLOCKS",
    "BIOME_GRASS_TINTS", "BIOME_FOLIAGE_TINTS",
    "apply_tint", "tints_for_block", "is_tintable",
]
