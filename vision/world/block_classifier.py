# vision/world/block_classifier.py
"""
Turn a small patch of the rendered world view into a block-id guess.

Strategy
--------
This module ships with a *baseline* colour-signature classifier:

1. Once at startup we build a signature per vanilla block from its
   primary face texture (``data/mc_assets/<v>/textures/block/*.png``).
   The signature is a fixed-size vector that compresses the colour
   distribution of the texture (mean RGB + per-channel std + a small
   HSV histogram), which is fast to compute and reasonably
   discriminative for "what biome surface am I looking at".
2. Per query, we compute the same signature for the input patch and
   pick the block whose signature has the smallest L1 distance.
3. Confidence is the *margin* between the top and runner-up scores
   normalised to [0, 1].

It's not a CNN — it WILL confuse visually similar surfaces
(``stone`` vs ``deepslate``, ``grass_block`` vs ``moss_block``). The
point is to give the AI a usable first signal today and a clean place
to swap in a real CNN-based classifier later. The classifier protocol
below defines the swap point.

Plugging in a stronger model
----------------------------
Any callable matching :class:`BlockClassifierProtocol` (i.e. has
``classify(patch_rgb) -> (Optional[block_id], confidence)``) is
accepted by :class:`vision.world.perception.WorldPerception`. A neural
classifier just has to load its weights once at init and implement
that single method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Protocol any classifier implementation must satisfy
# ---------------------------------------------------------------------------

class BlockClassifierProtocol(Protocol):
    """Common interface for every block classifier."""

    def classify(self, patch_rgb: np.ndarray) -> Tuple[Optional[str], float]:
        ...

    def template_count(self) -> int:
        ...


# ---------------------------------------------------------------------------
# Colour-signature baseline
# ---------------------------------------------------------------------------

# Signature layout (28 floats):
#   [0:3]  mean RGB           (3)
#   [3:6]  std  RGB           (3)
#   [6:14] HSV-hue histogram  (8 bins, 0..180 → 8 buckets)
#   [14:22] HSV-saturation histogram (8 bins)
#   [22:28] V (luma) histogram (6 bins)

_SIG_LEN = 28
_HUE_BINS = 8
_SAT_BINS = 8
_V_BINS   = 6


def _signature_from_rgb(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Compute a 28-d colour signature from an RGB(A) image.

    RGBA inputs honour the alpha channel: fully transparent pixels are
    dropped before the statistics are computed (otherwise a sparse
    sapling texture would average to "mostly transparent grey").
    """
    if rgb is None or rgb.size == 0:
        return None

    if rgb.ndim == 3 and rgb.shape[2] == 4:
        alpha = rgb[..., 3]
        if not alpha.any():
            return None
        rgb_full = rgb[..., :3]
        keep = alpha > 16
        if int(keep.sum()) < 16:
            return None
        pixels = rgb_full[keep]
    elif rgb.ndim == 3 and rgb.shape[2] == 3:
        pixels = rgb.reshape(-1, 3)
    elif rgb.ndim == 2:
        # Greyscale → fake an RGB.
        g = rgb.reshape(-1, 1)
        pixels = np.concatenate([g, g, g], axis=1)
    else:
        return None

    pixels = pixels.astype(np.float32)
    mean = pixels.mean(axis=0)
    std  = pixels.std(axis=0)

    # HSV histograms (compute once on the full image rather than the
    # masked pixel array — OpenCV doesn't take a mask arg cleanly here,
    # so we convert then histogram over the *kept* indices.)
    if rgb.ndim == 3:
        small = rgb[..., :3].astype(np.uint8)
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).reshape(-1, 3)
        if rgb.shape[2] == 4:
            keep_flat = (rgb[..., 3] > 16).reshape(-1)
            hsv = hsv[keep_flat]
    else:
        # Greyscale fallback.
        hsv = np.zeros((pixels.shape[0], 3), dtype=np.uint8)
        hsv[:, 2] = pixels[:, 0].astype(np.uint8)

    h_hist, _ = np.histogram(hsv[:, 0], bins=_HUE_BINS, range=(0, 180))
    s_hist, _ = np.histogram(hsv[:, 1], bins=_SAT_BINS, range=(0, 256))
    v_hist, _ = np.histogram(hsv[:, 2], bins=_V_BINS,   range=(0, 256))
    total = max(1, hsv.shape[0])
    h_hist = h_hist.astype(np.float32) * (255.0 / total)
    s_hist = s_hist.astype(np.float32) * (255.0 / total)
    v_hist = v_hist.astype(np.float32) * (255.0 / total)

    sig = np.concatenate([mean, std, h_hist, s_hist, v_hist]).astype(np.float32)
    assert sig.shape[0] == _SIG_LEN
    return sig


@dataclass
class _BlockSignature:
    block_id:  str
    signature: np.ndarray


# ---------------------------------------------------------------------------
# Biome-tint variants
# ---------------------------------------------------------------------------
# Minecraft applies a per-biome multiplicative tint to a small set of
# blocks (grass / foliage / water) at render time. The raw atlas
# textures are grayscale-ish, so the in-world appearance differs
# dramatically from a flat texture match. We don't have access to the
# player's biome at signature-build time, so we add an extra signature
# variant per tintable block for each "representative" biome. The
# classifier then accepts whichever variant happens to match the
# current view.
#
# The tint tables + formula live in :mod:`vision.world.biome_tint` so the
# CNN texture pre-training shares the exact same render knowledge (they
# must not drift). Aliased here under the historic private names to keep
# the rest of this module unchanged.
from vision.world.biome_tint import (
    GRASS_TINTED_BLOCKS as _GRASS_TINTED_BLOCKS,
    FOLIAGE_TINTED_BLOCKS as _FOLIAGE_TINTED_BLOCKS,
    FIXED_TINT_BLOCKS as _FIXED_TINT_BLOCKS,
    BIOME_GRASS_TINTS as _BIOME_GRASS_TINTS,
    BIOME_FOLIAGE_TINTS as _BIOME_FOLIAGE_TINTS,
    apply_tint as _apply_tint,
)


class ColourSignatureBlockClassifier:
    """
    Baseline block classifier — colour-signature nearest neighbour.

    Construction is fast (<200 ms for vanilla 1.21's ~1200 block
    textures). ``classify`` runs in a few hundred microseconds.
    """

    # Tuned against typical first-person captures. World surfaces are
    # noisier than slot icons (lighting + shadows + biome tint), so the
    # gates here are more permissive than the inventory recogniser's.
    # ``_MAX_DIST`` was bumped from 110 → 220 after profiling: tinted
    # in-world grass scores ~70-90 against its closest tinted variant,
    # while clearly-wrong matches sit at 250+. Splitting the cap at
    # 220 keeps the discriminator while admitting real surfaces.
    _MAX_DIST   = 220.0
    _MIN_MARGIN_CONFIDENCE = 0.05

    def __init__(self, assets, *, include_block_face_only: bool = True):
        self.assets = assets
        self._signatures: List[_BlockSignature] = []

        # ── 1. Raw atlas signatures (untinted, all blocks) ────────────
        # Face-stem suffixes the asset jar uses for multi-face blocks
        # (top / side / bottom / front / back / inner). Each face has
        # its own texture file (e.g. ``dirt_path_top.png``,
        # ``stripped_oak_log_side.png``), so the classifier needs ALL
        # faces for matching. But the COMMITTED block_id must be the
        # canonical block name (``dirt_path``), not the face stem
        # (``dirt_path_top``) — otherwise the WorldMap fills with
        # bogus ids like ``minecraft:dirt_path_top`` that no agent /
        # exporter / pathfinder will recognise.
        # MC's asset jar names multi-face textures with a few
        # canonical suffixes. We strip them to the base block name
        # so the classifier reports e.g. ``minecraft:hopper`` instead
        # of ``minecraft:hopper_outside``. The list grew in waves as
        # live tests surfaced new leaks:
        #   * ``_outside`` / ``_inside``: hopper + mushroom_block.
        #   * ``_pivot`` / ``_round``: grindstone parts.
        # Texture stems that don't follow the convention (e.g.
        # ``<wood>_shelf`` for the chiseled bookshelf) fall through
        # unchanged — they'd need a hand-curated alias which is
        # tracked separately.
        _FACE_SUFFIXES = (
            "_top", "_side", "_bottom", "_front", "_back",
            "_inner", "_inner_top", "_inner_bottom",
            "_side_overlay", "_top_overlay",
            "_outside", "_inside",
            "_pivot", "_round",
        )
        all_stems = set(assets.list_block_textures())
        # Build a base-stem index: a stem `X_top` and `X_side` BOTH
        # collapse to block `X` even when bare `X.png` doesn't exist
        # (true for dirt_path, mangrove_roots, brown_stained_glass_pane,
        # etc. — multi-face blocks that have no single canonical face).
        # Detection rule: ``X`` is a real block iff at least one of
        # ``X``, ``X_top``, ``X_side``, ``X_bottom`` exists AND ``X``
        # isn't itself a face stem (doesn't end in a face suffix).
        bases_set = set()
        for stem in all_stems:
            if any(stem.endswith(suf) for suf in _FACE_SUFFIXES):
                continue
            bases_set.add(stem)
        for stem in all_stems:
            for suf in sorted(_FACE_SUFFIXES, key=len, reverse=True):
                if stem.endswith(suf):
                    base = stem[: -len(suf)]
                    if base:
                        bases_set.add(base)
                    break

        for stem in assets.list_block_textures():
            tex = assets.block_texture(stem)
            sig = _signature_from_rgb(tex) if tex is not None else None
            if sig is None:
                continue
            # Reject textures with virtually zero saturation+variance —
            # those tend to be tint overlays (leather_chestplate_overlay,
            # foliage_overlay) that aren't surfaces on their own.
            if float(sig[3:6].sum()) < 4.0:
                continue
            # Normalise face-stems to the base block name. Try the
            # longest matching suffix first so ``_inner_top`` is
            # stripped before ``_top``. We accept any base in
            # ``bases_set`` — that includes bases that only exist
            # as face textures (no bare PNG).
            canonical = stem
            for suf in sorted(_FACE_SUFFIXES, key=len, reverse=True):
                if stem.endswith(suf):
                    candidate = stem[: -len(suf)]
                    if candidate and candidate in bases_set:
                        canonical = candidate
                        break
            self._signatures.append(_BlockSignature(
                block_id=f"minecraft:{canonical}",
                signature=sig,
            ))

        # ── 2. Biome-tinted variants (grass + foliage blocks) ─────────
        # Without these, a plains-biome grass surface (RGB ≈ 55,110,35
        # in-world) is many MAE units away from raw grass_block_top.png
        # (RGB ≈ 145,145,135) and gets rejected by the classifier.
        for stem in _GRASS_TINTED_BLOCKS:
            self._add_tinted_variants(stem, _BIOME_GRASS_TINTS)
        for stem in _FOLIAGE_TINTED_BLOCKS:
            self._add_tinted_variants(stem, _BIOME_FOLIAGE_TINTS)
        for stem, tint in _FIXED_TINT_BLOCKS:
            self._add_one_tinted_variant(stem, tint, label="fixed")

        if not self._signatures:
            raise RuntimeError("ColourSignatureBlockClassifier: no signatures "
                               "built — check that the asset cache exists.")
        self._sig_matrix = np.stack([s.signature for s in self._signatures], axis=0)
        self._ids        = [s.block_id for s in self._signatures]

    # ── Tinted variant helpers ────────────────────────────────────

    def _add_tinted_variants(self,
                             stem: str,
                             tints: List[Tuple[str, Tuple[int, int, int]]],
                             ) -> None:
        # grass_block uses the SIDE texture for the lower half of the
        # block and the TOP texture for the top; we add both with the
        # same tint so a grass-top patch matches whichever face the AI
        # is looking at. Other blocks fall back to the base stem.
        candidates: List[str] = [stem]
        if stem == "grass_block":
            candidates = ["grass_block_top", "grass_block_side_overlay"]
        for base in candidates:
            for biome, tint in tints:
                self._add_one_tinted_variant(base, tint, label=biome, alias=stem)

    def _add_one_tinted_variant(self,
                                base_texture: str,
                                tint: Tuple[int, int, int],
                                *,
                                label: str,
                                alias: Optional[str] = None) -> None:
        tex = self.assets.block_texture(base_texture)
        if tex is None:
            return
        tinted = _apply_tint(tex, tint)
        sig = _signature_from_rgb(tinted)
        if sig is None or float(sig[3:6].sum()) < 4.0:
            return
        block_id = f"minecraft:{alias or base_texture}"
        self._signatures.append(_BlockSignature(
            block_id=block_id,
            signature=sig,
        ))

    # ── Public API ─────────────────────────────────────────────────

    def template_count(self) -> int:
        return len(self._signatures)

    def classify(self, patch_rgb: np.ndarray) -> Tuple[Optional[str], float]:
        sig = _signature_from_rgb(patch_rgb)
        if sig is None:
            return None, 0.0
        dists = np.abs(self._sig_matrix - sig[None, :]).sum(axis=1)
        order = np.argsort(dists)
        best   = int(order[0])
        second = int(order[1]) if len(order) > 1 else best
        d1 = float(dists[best])
        d2 = float(dists[second])

        if d1 >= self._MAX_DIST:
            return None, max(0.0, 1.0 - d1 / self._MAX_DIST)

        margin = max(0.0, d2 - d1)
        confidence = float(min(1.0, margin / 60.0))
        if confidence < self._MIN_MARGIN_CONFIDENCE:
            return None, confidence
        return self._ids[best], confidence

    def classify_batch(self,
                       patches: List[np.ndarray]) -> List[Tuple[Optional[str], float]]:
        """Convenience: classify many patches at once."""
        out = []
        for patch in patches:
            out.append(self.classify(patch))
        return out


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_block_classifier(settings: Optional[Dict[str, Any]] = None,
                           *,
                           assets=None,
                           ) -> BlockClassifierProtocol:
    """
    Build the block classifier configured by ``settings.yaml``.

    Honoured keys (all optional):

      ``vision.world.block_classifier``
          ``baseline``  (default) — colour-signature nearest neighbour.
          ``nn``                  — placeholder for a future neural net;
                                    falls back to the baseline today
                                    with a one-line warning.
    """
    cfg = ((settings or {}).get("vision", {})
                            .get("world", {})
                            .get("block_classifier", "baseline"))
    if assets is None:
        from vision.mc_assets import MCAssets
        assets = MCAssets.load()
    if cfg == "nn":
        print("[world] block_classifier='nn' requested but no NN model "
              "is bundled yet — falling back to colour-signature baseline.")
    return ColourSignatureBlockClassifier(assets)


__all__ = [
    "BlockClassifierProtocol",
    "ColourSignatureBlockClassifier",
    "build_block_classifier",
]
