# vision/nn_recognizer.py
"""
Nearest-neighbour item recognizer against captured real samples.

The synthetic :class:`vision.inventory.ItemRecognizer` matches slot
crops against templates we generated ourselves — flat item PNGs from
the game jar and isometric block renders we approximate. That works for
many items but fundamentally CAN'T be perfect for anything MC renders
through a code path our synthetic renderer doesn't simulate: shulker
boxes (3D entity model), banners (pattern composite), chests, chains,
potions, decorated pots, signs, and anything with animated textures.

This module solves that by remembering real captures. Every time
Phase 2 (the hover-OCR inspector) confirms what's in a slot via the
tooltip, we save the captured pixels into a :class:`SampleStore`
labelled with the OCR'd ``minecraft:<id>``. Subsequent runs can then
NN-match new slot pixels against the saved samples — and because those
samples come from MC's actual renderer, they describe the visual truth
better than any template we could generate.

Why simple NN, not a learned model
----------------------------------
* Slot icons are 16×16. The full feature space is just 768 bytes.
  A direct L1 (MAE) comparison over the whole library runs in <5 ms
  even with thousands of samples.
* The "training set" grows organically from hovers; no separate
  training step or hyperparameter to tune.
* Adding a per-item conv-net later is an easy upgrade — the same
  SampleStore feeds it.

Confidence rule
---------------
The recognizer reports a match only when:
  * the best sample is below ``score_max`` MAE (must be a close
    pixel-level fit — these are EXACT real captures so the bar is
    higher than for synthetic templates), AND
  * the runner-up among DIFFERENT items is at least
    ``margin_min`` MAE worse (rules out look-alikes).

A multi-sample match where the top-K all agree on the same item is
treated as additional confirmation — confidence rises with agreement.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from vision.inventory import SlotContent
from vision.sample_store import SAMPLE_SIZE, SampleStore, StoredSample


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class NNRecognizerConfig:
    # Sample-vs-capture L1 distance must be below this to count as a
    # match candidate. Real-vs-real matches are typically below 4 MAE
    # (identical MC renders); a captured slot crop with the same item
    # often differs from a stored sample by 0–3 MAE.
    score_max: float = 8.0

    # Required MAE gap between the best item and the best DIFFERENT
    # item. 4 means the closest-matching sample of a different item
    # has to be at least 4 MAE worse than the winner.
    margin_min: float = 3.0

    # Top-K neighbours we look at. If K samples all vote for the same
    # item id, that's much stronger evidence than a single nearest
    # match.
    top_k: int = 5


# ---------------------------------------------------------------------------
# Recognizer
# ---------------------------------------------------------------------------

class SampleRecognizer:
    """
    Direct-pixel nearest-neighbour matcher over a :class:`SampleStore`.

    Build once per process (it loads every sample into a single packed
    tensor for fast vectorised distance). When new samples land on disk
    you can call :meth:`reload` to refresh.

    The result is a :class:`SlotContent` with ``source`` reporting that
    it came from a sample match — distinguishing it from synthetic
    template matches in downstream logging / training data collection.
    """

    def __init__(self,
                 store: SampleStore,
                 *,
                 config: Optional[NNRecognizerConfig] = None):
        self._store = store
        self.cfg = config or NNRecognizerConfig()
        self._samples: List[StoredSample] = []
        self._tensor: Optional[np.ndarray] = None
        self.reload()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Re-scan the disk store and rebuild the in-memory tensor."""
        self._samples = self._store.load_all()
        if not self._samples:
            self._tensor = None
            return
        self._tensor = np.stack([s.rgb for s in self._samples], axis=0)

    def sample_count(self) -> int:
        return len(self._samples)

    def item_count(self) -> int:
        return len({s.item_id for s in self._samples})

    def recognize(self, crop_rgb: np.ndarray) -> Optional[SlotContent]:
        """
        Return a :class:`SlotContent` if NN matched confidently, else
        ``None``. ``None`` means "let the caller fall back to the
        synthetic template recogniser".
        """
        if self._tensor is None or crop_rgb is None or crop_rgb.size == 0:
            return None
        if crop_rgb.ndim == 3 and crop_rgb.shape[2] == 4:
            crop_rgb = crop_rgb[..., :3]
        if crop_rgb.shape[:2] != (SAMPLE_SIZE, SAMPLE_SIZE):
            import cv2
            crop_rgb = cv2.resize(crop_rgb, (SAMPLE_SIZE, SAMPLE_SIZE),
                                  interpolation=cv2.INTER_AREA)
        crop_rgb = crop_rgb.astype(np.int32)

        # Per-sample average MAE over all 16×16×3 pixels.
        diff = np.abs(self._tensor.astype(np.int32) - crop_rgb[None, ...])
        per_sample_score = diff.reshape(len(self._samples), -1
                                       ).mean(axis=1)        # (N,)

        order = np.argsort(per_sample_score)
        best_idx = int(order[0])
        best_score = float(per_sample_score[best_idx])
        if best_score > self.cfg.score_max:
            return None
        best_item = self._samples[best_idx].item_id

        # Top-K agreement: count what fraction of the K nearest
        # neighbours share the winning item id.
        k = min(self.cfg.top_k, len(self._samples))
        top_items = [self._samples[int(i)].item_id for i in order[:k]]
        votes = Counter(top_items)
        winner_votes = votes[best_item]
        agreement = winner_votes / k

        # Best score among samples whose item id differs from the
        # winner — used to gate by margin.
        runner_up_score = float("inf")
        runner_up_item: Optional[str] = None
        for i in order[1:]:
            ii = int(i)
            sid = self._samples[ii].item_id
            if sid != best_item:
                runner_up_score = float(per_sample_score[ii])
                runner_up_item  = sid
                break

        margin = runner_up_score - best_score
        if margin < self.cfg.margin_min:
            return None

        # Confidence: scaled margin × agreement. A solid match (margin
        # ≥10 MAE, all K neighbours agree) → 1.0; a borderline one
        # (margin 3 MAE, 1 of 5 agree) → ~0.2.
        confidence = float(min(1.0, (margin / 10.0)) * agreement)
        # Boost confidence to a useful range so downstream gating
        # treats sample matches at least as confidently as template
        # matches at the same margin.
        confidence = max(confidence, 0.50)

        return SlotContent(
            item       = best_item,
            count      = 0,                          # filled by caller
            confidence = confidence,
            score      = best_score,
            second     = runner_up_item,
        )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def build_sample_recognizer(store: Optional[SampleStore] = None
                            ) -> SampleRecognizer:
    """Build a SampleRecognizer against the default sample store."""
    if store is None:
        from vision.sample_store import build_sample_store
        store = build_sample_store()
    return SampleRecognizer(store)


__all__ = [
    "SampleRecognizer", "NNRecognizerConfig", "build_sample_recognizer",
]
