# vision/world/sample_recognizer.py
"""
Nearest-neighbour block recogniser over real world captures.

Mirrors :mod:`vision.nn_recognizer` (which does the same thing for
inventory slots), but on world patches rather than 16-px slot icons.

How it works
------------
1. At construction we load every labelled patch out of a
   :class:`vision.world.sample_store.WorldSampleStore` and stack their
   pixels into one packed ``(N, S, S, 3)`` int32 tensor.
2. ``classify(patch_rgb)`` measures L1 (mean-absolute-error) against
   every stored sample, picks the closest, and reports it ONLY when
   the match is tight and well-separated from the nearest *different*
   block id.
3. ``classify`` also implements the
   :class:`vision.world.block_classifier.BlockClassifierProtocol` —
   so it can drop straight into ``WorldPerception`` in place of the
   colour-signature baseline (or beside it, via a hybrid).

Why MAE not cosine or feature-net
---------------------------------
The samples come from the actual game render at the player's exact
graphics settings. Two patches of the same block in similar lighting
differ by ~5–10 MAE at most; two clearly-different blocks usually sit
≥25 MAE apart. The signal/noise gap is wide enough that plain pixel
distance works — and a 100-sample library queries in <2 ms on CPU,
fast enough for the per-tick perception sweep.

When a CNN classifier eventually arrives, it can train on this same
sample store with no API changes — the recogniser just gets swapped at
the :class:`BlockClassifierProtocol` seam.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from vision.world.sample_store import (
    SAMPLE_SIZE,
    StoredWorldSample,
    WorldSampleStore,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SampleBlockRecognizerConfig:
    # Best-sample MAE must be below this to count as a candidate. The
    # threshold is more lenient than the inventory matcher's (8) because
    # world patches vary far more across lighting/biome conditions.
    score_max: float = 22.0

    # Required MAE gap between the best item and the best DIFFERENT
    # item. Smaller than inventory (3) because world surfaces with
    # similar palettes (stone vs cobblestone) sit closer in pixel space.
    margin_min: float = 2.5

    # K nearest neighbours we tally for the agreement vote.
    top_k: int = 5

    # Confidence floor — return None below this to keep the WorldMap
    # from absorbing low-quality guesses.
    min_confidence: float = 0.10


# ---------------------------------------------------------------------------
# Recogniser
# ---------------------------------------------------------------------------

class SampleBlockRecognizer:
    """
    Pixel-NN recogniser over a :class:`WorldSampleStore`.

    Build once per process. When new samples land on disk (because the
    perception layer just learned a new label), call :meth:`reload`.
    """

    def __init__(self,
                 store: WorldSampleStore,
                 *,
                 config: Optional[SampleBlockRecognizerConfig] = None):
        self._store = store
        self.cfg = config or SampleBlockRecognizerConfig()
        self._samples: List[StoredWorldSample] = []
        self._tensor:  Optional[np.ndarray] = None
        self._ids:     List[str] = []
        self.reload()

    # ── Public API ────────────────────────────────────────────────

    def reload(self) -> None:
        """Re-scan the disk store from scratch and rebuild the tensor."""
        self._samples = self._store.load_all()
        self._rebuild_tensor()

    def reload_incremental(self) -> None:
        """Append only samples that appeared on disk since the last load.

        The full ``reload`` re-decodes every PNG — fine at startup, but
        the perception layer calls it every few auto-saved samples, and
        re-reading the entire (growing) dataset each time was the single
        biggest perception cost in profiling. Here we read ONLY the new
        files (``skip_paths`` = what we already hold) and append, so a
        reload costs O(new samples) instead of O(all samples).
        """
        known = {s.path for s in self._samples}
        new = self._store.load_all(skip_paths=known)
        if not new:
            return
        self._samples.extend(new)
        self._rebuild_tensor()

    def _rebuild_tensor(self) -> None:
        if not self._samples:
            self._tensor = None
            self._ids = []
            return
        self._tensor = np.stack([s.rgb for s in self._samples], axis=0
                                ).astype(np.int32)
        self._ids = [s.block_id for s in self._samples]

    def sample_count(self) -> int:
        return len(self._samples)

    def block_count(self) -> int:
        return len(set(self._ids))

    def count_for(self, block_id: str) -> int:
        """Public accessor for the per-block sample count. Used by
        the perception layer's commit-gate to require ``>= N`` samples
        of the specific predicted block id before trusting it.

        Reads from the immutable in-memory list created during the
        last ``reload()``. Safe even if a concurrent ``reload()`` is
        in flight — the worst case is reading the previous snapshot,
        which is identical to the perception layer running one tick
        earlier."""
        return sum(1 for sid in self._ids if sid == block_id)

    # The two methods that match BlockClassifierProtocol.

    def template_count(self) -> int:
        # "Templates" for protocol parity = how many block signatures we
        # could discriminate. With NN that's distinct block ids loaded.
        return self.block_count()

    def classify(self,
                 patch_rgb: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Return ``(block_id, confidence)`` or ``(None, low_conf)`` if no
        confident match.

        ``confidence`` is in [0, 1]. ``None`` lets the perception
        orchestrator fall back to the colour-signature baseline.
        """
        if self._tensor is None or patch_rgb is None or patch_rgb.size == 0:
            return None, 0.0
        # When the store contains only ONE block id, there is no
        # negative example to gauge margin against — every query
        # returns the only known id with maximal confidence. Refuse
        # to classify under that condition; otherwise we'd flood the
        # WorldMap with cubes labelled as whatever happened to be
        # the first F3-confirmed block.
        if len(set(self._ids)) < 2:
            return None, 0.0

        rgb = patch_rgb
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[..., :3]
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return None, 0.0
        if rgb.shape[:2] != (SAMPLE_SIZE, SAMPLE_SIZE):
            rgb = cv2.resize(rgb, (SAMPLE_SIZE, SAMPLE_SIZE),
                             interpolation=cv2.INTER_AREA)
        q = rgb.astype(np.int32)

        # Per-sample MAE.
        diff = np.abs(self._tensor - q[None, ...])
        per_sample = diff.reshape(len(self._samples), -1).mean(axis=1)
        order = np.argsort(per_sample)
        best_idx   = int(order[0])
        best_score = float(per_sample[best_idx])
        best_id    = self._ids[best_idx]

        if best_score > self.cfg.score_max:
            return None, 0.0

        # Top-K vote agreement.
        k = min(self.cfg.top_k, len(self._samples))
        top_ids = [self._ids[int(i)] for i in order[:k]]
        votes   = Counter(top_ids)
        agreement = votes[best_id] / k

        # Best score among samples whose block id differs.
        runner_up_score = float("inf")
        for i in order[1:]:
            sid = self._ids[int(i)]
            if sid != best_id:
                runner_up_score = float(per_sample[int(i)])
                break
        margin = runner_up_score - best_score
        if margin < self.cfg.margin_min:
            return None, 0.0

        # Confidence: combines tightness (margin) with agreement, with
        # a soft floor so a near-perfect match never undersells itself.
        norm_margin = min(1.0, margin / 12.0)
        confidence  = max(0.0, min(1.0, 0.55 * agreement + 0.45 * norm_margin))
        if confidence < self.cfg.min_confidence:
            return None, confidence
        return best_id, confidence


# ---------------------------------------------------------------------------
# Hybrid: sample recogniser first, colour-signature baseline fallback
# ---------------------------------------------------------------------------

class HybridBlockClassifier:
    """
    Try the sample-NN recogniser first; if it can't confidently name
    the block, fall back to the colour-signature baseline.

    This is the recommended runtime classifier:
      * In a fresh project the sample store is empty → every classify
        call falls back to the baseline (current behaviour).
      * As the user holds F3 and the perception layer auto-collects
        samples, the recogniser learns this player's *actual* world →
        accuracy improves silently over time.

    The hybrid implements ``BlockClassifierProtocol`` so
    :class:`vision.world.perception.WorldPerception` can use it
    unchanged.
    """

    def __init__(self,
                 sample_recognizer: SampleBlockRecognizer,
                 baseline_classifier):
        self.sample = sample_recognizer
        self.baseline = baseline_classifier

    def template_count(self) -> int:
        return self.sample.block_count() + self.baseline.template_count()

    def classify(self,
                 patch_rgb: np.ndarray) -> Tuple[Optional[str], float]:
        block_id, conf = self.sample.classify(patch_rgb)
        if block_id is not None and conf >= self.sample.cfg.min_confidence:
            return block_id, conf
        # Sample store didn't have a confident match. Try the baseline.
        return self.baseline.classify(patch_rgb)

    def reload_samples(self) -> None:
        """Convenience pass-through for callers that just wrote a new
        sample and want it reflected immediately. Incremental: only the
        newly-saved patches are read from disk, not the whole dataset."""
        self.sample.reload_incremental()


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_sample_block_recognizer(
        store: Optional[WorldSampleStore] = None,
        *,
        config: Optional[SampleBlockRecognizerConfig] = None,
        ) -> SampleBlockRecognizer:
    if store is None:
        from vision.world.sample_store import build_world_sample_store
        store = build_world_sample_store()
    return SampleBlockRecognizer(store, config=config)


__all__ = [
    "SampleBlockRecognizer",
    "SampleBlockRecognizerConfig",
    "HybridBlockClassifier",
    "build_sample_block_recognizer",
]
