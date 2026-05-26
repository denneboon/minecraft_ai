# vision/world/sample_store.py
"""
Disk-backed store of labelled world-block patches.

Companion to :mod:`vision.sample_store` (which holds inventory-slot
samples). This one holds *world-view* samples: small RGB crops of how
a given block actually looks when rendered inside the gameplay window,
labelled by the block id MC told us was there.

Where the labels come from
--------------------------
Whenever the F3 overlay shows a "Targeted Block" / "Looking at block"
line (player is holding F3, or has the setting on "always"), the
:class:`vision.world.perception.WorldPerception` orchestrator knows
*exactly* what block is at the crosshair. It then crops a small patch
centered on the crosshair and hands the pair ``(patch, block_id)`` to
this store. Over time the store builds up real captures of every
block the player has ever crosshaired, across biomes, lighting
conditions, and weather.

Why this matters
----------------
The colour-signature classifier (``ColourSignatureBlockClassifier``)
matches a 28-dim signature to the *vanilla atlas texture*. That works
on a brightly-lit dry block in a flat-lighting daytime shot. It fails
on:

* night-time darkness (everything maps to "unknown"),
* biome-tinted surfaces it has no signature variant for,
* weather effects (rain, snow, fog),
* shaders / smooth-lighting palette shifts.

A sample-based recogniser sidesteps every one of those because it
matches against *the rendered pixels of this exact game install*, not
against the canonical atlas.

Layout
------
::

    data/training/world_samples/
        <block_short_id>/                  # "stone", "grass_block", …
            <pixel_hash>.png               # de-duplicated patch
            ...
        _manifest.json                     # {<id>: <sample_count>, …}

We store each patch resized to a fixed ``SAMPLE_SIZE × SAMPLE_SIZE``
RGB tile. Identical captures hash to the same filename and de-dup,
so a 30-tick stare at one block only contributes one sample.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Sample primitives
# ---------------------------------------------------------------------------

# Canonical world-patch size. Bigger than inventory's 16 (no in-game
# 16-px constraint here) and matches the default ``patch_size_px`` of
# the perception orchestrator, so a captured crosshair patch needs no
# resize before storage in the common case.
SAMPLE_SIZE = 24


@dataclass
class StoredWorldSample:
    """One on-disk sample: a block id, the pixels, and the file path."""
    block_id: str                       # ``"minecraft:stone"``
    path: Path
    rgb: np.ndarray                     # SAMPLE_SIZE x SAMPLE_SIZE x 3 uint8


def _short_id(block_id: str) -> str:
    """Strip the ``minecraft:`` namespace for use as a directory name."""
    return block_id.split(":", 1)[-1] if ":" in block_id else block_id


def _hash_pixels(rgb: np.ndarray) -> str:
    """SHA-1 of the raw pixel bytes — content-addressed filenames."""
    return hashlib.sha1(rgb.tobytes()).hexdigest()[:16]


def _normalise_patch(patch_rgb: np.ndarray) -> Optional[np.ndarray]:
    """Coerce a world patch into ``SAMPLE_SIZE × SAMPLE_SIZE × 3 uint8``."""
    if patch_rgb is None or patch_rgb.size == 0:
        return None
    if patch_rgb.ndim == 3 and patch_rgb.shape[2] == 4:
        patch_rgb = patch_rgb[..., :3]
    if patch_rgb.ndim != 3 or patch_rgb.shape[2] != 3:
        return None
    if patch_rgb.shape[:2] != (SAMPLE_SIZE, SAMPLE_SIZE):
        patch_rgb = cv2.resize(patch_rgb, (SAMPLE_SIZE, SAMPLE_SIZE),
                               interpolation=cv2.INTER_AREA)
    return patch_rgb.astype(np.uint8)


# ---------------------------------------------------------------------------
# Sample store
# ---------------------------------------------------------------------------

@dataclass
class WorldSampleStoreConfig:
    # Hard cap on per-block samples. Once a block has this many stored
    # samples, new captures are dropped. The de-dup hash already
    # collapses identical pixels; this cap bounds the diversity any
    # single block can contribute to the NN index so the matcher stays
    # cheap.
    max_samples_per_block: int = 80


class WorldSampleStore:
    """
    Add, query, and load labelled world-block patches on disk.

    Multi-tick safe — the only mutation site (:meth:`save`) is guarded
    by an in-process lock so background workers can't race.
    """

    def __init__(self,
                 root: Path,
                 *,
                 config: Optional[WorldSampleStoreConfig] = None):
        self.root = Path(root)
        self.cfg = config or WorldSampleStoreConfig()
        self._lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Mutators ──────────────────────────────────────────────────

    def save(self,
             block_id: str,
             patch_rgb: np.ndarray,
             *,
             metadata: Optional[Dict[str, object]] = None,
             ) -> Optional[Path]:
        """
        Record one sample for ``block_id``. Returns the on-disk path if
        the sample was written, ``None`` if it was a duplicate or the
        per-block cap was already reached.

        ``metadata`` (optional) is a JSON-serialisable dict written to
        a sibling ``.json`` file next to the PNG. Used to record the
        context in which the sample was captured — pose, distance from
        eye to block, time of day, the classifier guess that was
        rejected, etc. Future ML training can condition on these.
        """
        norm = _normalise_patch(patch_rgb)
        if norm is None:
            return None
        short = _short_id(block_id)
        block_dir = self.root / short
        with self._lock:
            block_dir.mkdir(parents=True, exist_ok=True)
            # Just count — we don't need a sorted list, only the size.
            n = sum(1 for _ in block_dir.glob("*.png"))
            if n >= self.cfg.max_samples_per_block:
                return None
            h = _hash_pixels(norm)
            path = block_dir / f"{h}.png"
            if path.is_file():
                return None
            bgr = cv2.cvtColor(norm, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), bgr)
            if metadata:
                meta_path = path.with_suffix(".json")
                try:
                    meta_path.write_text(
                        json.dumps(metadata, sort_keys=True, default=str),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
            self._write_manifest_unlocked()
            return path

    # ── Readers ───────────────────────────────────────────────────

    def load_all(self) -> List[StoredWorldSample]:
        """Load every sample on disk into memory."""
        out: List[StoredWorldSample] = []
        if not self.root.is_dir():
            return out
        for block_dir in sorted(self.root.iterdir()):
            if not block_dir.is_dir():
                continue
            block_id = f"minecraft:{block_dir.name}"
            for p in sorted(block_dir.glob("*.png")):
                bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if rgb.shape[:2] != (SAMPLE_SIZE, SAMPLE_SIZE):
                    rgb = cv2.resize(rgb, (SAMPLE_SIZE, SAMPLE_SIZE),
                                     interpolation=cv2.INTER_AREA)
                out.append(StoredWorldSample(block_id=block_id, path=p,
                                             rgb=rgb.astype(np.uint8)))
        return out

    def manifest(self) -> Dict[str, int]:
        """Return ``{block_id: sample_count}`` based on what's on disk."""
        out: Dict[str, int] = {}
        if not self.root.is_dir():
            return out
        for block_dir in sorted(self.root.iterdir()):
            if not block_dir.is_dir():
                continue
            block_id = f"minecraft:{block_dir.name}"
            out[block_id] = sum(1 for _ in block_dir.glob("*.png"))
        return out

    def total_samples(self) -> int:
        return sum(self.manifest().values())

    def block_count(self) -> int:
        return len(self.manifest())

    # ── Internal ──────────────────────────────────────────────────

    def _write_manifest_unlocked(self) -> None:
        """Refresh ``_manifest.json``. Caller must hold ``self._lock``."""
        manifest_path = self.root / "_manifest.json"
        m: Dict[str, int] = {}
        for block_dir in self.root.iterdir():
            if not block_dir.is_dir():
                continue
            n = sum(1 for _ in block_dir.glob("*.png"))
            if n:
                m[f"minecraft:{block_dir.name}"] = n
        manifest_path.write_text(json.dumps(m, indent=2, sort_keys=True),
                                 encoding="utf-8")


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def default_world_sample_root() -> Path:
    return (Path(__file__).resolve().parents[2]
            / "data" / "training" / "world_samples")


def build_world_sample_store(root: Optional[Path] = None
                             ) -> WorldSampleStore:
    if root is None:
        root = default_world_sample_root()
    return WorldSampleStore(root)


__all__ = [
    "WorldSampleStore", "WorldSampleStoreConfig", "StoredWorldSample",
    "SAMPLE_SIZE",
    "build_world_sample_store", "default_world_sample_root",
]
