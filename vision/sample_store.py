# vision/sample_store.py
"""
Disk-backed store of (slot crop, item id) training samples.

Every successful hover-OCR resolution in Phase 2 of the inventory
pipeline produces a labelled sample: "the pixels I just cropped from
this slot really *are* an instance of minecraft:<id> as drawn by MC's
inventory renderer". This module collects those samples on disk so that
later runs can:

  * skip the hover entirely for items we've seen before (matching the
    new slot pixels against the saved samples in
    :class:`vision.nn_recognizer.SampleRecognizer`),
  * eventually train a learned classifier in Phase 3.

Why disk + per-item directories
-------------------------------
* Captures from MC's actual renderer beat any synthetic isometric
  template we can generate (shulker boxes, banners, glass panes, chests
  and chains have no usable cube-projected template; samples ARE the
  template).
* Per-item directories make the dataset easy to browse, audit, and
  delete bad samples by hand.
* Disk persistence means the same dataset accumulates across
  invocations and across project sessions without re-extraction.

Layout
------
::

    data/training/inventory_samples/
        <item_short_id>/                   # e.g. "furnace"
            <pixel_hash>.png               # de-duplicated 16×16 crop
            ...
        _manifest.json                     # {<id>: <sample_count>, …}

We store the raw 16×16 RGBA crop (resized from the slot's screen-px
crop with INTER_AREA), one PNG per pixel-content-hash. Identical
captures don't re-enter the set, so a Phase-2 sweep that hovers the
same slot ten times only adds one sample.
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

# Canonical sample size. Resizing every slot crop to this size before
# storing means the NN matcher always compares like-sized arrays even
# if MC's GUI scale changes between runs.
SAMPLE_SIZE = 16


@dataclass
class StoredSample:
    """One on-disk sample: an item id, the pixels, and provenance."""
    item_id: str                       # "minecraft:furnace"
    path: Path
    rgb: np.ndarray                    # SAMPLE_SIZE × SAMPLE_SIZE × 3 uint8


def _short_id(item_id: str) -> str:
    """Strip the ``minecraft:`` namespace for use as a directory name."""
    return item_id.split(":", 1)[-1] if ":" in item_id else item_id


def _hash_pixels(rgb: np.ndarray) -> str:
    """SHA-1 of the raw pixel bytes. Used for content-addressed filenames."""
    return hashlib.sha1(rgb.tobytes()).hexdigest()[:16]


def _normalise_crop(crop_rgb: np.ndarray) -> Optional[np.ndarray]:
    """Coerce a slot crop into ``SAMPLE_SIZE × SAMPLE_SIZE × 3 uint8``."""
    if crop_rgb is None or crop_rgb.size == 0:
        return None
    if crop_rgb.ndim == 3 and crop_rgb.shape[2] == 4:
        crop_rgb = crop_rgb[..., :3]
    if crop_rgb.ndim != 3 or crop_rgb.shape[2] != 3:
        return None
    if crop_rgb.shape[:2] != (SAMPLE_SIZE, SAMPLE_SIZE):
        crop_rgb = cv2.resize(crop_rgb, (SAMPLE_SIZE, SAMPLE_SIZE),
                              interpolation=cv2.INTER_AREA)
    return crop_rgb.astype(np.uint8)


# ---------------------------------------------------------------------------
# Sample store
# ---------------------------------------------------------------------------

@dataclass
class SampleStoreConfig:
    # Hard cap on per-item samples. Once an item has this many stored
    # samples, new captures are dropped (de-dup already handles
    # identical pixels; this caps the diversity any single item can
    # contribute to the NN matcher's index, keeping it lightweight).
    max_samples_per_item: int = 50


class SampleStore:
    """
    Add, query, and load labelled slot crops on disk.

    Construct once per process; pass the same instance to the inspector
    (for writing during hover-OCR) and the NN recogniser (for reading).
    Multi-thread safe — the only mutation site is :meth:`save`, guarded
    by a lock so two hover threads can't race on the same item dir.
    """

    def __init__(self,
                 root: Path,
                 *,
                 config: Optional[SampleStoreConfig] = None):
        self.root = Path(root)
        self.cfg = config or SampleStoreConfig()
        self._lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def save(self, item_id: str, crop_rgb: np.ndarray) -> Optional[Path]:
        """
        Record one sample for ``item_id``. Returns the on-disk path if
        the sample was written, ``None`` if it was a duplicate or the
        per-item cap was already reached.
        """
        norm = _normalise_crop(crop_rgb)
        if norm is None:
            return None
        short = _short_id(item_id)
        item_dir = self.root / short
        with self._lock:
            item_dir.mkdir(parents=True, exist_ok=True)
            existing = list(item_dir.glob("*.png"))
            if len(existing) >= self.cfg.max_samples_per_item:
                return None
            h = _hash_pixels(norm)
            path = item_dir / f"{h}.png"
            if path.is_file():
                return None
            bgr = cv2.cvtColor(norm, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), bgr)
            self._write_manifest_unlocked()
            return path

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------

    def load_all(self) -> List[StoredSample]:
        """
        Load every sample on disk into memory. Cheap when the set is
        small (<10k items × 50 samples × 768 bytes ≈ 380 MB worst case;
        practically a few MB at any reasonable point in a project).
        """
        out: List[StoredSample] = []
        if not self.root.is_dir():
            return out
        for item_dir in sorted(self.root.iterdir()):
            if not item_dir.is_dir():
                continue
            item_id = f"minecraft:{item_dir.name}"
            for p in sorted(item_dir.glob("*.png")):
                bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if rgb.shape[:2] != (SAMPLE_SIZE, SAMPLE_SIZE):
                    rgb = cv2.resize(rgb, (SAMPLE_SIZE, SAMPLE_SIZE),
                                     interpolation=cv2.INTER_AREA)
                out.append(StoredSample(item_id=item_id, path=p,
                                        rgb=rgb.astype(np.uint8)))
        return out

    def manifest(self) -> Dict[str, int]:
        """Return ``{item_id: sample_count}`` based on what's on disk."""
        out: Dict[str, int] = {}
        if not self.root.is_dir():
            return out
        for item_dir in sorted(self.root.iterdir()):
            if not item_dir.is_dir():
                continue
            item_id = f"minecraft:{item_dir.name}"
            out[item_id] = sum(1 for _ in item_dir.glob("*.png"))
        return out

    def total_samples(self) -> int:
        return sum(self.manifest().values())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_manifest_unlocked(self) -> None:
        """Refresh ``_manifest.json``. Caller must hold ``self._lock``."""
        manifest_path = self.root / "_manifest.json"
        m = {}
        for item_dir in self.root.iterdir():
            if not item_dir.is_dir():
                continue
            n = sum(1 for _ in item_dir.glob("*.png"))
            if n:
                m[f"minecraft:{item_dir.name}"] = n
        manifest_path.write_text(json.dumps(m, indent=2, sort_keys=True),
                                 encoding="utf-8")


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def default_sample_root() -> Path:
    return (Path(__file__).resolve().parent.parent
            / "data" / "training" / "inventory_samples")


def build_sample_store(root: Optional[Path] = None) -> SampleStore:
    if root is None:
        root = default_sample_root()
    return SampleStore(root)


__all__ = [
    "SampleStore", "SampleStoreConfig", "StoredSample",
    "SAMPLE_SIZE", "build_sample_store", "default_sample_root",
]
