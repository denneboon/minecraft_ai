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
    # Hard cap on per-block samples (sliding window — see eviction below).
    # This is the single biggest lever on recogniser quality: it bounds how
    # much real-condition DIVERSITY a block can retain (positions, distances,
    # angles, lighting). 80 proved too low — a block saturates with near-
    # duplicate views from one spot and never generalises (oak_log was stuck
    # at ~0.5). 220 (validated by train_overnight) holds a full walk/day's
    # variety; crucially it's the DEFAULT so EVERY run (agent play, live
    # self-teach) retains diversity instead of evicting it back down to 80.
    max_samples_per_block: int = 220

    # When the cap is hit, evict the OLDEST sample to make room for the
    # new one (sliding window) instead of refusing the new capture. This
    # lets the store keep refreshing itself as the player/agent sees
    # blocks under newer / better conditions — and lets a capture-scheme
    # change (e.g. switching to distance-normalised crops) gradually
    # replace stale samples instead of being frozen out at the cap.
    evict_oldest_when_full: bool = True


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
        # In-memory per-block PNG counts (short_id -> n). Lazily built
        # from disk on first save so the per-block cap check and the
        # manifest don't re-glob the whole tree on every single sample —
        # that cost is O(blocks × files) and grows with the dataset this
        # feature is meant to accumulate. The manifest file is rewritten
        # on a throttle (it's only an external-inspection artifact;
        # ``manifest()`` reads the live cache).
        self._counts: Optional[Dict[str, int]] = None
        self._saves_since_manifest = 0
        self._manifest_every = 10

    # ── Mutators ──────────────────────────────────────────────────

    def _evict_oldest_locked(self, block_dir: Path, short: str,
                             counts: Dict[str, int], *, keep: int) -> bool:
        """Delete oldest PNGs (and their .json sidecars) for ``short`` until
        at most ``keep`` remain. Caller holds ``self._lock``. Returns True
        if there's now room (count <= keep). Best-effort: a failed unlink
        just leaves that file and tries the next."""
        try:
            pngs = sorted(block_dir.glob("*.png"),
                          key=lambda p: p.stat().st_mtime)
        except Exception:
            return False
        removed = 0
        for p in pngs:
            if len(pngs) - removed <= keep:
                break
            try:
                p.with_suffix(".json").unlink(missing_ok=True)
                p.unlink()
                removed += 1
            except Exception:
                continue
        if removed:
            counts[short] = max(0, counts.get(short, removed) - removed)
        return counts.get(short, 0) <= keep

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
            counts = self._ensure_counts_locked()
            n = counts.get(short, 0)
            if n >= self.cfg.max_samples_per_block:
                if not self.cfg.evict_oldest_when_full:
                    return None
                # Sliding window: drop the oldest sample(s) to make room.
                if not self._evict_oldest_locked(block_dir, short, counts,
                                                 keep=self.cfg.max_samples_per_block - 1):
                    return None
            h = _hash_pixels(norm)
            path = block_dir / f"{h}.png"
            if path.is_file():
                return None
            bgr = cv2.cvtColor(norm, cv2.COLOR_RGB2BGR)
            ok = cv2.imwrite(str(path), bgr)
            if not ok:
                # cv2.imwrite returns False on encoder error, disk full,
                # invalid extension, etc. Without surfacing this we'd
                # silently lose every sample of the affected block id.
                if not getattr(self, "_imwrite_warn_emitted", False):
                    self._imwrite_warn_emitted = True
                    print(f"[sample_store][WARN] cv2.imwrite returned False "
                          f"for {path} — sample lost. (further occurrences "
                          f"will stay silent)")
                return None
            if metadata:
                meta_path = path.with_suffix(".json")
                try:
                    meta_path.write_text(
                        json.dumps(metadata, sort_keys=True, default=str),
                        encoding="utf-8",
                    )
                except Exception as e:
                    # Metadata is best-effort — the PNG label is the
                    # primary signal — but a permission / encoding
                    # error here means EVERY future sidecar will also
                    # fail. Surface the first occurrence so we can
                    # diagnose; stay silent after that.
                    if not getattr(self, "_meta_warn_emitted", False):
                        self._meta_warn_emitted = True
                        print(f"[sample_store][WARN] metadata sidecar "
                              f"write failed for {meta_path.name}: {e!r}")
            counts[short] = n + 1
            self._saves_since_manifest += 1
            if self._saves_since_manifest >= self._manifest_every:
                self._saves_since_manifest = 0
                self._write_manifest_unlocked()
            return path

    # ── Readers ───────────────────────────────────────────────────

    def load_all(self, skip_paths: Optional[set] = None
                 ) -> List[StoredWorldSample]:
        """Load samples on disk into memory. Corrupt PNGs are
        skipped, but a non-zero corruption count is surfaced once per
        load — a partial dataset silently shrinking past a power-loss
        event would otherwise stay invisible until the gate's per-block
        floor stopped firing.

        ``skip_paths`` (optional) is a set of ``Path`` already held in
        memory by the caller; those files are not re-read. This lets the
        sample recogniser reload INCREMENTALLY (read only newly-saved
        patches) instead of re-decoding the entire — and ever-growing —
        dataset from disk on every reload.
        """
        out: List[StoredWorldSample] = []
        if not self.root.is_dir():
            return out
        n_corrupt = 0
        for block_dir in sorted(self.root.iterdir()):
            if not block_dir.is_dir():
                continue
            block_id = f"minecraft:{block_dir.name}"
            for p in sorted(block_dir.glob("*.png")):
                if skip_paths is not None and p in skip_paths:
                    continue
                bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if bgr is None:
                    n_corrupt += 1
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if rgb.shape[:2] != (SAMPLE_SIZE, SAMPLE_SIZE):
                    rgb = cv2.resize(rgb, (SAMPLE_SIZE, SAMPLE_SIZE),
                                     interpolation=cv2.INTER_AREA)
                out.append(StoredWorldSample(block_id=block_id, path=p,
                                             rgb=rgb.astype(np.uint8)))
        if n_corrupt:
            print(f"[sample_store][WARN] skipped {n_corrupt} corrupt PNG(s) "
                  f"under {self.root}. Loaded {len(out)} valid samples.")
        return out

    def manifest(self) -> Dict[str, int]:
        """Return ``{block_id: sample_count}``. Uses the in-memory count
        cache once it's been built (single-process authoritative); falls
        back to a disk scan before the first save."""
        if self._counts is not None:
            return {f"minecraft:{k}": v for k, v in self._counts.items() if v}
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

    def _ensure_counts_locked(self) -> Dict[str, int]:
        """Lazily build the per-block PNG-count cache from disk (once).
        Caller must hold ``self._lock``."""
        if self._counts is None:
            counts: Dict[str, int] = {}
            if self.root.is_dir():
                for block_dir in self.root.iterdir():
                    if block_dir.is_dir():
                        counts[block_dir.name] = sum(
                            1 for _ in block_dir.glob("*.png"))
            self._counts = counts
        return self._counts

    def _write_manifest_unlocked(self) -> None:
        """Refresh ``_manifest.json`` from the in-memory count cache (no
        disk re-glob). Caller must hold ``self._lock``."""
        manifest_path = self.root / "_manifest.json"
        m = {f"minecraft:{k}": v for k, v in (self._counts or {}).items() if v}
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
