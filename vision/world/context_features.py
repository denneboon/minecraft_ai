"""
Futureproof context-feature system for the block recogniser's fusion model.

THE design goal (per the user): make it trivial to add ANY new input that
could affect recognition — without touching the model architecture or
silently breaking old checkpoints. Each input is a named ``ContextFeature``
with a fixed encoded width that turns ONE sample's metadata dict into a
fixed-length vector, PLUS a presence flag so MISSING context (old samples
with no rich sidecar, F3 closed -> no biome, an unconfirmed neighbour)
degrades gracefully to "unknown" instead of a wrong value.

Adding an input = append one ``ContextFeature`` to ``DEFAULT_FEATURES``. The
fusion model derives its input width from the active set and records the
set's SIGNATURE (ordered name:dim list) in its checkpoint, so a model always
knows exactly which inputs it was trained with and can detect a mismatch on
load — no surprise silent corruption when the feature set evolves.

Pure numpy (no torch) so it's usable for analysis + both training/inference.

Encodings:
  * one-hot      — bounded categoricals (face, weather, dimension)
  * feature-hash — UNBOUNDED categoricals (biome, block ids, held item): any
                   new value just hashes into the fixed space, so a new biome
                   or a never-seen block needs zero code changes.
  * scalar       — numerics, normalised to ~[0,1] (brightness, light, distance,
                   y, fov)
  * bool         — 0/1
Each feature contributes ``dim`` encoded values + 1 presence flag.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Encoding primitives
# ---------------------------------------------------------------------------

def _hash_encode(value: Optional[str], dim: int) -> np.ndarray:
    """Signed feature-hashing of a string into ``dim`` slots. Deterministic,
    vocabulary-free (any new string hashes in), low collision at modest dim.
    None / empty -> all zeros (the presence flag carries 'missing')."""
    out = np.zeros(dim, dtype=np.float32)
    if not value:
        return out
    h = hashlib.md5(value.encode("utf-8")).digest()
    idx = int.from_bytes(h[:4], "little") % dim
    sign = 1.0 if (h[4] & 1) else -1.0
    out[idx] = sign
    return out


def _onehot(value: Optional[str], vocab: Tuple[str, ...]) -> np.ndarray:
    out = np.zeros(len(vocab), dtype=np.float32)
    if value is not None and value in vocab:
        out[vocab.index(value)] = 1.0
    return out


def _scalar(value: Optional[float], lo: float, hi: float) -> np.ndarray:
    if value is None:
        return np.zeros(1, dtype=np.float32)
    try:
        v = (float(value) - lo) / (hi - lo) if hi > lo else float(value)
    except (TypeError, ValueError):
        return np.zeros(1, dtype=np.float32)
    return np.array([float(np.clip(v, -2.0, 2.0))], dtype=np.float32)


# Stable small vocabularies for bounded categoricals.
_FACES = ("top", "bottom", "north", "south", "east", "west")
_WEATHER = ("clear", "rain", "snow", "thunder", "unknown")
_DIMS = ("minecraft:overworld", "minecraft:the_nether", "minecraft:the_end")
_NEIGHBOR_DIRS = ("py", "ny", "px", "nx", "pz", "nz")   # up/down then horizontal
_NEIGHBOR_HASH_DIM = 12     # per-direction hashed block-id width


# ---------------------------------------------------------------------------
# Feature definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextFeature:
    """One named input. ``extract(metadata) -> (value_vector, present)``.
    ``value_vector`` is length ``dim``; ``present`` is whether the metadata
    actually carried this signal (False -> the encoder returns zeros and the
    model sees the presence flag drop to 0)."""
    name: str
    dim: int
    extract: Callable[[Dict[str, Any]], Tuple[np.ndarray, bool]]


# ── Individual extractors ──────────────────────────────────────────────────

def _f_face(m):
    v = m.get("face")
    return _onehot(v, _FACES), v is not None

def _f_weather(m):
    v = m.get("weather")
    return _onehot(v if v in _WEATHER else ("unknown" if v else None), _WEATHER), v is not None

def _f_dimension(m):
    v = (m.get("pose") or {}).get("dimension") or m.get("dimension")
    return _onehot(v, _DIMS), v is not None

def _f_biome(m):
    v = m.get("biome")
    return _hash_encode(v, 16), v is not None

def _f_sky_brightness(m):
    v = m.get("sky_brightness")
    v = None if (v is None or v < 0) else v
    return _scalar(v, 0.0, 255.0), v is not None

def _f_sky_light(m):
    v = m.get("sky_light")
    return _scalar(v, 0.0, 15.0), v is not None

def _f_block_light(m):
    v = m.get("block_light")
    return _scalar(v, 0.0, 15.0), v is not None

def _f_distance(m):
    v = m.get("distance_blocks")
    return _scalar(v, 0.0, 48.0), v is not None

def _f_y(m):
    y = (m.get("pose") or {}).get("y")
    if y is None:
        tv = m.get("target_voxel")
        y = tv[1] if isinstance(tv, (list, tuple)) and len(tv) == 3 else None
    return _scalar(y, -64.0, 320.0), y is not None

def _f_is_surface(m):
    v = m.get("is_surface")
    return (np.array([1.0 if v else 0.0], dtype=np.float32), v is not None)

def _f_fov(m):
    v = m.get("h_fov_deg")
    return _scalar(v, 30.0, 130.0), v is not None

def _f_held(m):
    return _hash_encode(m.get("held_item"), 8), m.get("held_item") is not None

def _f_offhand(m):
    return _hash_encode(m.get("offhand_item"), 8), m.get("offhand_item") is not None

def _f_neighbors(m):
    """The 6 confirmed axis-neighbours' block ids, each feature-hashed. The
    strongest spatial prior — blocks cluster (a trunk is logs, ground is
    grass/dirt). Unknown/unconfirmed neighbours hash to zero. Present iff at
    least one neighbour is known."""
    nb = m.get("neighbors") or {}
    parts, any_known = [], False
    for d in _NEIGHBOR_DIRS:
        bid = nb.get(d)
        if bid:
            any_known = True
        parts.append(_hash_encode(bid, _NEIGHBOR_HASH_DIM))
    return np.concatenate(parts), any_known


# ---------------------------------------------------------------------------
# The DEFAULT (ordered) feature set. APPEND here to add an input.
# ---------------------------------------------------------------------------
DEFAULT_FEATURES: List[ContextFeature] = [
    ContextFeature("face",           len(_FACES),                _f_face),
    ContextFeature("weather",        len(_WEATHER),              _f_weather),
    ContextFeature("dimension",      len(_DIMS),                 _f_dimension),
    ContextFeature("biome",          16,                         _f_biome),
    ContextFeature("sky_brightness", 1,                          _f_sky_brightness),
    ContextFeature("sky_light",      1,                          _f_sky_light),
    ContextFeature("block_light",    1,                          _f_block_light),
    ContextFeature("distance",       1,                          _f_distance),
    ContextFeature("y",              1,                          _f_y),
    ContextFeature("is_surface",     1,                          _f_is_surface),
    ContextFeature("fov",            1,                          _f_fov),
    ContextFeature("held_item",      8,                          _f_held),
    ContextFeature("offhand_item",   8,                          _f_offhand),
    ContextFeature("neighbors",      len(_NEIGHBOR_DIRS) * _NEIGHBOR_HASH_DIM, _f_neighbors),
]


class ContextFeatureSet:
    """An ordered, versioned set of context features. ``encode(metadata)``
    returns the full fixed-length vector (each feature's ``dim`` values
    followed by its presence flag). The model derives its context-input width
    from ``total_dim`` and stores ``signature`` so a checkpoint records exactly
    which inputs (and widths) it was trained with."""

    def __init__(self, features: Optional[List[ContextFeature]] = None):
        self.features = list(features if features is not None else DEFAULT_FEATURES)

    @property
    def total_dim(self) -> int:
        return sum(f.dim + 1 for f in self.features)   # +1 presence flag each

    @property
    def signature(self) -> str:
        return ";".join(f"{f.name}:{f.dim}" for f in self.features)

    def names(self) -> List[str]:
        return [f.name for f in self.features]

    def encode(self, metadata: Optional[Dict[str, Any]]) -> np.ndarray:
        m = metadata or {}
        chunks: List[np.ndarray] = []
        for f in self.features:
            try:
                vec, present = f.extract(m)
                vec = np.asarray(vec, dtype=np.float32).ravel()
                if vec.shape[0] != f.dim:        # defensive: pad/trim to spec
                    fixed = np.zeros(f.dim, dtype=np.float32)
                    fixed[: min(f.dim, vec.shape[0])] = vec[: f.dim]
                    vec = fixed
            except Exception:
                vec, present = np.zeros(f.dim, dtype=np.float32), False
            chunks.append(vec)
            chunks.append(np.array([1.0 if present else 0.0], dtype=np.float32))
        return np.concatenate(chunks).astype(np.float32)


__all__ = ["ContextFeature", "ContextFeatureSet", "DEFAULT_FEATURES"]
