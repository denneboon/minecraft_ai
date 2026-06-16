# vision/world/map.py
"""
WorldMap — the AI's persistent 3-D memory of blocks, entities, and
item drops it has observed via vision.

Design
------
* **Sparse**: a Python dict keyed by ``(x, y, z)`` integer block
  coords. Minecraft worlds are ~30 M blocks per axis; a dense voxel
  grid is out of the question.
* **Dimension-aware**: one sub-store per dimension id
  (``minecraft:overworld``, ``…the_nether``, ``…the_end``). Travelling
  through a portal does not erase the other dimension's memory.
* **Confidence-weighted updates**: a high-confidence observation
  overwrites a lower-confidence one; a same-confidence observation
  refreshes the timestamp only. Stronger sources (looking_at_block,
  manual annotations) win over weaker ones (vision_patch).
* **Decay**: old observations can be aged out via
  ``forget_older_than`` or by capping the per-dimension store size.
  This stops long sessions from hoarding gigabytes of stale guesses.
* **Read-mostly**: the runtime pipeline writes new observations every
  tick; agents read sub-volumes, neighbourhoods, or single voxels at
  any time. All accessors are O(1) per voxel.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from vision.world.types import (
    BlockObservation,
    EntityObservation,
    ItemDropObservation,
)


# Source authority: higher number = more trusted. Used when deciding
# whether a new observation should overwrite an existing one.
_SOURCE_AUTHORITY: Dict[str, int] = {
    "broken":          6,   # the bot itself broke this block -> it's now air
    "manual":          5,
    "looking_at":      4,
    "ray_clear_air":   3,   # carved as air along a confirmed F3 sightline
    "vision_patch":    2,
    "extrapolation":   1,
    "unknown":         0,
}

# Sentinel block id used to mark a voxel as KNOWN-empty (air). Putting
# air observations into the same map (instead of a separate "free
# space" set) means a single ``get_block(pos)`` answers all three
# states an agent cares about: solid (id != AIR_BLOCK), air (id ==
# AIR_BLOCK), unknown (returns None). Renderers can colour the three
# differently.
AIR_BLOCK = "minecraft:air"


def _source_score(src: str) -> int:
    return _SOURCE_AUTHORITY.get(src, 0)


# ---------------------------------------------------------------------------
# Per-dimension sub-store
# ---------------------------------------------------------------------------

@dataclass
class _DimensionStore:
    """All observations seen in a single MC dimension."""
    blocks: Dict[Tuple[int, int, int], BlockObservation] = field(default_factory=dict)
    entities: Dict[str, EntityObservation] = field(default_factory=dict)   # keyed by detection-uid
    drops:   Dict[str, ItemDropObservation] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# WorldMap
# ---------------------------------------------------------------------------

class WorldMap:
    """
    Cross-dimension store of everything the AI has ever observed.

    The map is *append-mostly*: blocks are rarely removed (a missing
    update doesn't mean "the block is gone", just "I didn't look at
    it"). Entities and drops are *append-and-decay*: an entity not
    seen for ``entity_ttl_ticks`` is dropped from the store, because
    mobs and drops move.

    Thread-safety: not thread-safe. WorldPerception calls into this on
    the agent loop's thread; if you ever read from a second thread,
    wrap calls in a lock.
    """

    # Default memory limits — generous for one play session.
    DEFAULT_MAX_BLOCKS_PER_DIM     = 250_000
    DEFAULT_ENTITY_TTL_TICKS       = 100        # ~5 s at 20 Hz
    DEFAULT_DROP_TTL_TICKS         = 200        # ~10 s

    def __init__(self,
                 max_blocks_per_dim: int = DEFAULT_MAX_BLOCKS_PER_DIM,
                 entity_ttl_ticks:   int = DEFAULT_ENTITY_TTL_TICKS,
                 drop_ttl_ticks:     int = DEFAULT_DROP_TTL_TICKS):
        self._dims: Dict[str, _DimensionStore] = {}
        self._max_blocks_per_dim = int(max_blocks_per_dim)
        self._entity_ttl_ticks   = int(entity_ttl_ticks)
        self._drop_ttl_ticks     = int(drop_ttl_ticks)
        self._current_dim: str   = "minecraft:overworld"
        self._created_at: float  = time.time()

    # ── Dimension switching ────────────────────────────────────────

    def set_current_dimension(self, dimension: str) -> None:
        """Note the dimension the player is currently in. New
        observations default to this dimension if their own field is
        unset."""
        if dimension:
            self._current_dim = dimension
            self._dims.setdefault(dimension, _DimensionStore())

    def current_dimension(self) -> str:
        return self._current_dim

    def dimensions(self) -> List[str]:
        return sorted(self._dims.keys())

    def _store(self, dim: Optional[str] = None) -> _DimensionStore:
        d = dim or self._current_dim
        return self._dims.setdefault(d, _DimensionStore())

    # ── Block API ──────────────────────────────────────────────────

    def update_block(self, obs: BlockObservation) -> bool:
        """
        Insert or update a block observation.

        Returns True if the store was modified. We only overwrite an
        existing entry when the new observation is more authoritative
        OR strictly more confident at equal authority.
        """
        store = self._store(obs.dimension)
        prev = store.blocks.get(obs.pos)
        if prev is None:
            store.blocks[obs.pos] = obs
            self._maybe_evict_blocks(store)
            return True

        new_auth = _source_score(obs.source)
        old_auth = _source_score(prev.source)
        if new_auth > old_auth or (
                new_auth == old_auth and obs.confidence >= prev.confidence):
            store.blocks[obs.pos] = obs
            return True
        # Older entry wins — but still bump its last_seen_tick so the
        # decay logic knows we revisited the cell.
        prev.last_seen_tick = max(prev.last_seen_tick, obs.last_seen_tick)
        return False

    def get_block(self,
                  pos: Tuple[int, int, int],
                  dimension: Optional[str] = None,
                  ) -> Optional[BlockObservation]:
        """BELIEF map: the best observation at ``pos`` from ANY source
        (F3-confirmed OR a CNN/NN guess). Use when you want the fullest
        picture and can tolerate a guess being wrong."""
        return self._store(dimension).blocks.get(pos)

    # ── The two maps: CONFIRMED (ground truth) vs BELIEF (incl. guesses) ──
    # The belief map is ``get_block`` / ``iter_blocks`` above. The confirmed
    # map is the same store filtered to F3 "looking_at" observations — the
    # only source the multi-frame + ray + catalog gates guarantee is real.
    _CONFIRMED_SOURCES = ("looking_at", "ray_clear_air", "broken")

    def get_confirmed(self,
                      pos: Tuple[int, int, int],
                      dimension: Optional[str] = None,
                      ) -> Optional[BlockObservation]:
        """CONFIRMED map: the observation at ``pos`` ONLY if it was F3
        looking-at-confirmed (or carved air along a confirmed sightline) —
        i.e. ground truth. Returns None for guessed/extrapolated voxels, so a
        caller that must never act on a guess (build-safety, label export) can
        rely on it. Never emits a stray block from a CNN/NN guess."""
        obs = self._store(dimension).blocks.get(pos)
        if obs is not None and getattr(obs, "source", None) in self._CONFIRMED_SOURCES:
            return obs
        return None

    def iter_confirmed(self,
                       dimension: Optional[str] = None,
                       *, include_air: bool = False) -> Iterable[BlockObservation]:
        """Every CONFIRMED (looking_at) observation — the ground-truth map.
        Air (carved sightlines) is excluded unless ``include_air``."""
        # Snapshot: perception (and the background CNN trainer) mutate the store
        # on another path, so iterating .values() live can raise "dict changed
        # size during iteration" in a viewer mid-render.
        for o in list(self._store(dimension).blocks.values()):
            if getattr(o, "source", None) not in self._CONFIRMED_SOURCES:
                continue
            if (not include_air) and o.block_id == AIR_BLOCK:
                continue
            yield o

    def block_count(self, dimension: Optional[str] = None) -> int:
        return len(self._store(dimension).blocks)

    def iter_blocks_in_range(self,
                             center: Tuple[int, int, int],
                             radius: int,
                             *,
                             dimension: Optional[str] = None,
                             ) -> Iterable[BlockObservation]:
        """
        Yield every observed block within a Chebyshev radius (cube) of
        ``center``. Useful for "what's around me right now?" queries.
        """
        cx, cy, cz = center
        store = self._store(dimension)
        # Iterating the whole store is fine while we're under
        # ~250 K entries; if that ever changes we'll add a chunk
        # index keyed by 16-block bins.
        for pos, obs in list(store.blocks.items()):   # snapshot (concurrent mutation)
            if (abs(pos[0] - cx) <= radius
                    and abs(pos[1] - cy) <= radius
                    and abs(pos[2] - cz) <= radius):
                yield obs

    def iter_blocks(self, dimension: Optional[str] = None) -> Iterable[BlockObservation]:
        return iter(list(self._store(dimension).blocks.values()))   # snapshot

    def iter_solid_blocks(self,
                          dimension: Optional[str] = None
                          ) -> Iterable[BlockObservation]:
        """Yield only observations of SOLID blocks (excludes air)."""
        for o in list(self._store(dimension).blocks.values()):      # snapshot
            if o.block_id and o.block_id != AIR_BLOCK:
                yield o

    def mark_air_along(self,
                        voxels: Iterable[Tuple[int, int, int]],
                        *,
                        dimension: Optional[str] = None,
                        confidence: float = 0.95,
                        tick: int = 0,
                        ) -> int:
        """
        Bulk-mark a sequence of voxels as known-empty (air).

        Used after a confirmed F3 looking_at observation: every voxel
        the sight-line passes through MUST be air (or transparent),
        because we could see THROUGH it to the targeted block beyond.
        That's a strong free-space signal — the agent now knows it
        could walk along that line, and the renderer can show the
        carved-out volume.

        Returns the number of voxels that were inserted or updated.
        """
        n_changed = 0
        for pos in voxels:
            if self.update_block(BlockObservation(
                pos=pos, block_id=AIR_BLOCK,
                confidence=confidence,
                source="ray_clear_air",
                last_seen_tick=tick,
                dimension=dimension or self._current_dim,
            )):
                n_changed += 1
        return n_changed

    def mark_broken(self,
                    pos: Tuple[int, int, int],
                    *,
                    dimension: Optional[str] = None,
                    tick: int = 0,
                    confidence: float = 1.0) -> None:
        """Record that the bot just BROKE the block at ``pos`` — it is now air.

        Written at the highest authority (``broken``) so it OVERWRITES a stale
        confirmed solid: otherwise a mined log/stone lingers in the map forever
        (decay is unused), and the chopper keeps re-walking to a log that's
        already gone (the live "sees a log, then looks elsewhere / can't chop"
        symptom). A confirmed break is the strongest possible free-space signal,
        so it outranks even a fresh F3 ``looking_at`` read."""
        self.update_block(BlockObservation(
            pos=tuple(pos), block_id=AIR_BLOCK,
            confidence=confidence,
            source="broken",
            last_seen_tick=tick,
            dimension=dimension or self._current_dim,
        ))

    def forget_blocks_older_than(self,
                                 cutoff_tick: int,
                                 dimension: Optional[str] = None,
                                 ) -> int:
        """
        Drop block observations not refreshed since ``cutoff_tick``.

        Returns the number of entries removed. Don't use this casually
        — for the "world doesn't disappear when I look away" semantic,
        keep blocks indefinitely and rely on
        :meth:`forget_blocks_below_confidence` for cleanup instead.
        """
        store = self._store(dimension)
        before = len(store.blocks)
        store.blocks = {
            p: o for p, o in store.blocks.items() if o.last_seen_tick >= cutoff_tick
        }
        return before - len(store.blocks)

    def forget_blocks_below_confidence(self,
                                       min_confidence: float,
                                       dimension: Optional[str] = None,
                                       ) -> int:
        store = self._store(dimension)
        before = len(store.blocks)
        store.blocks = {
            p: o for p, o in store.blocks.items() if o.confidence >= min_confidence
        }
        return before - len(store.blocks)

    def _maybe_evict_blocks(self, store: _DimensionStore) -> None:
        """If a single dimension exceeds the size cap, drop the oldest
        low-confidence entries until we're back under it.

        Eviction TARGET is 90% of the cap so we have headroom before
        the next eviction. The previous behaviour dropped a fixed 10%
        which could lag behind insertions during heavy F3-confirm
        bursts — the store could grow past the cap unbounded over a
        long session. Now we evict at least ``cap - target`` entries
        (capped at half the store to bound CPU per call); over time
        the per-tick eviction rate matches the per-tick insertion
        rate by construction.
        """
        cap = self._max_blocks_per_dim
        if len(store.blocks) <= cap:
            return
        target = int(cap * 0.9)
        # Sort by (confidence asc, last_seen_tick asc) once. We need
        # to drop enough to hit ``target`` but never more than half
        # the store in a single eviction so a temporary overshoot
        # doesn't decimate good observations.
        # Drop exactly enough to reach ``target`` (worst entries first). The
        # previous half-the-store cap could leave the store ABOVE ``cap`` after
        # a big overshoot — the bound wasn't actually guaranteed. Dropping to
        # target in one pass restores it; the O(n log n) sort dominates cost
        # either way, so the cap bought nothing.
        n_drop = max(1, len(store.blocks) - target)
        # Evict by (authority, confidence, recency) ascending — lowest FIRST.
        # Authority must lead the key: a CNN/extrapolation guess (or carved air
        # at conf 0.95) must be dropped before an F3-confirmed solid, even when
        # the guess has a higher confidence number. Sorting on confidence alone
        # could evict hard-won ground truth and keep a guess.
        items = sorted(
            store.blocks.items(),
            key=lambda kv: (_source_score(kv[1].source),
                            kv[1].confidence, kv[1].last_seen_tick),
        )
        for k, _ in items[:n_drop]:
            del store.blocks[k]

    # ── Entity API ─────────────────────────────────────────────────

    def update_entity(self,
                      key: str,
                      obs: EntityObservation) -> None:
        """
        Insert or refresh an entity observation. ``key`` is a stable id
        chosen by the caller (e.g. tracker id, or a quantised world
        position hash) so the same physical entity isn't double-stored
        across frames.
        """
        self._store(obs.dimension).entities[key] = obs

    def get_entity(self, key: str,
                   dimension: Optional[str] = None) -> Optional[EntityObservation]:
        return self._store(dimension).entities.get(key)

    def iter_entities(self,
                      dimension: Optional[str] = None,
                      ) -> Iterable[EntityObservation]:
        return iter(self._store(dimension).entities.values())

    def decay_entities(self, current_tick: int,
                       dimension: Optional[str] = None) -> int:
        """Drop entities not seen in the last ``entity_ttl_ticks`` ticks."""
        store = self._store(dimension)
        cutoff = current_tick - self._entity_ttl_ticks
        before = len(store.entities)
        store.entities = {
            k: e for k, e in store.entities.items() if e.last_seen_tick >= cutoff
        }
        return before - len(store.entities)

    # ── Drops API ──────────────────────────────────────────────────

    def update_drop(self, key: str, obs: ItemDropObservation) -> None:
        self._store(obs.dimension).drops[key] = obs

    def iter_drops(self, dimension: Optional[str] = None
                  ) -> Iterable[ItemDropObservation]:
        return iter(self._store(dimension).drops.values())

    def decay_drops(self, current_tick: int,
                    dimension: Optional[str] = None) -> int:
        store = self._store(dimension)
        cutoff = current_tick - self._drop_ttl_ticks
        before = len(store.drops)
        store.drops = {
            k: d for k, d in store.drops.items() if d.last_seen_tick >= cutoff
        }
        return before - len(store.drops)

    # ── Bulk maintenance ───────────────────────────────────────────

    def clear(self, dimension: Optional[str] = None) -> None:
        if dimension is None:
            self._dims.clear()
        else:
            self._dims.pop(dimension, None)

    def stats(self) -> Dict[str, Any]:
        return {
            "dimensions": {
                d: {
                    "blocks":   len(s.blocks),
                    "entities": len(s.entities),
                    "drops":    len(s.drops),
                }
                for d, s in self._dims.items()
            },
            "current_dimension": self._current_dim,
            "max_blocks_per_dim": self._max_blocks_per_dim,
        }

    # ── Serialisation ──────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Snapshot the map as plain Python data. Used by debug tools
        and offline replay; not on the hot path."""
        out: Dict[str, Any] = {"current_dimension": self._current_dim,
                               "dimensions": {}}
        for d, s in self._dims.items():
            out["dimensions"][d] = {
                "blocks": [
                    {
                        "pos": list(p),
                        "block_id": o.block_id,
                        "confidence": round(float(o.confidence), 3),
                        "source": o.source,
                        "last_seen_tick": int(o.last_seen_tick),
                    }
                    for p, o in s.blocks.items()
                ],
                "entities": [
                    {
                        "entity_id": e.entity_id,
                        "screen_box": list(e.screen_box),
                        "world_pos": (None if e.world_pos is None
                                       else [round(float(v), 3) for v in e.world_pos]),
                        "confidence": round(float(e.confidence), 3),
                        "source": e.source,
                        "last_seen_tick": int(e.last_seen_tick),
                    }
                    for e in s.entities.values()
                ],
                "drops": [
                    {
                        "item_id": d.item_id,
                        "screen_box": list(d.screen_box),
                        "world_pos": (None if d.world_pos is None
                                       else [round(float(v), 3) for v in d.world_pos]),
                        "confidence": round(float(d.confidence), 3),
                        "last_seen_tick": int(d.last_seen_tick),
                    }
                    for d in s.drops.values()
                ],
            }
        return out

    def dump_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


__all__ = ["WorldMap"]
