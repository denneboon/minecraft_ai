# vision/world/temporal_vote.py
"""
Per-voxel temporal vote smoothing for the block recogniser.

A single-frame vision-patch classification can flip between visually
similar blocks (oak_leaves ↔ vine, grass_block ↔ moss) as lighting,
sub-pixel aim, and shading jitter frame-to-frame. When the agent looks
at the SAME voxel across several frames (dwelling, the walker holding
aim, the explorer's settle phase), those independent guesses are a free
ensemble — voting over them is far more stable than trusting any one.

Design goals
------------
* NEVER suppress a first sighting. With no history a voxel's guess is
  returned unchanged (so single-frame perception — and every offline
  test that feeds one frame — behaves exactly as before).
* Correct transient flips: once there's history, return the MAJORITY
  block id over a recent-tick window, not the latest noisy guess.
* Reward agreement: confidence is scaled by how strongly the window
  agrees, so a voxel seen as ``stone`` five times reads as more certain
  than one seen once — without ever exceeding the raw confidence.
* Bounded memory: old voxels (not seen within the window) are evicted
  lazily so an 8-hour run doesn't grow the history without limit.

This is pure book-keeping (no torch / no perception deps) so it unit
tests in isolation.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple


@dataclass
class TemporalVoteConfig:
    # How many ticks back a guess still counts toward a voxel's vote.
    window_ticks: int = 50
    # Most recent guesses kept per voxel (caps per-voxel memory).
    max_history: int = 9
    # Evict a voxel's history once it hasn't been seen this many ticks
    # (keeps the table bounded on long runs).
    evict_after_ticks: int = 200


Voxel = Tuple[int, int, int]


class TemporalVoter:
    """Smooths per-voxel block guesses over a recent-tick window."""

    def __init__(self, config: TemporalVoteConfig | None = None):
        self.cfg = config or TemporalVoteConfig()
        # voxel -> deque[(tick, block_id, conf)]
        self._hist: Dict[Voxel, Deque[Tuple[int, str, float]]] = {}
        self._last_evict_tick = 0

    def vote(self, voxel: Voxel, block_id: str, conf: float,
             tick: int) -> Tuple[str, float, bool]:
        """Record ``(block_id, conf)`` for ``voxel`` at ``tick`` and return
        the smoothed ``(block_id, confidence, stable)``.

        ``stable`` is True when at least two guesses in the window agree
        on the winner — a caller may use it to gate commits more tightly.
        With no prior history the input is returned unchanged and
        ``stable`` is False.
        """
        h = self._hist.get(voxel)
        if h is None:
            h = deque(maxlen=self.cfg.max_history)
            self._hist[voxel] = h
        h.append((tick, block_id, conf))

        # Drop entries outside the recent-tick window.
        lo = tick - self.cfg.window_ticks
        while h and h[0][0] < lo:
            h.popleft()

        # Tally votes + best confidence seen per candidate in the window.
        votes: Counter = Counter()
        best_conf: Dict[str, float] = {}
        for _, bid, c in h:
            votes[bid] += 1
            if c > best_conf.get(bid, -1.0):
                best_conf[bid] = c
        win_id, win_n = votes.most_common(1)[0]
        total = sum(votes.values())

        self._maybe_evict(tick)

        if total <= 1:
            # First sighting (or only one in-window) — unchanged.
            return block_id, conf, False

        agree = win_n / total
        # Confidence = the winner's best observed confidence, gently
        # scaled by agreement (full agreement keeps it; a split halves
        # toward 0.5×). Never inflated above the observed confidence.
        voted_conf = best_conf[win_id] * (0.5 + 0.5 * agree)
        return win_id, float(voted_conf), win_n >= 2

    def _maybe_evict(self, tick: int) -> None:
        # Cheap throttle: sweep the table at most once per ~window.
        if tick - self._last_evict_tick < self.cfg.window_ticks:
            return
        self._last_evict_tick = tick
        cutoff = tick - self.cfg.evict_after_ticks
        stale = [v for v, h in self._hist.items()
                 if not h or h[-1][0] < cutoff]
        for v in stale:
            del self._hist[v]

    def purge(self) -> None:
        """Drop all history (e.g. on a dimension change)."""
        self._hist.clear()


__all__ = ["TemporalVoter", "TemporalVoteConfig"]
