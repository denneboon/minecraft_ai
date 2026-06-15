# vision/pose_filter.py
"""
Robust filtering for noisy F3 pose reads.

Why this exists
---------------
The F3 OCR is the AI's only window onto the player's state, but it
isn't reliable on busy backgrounds: glyph misreads turn ``-110.500``
into ``-.L|.076`` or ``10.5`` depending on which substring the regex
matches first. Without filtering, a single garbled tick can:

  * relocate the perception layer's eye position 100+ blocks away,
  * fill the curiosity queue with voxels that don't exist there,
  * send the explorer agent panning toward fictional targets,
  * corrupt the mouse-to-degree auto-calibrator.

This module wraps each raw F3 read in a sanity check before it reaches
the perception layer. The check uses two simple physical priors:

  1. Player position (XYZ) cannot change by more than ``max_dxz_per_sec``
     blocks/sec horizontally or ``max_dy_per_sec`` blocks/sec vertically.
     Even with sprint + jump + creative-mode flying, plausible motion
     is < 25 blocks/sec.
  2. Yaw + pitch can change arbitrarily fast (a 360° flick is allowed)
     but the absolute value of pitch is clamped to [-90, +90] by MC,
     so any read outside that range is OCR garbage.

When a read fails the check, we hold the previous good pose (or
return None if there's no prior history). That way the agent always
sees a plausible pose or nothing — never a wild outlier.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PoseFilterConfig:
    # Hard caps on positional velocity (blocks/sec). Sprint+jump tops
    # out near 7 blocks/sec; creative-mode flying with sprint is
    # capped at 22 blocks/sec. 30 leaves comfortable headroom.
    max_dxz_per_sec: float = 30.0
    max_dy_per_sec:  float = 80.0   # elytra dives can be fast vertically

    # Hard caps on Y. Vanilla overworld is -64..320; nether 0..256;
    # end 0..256. We use the union plus margin.
    y_min: float = -100.0
    y_max: float = 400.0

    # Pitch is clamped to ±90 by MC. Allow a tiny margin for OCR
    # noise but reject anything past that.
    pitch_max_abs: float = 90.5

    # If we've been holding the previous pose for this long without
    # a fresh good read, the old continuity check no longer applies —
    # the player may have moved far while we were blind. But we do NOT
    # trust a single read after the gap (an in-range-but-wrong XYZ would
    # "teleport the eye" and become ground truth). Instead we RE-ACQUIRE:
    # accept only after this many consecutive reads that are mutually
    # consistent (low velocity between successive reads), so one spurious
    # jump can't stick while a genuine new position (stable across reads)
    # still re-locks quickly.
    max_hold_seconds: float = 1.5
    reacq_consistent_reads: int = 3      # consecutive consistent reads to re-lock
    reacq_max_gap_sec: float = 1.0       # max time between them to count as a streak

    # Reads where ALL fields are None are not "outliers" — they're
    # the OCR saying "I couldn't read this frame". Just pass them
    # through (the agent already handles no-pose by halting).


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

class PoseFilter:
    """
    Stateful single-stream pose validator.

    Usage::

        f = PoseFilter()
        ...
        good = f.accept(fresh_f3_info, now=time.perf_counter())
        # ``good`` is either the same F3Info passed in or the
        # previous good one (if the new one violated the priors).
    """

    def __init__(self, config: Optional[PoseFilterConfig] = None):
        self.cfg = config or PoseFilterConfig()
        self._last_good = None         # F3Info
        self._last_good_ts: float = 0.0
        self.n_accepted: int = 0
        self.n_rejected: int = 0
        self.last_reject_reason: Optional[str] = None
        # Re-acquisition streak (after a long blind gap, see accept()).
        self._reacq_candidate = None   # F3Info of the streak's last read
        self._reacq_ts: float = 0.0
        self._reacq_streak: int = 0

    def reset(self) -> None:
        self._last_good = None
        self._last_good_ts = 0.0
        self.n_accepted = 0
        self.n_rejected = 0
        self.last_reject_reason = None
        self._reacq_candidate = None
        self._reacq_ts = 0.0
        self._reacq_streak = 0

    def accept(self, info, now: Optional[float] = None):
        """
        Pass an F3Info through the filter. Returns the same info if
        accepted (and stores it as the new last-good), or the
        previous last-good if the new read failed the priors.

        If ``info`` is None or has no parsed position, passes it
        through unchanged (a no-read is not a bad read).
        """
        if info is None:
            return None
        if info.x is None and info.yaw is None and info.pitch is None:
            # Nothing parsed — just let it through. The agent
            # already handles "no pose" gracefully.
            return info

        # NaN / Inf guard. A malformed OCR can occasionally return
        # ``float('nan')`` or ``float('inf')`` for a numeric field
        # (e.g. when the glyph for a digit is mis-decoded as a
        # punctuation that parses to inf in some locales). Pythonic
        # comparisons with NaN always return False, so the existing
        # ``y_min <= y <= y_max`` check would let NaN through and
        # the perception eye would teleport to NaN-land. Detect
        # explicitly + reject.
        for fld in ("x", "y", "z", "yaw", "pitch"):
            v = getattr(info, fld, None)
            if v is not None and not math.isfinite(v):
                return self._reject(info, f"non-finite {fld}={v!r}")

        now = now if now is not None else time.perf_counter()

        # Hard physical limits first — these reject impossible reads
        # without needing a previous-good reference.
        if (info.y is not None
                and not (self.cfg.y_min <= info.y <= self.cfg.y_max)):
            return self._reject(info, f"y={info.y} out of [{self.cfg.y_min},"
                                       f"{self.cfg.y_max}]")
        if (info.pitch is not None
                and abs(info.pitch) > self.cfg.pitch_max_abs):
            return self._reject(info,
                                f"|pitch|={abs(info.pitch):.1f} > "
                                f"{self.cfg.pitch_max_abs}")

        # Continuity check against the previous good read.
        if self._last_good is None:
            return self._accept(info, now)

        dt = max(1e-3, now - self._last_good_ts)
        if dt > self.cfg.max_hold_seconds:
            # Held the same pose past the hold window — continuity vs the stale
            # last_good no longer applies (the player may have moved far while
            # we were blind). But DON'T trust a single read: an in-range-but-
            # wrong XYZ would teleport the eye. Re-acquire only after N reads
            # that are mutually consistent with each other.
            return self._reacquire(info, now)

        # A normal continuity-checked read resumes -> abandon any half-built
        # re-acquisition streak (we never lost the lock).
        self._reacq_streak = 0
        self._reacq_candidate = None

        last = self._last_good
        if (info.x is not None and last.x is not None
                and info.z is not None and last.z is not None):
            dxz = ((info.x - last.x) ** 2 + (info.z - last.z) ** 2) ** 0.5
            if dxz / dt > self.cfg.max_dxz_per_sec:
                return self._reject(info,
                    f"dxz={dxz:.1f} in {dt*1000:.0f}ms "
                    f"(>{self.cfg.max_dxz_per_sec} blocks/s)")
        if (info.y is not None and last.y is not None):
            dy = abs(info.y - last.y)
            if dy / dt > self.cfg.max_dy_per_sec:
                return self._reject(info,
                    f"dy={dy:.1f} in {dt*1000:.0f}ms "
                    f"(>{self.cfg.max_dy_per_sec} blocks/s)")

        return self._accept(info, now)

    # ── Internals ────────────────────────────────────────────────

    def _within_caps(self, info, ref, dt: float) -> bool:
        """True if moving from ``ref`` to ``info`` in ``dt`` s respects the
        positional velocity caps (fields that are None on either side skip)."""
        dt = max(1e-3, dt)
        if (info.x is not None and ref.x is not None
                and info.z is not None and ref.z is not None):
            dxz = ((info.x - ref.x) ** 2 + (info.z - ref.z) ** 2) ** 0.5
            if dxz / dt > self.cfg.max_dxz_per_sec:
                return False
        if info.y is not None and ref.y is not None:
            if abs(info.y - ref.y) / dt > self.cfg.max_dy_per_sec:
                return False
        return True

    def _reacquire(self, info, now: float):
        """After a long blind gap, re-lock only on a STREAK of consecutive
        reads that are mutually consistent (low velocity between them) — never
        on a single read, which could be an in-range-but-wrong jump. A read
        that breaks consistency restarts the streak from itself. Until the
        streak is long enough we keep returning the last good pose."""
        cand, cts = self._reacq_candidate, self._reacq_ts
        if (cand is not None
                and (now - cts) <= self.cfg.reacq_max_gap_sec
                and self._within_caps(info, cand, now - cts)):
            self._reacq_streak += 1
        else:
            self._reacq_streak = 1            # (re)start the streak at this read
        self._reacq_candidate = info
        self._reacq_ts = now
        if self._reacq_streak >= self.cfg.reacq_consistent_reads:
            self._reacq_streak = 0
            self._reacq_candidate = None
            return self._accept(info, now)
        return self._reject(
            info, f"re-acquiring "
                  f"({self._reacq_streak}/{self.cfg.reacq_consistent_reads})")

    def _accept(self, info, now: float):
        self._last_good = info
        self._last_good_ts = now
        self.n_accepted += 1
        self.last_reject_reason = None
        return info

    def _reject(self, info, reason: str):
        self.n_rejected += 1
        self.last_reject_reason = reason
        # Return the previous good read so downstream code sees a
        # plausible pose. If we have none, return None — the
        # agent handles that.
        return self._last_good


__all__ = ["PoseFilter", "PoseFilterConfig"]
