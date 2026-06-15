#!/usr/bin/env python3
"""Offline self-test for PoseFilter — the velocity/continuity validator that
guards the F3 pose stream. Focus: the RE-ACQUISITION behaviour after a long
blind gap (the fix for "a single in-range-but-wrong read teleports the eye")."""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

from vision.ocr import F3Info
from vision.pose_filter import PoseFilter, PoseFilterConfig

_fails = 0
def ok(m): print(f"  [OK]  {m}")
def bad(m):
    global _fails; _fails += 1; print(f"  [FAIL] {m}")


def _p(x, y, z, yaw=0.0, pitch=0.0):
    return F3Info(x=x, y=y, z=z, yaw=yaw, pitch=pitch)


def main() -> int:
    print("=" * 56); print(" pose_filter — offline self-test"); print("=" * 56)
    cfg = PoseFilterConfig()        # defaults: max_hold 1.5s, reacq 3 reads

    # 1. Basic continuity: a plausible step is accepted; an impossible jump
    #    within the hold window is rejected (returns the last good pose).
    print("\n[1] continuity within the hold window")
    f = PoseFilter(cfg)
    a = _p(0, 64, 0)
    (ok if f.accept(a, now=0.0) is a else bad)("first read accepted")
    b = _p(1, 64, 0)
    (ok if f.accept(b, now=0.1) is b else bad)("small step accepted")
    jump = _p(9999, 64, 0)          # ~10k blocks in 0.1s -> impossible
    (ok if f.accept(jump, now=0.2) is b else bad)(
        "impossible jump rejected -> returns last good")

    # 2. THE FIX: after a long blind gap, a SINGLE in-range read must NOT be
    #    trusted (it would teleport the eye). It takes a streak of mutually
    #    consistent reads to re-acquire.
    print("\n[2] re-acquisition after a long gap needs a consistent streak")
    f = PoseFilter(cfg)
    f.accept(_p(0, 64, 0), now=0.0)
    far = _p(500, 70, 500)          # plausible absolute values, far from last
    r1 = f.accept(far, now=10.0)    # gap >> max_hold -> first re-acq read
    (ok if r1 is not far else bad)("one read after the gap is NOT accepted")
    r2 = f.accept(_p(500, 70, 500), now=10.1)
    (ok if r2 is not None and (r2.x, r2.z) != (500, 500) else bad)(
        "second consistent read still not accepted (streak < 3)")
    locked = f.accept(_p(500, 70, 500), now=10.2)
    (ok if locked is not None and locked.x == 500 and locked.z == 500 else bad)(
        "third consistent read RE-ACQUIRES the new position")

    # 3. Garbage after a gap (every read a different wild value) never re-locks
    #    — the streak keeps resetting, so the eye is never teleported.
    print("\n[3] inconsistent reads after a gap never re-acquire")
    f = PoseFilter(cfg)
    good = _p(0, 64, 0)
    f.accept(good, now=0.0)
    for i, x in enumerate((3000, -3000, 8000, -8000, 1234)):
        out = f.accept(_p(x, 64, x), now=10.0 + i * 0.1)
        if out is not None and out.x == x:
            bad(f"wild read x={x} wrongly accepted"); break
    else:
        ok("5 mutually-inconsistent reads all rejected (no teleport)")

    # 4. A genuine teleport (player really moved, then sits still) re-locks
    #    quickly — recovery isn't broken by the stricter gate.
    print("\n[4] a real new position (held still) re-locks in N reads")
    f = PoseFilter(cfg)
    f.accept(_p(0, 64, 0), now=0.0)
    out = None
    for i in range(cfg.reacq_consistent_reads):
        out = f.accept(_p(-2000, 30, 1500), now=20.0 + i * 0.1)
    (ok if out is not None and out.x == -2000 and out.y == 30 else bad)(
        f"re-locked at the new spot after {cfg.reacq_consistent_reads} reads")

    # 5. Hard caps still reject before re-acquisition even considers a read.
    print("\n[5] hard Y cap rejects regardless of the gap")
    f = PoseFilter(cfg)
    f.accept(_p(0, 64, 0), now=0.0)
    out = f.accept(_p(0, 99999, 0), now=10.0)     # Y way out of range
    (ok if out is None or out.y != 99999 else bad)("out-of-range Y not accepted")

    print("\n" + ("ALL POSE_FILTER TESTS PASSED" if not _fails
                  else f"{_fails} CHECK(S) FAILED"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
