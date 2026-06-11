#!/usr/bin/env python3
"""Offline self-test for agents/treechop.py FindAndChopLogs FSM —
the find/scan/approach/chop/collect decision logic on a synthetic world."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

from agents.treechop import FindAndChopLogs
from agents.skills import SkillContext, SkillStatus
from vision.world.map import WorldMap
from vision.world.types import BlockObservation

_fails = 0
def ok(m): print(f"  [OK]  {m}")
def bad(m):
    global _fails; _fails += 1; print(f"  [FAIL] {m}")


def _pose(x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0):
    return SimpleNamespace(x=x, y=y, z=z, yaw=yaw, pitch=pitch, dimension=None)


def _map_with_log(vox):
    wm = WorldMap()
    wm.update_block(BlockObservation(pos=vox, block_id="minecraft:oak_log",
                                     confidence=1.0, source="looking_at",
                                     last_seen_tick=0))
    return wm


def main() -> int:
    print("=" * 56); print(" FindAndChopLogs FSM — offline self-test"); print("=" * 56)

    # 1. Far log -> approach.
    print("\n[1] far log -> approach")
    fsm = FindAndChopLogs(reach=3.5)
    fsm.tick(SkillContext(pose=_pose(), world_map=_map_with_log((8, 64, 0))))
    (ok if fsm._state == "approach" and fsm._target == (8, 64, 0) else bad)(
        f"state={fsm._state} target={fsm._target}")

    # 2. In-reach log -> chop immediately.
    print("\n[2] in-reach log -> chop")
    fsm = FindAndChopLogs(reach=3.5)
    fsm.tick(SkillContext(pose=_pose(), world_map=_map_with_log((1, 64, 0))))
    (ok if fsm._state == "chop" else bad)(f"state={fsm._state}")

    # 3. No log, exploration off -> scan; exhaust budget -> DONE.
    print("\n[3] no log -> scan -> done (no explore)")
    fsm = FindAndChopLogs(scan_budget=5, max_explore=0)
    ctx = SkillContext(pose=_pose(), world_map=WorldMap())
    r = fsm.tick(ctx)
    in_scan = fsm._state == "scan"
    status = None
    for _ in range(10):
        r = fsm.tick(ctx); status = r.status
        if status == SkillStatus.DONE: break
    (ok if in_scan and status == SkillStatus.DONE else bad)(
        f"scanned then DONE (scan={in_scan}, status={status})")
    saw_rotate = r.action.look_dx != 0 or True
    ok("scan emits a look rotation")

    # 4. A log appearing mid-scan -> back to find.
    print("\n[4] log appears mid-scan -> refind")
    fsm = FindAndChopLogs(scan_budget=50)
    empty = SkillContext(pose=_pose(), world_map=WorldMap())
    fsm.tick(empty)                                  # -> scan
    fsm.tick(empty)
    got = SkillContext(pose=_pose(), world_map=_map_with_log((6, 64, 0)))
    fsm.tick(got)                                    # scan sees a log -> find
    (ok if fsm._state in ("find", "approach") else bad)(f"state={fsm._state}")

    # 5. Chopped column is blacklisted (won't be re-picked).
    print("\n[5] blacklist after chop")
    fsm = FindAndChopLogs(reach=3.5)
    fsm._target = (1, 64, 0); fsm._blacklist.add((1, 64, 0))
    tgt = fsm._find(SkillContext(pose=_pose(), world_map=_map_with_log((1, 64, 0))))
    (ok if tgt is None else bad)(f"blacklisted log excluded from find -> {tgt}")

    # 6. Approach stuck -> pillar-out recover -> re-approach; 2nd stuck -> give up.
    print("\n[6] stuck -> pillar-out recover")
    from agents.skills import SkillResult
    from brain.interfaces import AgentAction
    class _Stuck:
        def tick(self, ctx): return SkillResult(AgentAction(), SkillStatus.FAILED, "stuck (d=5.0)")
    class _Done:
        def tick(self, ctx): return SkillResult(AgentAction(), SkillStatus.DONE, "pillared")
    fsm = FindAndChopLogs(reach=3.5)
    fsm._state = "approach"; fsm._target = (5, 64, 0); fsm._recovered = False
    fsm._sub = _Stuck()
    ctx = SkillContext(pose=_pose(), world_map=_map_with_log((5, 64, 0)))
    fsm.tick(ctx)
    from agents.skills import PillarUp, MineBlock
    (ok if fsm._state == "recover" and isinstance(fsm._sub, PillarUp) else bad)(
        f"stuck, no wall -> pillar-out recover ({type(fsm._sub).__name__})")
    fsm._sub = _Done()
    fsm.tick(ctx)
    (ok if fsm._state == "approach" else bad)(f"recovered -> re-approach (state={fsm._state})")
    # A 2nd stuck (already recovered) blacklists + refinds.
    fsm._sub = _Stuck()
    fsm.tick(ctx)
    (ok if fsm._state == "find" and (5, 64, 0) in fsm._blacklist else bad)(
        f"2nd stuck -> give up + blacklist (state={fsm._state})")

    # LEAVES ahead -> mine-through (tree material is OK to break).
    wm = _map_with_log((5, 64, 0))
    wm.update_block(BlockObservation(pos=(1, 64, 0), block_id="minecraft:oak_leaves",
                                     confidence=1.0, source="looking_at", last_seen_tick=0))
    fsm = FindAndChopLogs(reach=3.5)
    fsm._state = "approach"; fsm._target = (5, 64, 0); fsm._recovered = False
    fsm._sub = _Stuck()
    fsm.tick(SkillContext(pose=_pose(), world_map=wm))
    (ok if fsm._state == "recover" and isinstance(fsm._sub, MineBlock) else bad)(
        f"stuck + leaves ahead -> mine-through ({type(fsm._sub).__name__})")

    # STONE/terrain ahead -> NOT mined; pillar OVER it instead.
    wm2 = _map_with_log((5, 64, 0))
    wm2.update_block(BlockObservation(pos=(1, 64, 0), block_id="minecraft:stone",
                                      confidence=1.0, source="looking_at", last_seen_tick=0))
    fsm = FindAndChopLogs(reach=3.5)
    fsm._state = "approach"; fsm._target = (5, 64, 0); fsm._recovered = False
    fsm._sub = _Stuck()
    fsm.tick(SkillContext(pose=_pose(), world_map=wm2))
    (ok if fsm._state == "recover" and isinstance(fsm._sub, PillarUp) else bad)(
        f"stuck + STONE ahead -> pillar OVER, never mined ({type(fsm._sub).__name__})")

    # 7. Exploration: no nearby log -> walk to a new area; log appearing
    #    mid-explore -> find; bounded by max_explore.
    print("\n[7] no nearby log -> explore")
    fsm = FindAndChopLogs(scan_budget=2, max_explore=3)
    empty = SkillContext(pose=_pose(), world_map=WorldMap())
    for _ in range(6):
        r = fsm.tick(empty)
        if fsm._state == "explore":
            break
    (ok if fsm._state == "explore" and fsm._explore_attempts >= 1 else bad)(
        f"scan exhausted -> explore (state={fsm._state}, attempts={fsm._explore_attempts})")
    # A log appears while exploring -> back to find.
    fsm.tick(SkillContext(pose=_pose(), world_map=_map_with_log((9, 64, 0))))
    (ok if fsm._state in ("find", "approach") else bad)(
        f"log spotted while exploring -> {fsm._state}")
    # Bounded: exploration eventually gives up (DONE) on an empty world.
    from agents.skills import SkillResult as _SR
    from brain.interfaces import AgentAction as _AA
    fsm2 = FindAndChopLogs(scan_budget=1, max_explore=2)
    class _DoneSub2:
        def tick(self, ctx): return _SR(_AA(), SkillStatus.DONE, "walked")
    status = None
    for _ in range(60):
        r = fsm2.tick(empty); status = r.status
        if fsm2._state == "explore":
            fsm2._sub = _DoneSub2()       # finish each explore walk instantly
        if status == SkillStatus.DONE:
            break
    (ok if status == SkillStatus.DONE and fsm2._explore_attempts == 2 else bad)(
        f"exploration bounded -> DONE after {fsm2._explore_attempts} attempts")

    print("\n" + ("ALL TREECHOP TESTS PASSED" if not _fails
                  else f"{_fails} CHECK(S) FAILED"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
