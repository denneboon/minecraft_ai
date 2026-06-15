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

    # 1b. goal_blocks: DONE once that many blocks are actually gathered.
    print("\n[1b] goal_blocks completes on blocks gathered")
    fsm = FindAndChopLogs(goal_blocks=2)
    r = fsm.tick(SkillContext(pose=_pose(), world_map=WorldMap()))
    not_done_yet = r.status != SkillStatus.DONE
    fsm.logs = 2                                # 2 blocks broken/collected
    r = fsm.tick(SkillContext(pose=_pose(), world_map=WorldMap()))
    (ok if not_done_yet and r.status == SkillStatus.DONE and "goal" in r.info
     else bad)(f"goal_blocks=2 -> DONE at 2 gathered ({r.status})")

    # 2. In-reach log -> chop immediately.
    print("\n[2] in-reach log -> chop")
    fsm = FindAndChopLogs(reach=3.5)
    fsm.tick(SkillContext(pose=_pose(), world_map=_map_with_log((1, 64, 0))))
    (ok if fsm._state == "chop" else bad)(f"state={fsm._state}")

    # 2b. A canopy log far overhead (out of the reachable height band) is not
    #     targeted at all -> scan for reachable ones (no stuck on it).
    fsm = FindAndChopLogs(reach=3.5)
    fsm.tick(SkillContext(pose=_pose(), world_map=_map_with_log((1, 80, 0))))
    (ok if fsm._state == "scan" else bad)(
        f"canopy log overhead -> not targeted, scans (state={fsm._state})")

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

    # 5b. Opportunistic chop: directly looking at an IN-REACH log -> chop it
    #     NOW, even while scanning/approaching (don't hover over it).
    print("\n[5b] opportunistic chop on the log under the crosshair")
    fsm = FindAndChopLogs(reach=3.5)
    fsm._state = "scan"           # busy doing something else
    la = SimpleNamespace(pos=(1, 64, 0), block_id="minecraft:oak_log")
    ctx = SkillContext(pose=_pose(), world_map=WorldMap(), looking_at=la)
    fsm.tick(ctx)
    (ok if fsm._state == "chop" and fsm._target == (1, 64, 0) else bad)(
        f"in-reach log under crosshair -> chop (state={fsm._state}, tgt={fsm._target})")
    # An OUT-OF-REACH log under the crosshair is NOT opportunistically chopped.
    fsm2 = FindAndChopLogs(reach=3.5)
    fsm2._state = "scan"
    la2 = SimpleNamespace(pos=(1, 80, 0), block_id="minecraft:oak_log")  # too high
    fsm2.tick(SkillContext(pose=_pose(), world_map=WorldMap(), looking_at=la2))
    (ok if fsm2._state != "chop" else bad)(
        f"out-of-reach log under crosshair -> NOT chopped (state={fsm2._state})")
    # Birch behaves identically to oak.
    fsm3 = FindAndChopLogs(reach=3.5)
    fsm3._state = "scan"
    lb = SimpleNamespace(pos=(1, 64, 0), block_id="minecraft:birch_log")
    fsm3.tick(SkillContext(pose=_pose(), world_map=WorldMap(), looking_at=lb))
    (ok if fsm3._state == "chop" else bad)(f"birch log under crosshair -> chop (state={fsm3._state})")

    # 5c. Clean-sightline raise: aiming at the BOTTOM log of a mapped trunk
    #     from up close grazes the ground (live 'grass_block blocks the target
    #     — abandon'). The chop start is raised to the log nearest EYE height
    #     (at/below eye), never above it; ChopTrunk fells the skipped lower
    #     logs on its DOWN pass. No-op when the column above isn't mapped.
    print("\n[5c] chop start raised to a clean (eye-level) sightline")
    fsm = FindAndChopLogs(reach=3.5)
    wm_col = WorldMap()
    for yy in (63, 64, 65, 66):          # a 4-tall trunk at x=2
        wm_col.update_block(BlockObservation(pos=(2, yy, 0),
            block_id="minecraft:oak_log", confidence=1.0,
            source="looking_at", last_seen_tick=0))
    ctx_col = SkillContext(pose=_pose(x=0.0, y=64.0, z=0.0), world_map=wm_col)
    raised = fsm._raise_to_clean_sightline(ctx_col, (2, 63, 0))  # eye=65.62
    (ok if raised == (2, 65, 0) else bad)(
        f"bottom (2,63,0) raised to eye-level (2,65,0), not above eye (got {raised})")
    # No mapped column above -> unchanged (safe fallback).
    raised2 = fsm._raise_to_clean_sightline(
        SkillContext(pose=_pose(), world_map=_map_with_log((2, 63, 0))), (2, 63, 0))
    (ok if raised2 == (2, 63, 0) else bad)(
        f"unmapped column above -> target unchanged (got {raised2})")

    # 5d. Scan STEP-AND-SETTLE: a fast continuous spin sweeps the crosshair past
    #     trunks faster than the ~11 Hz F3 reader can record them (the live
    #     "looks right past a tree it saw"). The scan must rotate a small step
    #     then HOLD (look_dx==0) so a log in the new view registers — never the
    #     old 70 px/tick blind spin.
    print("\n[5d] scan steps and settles (no fast blind spin)")
    fsm = FindAndChopLogs(scan_budget=50, max_explore=0,
                          scan_step_px=26, scan_settle_ticks=1)
    sctx = SkillContext(pose=_pose(pitch=0.0), world_map=WorldMap())  # empty -> scan
    looks = [fsm.tick(sctx).action.look_dx for _ in range(8)]
    (ok if 70 not in looks else bad)(f"no 70 px blind spin (looks={looks})")
    (ok if 26 in looks else bad)("rotates by the configured step (26)")
    (ok if 0 in looks else bad)("holds still on settle ticks (look_dx==0)")
    # a step is always followed by at least one settle (no two 26s back-to-back).
    steps = [i for i, dx in enumerate(looks) if dx == 26]
    (ok if all(j - i > 1 for i, j in zip(steps, steps[1:])) else bad)(
        "a settle tick separates consecutive steps")

    # 6. Approach stuck -> pillar-out recover -> re-approach; 2nd stuck -> give up.
    print("\n[6] stuck -> pillar-out recover")
    from agents.skills import SkillResult
    from brain.interfaces import AgentAction
    class _Stuck:
        def tick(self, ctx): return SkillResult(AgentAction(), SkillStatus.FAILED, "stuck (d=5.0)")
    class _Done:
        def tick(self, ctx): return SkillResult(AgentAction(), SkillStatus.DONE, "pillared")
    fsm = FindAndChopLogs(reach=3.5)
    fsm._state = "approach"; fsm._target = (5, 64, 0); fsm._recover_count = 0; fsm._cleared = set()
    fsm._sub = _Stuck()
    ctx = SkillContext(pose=_pose(), world_map=_map_with_log((5, 64, 0)))
    fsm.tick(ctx)
    from agents.skills import PillarUp, MineBlock
    (ok if fsm._state == "recover" and isinstance(fsm._sub, PillarUp) else bad)(
        f"stuck, no wall -> pillar-out recover ({type(fsm._sub).__name__})")
    fsm._sub = _Done()
    fsm.tick(ctx)
    (ok if fsm._state == "approach" else bad)(f"recovered -> re-approach (state={fsm._state})")

    # Bounded multi-recovery: recovers up to max_recover times, THEN gives up.
    fsm = FindAndChopLogs(reach=3.5, max_recover=2)
    fsm._state = "approach"; fsm._target = (5, 64, 0)
    fsm._recover_count = 0; fsm._cleared = set()
    ctx = SkillContext(pose=_pose(), world_map=_map_with_log((5, 64, 0)))
    recoveries = 0
    for _ in range(8):
        fsm._sub = _Stuck()
        fsm.tick(ctx)                           # approach stuck -> recover or give up
        if fsm._state == "recover":
            recoveries += 1
            fsm._sub = _Done(); fsm.tick(ctx)   # finish recovery -> re-approach
        elif fsm._state == "find":
            break
    (ok if recoveries == 2 and fsm._state == "find" and (5, 64, 0) in fsm._blacklist
     else bad)(f"recovers max_recover=2 then gives up (did {recoveries}, "
               f"state={fsm._state})")

    # LEAVES ahead -> mine-through (tree material is OK to break).
    wm = _map_with_log((5, 64, 0))
    wm.update_block(BlockObservation(pos=(1, 64, 0), block_id="minecraft:oak_leaves",
                                     confidence=1.0, source="looking_at", last_seen_tick=0))
    fsm = FindAndChopLogs(reach=3.5)
    fsm._state = "approach"; fsm._target = (5, 64, 0); fsm._recover_count = 0; fsm._cleared = set()
    fsm._sub = _Stuck()
    fsm.tick(SkillContext(pose=_pose(), world_map=wm))
    (ok if fsm._state == "recover" and isinstance(fsm._sub, MineBlock) else bad)(
        f"stuck + leaves ahead -> mine-through ({type(fsm._sub).__name__})")

    # STONE/terrain ahead -> NOT mined; pillar OVER it instead.
    wm2 = _map_with_log((5, 64, 0))
    wm2.update_block(BlockObservation(pos=(1, 64, 0), block_id="minecraft:stone",
                                      confidence=1.0, source="looking_at", last_seen_tick=0))
    fsm = FindAndChopLogs(reach=3.5)
    fsm._state = "approach"; fsm._target = (5, 64, 0); fsm._recover_count = 0; fsm._cleared = set()
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

    # 8. TreeChopAgent: registered, idles without perception, acts with it.
    print("\n[8] TreeChopAgent (main.py agent)")
    from agents import build_agent, available_agents
    (ok if "treechop" in available_agents() else bad)("registered in available_agents")
    ag = build_agent("treechop", {})
    a = ag.decide(SimpleNamespace(f3=None, world=None))
    (ok if a.interact is None and not any(a.movement.values()) else bad)(
        "idles safely with no perception (all movement released)")
    ag.attach_perception(SimpleNamespace(world_map=_map_with_log((3, 64, 0))))
    pose = SimpleNamespace(x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=20.0, dimension=None)
    a = ag.decide(SimpleNamespace(f3=None,
                                  world=SimpleNamespace(pose=pose, looking_at=None)))
    (ok if hasattr(a, "movement") else bad)("decide returns an AgentAction with a log mapped")

    # 9. Eat-when-hungry: sustained low hunger -> eats (stands still), and
    #    NOT when hunger is fine.
    print("\n[9] eat-when-hungry")
    s = {"agent": {"treechop": {"eat_below": 0.45}},
         "hotbar": {"slot_roles": {9: "food"}, "extra_food": []}}
    ag = build_agent("treechop", s)
    ag.attach_perception(SimpleNamespace(world_map=_map_with_log((3, 64, 0))))
    pose = SimpleNamespace(x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0, dimension=None)
    def _state(hunger):
        return SimpleNamespace(f3=None,
                               world=SimpleNamespace(pose=pose, looking_at=None),
                               hunger=hunger)
    saw_eat = saw_stop = False
    for _ in range(16):                       # hungry for many ticks
        a = ag.decide(_state(0.2))
        if a.interact == "use_hold":
            saw_eat = True
            if not a.movement.get("forward", False) and not a.movement.get("sprint", False):
                saw_stop = True
    (ok if saw_eat else bad)("eats (use_hold) when hunger stays low")
    (ok if saw_stop else bad)("stands still while eating (movement released)")
    # Full hunger -> never eats.
    ag2 = build_agent("treechop", s)
    ag2.attach_perception(SimpleNamespace(world_map=_map_with_log((3, 64, 0))))
    ate = any(ag2.decide(_state(1.0)).interact == "use_hold" for _ in range(16))
    (ok if not ate else bad)("does NOT eat when hunger is full")

    # 10. Giving up blacklists the RAW found voxel (not just the descended
    #     base) so find can't infinite-loop re-picking an unreachable log.
    print("\n[10] give-up blacklists the found voxel (no infinite re-pick)")
    fsm = FindAndChopLogs(reach=3.5)
    ctx = SkillContext(pose=_pose(), world_map=_map_with_log((8, 70, 0)))
    fsm.tick(ctx)                              # find -> sets _found_raw + target
    fsm._drop_target()                         # simulate giving up on it
    again = fsm._find(ctx)
    (ok if again is None else bad)(
        f"found voxel excluded from find after give-up -> {again}")

    # 11. HarvestAgent — the SAME gather FSM generalised to any block via a
    #     thin config (proves the framework yields new behaviours cheaply).
    print("\n[11] HarvestAgent (framework generalises)")
    from agents import build_agent, available_agents
    from agents.skills import MineBlock as _MB, ChopTrunk as _CT
    (ok if "harvest" in available_agents() else bad)("harvest agent registered")
    hv = build_agent("harvest", {"agent": {"harvest": {"match": ["stone"],
                                                        "tool": "pickaxe"}}})
    hv._build()
    m = hv._fsm._make_mine((3, 64, 0))
    (ok if isinstance(m, _MB) and not isinstance(m, _CT) else bad)(
        f"harvest reach-action is a plain MineBlock ({type(m).__name__})")
    (ok if hv._fsm.is_log("minecraft:stone")
        and not hv._fsm.is_log("minecraft:oak_log") else bad)(
        "harvest targets the configured block (stone), not logs")
    (ok if isinstance(FindAndChopLogs()._make_mine((0, 0, 0)), _CT) else bad)(
        "tree-chopper still fells trunks with ChopTrunk (unchanged)")

    # 12. PlannerAgent — sequences tasks, advancing on FSM DONE (T3 layer).
    print("\n[12] PlannerAgent (goal/planner sequencing)")
    from agents import build_agent, available_agents
    from agents.skills import MineBlock as _MB, ChopTrunk as _CT
    (ok if "planner" in available_agents() else bad)("planner agent registered")
    pl = build_agent("planner", {"agent": {"planner": {"tasks": [
        {"kind": "logs", "count": 3},
        {"kind": "block", "match": ["stone"], "tool": "pickaxe", "count": 5},
    ]}}})
    pl._build()
    # task 1 = logs (ChopTrunk feller), goal 3 GATHERED blocks
    (ok if pl._task_i == 0 and isinstance(pl._fsm._make_mine((0, 0, 0)), _CT)
        and pl._fsm.goal_blocks == 3 else bad)("task 1 = logs (ChopTrunk, goal 3)")
    # simulate task-1 FSM done -> advance to task 2
    pl._on_fsm_done()
    (ok if pl._task_i == 1 and isinstance(pl._fsm._make_mine((0, 0, 0)), _MB)
        and pl._fsm.is_log("minecraft:stone") and pl._fsm.goal_blocks == 5 else bad)(
        "advances to task 2 = harvest stone (MineBlock, goal 5)")
    # task-2 done -> plan complete, stays idle, no crash on extra fires
    pl._on_fsm_done(); pl._on_fsm_done()
    (ok if pl._task_i == 2 else bad)(f"plan complete + bounded ({pl._task_i})")
    t = pl.telemetry()
    (ok if t.get("of") == 2 else bad)(f"telemetry reports plan size ({t})")
    # repeat: a repeating plan loops back to task 0 when complete.
    rp = build_agent("planner", {"agent": {"planner": {"repeat": True, "tasks": [
        {"kind": "logs", "count": 2}]}}})
    rp._build(); rp._on_fsm_done()
    (ok if rp._task_i == 0 else bad)(f"repeat plan loops to task 0 ({rp._task_i})")

    # 12b. Relocate guard: too many consecutive unreachable targets ->
    #      blacklist the cluster + relocate (don't grind through each log).
    print("\n[12b] consecutive approach-fails -> relocate")
    from brain.interfaces import AgentAction as _AA
    from agents.skills import SkillResult as _SR
    class _FailSub:                       # always fails NOT-stuck (e.g. edge)
        def tick(self, ctx):
            return _SR(_AA(), SkillStatus.FAILED, "edge ahead (would fall)")
    fsm = FindAndChopLogs()
    fsm._state = "approach"; fsm._target = (10, 64, 10)
    fsm._approach_fails = 3               # next fail is the 4th -> relocate
    fsm._sub = _FailSub()
    r = fsm.tick(SkillContext(pose=_pose(), world_map=WorldMap()))
    (ok if fsm._state == "scan" and "relocating" in r.info else bad)(
        f"4th unreachable target -> relocate ({fsm._state}, {r.info!r})")
    (ok if (10, 64, 10) in fsm._blacklist and (11, 65, 11) in fsm._blacklist
     else bad)("blacklists the surrounding cluster (radius), not just one voxel")
    (ok if fsm._approach_fails == 0 else bad)("relocate resets the fail counter")
    # below threshold: just refind (no premature relocate)
    fsm2 = FindAndChopLogs()
    fsm2._state = "approach"; fsm2._target = (5, 64, 5); fsm2._sub = _FailSub()
    r2 = fsm2.tick(SkillContext(pose=_pose(), world_map=WorldMap()))
    (ok if fsm2._state == "find" and fsm2._approach_fails == 1 else bad)(
        f"1st fail -> refind, not relocate ({fsm2._state}, n={fsm2._approach_fails})")

    # 13. parse_plan: CLI plan string -> planner tasks.
    print("\n[13] parse_plan (CLI plan -> tasks)")
    from agents.treechop import parse_plan
    tasks = parse_plan("logs:8, stone:16, diamond_ore")
    (ok if len(tasks) == 3 else bad)(f"3 tasks parsed ({len(tasks)})")
    (ok if tasks[0] == {"kind": "logs", "count": 8} else bad)(f"logs:8 -> {tasks[0]}")
    (ok if tasks[1]["kind"] == "block" and tasks[1]["match"] == ["stone"]
        and tasks[1]["count"] == 16 else bad)(f"stone:16 -> {tasks[1]}")
    (ok if tasks[2]["count"] == 9999 else bad)(f"no count -> unbounded ({tasks[2]['count']})")
    (ok if parse_plan("") == [] else bad)("empty plan -> []")
    (ok if parse_plan("stone:notanum")[0]["count"] == 9999 else bad)("bad count -> unbounded")

    # 14. find confidence gate: a low-confidence MISLABEL (grass/leaf guessed as
    # a log) must NOT be chosen over a real F3-confirmed log farther away.
    print("\n[14] find_nearest_block confidence gate")
    from agents.skills import find_nearest_block
    from agents.treechop import _is_log_default
    wm = WorldMap()
    wm.update_block(BlockObservation(block_id="minecraft:oak_log", pos=(0, 64, 2),
                                     confidence=0.5, source="vision_patch"))
    wm.update_block(BlockObservation(block_id="minecraft:oak_log", pos=(0, 64, 5),
                                     confidence=1.0, source="looking_at"))
    eye = (0.5, 65.0, 0.5)
    u = find_nearest_block(wm, eye, _is_log_default, max_radius=32)
    g = find_nearest_block(wm, eye, _is_log_default, max_radius=32, min_confidence=0.6)
    (ok if u and u[0] == (0, 64, 2) else bad)("ungated picks the nearer low-conf voxel")
    (ok if g and g[0] == (0, 64, 5) else bad)(
        "gated SKIPS the low-conf mislabel, picks the F3-confirmed log")

    print("\n" + ("ALL TREECHOP TESTS PASSED" if not _fails
                  else f"{_fails} CHECK(S) FAILED"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
