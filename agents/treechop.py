# agents/treechop.py
"""
FindAndChopLogs — the headline behaviour: find a log, walk to it, chop the
trunk, collect the drops, scan for the next one, repeat.

It's a tick-driven FSM that composes the live-verified skill primitives
(find_nearest_block + WalkToward + ChopTrunk) — so it stays non-blocking,
reuses pieces that are individually tested, and could later have any state
swapped for a learned policy. States:

    find    -> pick the nearest mapped log (not already chopped)
    scan    -> no log mapped: rotate the view so the recogniser maps more;
               give up after a couple of rotations
    approach-> WalkToward the log until within mining reach
    chop    -> ChopTrunk the column
    collect -> walk onto the trunk base to pick up the dropped logs
    (repeat) -> blacklist that column, back to find

Locomotion uses the reactive WalkToward (stuck/edge-safe). Heavier terrain
recovery (jump-over / pillar-out / mine-through) is the planner layer that
slots in where approach currently just FAILS-and-refinds.
"""

from __future__ import annotations

import math
from typing import Optional

from brain.interfaces import AgentAction
from agents.skills import (
    SkillStatus, SkillResult, SkillContext, WalkToward, ChopTrunk, PillarUp,
    find_nearest_block,
)


def _is_log_default(bid: str) -> bool:
    return bool(bid) and (str(bid).endswith("_log") or str(bid).endswith("_stem"))


class FindAndChopLogs:
    name = "find_and_chop_logs"

    def __init__(self, is_log=None, reach: float = 3.5, max_radius: int = 32,
                 scan_budget: int = 60, max_logs: int = 9999,
                 tool_role: Optional[str] = "axe"):
        self.is_log = is_log or _is_log_default
        self.reach = reach
        self.max_radius = max_radius
        self.scan_budget = scan_budget
        self.max_logs = max_logs
        self.tool_role = tool_role
        self.reset()

    def reset(self):
        self._state = "find"
        self._target = None
        self._sub = None
        self._scan_ticks = 0
        self._blacklist = set()
        self._recovered = False     # pillar-out attempted for this target?
        self.chopped = 0

    def _eye_vox(self, pose):
        return (int(math.floor(pose.x)), int(math.floor(pose.y)),
                int(math.floor(pose.z)))

    def _find(self, ctx):
        eye = self._eye_vox(ctx.pose)
        res = find_nearest_block(ctx.world_map, eye, self.is_log,
                                 max_radius=self.max_radius,
                                 dimension=ctx.dimension, exclude=self._blacklist)
        return res[0] if res else None

    def tick(self, ctx: SkillContext) -> SkillResult:
        pose = ctx.pose
        if pose is None:
            return SkillResult(AgentAction(), SkillStatus.BLOCKED, "no pose")
        if self.chopped >= self.max_logs:
            return SkillResult(AgentAction(), SkillStatus.DONE,
                               f"chopped {self.chopped} (limit)")

        st = self._state
        if st == "find":
            tgt = self._find(ctx)
            if tgt is None:
                self._state = "scan"; self._scan_ticks = 0
                return SkillResult(AgentAction(), SkillStatus.RUNNING, "no log mapped; scanning")
            self._target = tgt
            self._recovered = False
            ex, ez = pose.x, pose.z
            horiz = math.hypot(tgt[0] + 0.5 - ex, tgt[2] + 0.5 - ez)
            if horiz <= self.reach:
                self._sub = ChopTrunk(tgt, is_log=self.is_log, tool_role=self.tool_role)
                self._state = "chop"
                return SkillResult(AgentAction(), SkillStatus.RUNNING, f"log {tgt} in reach; chopping")
            self._sub = WalkToward(tgt, arrive_dist=self.reach)
            self._state = "approach"
            return SkillResult(AgentAction(), SkillStatus.RUNNING, f"log {tgt} at {horiz:.1f}; approaching")

        if st == "scan":
            self._scan_ticks += 1
            tgt = self._find(ctx)
            if tgt is not None:
                self._state = "find"
                return SkillResult(AgentAction(), SkillStatus.RUNNING, "log appeared; refind")
            if self._scan_ticks > self.scan_budget:
                return SkillResult(AgentAction(), SkillStatus.DONE,
                                   f"no logs found (scanned); chopped {self.chopped}")
            # Rotate the view to map more (oscillate pitch to catch trunks + ground).
            dy = int(18 * math.sin(self._scan_ticks * 0.5))
            return SkillResult(AgentAction(look_dx=70, look_dy=dy),
                               SkillStatus.RUNNING, f"scanning {self._scan_ticks}")

        if st == "approach":
            r = self._sub.tick(ctx)
            if r.status == SkillStatus.DONE:
                self._sub = ChopTrunk(self._target, is_log=self.is_log, tool_role=self.tool_role)
                self._state = "chop"
                return SkillResult(r.action, SkillStatus.RUNNING, "arrived; chopping")
            if r.status in (SkillStatus.FAILED, SkillStatus.BLOCKED):
                # Stuck (e.g. fell in a hole)? Try pillaring out once, then
                # re-approach from the new height. Other failures (or a 2nd
                # stuck) -> give up on this log.
                if "stuck" in r.info and not self._recovered:
                    self._recovered = True
                    self._sub = PillarUp(height=2)
                    self._state = "recover"
                    return SkillResult(r.action, SkillStatus.RUNNING,
                                       "stuck; pillaring out of the hole")
                self._blacklist.add(self._target); self._state = "find"
                return SkillResult(r.action, SkillStatus.RUNNING,
                                   f"approach {r.status.value}; refind")
            return r

        if st == "recover":
            r = self._sub.tick(ctx)
            if r.status in (SkillStatus.DONE, SkillStatus.FAILED, SkillStatus.BLOCKED):
                # Climbed out (or couldn't) -> re-approach the log once more.
                self._sub = WalkToward(self._target, arrive_dist=self.reach)
                self._state = "approach"
                return SkillResult(r.action, SkillStatus.RUNNING,
                                   f"recovered ({r.info}); re-approaching")
            return r

        if st == "chop":
            r = self._sub.tick(ctx)
            if r.status == SkillStatus.DONE:
                self.chopped += 1
                # Walk onto the base to collect the dropped logs, then refind.
                base = self._target
                self._blacklist.add(base)
                self._sub = WalkToward(base, arrive_dist=0.7, stuck_window=12)
                self._state = "collect"
                return SkillResult(r.action, SkillStatus.RUNNING, f"chopped #{self.chopped}; collecting")
            if r.status in (SkillStatus.FAILED, SkillStatus.BLOCKED):
                self._blacklist.add(self._target); self._state = "find"
                return SkillResult(r.action, SkillStatus.RUNNING, f"chop {r.status.value}; refind")
            return r

        if st == "collect":
            r = self._sub.tick(ctx)
            if r.status in (SkillStatus.DONE, SkillStatus.FAILED, SkillStatus.BLOCKED):
                self._target = None; self._state = "find"
                return SkillResult(r.action, SkillStatus.RUNNING, "collected; refind")
            return r

        return SkillResult(AgentAction(), SkillStatus.FAILED, f"bad state {st}")


__all__ = ["FindAndChopLogs"]
