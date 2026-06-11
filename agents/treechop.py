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
    MineBlock, find_nearest_block,
)
from vision.world.map import AIR_BLOCK


def _is_log_default(bid: str) -> bool:
    return bool(bid) and (str(bid).endswith("_log") or str(bid).endswith("_stem"))


class FindAndChopLogs:
    name = "find_and_chop_logs"

    def __init__(self, is_log=None, reach: float = 3.5, max_radius: int = 32,
                 scan_budget: int = 60, max_logs: int = 9999,
                 tool_role: Optional[str] = "axe",
                 explore_dist: float = 8.0, max_explore: int = 10,
                 explore_turn: float = 65.0, is_breakable=None):
        self.is_log = is_log or _is_log_default
        # The ONLY blocks tree-chopping may ever break: logs + leaves. The
        # path-clearing recovery (mine-through) is restricted to these, so
        # the bot never mines terrain or a player's build to reach a tree.
        self.is_breakable = is_breakable or (
            lambda b: self.is_log(b) or (bool(b) and str(b).endswith("_leaves")))
        self.reach = reach
        self.max_radius = max_radius
        self.scan_budget = scan_budget
        self.max_logs = max_logs
        self.tool_role = tool_role
        # Exploration: when no log is mapped, walk to a new area and re-scan
        # (fanning the heading each attempt) instead of giving up.
        self.explore_dist = explore_dist
        self.max_explore = max_explore
        self.explore_turn = explore_turn
        self.reset()

    def reset(self):
        self._state = "find"
        self._target = None
        self._sub = None
        self._scan_ticks = 0
        self._blacklist = set()
        self._recovered = False     # pillar-out attempted for this target?
        self._explore_attempts = 0
        self.chopped = 0

    def _eye_vox(self, pose):
        return (int(math.floor(pose.x)), int(math.floor(pose.y)),
                int(math.floor(pose.z)))

    def _obstacle_ahead(self, ctx):
        """The voxel of a KNOWN, BREAKABLE (leaf/log) block directly ahead
        (foot or head level, toward the target) that's blocking the path,
        or None. Only leaves/logs qualify — the bot mines THROUGH tree
        material to reach a trunk, but never through terrain or builds (it
        pillars over those instead)."""
        p, wm = ctx.pose, ctx.world_map
        if wm is None or self._target is None:
            return None
        dx = self._target[0] + 0.5 - p.x
        dz = self._target[2] + 0.5 - p.z
        sx = (1 if dx > 0 else -1) if abs(dx) >= abs(dz) else 0
        sz = 0 if sx != 0 else (1 if dz > 0 else -1)
        foot = int(math.floor(p.y))
        bx, bz = int(math.floor(p.x)) + sx, int(math.floor(p.z)) + sz
        for dyy in (0, 1):                       # foot + head height
            v = (bx, foot + dyy, bz)
            try:
                obs = wm.get_block(v, dimension=ctx.dimension)
            except TypeError:
                obs = wm.get_block(v)
            bid = getattr(obs, "block_id", None)
            if bid not in (AIR_BLOCK, None) and self.is_breakable(bid):
                return v
        return None

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
            self._explore_attempts = 0      # found one -> refresh explore budget
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
                # Nothing nearby -> walk to a new area and re-scan.
                if self._explore_attempts < self.max_explore:
                    self._explore_attempts += 1
                    h = math.radians(float(pose.yaw) + self._explore_attempts * self.explore_turn)
                    ex = (int(math.floor(pose.x + self.explore_dist * (-math.sin(h)))),
                          int(math.floor(pose.y)),
                          int(math.floor(pose.z + self.explore_dist * math.cos(h))))
                    self._sub = WalkToward(ex, arrive_dist=1.5)
                    self._state = "explore"
                    return SkillResult(AgentAction(), SkillStatus.RUNNING,
                                       f"explore {self._explore_attempts}/{self.max_explore} -> {ex}")
                return SkillResult(AgentAction(), SkillStatus.DONE,
                                   f"no logs found after exploring; chopped {self.chopped}")
            # Clean look-around: FIRST level the pitch to ~0 without
            # touching yaw, THEN rotate yaw only (pitch untouched) — no
            # up/down bobbing. Level scanning also sweeps the horizon where
            # tree trunks are, which is exactly what we're looking for.
            pitch = float(pose.pitch)
            if abs(pitch) > 5.0:
                dy = int(max(-120, min(120, (0.0 - pitch) * ctx.px_per_deg * 0.5)))
                return SkillResult(AgentAction(look_dy=dy), SkillStatus.RUNNING,
                                   f"leveling pitch ({pitch:.0f}deg)")
            return SkillResult(AgentAction(look_dx=70), SkillStatus.RUNNING,
                               f"scanning {self._scan_ticks}")

        if st == "explore":
            tgt = self._find(ctx)               # a log may appear as we walk
            if tgt is not None:
                self._state = "find"
                return SkillResult(AgentAction(), SkillStatus.RUNNING, "log spotted while exploring")
            r = self._sub.tick(ctx)
            if r.status in (SkillStatus.DONE, SkillStatus.FAILED, SkillStatus.BLOCKED):
                self._state = "scan"; self._scan_ticks = 0
                return SkillResult(r.action, SkillStatus.RUNNING,
                                   f"explored ({r.info}); scanning")
            return r

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
                    self._state = "recover"
                    obstacle = self._obstacle_ahead(ctx)
                    if obstacle is not None:
                        # A wall is blocking the path -> mine through it.
                        self._sub = MineBlock(obstacle, tool_role=None)
                        return SkillResult(r.action, SkillStatus.RUNNING,
                                           f"stuck; mining through {obstacle}")
                    # Otherwise assume a hole -> pillar out.
                    self._sub = PillarUp(height=2)
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


from types import SimpleNamespace
from brain.interfaces import BaseAgent


class TreeChopAgent(BaseAgent):
    """main.py agent wrapping FindAndChopLogs — runs the autonomous
    tree-chopper inside the normal agent loop (safety gate, panic-stop,
    attack-hold dispatch). Requires ``vision.world.enabled`` so it has a
    WorldMap + F3 targeted-block to work from.

    Run: ``python main.py --agent treechop``  (set ``hotbar.slot_roles`` so
    the axe/blocks slots are right; the bot only ever breaks logs+leaves)."""

    name = "treechop"

    def __init__(self, settings: dict):
        self._settings = settings or {}
        cfg = ((self._settings.get("agent", {}) or {}).get("treechop", {}) or {})
        self._max_logs = int(cfg.get("max_logs", 9999))
        self._px_per_deg = float(((self._settings.get("agent", {}) or {})
                                  .get("mouse_per_degree", 6.5)) or 6.5)
        self._wp = None
        self._hotbar = None
        self._fsm = None
        self._warned_no_world = False

    def attach_perception(self, wp) -> None:
        self._wp = wp

    def reset(self) -> None:
        self._fsm = None

    def _build(self) -> None:
        from knowledge.catalog import Catalog
        from vision.mc_assets import MCAssets
        from control.hotbar import build_hotbar_manager
        cat = Catalog.load(MCAssets.load())
        log_ids = {b.id for b in cat.blocks_in_tag("logs")}
        leaf_ids = {b.id for b in cat.blocks_in_tag("leaves")}
        is_log = lambda b: bool(b) and (b in log_ids or str(b).endswith("_log"))
        is_breakable = lambda b: bool(b) and (
            b in log_ids or b in leaf_ids
            or str(b).endswith("_log") or str(b).endswith("_leaves"))
        self._hotbar = build_hotbar_manager(self._settings, catalog=cat)
        self._fsm = FindAndChopLogs(is_log=is_log, is_breakable=is_breakable,
                                    tool_role="axe", max_logs=self._max_logs)

    @staticmethod
    def _pose_from_f3(f3):
        if f3 is None or f3.x is None or f3.yaw is None or f3.pitch is None:
            return None
        return SimpleNamespace(x=float(f3.x), y=float(f3.y), z=float(f3.z),
                               yaw=float(f3.yaw), pitch=float(f3.pitch),
                               dimension=getattr(f3, "dimension", None))

    def decide(self, state):
        from brain.interfaces import AgentAction as _AA
        if self._wp is None:
            if not self._warned_no_world:
                self._warned_no_world = True
                print("[treechop] needs vision.world.enabled: true — no "
                      "WorldMap to find logs. Idling.")
            return _AA()
        if self._fsm is None:
            self._build()
        world = getattr(state, "world", None)
        pose = (world.pose if world is not None and world.pose is not None
                else self._pose_from_f3(getattr(state, "f3", None)))
        if pose is None:
            return _AA()                       # wait for a pose this tick
        looking_at = world.looking_at if world is not None else None
        ctx = SkillContext(pose=pose, world_map=self._wp.world_map,
                           looking_at=looking_at, hotbar=self._hotbar,
                           px_per_deg=self._px_per_deg,
                           dimension=getattr(pose, "dimension", None))
        return self._fsm.tick(ctx).action


def build_treechop_agent(settings: dict) -> TreeChopAgent:
    return TreeChopAgent(settings)


__all__ = ["FindAndChopLogs", "TreeChopAgent", "build_treechop_agent"]
