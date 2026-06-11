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
    SkillStatus, SkillResult, SkillContext, WalkToward, NavigateTo, ChopTrunk,
    PillarUp, MineBlock, Eat, find_nearest_block, block_in_reach, PLAYER_REACH,
    _Aimer,
)
from vision.world.map import AIR_BLOCK


# Every wood/log variant across all species — bark logs (_log), 6-sided
# wood (_wood), nether stems (_stem) and hyphae (_hyphae), plus their
# stripped forms (which still end in those). Species-agnostic on purpose:
# birch_log, oak_log, spruce_log, … all match identically.
_LOG_SUFFIXES = ("_log", "_wood", "_stem", "_hyphae")


def _is_log_default(bid: str) -> bool:
    return bool(bid) and str(bid).endswith(_LOG_SUFFIXES)


_MOVE_KEYS = ("forward", "backward", "left", "right", "jump", "sprint", "sneak")


def _full_movement(mv):
    """Expand a skill's partial movement dict into a COMPLETE one (every
    key present, unset -> False). Movement keys are held state; if a tick
    omits a key, the dispatcher leaves it as-was — so an earlier jump/
    forward would stay stuck on. Fully specifying every tick guarantees
    anything not actively commanded is released (no jumping-in-place, no
    walking-while-mining). Every skill re-issues its movement each tick, so
    nothing relies on persistence."""
    full = {k: False for k in _MOVE_KEYS}
    if mv:
        for k, v in mv.items():
            if k in full:
                full[k] = bool(v)
    return full


class FindAndChopLogs:
    name = "find_and_chop_logs"

    def __init__(self, is_log=None, reach: float = 3.5, max_radius: int = 32,
                 scan_budget: int = 60, max_logs: int = 9999,
                 tool_role: Optional[str] = "axe",
                 explore_dist: float = 8.0, max_explore: int = 10,
                 explore_turn: float = 65.0, is_breakable=None,
                 max_recover: int = 5, mine_action=None, goal_blocks=None):
        # mine_action(voxel) -> Skill: what to do once IN REACH of a target.
        # Default fells the trunk (ChopTrunk); a flat block-gatherer passes a
        # plain MineBlock factory. This is the one knob that turns the
        # tree-chopper into a generic "find -> go -> mine -> collect" gatherer.
        self._mine_action = mine_action
        # goal_blocks: complete (DONE) once this many blocks have actually
        # been BROKEN (F3-confirmed -> ~= gathered, since auto-pickup happens
        # while standing there). This is the closed-loop goal metric — "get 8
        # logs" = 8 logs broken/collected — vs max_logs which caps trunks.
        self.goal_blocks = goal_blocks
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
        self.max_recover = max_recover
        self._scan_aimer = _Aimer(tol_deg=4.0)   # stale-pose-guarded pitch leveler
        self.reset()

    def reset(self):
        self._state = "find"
        self._target = None
        self._sub = None
        self._scan_ticks = 0
        self._blacklist = set()
        self._recover_count = 0         # recovery actions used for this target
        self._found_raw = None          # the voxel find returned (pre-descend)
        self._cleared = set()           # obstacles already mined (avoid re-mining)
        self._explore_attempts = 0
        self._explore_anchor_yaw = None # fixed origin to fan explore headings
        self._approach_ticks = 0        # cap time spent navigating to one log
        self._approach_fails = 0        # consecutive unreachable targets -> relocate
        self.chopped = 0                # trunk-columns chopped (>=1 log each)
        self.logs = 0                   # actual log blocks broken

    def _eye_vox(self, pose):
        return (int(math.floor(pose.x)), int(math.floor(pose.y)),
                int(math.floor(pose.z)))

    def _make_mine(self, voxel):
        """The reach-action for a target: configured mine_action, else the
        default trunk-feller."""
        if self._mine_action is not None:
            return self._mine_action(voxel)
        return ChopTrunk(voxel, is_log=self.is_log, tool_role=self.tool_role,
                         is_safe=self.is_breakable)

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
            if v in self._cleared:           # already mined (map may lag) -> skip
                continue
            if bid not in (AIR_BLOCK, None) and self.is_breakable(bid):
                return v
        return None

    def _find(self, ctx):
        eye = self._eye_vox(ctx.pose)
        py = float(ctx.pose.y)
        # Only consider logs within a REACHABLE height band of where we are
        # (feet to ~head+reach). Canopy logs 8-10 blocks overhead can't be
        # reached from the ground, so don't target them — the level scan maps
        # trunk-height logs to chop instead.
        def pos_ok(v):
            return -4.0 <= (v[1] - py) <= 5.0
        res = find_nearest_block(ctx.world_map, eye, self.is_log,
                                 max_radius=self.max_radius,
                                 dimension=ctx.dimension, exclude=self._blacklist,
                                 pos_ok=pos_ok)
        return res[0] if res else None

    def _drop_target(self):
        """Blacklist BOTH the raw found voxel and the descended base, so
        find() can't immediately re-pick the same (unreachable) log."""
        for v in (self._target, self._found_raw):
            if v is not None:
                self._blacklist.add(v)

    def _descend_to_base(self, voxel, ctx):
        """Lower the target to the bottom of its mapped log column so we
        chop the WHOLE trunk from the base up (not a stray canopy block).
        Stops at the lowest contiguous log the WorldMap knows about."""
        wm = ctx.world_map
        if wm is None:
            return voxel
        x, y, z = voxel
        for _ in range(20):
            below = (x, y - 1, z)
            try:
                obs = wm.get_block(below, dimension=ctx.dimension)
            except TypeError:
                obs = wm.get_block(below)
            if obs is not None and self.is_log(getattr(obs, "block_id", None)):
                y -= 1
            else:
                break
        return (x, y, z)

    def tick(self, ctx: SkillContext) -> SkillResult:
        pose = ctx.pose
        if pose is None:
            return SkillResult(AgentAction(), SkillStatus.BLOCKED, "no pose")
        if self.goal_blocks is not None and self.logs >= self.goal_blocks:
            return SkillResult(AgentAction(), SkillStatus.DONE,
                               f"goal reached: {self.logs} blocks gathered")
        if self.chopped >= self.max_logs:
            return SkillResult(AgentAction(), SkillStatus.DONE,
                               f"chopped {self.chopped} (limit)")

        # OPPORTUNISTIC CHOP: the crosshair is the authority. If F3 says we're
        # looking directly at a log that's in reach, chop it NOW — never hover
        # over a reachable log while busy navigating/aiming/scanning. (Only
        # pre-empts non-chop states; ChopTrunk owns the chop state once in it.)
        if self._state != "chop":
            la = ctx.looking_at
            la_id = getattr(la, "block_id", None) if la is not None else None
            la_pos = (tuple(la.pos) if la is not None
                      and getattr(la, "pos", None) is not None else None)
            if la_id and self.is_log(la_id) and la_pos is not None \
                    and la_pos not in self._blacklist \
                    and block_in_reach(pose, la_pos, PLAYER_REACH):
                self._target = la_pos
                self._found_raw = la_pos
                self._recover_count = 0
                self._cleared = set()
                self._sub = self._make_mine(la_pos)
                self._state = "chop"

        st = self._state
        if st == "find":
            tgt = self._find(ctx)
            if tgt is None:
                self._state = "scan"; self._scan_ticks = 0
                return SkillResult(AgentAction(), SkillStatus.RUNNING, "no log mapped; scanning")
            self._found_raw = tgt
            # Mine the log we actually found (ChopTrunk then works UP the
            # trunk). We do NOT descend to a hidden base — that made the bot
            # look down / walk past the log it could already see.
            self._target = tgt
            self._recover_count = 0
            self._cleared = set()
            self._explore_attempts = 0      # found one -> refresh explore budget
            self._explore_anchor_yaw = None
            # 3D reach (eye -> block): a too-high canopy log is NOT "in reach"
            # just because it's horizontally close, so we don't get stuck
            # clicking at something we can't touch.
            if block_in_reach(pose, tgt, PLAYER_REACH):
                self._sub = self._make_mine(tgt)
                self._state = "chop"
                return SkillResult(AgentAction(), SkillStatus.RUNNING, f"log {tgt} in reach; chopping")
            # A* route AROUND known gaps/obstacles to within reach of the log,
            # following the path with the reactive WalkToward.
            self._sub = NavigateTo(tgt, arrive_reach=PLAYER_REACH)
            self._state = "approach"; self._approach_ticks = 0
            return SkillResult(AgentAction(), SkillStatus.RUNNING, f"log {tgt}; navigating")

        if st == "scan":
            self._scan_ticks += 1
            tgt = self._find(ctx)
            if tgt is not None:
                self._state = "find"
                return SkillResult(AgentAction(), SkillStatus.RUNNING, "log appeared; refind")
            if self._scan_ticks > self.scan_budget:
                # Nothing nearby -> walk to a new area and re-scan.
                if self._explore_attempts < self.max_explore:
                    if self._explore_anchor_yaw is None:
                        self._explore_anchor_yaw = float(pose.yaw)
                    self._explore_attempts += 1
                    h = math.radians(self._explore_anchor_yaw
                                     + self._explore_attempts * self.explore_turn)
                    ex = (int(math.floor(pose.x + self.explore_dist * (-math.sin(h)))),
                          int(math.floor(pose.y)),
                          int(math.floor(pose.z + self.explore_dist * math.cos(h))))
                    self._sub = WalkToward(ex, arrive_dist=1.5)
                    self._state = "explore"
                    return SkillResult(AgentAction(), SkillStatus.RUNNING,
                                       f"explore {self._explore_attempts}/{self.max_explore} -> {ex}")
                return SkillResult(AgentAction(), SkillStatus.DONE,
                                   f"no logs found after exploring; chopped {self.chopped}")
            # Clean look-around: FIRST level the pitch to ~0 (yaw untouched),
            # THEN rotate yaw only. The leveling goes through the SAME
            # stale-pose-guarded aimer the mining uses, so it issues one
            # correction per pose update instead of over-applying every tick
            # and oscillating up/down.
            _, dy, level_ok = self._scan_aimer.step(ctx, float(pose.yaw), 0.0)
            if not level_ok:
                return SkillResult(AgentAction(look_dy=dy), SkillStatus.RUNNING,
                                   f"leveling pitch ({float(pose.pitch):.0f}deg)")
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
            # Time-cap: don't grind forever trying to reach one log (e.g. an
            # out-of-reach/across-a-gap target the router can't close on).
            self._approach_ticks += 1
            if self._approach_ticks > 160:
                self._drop_target(); self._state = "find"
                return SkillResult(AgentAction(movement={"forward": False}),
                                   SkillStatus.RUNNING, "approach timed out; refind")
            r = self._sub.tick(ctx)
            if r.status == SkillStatus.DONE:
                # Arrived. Only chop if the log is now actually in 3D reach;
                # if it's still too high/far (walking couldn't help), skip it.
                if not block_in_reach(pose, self._target, PLAYER_REACH):
                    self._drop_target(); self._state = "find"
                    return SkillResult(r.action, SkillStatus.RUNNING,
                                       "arrived but log still out of reach; refind")
                self._sub = self._make_mine(self._target)
                self._state = "chop"
                return SkillResult(r.action, SkillStatus.RUNNING, "arrived; chopping")
            if r.status in (SkillStatus.FAILED, SkillStatus.BLOCKED):
                # Stuck? Recover up to max_recover times for this log — each
                # time, mine THROUGH a leaf/log wall ahead (so a multi-block
                # wall clears across several recoveries) or pillar OUT of a
                # hole. Anything else, or budget exhausted -> give up on it.
                if "stuck" in r.info and self._recover_count < self.max_recover:
                    self._recover_count += 1
                    self._state = "recover"
                    obstacle = self._obstacle_ahead(ctx)
                    if obstacle is not None:
                        self._cleared.add(obstacle)        # don't re-mine (map lags)
                        self._sub = MineBlock(obstacle, tool_role=None,
                                              is_safe=self.is_breakable)
                        return SkillResult(r.action, SkillStatus.RUNNING,
                                           f"stuck; mining through {obstacle} "
                                           f"({self._recover_count}/{self.max_recover})")
                    self._sub = PillarUp(height=2)
                    return SkillResult(r.action, SkillStatus.RUNNING,
                                       f"stuck; pillaring out "
                                       f"({self._recover_count}/{self.max_recover})")
                failed = self._target
                self._approach_fails += 1
                self._drop_target()
                if self._approach_fails >= 4 and failed is not None:
                    # Repeatedly can't reach targets here (classic case: a whole
                    # cluster across an edge/gap). Blacklist the surrounding
                    # cluster so find stops re-picking it one log at a time, and
                    # relocate (explore) instead of grinding through them all.
                    fx, fy, fz = failed
                    for ddx in range(-2, 3):
                        for ddz in range(-2, 3):
                            for ddy in range(-3, 6):
                                self._blacklist.add((fx + ddx, fy + ddy, fz + ddz))
                    self._approach_fails = 0
                    self._scan_ticks = self.scan_budget + 1   # -> explore next scan tick
                    self._state = "scan"
                    return SkillResult(r.action, SkillStatus.RUNNING,
                                       f"unreachable cluster near {failed}; relocating")
                self._state = "find"
                return SkillResult(r.action, SkillStatus.RUNNING,
                                   f"approach failed [{r.info}]; refind")
            return r

        if st == "recover":
            r = self._sub.tick(ctx)
            if r.status in (SkillStatus.DONE, SkillStatus.FAILED, SkillStatus.BLOCKED):
                # Cleared a block / climbed (or couldn't) -> re-approach. A
                # remaining wall block or hole re-triggers recover (bounded
                # by max_recover), so multi-block obstacles clear over a few
                # passes without an infinite loop.
                self._sub = NavigateTo(self._target, arrive_reach=PLAYER_REACH)
                self._state = "approach"; self._approach_ticks = 0
                return SkillResult(r.action, SkillStatus.RUNNING,
                                   f"recovered ({r.info}); re-approaching")
            return r

        if st == "chop":
            r = self._sub.tick(ctx)
            if r.status == SkillStatus.DONE:
                # Honest counting: tally the actual logs broken; only count a
                # "trunk" when at least one log fell (a no-op ChopTrunk that
                # found the target already gone shouldn't inflate the total).
                mined = int(getattr(self._sub, "_mined", 0))
                self.logs += mined
                if mined >= 1:
                    self.chopped += 1
                    self._approach_fails = 0     # progress -> reset relocate guard
                # Walk onto the base to collect the dropped logs, then refind.
                # No jumping here (we're not climbing anything) and a roomier
                # arrive radius so it doesn't flail trying to stand on the
                # exact column.
                base = self._target
                self._drop_target()
                # Blacklist the WHOLE column we just chopped (base..base+mined)
                # so the opportunistic check doesn't re-target now-broken upper
                # voxels the map still shows as logs (lag) and thrash.
                if base is not None:
                    for dy in range(0, mined + 2):
                        self._blacklist.add((base[0], base[1] + dy, base[2]))
                self._sub = WalkToward(base, arrive_dist=1.4, stuck_window=8,
                                       jump_after=10 ** 9)
                self._state = "collect"
                return SkillResult(r.action, SkillStatus.RUNNING,
                                   f"chopped {mined} log(s); collecting")
            if r.status in (SkillStatus.FAILED, SkillStatus.BLOCKED):
                self._drop_target(); self._state = "find"
                return SkillResult(r.action, SkillStatus.RUNNING,
                                   f"chop failed [{r.info}]; refind")    # surface WHY
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
    cfg_key = "treechop"            # settings.agent.<cfg_key> section

    def __init__(self, settings: dict):
        self._settings = settings or {}
        cfg = ((self._settings.get("agent", {}) or {}).get(self.cfg_key, {}) or {})
        self._max_logs = int(cfg.get("max_logs", cfg.get("max_blocks", 9999)))
        # Eat when hunger drops below this (0-1; 0.45 ≈ 9/20 food — above
        # the 6/20 sprint cutoff so the bot keeps sprinting). 0 disables.
        self._eat_below = float(cfg.get("eat_below", 0.45))
        self._px_per_deg = float(((self._settings.get("agent", {}) or {})
                                  .get("mouse_per_degree", 6.5)) or 6.5)
        self._wp = None
        self._hotbar = None
        self._fsm = None
        self._warned_no_world = False
        self._last_state = None
        self._eat = None          # active Eat skill (eating in progress)
        self._hungry_ticks = 0    # debounce HUD misreads
        self._eat_fails = 0       # consecutive ineffective eats (cap the loop)
        self._build_failed = False

    def attach_perception(self, wp) -> None:
        self._wp = wp

    def telemetry(self) -> dict:
        """Labels each logged episode tick with the behaviour's state +
        progress (for analytics + future learning)."""
        if self._fsm is None:
            return {"state": "init"}
        t = {"state": self._fsm._state, "logs": self._fsm.logs,
             "trunks": self._fsm.chopped}
        if self._eat is not None:
            t["eating"] = True
        if getattr(self._fsm, "_target", None) is not None:
            t["target"] = list(self._fsm._target)
        return t

    def reset(self) -> None:
        self._fsm = None

    def _build(self) -> None:
        from knowledge.catalog import Catalog
        from vision.mc_assets import MCAssets
        from control.hotbar import build_hotbar_manager
        cat = Catalog.load(MCAssets.load())
        log_ids = {b.id for b in cat.blocks_in_tag("logs")}
        leaf_ids = {b.id for b in cat.blocks_in_tag("leaves")}
        # Species-agnostic: any log/wood variant (tag OR suffix) — so birch,
        # oak, spruce, … are all treated identically.
        is_log = lambda b: bool(b) and (b in log_ids or str(b).endswith(_LOG_SUFFIXES))
        is_breakable = lambda b: bool(b) and (
            is_log(b) or b in leaf_ids or str(b).endswith("_leaves"))
        self._hotbar = build_hotbar_manager(self._settings, catalog=cat)
        self._fsm = FindAndChopLogs(is_log=is_log, is_breakable=is_breakable,
                                    tool_role="axe", max_logs=self._max_logs)

    def _maybe_eat(self, ctx, state):
        """Returns an AgentAction if the bot should be eating this tick
        (standing still, holding use), else None to let the FSM run."""
        from brain.interfaces import AgentAction as _AA
        STOP = {"forward": False, "backward": False, "left": False,
                "right": False, "jump": False, "sprint": False}
        hunger = getattr(state, "hunger", None)
        h = float(hunger) if hunger is not None else None
        if self._eat is not None:                       # mid-eat
            r = self._eat.tick(ctx)
            if r.status in (SkillStatus.DONE, SkillStatus.FAILED, SkillStatus.BLOCKED):
                self._eat = None
                self._hungry_ticks = 0
                # If eating didn't restore hunger, count it — after a few
                # ineffective eats (no food, or a stuck-low HUD read) stop
                # trying so we don't loop forever standing still eating.
                if h is not None and 0.0 < h < self._eat_below:
                    self._eat_fails += 1
                else:
                    self._eat_fails = 0
                return None                             # resume FSM next tick
            a = r.action; a.movement = dict(STOP)       # stand still to eat
            return a
        if self._eat_below <= 0 or self._eat_fails >= 3 or self._hotbar is None \
                or self._hotbar.best_slot_for("food") is None:
            return None
        # Eat only on a PLAUSIBLE low reading. Exactly 0.0 is almost always a
        # HUD misread (you rarely sit at 0/20), and acting on it spam-eats.
        if h is not None and 0.0 < h < self._eat_below:
            self._hungry_ticks += 1
        else:
            self._hungry_ticks = 0
        if self._hungry_ticks >= 6:                     # sustained low -> eat
            self._eat = Eat()
            print(f"[{self.name}] hungry ({h:.2f}) -> eating")
            a = self._eat.tick(ctx).action; a.movement = dict(STOP)
            return a
        return None

    @staticmethod
    def _pose_from_f3(f3):
        if f3 is None or f3.x is None or f3.yaw is None or f3.pitch is None:
            return None
        return SimpleNamespace(x=float(f3.x), y=float(f3.y), z=float(f3.z),
                               yaw=float(f3.yaw), pitch=float(f3.pitch),
                               dimension=getattr(f3, "dimension", None))

    def decide(self, state):
        from brain.interfaces import AgentAction as _AA
        def _emit(a):
            # Always fully specify movement so a held jump/forward from an
            # earlier tick is released when not actively commanded.
            a.movement = _full_movement(a.movement)
            return a
        if self._wp is None:
            if not self._warned_no_world:
                self._warned_no_world = True
                print(f"[{self.name}] needs vision.world.enabled: true — no "
                      "WorldMap to find targets. Idling.")
            return _emit(_AA())
        if self._fsm is None:
            if self._build_failed:
                return _emit(_AA())          # idle quietly; don't re-raise every tick
            try:
                self._build()
            except Exception as e:
                self._build_failed = True
                print(f"[{self.name}] build failed ({e}); idling.")
                return _emit(_AA())
        world = getattr(state, "world", None)
        pose = (world.pose if world is not None and world.pose is not None
                else self._pose_from_f3(getattr(state, "f3", None)))
        if pose is None:
            return _emit(_AA())                # wait for a pose this tick
        looking_at = world.looking_at if world is not None else None
        ctx = SkillContext(pose=pose, world_map=self._wp.world_map,
                           looking_at=looking_at, hotbar=self._hotbar,
                           px_per_deg=self._px_per_deg,
                           dimension=getattr(pose, "dimension", None))

        # Eat-when-hungry: pauses the FSM, stands still, and eats one item
        # when hunger stays low (debounced against HUD misreads). Keeps the
        # bot alive + sprint-capable on long unattended runs.
        eat_action = self._maybe_eat(ctx, state)
        if eat_action is not None:
            return _emit(eat_action)

        res = self._fsm.tick(ctx)
        if res.status == SkillStatus.DONE:
            self._on_fsm_done()                      # planner hook (default: no-op)
        if self._fsm._state != self._last_state:     # observability
            self._last_state = self._fsm._state
            print(f"[{self.name}] {self._fsm._state} | got={self._fsm.logs} "
                  f"runs={self._fsm.chopped} | {res.info}")
        return _emit(res.action)

    def _on_fsm_done(self) -> None:
        """Hook fired when the active behaviour FSM reports DONE. Default
        no-op (treechop/harvest just sit DONE). The planner advances tasks."""
        return


class HarvestAgent(TreeChopAgent):
    """Generic block gatherer — the SAME find -> navigate -> mine -> collect
    -> explore behaviour as the tree-chopper, but for ANY block type (no
    trunk-climb): a plain MineBlock once in reach. Proves the framework
    yields a new behaviour from a thin config, and gives a future goal/
    planner a second behaviour to sequence.

    Configure under settings.agent.harvest:
        match: ["stone", "cobblestone"]   # substrings of block ids to gather
        tool:  pickaxe                     # hotbar role to select for it
        max_blocks: 9999
    Run: ``python main.py --agent harvest``."""

    name = "harvest"
    cfg_key = "harvest"

    def _build(self) -> None:
        from knowledge.catalog import Catalog
        from vision.mc_assets import MCAssets
        from control.hotbar import build_hotbar_manager
        cat = Catalog.load(MCAssets.load())
        cfg = ((self._settings.get("agent", {}) or {}).get(self.cfg_key, {}) or {})
        match = [str(m).lower() for m in (cfg.get("match") or ["stone"])]
        tool = cfg.get("tool", "pickaxe")
        is_target = lambda b: bool(b) and any(m in str(b).lower() for m in match)
        mine_action = (lambda v: MineBlock(v, tool_role=tool,
                                           is_target=is_target,
                                           is_passthrough=is_target))
        self._hotbar = build_hotbar_manager(self._settings, catalog=cat)
        # is_breakable == the target predicate: only ever break what we're
        # gathering (no incidental block-breaking).
        self._fsm = FindAndChopLogs(is_log=is_target, is_breakable=is_target,
                                    tool_role=tool, max_logs=self._max_logs,
                                    mine_action=mine_action)


def parse_plan(spec: str) -> list:
    """Parse a CLI plan string into planner tasks. Format: comma-separated
    `name:count` (count optional). 'logs'/'wood'/'tree(s)' -> fell trees;
    any other name -> mine blocks whose id contains it (pickaxe).
        "logs:8, stone:16, diamond_ore:3"
    """
    tasks = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, count = tok, 9999
        if ":" in tok:
            name, c = tok.rsplit(":", 1)
            try:
                count = int(c.strip())
            except ValueError:
                count = 9999
        name = name.strip().lower()
        if name in ("logs", "log", "wood", "tree", "trees"):
            tasks.append({"kind": "logs", "count": count})
        else:
            tasks.append({"kind": "block", "match": [name],
                          "tool": "pickaxe", "count": count})
    return tasks


def _fsm_for_task(task: dict, cat) -> "FindAndChopLogs":
    """Build the gather FSM for one plan task. kind 'logs' fells trunks;
    anything else gathers blocks whose id contains any of `match`."""
    count = int(task.get("count", 9999))
    kind = str(task.get("kind", "logs")).lower()
    if kind in ("logs", "log", "wood", "tree", "trees"):
        log_ids = {b.id for b in cat.blocks_in_tag("logs")}
        leaf_ids = {b.id for b in cat.blocks_in_tag("leaves")}
        is_log = lambda b: bool(b) and (b in log_ids or str(b).endswith(_LOG_SUFFIXES))
        is_brk = lambda b: bool(b) and (
            is_log(b) or b in leaf_ids or str(b).endswith("_leaves"))
        return FindAndChopLogs(is_log=is_log, is_breakable=is_brk,
                               tool_role=task.get("tool", "axe"), goal_blocks=count)
    match = [str(m).lower() for m in (task.get("match") or [kind])]
    tool = task.get("tool", "pickaxe")
    pred = lambda b: bool(b) and any(m in str(b).lower() for m in match)
    mine = (lambda v: MineBlock(v, tool_role=tool, is_target=pred, is_passthrough=pred))
    return FindAndChopLogs(is_log=pred, is_breakable=pred, tool_role=tool,
                           goal_blocks=count, mine_action=mine)


class PlannerAgent(TreeChopAgent):
    """Goal/planner layer — runs an ORDERED plan of gather tasks, advancing
    to the next when the current behaviour reports DONE (count reached, or
    the area is exhausted). The T3 step: an agent that executes a multi-step
    plan by sequencing behaviours, reusing the whole gather stack
    (navigation, reach, recovery, eat, episode-logging) per task.

    Configure under settings.agent.planner.tasks, e.g.:
        - {kind: logs, count: 8}
        - {kind: block, match: ["stone","cobblestone"], tool: pickaxe, count: 16}
    Run: ``python main.py --agent planner``."""

    name = "planner"
    cfg_key = "planner"

    def __init__(self, settings: dict):
        super().__init__(settings)
        cfg = ((self._settings.get("agent", {}) or {}).get(self.cfg_key, {}) or {})
        self._tasks = list(cfg.get("tasks") or [{"kind": "logs", "count": 9999}])
        self._repeat = bool(cfg.get("repeat", False))   # loop the plan forever?
        self._task_i = 0

    def reset(self) -> None:
        super().reset()
        self._task_i = 0

    def telemetry(self) -> dict:
        t = super().telemetry()
        t["task"] = min(self._task_i + 1, len(self._tasks))
        t["of"] = len(self._tasks)
        if self._task_i < len(self._tasks):
            t["doing"] = self._tasks[self._task_i].get("kind", "?")
        return t

    def _build(self) -> None:
        from knowledge.catalog import Catalog
        from vision.mc_assets import MCAssets
        from control.hotbar import build_hotbar_manager
        cat = Catalog.load(MCAssets.load())
        self._hotbar = build_hotbar_manager(self._settings, catalog=cat)
        if self._task_i >= len(self._tasks):
            self._fsm = FindAndChopLogs(max_logs=0)      # nothing left -> idle (DONE)
            return
        task = self._tasks[self._task_i]
        print(f"[planner] task {self._task_i + 1}/{len(self._tasks)}: {task}")
        self._fsm = _fsm_for_task(task, cat)
        self._last_state = None

    def _on_fsm_done(self) -> None:
        if self._task_i >= len(self._tasks):
            return                                        # plan already finished
        self._task_i += 1
        if self._task_i < len(self._tasks):
            self._build()                                 # next task -> fresh FSM
        elif self._repeat:
            print(f"[planner] plan complete — repeating ({len(self._tasks)} tasks)")
            self._task_i = 0
            self._build()                                 # loop for unattended runs
        else:
            print(f"[planner] plan complete ({len(self._tasks)} tasks)")


def build_treechop_agent(settings: dict) -> TreeChopAgent:
    return TreeChopAgent(settings)


def build_planner_agent(settings: dict) -> PlannerAgent:
    return PlannerAgent(settings)


def build_harvest_agent(settings: dict) -> HarvestAgent:
    return HarvestAgent(settings)


__all__ = ["FindAndChopLogs", "TreeChopAgent", "HarvestAgent", "PlannerAgent",
           "build_treechop_agent", "build_harvest_agent", "build_planner_agent",
           "parse_plan"]
