"""
DescendToStone — get stone (cobblestone) from ANYWHERE, safely, by digging a
simple descending staircase. Stone is a few blocks under the surface
everywhere, so the bot never needs an exposed face.

This works the way a person does, NOT by computing exact voxels and probing each
one (that was brittle — your own body occludes the down-forward cell and F3's id
garbles at steep close angles). Instead, per stair:

  1. Look DOWN-AND-FORWARD. F3's "Targeted Block" tells you the block right
     there (position survives an unreadable id).
  2. If it's lava/water -> STOP. Otherwise mine it.
  3. Walk forward — you step down into the hole you just made.
  4. Repeat until you've collected enough stone.

Safety (conservative): it stops on any lava/water it looks at, stops if it ever
falls more than a step (a cave under the floor), and only digs a shallow,
surface-depth pit — so it can't dig itself into a deep lava lake or a fatal
fall. If a direction is a wall/cliff it can't dig, it turns to another.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from brain.interfaces import AgentAction
from agents.skills import (Skill, SkillResult, SkillStatus, SkillContext,
                           MineBlock, _Aimer)

Voxel = Tuple[int, int, int]

_CARDINALS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_DOWN_PITCH = 62.0        # steep enough to target the IMMEDIATE next tread
                          # (~1 ahead, 1 down) — shallower aims a block too far
                          # away and the bot just walks across the surface.
_MAX_FALL = 2.2           # a drop bigger than this == a cave; stop
_STEP_REACH = 1.4         # walk at most this far per step before re-cutting

# Blocks safe to dig through (allowlist — anything else, incl. an unreadable id
# at depth, is treated as unsafe so we stop rather than mine blind).
_SAFE_DIG_STEMS = {
    "stone", "cobblestone", "deepslate", "cobbled_deepslate", "tuff",
    "andesite", "diorite", "granite", "calcite", "dripstone_block",
    "dirt", "grass_block", "coarse_dirt", "rooted_dirt", "podzol", "mycelium",
    "gravel", "sand", "red_sand", "sandstone", "red_sandstone", "clay",
    "moss_block", "mud", "packed_mud",
}


def _stem(bid: Optional[str]) -> str:
    s = str(bid or "")
    return s.split(":", 1)[-1] if ":" in s else s


def is_safe_dig(bid: Optional[str]) -> bool:
    """True only for a KNOWN dig-through block (the allowlist)."""
    return _stem(bid) in _SAFE_DIG_STEMS


def is_hazard(bid: Optional[str]) -> bool:
    """Liquids — never mine into these."""
    return _stem(bid) in ("lava", "water", "flowing_lava", "flowing_water",
                          "bubble_column")


def is_stone_like(bid: Optional[str]) -> bool:
    """Mining this drops cobblestone (or cobbled deepslate)."""
    s = _stem(bid)
    return (("stone" in s and "sandstone" not in s) or "deepslate" in s
            or s in ("andesite", "diorite", "granite", "tuff"))


def cardinal_step(yaw: float) -> Tuple[int, int]:
    """The (dx, dz) for the cardinal nearest ``yaw`` (MC: 0=+Z, 90=-X, 180=-Z,
    270=+X)."""
    dx, dz = -math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
    if abs(dx) >= abs(dz):
        return (1 if dx > 0 else -1), 0
    return 0, (1 if dz > 0 else -1)


def yaw_for(step: Tuple[int, int]) -> float:
    """The MC yaw that faces cardinal ``step`` (so 'forward' walks that way)."""
    sx, sz = step
    return math.degrees(math.atan2(-sx, sz))


class DescendToStone(Skill):
    """Dig a simple safe staircase down, collecting ``count`` stone blocks (→
    cobblestone). DONE carries however many were gathered."""
    name = "descend_to_stone"

    def __init__(self, count: int = 3, max_depth: int = 8,
                 tool_role: str = "pickaxe"):
        self.count = int(count)
        self.max_depth = int(max_depth)
        self.tool_role = tool_role
        self.reset()

    def reset(self):
        self.gathered = 0
        self._depth = 0
        self._phase = "face"        # face -> aim -> mine -> step -> (face)
        self._order = None          # cardinal try-order (nearest yaw first)
        self._dir_i = 0
        self._step = (1, 0)
        self._yaw = 0.0
        self._aim = _Aimer(tol_deg=5.0)
        self._sub = None            # active MineBlock
        self._t = 0                 # per-phase tick counter
        self._mine_pos = None       # voxel we're mining
        self._mine_yield = False    # is it stone (drops cobblestone)?
        self._step_y = None         # pose.y when the step (walk) began
        self._step_x0 = 0.0         # pose.x/z when the step began (cap walk dist)
        self._step_z0 = 0.0

    def _done(self, why: str) -> SkillResult:
        return SkillResult(AgentAction(), SkillStatus.DONE,
                           f"{why} ({self.gathered} gathered)")

    def _stop_or_turn(self, why: str) -> SkillResult:
        """A 'can't dig this way' before we've cut anything just means we faced
        a wall/cliff: turn to the next cardinal. A hazard, or any stop after
        we've started descending, is final."""
        if ("lava" not in why and "water" not in why and self._depth == 0
                and self.gathered == 0 and self._order is not None
                and self._dir_i + 1 < len(self._order)):
            self._dir_i += 1
            self._sub = None; self._t = 0; self._phase = "face"
            return SkillResult(AgentAction(), SkillStatus.RUNNING,
                               f"can't dig that way ({why}); turning")
        return self._done(why)

    def tick(self, ctx: SkillContext) -> SkillResult:
        pose = ctx.pose
        if pose is None:
            return SkillResult(AgentAction(), SkillStatus.BLOCKED, "no pose")
        if self.gathered >= self.count:
            return self._done("gathered enough")
        if self._depth >= self.max_depth:
            return self._done("reached depth limit")

        # 0. Pick / keep a dig direction.
        if self._phase == "face":
            if self._order is None:
                base = cardinal_step(float(pose.yaw))
                self._order = [base] + [c for c in _CARDINALS if c != base]
            self._step = self._order[self._dir_i]
            self._yaw = yaw_for(self._step)
            self._phase = "aim"; self._t = 0; self._aim = _Aimer(tol_deg=5.0)
            return SkillResult(AgentAction(), SkillStatus.RUNNING,
                               f"digging toward {self._step}")

        # 1. Look down-and-forward, then read the block there (by POSITION, which
        #    survives an unreadable id). Decide: hazard -> stop; air ahead ->
        #    nothing to dig this way -> turn; a safe block in front+below -> mine.
        if self._phase == "aim":
            dx, dy, aimed = self._aim.step(ctx, self._yaw, _DOWN_PITCH)
            self._t += 1
            if not aimed and self._t < 25:
                return SkillResult(AgentAction(look_dx=dx, look_dy=dy),
                                   SkillStatus.RUNNING, "aiming down-forward")
            tp = ctx.targeted_pos
            bid = getattr(ctx.looking_at, "block_id", None)
            if tp is None:
                return self._stop_or_turn("nothing to dig ahead")
            if is_hazard(bid):
                return self._done(f"stop: {_stem(bid)} ahead")
            if not self._in_front_below(pose, tp):
                # The crosshair grabbed our own footing / a side block — aim a
                # touch steeper and retry; give up this way if it won't resolve.
                if self._t < 40:
                    return SkillResult(AgentAction(look_dy=10),
                                       SkillStatus.RUNNING, "re-aiming lower")
                return self._stop_or_turn("can't sight the tread")
            # A readable non-safe block (e.g. bedrock) -> don't dig it.
            if bid is not None and not is_safe_dig(bid):
                return self._stop_or_turn(f"won't dig {_stem(bid)}")
            self._mine_pos = tuple(tp)
            self._mine_yield = is_stone_like(bid)
            self._phase = "mine"; self._t = 0
            return SkillResult(AgentAction(), SkillStatus.RUNNING, "tread -> mining")

        # 2. Mine that block. Detect the break by the F3 target moving OFF it
        #    (every block we dig matches is_safe_dig, so MineBlock's own id check
        #    can't tell when it broke — position can).
        if self._phase == "mine":
            if self._sub is None:
                self._sub = MineBlock(self._mine_pos, tool_role=self.tool_role,
                                      is_target=is_safe_dig, is_passthrough=is_safe_dig)
            r = self._sub.tick(ctx)
            self._t += 1
            tp = ctx.targeted_pos
            broke = (self._t > 5 and tp is not None and tuple(tp) != self._mine_pos)
            if broke or r.status == SkillStatus.DONE:
                self._sub = None
                self._step_y = float(pose.y)
                self._step_x0, self._step_z0 = float(pose.x), float(pose.z)
                # When we just broke STONE, hold still a moment so its dropped
                # cobblestone (right at the cut, ~1 block away) gets sucked in
                # BEFORE we step down past it — otherwise the drop is left in the
                # trench and the mined block never reaches the inventory.
                self._phase = "collect" if self._mine_yield else "step"
                self._t = 0
                return SkillResult(r.action, SkillStatus.RUNNING, "cut tread")
            if r.status in (SkillStatus.FAILED, SkillStatus.BLOCKED) or self._t > 50:
                self._sub = None
                return self._stop_or_turn("stop: couldn't cut the tread")
            return r

        # 2b. Collect: stand on the cut for a beat so the cobblestone is picked
        #     up (auto-pickup pulls items within ~1 block over ~0.5 s).
        if self._phase == "collect":
            self._t += 1
            if self._t >= 10:
                self.gathered += 1
                self._phase = "step"; self._t = 0
                return SkillResult(AgentAction(), SkillStatus.RUNNING, "collected")
            return SkillResult(AgentAction(movement={"forward": False}),
                               SkillStatus.RUNNING, "collecting drop")

        # 3. Walk forward — step down into the hole. Watch pose.y: a ~1-block
        #    drop is a stair; no drop for a while means a block is blocking head
        #    height ahead, so dig again; a big drop is a cave -> stop.
        if self._phase == "step":
            self._t += 1
            drop = (self._step_y - float(pose.y)) if self._step_y is not None else 0.0
            moved = math.hypot(float(pose.x) - self._step_x0,
                               float(pose.z) - self._step_z0)
            if drop >= 0.7:
                self._depth += 1
                if drop > _MAX_FALL:
                    return self._done("stopped: fell into open space (cave)")
                self._phase = "face"; self._t = 0
                return SkillResult(AgentAction(movement={"forward": False}),
                                   SkillStatus.RUNNING, f"descended (depth {self._depth})")
            if moved > _STEP_REACH or self._t > 16:
                # Walked a whole block without dropping — the tread wasn't right
                # below us (or a block is blocking head height). Re-cut: the aim
                # phase targets whatever is now in front+below. Stops us surfing
                # across flat ground instead of cutting DOWN.
                self._phase = "aim"; self._t = 0; self._aim = _Aimer(tol_deg=5.0)
                return SkillResult(AgentAction(movement={"forward": False}),
                                   SkillStatus.RUNNING, "no drop; cutting again")
            # hold our facing and push forward into the step.
            ddx, _, _ = self._aim.step(ctx, self._yaw, _DOWN_PITCH)
            return SkillResult(AgentAction(movement={"forward": True}, look_dx=ddx),
                               SkillStatus.RUNNING, "stepping down")

        return self._done("done")

    def _in_front_below(self, pose, tp) -> bool:
        """True if voxel ``tp`` is in front (along the dig step) and at/below
        the feet — i.e. a real staircase tread, not our own footing or a block
        behind/above us."""
        fx, fy, fz = (int(math.floor(pose.x)), int(math.floor(pose.y)),
                      int(math.floor(pose.z)))
        sx, sz = self._step
        ahead = (tp[0] - fx) * sx + (tp[2] - fz) * sz      # >0 == in the dig dir
        horiz = abs(tp[0] - fx) + abs(tp[2] - fz)
        return ahead >= 1 and horiz <= 3 and tp[1] <= fy


__all__ = ["DescendToStone", "cardinal_step", "yaw_for",
           "is_safe_dig", "is_hazard", "is_stone_like"]
