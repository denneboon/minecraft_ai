"""
DescendToStone — get stone (cobblestone) from ANYWHERE, safely, every time.

Stone is a few blocks under the surface everywhere, so the bot never needs
exposed stone — it digs DOWN to it. But digging straight down is the classic
way to die (drop into a cave or lava). This skill instead cuts a 1-wide
DESCENDING STAIRCASE and is conservative by construction:

  * It never mines the block under its own feet, so it can't fall into a hole
    it didn't expect — it always steps DOWN-AND-FORWARD onto a block it has
    already confirmed is solid.
  * Every dig needs POSITIVE confirmation the next move is safe: the block it's
    about to mine must read as a known SAFE-to-dig block (dirt/stone/gravel/…,
    never lava/water), and the next floor must read SOLID. On ANY hazard
    (lava/water) or uncertainty (a cave where the floor should be, an unreadable
    block, no pose) it STOPS with whatever it has gathered. It only ever acts on
    a confirmed-safe next step — so it cannot dig itself into lava or a fatal
    fall.

The geometry + safety predicates are a pure, offline-tested core
(:func:`plan_stair`, :func:`cardinal_step`, :func:`is_safe_dig`,
:func:`is_hazard`); the skill is the thin FSM that aims, reads, and drives
MineBlock / LookAtVoxel / WalkToward through them.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from brain.interfaces import AgentAction
from agents.skills import (Skill, SkillResult, SkillStatus, SkillContext,
                           MineBlock, LookAtVoxel, WalkToward, AIR_BLOCK)

Voxel = Tuple[int, int, int]

# Blocks safe to dig THROUGH on the way down (curated, version-stable stems).
# Deliberately a allowlist: anything NOT here (lava, water, bedrock, an
# unreadable id) is treated as unsafe, so the default is to STOP.
_SAFE_DIG_STEMS = {
    "stone", "cobblestone", "deepslate", "cobbled_deepslate", "tuff",
    "andesite", "diorite", "granite", "calcite", "dripstone_block",
    "dirt", "grass_block", "coarse_dirt", "rooted_dirt", "podzol", "mycelium",
    "gravel", "sand", "red_sand", "sandstone", "red_sandstone", "clay",
    "moss_block", "mud", "packed_mud",
}
# Blocks that drop the stone we're after (cobblestone). "stone" substring
# catches stone/cobblestone/*stone variants; deepslate handled explicitly.
def _default_is_yield(bid: Optional[str]) -> bool:
    s = str(bid or "")
    return ("stone" in s and "sandstone" not in s) or "deepslate" in s


def _stem(bid: Optional[str]) -> str:
    s = str(bid or "")
    return s.split(":", 1)[-1] if ":" in s else s


def is_safe_dig(bid: Optional[str]) -> bool:
    """True only for a KNOWN dig-through block — the allowlist. An unknown or
    unreadable id is unsafe (returns False), so the caller stops rather than
    mining blind."""
    return _stem(bid) in _SAFE_DIG_STEMS


def is_hazard(bid: Optional[str]) -> bool:
    """Liquids — never mine into these."""
    s = _stem(bid)
    return s in ("lava", "water", "flowing_lava", "flowing_water", "bubble_column")


def cardinal_step(yaw: float) -> Tuple[int, int]:
    """The (dx, dz) unit step for the cardinal direction nearest ``yaw`` (MC
    yaw: 0=+Z south, 90=-X west, 180=-Z north, 270=+X east). Digging along an
    axis keeps 'the block ahead' a single unambiguous column."""
    dx, dz = -math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
    if abs(dx) >= abs(dz):
        return (1 if dx > 0 else -1), 0
    return 0, (1 if dz > 0 else -1)


def plan_stair(feet: Voxel, step: Tuple[int, int]
               ) -> Tuple[Tuple[Voxel, Voxel, Voxel], Voxel, Voxel]:
    """One descending stair from ``feet`` going ``step``. Returns
    ``(cut, support, stand)``:

      * ``cut`` = the THREE cells to clear so the player (2 tall) can step
        forward-and-down: ``(head, mid, low)`` = ahead at head height, ahead at
        foot height, and the drop cell one below. All must be air to move into.
      * ``support`` = the block one below the drop cell; it MUST be solid — it's
        what the player lands on. (Confirming it is what makes the descent safe.)
      * ``stand`` = the player's new feet cell (forward 1, down 1)."""
    fx, fy, fz = feet
    sx, sz = step
    ax, az = fx + sx, fz + sz
    head = (ax, fy + 1, az)         # ahead at head height
    mid = (ax, fy, az)              # ahead at foot height
    low = (ax, fy - 1, az)          # the drop cell (becomes new feet)
    support = (ax, fy - 2, az)      # MUST be solid — landed-on floor
    stand = (ax, fy - 1, az)        # new feet
    return (head, mid, low), support, stand


class DescendToStone(Skill):
    """Cut a safe descending staircase until ``count`` stone blocks (→
    cobblestone) have been broken, or ``max_depth`` stairs cut, or a hazard /
    uncertainty forces a conservative stop. DONE carries however much was
    gathered (``self.gathered``); it never performs an unconfirmed-unsafe dig."""
    name = "descend_to_stone"

    def __init__(self, count: int = 3, max_depth: int = 10,
                 tool_role: str = "pickaxe"):
        self.count = int(count)
        self.max_depth = int(max_depth)
        self.tool_role = tool_role
        self.reset()

    def reset(self):
        self.gathered = 0
        self._depth = 0
        # face -> [look/mine the 3 ahead cells] -> verify support -> advance ...
        self._phase = "face"
        self._step = None               # (sx, sz) cardinal
        self._plan = None               # ((head,mid,low), support, stand)
        self._cut_i = 0                 # which of the 3 ahead cells we're on
        self._sub = None                # active MineBlock / LookAtVoxel / WalkToward
        self._look_ticks = 0            # cap on how long we aim to read a block
        self._cut_yield = False         # did the block we're cutting read as stone?

    def _next_cut(self):
        """Advance to the next of the 3 ahead cells, or to support-verify."""
        self._cut_i += 1
        self._phase = "verify_support" if self._cut_i >= 3 else "look"

    # -- helpers -------------------------------------------------------
    def _feet(self, pose) -> Voxel:
        return (int(math.floor(pose.x)), int(math.floor(pose.y)),
                int(math.floor(pose.z)))

    def _done(self, why: str) -> SkillResult:
        return SkillResult(AgentAction(), SkillStatus.DONE,
                           f"{why} ({self.gathered} gathered)")

    def _probe(self, ctx, pose, target) -> Tuple[str, Optional[str]]:
        """Classify ``target`` from the crosshair read: ``('solid', id)`` when
        the crosshair rests ON it; ``('clear', None)`` when the ray passed
        THROUGH it to a farther block (so it's air/transparent — nothing to
        cut); ``('unknown', None)`` when there's no usable read yet (keep aiming,
        and if it never resolves, STOP — never act on an unknown)."""
        la = ctx.looking_at
        p = (tuple(la.pos) if la is not None and getattr(la, "pos", None) is not None
             else None)
        if p is None:
            return "unknown", None
        if p == tuple(target):
            return "solid", getattr(la, "block_id", None)
        ex, ey, ez = float(pose.x), float(pose.y) + 1.62, float(pose.z)

        def _d2(v):
            return (v[0] + 0.5 - ex) ** 2 + (v[1] + 0.5 - ey) ** 2 + (v[2] + 0.5 - ez) ** 2
        # Crosshair landed on a block BEYOND the target along the ray => the
        # target voxel is see-through (air). Closer/sideways => can't vouch.
        return ("clear" if _d2(p) > _d2(target) else "unknown"), None

    def tick(self, ctx: SkillContext) -> SkillResult:
        pose = ctx.pose
        if pose is None:
            return SkillResult(AgentAction(), SkillStatus.BLOCKED, "no pose")
        if self.gathered >= self.count:
            return self._done("gathered enough")
        if self._depth >= self.max_depth:
            return self._done("max depth")

        # 0. Align to a cardinal so 'ahead' is one clean, unambiguous column.
        if self._phase == "face":
            self._step = cardinal_step(float(pose.yaw))
            self._plan = plan_stair(self._feet(pose), self._step)
            self._cut_i = 0
            self._phase = "look"
            return SkillResult(AgentAction(), SkillStatus.RUNNING, "facing set")

        cut, support, stand = self._plan

        # 1. Clear the 3 cells ahead (head, mid, drop), one at a time: AIM, READ,
        #    then mine ONLY a confirmed safe-dig block. Air is skipped; a hazard
        #    or an unreadable/unknown block STOPS the descent.
        if self._phase == "look":
            target = cut[self._cut_i]
            if self._sub is None:
                self._sub = LookAtVoxel(target, tol_deg=4.0)
            r = self._sub.tick(ctx)
            kind, bid = self._probe(ctx, pose, target)
            self._look_ticks += 1
            if kind in ("solid", "clear") or self._look_ticks > 16:
                self._look_ticks = 0; self._sub = None
                if kind == "clear":                  # air ahead -> nothing to cut
                    self._next_cut()
                    return SkillResult(AgentAction(), SkillStatus.RUNNING, "clear")
                if kind != "solid":                  # never resolved -> conservative stop
                    return self._done("stop: can't read the block ahead")
                if is_hazard(bid):
                    return self._done(f"stop: {_stem(bid)} ahead")
                if not is_safe_dig(bid):
                    return self._done(f"stop: won't dig {_stem(bid)}")
                self._cut_yield = _default_is_yield(bid)
                self._phase = "mine"
                return SkillResult(AgentAction(), SkillStatus.RUNNING, "safe -> mining")
            return SkillResult(r.action, SkillStatus.RUNNING, "aiming")

        if self._phase == "mine":
            target = cut[self._cut_i]
            if self._sub is None:
                self._sub = MineBlock(target, tool_role=self.tool_role,
                                      is_target=is_safe_dig, is_passthrough=is_safe_dig)
            r = self._sub.tick(ctx)
            if r.status == SkillStatus.DONE:
                if self._cut_yield:
                    self.gathered += 1
                    self._cut_yield = False
                self._sub = None
                self._next_cut()
                return SkillResult(r.action, SkillStatus.RUNNING, "cut")
            if r.status in (SkillStatus.FAILED, SkillStatus.BLOCKED):
                self._sub = None
                return self._done("stop: can't cut safely")
            return r

        if self._phase == "verify_support":
            if self._sub is None:
                self._sub = LookAtVoxel(support, tol_deg=4.0)
            r = self._sub.tick(ctx)
            kind, bid = self._probe(ctx, pose, support)
            self._look_ticks += 1
            if kind in ("solid", "clear") or self._look_ticks > 16:
                self._look_ticks = 0; self._sub = None
                # The floor we'd land on MUST be positively confirmed solid +
                # safe. 'clear' (air below) is a cave; 'unknown' we can't vouch
                # for; a hazard is lava/water — in every non-solid case we STOP
                # rather than step into a possible fall.
                if kind != "solid" or is_hazard(bid) or not is_safe_dig(bid):
                    why = ("cave below the next step" if kind == "clear"
                           else f"{_stem(bid)} below" if is_hazard(bid)
                           else "no confirmed solid floor")
                    return self._done(f"stop: {why}")
                self._phase = "advance"
                return SkillResult(AgentAction(), SkillStatus.RUNNING,
                                   "floor solid -> step down")
            return SkillResult(r.action, SkillStatus.RUNNING, "aiming at floor")

        if self._phase == "advance":
            if self._sub is None:
                self._sub = WalkToward(stand, arrive_dist=0.8,
                                       avoid_fall=False, jump_after=0)
            r = self._sub.tick(ctx)
            if r.status in (SkillStatus.DONE, SkillStatus.FAILED, SkillStatus.BLOCKED):
                self._sub = None
                self._depth += 1
                self._phase = "face"      # re-derive facing + plan from new pose
                return SkillResult(r.action, SkillStatus.RUNNING,
                                   f"descended (depth {self._depth})")
            return r

        return self._done("done")


__all__ = ["DescendToStone", "plan_stair", "cardinal_step",
           "is_safe_dig", "is_hazard"]
