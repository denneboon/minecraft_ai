"""
DescendToStone — get stone (cobblestone) from ANYWHERE by digging STRAIGHT DOWN.

Stone is a few blocks under the surface everywhere, so the bot never needs an
exposed face — it just digs down to it. Straight down (not a forward staircase)
because it's both robust and self-collecting:

  * The block directly below the feet is unambiguous to target (look straight
    down) — no fragile "aim at the cell ahead" that trips over slopes, trees, or
    your own body occluding the view.
  * In a 1-wide vertical shaft the mined block's drop lands at your feet as you
    fall into the gap, so the cobblestone is picked up automatically.

Per step: look straight down, read the block below; if it's lava/water STOP,
otherwise mine it and drop one block; pause a beat so the drop is collected;
repeat until enough stone is gathered.

Safety (conservative, surface-shallow): STOP on any lava/water read; STOP if a
single mine drops us more than a block (a cave beneath the floor); and only dig
a shallow, surface-depth shaft (``max_depth``) where lava doesn't occur — so it
can't dig itself into a deep lava lake or a fatal fall. Not provably 100% (the
block UNDER the one we mine is unobservable until we've dropped onto it), but
safe in practice for the shallow surface digging this is used for.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from brain.interfaces import AgentAction
from agents.skills import (Skill, SkillResult, SkillStatus, SkillContext,
                           MineBlock)

Voxel = Tuple[int, int, int]

_MAX_FALL = 2.0           # a single mine that drops us more than this == a cave

# Blocks safe to dig through (allowlist — anything else, incl. an unreadable id
# at depth, is treated as unsafe so we stop rather than mine blind).
_SAFE_DIG_STEMS = {
    "stone", "cobblestone", "deepslate", "cobbled_deepslate", "tuff",
    "andesite", "diorite", "granite", "calcite", "dripstone_block",
    "dirt", "grass_block", "coarse_dirt", "rooted_dirt", "podzol", "mycelium",
    "gravel", "sand", "red_sand", "sandstone", "red_sandstone", "clay",
    "moss_block", "mud", "packed_mud",
}
# Plant/foliage suffixes always safe to dig through (a tree in the column
# shouldn't stop the descent — leaves/saplings/logs break harmlessly).
_SAFE_DIG_SUFFIXES = ("_leaves", "_log", "_wood", "_sapling", "_roots")


def _stem(bid: Optional[str]) -> str:
    s = str(bid or "")
    return s.split(":", 1)[-1] if ":" in s else s


def is_safe_dig(bid: Optional[str]) -> bool:
    """True for a KNOWN dig-through block: the terrain allowlist + tree material.
    Anything else (unreadable id, lava/water, bedrock) is unsafe -> we stop."""
    stem = _stem(bid)
    return stem in _SAFE_DIG_STEMS or stem.endswith(_SAFE_DIG_SUFFIXES)


def is_hazard(bid: Optional[str]) -> bool:
    """Liquids — never mine into these."""
    return _stem(bid) in ("lava", "water", "flowing_lava", "flowing_water",
                          "bubble_column")


def is_stone_like(bid: Optional[str]) -> bool:
    """Mining this drops cobblestone (or cobbled deepslate)."""
    s = _stem(bid)
    return (("stone" in s and "sandstone" not in s) or "deepslate" in s
            or s in ("andesite", "diorite", "granite", "tuff"))


class DescendToStone(Skill):
    """Dig straight down, collecting ``count`` stone blocks (-> cobblestone).
    DONE carries however many were gathered."""
    name = "descend_to_stone"

    def __init__(self, count: int = 3, max_depth: int = 12,
                 tool_role: str = "pickaxe"):
        self.count = int(count)
        self.max_depth = int(max_depth)
        self.tool_role = tool_role
        self.reset()

    def reset(self):
        self.gathered = 0
        self._depth = 0
        self._phase = "mine"       # mine -> settle -> mine ...
        self._sub = None           # active MineBlock(below)
        self._t = 0
        self._mine_pos = None      # the block we're mining (directly below)
        self._mine_yield = False   # has it read as stone while mining?
        self._pre_y = None         # pose.y before the mine started (fall detect)

    def _done(self, why: str) -> SkillResult:
        return SkillResult(AgentAction(), SkillStatus.DONE,
                           f"{why} ({self.gathered} gathered)")

    def _below(self, pose) -> Voxel:
        return (int(math.floor(pose.x)), int(math.floor(pose.y)) - 1,
                int(math.floor(pose.z)))

    def _reads_below(self, ctx, below):
        """The F3 block id IF the crosshair is on the block directly below."""
        la = ctx.looking_at
        if la is not None and tuple(getattr(la, "pos", ()) or ()) == tuple(below):
            return getattr(la, "block_id", None)
        return None

    def tick(self, ctx: SkillContext) -> SkillResult:
        pose = ctx.pose
        if pose is None:
            return SkillResult(AgentAction(), SkillStatus.BLOCKED, "no pose")
        if self.gathered >= self.count:
            return self._done("gathered enough")
        if self._depth >= self.max_depth:
            return self._done("reached depth limit")

        below = self._below(pose)

        # 1. Mine the block directly below with the proven MineBlock skill (it
        #    aims straight down at the voxel and only mines a safe-dig block —
        #    it abandons lava/water on its own). We drop into the gap as it
        #    breaks; in the 1-wide shaft the cobblestone lands at our feet.
        if self._phase == "mine":
            # FIRST: did the block we were mining break? We detect that by the
            # DROP into its space (pose.y fell). Check it BEFORE re-targeting,
            # because falling changes 'below'.
            if self._sub is not None and self._pre_y is not None:
                fall = self._pre_y - float(pose.y)
                if fall >= 0.6:
                    if self._mine_yield:
                        self.gathered += 1
                    self._sub = None
                    if fall > _MAX_FALL:
                        return self._done("stopped: dropped into open space (cave)")
                    self._depth += 1
                    self._phase = "settle"; self._t = 0
                    return SkillResult(AgentAction(), SkillStatus.RUNNING,
                                       f"dug down (depth {self._depth}, got {self.gathered})")
            # Mine the block currently below.
            if self._sub is None or self._mine_pos != below:
                self._sub = MineBlock(below, tool_role=self.tool_role,
                                      is_target=is_safe_dig, is_passthrough=is_safe_dig)
                self._mine_pos = below; self._pre_y = float(pose.y)
                self._mine_yield = False; self._t = 0
            # Read the floor as we mine it: lava -> stop; remember if it's stone.
            bid = self._reads_below(ctx, below)
            if is_hazard(bid):
                self._sub = None
                return self._done(f"stop: {_stem(bid)} below")
            if is_stone_like(bid):
                self._mine_yield = True
            r = self._sub.tick(ctx)
            self._t += 1
            if r.status == SkillStatus.DONE:        # broke without a detected fall
                if self._mine_yield:
                    self.gathered += 1
                self._sub = None; self._phase = "settle"; self._t = 0
                return SkillResult(r.action, SkillStatus.RUNNING,
                                   f"dug down (depth {self._depth}, got {self.gathered})")
            if r.status in (SkillStatus.FAILED, SkillStatus.BLOCKED) or self._t > 70:
                self._sub = None
                return self._done("stop: couldn't dig down here")
            return r

        # 2. Settle: hold still a beat so the cobblestone drop at our feet is
        #    picked up before we mine the next block down.
        if self._phase == "settle":
            self._t += 1
            if self._t >= 8:
                self._phase = "mine"; self._t = 0
                return SkillResult(AgentAction(), SkillStatus.RUNNING, "collected")
            return SkillResult(AgentAction(movement={"forward": False}),
                               SkillStatus.RUNNING, "settling")

        return self._done("done")


__all__ = ["DescendToStone", "is_safe_dig", "is_hazard", "is_stone_like"]
