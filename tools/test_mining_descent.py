#!/usr/bin/env python3
"""Offline self-test for agents/mining_descent.py — the straight-down
DescendToStone: dig/hazard/stone predicates (incl. digging through tree leaves),
and the skill's conservative stops (no pose; lava directly below => halt without
mining)."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

from agents.mining_descent import (DescendToStone, is_safe_dig, is_hazard,
                                   is_stone_like)
from agents.skills import SkillContext, SkillStatus

_fails = 0
def ok(m): print(f"  [OK]  {m}")
def bad(m):
    global _fails; _fails += 1; print(f"  [FAIL] {m}")


def _pose(x=0.5, y=64.0, z=0.5, yaw=0.0, pitch=0.0):
    return SimpleNamespace(x=x, y=y, z=z, yaw=yaw, pitch=pitch, dimension=None)


def main() -> int:
    print("=" * 56); print(" descend-to-stone (straight down) — offline self-test"); print("=" * 56)

    # 1. dig / hazard / stone predicates (incl. tree material via suffix).
    print("\n[1] block predicates")
    for b in ("minecraft:stone", "minecraft:dirt", "minecraft:gravel",
              "minecraft:deepslate", "minecraft:grass_block",
              "minecraft:birch_leaves", "minecraft:oak_log"):
        (ok if is_safe_dig(b) else bad)(f"{b.split(':')[-1]} safe to dig")
    for b in ("minecraft:bedrock", "minecraft:lava", "minecraft:water", None):
        (ok if not is_safe_dig(b) else bad)(f"{b} NOT auto-dug")
    (ok if is_hazard("minecraft:lava") and is_hazard("minecraft:water")
        and not is_hazard("minecraft:stone") else bad)("lava/water are hazards, stone isn't")
    (ok if is_stone_like("minecraft:stone") and is_stone_like("minecraft:granite")
        and not is_stone_like("minecraft:dirt")
        and not is_stone_like("minecraft:sandstone") else bad)(
        "stone/granite drop cobble; dirt/sandstone don't")

    # 2. no pose -> BLOCKED (never acts blind).
    print("\n[2] conservative: no pose")
    r = DescendToStone(count=3).tick(SkillContext(pose=None, world_map=None))
    (ok if r.status == SkillStatus.BLOCKED else bad)(f"no pose -> BLOCKED ({r.status})")

    # 3. lava directly below -> HALT (DONE, 0 gathered), never mines.
    print("\n[3] lava below -> halt, gather nothing")
    sk = DescendToStone(count=3)
    pose = _pose(x=0.5, y=64.0, z=0.5)
    below = (0, 63, 0)
    lava = SimpleNamespace(pos=below, block_id="minecraft:lava")
    ctx = SkillContext(pose=pose, world_map=None, looking_at=lava, targeted_pos=below)
    res = None
    for _ in range(30):
        res = sk.tick(ctx)
        if res.status == SkillStatus.DONE:
            break
    (ok if res.status == SkillStatus.DONE and sk.gathered == 0 else bad)(
        f"halts on lava, 0 gathered (status={res.status}, got {sk.gathered})")
    (ok if "lava" in res.info else bad)(f"reports the hazard ({res.info})")

    # 4. stone below + a drop (we fell into the gap) -> counts a cobblestone.
    print("\n[4] stone below, fell into the gap -> counts the cobblestone")
    sk2 = DescendToStone(count=3)
    stone = SimpleNamespace(pos=below, block_id="minecraft:stone")
    p0 = _pose(x=0.5, y=64.0, z=0.5)
    sk2.tick(SkillContext(pose=p0, world_map=None, looking_at=stone, targeted_pos=below))
    # next tick the bot has dropped a block (the floor broke) while reading stone
    p1 = _pose(x=0.5, y=63.0, z=0.5)
    r2 = sk2.tick(SkillContext(pose=p1, world_map=None, looking_at=stone, targeted_pos=below))
    (ok if sk2.gathered == 1 else bad)(f"counted 1 cobblestone on the drop (got {sk2.gathered})")
    (ok if "dug down" in r2.info else bad)(f"reports digging down ({r2.info})")

    print("\n" + ("ALL DESCEND-TO-STONE TESTS PASSED" if not _fails
                  else f"{_fails} CHECK(S) FAILED"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
