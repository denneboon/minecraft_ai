#!/usr/bin/env python3
"""Offline self-test for agents/mining_descent.py — the simple SAFE
descend-to-stone staircase: direction/facing math, dig/hazard/stone predicates,
and the skill's conservative stops (no pose; lava ahead => halt, never dig)."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

from agents.mining_descent import (DescendToStone, cardinal_step, yaw_for,
                                   is_safe_dig, is_hazard, is_stone_like)
from agents.skills import SkillContext, SkillStatus

_fails = 0
def ok(m): print(f"  [OK]  {m}")
def bad(m):
    global _fails; _fails += 1; print(f"  [FAIL] {m}")


def _pose(x=0.5, y=64.0, z=0.5, yaw=0.0, pitch=0.0):
    return SimpleNamespace(x=x, y=y, z=z, yaw=yaw, pitch=pitch, dimension=None)


def main() -> int:
    print("=" * 56); print(" descend-to-stone (simple, safe) — offline self-test"); print("=" * 56)

    # 1. cardinal step + the yaw that faces it (round-trip).
    print("\n[1] facing math")
    for yaw, want in {0.0: (0, 1), 90.0: (-1, 0), 180.0: (0, -1), 270.0: (1, 0)}.items():
        (ok if cardinal_step(yaw) == want else bad)(
            f"yaw {yaw} -> step {want} (got {cardinal_step(yaw)})")
    for step in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        (ok if cardinal_step(yaw_for(step)) == step else bad)(
            f"yaw_for{step} faces back to {step}")

    # 2. dig / hazard / stone predicates.
    print("\n[2] block predicates")
    for b in ("minecraft:stone", "minecraft:dirt", "minecraft:gravel",
              "minecraft:deepslate", "minecraft:grass_block"):
        (ok if is_safe_dig(b) else bad)(f"{b.split(':')[-1]} safe to dig")
    for b in ("minecraft:bedrock", "minecraft:lava", "minecraft:water", None):
        (ok if not is_safe_dig(b) else bad)(f"{b} NOT auto-dug")
    (ok if is_hazard("minecraft:lava") and is_hazard("minecraft:water")
        and not is_hazard("minecraft:stone") else bad)("lava/water are hazards, stone isn't")
    (ok if is_stone_like("minecraft:stone") and is_stone_like("minecraft:granite")
        and not is_stone_like("minecraft:dirt")
        and not is_stone_like("minecraft:sandstone") else bad)(
        "stone/granite drop cobble; dirt/sandstone don't")

    # 3. no pose -> BLOCKED (never acts blind).
    print("\n[3] conservative: no pose")
    r = DescendToStone(count=3).tick(SkillContext(pose=None, world_map=None))
    (ok if r.status == SkillStatus.BLOCKED else bad)(f"no pose -> BLOCKED ({r.status})")

    # 4. lava in front+below -> HALT (DONE, 0 gathered), never mines.
    print("\n[4] lava ahead -> halt, gather nothing")
    sk = DescendToStone(count=3)
    pose = _pose(yaw=0.0)                       # faces +Z; step = (0,1)
    lava = SimpleNamespace(pos=(0, 63, 1), block_id="minecraft:lava")
    ctx = SkillContext(pose=pose, world_map=None, looking_at=lava,
                       targeted_pos=(0, 63, 1))
    res = None
    for _ in range(40):
        res = sk.tick(ctx)
        if res.status == SkillStatus.DONE:
            break
    (ok if res.status == SkillStatus.DONE and sk.gathered == 0 else bad)(
        f"halts on lava, 0 gathered (status={res.status}, got {sk.gathered})")
    (ok if "lava" in res.info else bad)(f"reports the hazard ({res.info})")

    print("\n" + ("ALL DESCEND-TO-STONE TESTS PASSED" if not _fails
                  else f"{_fails} CHECK(S) FAILED"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
