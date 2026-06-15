#!/usr/bin/env python3
"""Offline self-test for agents/mining_descent.py — the SAFE descend-to-stone
strategy. Covers the pure safety/geometry core (cardinal step, stair plan,
dig/hazard predicates) and the skill's conservative stops (no pose; lava ahead
=> halt with nothing gathered, never an unsafe dig)."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

from agents.mining_descent import (DescendToStone, plan_stair, cardinal_step,
                                   is_safe_dig, is_hazard)
from agents.skills import SkillContext, SkillStatus

_fails = 0
def ok(m): print(f"  [OK]  {m}")
def bad(m):
    global _fails; _fails += 1; print(f"  [FAIL] {m}")


def _pose(x=0.5, y=64.0, z=0.5, yaw=0.0, pitch=0.0):
    return SimpleNamespace(x=x, y=y, z=z, yaw=yaw, pitch=pitch, dimension=None)


def main() -> int:
    print("=" * 56); print(" descend-to-stone (safe) — offline self-test"); print("=" * 56)

    # 1. cardinal step from yaw (MC: 0=+Z south, 90=-X west, 180=-Z north, 270=+X east).
    print("\n[1] cardinal step from yaw")
    cases = {0.0: (0, 1), 90.0: (-1, 0), 180.0: (0, -1), 270.0: (1, 0), -90.0: (1, 0)}
    for yaw, want in cases.items():
        got = cardinal_step(yaw)
        (ok if got == want else bad)(f"yaw {yaw} -> {want} (got {got})")

    # 2. stair geometry: 3 cells to clear (head, mid, drop) + a solid support
    #    one below the drop, and the new feet at forward+down-1.
    print("\n[2] stair plan geometry")
    cut, support, stand = plan_stair((0, 64, 0), (0, 1))
    (ok if cut == ((0, 65, 1), (0, 64, 1), (0, 63, 1)) else bad)(
        f"clears head/mid/drop ahead (got {cut})")
    (ok if support == (0, 62, 1) else bad)(f"support is one below the drop (got {support})")
    (ok if stand == (0, 63, 1) else bad)(f"stands forward+down-1 (got {stand})")

    # 3. dig/hazard predicates — allowlist for digging, liquids are hazards.
    print("\n[3] safe-dig allowlist + hazard liquids")
    for b in ("minecraft:stone", "minecraft:dirt", "minecraft:deepslate",
              "minecraft:gravel", "minecraft:grass_block"):
        (ok if is_safe_dig(b) else bad)(f"{b.split(':')[-1]} is safe to dig")
    for b in ("minecraft:bedrock", "minecraft:lava", "minecraft:water", None):
        (ok if not is_safe_dig(b) else bad)(f"{b} is NOT auto-dug")
    (ok if is_hazard("minecraft:lava") and is_hazard("minecraft:water") else bad)(
        "lava/water flagged as hazards")
    (ok if not is_hazard("minecraft:stone") else bad)("stone is not a hazard")

    # 4. conservative stops: no pose -> BLOCKED (never acts blind).
    print("\n[4] skill is conservative")
    sk = DescendToStone(count=3)
    r = sk.tick(SkillContext(pose=None, world_map=None))
    (ok if r.status == SkillStatus.BLOCKED else bad)(f"no pose -> BLOCKED ({r.status})")

    # 5. lava directly ahead -> the skill HALTS (DONE, 0 gathered), never mines.
    print("\n[5] lava ahead -> halt, gather nothing, no dig")
    sk = DescendToStone(count=3)
    pose = _pose(yaw=0.0)                       # faces +Z -> first cell ahead = (0,65,1)
    lava = SimpleNamespace(pos=(0, 65, 1), block_id="minecraft:lava")
    # tick 1: face (computes the plan); tick 2+: look at the head cell -> lava.
    sk.tick(SkillContext(pose=pose, world_map=None, looking_at=lava))
    res = None
    for _ in range(4):
        res = sk.tick(SkillContext(pose=pose, world_map=None, looking_at=lava))
        if res.status == SkillStatus.DONE:
            break
    (ok if res.status == SkillStatus.DONE and sk.gathered == 0 else bad)(
        f"halts on lava with 0 gathered (status={res.status}, got {sk.gathered})")
    (ok if "lava" in res.info or "stop" in res.info else bad)(
        f"reports the hazard ({res.info})")

    print("\n" + ("ALL DESCEND-TO-STONE TESTS PASSED" if not _fails
                  else f"{_fails} CHECK(S) FAILED"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
