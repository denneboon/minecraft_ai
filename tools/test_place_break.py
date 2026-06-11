#!/usr/bin/env python3
"""Offline self-test for placement/breaking — can_place_block decision,
PlaceBlock selection+place, BreakLookedAt break+confirm."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

from agents.skills import (
    SkillContext, SkillStatus, PlaceBlock, BreakLookedAt,
    can_place_block, placement_voxel, SKILLS,
)
from vision.world.map import AIR_BLOCK

_fails = 0
def ok(m): print(f"  [OK]  {m}")
def bad(m):
    global _fails; _fails += 1; print(f"  [FAIL] {m}")


def _pose(x=0.5, y=64.0, z=0.5, yaw=0.0, pitch=56.0):
    return SimpleNamespace(x=x, y=y, z=z, yaw=yaw, pitch=pitch,
                           eye_y=y + 1.62, dimension=None)

def _la(pos, face="up", block_id="minecraft:grass_block"):
    return SimpleNamespace(pos=pos, face=face, block_id=block_id)

def _wm(solid=()):
    solid = set(solid)
    return SimpleNamespace(get_block=lambda v, dimension=None: SimpleNamespace(
        block_id=("minecraft:stone" if v in solid else AIR_BLOCK)))

def _hotbar(roles):
    return SimpleNamespace(best_slot_for=lambda r: roles.get(r))


def main() -> int:
    print("=" * 56); print(" place / break — offline self-test"); print("=" * 56)

    # 1. placement_voxel: block pos offset by the targeted face normal.
    print("\n[1] placement_voxel from face")
    (ok if placement_voxel(_la((0, 63, 1), "up")) == (0, 64, 1) else bad)("up -> +y")
    (ok if placement_voxel(_la((0, 63, 1), "north")) == (0, 63, 0) else bad)("north -> -z")
    (ok if placement_voxel(_la((5, 64, 5), "east")) == (6, 64, 5) else bad)("east -> +x")
    (ok if placement_voxel(None) is None and placement_voxel(_la((0, 0, 0), None)) is None
     else bad)("no target / no face -> None")

    # 2. can_place_block decision.
    print("\n[2] can_place_block")
    p = _pose()                              # feet voxel (0,64,0), head (0,65,0)
    # ground block in front (0,63,1), top face -> place (0,64,1): valid
    (ok if can_place_block(p, _la((0, 63, 1), "up"), _wm()) == (0, 64, 1)
     else bad)("reachable ground in front -> placeable")
    # the block directly under us -> place would be our feet voxel -> blocked
    (ok if can_place_block(p, _la((0, 63, 0), "up"), _wm()) is None else bad)(
        "placing into our own feet -> None")
    # too far
    (ok if can_place_block(p, _la((0, 60, 9), "up"), _wm()) is None else bad)(
        "target out of reach -> None")
    # resulting voxel already solid
    (ok if can_place_block(p, _la((0, 63, 1), "up"), _wm(solid=[(0, 64, 1)])) is None
     else bad)("placement voxel occupied -> None")

    # 3. PlaceBlock: select slot, then place when possible.
    print("\n[3] PlaceBlock skill")
    (ok if "place_block" in SKILLS and "break_looked_at" in SKILLS else bad)(
        "skills registered")
    pb = PlaceBlock("blocks")
    ctx = SkillContext(pose=_pose(), looking_at=_la((0, 63, 1), "up"),
                       world_map=_wm(), hotbar=_hotbar({"blocks": 5}), px_per_deg=6.5)
    r = pb.tick(ctx)
    (ok if r.action.hotbar == 5 else bad)(f"first selects the blocks slot ({r.action.hotbar})")
    last = None
    for _ in range(8):                       # aim (pitch already at target) -> place
        last = pb.tick(ctx)
        if last.status == SkillStatus.DONE:
            break
    (ok if last.status == SkillStatus.DONE and last.action.interact == "use_item"
        and pb.placed_at == (0, 64, 1) else bad)(
        f"places via use_item at the placement voxel ({last.status}, {pb.placed_at})")
    # no blocks in hotbar -> FAILED
    pb2 = PlaceBlock("blocks")
    r2 = pb2.tick(SkillContext(pose=_pose(), hotbar=_hotbar({}), px_per_deg=6.5))
    (ok if r2.status == SkillStatus.FAILED else bad)("no block in hotbar -> FAILED")

    # 4. BreakLookedAt: attack while a block is there, DONE when gone.
    print("\n[4] BreakLookedAt skill")
    bl = BreakLookedAt()
    ctx_block = SkillContext(looking_at=_la((0, 64, 1), block_id="minecraft:crafting_table"))
    r = bl.tick(ctx_block)
    (ok if r.action.interact == "attack" and r.status == SkillStatus.RUNNING else bad)(
        "holds attack on the looked-at block")
    ctx_gone = SkillContext(looking_at=None)
    done = None
    for _ in range(5):
        done = bl.tick(ctx_gone)
        if done.status == SkillStatus.DONE:
            break
    (ok if done.status == SkillStatus.DONE and bl.broke else bad)(
        f"DONE + broke once the block is gone ({done.status}, broke={bl.broke})")
    # refuses a protected block
    bl2 = BreakLookedAt(avoid=lambda b: b == "minecraft:chest")
    r = bl2.tick(SkillContext(looking_at=_la((0, 64, 1), block_id="minecraft:chest")))
    (ok if r.status == SkillStatus.FAILED else bad)("refuses to break a protected block")

    print("\n" + ("ALL PLACE/BREAK TESTS PASSED" if not _fails
                  else f"{_fails} CHECK(S) FAILED"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
