#!/usr/bin/env python3
"""
Offline self-test for agents/skills.py — the aim geometry and each skill's
tick-FSM, driven by a synthetic SkillContext (no Minecraft). Verifies the
DECISION logic + transitions; the on-screen execution (does attack break
the block, does place land) is confirmed in a live session.
"""
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
    SkillStatus, SkillContext, SelectRole, LookAtVoxel, Eat, MineBlock,
    PillarUp, Bridge, aim_angles, norm_angle,
)
from control.hotbar import HotbarManager, HotbarConfig
from vision.world.map import AIR_BLOCK

_fails = 0
def ok(m): print(f"  [OK]  {m}")
def bad(m):
    global _fails; _fails += 1; print(f"  [FAIL] {m}")


def _pose(x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0):
    return SimpleNamespace(x=x, y=y, z=z, yaw=yaw, pitch=pitch)


class _FakeMap:
    def __init__(self): self.blocks = {}
    def set(self, vox, bid): self.blocks[vox] = SimpleNamespace(block_id=bid)
    def get_block(self, vox, dimension=None): return self.blocks.get(vox)


def _hotbar(cat):
    hb = HotbarManager(HotbarConfig(slot_roles={2: "axe", 3: "pickaxe",
                                                5: "blocks", 9: "food"}),
                       catalog=cat)
    hb.update(["minecraft:diamond_sword", "minecraft:diamond_axe",
               "minecraft:iron_pickaxe", None, "minecraft:oak_planks",
               None, None, None, "minecraft:bread"])
    return hb


def test_geometry():
    print("\n[1] aim geometry")
    eye = (0.0, 0.0, 0.0)
    checks = [
        ((0, 0, 5), 0.0, 0.0, "straight ahead +Z"),
        ((5, 0, 0), -90.0, 0.0, "to +X (right)"),
        ((-5, 0, 0), 90.0, 0.0, "to -X (left)"),
        ((0, 5, 0), 0.0, -90.0, "straight up"),
        ((0, -5, 0), 0.0, 90.0, "straight down"),
    ]
    for tgt, wy, wp, desc in checks:
        y, p = aim_angles(eye, tgt)
        good = abs(norm_angle(y - wy)) < 0.5 and abs(p - wp) < 0.5
        (ok if good else bad)(f"{desc}: yaw={y:.1f}(want {wy}) pitch={p:.1f}(want {wp})")


def test_select_role(cat):
    print("\n[2] SelectRole")
    hb = _hotbar(cat)
    r = SelectRole("axe").tick(SkillContext(pose=_pose(), hotbar=hb))
    (ok if r.status == SkillStatus.DONE and r.action.hotbar == 2 else bad)(
        f"axe -> DONE slot {r.action.hotbar}")
    hb.update([None] * 9)
    r = SelectRole("axe").tick(SkillContext(pose=_pose(), hotbar=hb))
    (ok if r.status == SkillStatus.FAILED else bad)(f"missing axe -> {r.status}")


def test_look_at_voxel():
    print("\n[3] LookAtVoxel")
    vox = (3, 64, 0)
    eye = (0.0, 64.0 + 1.62, 0.0)
    yaw, pitch = aim_angles(eye, (vox[0] + 0.5, vox[1] + 0.5, vox[2] + 0.5))
    # Aligned pose -> DONE.
    r = LookAtVoxel(vox).tick(SkillContext(pose=_pose(yaw=yaw, pitch=pitch)))
    (ok if r.status == SkillStatus.DONE else bad)(f"aligned -> {r.status}")
    # Misaligned -> RUNNING with a non-zero look.
    r = LookAtVoxel(vox).tick(SkillContext(pose=_pose(yaw=yaw - 40, pitch=0)))
    moving = r.action.look_dx != 0 or r.action.look_dy != 0
    (ok if r.status == SkillStatus.RUNNING and moving else bad)(
        f"misaligned -> RUNNING look_dx={r.action.look_dx}")
    # No pose -> BLOCKED.
    r = LookAtVoxel(vox).tick(SkillContext(pose=None))
    (ok if r.status == SkillStatus.BLOCKED else bad)(f"no pose -> {r.status}")


def test_eat(cat):
    print("\n[4] Eat")
    hb = _hotbar(cat)
    sk = Eat(hold_ticks=3)
    ctx = SkillContext(pose=_pose(), hotbar=hb)
    r = sk.tick(ctx)                                  # select food
    sel_ok = r.action.hotbar == 9 and r.status == SkillStatus.RUNNING
    statuses = [sk.tick(ctx).status for _ in range(3)]
    (ok if sel_ok and SkillStatus.DONE in statuses else bad)(
        f"select slot 9 then eat -> {statuses}")
    hb.update([None] * 9)
    r = Eat().tick(SkillContext(pose=_pose(), hotbar=hb))
    (ok if r.status == SkillStatus.FAILED else bad)(f"no food -> {r.status}")


def test_mine_block(cat):
    print("\n[5] MineBlock")
    vox = (2, 64, 0)
    eye = (0.0, 64.0 + 1.62, 0.0)
    yaw, pitch = aim_angles(eye, (vox[0] + 0.5, vox[1] + 0.5, vox[2] + 0.5))
    wm = _FakeMap(); wm.set(vox, "minecraft:oak_log")
    hb = _hotbar(cat)
    sk = MineBlock(vox, tool_role="axe", max_ticks=50)
    ctx = SkillContext(pose=_pose(yaw=yaw, pitch=pitch), world_map=wm, hotbar=hb)
    saw_tool = saw_attack = False
    for _ in range(8):
        r = sk.tick(ctx)
        if r.action.hotbar == 2: saw_tool = True
        if r.action.interact == "attack": saw_attack = True
    (ok if saw_tool else bad)("selected axe before mining")
    (ok if saw_attack else bad)("issued attack while aimed")
    # Now the block breaks (world carves it to air) -> DONE.
    wm.set(vox, AIR_BLOCK)
    r = sk.tick(ctx)
    (ok if r.status == SkillStatus.DONE else bad)(f"voxel->air -> {r.status}")
    # Timeout path.
    wm2 = _FakeMap(); wm2.set(vox, "minecraft:stone")
    sk2 = MineBlock(vox, tool_role=None, max_ticks=5)
    ctx2 = SkillContext(pose=_pose(yaw=yaw, pitch=pitch), world_map=wm2)
    st = SkillStatus.RUNNING
    for _ in range(20):
        st = sk2.tick(ctx2).status
        if st in (SkillStatus.DONE, SkillStatus.FAILED): break
    (ok if st == SkillStatus.FAILED else bad)(f"never breaks -> {st}")


def test_pillar_up(cat):
    print("\n[6] PillarUp")
    hb = _hotbar(cat)
    sk = PillarUp(height=2, per_block_budget=40)
    pose = _pose(y=64.0)
    ctx = SkillContext(pose=pose, hotbar=hb)
    status = SkillStatus.RUNNING
    # Simulate: every few ticks the player rises 1 block (a placed pillar).
    for i in range(60):
        r = sk.tick(ctx)
        status = r.status
        if status in (SkillStatus.DONE, SkillStatus.FAILED):
            break
        if i in (3, 9):           # two successful placements
            pose.y += 1.0
    (ok if status == SkillStatus.DONE else bad)(f"rose 2 blocks -> {status}")
    # FAILED when no upward progress.
    sk2 = PillarUp(height=1, per_block_budget=5)
    pose2 = _pose(y=64.0)
    st = SkillStatus.RUNNING
    for _ in range(20):
        st = sk2.tick(SkillContext(pose=pose2, hotbar=hb)).status
        if st in (SkillStatus.DONE, SkillStatus.FAILED): break
    (ok if st == SkillStatus.FAILED else bad)(f"stuck (no rise) -> {st}")


def test_bridge(cat):
    print("\n[7] Bridge (safe = keep_sneak)")
    hb = _hotbar(cat)
    sk = Bridge(length=3, keep_sneak=True, per_block_budget=40)
    pose = _pose(x=0.0, z=0.0)
    saw_sneak = False
    status = SkillStatus.RUNNING
    for i in range(60):
        r = sk.tick(SkillContext(pose=pose, hotbar=hb))
        status = r.status
        if r.action.movement.get("sneak"): saw_sneak = True
        if status in (SkillStatus.DONE, SkillStatus.FAILED): break
        if i % 4 == 3: pose.x += 1.05      # advanced one block
    (ok if status == SkillStatus.DONE else bad)(f"bridged 3 -> {status}")
    (ok if saw_sneak else bad)("keep_sneak held during safe bridge")


def test_sequence_and_query(cat):
    print("\n[8] SkillSequence + find_nearest_block")
    from agents.skills import SkillSequence, find_nearest_block, SelectRole
    hb = _hotbar(cat)
    # Sequence of two selects -> DONE after both.
    seq = SkillSequence([SelectRole("axe"), SelectRole("food")])
    ctx = SkillContext(pose=_pose(), hotbar=hb)
    sts = []
    for _ in range(5):
        r = seq.tick(ctx); sts.append(r.status)
        if r.status in (SkillStatus.DONE, SkillStatus.FAILED): break
    (ok if sts[-1] == SkillStatus.DONE else bad)(f"two-step sequence -> {sts}")
    # A failing step stops the sequence with FAILED.
    hb.update([None] * 9)   # no axe now
    seq2 = SkillSequence([SelectRole("axe"), SelectRole("food")])
    r = seq2.tick(SkillContext(pose=_pose(), hotbar=hb))
    (ok if r.status == SkillStatus.FAILED else bad)(f"failing step -> {r.status}")

    # find_nearest_block over a synthetic WorldMap of logs.
    from vision.world.map import WorldMap
    from vision.world.types import BlockObservation
    wm = WorldMap()
    for v, bid in [((10, 64, 0), "minecraft:oak_log"),
                   ((3, 64, 0), "minecraft:oak_log"),
                   ((1, 64, 0), "minecraft:stone"),
                   ((2, 64, 0), "minecraft:air")]:
        wm.update_block(BlockObservation(pos=v, block_id=bid, confidence=1.0,
                                         source="looking_at", last_seen_tick=0))
    res = find_nearest_block(wm, (0, 64, 0),
                             lambda b: b == "minecraft:oak_log", max_radius=48)
    (ok if res and res[0] == (3, 64, 0) else bad)(
        f"nearest oak_log from origin -> {res[0] if res else None} (want (3,64,0))")
    res2 = find_nearest_block(wm, (0, 64, 0),
                              lambda b: b == "minecraft:diamond_ore")
    (ok if res2 is None else bad)("no match -> None")


def main() -> int:
    print("=" * 60); print(" Skills — offline self-test"); print("=" * 60)
    from vision.mc_assets import MCAssets
    from knowledge.catalog import Catalog
    cat = Catalog.load(MCAssets.load())
    test_geometry()
    test_select_role(cat)
    test_look_at_voxel()
    test_eat(cat)
    test_mine_block(cat)
    test_pillar_up(cat)
    test_bridge(cat)
    test_sequence_and_query(cat)
    print("\n" + ("ALL SKILL TESTS PASSED" if not _fails
                  else f"{_fails} CHECK(S) FAILED"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
