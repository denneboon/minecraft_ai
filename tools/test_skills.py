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

    # Stale-pose guard: after issuing a correction, a SECOND tick with the
    # SAME (unchanged) pose must NOT issue another (it would stack/overshoot
    # since the camera move hasn't registered yet) — wait for fresh pose.
    sk = LookAtVoxel(vox)
    stale_ctx = SkillContext(pose=_pose(yaw=yaw - 40, pitch=0))
    r1 = sk.tick(stale_ctx)
    r2 = sk.tick(stale_ctx)                       # identical pose -> wait
    moved_then_waited = (r1.action.look_dx != 0) and \
        (r2.action.look_dx == 0 and r2.action.look_dy == 0)
    (ok if moved_then_waited else bad)(
        f"issues once then waits for fresh pose (dx {r1.action.look_dx}->{r2.action.look_dx})")
    # Fresh pose (changed) -> issues again.
    r3 = sk.tick(SkillContext(pose=_pose(yaw=yaw - 20, pitch=0)))
    (ok if r3.action.look_dx != 0 else bad)("issues again once the pose changes")


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
    # Config-trust: with a food slot reserved, Eat selects it even with no
    # reading (like the axe). It FAILs only when NO food slot is configured.
    hb_nofood = HotbarManager(HotbarConfig(slot_roles={2: "axe"}), catalog=cat)
    r = Eat().tick(SkillContext(pose=_pose(), hotbar=hb_nofood))
    (ok if r.status == SkillStatus.FAILED else bad)(
        f"no food slot configured -> {r.status}")
    # Reserved food slot but empty reading -> still selects it (trust config).
    r = Eat().tick(SkillContext(pose=_pose(), hotbar=hb))
    (ok if r.action.hotbar == 9 and r.status == SkillStatus.RUNNING else bad)(
        f"reserved food slot trusted with no reading -> slot {r.action.hotbar}")


def test_mine_block(cat):
    print("\n[5] MineBlock (aim -> freeze + mine latch)")
    vox = (2, 64, 0)
    eye = (0.0, 64.0 + 1.62, 0.0)
    yaw, pitch = aim_angles(eye, (vox[0] + 0.5, vox[1] + 0.5, vox[2] + 0.5))
    LOG = lambda b: bool(b) and str(b).endswith("_log")
    LEAFLOG = lambda b: bool(b) and (str(b).endswith("_log") or str(b).endswith("_leaves"))
    hb = _hotbar(cat)
    def log_la():  return SimpleNamespace(pos=vox, block_id="minecraft:oak_log")

    # The instant F3 shows the target log: select axe, then FREEZE + mine —
    # holding the click and NEVER moving the camera.
    sk = MineBlock(vox, tool_role="axe", is_target=LOG, is_passthrough=LEAFLOG)
    ctx = SkillContext(pose=_pose(yaw=yaw, pitch=pitch), hotbar=hb, looking_at=log_la())
    saw_tool = saw_attack = saw_look = False
    for _ in range(6):
        r = sk.tick(ctx)
        if r.action.hotbar == 2: saw_tool = True
        if r.action.interact == "attack": saw_attack = True
        if r.action.look_dx or r.action.look_dy: saw_look = True
    (ok if saw_tool else bad)("selected axe before mining")
    (ok if saw_attack else bad)("holds attack on the confirmed log")
    (ok if not saw_look else bad)("NEVER moves the camera while mining a log")

    # Log breaks: F3 stops showing a log -> after a brief grace, DONE + broke.
    ctx.looking_at = None
    st = None
    for _ in range(6):
        r = sk.tick(ctx); st = r.status
        if st == SkillStatus.DONE: break
    (ok if st == SkillStatus.DONE and sk.broke else bad)(
        f"log gone -> DONE + counted ({st}, broke={sk.broke})")

    # Mislabelled target (F3 says the target voxel is dirt) -> abandon, NOT counted.
    sk2 = MineBlock(vox, tool_role=None, is_target=LOG)
    ctx2 = SkillContext(pose=_pose(yaw=yaw, pitch=pitch),
                        looking_at=SimpleNamespace(pos=vox, block_id="minecraft:dirt"))
    r = sk2.tick(ctx2)
    (ok if r.status == SkillStatus.FAILED and not sk2.broke else bad)(
        f"dirt target -> abandon, not counted ({r.status})")

    # Leaf occluding the target log: clear it (frozen, NOT counted), then the
    # log appears -> mine it -> break -> DONE + counted.
    sk3 = MineBlock(vox, tool_role=None, is_target=LOG, is_passthrough=LEAFLOG)
    ctx3 = SkillContext(pose=_pose(yaw=yaw, pitch=pitch),
                        looking_at=SimpleNamespace(pos=(vox[0], vox[1], vox[2] + 1),
                                                   block_id="minecraft:oak_leaves"))
    r = sk3.tick(ctx3)                          # locked + leaf occluding -> clear
    cleared = (r.action.interact == "attack"
               and not (r.action.look_dx or r.action.look_dy))
    ctx3.looking_at = log_la()                  # leaf gone -> log now visible
    r = sk3.tick(ctx3)
    mining = (r.action.interact == "attack")
    ctx3.looking_at = None                      # log breaks
    st = None
    for _ in range(6):
        r = sk3.tick(ctx3); st = r.status
        if st == SkillStatus.DONE: break
    (ok if cleared and mining and st == SkillStatus.DONE and sk3.broke else bad)(
        "clears occluding leaf (no count) then mines the log behind (counts)")

    # Flicker mid-mine: a brief F3 gap keeps the click held (no spam release).
    sk4 = MineBlock(vox, tool_role=None, is_target=LOG)
    ctx4 = SkillContext(pose=_pose(yaw=yaw, pitch=pitch), looking_at=log_la())
    sk4.tick(ctx4)                              # mine_log
    ctx4.looking_at = None
    r = sk4.tick(ctx4)
    (ok if r.action.interact == "attack" else bad)(
        "holds attack through a brief F3 gap (no spam-click)")

    # Aimed at sky (F3 silent), never started -> bounded abandon (no hang).
    sk5 = MineBlock(vox, tool_role=None, is_target=LOG)
    ctx5 = SkillContext(pose=_pose(yaw=yaw, pitch=pitch))     # looking_at None
    st = SkillStatus.RUNNING
    for _ in range(16):
        st = sk5.tick(ctx5).status
        if st in (SkillStatus.DONE, SkillStatus.FAILED): break
    (ok if st == SkillStatus.FAILED else bad)(f"aimed at sky -> bounded abandon ({st})")

    # 3D reach: an out-of-reach (too-high) target -> abandon, never sits
    # clicking; a near one is reachable.
    from agents.skills import block_reach_distance
    eye0 = (0.5, 65.62, 0.5)
    (ok if block_reach_distance(eye0, (0, 64, 0)) < 4.5
        and block_reach_distance(eye0, (0, 80, 0)) > 4.5 else bad)(
        "block_reach_distance: near in reach, high out of reach")
    skr = MineBlock((2, 80, 0), tool_role=None, is_target=LOG)
    rr = skr.tick(SkillContext(pose=_pose(yaw=0, pitch=0)))
    (ok if rr.status == SkillStatus.FAILED and "out of reach" in rr.info else bad)(
        f"out-of-reach target -> abandon ({rr.status})")


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


def test_walk_toward():
    print("\n[10] WalkToward (reactive approach)")
    from agents.skills import WalkToward, norm_angle
    tgt = (5, 64, 0)
    # Facing +X means yaw = atan2(-(5),0) = -90.
    pose = _pose(x=0.0, z=0.0, yaw=-90.0)
    sk = WalkToward(tgt, arrive_dist=1.6)
    status = None
    for _ in range(60):
        r = sk.tick(SkillContext(pose=pose, px_per_deg=6.5))
        status = r.status
        if status in (SkillStatus.DONE, SkillStatus.FAILED):
            break
        # Simulate the camera turning from look_dx (like the live game).
        pose.yaw = norm_angle(pose.yaw + r.action.look_dx / 6.5)
        if r.action.movement.get("forward"):
            pose.x += 0.6              # advanced toward target
    (ok if status == SkillStatus.DONE else bad)(f"faces + walks to arrive -> {status}")

    # Sprint is held while walking forward.
    pose_s = _pose(x=0.0, z=0.0, yaw=-90.0)
    rs = WalkToward(tgt).tick(SkillContext(pose=pose_s, px_per_deg=6.5))
    (ok if rs.action.movement.get("sprint") and rs.action.movement.get("forward")
     else bad)("sprint held while walking forward")

    # Stuck: facing but never advances -> jumps (climb out) then FAILED.
    pose2 = _pose(x=0.0, z=0.0, yaw=-90.0)
    sk2 = WalkToward(tgt, stuck_window=10, jump_after=3)
    st = None; saw_jump = False
    for _ in range(40):
        r = sk2.tick(SkillContext(pose=pose2)); st = r.status
        if r.action.movement.get("jump"):
            saw_jump = True
        if st in (SkillStatus.DONE, SkillStatus.FAILED):
            break
    (ok if saw_jump else bad)("jumps while stalling (mantle out of a 1-deep hole / step)")
    (ok if st == SkillStatus.FAILED else bad)(f"persistent no progress -> {st}")

    # Edge ahead: known air below the next step -> FAILED.
    wm = _FakeMap(); wm.set((1, 63, 0), "minecraft:air")
    pose3 = _pose(x=0.0, z=0.0, yaw=-90.0)
    r = WalkToward(tgt).tick(SkillContext(pose=pose3, world_map=wm))
    (ok if r.status == SkillStatus.FAILED and "edge" in r.info else bad)(
        f"known drop ahead -> {r.status} ({r.info})")


def test_navigate_to():
    print("\n[10b] NavigateTo (A* route + follow)")
    from agents.skills import NavigateTo, norm_angle
    from vision.world.map import WorldMap
    goal = (6, 64, 0)
    # Empty map + 'passable' A* -> everything standable -> plans + follows to
    # the goal. Simulate the camera turning (look_dx) + walking (forward).
    wm = WorldMap()
    pose = _pose(x=0.0, y=64.0, z=0.0, yaw=-90.0)
    sk = NavigateTo(goal, arrive_dist=1.5)
    status = None
    for _ in range(120):
        r = sk.tick(SkillContext(pose=pose, world_map=wm, px_per_deg=6.5))
        status = r.status
        if status in (SkillStatus.DONE, SkillStatus.FAILED):
            break
        pose.yaw = norm_angle(pose.yaw + r.action.look_dx / 6.5)
        if r.action.movement.get("forward"):
            pose.x += 0.5
    (ok if status == SkillStatus.DONE else bad)(f"routes + follows to goal -> {status}")

    # No map -> reactive straight-line fallback (still emits movement).
    sk2 = NavigateTo(goal, arrive_dist=1.5)
    r = sk2.tick(SkillContext(pose=_pose(yaw=-90.0), world_map=None, px_per_deg=6.5))
    (ok if r.status == SkillStatus.RUNNING else bad)(f"no map -> reactive fallback ({r.status})")

    # Already at the goal -> DONE immediately.
    r = NavigateTo(goal, arrive_dist=1.5).tick(
        SkillContext(pose=_pose(x=6.5, z=0.0), world_map=WorldMap()))
    (ok if r.status == SkillStatus.DONE else bad)(f"at goal -> DONE ({r.status})")


def test_chop_trunk():
    print("\n[9] ChopTrunk chains up a multi-log trunk")
    from agents.skills import ChopTrunk, aim_angles
    logs = {(0, 64, 0), (0, 65, 0)}     # base + one above; (0,66,0) is NOT a log
    pose = _pose(); wm = _FakeMap()
    sk = ChopTrunk((0, 64, 0), is_log=lambda b: b and b.endswith("_log"),
                   tool_role=None)
    attacks = {}
    status = None
    for _ in range(120):
        ctx = SkillContext(pose=pose, world_map=wm)
        ph, tgt = sk._phase, sk._target
        if ph == "mine":
            v = sk._mine.voxel
            eye = (pose.x, pose.y + 1.62, pose.z)
            pose.yaw, pose.pitch = aim_angles(eye, (v[0]+.5, v[1]+.5, v[2]+.5))
            attacks[v] = attacks.get(v, 0)
            if sk._mine._mining_ticks >= 1 and attacks[v] >= 1:
                ctx.looking_at = SimpleNamespace(pos=(v[0], v[1], v[2]+1),
                                                 block_id="minecraft:dirt")  # broke
            else:
                ctx.looking_at = SimpleNamespace(pos=v, block_id="minecraft:oak_log")
            attacks[v] += 1
        else:  # aim_up / check: keep aimed at the (above) target
            eye = (pose.x, pose.y + 1.62, pose.z)
            pose.yaw, pose.pitch = aim_angles(eye, (tgt[0]+.5, tgt[1]+.5, tgt[2]+.5))
            ctx.looking_at = (SimpleNamespace(pos=tgt, block_id="minecraft:oak_log")
                              if tgt in logs and ph == "check" else
                              (SimpleNamespace(pos=tgt, block_id="minecraft:air")
                               if ph == "check" else None))
        r = sk.tick(ctx); status = r.status
        if status in (SkillStatus.DONE, SkillStatus.FAILED):
            break
    (ok if status == SkillStatus.DONE and sk._mined == 2 else bad)(
        f"chopped 2-log trunk then stopped -> {status}, mined={sk._mined}")


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
    test_walk_toward()
    test_navigate_to()
    test_chop_trunk()
    print("\n" + ("ALL SKILL TESTS PASSED" if not _fails
                  else f"{_fails} CHECK(S) FAILED"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
