#!/usr/bin/env python3
"""
Offline self-test for the WorldMap A* pathfinder.

Builds synthetic WorldMap topologies and asserts the pathfinder
produces correct waypoints — independent of Minecraft, screen capture,
or any OCR. Suitable for CI.

Run:
    python tools/test_pathfind.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.world.map import WorldMap, AIR_BLOCK
from vision.world.types import BlockObservation
from vision.world.pathfind import (
    PathfinderConfig,
    find_path,
    is_passable,
    is_solid,
    is_hazard,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fail(msg: str) -> None:
    print(f"\n[FAIL] {msg}")
    sys.exit(1)


def _ok(msg: str) -> None:
    print(f"  [ok] {msg}")


def _put(wm: WorldMap, pos: Tuple[int, int, int], block_id: str) -> None:
    """Insert a synthetic block observation."""
    wm.update_block(BlockObservation(
        pos=pos, block_id=block_id,
        confidence=1.0, source="manual", last_seen_tick=1,
    ))


def _flat_floor(wm: WorldMap,
                x_range: Tuple[int, int],
                y: int,
                z_range: Tuple[int, int],
                block_id: str = "minecraft:grass_block",
                ) -> None:
    """Fill a rectangle of solid blocks at height y."""
    for x in range(x_range[0], x_range[1] + 1):
        for z in range(z_range[0], z_range[1] + 1):
            _put(wm, (x, y, z), block_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_block_classification() -> None:
    print("\n[1] block-classification helpers")
    assert is_passable("minecraft:short_grass"), "short_grass is passable"
    assert is_passable(AIR_BLOCK)
    assert is_passable(None), "unknown defaults to passable"
    assert is_solid("minecraft:grass_block")
    assert not is_solid(None), "unknown is not a foothold"
    assert not is_solid("minecraft:short_grass")
    assert is_hazard("minecraft:lava")
    assert not is_hazard("minecraft:grass_block")
    _ok("passable / solid / hazard classifications correct")


def test_same_cell_returns_trivial_path() -> None:
    print("\n[2] start == goal returns the single-cell path")
    wm = WorldMap()
    _flat_floor(wm, (-1, 1), 63, (-1, 1))
    res = find_path(wm, (0, 64, 0), (0, 64, 0))
    if not res or res.waypoints != [(0, 64, 0)]:
        _fail(f"trivial path malformed: {res}")
    _ok("trivial same-cell path returned correctly")


def test_flat_field_finds_shortest_path() -> None:
    print("\n[3] flat 5x5 field — A* returns Manhattan-shortest path")
    wm = WorldMap()
    _flat_floor(wm, (-1, 5), 63, (-1, 5))
    res = find_path(wm, (0, 64, 0), (4, 64, 4))
    if not res:
        _fail(f"expected path, got {res}")
    if res.waypoints[0] != (0, 64, 0) or res.waypoints[-1] != (4, 64, 4):
        _fail(f"path endpoints wrong: {res.waypoints[:3]} ... {res.waypoints[-3:]}")
    # 4 east + 4 south = 8 cardinal moves. Manhattan distance is 8;
    # waypoint count = moves + 1 = 9.
    if len(res.waypoints) != 9:
        _fail(f"expected 9 waypoints (8 cardinal moves + start), got "
              f"{len(res.waypoints)}: {res.waypoints}")
    # Each step must change exactly one of (x, z) by 1.
    for a, b in zip(res.waypoints[:-1], res.waypoints[1:]):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        dz = abs(a[2] - b[2])
        if dx + dy + dz != 1:
            _fail(f"non-unit step in path: {a} -> {b}")
    _ok(f"flat path length={len(res.waypoints)} cost={res.cost:.1f} "
        f"nodes={res.nodes_expanded}")


def test_wall_routes_around() -> None:
    print("\n[4] wall in the middle — path routes around")
    wm = WorldMap()
    # Floor at y=63 for x=-1..5, z=-1..5
    _flat_floor(wm, (-1, 5), 63, (-1, 5))
    # Wall at x=2, y=64, z=0..5 — blocks any direct line east-west
    # through that column. Make it 2 tall so the agent can't jump over.
    for z in range(0, 6):
        _put(wm, (2, 64, z), "minecraft:stone")
        _put(wm, (2, 65, z), "minecraft:stone")
        _put(wm, (2, 66, z), "minecraft:stone")

    res = find_path(wm, (0, 64, 1), (4, 64, 1))
    if not res:
        _fail(f"wall-routing failed: {res}")
    # Path must NOT cross the wall at x=2 on any of z=0..5.
    for wp in res.waypoints:
        if wp[0] == 2 and 0 <= wp[2] <= 5:
            _fail(f"path crosses wall at {wp}: {res.waypoints}")
    _ok(f"path routes around wall (length={len(res.waypoints)})")


def test_step_up_uses_jump() -> None:
    print("\n[5] 1-block step up — pathfinder uses the jump-up move")
    wm = WorldMap()
    _flat_floor(wm, (-1, 4), 63, (-1, 1))
    # Add a step: floor at y=64 from x=2..4. So x=0,1 stand at y=64,
    # x=2,3,4 stand at y=65.
    _flat_floor(wm, (2, 4), 64, (-1, 1))

    res = find_path(wm, (0, 64, 0), (4, 65, 0))
    if not res:
        _fail(f"expected step-up path, got {res}")
    if res.waypoints[0] != (0, 64, 0) or res.waypoints[-1] != (4, 65, 0):
        _fail(f"step-up endpoints wrong: {res.waypoints[0]} -> "
              f"{res.waypoints[-1]}")
    # Should be exactly one Y change of +1 somewhere in the path.
    y_jumps = [b[1] - a[1] for a, b in zip(res.waypoints[:-1], res.waypoints[1:])]
    if sum(j for j in y_jumps if j > 0) != 1:
        _fail(f"expected exactly one +1 y-step, got jumps={y_jumps}")
    _ok(f"step-up path length={len(res.waypoints)}")


def test_drop_down() -> None:
    print("\n[6] 2-block drop is reachable, 5-block drop is not")
    wm = WorldMap()
    # High platform at y=64 for x=0..1, low platform at y=62 for x=2..4.
    _flat_floor(wm, (0, 1), 63, (-1, 1))
    _flat_floor(wm, (2, 4), 61, (-1, 1))
    res = find_path(wm, (0, 64, 0), (3, 62, 0))
    if not res:
        _fail(f"expected drop path, got {res}")
    if res.waypoints[-1] != (3, 62, 0):
        _fail(f"drop goal wrong: {res.waypoints[-1]}")
    _ok(f"2-block drop path length={len(res.waypoints)}")

    # Replace the low platform with one 5 blocks below — beyond
    # max_fall_blocks=3. Pathfinder should refuse.
    wm2 = WorldMap()
    _flat_floor(wm2, (0, 1), 63, (-1, 1))
    _flat_floor(wm2, (2, 4), 58, (-1, 1))
    res2 = find_path(wm2, (0, 64, 0), (3, 59, 0))
    if res2:
        _fail(f"expected 5-block drop to be impassable, got {res2}")
    if res2.reason != "blocked":
        _fail(f"expected reason=blocked, got reason={res2.reason!r}")
    _ok("5-block drop correctly rejected as impassable")


def test_blocked_returns_none() -> None:
    print("\n[7] target inside a solid block returns blocked")
    wm = WorldMap()
    _flat_floor(wm, (-1, 4), 63, (-1, 1))
    _put(wm, (3, 64, 0), "minecraft:stone")     # the goal cell itself is solid
    res = find_path(wm, (0, 64, 0), (3, 64, 0))
    if res:
        _fail(f"expected blocked, got {res}")
    if res.reason != "blocked":
        _fail(f"expected reason=blocked, got {res.reason!r}")
    _ok("path into solid block correctly blocked")


def test_unknown_policy_truth_table() -> None:
    print("\n[8] unknown-policy: passable vs ground_only truth table")
    wm = WorldMap()
    # Observed stone ground along x=0..3 at y=63, so (0..3, 64, 0)
    # are standable under any policy. Cells off this corridor are
    # unobserved.
    for x in range(0, 4):
        _put(wm, (x, 63, 0), "minecraft:stone")

    # Within the observed corridor: both policies should find the path.
    for policy in ("passable", "ground_only"):
        res = find_path(
            wm, (0, 64, 0), (3, 64, 0),
            config=PathfinderConfig(unknown_policy=policy),
        )
        if not res:
            _fail(f"{policy}: should find path through observed corridor, got {res}")
        _ok(f"{policy}: observed-corridor path length={len(res.waypoints)}")

    # Now ask for a target where the FLOOR is unobserved.
    # passable     -> finds it (treats unknown floor as solid)
    # ground_only  -> refuses (no observed ground beneath goal)
    res_passable = find_path(
        wm, (0, 64, 0), (3, 64, 5),
        config=PathfinderConfig(unknown_policy="passable",
                                max_nodes_expanded=5000),
    )
    if not res_passable:
        _fail(f"passable should find path to unobserved-floor goal, got {res_passable}")
    _ok(f"passable: unobserved-floor goal reached "
        f"(length={len(res_passable.waypoints)})")

    res_ground_only = find_path(
        wm, (0, 64, 0), (3, 64, 5),
        config=PathfinderConfig(unknown_policy="ground_only"),
    )
    if res_ground_only:
        _fail(f"ground_only should refuse unobserved-floor goal, got {res_ground_only}")
    _ok("ground_only: unobserved-floor goal correctly refused")


def test_budget_limit() -> None:
    print("\n[9] expand-budget abort returns reason='budget'")
    wm = WorldMap()
    # Large empty maze that exceeds the tiny budget we'll request.
    _flat_floor(wm, (-50, 50), 63, (-50, 50))
    res = find_path(
        wm, (0, 64, 0), (40, 64, 40),
        config=PathfinderConfig(max_nodes_expanded=10),
    )
    if res.reason != "budget":
        _fail(f"expected reason=budget, got reason={res.reason!r}: {res}")
    _ok(f"budget abort at {res.nodes_expanded} nodes expanded")


def test_drop_through_solid_column_rejected() -> None:
    """A solid block in the fall column must block the drop, even if
    the landing cell on the far side is technically standable."""
    print("\n[11] drop blocked by a solid in the fall column")
    wm = WorldMap()
    # Floor at y=63 for x=0..1, z=0. Drop target at y=61.
    _flat_floor(wm, (0, 1), 63, (-1, 1))
    _flat_floor(wm, (2, 4), 61, (-1, 1))
    # Add a solid block at (2, 63, 0) — directly in the path of the
    # horizontal step-off. The player would crash into it before
    # falling.
    _put(wm, (2, 63, 0), "minecraft:stone")
    res = find_path(wm, (0, 64, 0), (3, 62, 0))
    if res:
        # Some path might still exist via z=±1 lanes, so we need to
        # check the (2, 63, 0) cell wasn't traversed. The CURRENT
        # path must NOT include (2, 62, 0) as a direct drop from
        # (1, 64, 0) — the column is blocked.
        for a, b in zip(res.waypoints[:-1], res.waypoints[1:]):
            if a == (1, 64, 0) and b == (2, 62, 0):
                _fail(f"path tried to drop through solid at (2, 63, 0): "
                      f"{res.waypoints}")
        _ok(f"alt-lane path length={len(res.waypoints)}; never drops "
            f"through the solid column")
    else:
        _ok(f"drop through solid column correctly refused (reason="
            f"{res.reason})")


def test_hazard_blocks_avoided() -> None:
    print("\n[12] hazard voxels (lava, sweet_berry_bush) are avoided")
    wm = WorldMap()
    _flat_floor(wm, (-1, 4), 63, (-1, 1))
    # Place lava at (2, 64, 0) — blocks the direct east path. Player
    # must route around via z=-1 or z=+1.
    _put(wm, (2, 64, 0), "minecraft:lava")
    res = find_path(wm, (0, 64, 0), (4, 64, 0))
    if not res:
        _fail(f"path should route around lava, got {res}")
    for wp in res.waypoints:
        if wp == (2, 64, 0):
            _fail(f"path walks INTO lava: {res.waypoints}")
    _ok(f"lava avoided; path length={len(res.waypoints)}")

    # Repeat with sweet_berry_bush — damage on contact.
    wm2 = WorldMap()
    _flat_floor(wm2, (-1, 4), 63, (-1, 1))
    _put(wm2, (2, 64, 0), "minecraft:sweet_berry_bush")
    res2 = find_path(wm2, (0, 64, 0), (4, 64, 0))
    if not res2:
        _fail(f"path should route around berry bush, got {res2}")
    for wp in res2.waypoints:
        if wp == (2, 64, 0):
            _fail(f"path walks INTO sweet_berry_bush: {res2.waypoints}")
    _ok(f"sweet_berry_bush avoided; path length={len(res2.waypoints)}")


def test_magma_block_floor_rejected() -> None:
    """Walking ON TOP of magma damages the player. The pathfinder
    treats magma as a non-floor so the agent can't accidentally pick
    a route across a magma surface."""
    print("\n[13] magma_block is not a valid floor")
    wm = WorldMap()
    _flat_floor(wm, (-1, 4), 63, (-1, 1))
    # Replace one floor block with magma.
    _put(wm, (2, 63, 0), "minecraft:magma_block")
    res = find_path(wm, (0, 64, 0), (4, 64, 0))
    if res:
        for wp in res.waypoints:
            if wp == (2, 64, 0):
                _fail(f"path stands on magma at {wp}: {res.waypoints}")
        _ok(f"magma floor avoided; path length={len(res.waypoints)}")
    else:
        # If the only route was via the magma column and there's no
        # z=±1 alternative, "blocked" is also acceptable. The test
        # field has z=±1 paths, so we expect a successful detour.
        _fail(f"path should still exist via z=±1, got {res}")


def test_descent_in_unobserved_world() -> None:
    """Regression: in ``passable`` mode (every unknown cell is
    optimistic-standable) the pathfinder used to skip drop neighbours
    when the same-Y walk also succeeded — which is ALWAYS the case in
    fully-unobserved space. The result was that A* could only plan
    paths at the same-Y or higher; any descent goal blew the node
    budget. The fix: only skip drops when the floor is OBSERVED solid
    (in which case the player physically cannot fall through it)."""
    print("\n[14] passable mode plans paths through descent in unobserved world")
    wm = WorldMap()  # fully empty
    cases = [
        ((0, 64, 0), (0, 63, 0), 'straight-down 1'),
        ((0, 64, 0), (5, 60, 5), 'diagonal descent'),
        ((-91, 91, -96), (-95, 90, -94), 'live-test target'),
    ]
    for s, g, lbl in cases:
        res = find_path(wm, s, g,
                        config=PathfinderConfig(
                            unknown_policy='passable',
                            max_nodes_expanded=5000))
        if not res:
            _fail(f"{lbl}: passable mode should find path "
                  f"through unobserved descent, got "
                  f"reason={res.reason} nodes={res.nodes_expanded}")
        _ok(f"{lbl}: nodes={res.nodes_expanded} cost={res.cost:.1f} "
            f"wp={len(res.waypoints)}")


def test_jump_blocked_by_ceiling() -> None:
    print("\n[10] jump-up blocked when a ceiling is directly overhead")
    wm = WorldMap()
    # Floor at y=63 everywhere, step up at x=2..3 (y=64 floor).
    _flat_floor(wm, (0, 4), 63, (-1, 1))
    _flat_floor(wm, (2, 3), 64, (-1, 1))
    # Ceiling at y=66 over x=0,1 — would block the jump arc at y+2.
    for x in range(0, 2):
        _put(wm, (x, 66, 0), "minecraft:stone")
    # Should still find a route — diagonal cells without ceiling.
    res = find_path(wm, (0, 64, 1), (3, 65, 1))
    if not res:
        _fail(f"path should still exist via z=-1/+1 lane, got {res}")
    _ok(f"jump+ceiling: path length={len(res.waypoints)}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 68)
    print(" Pathfinder offline self-test")
    print("=" * 68)
    test_block_classification()
    test_same_cell_returns_trivial_path()
    test_flat_field_finds_shortest_path()
    test_wall_routes_around()
    test_step_up_uses_jump()
    test_drop_down()
    test_blocked_returns_none()
    test_unknown_policy_truth_table()
    test_budget_limit()
    test_drop_through_solid_column_rejected()
    test_hazard_blocks_avoided()
    test_magma_block_floor_rejected()
    test_descent_in_unobserved_world()
    test_jump_blocked_by_ceiling()
    print("\n" + "=" * 68)
    print(" ALL TESTS PASSED")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
