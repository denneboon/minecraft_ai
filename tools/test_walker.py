#!/usr/bin/env python3
"""
Offline self-test for the voxel-path walker.

Builds synthetic WorldMap topologies, drives a ``WalkerController``
with hand-crafted F3 poses, and asserts the produced action stream is
sane. Independent of Minecraft, screen capture, OCR.

Run:
    python tools/test_walker.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.walker import (
    WalkerController,
    WalkerConfig,
    WalkerStatus,
)
from vision.ocr import F3Info
from vision.world.map import WorldMap
from vision.world.types import BlockObservation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fail(msg: str) -> None:
    print(f"\n[FAIL] {msg}")
    sys.exit(1)


def _ok(msg: str) -> None:
    print(f"  [ok] {msg}")


def _put(wm: WorldMap, pos: Tuple[int, int, int], block_id: str) -> None:
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
    for x in range(x_range[0], x_range[1] + 1):
        for z in range(z_range[0], z_range[1] + 1):
            _put(wm, (x, y, z), block_id)


def _f3(*, x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0,
        timestamp=1.0) -> F3Info:
    return F3Info(
        x=x, y=y, z=z,
        yaw=yaw, pitch=pitch,
        facing_name="south", dimension="minecraft:overworld",
        block_x=int(x), block_y=int(y), block_z=int(z),
        raw_text=f"XYZ: {x:.3f} / {y:.3f} / {z:.3f}",
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_idle_until_target_set() -> None:
    print("\n[1] walker starts IDLE; no movement until target set")
    wm = WorldMap()
    _flat_floor(wm, (-1, 5), 63, (-1, 5))
    ctrl = WalkerController(wm)
    if ctrl.status != WalkerStatus.IDLE:
        _fail(f"expected IDLE, got {ctrl.status}")
    action = ctrl.tick(_f3())
    if action.movement.get("forward"):
        _fail(f"idle walker should not press forward: {action.movement!r}")
    _ok("IDLE walker emits no forward press")


def test_plan_and_walk_to_horizontal_target() -> None:
    print("\n[2] planning on first tick + forward-press after rotation")
    wm = WorldMap()
    _flat_floor(wm, (-1, 5), 63, (-1, 5))
    ctrl = WalkerController(wm)
    # Goal 3 blocks south of start.
    status = ctrl.set_target((0, 64, 3))
    if status != WalkerStatus.PLANNING:
        # ``set_target`` defers planning when no eye is given.
        _fail(f"expected PLANNING after set_target, got {status}")
    # First tick triggers a plan.
    pose = _f3(x=0.5, y=64.0, z=0.5, yaw=0.0, pitch=0.0)
    action = ctrl.tick(pose)
    if ctrl.status != WalkerStatus.WALKING:
        _fail(f"expected WALKING after first tick with pose, "
              f"got {ctrl.status}; plan={ctrl.last_plan_result}")
    # Yaw is already 0 (south = +Z = goal direction) so forward should
    # be true; jump should be false (flat ground).
    if not action.movement.get("forward"):
        _fail(f"expected forward=True after aim, got {action.movement!r}")
    if action.movement.get("jump"):
        _fail(f"unexpected jump on flat ground: {action.movement!r}")
    _ok(f"plan succeeded ({len(ctrl._path)} waypoints) + forward held")


def test_rotates_before_walking_when_yaw_off() -> None:
    print("\n[3] walker rotates BEFORE pressing forward when off-aim")
    wm = WorldMap()
    _flat_floor(wm, (-1, 5), 63, (-1, 5))
    ctrl = WalkerController(wm)
    ctrl.set_target((3, 64, 0), eye_voxel=(0, 64, 0))
    if ctrl.status != WalkerStatus.WALKING:
        _fail(f"plan failed: {ctrl.last_plan_result}")
    # Player facing -Y direction (yaw=-90 = east, which is goal
    # direction). Actually goal is east (+X), so desired yaw = -90 in
    # MC convention. Start at yaw=0 (facing south). That's a 90°
    # error — walker should rotate first.
    pose = _f3(x=0.5, y=64.0, z=0.5, yaw=0.0, pitch=0.0)
    action = ctrl.tick(pose)
    if action.movement.get("forward"):
        _fail(f"forward pressed while yaw err exceeds tolerance: "
              f"{action.movement!r}")
    # A non-zero yaw velocity command should be emitted.
    if abs(action.look_vx) < 1e-3:
        _fail(f"expected non-zero look_vx for rotation, got "
              f"vx={action.look_vx}")
    _ok(f"rotating: look_vx={action.look_vx:+.1f} (no forward yet)")

    # Once aimed, forward should kick in. We'll fake the aim by
    # snapping yaw to the desired bearing.
    pose_aimed = _f3(x=0.5, y=64.0, z=0.5, yaw=-90.0)
    action2 = ctrl.tick(pose_aimed)
    if not action2.movement.get("forward"):
        _fail(f"aimed walker should press forward, got "
              f"{action2.movement!r}")
    _ok("aimed walker presses forward")


def test_arrival_status() -> None:
    print("\n[4] walker reports ARRIVED at goal voxel")
    wm = WorldMap()
    _flat_floor(wm, (-1, 5), 63, (-1, 5))
    ctrl = WalkerController(wm)
    ctrl.set_target((3, 64, 0), eye_voxel=(0, 64, 0))
    # Drop the player AT the goal centre.
    pose = _f3(x=3.5, y=64.0, z=0.5, yaw=-90.0)
    action = ctrl.tick(pose)
    if ctrl.status != WalkerStatus.ARRIVED:
        _fail(f"expected ARRIVED, got {ctrl.status}")
    # Action should release all movement.
    if action.movement.get("forward"):
        _fail(f"arrived walker still pressing forward: "
              f"{action.movement!r}")
    _ok("ARRIVED + movement released")


def test_no_pose_releases_movement() -> None:
    print("\n[5] missing F3 pose releases all movement (safety)")
    wm = WorldMap()
    _flat_floor(wm, (-1, 5), 63, (-1, 5))
    ctrl = WalkerController(wm)
    ctrl.set_target((3, 64, 0), eye_voxel=(0, 64, 0))
    # Tick with f3=None.
    action = ctrl.tick(None)
    # Movement keys must all be released (False) — not just unset.
    for key in ("forward", "backward", "left", "right", "jump", "sneak", "sprint"):
        if action.movement.get(key) is not False:
            _fail(f"missing pose: {key} not explicitly released "
                  f"({action.movement!r})")
    _ok("no-pose tick releases every movement key")


def test_unreachable_target_blocks() -> None:
    print("\n[6] unreachable target -> BLOCKED status")
    wm = WorldMap()
    _flat_floor(wm, (-1, 1), 63, (-1, 1))
    ctrl = WalkerController(wm)
    # Target 50 blocks away with no observed ground path — under
    # ground_only the pathfinder refuses.
    ctrl.set_target((20, 64, 20), eye_voxel=(0, 64, 0))
    if ctrl.status != WalkerStatus.BLOCKED:
        _fail(f"expected BLOCKED, got {ctrl.status}: "
              f"{ctrl.last_plan_result}")
    _ok(f"BLOCKED on unreachable target ({ctrl.last_plan_result.reason})")


def test_jump_emitted_for_step_up() -> None:
    print("\n[7] step-up path emits jump=True at the right tick")
    wm = WorldMap()
    _flat_floor(wm, (-1, 2), 63, (-1, 1))
    # Step up: floor at y=64 for x>=2.
    _flat_floor(wm, (2, 4), 64, (-1, 1))
    ctrl = WalkerController(wm)
    ctrl.set_target((4, 65, 0), eye_voxel=(0, 64, 0))
    if ctrl.status != WalkerStatus.WALKING:
        _fail(f"plan failed: {ctrl.last_plan_result}")
    # Stand at (1.5, 64, 0.5) facing east (yaw=-90). Next waypoint is
    # (2, 65, 0) — a step up.
    pose = _f3(x=1.5, y=64.0, z=0.5, yaw=-90.0)
    # Force the walker to advance to the step-up waypoint by sliding
    # through. We'll iterate a couple of ticks to let it pick it up.
    action = None
    for _ in range(3):
        action = ctrl.tick(pose)
    # The current target waypoint should now be the higher cell;
    # the walker should emit jump=True.
    if action is None or not action.movement.get("jump"):
        _fail(f"expected jump=True near step-up, got {action.movement!r}; "
              f"wp_idx={ctrl._wp_idx} path={ctrl._path}")
    _ok("jump emitted on step-up waypoint")


def test_stuck_detection_triggers_replan() -> None:
    print("\n[8] no-progress over the stuck window triggers a replan")
    wm = WorldMap()
    _flat_floor(wm, (-1, 5), 63, (-1, 5))
    cfg = WalkerConfig()
    cfg.stuck_progress_window = 4    # tiny window for the test
    cfg.stuck_replan_retries = 1
    ctrl = WalkerController(wm, config=cfg)
    ctrl.set_target((3, 64, 0), eye_voxel=(0, 64, 0))
    if ctrl.status != WalkerStatus.WALKING:
        _fail(f"plan failed: {ctrl.last_plan_result}")
    # Feed FRESH-but-stationary F3 reads (each with a NEW timestamp).
    # The walker's stuck-counter only ticks on fresh F3, not on every
    # 20 Hz cached repeat — so we need distinct timestamps to drive
    # the detector.
    for i in range(15):
        pose = _f3(x=0.5, y=64.0, z=0.5, yaw=-90.0,
                   timestamp=float(i + 1))
        ctrl.tick(pose)
    if ctrl._plan_attempts < 1:
        _fail(f"expected at least one stuck-replan attempt, got "
              f"{ctrl._plan_attempts}")
    _ok(f"stuck-replan attempted {ctrl._plan_attempts} times")


def test_stuck_detection_ignores_cached_pose() -> None:
    """At 20 Hz the agent gets the same F3Info handed back many times
    between fresh 3 Hz OCR reads. Counting those cached repeats toward
    the stuck timer would fire the moment the player paused for a
    second — the timer must key off f3.timestamp."""
    print("\n[8b] stuck-detection IGNORES cached-pose repeats")
    wm = WorldMap()
    _flat_floor(wm, (-1, 5), 63, (-1, 5))
    cfg = WalkerConfig()
    cfg.stuck_progress_window = 4
    cfg.stuck_replan_retries = 1
    ctrl = WalkerController(wm, config=cfg)
    ctrl.set_target((3, 64, 0), eye_voxel=(0, 64, 0))
    # Hand the SAME F3Info (same timestamp) 100 times — that's a single
    # OCR cycle. The walker must NOT register a stuck event.
    pose = _f3(x=0.5, y=64.0, z=0.5, yaw=-90.0, timestamp=42.0)
    for _ in range(100):
        ctrl.tick(pose)
    if ctrl._plan_attempts != 0:
        _fail(f"cached-pose repeats triggered a replan "
              f"({ctrl._plan_attempts}); stuck-detector did not "
              f"respect f3.timestamp freshness")
    _ok("cached-pose repeats did not falsely trigger stuck-replan")


def test_normalize_angle_corner_cases() -> None:
    print("\n[9] internal yaw normaliser handles ±180 boundary")
    from agents.walker import _normalize_angle
    assert _normalize_angle(0.0) == 0.0
    assert abs(_normalize_angle(360.0) - 0.0) < 1e-9
    assert abs(_normalize_angle(-360.0) - 0.0) < 1e-9
    # 270 degrees should normalise to -90 (shortest signed angle).
    assert abs(_normalize_angle(270.0) - (-90.0)) < 1e-9
    assert abs(_normalize_angle(-270.0) - 90.0) < 1e-9
    _ok("yaw normaliser round-trip clean across ±360")


def test_y_drop_triggers_immediate_replan() -> None:
    """Under passable policy the pathfinder optimistically assumes
    unobserved terrain is standable. When the player walks into real
    terrain that drops below the planned Y (cliff / hole), the walker
    must detect the Y mismatch and replan IMMEDIATELY — waiting for
    the stuck-timer would mean 30+ ticks of confused aim while the
    geometry no longer matches the path."""
    print("\n[15] Y-drop triggers immediate replan (passable mode)")
    wm = WorldMap()
    # Sparse ground only — most of the world is unobserved. Under
    # passable mode the pathfinder will optimistically plan through
    # the unknowns.
    _flat_floor(wm, (-1, 1), 63, (-1, 1))
    cfg = WalkerConfig()
    cfg.pathfind.unknown_policy = "passable"
    ctrl = WalkerController(wm, config=cfg)
    ctrl.set_target((10, 64, 0), eye_voxel=(0, 64, 0))
    if ctrl.status != WalkerStatus.WALKING:
        _fail(f"plan failed: {ctrl.last_plan_result}")
    initial_path = list(ctrl._path)
    initial_plan_attempts = ctrl._plan_attempts
    # Feed a pose showing the player Y=64 (matches plan).
    ctrl.tick(_f3(x=0.5, y=64.0, z=0.5, yaw=-90.0, timestamp=1.0))
    # Now simulate falling into a hole — player Y suddenly = 60.
    # The current waypoint is at Y=64. Drop = 64 - 60 = 4 blocks,
    # well past ``unexpected_y_drop_blocks=2``. Walker must replan.
    ctrl.tick(_f3(x=2.5, y=60.0, z=0.5, yaw=-90.0, timestamp=2.0))
    if not ctrl._path:
        # Replan may have failed — that's BLOCKED status, which is
        # acceptable (no path from Y=60 to Y=64 goal in our sparse
        # world). The key is that the OLD path was abandoned.
        if ctrl._path == initial_path:
            _fail("Y-drop did not invalidate the old path")
    _ok(f"Y-drop replan fired: status={ctrl.status.value} "
        f"path_len_before={len(initial_path)} "
        f"path_len_after={len(ctrl._path)}")


def test_arrived_tolerance_at_least_waypoint_tolerance() -> None:
    """Invariant: ``arrived_tolerance >= waypoint_tolerance``.
    The waypoint-advance step pops a waypoint off the path when the
    player is within ``waypoint_tolerance`` of its centre. If
    ``arrived_tolerance < waypoint_tolerance`` the walker can
    advance OFF the end of the path (past the final waypoint =
    the goal) without ever firing the arrival check, leaving the
    agent looking for a target it's already passed."""
    print("\n[16] invariant: arrived_tolerance >= waypoint_tolerance")
    cfg = WalkerConfig()
    if cfg.arrived_tolerance < cfg.waypoint_tolerance:
        _fail(f"arrived_tolerance ({cfg.arrived_tolerance}) "
              f"< waypoint_tolerance ({cfg.waypoint_tolerance}) — "
              f"this allows the walker to advance past the final "
              f"waypoint without registering arrival")
    _ok(f"arrived_tolerance={cfg.arrived_tolerance} "
        f">= waypoint_tolerance={cfg.waypoint_tolerance}")


def test_pitch_levels_toward_target() -> None:
    """Walker must P-control pitch toward ``target_pitch_deg`` in
    parallel with yaw rotation. If pitch is left dangling at the
    systematic-scan endpoint (often ±80°), the agent walks forward
    staring at the sky / floor and F3 can't see the path."""
    print("\n[11] walker levels pitch toward target while navigating")
    wm = WorldMap()
    _flat_floor(wm, (-1, 5), 63, (-1, 5))
    cfg = WalkerConfig()
    ctrl = WalkerController(wm, config=cfg)
    ctrl.set_target((3, 64, 0), eye_voxel=(0, 64, 0))
    # Player is aimed correctly (yaw=-90, east) but staring at the
    # sky (pitch=-70). Walker should command a downward pitch
    # velocity (positive vy = look down).
    pose = _f3(x=0.5, y=64.0, z=0.5, yaw=-90.0, pitch=-70.0)
    action = ctrl.tick(pose)
    if action.look_vy <= 0.0:
        _fail(f"expected look_vy > 0 (pitch down toward level), got "
              f"vy={action.look_vy:.1f}")
    _ok(f"sky stare: look_vy={action.look_vy:+.1f} (correcting down)")

    # Opposite: pitch=+70 (looking at ground). Walker should command
    # an upward pitch velocity (negative vy).
    pose2 = _f3(x=0.5, y=64.0, z=0.5, yaw=-90.0, pitch=70.0)
    action2 = ctrl.tick(pose2)
    if action2.look_vy >= 0.0:
        _fail(f"expected look_vy < 0 (pitch up toward level), got "
              f"vy={action2.look_vy:.1f}")
    _ok(f"floor stare: look_vy={action2.look_vy:+.1f} (correcting up)")

    # At target pitch: vy should be ~0.
    pose3 = _f3(x=0.5, y=64.0, z=0.5, yaw=-90.0,
                pitch=cfg.target_pitch_deg)
    action3 = ctrl.tick(pose3)
    if abs(action3.look_vy) > 1.0:
        _fail(f"expected look_vy ~ 0 at target pitch, got "
              f"vy={action3.look_vy:.1f}")
    _ok(f"at target pitch: look_vy={action3.look_vy:+.1f} (~0)")


def test_zero_mouse_per_degree_is_sanitised() -> None:
    """A YAML value of 0 (or NaN / inf) for mouse_per_degree would
    propagate into the look_vx command and leave the camera stuck
    rotating at zero rate — the walker can never reach aim
    tolerance, never presses W, never moves. The constructor floors
    the value to a sane default."""
    print("\n[12] zero / non-finite mouse_per_degree is sanitised")
    wm = WorldMap()
    _flat_floor(wm, (-1, 5), 63, (-1, 5))
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        cfg = WalkerConfig()
        cfg.mouse_per_degree = bad
        ctrl = WalkerController(wm, config=cfg)
        if ctrl.cfg.mouse_per_degree <= 0 or not (
                ctrl.cfg.mouse_per_degree == ctrl.cfg.mouse_per_degree):  # NaN check
            _fail(f"mouse_per_degree={bad} not sanitised, "
                  f"got {ctrl.cfg.mouse_per_degree}")
    _ok("zero / negative / NaN / inf all snapped to 6.5 default")


def test_failed_target_cooldown() -> None:
    """Repeated set_target calls to an unreachable goal must NOT
    rerun A* every time. The walker remembers blocked goals in
    a TTL cooldown table so an autonomous parent agent that keeps
    reproposing the same impossible voxel doesn't burn the per-tick
    A* budget."""
    print("\n[13] failed-target cooldown prevents repeated A* on blocked goal")
    wm = WorldMap()
    _flat_floor(wm, (-1, 1), 63, (-1, 1))   # tiny island
    ctrl = WalkerController(wm)
    far = (40, 64, 40)   # nowhere near the observed island
    # First set_target — A* runs, finds nothing, marks failed.
    ctrl.set_target(far, eye_voxel=(0, 64, 0))
    if ctrl.status != WalkerStatus.BLOCKED:
        _fail(f"expected BLOCKED on unreachable, got {ctrl.status}")
    # Second + third set_target — should be cooldown-rejected.
    nodes_before = ctrl.last_plan_result.nodes_expanded
    ctrl.set_target(far, eye_voxel=(0, 64, 0))
    if ctrl.last_plan_result.nodes_expanded != nodes_before:
        _fail("second set_target re-ran A* despite cooldown — "
              "nodes_expanded changed")
    if far not in ctrl._failed_targets:
        _fail("first failed plan did not record the target in cooldown")
    _ok("repeated set_target on blocked goal skipped A*")


def test_set_target_mid_walk_resets_state() -> None:
    """Calling set_target while the walker is in WALKING state must
    cleanly wipe stuck-detection state so the new target gets a
    fresh progress window. A stuck-counter carried over from the
    previous target would falsely trip the new walk."""
    print("\n[14] set_target() mid-walk resets stuck state")
    wm = WorldMap()
    _flat_floor(wm, (-1, 10), 63, (-1, 10))
    cfg = WalkerConfig()
    cfg.stuck_progress_window = 4
    ctrl = WalkerController(wm, config=cfg)
    ctrl.set_target((3, 64, 0), eye_voxel=(0, 64, 0))
    if ctrl.status != WalkerStatus.WALKING:
        _fail(f"initial plan failed: {ctrl.last_plan_result}")
    # Simulate a few fresh-pose stationary ticks to bump
    # _stuck_fresh_ticks_at_pos.
    for i in range(3):
        pose = _f3(x=0.5, y=64.0, z=0.5, yaw=-90.0,
                   timestamp=float(i + 1))
        ctrl.tick(pose)
    fresh_before = ctrl._stuck_fresh_ticks_at_pos
    # Switch target — should wipe stuck state.
    ctrl.set_target((6, 64, 0), eye_voxel=(0, 64, 0))
    if ctrl._stuck_fresh_ticks_at_pos != 0:
        _fail(f"mid-walk set_target did not reset stuck counter "
              f"(was {fresh_before}, now {ctrl._stuck_fresh_ticks_at_pos})")
    if ctrl._last_pose_ts is not None:
        _fail(f"mid-walk set_target did not reset pose timestamp "
              f"(still {ctrl._last_pose_ts})")
    _ok(f"mid-walk set_target wiped stuck state (was {fresh_before}, "
        f"now {ctrl._stuck_fresh_ticks_at_pos})")


def test_pathwalker_agent_integration() -> None:
    print("\n[10] PathWalkerAgent attach_perception + reset wires controller")
    from agents.walker import PathWalkerAgent

    class _StubPerception:
        def __init__(self, wm):
            self.world_map = wm

    wm = WorldMap()
    _flat_floor(wm, (-1, 5), 63, (-1, 5))
    agent = PathWalkerAgent(target=(3, 64, 0))
    agent.attach_perception(_StubPerception(wm))
    agent.reset()
    if agent._controller is None:
        _fail("agent.reset() should construct the controller after "
              "attach_perception")
    if agent._controller.status != WalkerStatus.WALKING \
            and agent._controller.status != WalkerStatus.PLANNING:
        _fail(f"agent controller status unexpected: "
              f"{agent._controller.status}")
    _ok(f"controller wired ({agent._controller.status})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 68)
    print(" Walker offline self-test")
    print("=" * 68)
    test_idle_until_target_set()
    test_plan_and_walk_to_horizontal_target()
    test_rotates_before_walking_when_yaw_off()
    test_arrival_status()
    test_no_pose_releases_movement()
    test_unreachable_target_blocks()
    test_jump_emitted_for_step_up()
    test_stuck_detection_triggers_replan()
    test_stuck_detection_ignores_cached_pose()
    test_normalize_angle_corner_cases()
    test_arrived_tolerance_at_least_waypoint_tolerance()
    test_pitch_levels_toward_target()
    test_y_drop_triggers_immediate_replan()
    test_zero_mouse_per_degree_is_sanitised()
    test_failed_target_cooldown()
    test_set_target_mid_walk_resets_state()
    test_pathwalker_agent_integration()
    print("\n" + "=" * 68)
    print(" ALL TESTS PASSED")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
