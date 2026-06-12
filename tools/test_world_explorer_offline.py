#!/usr/bin/env python3
"""
Offline self-test for the world-explorer + perception pipeline.

Runs the same agent + perception classes the live agent uses, but feeds
synthetic frames and F3 readings instead of real screen captures. The
goal is to verify, without Minecraft running:

  * Parser path:  Targeted-Block lines -> LookingAtBlock -> WorldMap commit.
  * Logging path: per-confirmation [perception] LOGGED prints.
  * Debug path:   [F3] dump fires on every fresh OCR reading.
  * Motion path:  agent emits pitch-locked sweep velocities (vy=0).
  * Startup path: open-loop scan does NOT pitch-drift before first pose.

If any assertion fails the script exits non-zero with a clear message —
suitable for a CI smoke test.

Run:
    python tools/test_world_explorer_offline.py
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from agents.world_explorer import (
    WorldExplorerAgent,
    build_world_explorer_agent,
)
from control.mouse_calibration import MouseCalibrator, MouseCalibrationConfig
from brain.interfaces import AgentAction
from vision.ocr import F3Info
from vision.processing import GameState, ScreenState
from vision.world import build_world_perception
from vision.world.f3_target import parse_looking_at_block


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings() -> dict:
    """Load real settings.yaml so the test exercises the production
    config the live agent will actually use."""
    with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
        settings = yaml.safe_load(f) or {}
    # These sub-tests exercise the COMMIT/EXPLORE pipeline, not the
    # multi-frame + crosshair-ray confirmation gates (those are covered by
    # test_world_perception's confirm_gate). Disable them here so a single
    # synthetic read commits as the pipeline assertions expect.
    settings.setdefault("vision", {}).setdefault("world", {}).update(
        {"min_confirm_reads": 1, "ray_consistency_max_dist": 0.0})
    return settings


def _build_agent_sandboxed(settings: dict) -> WorldExplorerAgent:
    """
    Build a WorldExplorerAgent whose calibrator does NOT persist to
    ``data/calibration/mouse_calibration.json``. The default builder
    wires the real persist path, which means every synthetic pose this
    test feeds the agent would corrupt the live calibration the user
    relies on between sessions.

    We swap in an ephemeral calibrator AFTER the builder has applied
    settings -> cfg, so all the behavioural knobs (gains, tolerances,
    sweep config) are still picked up from settings.yaml unchanged.
    """
    agent = build_world_explorer_agent(settings)
    agent.calibrator = MouseCalibrator(
        cfg=MouseCalibrationConfig(
            default_px_per_deg=agent.cfg.mouse_per_degree,
        ),
        persist_path=None,    # in-memory only
    )
    return agent


def _f3(*, x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0,
        block_x=0, block_y=64, block_z=0,
        looking_at_id: str | None = "minecraft:dirt",
        looking_at_pos: tuple[int, int, int] | None = (0, 64, 0),
        timestamp: float = 0.0) -> F3Info:
    """Build an F3Info exactly the way the live OCR pipeline does:
    raw_text is the multi-line text the parser walks, every other
    field is the post-parse structured result."""
    lines = [
        f"XYZ: {x:.3f} / {y:.3f} / {z:.3f}",
        f"Block: {int(x)} {int(y)} {int(z)}",
        f"Facing: south (Towards positive Z) ({yaw:.1f} / {pitch:.1f})",
    ]
    if looking_at_id is not None and looking_at_pos is not None:
        lines.append(
            f"Targeted Block: {looking_at_pos[0]}, "
            f"{looking_at_pos[1]}, {looking_at_pos[2]}")
        lines.append(looking_at_id)
    return F3Info(
        x=x, y=y, z=z,
        yaw=yaw, pitch=pitch,
        facing_name="south", dimension="minecraft:overworld",
        block_x=block_x, block_y=block_y, block_z=block_z,
        timestamp=timestamp,
        raw_text="\n".join(lines),
    )


def _frame(h: int = 1094, w: int = 1920) -> np.ndarray:
    """Solid dirt-brown frame. Perception only uses it for sampling
    and screen-ray intrinsics; we don't care what's painted on it."""
    return np.full((h, w, 3), (96, 64, 48), dtype=np.uint8)


def _fail(msg: str) -> None:
    print(f"\n[FAIL] {msg}")
    sys.exit(1)


def _ok(msg: str) -> None:
    print(f"  [ok] {msg}")


# ---------------------------------------------------------------------------
# Test 1: parser handles every MC F3 layout we expect
# ---------------------------------------------------------------------------

def test_parser_layouts() -> None:
    print("\n[1/5] parse_looking_at_block — F3 layout variants")
    cases = [
        # MC 1.20+: separate lines for label + id
        ("modern multi-line",
         ["Targeted Block: 12, 64, -7", "minecraft:stone"],
         "minecraft:stone", (12, 64, -7)),
        # Pre-1.20: single line
        ("legacy single-line",
         ["Targeted Block: 12, 64, -7  minecraft:stone"],
         "minecraft:stone", (12, 64, -7)),
        # "Looking at block:" variant
        ("Looking at block label",
         ["Looking at block: -5, 70, 3", "minecraft:grass_block"],
         "minecraft:grass_block", (-5, 70, 3)),
        # Tag lines interleaved
        ("modern with tag lines",
         ["Targeted Block: 100, 64, 50",
          "minecraft:oak_log",
          "axis: y",
          "#minecraft:mineable/axe",
          "#minecraft:logs"],
         "minecraft:oak_log", (100, 64, 50)),
        # Single decimal-coords-on-Targeted-line (older release)
        ("coords with commas + spaces",
         ["Targeted Block: -110,  64,   200",
          "minecraft:cobblestone"],
         "minecraft:cobblestone", (-110, 64, 200)),
    ]
    for name, lines, expect_id, expect_pos in cases:
        la = parse_looking_at_block(lines)
        if la is None:
            _fail(f"{name}: parser returned None")
        if la.block_id != expect_id:
            _fail(f"{name}: id={la.block_id!r}, expected {expect_id!r}")
        if la.pos != expect_pos:
            _fail(f"{name}: pos={la.pos}, expected {expect_pos}")
        _ok(f"{name}: {la.block_id} @ {la.pos}")

    # And negative cases
    no_match = [
        ("empty input", []),
        ("XYZ only, no targeted line",
            ["XYZ: 0 / 64 / 0", "Block: 0 64 0", "Facing: south"]),
        ("Targeted label but no coords or id",
            ["Targeted Block:", "Facing: south"]),
    ]
    for name, lines in no_match:
        la = parse_looking_at_block(lines)
        if la is not None:
            _fail(f"negative {name}: parser returned {la!r}")
        _ok(f"negative {name}: correctly returned None")


# ---------------------------------------------------------------------------
# Test 2: perception commits + logs on F3 confirmation
# ---------------------------------------------------------------------------

def test_perception_commit_and_log() -> None:
    print("\n[2/5] perception.update — commit + LOG line + F3 dump")
    settings = _settings()
    wp = build_world_perception(settings)
    frame = _frame()

    # Capture stdout to verify the [F3] dump and [perception] LOGGED prints.
    # The target (5,64,10) must lie ON the crosshair ray (the bot stands at
    # x=5.5,z=7.5 looking straight +Z at it) and be read on >=2 fresh frames
    # — both required now by the multi-frame + ray confirmation gates.
    buf = io.StringIO()
    with redirect_stdout(buf):
        for ts in (1.0, 2.0):
            wp.update(frame, _f3(
                x=5.5, y=63.0, z=7.5, yaw=0.0, pitch=0.0,
                looking_at_id="minecraft:dirt",
                looking_at_pos=(5, 64, 10),
                timestamp=ts,
            ))
    out = buf.getvalue()

    if "[F3] tick=" not in out:
        _fail("debug_f3_dump line did not appear in stdout")
    if "[perception] LOGGED dirt @ (5, 64, 10)" not in out:
        _fail(f"LOGGED line missing.  stdout was:\n{out!r}")
    if wp.world_map.get_block((5, 64, 10)) is None:
        _fail("block not committed to WorldMap")
    _ok("first confirmation: dirt committed at (5, 64, 10)")
    _ok("F3 dump line present")
    _ok("LOGGED heartbeat present")

    # Another read of the SAME voxel: already confirmed -> must NOT re-LOG.
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        wp.update(frame, _f3(
            x=5.5, y=63.0, z=7.5, yaw=0.0, pitch=0.0,
            looking_at_id="minecraft:dirt",
            looking_at_pos=(5, 64, 10),
            timestamp=3.0,
        ))
    out2 = buf2.getvalue()
    if "LOGGED" in out2:
        _fail(f"LOGGED fired twice for the same voxel.  out:\n{out2!r}")
    _ok("repeat confirmation of same voxel does NOT re-log")

    # A NEW voxel (stone at (5,63,11)) on the crosshair ray, read on 2 fresh
    # frames -> confirms + LOGs.
    buf3 = io.StringIO()
    with redirect_stdout(buf3):
        for ts in (4.0, 5.0):
            wp.update(frame, _f3(
                x=5.5, y=63.0, z=9.0, yaw=0.0, pitch=24.0,
                looking_at_id="minecraft:stone",
                looking_at_pos=(5, 63, 11),
                timestamp=ts,
            ))
    out3 = buf3.getvalue()
    if "LOGGED stone @ (5, 63, 11)" not in out3:
        _fail(f"second voxel did not LOG.  out:\n{out3!r}")
    _ok("second confirmation: stone logged at (5, 63, 11)")

    # World map should now have two confirmed blocks.
    confirmed = wp.stats()["confirmed_count"]
    if confirmed != 2:
        _fail(f"expected confirmed_count=2, got {confirmed}")
    _ok(f"confirmed_count={confirmed} after two distinct confirmations")


# ---------------------------------------------------------------------------
# Test 3: F3 dump fires only on FRESH reads (not on cached repeats)
# ---------------------------------------------------------------------------

def test_f3_dump_freshness() -> None:
    print("\n[3/5] F3 dump throttled to fresh OCR timestamps")
    settings = _settings()
    wp = build_world_perception(settings)
    frame = _frame()
    same = _f3(timestamp=42.0)

    buf = io.StringIO()
    with redirect_stdout(buf):
        for _ in range(7):       # 7 ticks reusing the same F3 (as main.py does between OCR calls)
            wp.update(frame, same)
    dumps = buf.getvalue().count("[F3] tick=")
    if dumps != 1:
        _fail(f"expected 1 [F3] dump for 7 repeated reads, got {dumps}")
    _ok(f"{dumps} dump line for 7 cached reads — correct")

    # Now a fresh timestamp: should dump again.
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        wp.update(frame, _f3(timestamp=43.0))
    if "[F3] tick=" not in buf2.getvalue():
        _fail("fresh F3 did not produce a new dump line")
    _ok("fresh OCR timestamp produces a new dump")


# ---------------------------------------------------------------------------
# Test 4: agent emits pitch-LOCKED sweeps (vy=0)
# ---------------------------------------------------------------------------

def test_agent_pitch_locked_sweep() -> None:
    print("\n[4/5] agent sweep velocity is pitch-LOCKED (vy=0)")
    settings = _settings()
    agent = _build_agent_sandboxed(settings)
    wp = build_world_perception(settings)
    agent.attach_perception(wp)
    agent.reset()

    # Drive past the orient + initial settle by jumping the agent
    # directly into the SWEEP_LEVEL phase with a pose already at
    # the target. We monkey-patch the state directly.
    agent._sys_phase_idx = 1   # sweep_level
    agent._sweep_yaw_start = 0.0
    agent._sweep_yaw_unwrapped = 0.0
    agent._sweep_prev_yaw = 0.0

    pose = _f3(x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0, timestamp=100.0)
    state = GameState(
        frame=_frame(64, 64),  # small dummy
        frame_raw=None,
        timestamp=100.0,
        screen_state=ScreenState.PLAYING,
        f3=pose,
    )

    # Suppress agent log spam.
    with redirect_stdout(io.StringIO()):
        action = agent.decide(state)

    if not isinstance(action, AgentAction):
        _fail(f"decide() returned {type(action)}, expected AgentAction")
    if action.look_vy != 0.0:
        _fail(f"sweep emitted non-zero vy={action.look_vy}; expected vy=0 "
              f"(pitch lock).  full action: {action!r}")
    if action.look_vx == 0.0:
        _fail("sweep emitted vx=0; expected non-zero yaw rate")
    if not action.force_velocity:
        _fail("sweep did not set force_velocity=True")
    _ok(f"sweep velocity: vx={action.look_vx:+.1f}, vy={action.look_vy:+.1f}, "
        f"force_velocity={action.force_velocity}")


# ---------------------------------------------------------------------------
# Test 5: open-loop scan no longer drifts pitch at startup
# ---------------------------------------------------------------------------

def test_open_loop_no_startup_drift() -> None:
    print("\n[5/5] open-loop scan does NOT pitch-drift before first pose")
    settings = _settings()
    agent = _build_agent_sandboxed(settings)
    wp = build_world_perception(settings)
    agent.attach_perception(wp)
    agent.reset()

    # Simulate the startup pose-less window: ~50 ticks with f3=None.
    state = GameState(
        frame=_frame(64, 64),
        frame_raw=None,
        timestamp=0.0,
        screen_state=ScreenState.PLAYING,
        f3=None,
    )

    pitch_vy_samples = []
    with redirect_stdout(io.StringIO()):
        for _ in range(50):
            action = agent.decide(state)
            pitch_vy_samples.append(action.look_vy)
    if any(abs(vy) > 1e-9 for vy in pitch_vy_samples):
        _fail(f"open_loop_scan emitted non-zero pitch velocity before "
              f"first pose: {pitch_vy_samples[:5]}... "
              f"This was the bug that pitched the camera 16° down at "
              f"agent startup. It MUST stay zero.")
    _ok("50 pre-baseline ticks: all look_vy = 0.0 (no startup drift)")

    # Now hand over a real pose. The agent should transition to systematic
    # scan and emit motion commands that include pitch correction since
    # phase=orient and current pitch=0 already.
    state_with_pose = GameState(
        frame=_frame(),
        frame_raw=None,
        timestamp=1.0,
        screen_state=ScreenState.PLAYING,
        f3=_f3(x=0.0, y=64.0, z=0.0, yaw=10.0, pitch=20.0, timestamp=1.0),
    )
    with redirect_stdout(io.StringIO()):
        action = agent.decide(state_with_pose)
    # In orient with pitch=20 -> target=0, we expect vy < 0 (move up).
    if action.look_vy >= 0.0:
        _fail(f"orient with pitch=20 should command UP (negative vy), "
              f"got vy={action.look_vy}")
    _ok(f"orient with pitch=20 commands vy={action.look_vy:+.1f} (UP toward 0)")


# ---------------------------------------------------------------------------
# Test 6: orient phase converges (closed-loop simulation)
# ---------------------------------------------------------------------------

def test_orient_converges_in_budget() -> None:
    """
    Closed-loop simulation of the ORIENT phase:

      pitch_now ← pitch_now + (commanded_vy / px_per_deg_pitch) × tick_period

    This catches the bug from the live run: orient was hitting its
    80-tick timeout with pitch still at 6° because the P-gain was too
    gentle (0.7) and the agent was fighting open-loop pitch drift from
    startup. With the new settle_p_gain (1.6) + settle_max_deg_per_sec
    (60) + zero startup drift, orient should reach the 2° tolerance
    well before the timeout — for a wide range of starting pitches
    AND for a 4× miscalibration of pitch_per_degree (worst-case in
    practice when the calibrator has bad historical samples).
    """
    print("\n[6/6] orient phase converges within budget (closed-loop)")
    settings = _settings()

    for start_pitch in (16.0, -16.0, 30.0, -30.0, 60.0, -60.0):
        for cal_skew in (1.0, 2.0, 4.0):
            agent = _build_agent_sandboxed(settings)
            wp = build_world_perception(settings)
            agent.attach_perception(wp)
            agent.reset()
            tick_period = agent.tick_period_sec    # 0.05 s at 20 Hz

            # Simulate calibrator returning a SKEWED value: the agent
            # commands px/sec, but the world only delivers
            # 1/cal_skew degrees per commanded pixel. cal_skew=4 means
            # agent thinks 1 px = X°, but reality is 1 px = X°/4.
            true_px_per_deg = (
                agent.calibrator.px_per_deg_pitch() * cal_skew
            )

            pitch_now = start_pitch
            yaw_now = 0.0
            tick = 0
            settled_at = None
            timeout_ticks = agent.cfg.systematic_settle_max_ticks
            timestamp = 0.0
            with redirect_stdout(io.StringIO()):
                while tick < timeout_ticks + 5:
                    timestamp += tick_period
                    state = GameState(
                        frame=_frame(64, 64),
                        frame_raw=None,
                        timestamp=timestamp,
                        screen_state=ScreenState.PLAYING,
                        f3=_f3(yaw=yaw_now, pitch=pitch_now,
                                timestamp=timestamp),
                    )
                    action = agent.decide(state)
                    # Apply velocity -> pose change for this tick.
                    dpitch_deg = (action.look_vy / true_px_per_deg
                                  * tick_period) if true_px_per_deg > 0 else 0
                    dyaw_deg = (action.look_vx / true_px_per_deg
                                * tick_period) if true_px_per_deg > 0 else 0
                    pitch_now = max(-90.0, min(90.0, pitch_now + dpitch_deg))
                    yaw_now = ((yaw_now + dyaw_deg + 180.0) % 360.0) - 180.0
                    # Detect orient -> next phase transition.
                    if agent._sys_phase_idx > 0 and settled_at is None:
                        settled_at = tick
                        break
                    tick += 1

            if settled_at is None:
                _fail(f"start_pitch={start_pitch:+.0f}° cal_skew={cal_skew}× "
                      f"— orient NEVER advanced past phase 0 in {tick} ticks. "
                      f"final pitch={pitch_now:.2f}°")
            else:
                _ok(f"start_pitch={start_pitch:+.0f}° cal_skew={cal_skew}× "
                    f"settled in {settled_at} ticks "
                    f"({settled_at * tick_period:.2f}s), "
                    f"final pitch={pitch_now:+.2f}°")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test 7: SCAN -> INVESTIGATE -> confirm round-trip
# ---------------------------------------------------------------------------

def test_investigate_lifecycle_confirm() -> None:
    """
    With ``scan_only_mode=False`` and the systematic plan marked done,
    the agent should:
      * enter INVESTIGATE when the curiosity queue has reachable
        voxels after ``scan_ticks_before_investigate`` SCAN ticks;
      * aim at the target via the velocity P-controller;
      * detect ``perception.is_confirmed(target)`` and return to SCAN.
    """
    print("\n[7] SCAN -> INVESTIGATE -> CONFIRM round-trip")
    settings = _settings()
    agent = _build_agent_sandboxed(settings)
    wp = build_world_perception(settings)
    agent.attach_perception(wp)
    agent.reset()
    agent.cfg.scan_only_mode = False
    agent._sys_done = True   # skip the systematic intro

    # Plant a curiosity entry 3 blocks ahead of the player (yaw 0 = +Z).
    # Eye is at (0, 65.62, 0); voxel centred at (0, 65, 3) -> 2.41 blocks
    # away horizontally, ~2.45 with the eye-y offset. Well within the
    # 5-block reach.
    eye = (0.0, 65.62, 0.0)
    target_voxel = (0, 65, 3)
    wp._curiosity[target_voxel] = {
        "block_id": "minecraft:acacia_leaves",
        "confidence": 0.8,
        "seen_tick": 0,
    }

    # Tick the agent forward enough scan ticks to trigger investigate.
    scan_ticks_before = agent.cfg.scan_ticks_before_investigate
    pose = _f3(x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0, timestamp=1.0)
    state = GameState(
        frame=_frame(64, 64), frame_raw=None,
        timestamp=1.0, screen_state=ScreenState.PLAYING, f3=pose,
    )
    with redirect_stdout(io.StringIO()):
        for _ in range(scan_ticks_before + 2):
            agent.decide(state)

    if agent._mode != agent._INVESTIGATE:
        _fail(f"agent did not enter INVESTIGATE after "
              f"{scan_ticks_before+2} scan ticks. Mode={agent._mode!r}, "
              f"target_voxel={agent._target_voxel}")
    if agent._target_voxel != target_voxel:
        _fail(f"agent investigating wrong voxel: {agent._target_voxel} "
              f"(expected {target_voxel})")
    _ok(f"agent entered INVESTIGATE on {target_voxel}")

    # Now mark the target as confirmed (simulating F3 looking_at firing
    # while the agent was aiming) and tick once. Agent should return
    # to SCAN.
    wp._confirmed.add(target_voxel)
    with redirect_stdout(io.StringIO()):
        action = agent.decide(state)

    if agent._mode != agent._SCAN:
        _fail(f"agent did not return to SCAN after confirmation. "
              f"Mode={agent._mode!r}")
    if agent._target_voxel is not None:
        _fail(f"agent did not clear target after confirmation: "
              f"{agent._target_voxel}")
    if agent._n_investigated != 1:
        _fail(f"agent didn't increment n_investigated. "
              f"Got {agent._n_investigated}")
    if not (action.look_vx == 0.0 and action.look_vy == 0.0):
        _fail(f"action after confirm should be halt, got "
              f"vx={action.look_vx} vy={action.look_vy}")
    _ok(f"agent confirmed and returned to SCAN. "
        f"n_investigated={agent._n_investigated}")


def test_investigate_give_up_records_cooldown() -> None:
    """
    If F3 never confirms the target, the agent should GIVE UP after
    ``investigate_max_ticks`` and record the voxel in ``_failed_targets``
    so the same voxel isn't tried again immediately when the patch
    sweep re-adds it to the curiosity queue.
    """
    print("\n[8] INVESTIGATE GIVE-UP records cooldown")
    settings = _settings()
    agent = _build_agent_sandboxed(settings)
    wp = build_world_perception(settings)
    agent.attach_perception(wp)
    agent.reset()
    agent.cfg.scan_only_mode = False
    agent._sys_done = True

    # Plant an unreachable-but-in-distance target — far enough that
    # the agent's aim will never resolve to a confirmation, but within
    # ``max_reach_blocks`` so the agent attempts it. Use ``yaw=0,
    # pitch=0`` initial pose so the agent has to actually rotate to
    # face this voxel.
    target = (3, 65, 3)
    wp._curiosity[target] = {
        "block_id": "minecraft:mangrove_leaves",
        "confidence": 0.7,
        "seen_tick": 0,
    }

    pose = _f3(x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0, timestamp=1.0)
    state = GameState(
        frame=_frame(64, 64), frame_raw=None,
        timestamp=1.0, screen_state=ScreenState.PLAYING, f3=pose,
    )

    # Drive the agent past the SCAN burst + the entire investigate
    # budget without ever calling is_confirmed. Agent will eventually
    # log GIVE UP.
    total_ticks = (agent.cfg.scan_ticks_before_investigate
                    + agent.cfg.investigate_max_ticks
                    + agent.cfg.close_confirm_grace_ticks
                    + 50)
    with redirect_stdout(io.StringIO()):
        for _ in range(total_ticks):
            agent.decide(state)

    if target not in agent._failed_targets:
        _fail(f"GIVE UP / DROP did not record {target} in failed_targets. "
              f"failed_targets={agent._failed_targets!r}")
    _ok(f"GIVE-UP recorded {target} in failed_targets at tick "
        f"{agent._failed_targets[target]}")

    # Re-populate the curiosity queue with the same voxel. The agent
    # should SKIP it because the cooldown is active.
    wp._curiosity[target] = {
        "block_id": "minecraft:mangrove_leaves",
        "confidence": 0.7,
        "seen_tick": agent._tick,
    }
    with redirect_stdout(io.StringIO()):
        # Try a few times — the agent loops up to 16 candidates per
        # _maybe_enter_investigate call.
        for _ in range(agent.cfg.scan_ticks_before_investigate + 5):
            agent.decide(state)

    if agent._mode == agent._INVESTIGATE and agent._target_voxel == target:
        _fail(f"agent re-investigated the same failed voxel {target} "
              f"while it was still in cooldown")
    if target in wp._curiosity:
        # We don't strictly require the agent to remove it — only
        # to not investigate it. But if it's still there AND we're
        # not investigating it, that's correct.
        pass
    _ok(f"agent skipped failed target {target} during cooldown window")


def test_investigate_out_of_reach_discarded() -> None:
    """
    Curiosity voxels beyond ``max_reach_blocks`` from the eye should be
    silently dropped without entering INVESTIGATE. They can never
    produce an F3 confirmation no matter how long we aim.
    """
    print("\n[9] out-of-reach curiosity entries are discarded")
    settings = _settings()
    agent = _build_agent_sandboxed(settings)
    wp = build_world_perception(settings)
    agent.attach_perception(wp)
    agent.reset()
    agent.cfg.scan_only_mode = False
    agent._sys_done = True

    # 50 blocks ahead — well past MC's 4.5-5 block reach.
    far_target = (0, 65, 50)
    wp._curiosity[far_target] = {
        "block_id": "minecraft:stone",
        "confidence": 0.9,
        "seen_tick": 0,
    }

    pose = _f3(x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0, timestamp=1.0)
    state = GameState(
        frame=_frame(64, 64), frame_raw=None,
        timestamp=1.0, screen_state=ScreenState.PLAYING, f3=pose,
    )
    with redirect_stdout(io.StringIO()):
        for _ in range(agent.cfg.scan_ticks_before_investigate + 5):
            agent.decide(state)

    if agent._mode == agent._INVESTIGATE:
        _fail(f"agent entered INVESTIGATE on out-of-reach target "
              f"{far_target} at distance "
              f"{((50)**2 + 0.62**2) ** 0.5:.1f} blocks (reach="
              f"{agent.cfg.max_reach_blocks})")
    if far_target in wp._curiosity:
        _fail(f"out-of-reach target {far_target} stayed in queue "
              f"(expected to be popped + discarded)")
    _ok(f"out-of-reach target {far_target} popped and discarded "
        f"without entering INVESTIGATE")


def test_dimension_change_purges_curiosity() -> None:
    """
    Voxel coords are dimension-local: (12, 64, 8) in the overworld
    has no relationship to (12, 64, 8) in the nether. When the
    player crosses dimensions, the curiosity queue + confirmed-voxel
    set from the prior dimension are stale and must be cleared so
    the agent doesn't aim at phantom coords.
    """
    print("\n[13] dimension change purges stale curiosity / confirmed")
    settings = _settings()
    wp = build_world_perception(settings)
    frame = _frame()

    # Seed both sets while in the overworld.
    with redirect_stdout(io.StringIO()):
        wp.update(frame, _f3(
            x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0,
            looking_at_id="minecraft:stone",
            looking_at_pos=(0, 64, 1),
            timestamp=1.0,
        ))
    wp._curiosity[(2, 64, 2)] = {
        "block_id": "minecraft:dirt",
        "confidence": 0.7,
        "seen_tick": wp._tick,
    }
    if not wp._curiosity or not wp._confirmed:
        _fail("setup failed — curiosity or confirmed empty before "
              "dimension change")

    # Manually build a nether-dimension F3 reading.
    nether_f3 = _f3(
        x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0,
        looking_at_id=None, looking_at_pos=None,
        timestamp=2.0,
    )
    nether_f3.dimension = "minecraft:the_nether"

    buf = io.StringIO()
    with redirect_stdout(buf):
        wp.update(frame, nether_f3)
    out = buf.getvalue()

    if "DIMENSION CHANGE" not in out:
        _fail(f"dimension-change log line did not fire. stdout:\n{out!r}")
    if wp._curiosity:
        _fail(f"curiosity not purged after dimension change: "
              f"{wp._curiosity!r}")
    if wp._confirmed:
        _fail(f"confirmed not purged after dimension change: "
              f"{wp._confirmed!r}")
    _ok("dimension change purged curiosity + confirmed sets")


def test_curiosity_correction_tallied() -> None:
    """
    A curiosity entry is the patch sweep's guess that the commit gate
    rejected. When F3 later confirms a DIFFERENT block id at the same
    voxel, the disagreement is the cleanest "classifier was wrong"
    signal we get — bookkeep it under ``curiosity_corrections`` keyed
    by (guessed_block_id -> actual_block_id) so we can see which
    block ids the baseline classifier hallucinates most.
    """
    print("\n[12] curiosity-correction stat captured (guess -> actual)")
    settings = _settings()
    wp = build_world_perception(settings)
    frame = _frame()

    # Plant a curiosity entry: patch sweep "guessed" acacia_leaves
    # at (5, 64, 10). Then F3 confirms it's actually oak_leaves.
    guess_voxel = (5, 64, 10)
    wp._curiosity[guess_voxel] = {
        "block_id": "minecraft:acacia_leaves",
        "confidence": 0.8,
        "seen_tick": 1,
    }

    with redirect_stdout(io.StringIO()):
        wp.update(frame, _f3(
            x=4.5, y=63.0, z=10.5, yaw=0.0, pitch=10.0,
            looking_at_id="minecraft:oak_leaves",
            looking_at_pos=guess_voxel,
            timestamp=1.0,
        ))

    s = wp.stats()
    curio_cor = s.get("curiosity_corrections", {})
    expected_key = "minecraft:acacia_leaves->minecraft:oak_leaves"
    if curio_cor.get(expected_key) != 1:
        _fail(f"curiosity_corrections missing {expected_key!r}. "
              f"Got: {curio_cor!r}")
    _ok(f"recorded curio-correction {expected_key} x{curio_cor[expected_key]}")

    # If F3 confirms the SAME block id the curiosity guessed, no
    # correction should be tallied — patch sweep was right.
    wp._curiosity[(6, 64, 10)] = {
        "block_id": "minecraft:dirt",
        "confidence": 0.9,
        "seen_tick": 2,
    }
    with redirect_stdout(io.StringIO()):
        wp.update(frame, _f3(
            x=5.5, y=63.0, z=10.5, yaw=0.0, pitch=10.0,
            looking_at_id="minecraft:dirt",
            looking_at_pos=(6, 64, 10),
            timestamp=2.0,
        ))
    curio_cor2 = wp.stats().get("curiosity_corrections", {})
    # No NEW entry should appear.
    if len(curio_cor2) > 1:
        _fail(f"unexpected extra curio_correction entry: {curio_cor2!r}")
    _ok("correct curiosity guess produced no false correction entry")


def test_f3_confirm_autosaves_sample() -> None:
    """
    When F3 confirms a block under the crosshair, perception should
    auto-save a labelled patch to the sample store — this is the
    mechanism that grows the dataset over time. Without this firing,
    the per-block gate stays starved for new block ids forever.

    We swap in a temp sample store, drive ONE F3 confirmation, and
    assert the store now has one labelled sample for that block id.
    """
    print("\n[11] F3 confirmation auto-saves a labelled sample")
    import tempfile
    from pathlib import Path
    from vision.world.sample_store import WorldSampleStore

    settings = _settings()
    with tempfile.TemporaryDirectory() as td:
        store = WorldSampleStore(Path(td))
        wp = build_world_perception(settings)
        # Build wires its OWN sample_store from data/training/world_samples;
        # swap in the temp one so the test doesn't bleed into the real
        # training set.
        wp.sample_store = store

        frame = _frame()   # full-res 1920x1094 so the crosshair patch
                            # has real pixels to crop.
        before = store.total_samples()
        with redirect_stdout(io.StringIO()):
            wp.update(frame, _f3(
                x=4.5, y=63.0, z=10.5, yaw=0.0, pitch=10.0,
                looking_at_id="minecraft:acacia_leaves",
                looking_at_pos=(5, 64, 10),
                timestamp=1.0,
            ))
        after = store.total_samples()
        if after <= before:
            _fail(f"sample store did not grow on F3 confirm. "
                  f"before={before} after={after}")

        # Verify the sample landed in the acacia_leaves folder
        # specifically — wrong-folder bugs have happened before
        # when parser quirks bled into the labelling step.
        acacia_dir = Path(td) / "minecraft__acacia_leaves"
        if not acacia_dir.exists():
            # Some store implementations sanitise the prefix
            # differently; accept any directory with a colon-stripped
            # variant containing the block name.
            candidates = list(Path(td).iterdir())
            if not any("acacia_leaves" in c.name for c in candidates):
                _fail(f"no acacia_leaves directory created. "
                      f"contents={[c.name for c in candidates]}")
        _ok(f"sample store grew {before} -> {after} on one F3 confirm")


def test_failed_target_cooldown_expires() -> None:
    """
    After ``failed_target_ttl_ticks`` have passed since GIVE-UP, the
    cooldown expires and the voxel becomes eligible again. Prevents a
    transient occlusion (clouds, mobs, particles) from permanently
    blacklisting a voxel.
    """
    print("\n[10] failed_target cooldown expires after TTL")
    settings = _settings()
    agent = _build_agent_sandboxed(settings)
    wp = build_world_perception(settings)
    agent.attach_perception(wp)
    agent.reset()
    agent.cfg.scan_only_mode = False
    agent._sys_done = True
    agent.cfg.failed_target_ttl_ticks = 5   # tiny TTL for the test

    target = (0, 65, 3)
    # Manually inject a "stale" failure: recorded at tick 0; we're
    # going to advance _tick past the TTL.
    agent._failed_targets[target] = 0
    agent._tick = agent.cfg.failed_target_ttl_ticks + 2

    # Re-add the target to curiosity and verify the agent picks it up.
    wp._curiosity[target] = {
        "block_id": "minecraft:dirt",
        "confidence": 0.8,
        "seen_tick": agent._tick,
    }

    pose = _f3(x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0, timestamp=1.0)
    state = GameState(
        frame=_frame(64, 64), frame_raw=None,
        timestamp=1.0, screen_state=ScreenState.PLAYING, f3=pose,
    )
    with redirect_stdout(io.StringIO()):
        for _ in range(agent.cfg.scan_ticks_before_investigate + 5):
            agent.decide(state)

    if target in agent._failed_targets:
        _fail(f"expired failed-target entry {target} was not "
              f"garbage-collected. "
              f"failed_targets={agent._failed_targets!r}")
    if agent._mode != agent._INVESTIGATE or agent._target_voxel != target:
        _fail(f"agent did not pick up the now-eligible target "
              f"{target}. Mode={agent._mode!r}, "
              f"target_voxel={agent._target_voxel}")
    _ok(f"expired cooldown allowed re-investigation of {target}")


# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 68)
    print(" Offline world-explorer + perception self-test")
    print("=" * 68)
    test_parser_layouts()
    test_perception_commit_and_log()
    test_f3_dump_freshness()
    test_agent_pitch_locked_sweep()
    test_open_loop_no_startup_drift()
    test_orient_converges_in_budget()
    test_investigate_lifecycle_confirm()
    test_investigate_give_up_records_cooldown()
    test_investigate_out_of_reach_discarded()
    test_curiosity_correction_tallied()
    test_dimension_change_purges_curiosity()
    test_f3_confirm_autosaves_sample()
    test_failed_target_cooldown_expires()
    print("\n" + "=" * 68)
    print(" ALL TESTS PASSED")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
