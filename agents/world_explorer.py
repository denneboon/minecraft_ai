# agents/world_explorer.py
"""
World-exploration agent — pans the camera through a fixed set of
360° sweeps at varied pitches and lets the perception layer log every
block that F3's "Looking at block" overlay confirms under the crosshair.

Current behaviour (scan-only mode, default)
-------------------------------------------
The agent runs a fixed plan forever:

  1. ORIENT          — yaw=0, pitch=0    (run once at start)
  2. SWEEP_LEVEL     — full 360° at pitch=0
  3. SWEEP_DOWN      — full 360° at pitch=+45°
  4. SWEEP_FLOOR     — full 360° at pitch=+80°
  5. SWEEP_UP        — full 360° at pitch=-45°
  6. SWEEP_CEIL      — full 360° at pitch=-80°
  7. → back to phase 2 (level), loop indefinitely.

While the camera moves, ``vision.world.WorldPerception`` watches the F3
overlay every ~3 Hz. Whenever F3 reports a "Looking at block:
minecraft:X" line, perception commits a single observation to the
WorldMap with the block's exact coordinates (the integers F3 also
prints on the same line). NOTHING else commits — vision-patch
guesses, sample-NN matches, and inverse-renderer extrapolation are
all gated off via ``commit_only_from_looking_at: true`` in
settings.yaml. The map only ever contains blocks the AI saw
*directly under the crosshair*.

Why this matters for future learning
------------------------------------
Each F3 confirmation also writes a labelled screen patch to
``data/samples/<block_id>/`` (crosshair pixels inpainted out so they
don't contaminate the texture). Over time this builds a dataset of
``texture → block id`` pairs harvested at the AI's own viewpoint and
lighting.

The next phase (gated behind ``commit_only_from_looking_at: false``)
will let the agent identify blocks it ISN'T directly looking at:

  * For every visible voxel the WorldMap already knows about,
    :class:`vision.world.screen_ray.ScreenRay` projects its centre
    onto the current frame using ``(pose.x, pose.y, pose.z, yaw,
    pitch)`` and the calibrated camera intrinsics (FOV + aspect).
  * That screen patch is compared against the labelled samples via
    :class:`vision.world.sample_recognizer.SampleBlockRecognizer`.
  * Confident matches at non-crosshair locations get committed as
    ``source="extrapolation"`` observations — the agent has now
    "seen" dirt without ever centring the crosshair on it.

The infrastructure (ScreenRay, SampleBlockRecognizer,
HybridBlockClassifier, InverseRenderer) is already in place; the
present-phase agent intentionally does NOT use it so the dataset
stays clean.

Safety
------
The agent ONLY moves the mouse (camera). It never presses movement
keys, never clicks, never interacts. Even an untrained run can't
break anything in-game — at worst it spins in place.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from brain.interfaces import AgentAction, BaseAgent
from control.mouse_calibration import (
    MouseCalibrator, MouseCalibrationConfig,
)
from vision.processing import GameState, ScreenState


# ---------------------------------------------------------------------------
# Systematic scan phase plan
# ---------------------------------------------------------------------------
# When the agent starts (or resets), it works through a fixed sequence
# of camera poses to maximise F3 "Targeted Block" confirmations:
#
#   1. ORIENT — reset to yaw=0, pitch=0 (looking horizon-south).
#   2. SWEEP_LEVEL — full 360° yaw rotation at pitch=0.
#   3. PITCH 45° down + SWEEP_DOWN.
#   4. PITCH ~85° down (looking at floor) + SWEEP_FLOOR.
#   5. PITCH ~-45° (up toward sky) + SWEEP_UP.
#   6. PITCH ~-85° (looking straight up) + SWEEP_CEIL.
#
# After all phases complete, the agent drops into the reactive
# SCAN ↔ INVESTIGATE state machine that existed before.

@dataclass
class _ScanPhase:
    name: str
    target_pitch: float   # absolute pitch (degrees, MC convention: + = down)
    sweep_360: bool       # rotate yaw 360° at this pitch?


SYSTEMATIC_SCAN_PHASES: Tuple[_ScanPhase, ...] = (
    _ScanPhase("orient",      0.0, False),
    _ScanPhase("sweep_level", 0.0, True),
    _ScanPhase("sweep_down",  45.0, True),
    _ScanPhase("sweep_floor", 80.0, True),
    _ScanPhase("sweep_up",   -45.0, True),
    _ScanPhase("sweep_ceil", -80.0, True),
)


# Short plan used in REACTIVE mode (``scan_only_mode=False``). We only
# want a quick coverage pass to seed the WorldMap + curiosity queue
# before handing over to the SCAN ↔ INVESTIGATE loop. The full 6-phase
# plan takes ~5 minutes at the configured 30°/s yaw rate — way too long
# when the whole point is to exercise the curiosity loop. orient +
# one 360° sweep gives the player enough confirmations to seed
# extrapolation + populate the curiosity queue in ~30-60 s.
SYSTEMATIC_SCAN_PHASES_SHORT: Tuple[_ScanPhase, ...] = (
    _ScanPhase("orient",      0.0, False),
    _ScanPhase("sweep_level", 0.0, True),
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class WorldExplorerConfig:
    # Mouse-pixels per degree of yaw at MC sensitivity 100 %. Used
    # as a fallback when the auto-calibrator has no samples yet; the
    # learned value supersedes this as soon as we have enough data.
    mouse_per_degree: float = 6.5

    # Cap per-tick mouse delta (legacy one-shot easing path). Kept
    # for backwards compatibility — the agent now emits velocities
    # not per-tick deltas, but a few helpers still clamp here.
    max_mouse_dx: int = 45
    max_mouse_dy: int = 35

    # Per-tick correction fraction. A lower gain feels less twitchy:
    # the agent approaches its target with a few slow nudges rather
    # than one big jump. The downside is convergence is slower; 0.30
    # converges in ~3-4 ticks for typical yaw errors.
    mouse_gain: float = 0.30

    # ── Humanlike velocity-mode tuning ─────────────────────────────
    # All angular speeds in DEGREES PER SECOND. The agent converts
    # these to px/sec via the auto-calibrated px-per-degree.
    # Cap is intentionally low because F3 OCR runs at ~3 Hz (333 ms
    # between fresh-pose reads) — at 30°/s the camera moves at most
    # 10° between corrections, far less than a typical aim error,
    # so the controller can't overshoot and slam into the pitch
    # clamp before the next correction arrives.
    aim_max_deg_per_sec:    float = 30.0
    # While SCAN-mode panning, this is the constant yaw rate.
    # ~18°/s feels like a deliberate human glance, not a head shake.
    scan_yaw_deg_per_sec:   float = 18.0
    # Pitch sweep velocity.
    scan_pitch_deg_per_sec: float = 8.0
    # Investigate P-controller gain. 0.7/sec converges in ~1.5 s
    # for typical 10° errors — slow enough to never overshoot
    # between OCR reads, fast enough to feel responsive.
    investigate_p_gain:     float = 0.7

    # Systematic-settle P-controller gain. Used by orient + the
    # per-phase pitch-aim step. Larger than investigate_p_gain
    # because settle errors are usually big (e.g. swinging pitch
    # from 16° to 0° between phases), and we want to converge
    # within the settle-timeout budget instead of leaving the
    # camera mid-pitch and hoping the sweep self-corrects.
    settle_p_gain:          float = 1.6
    # Higher angular-velocity cap during settle than during
    # investigate aiming. Investigates target a specific voxel
    # and overshoot is bad; settles just need to reach a coarse
    # target pose quickly. 60°/s = full 16° pitch correction in
    # ~0.25 s under the new gain.
    settle_max_deg_per_sec: float = 60.0

    # Stop applying mouse correction when within this many degrees
    # of the target aim. Below this, F3 OCR will register the voxel
    # under the crosshair as the targeted block.
    aim_tolerance_deg: float = 1.4

    # Tolerance for the SYSTEMATIC settle step. Tighter than it used
    # to be (was 5°) because the sweep that follows is now pure-yaw —
    # if pitch is off when the sweep starts, it stays off for the
    # whole 360°, so we want it pretty close to target. 2° still
    # converges in a couple of ticks even with OCR jitter.
    systematic_settle_tolerance_deg: float = 2.0

    # During the 360° sweep we lock pitch (vy=0) so the camera turns
    # in a clean horizontal arc instead of wobbling diagonally. If
    # pitch DOES drift beyond this threshold mid-sweep (e.g. the
    # auto-calibrator overshot), the agent pauses the sweep, re-
    # settles pitch, and resumes. This is the safety net — in
    # practice pitch doesn't drift because nothing in MC moves it
    # without an explicit vertical mouse input.
    sweep_pitch_drift_tol_deg: float = 5.0

    # Hard cap on settle-phase iterations — after this many ticks
    # we accept whatever camera position we're in and advance the
    # phase plan anyway. Prevents an unsettleable axis from blocking
    # the whole sweep schedule.
    systematic_settle_max_ticks: int = 80

    # Cap how long we'll spend trying to align on one target before
    # giving up and moving to the next. Prevents the agent from
    # getting stuck on a voxel behind a wall.
    investigate_max_ticks: int = 80

    # After the crosshair is "close enough" (yaw + pitch error both
    # under aim_tolerance_deg), how many extra ticks we wait for F3
    # to register the target before declaring this voxel a dud.
    # MC's F3 OCR runs at ~3 Hz, so ~10 ticks at 20 Hz is plenty for
    # one OCR cycle. If we still have no confirmation after this,
    # the voxel was likely a hallucination from a misread pose and
    # we should move on.
    close_confirm_grace_ticks: int = 12

    # MC's survival-mode block reach is 4.5 blocks. F3 "Looking at
    # block" doesn't fire on voxels beyond this distance from the
    # eye, so investigating them is wasted. We use a slight margin
    # over the canonical value to handle off-centre crosshair aiming.
    max_reach_blocks: float = 5.0

    # Pre-baseline grace window: how many ticks we wait for F3 OCR
    # to produce its first parseable pose before falling into the
    # BLIND yaw sweep. 60 ticks at 20 Hz = 3 s — enough for ~9 OCR
    # cycles, which is plenty to recover yaw/pitch from a quiet
    # frame. After this window expires we sweep yaw blindly so the
    # camera doesn't sit on a single block forever when Facing-line
    # OCR is degraded.
    open_loop_initial_idle_ticks: int = 60

    # ── Scan-only mode (current default) ──────────────────────────
    # When True (the current phase of the project) the agent ONLY runs
    # the systematic 360° plan and loops it forever — no curiosity
    # queue, no investigate, no patch sweep. The WorldMap is built
    # exclusively from F3 "Looking at block" confirmations. Flip to
    # False once the sample-recogniser is trusted enough to commit
    # blocks the AI ISN'T directly looking at (see module docstring).
    scan_only_mode: bool = True

    # When True (scan-only mode) and the systematic plan completes,
    # restart from the second phase (skip the one-shot ORIENT) so
    # the camera keeps producing F3 confirmations indefinitely.
    loop_systematic: bool = True

    # SCAN behaviour: which curiosity-queue ordering to prefer
    # ("closest", "oldest", "lowest_conf"). Only consulted when
    # ``scan_only_mode`` is False.
    queue_priority: str = "closest"

    # In SCAN mode the agent sweeps the camera with this many pixels
    # of horizontal mouse motion per tick. Small enough that the
    # perception layer's patch sweep can keep up. ~3-4 px per tick
    # at 20 Hz = ~10-13°/sec — a relaxed, humanlike pan.
    scan_yaw_pixels_per_tick: int = 4

    # SCAN also varies pitch in a slow sinusoidal sweep so we look
    # both at the ground and the sky over time. Period is in ticks.
    # The sweep is BIASED toward looking DOWN (centered at +20°)
    # because F3 "Looking at block" only fires when there's a block
    # within MC's reach (~5 blocks); empty sky returns nothing. By
    # spending most of the sweep aimed at the ground we maximise
    # the chance of getting ground-truth confirmations.
    scan_pitch_amplitude_deg: float = 25.0
    scan_pitch_period_ticks:  int   = 240    # 12 s at 20 Hz
    scan_pitch_center_deg:    float = 25.0   # downward bias

    # When the perception layer reports NO looking_at target for this
    # many consecutive ticks, the agent forces the camera DOWN
    # regardless of the sweep schedule — that's the fastest way to
    # find a block within reach of the cursor.
    no_target_pitch_down_after_ticks: int  = 30
    no_target_pitch_down_per_tick:    int  = 8    # px/tick (~1.2°/tick)

    # The scan cycles between SCAN_FOR_N → INVESTIGATE → SCAN_FOR_N.
    # Tunes how long the agent looks around before pausing to
    # investigate. If we investigated every tick the camera would
    # never move.
    scan_ticks_before_investigate: int = 14

    # If the curiosity queue is empty after a scan burst, keep
    # scanning for this many ticks before falling back to a slow
    # "just rotate" mode that gradually exposes new geometry.
    idle_scan_ticks: int = 200

    # After GIVE-UP / DROP on a voxel, remember it for this many
    # agent ticks before allowing another attempt. 20 Hz × 600 ticks
    # = 30 s — long enough that the player has likely moved or the
    # camera angle has changed (so geometry differs), short enough
    # that a transient occlusion clears within one TTL window. The
    # patch sweep WILL keep re-adding the same voxel to the
    # curiosity queue every time it sees the same view; without
    # this cooldown the agent burns its investigate budget on the
    # same hallucinated voxel forever.
    failed_target_ttl_ticks: int = 600


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class WorldExplorerAgent(BaseAgent):
    """
    Camera-only agent that drives the curiosity → confirmation loop.

    Requires ``vision.world.enabled: true`` in settings — the agent
    reads from ``state.world`` and from the underlying
    ``WorldPerception`` instance to access the curiosity queue.
    """

    name = "world_explorer"

    # Modes.
    _SYSTEMATIC  = "systematic"  # initial structured scan plan
    _SCAN        = "scan"
    _INVESTIGATE = "investigate"

    def __init__(self, config: Optional[WorldExplorerConfig] = None,
                 *,
                 calibrator: Optional[MouseCalibrator] = None):
        self.cfg = config or WorldExplorerConfig()
        # Mouse-to-angle auto-calibrator. The agent learns its actual
        # pixels-per-degree from every (emitted delta, observed yaw
        # change) pair, so aim math is exact regardless of MC's
        # sensitivity setting or system mouse DPI.
        self.calibrator = calibrator or MouseCalibrator(
            cfg=MouseCalibrationConfig(
                default_px_per_deg=self.cfg.mouse_per_degree,
            ),
            persist_path=os.path.join(
                "data", "calibration", "mouse_calibration.json",
            ),
        )
        self._tick = 0
        self._mode = self._SCAN
        self._scan_tick = 0
        self._inv_tick  = 0
        self._target_voxel: Optional[Tuple[int, int, int]] = None
        self._n_investigated = 0
        self._last_log_tick = -1000
        self._perception_ref = None     # set by build_world_explorer_agent
        # When the perception layer's looking_at has been None for a
        # while, the cursor is staring at empty sky — we want to pitch
        # down. This counter tracks the consecutive null-target ticks.
        self._no_target_streak = 0
        # How long the crosshair has been "close enough" to the current
        # investigate target without an F3 confirmation. After
        # ``close_confirm_grace_ticks`` we give up — the voxel is
        # almost certainly a hallucination from a misread pose.
        self._close_ticks = 0
        # Consecutive INVESTIGATE failures (drops + give-ups). When
        # this gets high the curiosity queue is full of bad voxels
        # from misread poses — we purge and restart from scan.
        self._consecutive_drops = 0
        # Voxels we've already investigated and failed to confirm,
        # keyed by voxel pos → tick at which we gave up. Used to skip
        # the SAME hallucinated voxel on future investigate cycles —
        # otherwise the patch sweep keeps re-adding it to the curiosity
        # queue and the agent re-attempts an unwinnable aim every few
        # seconds. Entries expire after ``failed_target_ttl_ticks`` so
        # a voxel can be re-tried later if the player has moved /
        # rotated meaningfully and the geometry has changed.
        self._failed_targets: Dict[Tuple[int, int, int], int] = {}
        # Last good pose (cached when state.f3 misses on later ticks).
        self._last_yaw: Optional[float] = None
        self._last_pitch: Optional[float] = None
        # Timestamp of the most recently absorbed F3 read. Used to
        # detect a stale pose so we don't keep commanding velocity
        # while waiting for the next OCR cycle.
        self._last_pose_ts: Optional[float] = None
        # ── Loop pacing ───────────────────────────────────────────
        # Wall-clock seconds per agent tick. Used by ``_emit`` to map
        # px/sec velocity commands into per-tick deltas for the
        # auto-calibrator. The builder overrides this from
        # ``agent.tick_rate``; the default keeps the historic 20 Hz
        # behaviour for direct ``WorldExplorerAgent()`` construction.
        self.tick_period_sec: float = 0.05
        # ── Systematic-scan state ─────────────────────────────────
        # Index into ``SYSTEMATIC_SCAN_PHASES``. -1 means we haven't
        # started yet (will be set on the first decide() with pose).
        # >= len(phases) means we're done and have transitioned to
        # the reactive SCAN mode.
        self._sys_phase_idx: int = -1
        # Yaw at which the current 360° sweep started. None until
        # a sweep begins. We track unwrapped degrees travelled
        # to know when we've completed a full revolution.
        self._sweep_yaw_start: Optional[float] = None
        self._sweep_yaw_unwrapped: float = 0.0
        self._sweep_prev_yaw: Optional[float] = None
        # Ticks spent settling the current phase's pitch (and yaw,
        # for orient). Used to enforce ``systematic_settle_max_ticks``
        # so an OCR-jittery axis can't block the sweep schedule.
        self._sys_settle_ticks: int = 0
        # Mark systematic scan as completed once we've gone through
        # the plan; we never re-run it within a single session.
        self._sys_done: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────

    def _phase_plan(self) -> Tuple[_ScanPhase, ...]:
        """Return the systematic phase list appropriate for the current
        config. In reactive mode (``scan_only_mode=False``) the short
        plan runs — orient + one horizontal sweep — so the agent hands
        off to the curiosity-driven INVESTIGATE loop in ~30-60 s
        instead of ~5 min. Scan-only mode (the conservative deployment)
        keeps the full 6-phase plan because that IS the operating
        mode there."""
        if self.cfg.scan_only_mode:
            return SYSTEMATIC_SCAN_PHASES
        return SYSTEMATIC_SCAN_PHASES_SHORT

    def _print_startup_banner(self) -> None:
        s = self.calibrator.stats()
        if s["calibrated"]:
            self._log(
                f"calibrated: yaw={s['px_per_deg_yaw']} px/deg "
                f"pitch={s['px_per_deg_pitch']} px/deg "
                f"({s['yaw_samples']} yaw + {s['pitch_samples']} pitch samples)"
            )
        else:
            self._log(
                f"calibration starting from default px/deg = "
                f"{self.cfg.mouse_per_degree:.2f} "
                f"(learning from observed motion)"
            )

    def reset(self) -> None:
        self._tick = 0
        self._mode = self._SCAN
        self._scan_tick = 0
        self._inv_tick  = 0
        self._target_voxel = None
        self._n_investigated = 0
        self._last_log_tick = -1000
        self._failed_targets.clear()
        # Re-run the systematic scan plan from scratch on every reset.
        self._sys_phase_idx = -1
        self._sys_done = False
        self._sweep_yaw_start = None
        self._sweep_yaw_unwrapped = 0.0
        self._sweep_prev_yaw = None
        self._sys_settle_ticks = 0
        self._print_startup_banner()

    def attach_perception(self, perception) -> None:
        """Called by the builder to give the agent direct access to
        the running :class:`vision.world.WorldPerception` (so it can
        read the curiosity queue and pop targets from it)."""
        self._perception_ref = perception

    # ── Decide ────────────────────────────────────────────────────

    def decide(self, state: GameState) -> AgentAction:
        self._tick += 1
        if state.screen_state != ScreenState.PLAYING:
            self._log_periodic(
                f"[idle] screen_state={state.screen_state} (waiting for PLAYING)"
            )
            return self._halt()
        if self._perception_ref is None:
            self._log_periodic("[idle] perception not attached")
            return self._halt()

        f3 = state.f3
        have_pose = (f3 is not None and f3.x is not None
                      and f3.yaw is not None and f3.pitch is not None)

        # Detect STALE pose. The F3 OCR runs at ~3 Hz; between reads
        # main.py re-attaches the previous F3Info, so the timestamp
        # field is the only way to tell whether the pose we're
        # looking at is fresh. Without this check, velocity-mode
        # control overshoots wildly because the agent acts on the
        # SAME yaw error across multiple ticks while the mouse
        # worker keeps spinning the camera.
        pose_fresh = False
        if have_pose:
            ts = getattr(f3, "timestamp", None)
            if ts is None or ts != self._last_pose_ts:
                pose_fresh = True
                self._last_pose_ts = ts
            elif self._last_pose_ts is None:
                pose_fresh = True

        # Cache the latest known yaw/pitch so we can keep scanning even
        # if the OCR briefly fails. We DON'T cache x/y/z because we
        # never aim absolutely at world coords without fresh xy.
        if pose_fresh:
            self._last_yaw   = float(f3.yaw)
            self._last_pitch = float(f3.pitch)
            # Feed the auto-calibrator. If we've sent any mouse motion
            # since the last fresh pose, this teaches it the exact
            # pixels-per-degree on the user's machine.
            self.calibrator.observed_pose(float(f3.yaw), float(f3.pitch))

        # Update no-target streak. F3 "Looking at block" goes BLANK
        # whenever the cursor isn't pointed at a block within MC's
        # ~5-block reach — empty sky / horizon returns nothing. We
        # use this as a signal that we should pitch DOWN to find
        # something solid.
        # Narrow exception — ``getattr`` only ever raises AttributeError
        # on missing-attribute, so a broad ``except Exception`` here
        # hides genuine bugs in the perception layer behind a silent
        # "no target" report.
        try:
            wf = state.world
            looking_at = getattr(wf, "looking_at", None) if wf else None
        except AttributeError:
            looking_at = None
        if looking_at is None:
            self._no_target_streak += 1
        else:
            self._no_target_streak = 0

        # If we don't have fresh pose, fall back to open-loop SCAN —
        # just keep panning so the OCR has more chances to catch a
        # parseable frame. We log periodically so the user knows.
        if not have_pose:
            self._log_periodic(
                f"[idle] no F3 pose — open-loop scanning "
                f"(streak={self._no_target_streak})"
            )
            return self._emit(self._open_loop_scan())

        eye = (float(f3.x), float(f3.y) + 1.62, float(f3.z))

        # Systematic scan runs FIRST when an agent session starts:
        # orient → 360° at horizon → 360° looking down → 360° at
        # the floor → 360° looking up → 360° at the ceiling. This
        # guarantees every block within F3 reach gets a chance to
        # be confirmed before we drop into reactive mode.
        if not self._sys_done:
            if self._sys_phase_idx < 0:
                self._sys_phase_idx = 0
                self._reset_sweep_tracking(float(f3.yaw))
                self._log(f"[sys] starting systematic scan — "
                          f"{len(self._phase_plan())} phases "
                          f"({'short' if not self.cfg.scan_only_mode else 'full'})")
            return self._emit(self._tick_systematic(f3, eye, pose_fresh))

        # Past systematic completion. In ``scan_only_mode`` we never get
        # here (the systematic plan loops indefinitely), but guard
        # explicitly so a misconfigured ``loop_systematic=False`` doesn't
        # silently activate the reactive curiosity-driven path the
        # scan-only project phase is meant to suppress.
        if self.cfg.scan_only_mode:
            self._log_periodic("[sys] scan-only mode: systematic done — idling")
            return self._halt()

        # Legacy reactive mode. Feedback control: the velocity command
        # computed below remains in effect until the next agent decide().
        # With F3 OCR at ~3 Hz and the conservative aim_max_deg_per_sec
        # cap, the velocity moves the camera at most ~10° per OCR
        # cycle — well under any realistic aim error — so re-running
        # the controller every tick from stale data simply re-emits
        # roughly the same velocity. No overshoot.
        if self._mode == self._SCAN:
            return self._emit(self._tick_scan(f3, eye))
        return self._emit(self._tick_investigate(f3, eye))

    def _halt(self) -> AgentAction:
        """Return a velocity-zero action so the mouse worker stops
        moving when we're idle. Without ``force_velocity`` an earlier
        ``set_velocity`` would keep the camera spinning even after the
        agent decided to wait — the runtime's dispatcher otherwise
        short-circuits on the (0, 0) default."""
        return AgentAction(look_vx=0.0, look_vy=0.0, force_velocity=True)

    def _emit(self, action: AgentAction) -> AgentAction:
        """Last-stop helper before returning to main: tell the
        calibrator how much mouse motion this tick will emit. The
        calibrator pairs this with the next observed yaw/pitch
        change to learn the exact pixels-per-degree.

        For velocity-mode motion, we approximate the per-tick px
        delta as ``velocity × tick_period`` (tick rate is 20 Hz
        elsewhere in the project). This is what the F3 reader will
        observe between consecutive pose reads.
        """
        # One-shot easing path (legacy).
        if action.look_dx or action.look_dy:
            self.calibrator.emitted(action.look_dx, action.look_dy)
        # Velocity path: approximate px-this-tick as v × tick_period.
        if action.look_vx or action.look_vy:
            dx_eq = action.look_vx * self.tick_period_sec
            dy_eq = action.look_vy * self.tick_period_sec
            self.calibrator.emitted(dx_eq, dy_eq)
        return action

    # ── Fallback scan (no pose) ───────────────────────────────────

    def _open_loop_scan(self) -> AgentAction:
        """
        Fallback when F3 OCR hasn't produced a parseable pose yet.

        Two regimes, separated by whether we've ever seen yaw/pitch
        in this session:

        * **Pre-baseline** (no pose ever): sit still for the FIRST
          ``open_loop_initial_idle_ticks`` ticks so F3 OCR gets a
          chance to catch up without us drifting the camera. After
          that grace window, switch to BLIND YAW SCAN — emit a
          constant yaw velocity even though we can't close the loop
          on pitch. This is exactly the case where the user has F3
          panel on but the Facing line is too OCR-garbled to recover
          yaw/pitch: ``looking_at`` and XYZ still parse fine, so the
          perception layer can keep committing blocks while the
          camera sweeps. Without this the agent would be stuck on
          a single frame forever and only ever log one block.

        * **Post-baseline** (we've seen pose at least once): gentle
          yaw scan, pitch forced down only if we've also been on
          empty sky for a while.
        """
        px_yaw   = self.calibrator.px_per_deg_yaw()
        px_pitch = self.calibrator.px_per_deg_pitch()

        if self._last_yaw is None or self._last_pitch is None:
            # Pre-baseline — give F3 OCR a brief grace window to
            # produce a clean read before we start panning blindly.
            if self._tick < self.cfg.open_loop_initial_idle_ticks:
                return AgentAction(look_vx=0.0, look_vy=0.0,
                                    force_velocity=True)
            # No pose yet but we've waited long enough; sweep the
            # camera blindly so perception can see new Targeted
            # Blocks as the crosshair traverses the world. Pitch
            # stays at whatever it was — we have no feedback to
            # correct it, but MC doesn't drift pitch on its own.
            vx = self.cfg.scan_yaw_deg_per_sec * px_yaw
            return AgentAction(look_vx=vx, look_vy=0.0,
                                force_velocity=True)

        # Post-baseline: gentle yaw scan, pitch only forced down if
        # we've also been on empty sky for a while.
        vx = self.cfg.scan_yaw_deg_per_sec * px_yaw     # px/sec
        vy = 0.0
        if self._no_target_streak >= self.cfg.no_target_pitch_down_after_ticks:
            vy = self.cfg.scan_pitch_deg_per_sec * px_pitch
        return AgentAction(look_vx=vx, look_vy=vy, force_velocity=True)

    # ── Systematic scan mode ──────────────────────────────────────

    def _reset_sweep_tracking(self, yaw_now: float) -> None:
        self._sweep_yaw_start    = yaw_now
        self._sweep_yaw_unwrapped = 0.0
        self._sweep_prev_yaw      = yaw_now

    # Max per-OCR-cycle yaw delta we accept as a real sweep step.
    # We command ~18°/sec and OCR fires at ~3 Hz, so true per-cycle
    # deltas sit around 6°. Anything > 60° is either an OCR misread
    # or a pose-filter glitch — drop it to avoid prematurely
    # completing the 360° check off the back of a single bad read.
    _MAX_SWEEP_STEP_DEG = 60.0

    def _update_sweep_progress(self, yaw_now: float) -> float:
        """Update unwrapped degrees travelled since the sweep began.
        Returns the absolute total swept so far.

        Guards against absurd per-update deltas (OCR jumps, pose-
        filter glitches) which could otherwise mis-count a single
        misread as a full revolution.
        """
        if self._sweep_prev_yaw is None:
            self._sweep_prev_yaw = yaw_now
            return 0.0
        delta = self._normalize_angle(yaw_now - self._sweep_prev_yaw)
        if abs(delta) > self._MAX_SWEEP_STEP_DEG:
            # Discard this sample's contribution to the unwrapped
            # total — but still update the anchor so the NEXT
            # delta is computed against the latest reading.
            self._sweep_prev_yaw = yaw_now
            return abs(self._sweep_yaw_unwrapped)
        self._sweep_yaw_unwrapped += delta
        self._sweep_prev_yaw = yaw_now
        return abs(self._sweep_yaw_unwrapped)

    def _tick_systematic(self, f3, eye, pose_fresh: bool) -> AgentAction:
        """
        One tick of the systematic scan plan.

        Two STRICTLY SEPARATED axes of motion:

        1. SETTLE step — pitch (and yaw for the orient phase) eased
           toward the phase's target. NO yaw rotation during this
           step except for orient. Once pitch is within tolerance the
           settle step is done and we hand off to the sweep step.

        2. SWEEP step — pure yaw rotation at a constant rate; pitch
           velocity is HARD-LOCKED at zero. If pitch happens to drift
           beyond ``sweep_pitch_drift_tol_deg`` mid-sweep the sweep
           is paused and we drop back to the settle step. This is the
           fix for the "wobble down-right then up-left" pattern: by
           keeping the two axes orthogonal, the camera describes a
           clean horizon-aligned arc instead of a diagonal corkscrew.
        """
        phase = self._phase_plan()[self._sys_phase_idx]
        cur_pitch = float(f3.pitch)
        cur_yaw   = float(f3.yaw)

        px_per_deg_yaw   = self.calibrator.px_per_deg_yaw()
        px_per_deg_pitch = self.calibrator.px_per_deg_pitch()
        # Settle uses its own (faster) P-gain + speed cap so it
        # actually converges within the settle-timeout budget. The
        # sweep below still uses investigate-grade smoothness.
        p_gain  = self.cfg.settle_p_gain
        max_dps = self.cfg.settle_max_deg_per_sec
        tol     = self.cfg.systematic_settle_tolerance_deg

        pitch_err = phase.target_pitch - cur_pitch
        pitch_ok  = abs(pitch_err) < tol
        # For the orient phase we also aim yaw to 0.
        yaw_target_for_orient = 0.0
        yaw_err_for_orient = self._normalize_angle(yaw_target_for_orient
                                                    - cur_yaw)
        yaw_orient_ok = (phase.name != "orient"
                          or abs(yaw_err_for_orient) < tol)

        # Hard timeout — if a particular axis can't settle (e.g. OCR
        # keeps jittering), advance anyway so the rest of the sweep
        # plan still runs.
        settle_timeout_hit = (
            self._sys_settle_ticks >= self.cfg.systematic_settle_max_ticks
        )

        # ── SETTLE: pitch first, yaw only for orient ──────────────
        if (not pitch_ok or not yaw_orient_ok) and not settle_timeout_hit:
            self._sys_settle_ticks += 1
            pitch_dps = max(-max_dps, min(max_dps, pitch_err * p_gain))
            if phase.name == "orient":
                yaw_dps = max(-max_dps, min(max_dps,
                                            yaw_err_for_orient * p_gain))
            else:
                # Pure pitch settle — never move yaw during a non-orient
                # settle. This is what the old code did already; keeping
                # it explicit for the symmetry with the sweep below.
                yaw_dps = 0.0
            vx = yaw_dps   * px_per_deg_yaw
            vy = pitch_dps * px_per_deg_pitch
            self._log_periodic(
                f"[sys:{phase.name}] settling[{self._sys_settle_ticks}]: "
                f"yaw={cur_yaw:.1f} pitch={cur_pitch:.1f} "
                f"→ target_pitch={phase.target_pitch:.0f}"
                f"  v=({vx:+.0f},{vy:+.0f})px/s"
            )
            return AgentAction(look_vx=vx, look_vy=vy)

        if settle_timeout_hit and not (pitch_ok and yaw_orient_ok):
            self._log(f"[sys:{phase.name}] settle TIMEOUT — proceeding "
                      f"with yaw={cur_yaw:.1f} pitch={cur_pitch:.1f}")
        self._sys_settle_ticks = 0

        # Once aim is achieved, either advance (no sweep) or sweep.
        if not phase.sweep_360:
            self._log(f"[sys:{phase.name}] settled — advancing")
            self._advance_systematic_phase(cur_yaw)
            return self._halt()

        # Begin / continue the 360° sweep.
        if self._sweep_yaw_start is None:
            self._reset_sweep_tracking(cur_yaw)

        if pose_fresh:
            swept_abs = self._update_sweep_progress(cur_yaw)
            if swept_abs >= 360.0:
                self._log(f"[sys:{phase.name}] 360° complete (swept "
                          f"{swept_abs:.1f}°) — advancing")
                self._advance_systematic_phase(cur_yaw)
                return self._halt()
        else:
            swept_abs = abs(self._sweep_yaw_unwrapped)

        # ── SWEEP: pure yaw, pitch HARD-LOCKED at zero ────────────
        # If pitch has somehow drifted (it shouldn't, but a wonky
        # auto-calibration could overshoot during the previous settle)
        # pause the sweep and re-enter the settle path. The unwrapped
        # yaw counter is preserved so we resume the same 360° later.
        if abs(pitch_err) > self.cfg.sweep_pitch_drift_tol_deg:
            self._sys_settle_ticks = 0  # restart the settle budget
            pitch_dps = max(-max_dps, min(max_dps, pitch_err * p_gain))
            self._log_periodic(
                f"[sys:{phase.name}] pitch drift {pitch_err:+.1f}° — "
                f"pausing sweep, re-settling"
            )
            return AgentAction(look_vx=0.0,
                                look_vy=pitch_dps * px_per_deg_pitch,
                                force_velocity=True)

        sweep_yaw_dps = self.cfg.scan_yaw_deg_per_sec
        vx = sweep_yaw_dps * px_per_deg_yaw

        self._log_periodic(
            f"[sys:{phase.name}] sweeping: yaw={cur_yaw:.1f} "
            f"pitch={cur_pitch:.1f} swept={swept_abs:.0f}/360°  "
            f"v=({vx:+.0f},+0)px/s"
        )
        return AgentAction(look_vx=vx, look_vy=0.0, force_velocity=True)

    def _advance_systematic_phase(self, cur_yaw: float) -> None:
        plan = self._phase_plan()
        self._sys_phase_idx += 1
        self._sys_settle_ticks = 0    # fresh budget for next phase
        if self._sys_phase_idx >= len(plan):
            if self.cfg.scan_only_mode and self.cfg.loop_systematic:
                # Loop the plan forever, skipping the one-shot ORIENT
                # phase on subsequent passes (we know the camera is
                # already roughly oriented from the previous cycle).
                self._sys_phase_idx = 1
                self._reset_sweep_tracking(cur_yaw)
                confirmed = self._confirmed_count()
                self._log(f"[sys] full pass complete (confirmed={confirmed}) "
                          f"— restarting at "
                          f"{plan[self._sys_phase_idx].name}")
                return
            # Legacy path: drop into reactive SCAN/INVESTIGATE — only
            # reachable when ``scan_only_mode`` is False, which is the
            # eventual self-improvement phase.
            self._sys_done = True
            self._mode = self._SCAN
            self._scan_tick = 0
            self._log("[sys] systematic scan COMPLETE — switching to "
                      "reactive SCAN")
        else:
            self._reset_sweep_tracking(cur_yaw)
            self._log(f"[sys] phase -> {plan[self._sys_phase_idx].name}")

    def _confirmed_count(self) -> int:
        """Cheap read of how many distinct voxels F3 has confirmed so
        far this session. Best-effort — perception versions without
        ``stats()`` fall through to a 0 so the log still prints."""
        try:
            return int(self._perception_ref.stats().get(
                "confirmed_count", 0))
        except (AttributeError, KeyError, TypeError):
            return 0

    # ── SCAN mode ─────────────────────────────────────────────────

    def _tick_scan(self, f3, eye) -> AgentAction:
        self._scan_tick += 1
        # Try to switch to INVESTIGATE every N ticks.
        if (self._scan_tick >= self.cfg.scan_ticks_before_investigate
                and self._maybe_enter_investigate(eye)):
            return self._tick_investigate(f3, eye)

        px_yaw   = self.calibrator.px_per_deg_yaw()
        px_pitch = self.calibrator.px_per_deg_pitch()
        # Yaw: constant pan velocity.
        vx = self.cfg.scan_yaw_deg_per_sec * px_yaw
        # Pitch: P-controller toward the desired sinusoidal sweep,
        # OR a forced descent if we're stuck looking at sky.
        if self._no_target_streak >= self.cfg.no_target_pitch_down_after_ticks:
            vy = self.cfg.scan_pitch_deg_per_sec * px_pitch
        else:
            # Modulo the tick by the period so ``t`` always lives in
            # [0, 1) — without it long sessions accumulate enough float
            # magnitude to chew through ``math.sin``'s precision (an
            # eight-hour run at 20 Hz is 576 000 ticks, well past where
            # sin/cos starts visibly stepping).
            period = max(1, self.cfg.scan_pitch_period_ticks)
            t = (self._tick % period) / period
            desired_pitch = (self.cfg.scan_pitch_center_deg
                             + self.cfg.scan_pitch_amplitude_deg * math.sin(
                                 2.0 * math.pi * t))
            pitch_err = desired_pitch - float(f3.pitch)
            # Convert pitch error → angular velocity (P-control).
            deg_per_sec = max(-self.cfg.scan_pitch_deg_per_sec,
                              min(self.cfg.scan_pitch_deg_per_sec,
                                  pitch_err * 0.8))
            vy = deg_per_sec * px_pitch

        self._log_periodic(
            f"[scan] tick={self._tick} pose=({f3.x:.1f},{f3.z:.1f}) "
            f"yaw={f3.yaw:.0f} pitch={f3.pitch:.0f} "
            f"queue={len(self._perception_ref.curiosity_queue())} "
            f"px/deg=({px_yaw:.2f},{px_pitch:.2f}) "
            f"v=({vx:+.0f},{vy:+.0f})px/s "
            f"no_target_streak={self._no_target_streak}"
        )
        return AgentAction(look_vx=vx, look_vy=vy)

    def _maybe_enter_investigate(self, eye) -> bool:
        """Look for a curiosity target within reach; switch mode if
        one is found."""
        # Garbage-collect expired entries from the failed-target
        # cooldown table so the dict doesn't grow without bound
        # over a long-running session.
        if self._failed_targets:
            ttl = self.cfg.failed_target_ttl_ticks
            expired = [k for k, t in self._failed_targets.items()
                       if self._tick - t > ttl]
            for k in expired:
                self._failed_targets.pop(k, None)

        # Pop up to N candidates, skipping any that lie beyond MC's
        # block-reach distance from the current eye (those would
        # never produce an F3 looking_at confirmation no matter how
        # long we aimed at them).
        cur_yaw = self._last_yaw  # cached last good yaw
        for _ in range(16):
            target = self._perception_ref.take_curiosity_target(
                eye=eye,
                prefer=self.cfg.queue_priority,
                yaw_deg=cur_yaw,
            )
            if target is None:
                return False
            # Skip recent failures so we don't burn the investigate
            # budget on the same hallucinated voxel cycle after cycle.
            if target in self._failed_targets:
                self._log_periodic(
                    f"[invst] skip recently-failed target {target} "
                    f"(cooldown ends in "
                    f"{self.cfg.failed_target_ttl_ticks - (self._tick - self._failed_targets[target])} ticks)"
                )
                continue
            tcx = target[0] + 0.5
            tcy = target[1] + 0.5
            tcz = target[2] + 0.5
            dist = math.sqrt((tcx - eye[0]) ** 2
                              + (tcy - eye[1]) ** 2
                              + (tcz - eye[2]) ** 2)
            if dist > self.cfg.max_reach_blocks:
                # Out-of-reach voxels are noise from a misread pose
                # earlier — just discard.
                self._log_periodic(
                    f"[invst] discarded out-of-reach target {target} "
                    f"(dist={dist:.1f} > {self.cfg.max_reach_blocks})"
                )
                continue
            self._target_voxel = target
            self._mode = self._INVESTIGATE
            self._inv_tick = 0
            self._scan_tick = 0
            self._close_ticks = 0
            return True
        return False

    # ── INVESTIGATE mode ──────────────────────────────────────────

    def _tick_investigate(self, f3, eye) -> AgentAction:
        self._inv_tick += 1
        target = self._target_voxel
        if target is None:
            self._mode = self._SCAN
            self._scan_tick = 0
            return self._halt()

        # Compute desired yaw + pitch to look AT this voxel's centre.
        tx = target[0] + 0.5
        ty = target[1] + 0.5
        tz = target[2] + 0.5
        dx_world = tx - eye[0]
        dy_world = ty - eye[1]
        dz_world = tz - eye[2]
        # MC convention: yaw 0 = +Z, increases clockwise from above.
        # yaw = atan2(-x, z) in degrees.
        desired_yaw = math.degrees(math.atan2(-dx_world, dz_world))
        horiz = math.hypot(dx_world, dz_world)
        # MC pitch: +90 = down, -90 = up. pitch = -atan2(dy, horiz).
        desired_pitch = math.degrees(-math.atan2(dy_world, max(0.001, horiz)))

        yaw_err   = self._normalize_angle(desired_yaw - float(f3.yaw))
        pitch_err = desired_pitch - float(f3.pitch)

        # If the F3 OCR has confirmed THIS voxel — or any 1-block
        # neighbour — since we chose it, we're done. Why the
        # neighbourhood check: even with perfect aim, MC's block
        # raytrace can resolve to a NEIGHBOURING voxel when the
        # crosshair lands on a face edge — especially for short blocks
        # (grass, fern) on top of a full block, where the raytrace
        # tie-breaks between the grass voxel and the grass_block
        # below. Without this relaxation the agent would aim, MC
        # would happily confirm the neighbour, the agent would still
        # see ``is_confirmed(target) == False``, and time out. The
        # neighbour confirm is just as valuable — the sample-store
        # auto-collect already wrote a labelled patch for it.
        tx0, ty0, tz0 = target
        for dx, dy, dz in ((0,0,0), (0,1,0), (0,-1,0),
                            (1,0,0), (-1,0,0),
                            (0,0,1), (0,0,-1)):
            if self._perception_ref.is_confirmed((tx0+dx, ty0+dy, tz0+dz)):
                self._n_investigated += 1
                self._consecutive_drops = 0
                hit_voxel = (tx0+dx, ty0+dy, tz0+dz)
                via = "exact" if hit_voxel == target else f"neighbour@{hit_voxel}"
                self._log(f"[invst] CONFIRMED {target} ({via}) after "
                          f"{self._inv_tick} ticks — total confirmed by "
                          f"agent: {self._n_investigated}")
                self._target_voxel = None
                self._mode = self._SCAN
                self._scan_tick = 0
                return self._halt()

        # Timeout — voxel is unreachable / occluded behind another
        # block. Drop it and resume scan.
        if self._inv_tick > self.cfg.investigate_max_ticks:
            self._log(f"[invst] GIVE UP on {target} after "
                      f"{self._inv_tick} ticks (yaw_err={yaw_err:+.1f}, "
                      f"pitch_err={pitch_err:+.1f})")
            self._failed_targets[target] = self._tick
            self._target_voxel = None
            self._mode = self._SCAN
            self._scan_tick = 0
            return self._halt()

        # Velocity-mode P-controller: angular velocity proportional
        # to remaining error, clamped to a humanlike max speed. This
        # produces a smooth ease-in: fast at the start of a long
        # rotation, decelerating smoothly to zero as we arrive.
        # Uses the auto-calibrated px-per-degree for exact aim.
        px_per_deg_yaw   = self.calibrator.px_per_deg_yaw()
        px_per_deg_pitch = self.calibrator.px_per_deg_pitch()
        max_dps = self.cfg.aim_max_deg_per_sec
        p_gain  = self.cfg.investigate_p_gain
        yaw_dps   = max(-max_dps, min(max_dps, yaw_err   * p_gain))
        pitch_dps = max(-max_dps, min(max_dps, pitch_err * p_gain))
        vx = yaw_dps   * px_per_deg_yaw
        vy = pitch_dps * px_per_deg_pitch

        # When close enough, halt velocity and let F3 catch up.
        close_enough = (abs(yaw_err) < self.cfg.aim_tolerance_deg
                        and abs(pitch_err) < self.cfg.aim_tolerance_deg)
        if close_enough:
            vx = 0.0
            vy = 0.0
            self._close_ticks += 1
        else:
            self._close_ticks = 0

        # Fast give-up: if we've been steady on this voxel for the
        # grace window without F3 confirming, the target is bogus
        # (typically a hallucinated voxel from a misread pose).
        if self._close_ticks > self.cfg.close_confirm_grace_ticks:
            self._log(f"[invst] DROP {target} — close for "
                      f"{self._close_ticks} ticks with no F3 confirm "
                      f"(probably out-of-reach hallucination)")
            self._failed_targets[target] = self._tick
            self._target_voxel = None
            self._close_ticks = 0
            self._consecutive_drops += 1
            # If we're dropping target after target, the curiosity
            # queue is full of bad voxels from misread poses. Purge it
            # and force a long scan window so the perception layer
            # repopulates from current frames.
            if self._consecutive_drops >= 5:
                # purge_curiosity_queue is part of WorldPerception's
                # public API; only AttributeError is plausible here
                # (older perception versions without the method).
                try:
                    n_purged = self._perception_ref.purge_curiosity_queue()
                    self._log(f"[invst] PURGED curiosity queue ({n_purged} "
                              f"stale entries) — too many consecutive drops")
                except AttributeError:
                    self._log("[invst] cannot purge: perception "
                              "implementation lacks purge_curiosity_queue()")
                self._consecutive_drops = 0
            self._mode = self._SCAN
            self._scan_tick = 0
            return self._halt()

        self._log_periodic(
            f"[invst] tick={self._inv_tick}/{self.cfg.investigate_max_ticks} "
            f"target={target} yaw_err={yaw_err:+5.1f} "
            f"pitch_err={pitch_err:+5.1f} v=({vx:+.0f},{vy:+.0f})px/s "
            f"close={close_enough} ({self._close_ticks})"
        )
        return AgentAction(look_vx=vx, look_vy=vy)

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _normalize_angle(deg: float) -> float:
        # Symmetric range (-180, 180]. The earlier asymmetry between
        # ``>`` and ``<=`` caused a sample at exactly -180 to wrap to
        # +180 while +180 stayed put — harmless in practice but easier
        # to reason about with a symmetric rule.
        while deg >  180.0: deg -= 360.0
        while deg < -180.0: deg += 360.0
        return deg

    def _log(self, msg: str) -> None:
        print(f"[agent.world_explorer] {msg}")
        self._last_log_tick = self._tick

    def _log_periodic(self, msg: str) -> None:
        if self._tick - self._last_log_tick >= 40:
            self._log(msg)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_world_explorer_agent(settings: dict) -> WorldExplorerAgent:
    cfg_raw = ((settings or {}).get("agent", {}) or {}).get(
        "world_explorer", {}) or {}
    cfg = WorldExplorerConfig()
    # Float-typed config keys. Every velocity-mode tuning knob lives
    # here so the user can tune behaviour in settings.yaml without
    # editing code — previously aim_max_deg_per_sec, investigate_p_gain,
    # scan_yaw_deg_per_sec, scan_pitch_deg_per_sec and
    # systematic_settle_tolerance_deg were silently hardcoded.
    for key in (
        "mouse_per_degree", "aim_tolerance_deg",
        "scan_pitch_amplitude_deg", "mouse_gain",
        "scan_pitch_center_deg", "max_reach_blocks",
        "aim_max_deg_per_sec", "scan_yaw_deg_per_sec",
        "scan_pitch_deg_per_sec", "investigate_p_gain",
        "systematic_settle_tolerance_deg",
        "sweep_pitch_drift_tol_deg",
        "settle_p_gain", "settle_max_deg_per_sec",
    ):
        if key in cfg_raw and cfg_raw[key] is not None:
            setattr(cfg, key, float(cfg_raw[key]))
    for key in (
        "max_mouse_dx", "max_mouse_dy",
        "investigate_max_ticks",
        "scan_yaw_pixels_per_tick", "scan_pitch_period_ticks",
        "scan_ticks_before_investigate", "idle_scan_ticks",
        "no_target_pitch_down_after_ticks",
        "no_target_pitch_down_per_tick",
        "close_confirm_grace_ticks",
        "systematic_settle_max_ticks",
        "open_loop_initial_idle_ticks",
        "failed_target_ttl_ticks",
    ):
        if key in cfg_raw and cfg_raw[key] is not None:
            setattr(cfg, key, int(cfg_raw[key]))
    for key in ("scan_only_mode", "loop_systematic"):
        if key in cfg_raw and cfg_raw[key] is not None:
            setattr(cfg, key, bool(cfg_raw[key]))
    if "queue_priority" in cfg_raw and cfg_raw["queue_priority"]:
        cfg.queue_priority = str(cfg_raw["queue_priority"])
    # Tick rate ultimately drives the velocity-to-px-per-tick mapping
    # inside ``_emit``. Plumbed through here so a future settings change
    # to ``agent.tick_rate`` is honoured instead of silently using 20 Hz.
    tick_rate = ((settings or {}).get("agent", {}) or {}).get("tick_rate", 20)
    try:
        tick_hz = float(tick_rate)
    except (TypeError, ValueError):
        tick_hz = 20.0
    agent = WorldExplorerAgent(config=cfg)
    agent.tick_period_sec = 1.0 / max(1.0, tick_hz)
    return agent


__all__ = [
    "WorldExplorerAgent",
    "WorldExplorerConfig",
    "build_world_explorer_agent",
]
