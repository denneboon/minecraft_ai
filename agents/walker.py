# agents/walker.py
"""
Voxel-path walker — translates a sequence of waypoint cells into
keyboard / mouse commands that move the player along that path.

Plug-in points
--------------
Two layers:

  ``WalkerController`` — stateless-ish state machine. ``set_target()``
    plans a path (or accepts a pre-planned one); ``tick(f3)`` returns
    one ``AgentAction`` per call given the freshest F3 pose. Used as a
    primitive by higher-level agents (resource-finder, navigate-to-X,
    etc.) without forcing the BaseAgent shape on the caller.

  ``NavigationAgent`` — a thin BaseAgent shell wrapping the controller
    so ``main.py --agent navigation`` can drive it directly. Its only
    job is to read the target from settings + thread world-perception
    in.

Movement model
--------------
The path is a list of standable feet-position voxels. The walker:

  1. Picks the FURTHEST waypoint on the path that is still "directly
     reachable" from the current pose (same Y, or a single ±1 Y step
     away). This skips redundant intermediate cardinals so the agent
     doesn't zigzag at every microblock.
  2. Computes the yaw needed to face that waypoint's centre.
  3. While yaw_err exceeds ``aim_tolerance_deg``, rotates the camera
     with a P-controlled velocity command. NO forward-key during
     rotation — strafing-while-turning corkscrews into walls.
  4. Once aimed, presses ``forward``. Adds ``jump`` for any +Y step;
     skips ``forward`` while falling (free vertical motion).
  5. Detects ARRIVAL when current XYZ is within ``arrived_tolerance``
     of the GOAL waypoint horizontally AND within ``arrived_y_tol``
     vertically.

Robustness
----------
Edge cases the walker handles WITHOUT crashing the agent loop:

  * ``f3 is None`` (no fresh pose) — emit ``release-all-movement`` so
    the player doesn't drift forward through unknown terrain while we
    wait for the next OCR cycle.
  * Path becomes stale (we measure no XYZ progress for
    ``stuck_replan_ticks`` ticks despite walking) — invalidate the
    path and re-plan from the current pose. If replan fails twice in
    a row, fall back to STATUS_BLOCKED so the caller can decide
    what to do.
  * Path lies through unobserved space (passable policy) and we hit
    an unexpected solid — the same stuck-detection triggers a replan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from brain.interfaces import AgentAction, BaseAgent
from vision.ocr import F3Info
from vision.processing import GameState, ScreenState
from vision.world.map import WorldMap
from vision.world.pathfind import (
    PathfinderConfig,
    PathResult,
    find_path,
)


# ---------------------------------------------------------------------------
# Walker state + config
# ---------------------------------------------------------------------------


class WalkerStatus(str, Enum):
    IDLE     = "idle"
    PLANNING = "planning"
    WALKING  = "walking"
    ARRIVED  = "arrived"
    BLOCKED  = "blocked"   # plan failed or repeatedly stuck


@dataclass
class WalkerConfig:
    # Tolerance for "we've reached the goal" in horizontal blocks.
    # MUST be >= ``waypoint_tolerance`` — otherwise the walker can
    # advance PAST the final waypoint (waypoint-advance fires at
    # waypoint_tolerance distance) without ever registering ARRIVED
    # (which requires arrived_tolerance). Live testing showed this
    # exact failure: the player got within ~0.7 blocks of the goal
    # voxel centre, the walker's path index advanced off the end,
    # and the agent kept walking forward looking for a target it
    # had already passed.
    arrived_tolerance: float = 0.8
    # Vertical leeway. 1.0 = the goal level OR the level just above
    # (e.g. fell 1 block past the goal). We accept this band so a
    # one-block fall on arrival doesn't loop forever.
    arrived_y_tol: float = 1.5

    # Horizontal distance at which a SUB-waypoint is considered
    # reached and we advance to the next one. Lower = more precise
    # turns; higher = smoother path.
    waypoint_tolerance: float = 0.65

    # Aim tolerance for the rotation step. We start pressing W only
    # after yaw is within this many degrees of the target bearing.
    aim_tolerance_deg: float = 6.0

    # Target pitch while walking. Humans walk with their head
    # roughly level — and a level pitch is what F3 "Looking at
    # block" needs to register the voxel in front of the player.
    # The walker P-controls pitch toward this value in parallel
    # with the yaw rotation, so the agent doesn't end up sprinting
    # forward while staring straight up at the sky (the systematic-
    # scan plan often leaves pitch at ±80° when handing off).
    target_pitch_deg:    float = 5.0    # slight downward bias
    pitch_p_gain:        float = 0.5
    pitch_max_deg_per_sec: float = 25.0
    # P-controller for camera rotation. Velocity command in px/sec
    # per degree of yaw error, capped at ``aim_max_deg_per_sec``.
    # Live testing showed the *commanded* deg/sec achieves ~25 %
    # of the rate in MC (mouse-input lag + MC sensitivity ramp), so
    # the cap is set 3-4x higher than what looks human at face
    # value. Net actual rate ≈ 25 °/s which IS human-eyeglance
    # territory. Lowering further makes 180° turns take 10+ s.
    aim_p_gain:           float = 0.7
    aim_max_deg_per_sec:  float = 90.0

    # Default mouse px-per-degree when no calibrator is attached.
    # Will be overridden by the calibrator at run time.
    mouse_per_degree: float = 6.5

    # Ticks of no horizontal progress (>0.1 blocks in
    # ``stuck_progress_window`` ticks) before we mark the path as
    # stale and replan.
    stuck_progress_window: int = 30
    stuck_min_progress:    float = 0.1
    # If a replan immediately after a stuck event also fails to
    # produce progress, give up and report BLOCKED.
    stuck_replan_retries:  int = 2

    # Pathfinder config — leak through so a caller can swap the
    # unknown-block policy without subclassing the walker.
    pathfind: PathfinderConfig = field(default_factory=PathfinderConfig)

    # Hard cap on plan attempts per ``set_target`` call. Prevents an
    # impossible target from blowing the agent loop's per-tick
    # budget with repeated A* calls.
    max_plan_attempts: int = 4

    # Failed-target cooldown TTL (agent ticks). When a plan to a
    # specific goal fails, that goal is remembered for this many
    # ticks before the walker accepts another ``set_target`` call
    # to the same voxel. Stops an autonomous parent agent that
    # keeps proposing an impossible target from running a full A*
    # every tick. Mirrors ``world_explorer``'s ``failed_target_ttl_ticks``.
    failed_target_ttl_ticks: int = 600

    # Y-drop replan threshold (blocks). Under ``passable`` mode the
    # pathfinder optimistically assumes unobserved terrain is solid
    # ground at the start Y. If the player walks east into the
    # planned cell and falls into a real-world cliff the actual Y
    # diverges from the waypoint Y. Once the divergence exceeds
    # this threshold the walker invalidates the plan and re-plans
    # from the new pose — much faster than waiting for the
    # stuck-timer (which only fires on no-progress, not on
    # "progressed-but-into-the-wrong-Y").
    unexpected_y_drop_blocks: int = 2

    # Goal-rescue radius (blocks). When the exact goal isn't
    # standable (typical cause: air-carving observed the floor as
    # AIR earlier, so the floor-check rejects the goal cell), the
    # walker scans this many blocks in each axis around the goal
    # for a standable neighbour and plans to that instead. 0 =
    # off. The cost of the rescue search scales as
    # (2R+1)^3 plan attempts; keep small. 1 = 26 alternatives.
    goal_rescue_radius: int = 1


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class WalkerController:
    """
    State machine that produces one ``AgentAction`` per ``tick()``
    call. Internally tracks the path, the current waypoint index,
    and stuck-detection state.

    Use:
        ctrl = WalkerController(world_map)
        ctrl.set_target((10, 64, 5))
        while ctrl.status not in (WalkerStatus.ARRIVED, WalkerStatus.BLOCKED):
            action = ctrl.tick(f3_info)
            ...
    """

    def __init__(self,
                 world_map: WorldMap,
                 *,
                 config: Optional[WalkerConfig] = None,
                 dimension: Optional[str] = None):
        self.world_map = world_map
        self.cfg = config or WalkerConfig()
        self.dimension = dimension
        self.status: WalkerStatus = WalkerStatus.IDLE
        # Plan state.
        self._goal: Optional[Tuple[int, int, int]] = None
        self._path: List[Tuple[int, int, int]] = []
        self._wp_idx: int = 0
        # Mouse calibrator (optional — supplied by NavigationAgent when
        # wired through ``main.py``). Without it we use the default
        # ``mouse_per_degree`` from the config.
        self._px_per_deg: Optional[float] = None
        # Stuck-detection. We only count FRESH F3 reads toward the
        # progress window — cached repeats of the same pose between
        # 3 Hz OCR cycles produce no XYZ change and would falsely
        # trigger the stuck timer at 20 Hz.
        self._stuck_pos: Optional[Tuple[float, float, float]] = None
        self._stuck_fresh_ticks_at_pos: int = 0
        self._last_pose_ts: Optional[float] = None
        self._tick_count: int = 0
        self._plan_attempts: int = 0
        # Diagnostics.
        self.last_plan_result: Optional[PathResult] = None
        self._last_aim_yaw: Optional[float] = None
        # ``_diag_due`` is set on the periodic-status branch inside
        # ``tick()`` and read again at the end of the same tick for
        # the per-action log. Initialise here so the first-tick guard
        # and any path that short-circuits before the assignment can
        # safely read it without ``getattr``-shielded undefined behaviour.
        self._diag_due: bool = False
        # Sanity-floor the configured px-per-degree. A YAML value of
        # 0 / NaN / inf would propagate into the look_vx command and
        # leave the camera stuck rotating at zero rate forever — the
        # walker can never reach aim tolerance, never presses W,
        # never moves. We refuse to construct with a non-positive
        # default and pin to the dataclass minimum.
        if (not math.isfinite(self.cfg.mouse_per_degree)
                or self.cfg.mouse_per_degree < 0.1):
            print(f"[pathwalker][WARN] mouse_per_degree="
                  f"{self.cfg.mouse_per_degree!r} is non-positive / "
                  f"non-finite; falling back to 6.5 to keep the "
                  f"yaw P-controller alive.")
            self.cfg.mouse_per_degree = 6.5
        # Invariant: arrived_tolerance MUST be >= waypoint_tolerance, or the
        # walker can pass the goal (within waypoint_tolerance) without ever
        # registering ARRIVED -> "walks past the goal forever". The defaults
        # honour this, but YAML can override either independently, so clamp.
        if self.cfg.arrived_tolerance < self.cfg.waypoint_tolerance:
            print(f"[pathwalker][WARN] arrived_tolerance "
                  f"{self.cfg.arrived_tolerance} < waypoint_tolerance "
                  f"{self.cfg.waypoint_tolerance}; raising it to match so "
                  f"arrival can register.")
            self.cfg.arrived_tolerance = self.cfg.waypoint_tolerance
        # Failed-target cooldown table. Mirrors the pattern in
        # ``world_explorer``: when a plan to ``goal`` fails the goal
        # goes here with the tick we gave up. Repeated ``set_target``
        # calls with that goal skip the A* attempt for a window so
        # an autonomous parent agent that keeps proposing the same
        # impossible voxel doesn't burn the agent loop's per-tick
        # budget on full pathfinds. Entries expire after
        # ``failed_target_ttl_ticks``.
        self._failed_targets: Dict[Tuple[int, int, int], int] = {}

    # ── Setup ────────────────────────────────────────────────────────

    def attach_calibrator_fn(self, fn) -> None:
        """``fn`` should return current px/degree (float). Called once
        per tick; the walker uses the returned value to convert the
        deg/sec velocity command to a px/sec mouse rate."""
        self._px_per_deg = fn

    def set_target(self,
                   goal: Tuple[int, int, int],
                   *,
                   eye_voxel: Optional[Tuple[int, int, int]] = None,
                   ) -> WalkerStatus:
        """
        Plan a path to ``goal``. If ``eye_voxel`` is supplied, plan
        from there; otherwise the walker will plan lazily on the
        first ``tick()`` call (when we have a fresh F3 pose).

        Returns the new status.

        If the goal voxel is currently in the failed-target cooldown
        table the call is rejected outright (returns BLOCKED) so an
        autonomous parent agent that keeps reproposing the same
        impossible voxel doesn't trigger a full A* every call. The
        cooldown is keyed by tick count, so it survives across
        ticks but expires after ``failed_target_ttl_ticks``.
        """
        # Expire stale cooldown entries before the lookup so we never
        # over-block a recovered geometry.
        if self._failed_targets:
            ttl = self.cfg.failed_target_ttl_ticks
            expired = [k for k, t in self._failed_targets.items()
                       if self._tick_count - t > ttl]
            for k in expired:
                self._failed_targets.pop(k, None)
        if goal in self._failed_targets:
            remaining = self.cfg.failed_target_ttl_ticks - (
                self._tick_count - self._failed_targets[goal])
            print(f"[pathwalker] set_target({goal}) refused — goal is in "
                  f"failed-target cooldown for {remaining} more ticks")
            self.status = WalkerStatus.BLOCKED
            return self.status
        # Warn on a mid-walk target change: the caller is rebinding
        # while the walker still has an active plan. Wiping state
        # silently could mask a parent-agent state-machine bug; the
        # behaviour itself is intentional (latest target wins).
        if self.status == WalkerStatus.WALKING and self._goal != goal:
            print(f"[pathwalker] set_target({goal}) called mid-walk; "
                  f"previous goal={self._goal} wp_idx={self._wp_idx} "
                  f"path_len={len(self._path)} — replanning")
        self._goal = goal
        self._path = []
        self._wp_idx = 0
        self._plan_attempts = 0
        self._stuck_pos = None
        self._stuck_fresh_ticks_at_pos = 0
        # Reset the freshness key so the new target's stuck-window
        # starts counting from the next FRESH F3 read, not the
        # cached one from the previous target.
        self._last_pose_ts = None
        if eye_voxel is not None:
            self._replan(eye_voxel)
        else:
            self.status = WalkerStatus.PLANNING
        return self.status

    # ── Per-tick entry ──────────────────────────────────────────────

    def tick(self, f3: Optional[F3Info]) -> AgentAction:
        self._tick_count += 1
        # Diagnostic: on tick 1, log entry conditions so a silently-
        # idle walker is debuggable. After that we only log on the
        # 40-tick cadence further down.
        if self._tick_count == 1:
            f3_ok = (f3 is not None and f3.x is not None
                     and f3.y is not None and f3.z is not None
                     and f3.yaw is not None)
            print(f"[pathwalker] tick=1 status={self.status.value} "
                  f"goal={self._goal} f3_ok={f3_ok} "
                  f"path_len={len(self._path)}")

        if self.status in (WalkerStatus.IDLE,
                            WalkerStatus.ARRIVED,
                            WalkerStatus.BLOCKED):
            return self._halt()

        # No fresh pose? Release every movement key and wait. Holding
        # W into unknown terrain while the OCR catches up risks
        # walking off cliffs / into lava.
        if f3 is None or f3.x is None or f3.y is None or f3.z is None \
                or f3.yaw is None:
            if self._tick_count % 80 == 1:
                print(f"[pathwalker] tick={self._tick_count} idle: "
                      f"no F3 pose yet (f3={f3 is not None}, "
                      f"x={getattr(f3,'x',None)}, "
                      f"yaw={getattr(f3,'yaw',None)})")
            return AgentAction(movement=BaseAgent.release_all_movement())

        eye_pos = (float(f3.x), float(f3.y), float(f3.z))
        eye_voxel = (int(math.floor(eye_pos[0])),
                     int(math.floor(eye_pos[1])),
                     int(math.floor(eye_pos[2])))
        cur_yaw = float(f3.yaw)

        # Lazy plan: if PLANNING, do it now.
        if self.status == WalkerStatus.PLANNING:
            self._replan(eye_voxel)
            if self.status != WalkerStatus.WALKING:
                return self._halt()

        # Arrival check (horizontal Manhattan within tol; vertical
        # within tol). We compare against the GOAL, not the current
        # waypoint, so a goal exactly between waypoints still triggers.
        if self._goal is not None:
            gx, gy, gz = self._goal
            horiz_d = math.hypot((gx + 0.5) - eye_pos[0],
                                 (gz + 0.5) - eye_pos[2])
            y_d = abs((gy + 0.5) - (eye_pos[1] + 0.5))
            if (horiz_d <= self.cfg.arrived_tolerance
                    and y_d <= self.cfg.arrived_y_tol):
                self.status = WalkerStatus.ARRIVED
                return AgentAction(movement=BaseAgent.release_all_movement(),
                                   look_vx=0.0, look_vy=0.0,
                                   force_velocity=True)

        # Advance waypoint index past anything we've already passed.
        self._advance_waypoint_idx(eye_pos)

        # Y-drop replan: if the player's actual Y is more than
        # ``unexpected_y_drop_blocks`` below the current waypoint's Y,
        # the terrain didn't match the optimistic plan (we walked off
        # a cliff the passable-policy didn't know about). Replan
        # immediately from the new pose rather than wait for the
        # stuck-timer — the existing path is geometrically wrong
        # now, and continuing to aim at the old waypoint produces
        # confusing motion (yaw swings + forward into terrain).
        if self._wp_idx < len(self._path):
            wp_y = self._path[self._wp_idx][1]
            drop = wp_y - eye_voxel[1]
            if drop > self.cfg.unexpected_y_drop_blocks:
                print(f"[pathwalker] Y-drop {drop} blocks below current "
                      f"waypoint {self._path[self._wp_idx]} (eye at "
                      f"Y={eye_voxel[1]}) — replanning from new pose")
                self._plan_attempts = 0   # not a stuck event, just terrain
                self._replan(eye_voxel)
                self._stuck_pos = eye_pos
                self._stuck_fresh_ticks_at_pos = 0
                if self.status != WalkerStatus.WALKING:
                    return self._halt()

        # Periodic status diagnostic — every 20 ticks (~4 s at 5 Hz)
        # so a long live run produces enough breadcrumbs to debug
        # "agent didn't move" issues without spamming. We log
        # again at the END of tick() to include the dispatched
        # action — that's the actual source of truth for whether
        # W is being held / what mouse velocity is being commanded.
        self._diag_due = (self._tick_count % 20 == 1)
        if self._diag_due:
            wp = (self._path[self._wp_idx]
                  if self._wp_idx < len(self._path) else None)
            print(f"[pathwalker] tick={self._tick_count} status={self.status.value} "
                  f"eye=({eye_pos[0]:+.1f},{eye_pos[1]:+.1f},{eye_pos[2]:+.1f}) "
                  f"yaw={cur_yaw:+.0f} goal={self._goal} wp[{self._wp_idx}]={wp}")

        # Stuck detection — measure progress over the recent FRESH-F3
        # window. F3 OCR runs at ~3 Hz; main.py hands us the same
        # F3Info on every cached tick in between. Counting cached
        # ticks toward "no progress" would falsely trigger stuck at
        # 20 Hz the moment the player paused. We use the F3
        # timestamp to count only fresh reads.
        ts = getattr(f3, "timestamp", None)
        pose_fresh = (ts is None
                      or ts != self._last_pose_ts)
        if pose_fresh:
            self._last_pose_ts = ts
            if self._stuck_pos is None:
                self._stuck_pos = eye_pos
                self._stuck_fresh_ticks_at_pos = 0
            else:
                dpos = math.hypot(eye_pos[0] - self._stuck_pos[0],
                                  eye_pos[2] - self._stuck_pos[2])
                if dpos > self.cfg.stuck_min_progress:
                    self._stuck_pos = eye_pos
                    self._stuck_fresh_ticks_at_pos = 0
                else:
                    self._stuck_fresh_ticks_at_pos += 1
                    if self._stuck_fresh_ticks_at_pos > self.cfg.stuck_progress_window:
                        # Stuck — replan from here.
                        self._plan_attempts += 1
                        if self._plan_attempts > self.cfg.stuck_replan_retries:
                            self.status = WalkerStatus.BLOCKED
                            return self._halt()
                        self._replan(eye_voxel)
                        self._stuck_pos = eye_pos
                        self._stuck_fresh_ticks_at_pos = 0
                        if self.status != WalkerStatus.WALKING:
                            return self._halt()

        # Compute aim to current waypoint.
        if self._wp_idx >= len(self._path):
            # Past the end — shouldn't normally happen because we
            # report ARRIVED above, but guard anyway.
            self.status = WalkerStatus.ARRIVED
            return self._halt()
        wp = self._path[self._wp_idx]
        wp_centre = (wp[0] + 0.5, wp[1] + 0.5, wp[2] + 0.5)
        dx_w = wp_centre[0] - eye_pos[0]
        dz_w = wp_centre[2] - eye_pos[2]
        # MC convention: yaw 0 looks at +Z, increases clockwise.
        desired_yaw = math.degrees(math.atan2(-dx_w, dz_w))
        yaw_err = _normalize_angle(desired_yaw - cur_yaw)
        self._last_aim_yaw = desired_yaw

        # Velocity command for rotation. Overshoot guard: when the
        # error is INSIDE the aim tolerance, hard-zero the yaw
        # velocity. With 3 Hz F3 OCR the agent only sees yaw
        # updates every ~333 ms; if the P-controller commands a
        # small but non-zero rate inside the tolerance, the camera
        # keeps rotating past the target and the next OCR cycle
        # sees a large opposite error, producing the
        # oscillation visible in live runs (yaw seesaws ±20°
        # around target indefinitely).
        max_dps = self.cfg.aim_max_deg_per_sec
        p_gain = self.cfg.aim_p_gain
        # ``<=`` rather than ``<``: live testing showed yaw_err
        # plateauing at exactly the tolerance value (e.g. -6.0 with
        # tolerance=6.0). With ``<`` neither branch fires the forward
        # press nor zeroes the velocity at the boundary, so the agent
        # keeps creeping past in tiny increments forever.
        if abs(yaw_err) <= self.cfg.aim_tolerance_deg:
            yaw_dps = 0.0
        else:
            yaw_dps = max(-max_dps, min(max_dps, yaw_err * p_gain))
        px_per_deg = self._current_px_per_deg()
        look_vx = yaw_dps * px_per_deg

        # Pitch-level controller — bring the camera back toward
        # ``target_pitch_deg`` in parallel with the yaw turn. Keeps
        # the head roughly level so F3 picks up the block in front
        # and so the agent doesn't sprint forward staring at the sky.
        # Same overshoot guard as yaw: zero velocity inside the
        # tolerance so 3 Hz OCR doesn't see post-overshoot drift.
        cur_pitch = float(getattr(f3, "pitch", 0.0) or 0.0)
        pitch_err = self.cfg.target_pitch_deg - cur_pitch
        pitch_max = self.cfg.pitch_max_deg_per_sec
        if abs(pitch_err) <= self.cfg.aim_tolerance_deg:
            pitch_dps = 0.0
        else:
            pitch_dps = max(-pitch_max, min(pitch_max,
                                             pitch_err * self.cfg.pitch_p_gain))
        look_vy = pitch_dps * px_per_deg

        # Build the movement dict. We ALWAYS emit explicit True/False
        # for every key we want set this tick, because the runtime
        # leaves unset keys alone.
        movement: dict = {
            "forward": False,
            "backward": False,
            "left": False,
            "right": False,
            "jump": False,
            "sneak": False,
            "sprint": False,
        }

        if abs(yaw_err) <= self.cfg.aim_tolerance_deg:
            # Aimed — walk forward toward the waypoint. ``<=`` so the
            # err==tolerance boundary fires; live trace showed yaw
            # plateauing AT tolerance and the agent never pressing W.
            movement["forward"] = True
            # Jump if the next waypoint is higher than current voxel.
            if wp[1] > eye_voxel[1]:
                movement["jump"] = True

        # Defense in depth: main.py already clamps NaN/inf/extreme
        # values before dispatch, but a buggy P-controller divide-
        # by-near-zero or a calibrator returning inf could surface
        # garbage here. Snap to zero on non-finite so the action
        # is at least benign.
        if not math.isfinite(look_vx):
            look_vx = 0.0
        if not math.isfinite(look_vy):
            look_vy = 0.0
        action = AgentAction(
            movement=movement,
            look_vx=look_vx,
            look_vy=look_vy,
            force_velocity=True,
        )
        if getattr(self, "_diag_due", False):
            print(f"  -> action: forward={movement['forward']} "
                  f"jump={movement['jump']} "
                  f"yaw_err={yaw_err:+.0f} pitch_err={pitch_err:+.0f} "
                  f"look_v=({look_vx:+.0f},{look_vy:+.0f})")
        return action

    # ── BaseAgent-style helpers ─────────────────────────────────────

    def reset(self) -> None:
        self.status = WalkerStatus.IDLE
        self._goal = None
        self._path = []
        self._wp_idx = 0
        self._stuck_pos = None
        self._stuck_fresh_ticks_at_pos = 0
        self._last_pose_ts = None
        self._tick_count = 0
        self._plan_attempts = 0
        self._last_aim_yaw = None
        self.last_plan_result = None
        self._diag_due = False
        self._failed_targets.clear()

    # ── Internals ───────────────────────────────────────────────────

    def _replan(self, eye_voxel: Tuple[int, int, int]) -> None:
        if self._goal is None:
            self.status = WalkerStatus.IDLE
            return
        # First try the exact goal.
        res = find_path(
            self.world_map, eye_voxel, self._goal,
            dimension=self.dimension, config=self.cfg.pathfind,
        )
        # Goal-rescue: if the exact goal isn't reachable but a NEAR
        # voxel is, try a small ±Y / ±1 horizontal sweep around the
        # goal before giving up. Common reason for the exact goal
        # failing: air-carving observed the floor as AIR (the agent
        # looked down through it earlier), so the goal cell's
        # standability check rejects it. The agent ALMOST got there
        # — landing one cell up or to the side is usually fine.
        if not res and self.cfg.goal_rescue_radius > 0:
            gx, gy, gz = self._goal
            best: Optional[PathResult] = None
            for dy in range(-self.cfg.goal_rescue_radius,
                             self.cfg.goal_rescue_radius + 1):
                for dx in range(-self.cfg.goal_rescue_radius,
                                 self.cfg.goal_rescue_radius + 1):
                    for dz in range(-self.cfg.goal_rescue_radius,
                                     self.cfg.goal_rescue_radius + 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        alt = (gx + dx, gy + dy, gz + dz)
                        cand = find_path(
                            self.world_map, eye_voxel, alt,
                            dimension=self.dimension,
                            config=self.cfg.pathfind,
                        )
                        if cand:
                            # Take the LOWEST-cost rescue (closest
                            # neighbour the agent can actually reach).
                            if best is None or cand.cost < best.cost:
                                best = cand
            if best is not None:
                print(f"[pathwalker] exact goal {self._goal} unreachable, "
                      f"rescued via nearby alternative end="
                      f"{best.waypoints[-1]} cost={best.cost:.1f}")
                res = best
        self.last_plan_result = res
        if not res:
            self.status = WalkerStatus.BLOCKED
            self._path = []
            # Record this goal in the cooldown table so an autonomous
            # parent agent reproposing the same impossible voxel
            # doesn't burn the agent loop's per-tick budget on full
            # A* attempts. The cooldown expires after
            # ``failed_target_ttl_ticks`` (default 600 ticks = ~30 s).
            if self._goal is not None:
                self._failed_targets[self._goal] = self._tick_count
            # Surface the failure reason so the agent doesn't silently
            # become a no-op. find_path returns ``reason`` from
            # {"blocked", "budget", "ok"} — both failure modes deserve
            # a one-line breadcrumb.
            print(f"[pathwalker] PLAN FAILED: start={eye_voxel} "
                  f"goal={self._goal} reason={res.reason} "
                  f"nodes={res.nodes_expanded}")
            return
        self._path = list(res.waypoints)
        self._wp_idx = 0
        # Skip waypoint[0] if it IS the eye voxel — we don't need to
        # walk to "where we already are".
        if self._path and self._path[0] == eye_voxel and len(self._path) > 1:
            self._wp_idx = 1
        self.status = WalkerStatus.WALKING
        print(f"[pathwalker] PLAN OK: {len(self._path)} waypoints  "
              f"start={eye_voxel} goal={self._goal} "
              f"cost={res.cost:.1f}")

    def _advance_waypoint_idx(self,
                              eye_pos: Tuple[float, float, float]) -> None:
        """Skip waypoints we've already reached. Look-ahead is bounded
        to two cells past the current index so a misread XYZ can't
        accidentally jump to the goal."""
        if not self._path:
            return
        max_look = min(len(self._path) - 1, self._wp_idx + 2)
        while self._wp_idx < max_look:
            wp = self._path[self._wp_idx]
            horiz = math.hypot((wp[0] + 0.5) - eye_pos[0],
                               (wp[2] + 0.5) - eye_pos[2])
            if horiz < self.cfg.waypoint_tolerance:
                self._wp_idx += 1
            else:
                break

    def _current_px_per_deg(self) -> float:
        if callable(self._px_per_deg):
            try:
                v = float(self._px_per_deg())
                if math.isfinite(v) and v > 0:
                    return v
            except Exception:
                pass
        return self.cfg.mouse_per_degree

    def _halt(self) -> AgentAction:
        return AgentAction(
            movement=BaseAgent.release_all_movement(),
            look_vx=0.0, look_vy=0.0, force_velocity=True,
        )


def _normalize_angle(deg: float) -> float:
    while deg >  180.0: deg -= 360.0
    while deg < -180.0: deg += 360.0
    return deg


# ---------------------------------------------------------------------------
# BaseAgent wrapper
# ---------------------------------------------------------------------------


class PathWalkerAgent(BaseAgent):
    """
    BaseAgent shell that points a ``WalkerController`` at a fixed
    target voxel read from settings (``agent.pathwalker.target_x/y/z``).
    Useful for ``python main.py --agent pathwalker`` smoke tests.

    The older ``navigation`` agent stays in place — it's a much simpler
    "turn-and-walk-in-a-straight-line" baseline that doesn't use the
    WorldMap. ``pathwalker`` plans a full A* path through the observed
    voxels and handles step-ups / drops / stuck-replans.

    For autonomous behaviour (pick own targets), build on top of
    ``WalkerController`` directly.
    """

    name = "pathwalker"

    def __init__(self,
                 target: Optional[Tuple[int, int, int]] = None,
                 *,
                 config: Optional[WalkerConfig] = None,
                 tick_hz: float = 20.0):
        self.target = target
        self.cfg = config or WalkerConfig()
        # Loop period used to convert the controller's px/sec look
        # velocity into px-this-tick for the calibrator. MUST match the
        # real agent-loop rate (``agent.tick_rate``) or the learned
        # px/degree is scaled by ``real_hz / tick_hz`` and every mouse
        # command inherits that error. Plumbed from settings via
        # ``build_pathwalker_agent`` instead of being hard-coded to 20 Hz.
        self._tick_period = 1.0 / float(tick_hz) if tick_hz else 1.0 / 20.0
        self._controller: Optional[WalkerController] = None
        self._perception_ref = None     # set by build_pathwalker_agent
        # Mouse-to-degree auto-calibrator. Without it the walker
        # would use the static ``mouse_per_degree`` default (6.5),
        # which is roughly 5x the actual rate on the user's machine
        # (real ~3 px/deg). Result: the walker thought it was
        # commanding 30°/s and got ~6°/s in practice — 180° turns
        # took 30 s instead of 6 s. The calibrator learns from
        # observed yaw / pitch deltas vs emitted mouse motion and
        # converges within ~50 fresh F3 reads.
        from control.mouse_calibration import (
            MouseCalibrator,
            MouseCalibrationConfig,
        )
        import os as _os
        self.calibrator = MouseCalibrator(
            cfg=MouseCalibrationConfig(
                default_px_per_deg=self.cfg.mouse_per_degree,
            ),
            persist_path=_os.path.join(
                "data", "calibration", "mouse_calibration.json",
            ),
        )
        # Cached for the per-tick observed-pose feed.
        self._last_pose_ts_for_calib: Optional[float] = None

    def attach_perception(self, perception) -> None:
        self._perception_ref = perception

    def reset(self) -> None:
        # Defer controller construction until we have a perception
        # reference (we need its world_map). Silent no-op is bad UX
        # — surface the "agent can't operate" reason once so a user
        # running ``--agent pathwalker`` without ``vision.world.enabled``
        # in settings learns WHY their bot does nothing.
        if self._perception_ref is None:
            self._controller = None
            print("[pathwalker][WARN] no perception reference attached "
                  "(set ``vision.world.enabled: true`` in settings or "
                  "wire perception manually). Agent will idle.")
            return
        wm = getattr(self._perception_ref, "world_map", None)
        if wm is None:
            self._controller = None
            print("[pathwalker][WARN] perception has no world_map; "
                  "agent will idle.")
            return
        self._controller = WalkerController(wm, config=self.cfg)
        # Wire the calibrator's per-tick estimate into the controller
        # so each yaw/pitch P-controller emission uses the latest
        # observed pixels-per-degree.
        self._controller.attach_calibrator_fn(self.calibrator.px_per_deg_yaw)
        # Banner so the user can see the calibration state.
        s = self.calibrator.stats()
        if s["calibrated"]:
            print(f"[pathwalker] calibrator loaded: yaw="
                  f"{s['px_per_deg_yaw']:.3f} px/deg pitch="
                  f"{s['px_per_deg_pitch']:.3f} px/deg "
                  f"({s['yaw_samples']} yaw + {s['pitch_samples']} "
                  f"pitch samples)")
        else:
            print(f"[pathwalker] calibrator cold-start at default "
                  f"{self.cfg.mouse_per_degree:.2f} px/deg "
                  f"(learning from observed motion)")
        if self.target is not None:
            self._controller.set_target(self.target)
            print(f"[pathwalker] target set to {self.target}")
        else:
            print("[pathwalker] no target configured "
                  "(``agent.pathwalker.target_x/y/z`` in settings); "
                  "agent will idle. Use the controller API to set one.")

    def decide(self, state: GameState) -> AgentAction:
        if self._controller is None or state.screen_state != ScreenState.PLAYING:
            return AgentAction(movement=BaseAgent.release_all_movement())
        # Feed the calibrator with the fresh F3 pose (only when the
        # pose's timestamp is new — main.py re-attaches the same
        # F3Info between OCR cycles, which would falsely teach the
        # calibrator zero motion).
        f3 = state.f3
        if (f3 is not None
                and f3.yaw is not None and f3.pitch is not None):
            ts = getattr(f3, "timestamp", None)
            if ts is None or ts != self._last_pose_ts_for_calib:
                self._last_pose_ts_for_calib = ts
                try:
                    self.calibrator.observed_pose(
                        float(f3.yaw), float(f3.pitch),
                    )
                except Exception:
                    pass
        action = self._controller.tick(f3)
        # After the controller emits velocity, teach the calibrator
        # how much mouse delta corresponds to a single tick at that
        # velocity. Velocity is in px/sec; multiply by tick period
        # to get px-this-tick. The agent loop runs at ~50 ms per
        # tick (configured via ``agent.tick_rate`` in settings, plumbed
        # into ``self._tick_period`` at construction).
        if action.look_vx or action.look_vy:
            dx_eq = action.look_vx * self._tick_period
            dy_eq = action.look_vy * self._tick_period
            try:
                self.calibrator.emitted(dx_eq, dy_eq)
            except Exception:
                pass
        return action

    def shutdown(self) -> None:
        # Persist the calibrator on clean shutdown so the next run
        # starts warm. The MouseCalibrator already triggers a save on
        # each new sample; this is a final flush.
        try:
            self.calibrator._try_save()
        except Exception:
            pass
        # Surface the final calibrator state so the user can see
        # whether this session improved the estimate.
        s = self.calibrator.stats()
        print(f"[pathwalker] calibrator final: yaw={s['px_per_deg_yaw']:.3f} "
              f"px/deg pitch={s['px_per_deg_pitch']:.3f} px/deg "
              f"({s['yaw_samples']} yaw + {s['pitch_samples']} "
              f"pitch samples, calibrated={s['calibrated']})")


def build_pathwalker_agent(settings: dict) -> PathWalkerAgent:
    cfg_raw = ((settings or {}).get("agent", {}) or {}).get(
        "pathwalker", {}) or {}
    target: Optional[Tuple[int, int, int]] = None
    if all(k in cfg_raw and cfg_raw[k] is not None
           for k in ("target_x", "target_y", "target_z")):
        target = (int(cfg_raw["target_x"]),
                  int(cfg_raw["target_y"]),
                  int(cfg_raw["target_z"]))
    cfg = WalkerConfig()
    for key in ("arrived_tolerance", "arrived_y_tol",
                "waypoint_tolerance", "aim_tolerance_deg",
                "aim_p_gain", "aim_max_deg_per_sec",
                "target_pitch_deg", "pitch_p_gain",
                "pitch_max_deg_per_sec",
                "mouse_per_degree", "stuck_min_progress"):
        if key in cfg_raw and cfg_raw[key] is not None:
            setattr(cfg, key, float(cfg_raw[key]))
    for key in ("stuck_progress_window", "stuck_replan_retries",
                "max_plan_attempts", "failed_target_ttl_ticks",
                "unexpected_y_drop_blocks", "goal_rescue_radius"):
        if key in cfg_raw and cfg_raw[key] is not None:
            setattr(cfg, key, int(cfg_raw[key]))
    # Pathfinder sub-config. ``unknown_policy`` is the single most
    # impactful pathfinder knob — exposing it through YAML lets users
    # tighten safety in caves / loosen for open exploration without
    # editing code.
    policy = cfg_raw.get("unknown_policy")
    if policy in ("passable", "ground_only"):
        cfg.pathfind.unknown_policy = policy
    elif policy is not None:
        print(f"[pathwalker][WARN] unknown_policy={policy!r} not "
              f"recognised; using default {cfg.pathfind.unknown_policy!r}")
    for key in ("max_fall_blocks", "max_nodes_expanded"):
        if key in cfg_raw and cfg_raw[key] is not None:
            setattr(cfg.pathfind, key, int(cfg_raw[key]))
    # Honour the configured loop rate so the calibrator's px/sec → px/tick
    # conversion matches reality (default 20 Hz). Falls back gracefully on
    # a bad value.
    tick_rate = ((settings or {}).get("agent", {}) or {}).get("tick_rate", 20)
    try:
        tick_hz = float(tick_rate)
        if tick_hz <= 0:
            tick_hz = 20.0
    except (TypeError, ValueError):
        tick_hz = 20.0
    return PathWalkerAgent(target=target, config=cfg, tick_hz=tick_hz)


__all__ = [
    "WalkerStatus",
    "WalkerConfig",
    "WalkerController",
    "PathWalkerAgent",
    "build_pathwalker_agent",
]
