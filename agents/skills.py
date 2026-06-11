# agents/skills.py
"""
Reusable low-level SKILLS for the bot — the action primitives that
behaviours (tree-chopping, exploring) compose.

Design (futureproof + composable)
---------------------------------
A Skill is a TICK-DRIVEN mini-controller, exactly like ``WalkerController``:
each tick it's handed a :class:`SkillContext` (the current perception) and
returns a :class:`SkillResult` = one ``AgentAction`` to dispatch + a status
(RUNNING / DONE / FAILED / BLOCKED). It holds its own progress state across
ticks. This keeps everything:
  * non-blocking — the agent loop stays responsive (safety / panic work),
  * composable — a task FSM just runs the current skill until it's DONE,
  * testable — the FSM transitions + geometry are exercised offline with a
    synthetic context; only the on-screen *execution* needs a live game,
  * swappable — a learned policy can implement the same Skill interface.

What's here vs. planned: see ``docs/bot_actions.md`` for the full action
catalogue. This module implements the foundational, proven primitives
(select-role, look-at-voxel, eat, mine-block, pillar-up, bridge — the
last two factoring the working god-bridge mechanic, where bridging with
sneak held is just *safe* bridging). Higher-level skills build on these.

NOTE on sign conventions: look_dx>0 turns the view right (yaw increases),
look_dy>0 looks down (MC pitch increases). The angle math matches the
walker/god-bridge; signs are confirmed against the live game when a skill
is first run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from brain.interfaces import AgentAction
from vision.world.map import AIR_BLOCK

Voxel = Tuple[int, int, int]


class SkillStatus(Enum):
    RUNNING = "running"     # still working; keep ticking
    DONE = "done"           # succeeded
    FAILED = "failed"       # could not complete (e.g. timed out)
    BLOCKED = "blocked"     # needs something it doesn't have (no pose, no item)


@dataclass
class SkillResult:
    action: AgentAction
    status: SkillStatus
    info: str = ""


@dataclass
class SkillContext:
    """Everything a skill needs to decide, gathered by the agent each tick.

    ``pose`` is any object exposing ``x, y, z, yaw, pitch`` and an
    ``eye_y`` (the F3/PlayerPose), or ``None`` when pose is unknown.
    """
    pose: object = None
    world_map: object = None          # vision.world.map.WorldMap (or None)
    hotbar: object = None             # control.hotbar.HotbarManager (or None)
    looking_at: object = None         # F3 LookingAtBlock (.pos, .block_id) or None
    px_per_deg: float = 6.5           # mouse sensitivity (calibrated upstream)
    tick: int = 0
    dimension: Optional[str] = None


# ── Geometry helpers (pure, offline-testable) ──────────────────────────

def norm_angle(a: float) -> float:
    """Wrap degrees to (-180, 180]."""
    a = (a + 180.0) % 360.0 - 180.0
    return a + 360.0 if a <= -180.0 else a


def aim_angles(eye: Tuple[float, float, float],
               target: Tuple[float, float, float]) -> Tuple[float, float]:
    """Desired (yaw, pitch) in MC degrees to look from ``eye`` at
    ``target`` (both world xyz). MC: yaw 0 = +Z increasing clockwise;
    pitch +90 = straight down. atan2(dy,0) is well-defined (±90)."""
    dx = target[0] - eye[0]
    dy = target[1] - eye[1]
    dz = target[2] - eye[2]
    yaw = math.degrees(math.atan2(-dx, dz))
    horiz = math.hypot(dx, dz)
    pitch = math.degrees(-math.atan2(dy, horiz))
    return yaw, pitch


def _eye(pose) -> Optional[Tuple[float, float, float]]:
    if pose is None:
        return None
    try:
        ey = pose.eye_y if hasattr(pose, "eye_y") else (pose.y + 1.62)
        return (float(pose.x), float(ey), float(pose.z))
    except Exception:
        return None


# ── Skill base ─────────────────────────────────────────────────────────

class Skill:
    name: str = "skill"

    def reset(self) -> None:
        """Clear per-run progress so the skill can be reused."""

    def tick(self, ctx: SkillContext) -> SkillResult:   # pragma: no cover
        raise NotImplementedError

    # Convenience for subclasses.
    @staticmethod
    def _idle(status=SkillStatus.RUNNING, info="") -> SkillResult:
        return SkillResult(AgentAction(), status, info)


class _Aimer:
    """Shared aim helper: one-shot proportional look toward (yaw,pitch).
    Returns (look_dx, look_dy, aimed?) for a target angle pair."""

    def __init__(self, tol_deg: float = 2.0, gain: float = 0.6,
                 max_px: int = 140):
        self.tol = tol_deg
        self.gain = gain
        self.max_px = max_px
        self._last_pose = None        # (yaw,pitch) we last issued a move at

    def step(self, ctx: SkillContext, want_yaw: float, want_pitch: float
             ) -> Tuple[int, int, bool]:
        p = ctx.pose
        cur = (round(float(p.yaw), 2), round(float(p.pitch), 2))
        yaw_err = norm_angle(want_yaw - float(p.yaw))
        pitch_err = float(want_pitch) - float(p.pitch)
        aimed = abs(yaw_err) <= self.tol and abs(pitch_err) <= self.tol
        if aimed:
            self._last_pose = None        # no correction pending
            return 0, 0, True
        # Stale-pose guard. Pose feedback (F3 OCR, a few Hz) lags the
        # control tick (~12-20 Hz). After we ISSUE a camera correction, the
        # pose won't reflect it for a tick or two; issuing another in the
        # meantime STACKS corrections and overshoots/oscillates (the
        # crosshair wanders diagonally and never settles). So once we've
        # issued a move, wait until the pose actually changes before issuing
        # the next. (Only set when we issue a non-zero look — a stationary
        # pose while merely walking must NOT block the next turn.)
        if self._last_pose is not None and cur == self._last_pose:
            return 0, 0, False
        self._last_pose = cur
        ppd = max(0.5, float(ctx.px_per_deg))
        dx = int(max(-self.max_px, min(self.max_px, yaw_err * ppd * self.gain)))
        dy = int(max(-self.max_px, min(self.max_px, pitch_err * ppd * self.gain)))
        return dx, dy, False


# ── Concrete skills ────────────────────────────────────────────────────

class SelectRole(Skill):
    """Select the hotbar slot reserved for a role (axe/blocks/food/…).
    DONE once the held slot matches; FAILED if no such item exists."""
    name = "select_role"

    def __init__(self, role: str):
        self.role = role

    def tick(self, ctx: SkillContext) -> SkillResult:
        hb = ctx.hotbar
        if hb is None:
            return SkillResult(AgentAction(), SkillStatus.BLOCKED, "no hotbar")
        slot = hb.slot_for_role(self.role)
        if slot is None:
            return SkillResult(AgentAction(), SkillStatus.FAILED,
                               f"no {self.role} in hotbar")
        return SkillResult(AgentAction(hotbar=slot), SkillStatus.DONE,
                           f"selected slot {slot} for {self.role}")


class LookAtVoxel(Skill):
    """Aim the crosshair at the centre of a world voxel. DONE when within
    tolerance, BLOCKED without a pose. Foundation for mining/placing."""
    name = "look_at_voxel"

    def __init__(self, voxel: Voxel, tol_deg: float = 2.0):
        self.voxel = voxel
        self.aim = _Aimer(tol_deg=tol_deg)

    def tick(self, ctx: SkillContext) -> SkillResult:
        eye = _eye(ctx.pose)
        if eye is None:
            return SkillResult(AgentAction(), SkillStatus.BLOCKED, "no pose")
        tgt = (self.voxel[0] + 0.5, self.voxel[1] + 0.5, self.voxel[2] + 0.5)
        yaw, pitch = aim_angles(eye, tgt)
        dx, dy, aimed = self.aim.step(ctx, yaw, pitch)
        if aimed:
            return SkillResult(AgentAction(), SkillStatus.DONE, "aimed")
        return SkillResult(AgentAction(look_dx=dx, look_dy=dy),
                           SkillStatus.RUNNING, f"aiming dx={dx} dy={dy}")


class Eat(Skill):
    """Select the food slot, then hold use-item for ``hold_ticks`` (eating
    takes ~1.6 s ≈ 32 ticks @20 Hz). DONE after the hold; FAILED if no
    food. (Whether hunger actually refilled is verified by the caller via
    the HUD; this skill just performs the eat action reliably.)"""
    name = "eat"

    def __init__(self, hold_ticks: int = 34):
        self.hold_ticks = hold_ticks
        self._held = 0
        self._selected = False

    def reset(self):
        self._held = 0; self._selected = False

    def tick(self, ctx: SkillContext) -> SkillResult:
        hb = ctx.hotbar
        slot = hb.best_slot_for("food") if hb is not None else None
        if slot is None:
            return SkillResult(AgentAction(), SkillStatus.FAILED, "no food")
        if not self._selected:
            self._selected = True
            return SkillResult(AgentAction(hotbar=slot), SkillStatus.RUNNING,
                               f"select food slot {slot}")
        self._held += 1
        if self._held >= self.hold_ticks:
            return SkillResult(AgentAction(), SkillStatus.DONE, "ate")
        # Sustained right-click hold to eat ("use_hold" — main holds the
        # button down; a per-tick click would restart the eat each tick).
        return SkillResult(AgentAction(interact="use_hold"),
                           SkillStatus.RUNNING, f"eating {self._held}/{self.hold_ticks}")


class MineBlock(Skill):
    """Mine the block at ``voxel``: select the right tool (``tool_role``,
    e.g. 'axe' for logs), aim at it, then hold attack until the voxel
    becomes AIR in the WorldMap (or vanishes) — FAILED on timeout.

    The mined-check is WorldMap-driven so it's verifiable offline; live,
    the world recogniser / F3 carve the voxel to air once it breaks."""
    name = "mine_block"

    def __init__(self, voxel: Voxel, tool_role: Optional[str] = "axe",
                 max_ticks: int = 200, tol_deg: float = 3.0,
                 is_safe=None, is_target=None, is_passthrough=None):
        self.voxel = voxel
        self.tool_role = tool_role
        self.max_ticks = max_ticks
        # Two predicates (block_id -> bool), verified against F3's targeted
        # block: is_target = what the TARGET must be to mine (a LOG);
        # is_passthrough = blocks we may break THROUGH to reach it (leaves).
        # ``is_safe`` is the simple alias that sets both.
        self.is_target = is_target if is_target is not None else is_safe
        self.is_passthrough = (is_passthrough if is_passthrough is not None
                               else (is_safe if is_safe is not None else self.is_target))
        self.aim = _Aimer(tol_deg=tol_deg)
        self._tool_ok = tool_role is None
        # State machine: 'aim' (the ONLY state that moves the camera) ->
        # 'mine_log' / 'clear_leaf' (camera FROZEN, click held). Latching
        # like this means: the instant F3 shows a log we stop moving and
        # just mine — the camera never drifts off the block mid-swing.
        self._mode = "aim"
        self._mining_ticks = 0
        self._silent = 0               # F3-gap ticks while frozen+mining
        self._await_ticks = 0          # aimed-but-F3-silent ticks (acquisition)
        self.broke = False             # did a LOG actually break? (for counting)

    def reset(self):
        self._tool_ok = self.tool_role is None
        self._mode = "aim"
        self._mining_ticks = 0
        self._silent = 0
        self._await_ticks = 0
        self.broke = False

    def _is_log(self, bid) -> bool:
        return bid is not None and (self.is_target is None or self.is_target(bid))

    def _is_pass(self, bid) -> bool:
        return bid is not None and (self.is_passthrough is None or self.is_passthrough(bid))

    def tick(self, ctx: SkillContext) -> SkillResult:
        la = ctx.looking_at
        la_id = getattr(la, "block_id", None) if la is not None else None
        la_pos = tuple(la.pos) if (la is not None and getattr(la, "pos", None) is not None) else None
        on_target = (la_pos == tuple(self.voxel))

        def _hold(info):                 # camera FROZEN (no look), click held
            self._mining_ticks += 1
            if self._mining_ticks > self.max_ticks:
                return SkillResult(AgentAction(), SkillStatus.FAILED, "timed out mining")
            return SkillResult(AgentAction(interact="attack"), SkillStatus.RUNNING, info)

        # ── MINING a log: frozen camera, hold click until the log is gone ──
        if self._mode == "mine_log":
            if self._is_log(la_id):
                self._silent = 0
                return _hold("mining log")
            self._silent += 1
            if self._silent <= 3:        # ride out a brief F3 OCR gap, still frozen
                return _hold("mining log (F3 gap)")
            self.broke = True            # the log we held on is gone -> counted
            return SkillResult(AgentAction(), SkillStatus.DONE, "mined log")

        # ── CLEARING an occluding leaf: frozen, hold click; NOT counted ──
        if self._mode == "clear_leaf":
            if self._is_pass(la_id) and not self._is_log(la_id):
                self._silent = 0
                return _hold("clearing leaf")
            self._mode = "aim"; self._silent = 0   # leaf gone / now a log -> re-evaluate

        if _eye(ctx.pose) is None:
            return SkillResult(AgentAction(), SkillStatus.BLOCKED, "no pose")
        if not self._tool_ok:
            hb = ctx.hotbar
            slot = (hb.best_slot_for(self.tool_role) if hb is not None else None)
            self._tool_ok = True
            if slot is not None:
                return SkillResult(AgentAction(hotbar=slot), SkillStatus.RUNNING,
                                   f"select {self.tool_role} slot {slot}")

        # THE MOMENT F3 shows a LOG under the crosshair: LATCH -> freeze + mine.
        if self._is_log(la_id):
            self._mode = "mine_log"; self._silent = 0
            return _hold("log in crosshair -> mining")

        # Otherwise AIM toward the target voxel — the only state that moves.
        eye = _eye(ctx.pose)
        tgt = (self.voxel[0] + 0.5, self.voxel[1] + 0.5, self.voxel[2] + 0.5)
        yaw, pitch = aim_angles(eye, tgt)
        dx, dy, aimed = self.aim.step(ctx, yaw, pitch)
        locked = on_target or aimed

        if la_id is not None:            # a NON-log block under the crosshair
            if not locked:
                return SkillResult(AgentAction(look_dx=dx, look_dy=dy),
                                   SkillStatus.RUNNING, "aiming at target")
            if on_target:                # the target voxel itself isn't a log
                return SkillResult(AgentAction(), SkillStatus.FAILED,
                                   f"target is {la_id}, not a log — abandon")
            if self._is_pass(la_id):     # a leaf occluding the target log
                self._mode = "clear_leaf"; self._silent = 0
                return _hold("leaf occludes target -> clearing")
            return SkillResult(AgentAction(), SkillStatus.FAILED,
                               f"{la_id} blocks the target — abandon")

        # F3 silent (crosshair on sky / unreadable).
        if not locked:
            return SkillResult(AgentAction(look_dx=dx, look_dy=dy),
                               SkillStatus.RUNNING, "aiming at target")
        self._await_ticks += 1
        if self._await_ticks > 10:
            return SkillResult(AgentAction(), SkillStatus.FAILED,
                               "locked but F3 showed no block — abandon")
        return SkillResult(AgentAction(), SkillStatus.RUNNING, "locked; awaiting F3")


class PillarUp(Skill):
    """Build straight up ``height`` blocks (the god-bridge pillar mechanic):
    per block — select the blocks slot, look ~straight down, place a block
    under the feet (use-item) while jumping onto it, then confirm the
    player's Y rose by ~1. DONE after ``height`` blocks, FAILED if a block
    won't place (no upward progress within a budget). Used to climb OUT of
    a hole or to gain height."""
    name = "pillar_up"

    def __init__(self, height: int = 1, place_pitch: float = 80.0,
                 per_block_budget: int = 40):
        self.height = height
        self.place_pitch = place_pitch
        self.per_block_budget = per_block_budget
        self.aim = _Aimer(tol_deg=4.0)
        self._done_blocks = 0
        self._base_y: Optional[float] = None
        self._ticks_this_block = 0
        self._selected = False

    def reset(self):
        self._done_blocks = 0; self._base_y = None
        self._ticks_this_block = 0; self._selected = False

    def tick(self, ctx: SkillContext) -> SkillResult:
        p = ctx.pose
        if p is None:
            return SkillResult(AgentAction(), SkillStatus.BLOCKED, "no pose")
        if self._base_y is None:
            self._base_y = float(p.y)
        if self._done_blocks >= self.height:
            return SkillResult(AgentAction(), SkillStatus.DONE,
                               f"pillared {self.height}")
        # Confirm a finished block: Y rose ~1 above this block's base.
        if float(p.y) >= self._base_y + 0.9:
            self._done_blocks += 1
            self._base_y = float(p.y)
            self._ticks_this_block = 0
            self._selected = False
            if self._done_blocks >= self.height:
                return SkillResult(AgentAction(), SkillStatus.DONE,
                                   f"pillared {self.height}")
        self._ticks_this_block += 1
        if self._ticks_this_block > self.per_block_budget:
            return SkillResult(AgentAction(), SkillStatus.FAILED,
                               "no upward progress (blocked / out of blocks)")
        # Select blocks slot once.
        hb = ctx.hotbar
        if not self._selected and hb is not None:
            slot = hb.best_slot_for("blocks")
            self._selected = True
            if slot is None:
                return SkillResult(AgentAction(), SkillStatus.FAILED, "no blocks")
            return SkillResult(AgentAction(hotbar=slot), SkillStatus.RUNNING,
                               f"select blocks slot {slot}")
        # Look down, jump + place under feet.
        _, _, aimed = self.aim.step(ctx, float(p.yaw), self.place_pitch)
        dy = 0 if aimed else int((self.place_pitch - float(p.pitch))
                                 * ctx.px_per_deg * 0.6)
        return SkillResult(
            AgentAction(movement={"jump": True}, look_dy=dy, interact="use_item"),
            SkillStatus.RUNNING, f"placing block {self._done_blocks+1}/{self.height}")


class Bridge(Skill):
    """Extend a 1-wide bridge forward by ``length`` blocks. With
    ``keep_sneak=True`` this is SAFE bridging (sneak prevents walking off
    the edge); with ``keep_sneak=False`` it's the faster god-bridge. Per
    block: ensure blocks selected, hold backward/forward + sneak as
    configured, look down at the edge and place. Progress is measured by
    horizontal distance travelled from the start.

    The precise look/timing is tuned against the live game (see
    tools/run_god_bridge.py); this skill exposes it as a composable,
    parameterised primitive. Execution is verified in a live session."""
    name = "bridge"

    def __init__(self, length: int = 4, keep_sneak: bool = True,
                 place_pitch: float = 60.0, per_block_budget: int = 30):
        self.length = length
        self.keep_sneak = keep_sneak
        self.place_pitch = place_pitch
        self.per_block_budget = per_block_budget
        self._placed = 0
        self._start: Optional[Tuple[float, float]] = None
        self._ticks_this_block = 0
        self._selected = False
        self._max_dist = 0.0

    def reset(self):
        self._placed = 0; self._start = None
        self._ticks_this_block = 0; self._selected = False
        self._max_dist = 0.0

    def tick(self, ctx: SkillContext) -> SkillResult:
        p = ctx.pose
        if p is None:
            return SkillResult(AgentAction(), SkillStatus.BLOCKED, "no pose")
        if self._start is None:
            self._start = (float(p.x), float(p.z))
        dist = math.hypot(float(p.x) - self._start[0], float(p.z) - self._start[1])
        # Reset the stall budget on ANY real forward progress (not just at
        # whole-block crossings) so slow, steady sneak-bridging isn't failed.
        if dist > self._max_dist + 0.1:
            self._max_dist = dist
            self._ticks_this_block = 0
        placed_now = int(dist)
        if placed_now > self._placed:
            self._placed = placed_now
        if self._placed >= self.length:
            return SkillResult(AgentAction(movement={"sneak": self.keep_sneak,
                                                     "backward": False}),
                               SkillStatus.DONE, f"bridged {self.length}")
        self._ticks_this_block += 1
        if self._ticks_this_block > self.per_block_budget:
            return SkillResult(AgentAction(), SkillStatus.FAILED,
                               "bridge stalled (no forward progress)")
        hb = ctx.hotbar
        if not self._selected and hb is not None:
            slot = hb.best_slot_for("blocks")
            self._selected = True
            if slot is None:
                return SkillResult(AgentAction(), SkillStatus.FAILED, "no blocks")
            return SkillResult(AgentAction(hotbar=slot), SkillStatus.RUNNING,
                               f"select blocks slot {slot}")
        dy = int((self.place_pitch - float(p.pitch)) * ctx.px_per_deg * 0.6)
        # Move backward while facing forward + place = the god-bridge mechanic;
        # sneak makes it the safe variant.
        return SkillResult(
            AgentAction(movement={"backward": True, "sneak": self.keep_sneak},
                        look_dy=dy, interact="use_item"),
            SkillStatus.RUNNING,
            f"{'safe-' if self.keep_sneak else 'god-'}bridge {self._placed+1}/{self.length}")


class WalkToward(Skill):
    """Reactive 'approach a target voxel' locomotion: face the target
    (yaw), then hold forward until within ``arrive_dist`` horizontally.
    DONE on arrival; FAILED if no forward progress for a while (stuck), or
    if the next step would walk off a KNOWN edge (the block it would step
    onto is air in the WorldMap — basic fall avoidance).

    Robust for short approaches on roughly-flat ground without needing a
    pre-built A* path (the A* WalkerController is for complex routing).
    The locomotion-planner layer adds jump-over / pillar-out / mine-through
    on top of this primitive."""
    name = "walk_toward"

    def __init__(self, target: Voxel, arrive_dist: float = 1.6,
                 face_tol_deg: float = 14.0, stuck_window: int = 18,
                 min_progress: float = 0.12, avoid_fall: bool = True,
                 sprint: bool = True, jump_after: int = 4):
        self.target = tuple(target)
        self.arrive_dist = arrive_dist
        self.face_tol = face_tol_deg
        self.stuck_window = stuck_window
        self.min_progress = min_progress
        self.avoid_fall = avoid_fall
        self.sprint = sprint
        # When forward progress stalls for this many ticks, start jumping
        # while walking — MC then mantles over a 1-block step or out of a
        # 1-deep hole. (Deeper holes escalate to PillarUp in the planner.)
        self.jump_after = jump_after
        self.aim = _Aimer(tol_deg=face_tol_deg)
        self.reset()

    def reset(self):
        self._best_d = None
        self._stuck = 0

    def _edge_ahead(self, ctx: SkillContext, px, py, pz, tx, tz) -> bool:
        """True if a block we'd step onto next is a KNOWN air block (a
        drop). We walk along the actual direction to the target, so check
        every ground cell the foot could land on — the cell ~0.8 blocks
        ahead AND both cardinal neighbours when the move is diagonal — so a
        diagonal step can't sidestep the fall check. Unknown ground -> not
        an edge (proceed)."""
        wm = ctx.world_map
        if wm is None or not self.avoid_fall:
            return False
        dx, dz = tx - px, tz - pz
        mag = math.hypot(dx, dz)
        if mag < 1e-6:
            return False
        ux, uz = dx / mag, dz / mag
        foot = int(math.floor(py))
        fx, fz = int(math.floor(px)), int(math.floor(pz))
        ahead = (int(math.floor(px + ux * 0.8)), int(math.floor(pz + uz * 0.8)))
        cells = {ahead}
        if abs(dx) > 0.35 and abs(dz) > 0.35:        # diagonal -> guard both sides
            cells.add((fx + (1 if dx > 0 else -1), fz))
            cells.add((fx, fz + (1 if dz > 0 else -1)))
        for cx, cz in cells:
            if (cx, cz) == (fx, fz):                 # the cell we're already on
                continue
            ground = (cx, foot - 1, cz)
            try:
                obs = wm.get_block(ground, dimension=ctx.dimension)
            except TypeError:
                obs = wm.get_block(ground)
            if obs is not None and obs.block_id == AIR_BLOCK:
                return True
        return False

    def tick(self, ctx: SkillContext) -> SkillResult:
        p = ctx.pose
        if p is None:
            return SkillResult(AgentAction(), SkillStatus.BLOCKED, "no pose")
        px, pz = float(p.x), float(p.z)
        tx, tz = self.target[0] + 0.5, self.target[2] + 0.5
        dx, dz = tx - px, tz - pz
        dist = math.hypot(dx, dz)
        if dist <= self.arrive_dist:
            return SkillResult(AgentAction(movement={"forward": False}),
                               SkillStatus.DONE, f"arrived (d={dist:.1f})")
        # Progress / stuck tracking.
        if self._best_d is None or dist < self._best_d - self.min_progress:
            self._best_d = dist
            self._stuck = 0
        else:
            self._stuck += 1
        if self._stuck > self.stuck_window:
            return SkillResult(AgentAction(movement={"forward": False}),
                               SkillStatus.FAILED, f"stuck (d={dist:.1f})")
        if self._edge_ahead(ctx, px, float(p.y), pz, tx, tz):
            return SkillResult(AgentAction(movement={"forward": False}),
                               SkillStatus.FAILED, "edge ahead (would fall)")
        # Face the target (yaw); keep pitch ~level for walking.
        want_yaw = math.degrees(math.atan2(-(tx - px), (tz - pz)))
        ddx, ddy, facing = self.aim.step(ctx, want_yaw, 0.0)
        if not facing:
            # Turn in place first; don't walk off-course.
            return SkillResult(AgentAction(look_dx=ddx, look_dy=ddy,
                                           movement={"forward": False}),
                               SkillStatus.RUNNING, f"facing (d={dist:.1f})")
        # Facing -> walk forward (sprint by default), nudging yaw to stay on
        # course. If progress has stalled, jump too: MC mantles over a
        # 1-block step / out of a 1-deep hole. Don't jump while edge-avoiding
        # (we already returned above if a known drop is ahead).
        # Always specify jump explicitly (True only while stalled) — never
        # omit it, or a previously-set jump would stay held across ticks.
        jumping = self._stuck >= self.jump_after
        mv = {"forward": True, "sprint": self.sprint, "jump": jumping}
        info = "jump-walking" if jumping else "walking"
        return SkillResult(AgentAction(movement=mv, look_dx=ddx),
                           SkillStatus.RUNNING, f"{info} (d={dist:.1f})")


class ChopTrunk(Skill):
    """Chop a vertical run of logs IN PLACE (no walking): mine the start
    voxel, look up to the block above, and if it's still a log, mine that
    too — repeating up the trunk until the block above isn't a log (or a
    safety height cap). Composes the live-verified MineBlock + LookAtVoxel.

    The 'is the block above a log?' check reads F3's targeted block after
    aiming up, with a few ticks of slack for the ~3 Hz OCR to catch up."""
    name = "chop_trunk"

    def __init__(self, start_voxel: Voxel, is_log=None,
                 max_height: int = 10, tool_role: Optional[str] = "axe",
                 is_safe=None):
        self.start = tuple(start_voxel)
        self.is_log = is_log or (lambda b: bool(b) and str(b).endswith("_log"))
        self.max_height = max_height
        self.tool_role = tool_role
        # Safety gate passed to each MineBlock so trunk mining can't strike
        # terrain even if aim drifts (defaults to "only this skill's logs").
        self.is_safe = is_safe or self.is_log
        self.reset()

    def reset(self):
        self._target = self.start
        self._mined = 0
        self._phase = "mine"
        self._mine = MineBlock(self._target, tool_role=self.tool_role,
                               is_target=self.is_log, is_passthrough=self.is_safe)
        self._aim = None
        self._check_ticks = 0

    def tick(self, ctx: SkillContext) -> SkillResult:
        if self._mined >= self.max_height:
            return SkillResult(AgentAction(), SkillStatus.DONE,
                               f"reached height cap ({self._mined})")
        if self._phase == "mine":
            r = self._mine.tick(ctx)
            if r.status == SkillStatus.DONE:
                # Only advance up the trunk + count when a LOG actually broke.
                # A DONE without a confirmed break (brief F3 loss, etc.) ends
                # the trunk without inflating the count.
                if not getattr(self._mine, "broke", False):
                    return SkillResult(r.action, SkillStatus.DONE,
                                       f"trunk done ({self._mined} logs)")
                self._mined += 1
                self._target = (self._target[0], self._target[1] + 1, self._target[2])
                self._aim = LookAtVoxel(self._target, tol_deg=3.0)
                self._phase = "aim_up"; self._check_ticks = 0
                return SkillResult(r.action, SkillStatus.RUNNING,
                                   f"mined {self._mined}; looking up")
            return r                          # RUNNING / FAILED / BLOCKED
        if self._phase == "aim_up":
            r = self._aim.tick(ctx)
            if r.status == SkillStatus.DONE:
                self._phase = "check"
                return SkillResult(AgentAction(), SkillStatus.RUNNING,
                                   "checking for log above")
            return r
        # phase == "check": wait for F3 to confirm a log at the new target.
        self._check_ticks += 1
        la = ctx.looking_at
        if la is not None and getattr(la, "pos", None) is not None \
                and tuple(la.pos) == self._target and self.is_log(la.block_id):
            self._mine = MineBlock(self._target, tool_role=self.tool_role,
                                   is_target=self.is_log, is_passthrough=self.is_safe)
            self._phase = "mine"; self._check_ticks = 0
            return SkillResult(AgentAction(), SkillStatus.RUNNING,
                               f"log above at {self._target}; mining")
        if self._check_ticks >= 8:
            return SkillResult(AgentAction(), SkillStatus.DONE,
                               f"trunk cleared ({self._mined} logs)")
        # hold aim while the OCR catches up
        return (self._aim.tick(ctx) if self._aim is not None
                else SkillResult(AgentAction(), SkillStatus.RUNNING, "wait"))


class SkillSequence(Skill):
    """Run a list of skills in order — the glue that composes primitives
    into a behaviour. Advances to the next skill when the current one is
    DONE; the sequence is DONE when all finish. On a child FAILED/BLOCKED
    it stops with that status (a task FSM decides what to do next). Being
    itself a Skill, sequences nest."""
    name = "sequence"

    def __init__(self, skills, stop_on_block: bool = True):
        self._skills = list(skills)
        self._i = 0
        self.stop_on_block = stop_on_block

    def reset(self):
        self._i = 0
        for s in self._skills:
            s.reset()

    @property
    def current(self) -> Optional[Skill]:
        return self._skills[self._i] if self._i < len(self._skills) else None

    def tick(self, ctx: SkillContext) -> SkillResult:
        if self._i >= len(self._skills):
            return SkillResult(AgentAction(), SkillStatus.DONE, "sequence done")
        res = self._skills[self._i].tick(ctx)
        if res.status == SkillStatus.DONE:
            self._i += 1
            label = f"step {self._i}/{len(self._skills)} done"
            if self._i >= len(self._skills):
                return SkillResult(res.action, SkillStatus.DONE, "sequence done")
            # Carry this tick's action through; next skill starts next tick.
            return SkillResult(res.action, SkillStatus.RUNNING, label)
        if res.status == SkillStatus.FAILED or (
                res.status == SkillStatus.BLOCKED and self.stop_on_block):
            return SkillResult(res.action, res.status,
                               f"step {self._i+1} {res.status.value}: {res.info}")
        return res


def find_nearest_block(world_map, origin, match, *, max_radius: int = 48,
                       dimension: Optional[str] = None, exclude=None):
    """Nearest observed block satisfying ``match(block_id) -> bool`` within
    ``max_radius`` (Chebyshev) of ``origin`` voxel. Returns ``(voxel, obs)``
    or ``None``. Used to spot the closest oak log the recogniser has mapped.
    """
    if world_map is None:
        return None
    try:
        it = world_map.iter_blocks_in_range(origin, max_radius, dimension=dimension)
    except TypeError:
        it = world_map.iter_blocks_in_range(origin, max_radius)
    best = None
    best_d2 = None
    ox, oy, oz = origin
    for obs in it:
        bid = getattr(obs, "block_id", None)
        if bid == AIR_BLOCK or bid is None or not match(bid):
            continue
        if exclude is not None and tuple(obs.pos) in exclude:
            continue
        vx, vy, vz = obs.pos
        d2 = (vx - ox) ** 2 + (vy - oy) ** 2 + (vz - oz) ** 2
        if best_d2 is None or d2 < best_d2:
            best_d2, best = d2, (obs.pos, obs)
    return best


# Registry of the currently-implemented skills (name -> class). The task
# layer / a future planner can look skills up by name.
SKILLS = {s.name: s for s in (SelectRole, LookAtVoxel, Eat, MineBlock,
                              PillarUp, Bridge, WalkToward, ChopTrunk, SkillSequence)}


__all__ = [
    "SkillStatus", "SkillResult", "SkillContext", "Skill",
    "SelectRole", "LookAtVoxel", "Eat", "MineBlock", "PillarUp", "Bridge",
    "WalkToward", "ChopTrunk", "SkillSequence", "find_nearest_block",
    "aim_angles", "norm_angle", "SKILLS",
]
