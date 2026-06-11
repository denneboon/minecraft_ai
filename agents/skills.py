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

    def step(self, ctx: SkillContext, want_yaw: float, want_pitch: float
             ) -> Tuple[int, int, bool]:
        p = ctx.pose
        yaw_err = norm_angle(want_yaw - float(p.yaw))
        pitch_err = float(want_pitch) - float(p.pitch)
        aimed = abs(yaw_err) <= self.tol and abs(pitch_err) <= self.tol
        if aimed:
            return 0, 0, True
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
        slot = hb.food_slot() if hb is not None else None
        if slot is None:
            return SkillResult(AgentAction(), SkillStatus.FAILED, "no food")
        if not self._selected:
            self._selected = True
            return SkillResult(AgentAction(hotbar=slot), SkillStatus.RUNNING,
                               f"select food slot {slot}")
        self._held += 1
        if self._held >= self.hold_ticks:
            return SkillResult(AgentAction(), SkillStatus.DONE, "ate")
        # Right-click hold to eat.
        return SkillResult(AgentAction(interact="use_item"),
                           SkillStatus.RUNNING, f"eating {self._held}/{self.hold_ticks}")


class MineBlock(Skill):
    """Mine the block at ``voxel``: select the right tool (``tool_role``,
    e.g. 'axe' for logs), aim at it, then hold attack until the voxel
    becomes AIR in the WorldMap (or vanishes) — FAILED on timeout.

    The mined-check is WorldMap-driven so it's verifiable offline; live,
    the world recogniser / F3 carve the voxel to air once it breaks."""
    name = "mine_block"

    def __init__(self, voxel: Voxel, tool_role: Optional[str] = "axe",
                 max_ticks: int = 200, tol_deg: float = 3.0):
        self.voxel = voxel
        self.tool_role = tool_role
        self.max_ticks = max_ticks
        self.aim = _Aimer(tol_deg=tol_deg)
        self._tool_ok = tool_role is None
        self._mining_ticks = 0

    def reset(self):
        self._tool_ok = self.tool_role is None
        self._mining_ticks = 0

    def _is_gone(self, ctx: SkillContext) -> bool:
        wm = ctx.world_map
        if wm is None:
            return False
        try:
            obs = wm.get_block(self.voxel, dimension=ctx.dimension)
        except TypeError:
            obs = wm.get_block(self.voxel)
        return obs is not None and obs.block_id == AIR_BLOCK

    def tick(self, ctx: SkillContext) -> SkillResult:
        # Robust break signal: once we've been attacking the voxel, if F3's
        # targeted block is no longer THIS voxel, it broke (we're now
        # looking past it). This is more reliable than waiting for the
        # WorldMap to carve the mined voxel to air, which lags. Either
        # signal completes the skill.
        la = ctx.looking_at
        if self._mining_ticks >= 1 and la is not None and \
                getattr(la, "pos", None) is not None and \
                tuple(la.pos) != tuple(self.voxel):
            return SkillResult(AgentAction(), SkillStatus.DONE, "mined (target moved off)")
        # Already air in the map? done.
        if self._is_gone(ctx):
            return SkillResult(AgentAction(), SkillStatus.DONE, "mined")
        if _eye(ctx.pose) is None:
            return SkillResult(AgentAction(), SkillStatus.BLOCKED, "no pose")
        # 1. Select the tool once.
        if not self._tool_ok:
            hb = ctx.hotbar
            slot = (hb.best_slot_for(self.tool_role)
                    if hb is not None else None)
            self._tool_ok = True   # don't loop forever if the tool is missing
            if slot is not None:
                return SkillResult(AgentAction(hotbar=slot), SkillStatus.RUNNING,
                                   f"select {self.tool_role} slot {slot}")
        # 2. Aim, then 3. hold attack.
        eye = _eye(ctx.pose)
        tgt = (self.voxel[0] + 0.5, self.voxel[1] + 0.5, self.voxel[2] + 0.5)
        yaw, pitch = aim_angles(eye, tgt)
        dx, dy, aimed = self.aim.step(ctx, yaw, pitch)
        if not aimed:
            return SkillResult(AgentAction(look_dx=dx, look_dy=dy),
                               SkillStatus.RUNNING, "aiming at block")
        self._mining_ticks += 1
        if self._mining_ticks > self.max_ticks:
            return SkillResult(AgentAction(), SkillStatus.FAILED,
                               "timed out mining")
        return SkillResult(AgentAction(interact="attack"), SkillStatus.RUNNING,
                           f"mining {self._mining_ticks}/{self.max_ticks}")


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

    def reset(self):
        self._placed = 0; self._start = None
        self._ticks_this_block = 0; self._selected = False

    def tick(self, ctx: SkillContext) -> SkillResult:
        p = ctx.pose
        if p is None:
            return SkillResult(AgentAction(), SkillStatus.BLOCKED, "no pose")
        if self._start is None:
            self._start = (float(p.x), float(p.z))
        dist = math.hypot(float(p.x) - self._start[0], float(p.z) - self._start[1])
        placed_now = int(dist)
        if placed_now > self._placed:
            self._placed = placed_now
            self._ticks_this_block = 0
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
        """True if the block we'd step onto next (toward the target) is a
        KNOWN air block (a drop). Unknown ground -> not an edge (proceed)."""
        wm = ctx.world_map
        if wm is None or not self.avoid_fall:
            return False
        sx = (1 if tx > px else -1) if abs(tx - px) >= abs(tz - pz) else 0
        sz = 0 if sx != 0 else (1 if tz > pz else -1)
        foot = int(math.floor(py))
        ground = (int(math.floor(px)) + sx, foot - 1, int(math.floor(pz)) + sz)
        try:
            obs = wm.get_block(ground, dimension=ctx.dimension)
        except TypeError:
            obs = wm.get_block(ground)
        return obs is not None and obs.block_id == AIR_BLOCK

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
        mv = {"forward": True, "sprint": self.sprint}
        if self._stuck >= self.jump_after:
            mv["jump"] = True
        info = "jump-walking" if mv.get("jump") else "walking"
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
                 max_height: int = 10, tool_role: Optional[str] = "axe"):
        self.start = tuple(start_voxel)
        self.is_log = is_log or (lambda b: bool(b) and str(b).endswith("_log"))
        self.max_height = max_height
        self.tool_role = tool_role
        self.reset()

    def reset(self):
        self._target = self.start
        self._mined = 0
        self._phase = "mine"
        self._mine = MineBlock(self._target, tool_role=self.tool_role)
        self._aim = None
        self._check_ticks = 0

    def tick(self, ctx: SkillContext) -> SkillResult:
        if self._mined >= self.max_height:
            return SkillResult(AgentAction(), SkillStatus.DONE,
                               f"reached height cap ({self._mined})")
        if self._phase == "mine":
            r = self._mine.tick(ctx)
            if r.status == SkillStatus.DONE:
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
            self._mine = MineBlock(self._target, tool_role=self.tool_role)
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
