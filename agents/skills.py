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
from vision.world.pathfind import find_path, PathfinderConfig

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
    targeted_pos: Optional[tuple] = None  # raw F3 targeted-block coords even when
                                      # the id is unreadable (placement confirm)
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


# Survival block-interaction reach (eye -> nearest point of the block).
PLAYER_REACH = 4.5


def block_reach_distance(eye: Tuple[float, float, float], voxel: Voxel) -> float:
    """Distance from the eye to the NEAREST point of the voxel's 1×1×1 box —
    the right measure for 'can the player reach this block', using the eye's
    fractional position and the block's integer corner. Straight-up logs have
    a small HORIZONTAL distance but a large 3D one, so this (not a 2D check)
    is what tells us a canopy log is out of reach."""
    nx = min(max(eye[0], voxel[0]), voxel[0] + 1.0)
    ny = min(max(eye[1], voxel[1]), voxel[1] + 1.0)
    nz = min(max(eye[2], voxel[2]), voxel[2] + 1.0)
    return math.sqrt((eye[0] - nx) ** 2 + (eye[1] - ny) ** 2 + (eye[2] - nz) ** 2)


def block_in_reach(pose, voxel, max_reach: float = PLAYER_REACH) -> bool:
    eye = _eye(pose)
    return eye is not None and block_reach_distance(eye, voxel) <= max_reach


# Unit normal of each block face the player can target (F3 "axis"/face line).
_FACE_NORMAL = {
    "up": (0, 1, 0), "down": (0, -1, 0),
    "north": (0, 0, -1), "south": (0, 0, 1),
    "east": (1, 0, 0), "west": (-1, 0, 0),
}


def _player_voxels(pose) -> Tuple[Voxel, Voxel]:
    """The two voxels the player body occupies: feet + head."""
    fx, fy, fz = (int(math.floor(pose.x)), int(math.floor(pose.y)),
                  int(math.floor(pose.z)))
    return (fx, fy, fz), (fx, fy + 1, fz)


# Player collision box: 0.6 wide (±0.30 around x,z) and 1.80 tall from the feet.
_PLAYER_HALF_W = 0.30
_PLAYER_HEIGHT = 1.80


def _intersects_player(pose, place: Voxel) -> bool:
    """True iff the 1×1×1 block at ``place`` would overlap the player's
    collision box — the ONLY geometric reason MC silently refuses a placement.

    A real 3-axis AABB test, NOT a horizontal-distance approximation: a block
    that's horizontally close but VERTICALLY clear (the ground directly below
    you when looking down, or a 1-block step down on uneven terrain) does NOT
    intersect and places fine. The old check rejected on horizontal distance
    alone, which ruled out a huge number of perfectly legal placements."""
    try:
        px, py, pz = float(pose.x), float(pose.y), float(pose.z)
    except Exception:
        return False
    bx, by, bz = place
    r = _PLAYER_HALF_W
    x_over = (bx < px + r) and (bx + 1.0 > px - r)
    y_over = (by < py + _PLAYER_HEIGHT) and (by + 1.0 > py)
    z_over = (bz < pz + r) and (bz + 1.0 > pz - r)
    return x_over and y_over and z_over


def placement_voxel(looking_at) -> Optional[Voxel]:
    """The voxel a block would be PLACED INTO if the player right-clicked the
    block they're looking at now: the targeted block's position offset by its
    targeted FACE normal. None if there's no valid target/face."""
    if looking_at is None:
        return None
    pos = getattr(looking_at, "pos", None)
    face = getattr(looking_at, "face", None)
    if pos is None or face not in _FACE_NORMAL:
        return None
    n = _FACE_NORMAL[face]
    return (pos[0] + n[0], pos[1] + n[1], pos[2] + n[2])


# Blocks MC silently REPLACES when you place into them — so the placement voxel
# being one of these is NOT an obstruction. Without this, no spot in a grassy /
# flowery biome is ever 'placeable' (short_grass covers the ground), and the
# table placer scans every view and times out (PC-observed on flower_forest).
_REPLACEABLE_PLACE_INTO = frozenset({
    "minecraft:short_grass", "minecraft:grass", "minecraft:tall_grass",
    "minecraft:fern", "minecraft:large_fern", "minecraft:dead_bush",
    "minecraft:seagrass", "minecraft:tall_seagrass", "minecraft:snow",
    "minecraft:vine", "minecraft:glow_lichen", "minecraft:hanging_roots",
    "minecraft:water", "minecraft:lava", "minecraft:fire", "minecraft:light",
})


def _is_replaceable_place_into(bid: Optional[str]) -> bool:
    if not bid:
        return True
    if bid in _REPLACEABLE_PLACE_INTO:
        return True
    b = bid.split(":")[-1]
    # All flowers (incl. 2-tall bottoms), saplings, mushrooms, crops, ferns,
    # tulips, and the *_grass plants are replaceable. ``grass_block`` is a real
    # solid ground block and is intentionally NOT matched (it doesn't end with
    # ``_grass``? it does — guard it explicitly).
    if b == "grass_block":
        return False
    return (b.endswith("_grass") or b.endswith("_fern") or b.endswith("_tulip")
            or b.endswith("_sapling") or b.endswith("_mushroom")
            or b in ("dandelion", "poppy", "blue_orchid", "allium",
                     "azure_bluet", "oxeye_daisy", "cornflower", "torchflower",
                     "lily_of_the_valley", "wither_rose", "sunflower", "lilac",
                     "rose_bush", "peony", "pink_petals", "snow"))


def can_place_block(pose, looking_at, world_map=None, dimension=None,
                    max_reach: float = PLAYER_REACH,
                    assume_face: Optional[str] = None) -> Optional[Voxel]:
    """Decide whether placing a block is possible RIGHT NOW, returning the
    voxel it would occupy, or None. A placement is valid when the targeted
    block AND the resulting voxel are within reach, the voxel isn't inside the
    player's own body (MC forbids it), and the voxel isn't already a known
    solid block. This is the "is placing possible" check the placer scans
    around to satisfy.

    ``assume_face`` is used when F3 doesn't report the targeted face (it
    often doesn't): a downward-looking placer can assume "up" (placing on a
    block's top), which is correct for putting something on the ground."""
    if pose is None or looking_at is None:
        return None
    tgt = getattr(looking_at, "pos", None)
    if tgt is None:
        return None
    face = getattr(looking_at, "face", None) or assume_face
    if face not in _FACE_NORMAL:
        return None
    n = _FACE_NORMAL[face]
    place = (tgt[0] + n[0], tgt[1] + n[1], tgt[2] + n[2])
    # Reach is to the block you CLICK (the targeted block). MC then places into
    # the adjacent cell — which may itself sit a hair past the reach sphere, so
    # we deliberately do NOT also require ``place`` to be in reach (that wrongly
    # rejected boundary placements where the clicked block is clearly reachable).
    if not block_in_reach(pose, tgt, max_reach):     # clicked block too far
        return None
    # The ONLY geometric block: the placed cube can't intersect our hitbox.
    # Full 3-axis test — horizontally-close-but-vertically-clear spots place fine.
    if _intersects_player(pose, place):
        return None
    if world_map is not None:                        # already occupied?
        try:
            obs = world_map.get_block(place, dimension=dimension)
        except TypeError:
            obs = world_map.get_block(place)
        bid = getattr(obs, "block_id", None)
        if bid not in (AIR_BLOCK, None) and not _is_replaceable_place_into(bid):
            # Only a CONFIRMED or high-confidence solid actually blocks the
            # placement. A low-confidence belief GUESS (a mislabel hovering over
            # the ground, e.g. grass guessed as oak_leaves) must NOT — otherwise
            # no spot in a forest is ever 'placeable' and the table never lands.
            src = getattr(obs, "source", None)
            conf = float(getattr(obs, "confidence", 0.0) or 0.0)
            if src in _CONFIRMED_SOURCES or conf >= 0.85:
                return None
    return place


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
                 max_px: int = 140, stuck_limit: int = 6):
        self.tol = tol_deg
        self.gain = gain
        self.max_px = max_px
        self.stuck_limit = stuck_limit  # waits before forcing a re-issue
        self._last_pose = None        # (yaw,pitch) we last issued a move at
        self._stuck = 0               # consecutive ticks the pose hasn't moved

    def step(self, ctx: SkillContext, want_yaw: float, want_pitch: float
             ) -> Tuple[int, int, bool]:
        p = ctx.pose
        cur = (round(float(p.yaw), 2), round(float(p.pitch), 2))
        yaw_err = norm_angle(want_yaw - float(p.yaw))
        pitch_err = float(want_pitch) - float(p.pitch)
        aimed = abs(yaw_err) <= self.tol and abs(pitch_err) <= self.tol
        if aimed:
            self._last_pose = None        # no correction pending
            self._stuck = 0
            return 0, 0, True
        # Stale-pose guard. Pose feedback (F3 OCR, a few Hz) lags the
        # control tick (~12-20 Hz). After we ISSUE a camera correction, the
        # pose won't reflect it for a tick or two; issuing another in the
        # meantime STACKS corrections and overshoots/oscillates (the
        # crosshair wanders diagonally and never settles). So once we've
        # issued a move, wait until the pose actually changes before issuing
        # the next. (Only set when we issue a non-zero look — a stationary
        # pose while merely walking must NOT block the next turn.)
        #
        # ESCAPE HATCH: if the pose hasn't budged after ``stuck_limit`` waits,
        # the camera input isn't landing (a transient raw-input desync — seen
        # after a use_item right-click freezes the view) or the frame is stale.
        # Re-ISSUE the correction instead of waiting forever; a fresh
        # track_target re-engages MC's raw-input listener and unsticks the aim.
        if self._last_pose is not None and cur == self._last_pose:
            self._stuck += 1
            if self._stuck < self.stuck_limit:
                return 0, 0, False
            # fall through to re-issue the move (don't reset _last_pose: we're
            # still at the same pose, so the next no-change tick keeps counting)
        else:
            self._stuck = 0
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
    e.g. 'axe' for logs), aim at it, then a hard aim->freeze+mine LATCH —
    the instant F3's targeted block is a log it FREEZES the camera and holds
    attack (no drift/circling) until that block is gone. The break signal is
    F3-driven: the log disappears from the crosshair for a few ticks (the
    lagging WorldMap-air check proved unreliable, so F3 'looking at' is the
    sole authority). Out-of-reach / mislabelled / sky targets abandon fast;
    occluding leaves are cleared (frozen) only with a log behind them.
    FAILED on timeout."""
    name = "mine_block"

    def __init__(self, voxel: Voxel, tool_role: Optional[str] = "axe",
                 max_ticks: int = 200, tol_deg: float = 3.0,
                 is_safe=None, is_target=None, is_passthrough=None,
                 max_reach: float = PLAYER_REACH):
        self.voxel = voxel
        self.tool_role = tool_role
        self.max_ticks = max_ticks
        self.max_reach = max_reach
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
        self._clear_ticks = 0          # ticks spent clearing occluding leaves
        self._aim_ticks = 0            # consecutive ticks trying to aim/acquire
        self.broke = False             # did a LOG actually break? (for counting)

    def reset(self):
        self._tool_ok = self.tool_role is None
        self._mode = "aim"
        self._mining_ticks = 0
        self._silent = 0
        self._await_ticks = 0
        self._clear_ticks = 0
        self._aim_ticks = 0            # ticks spent trying to aim/acquire a log
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
        # Raw F3 targeted-block POSITION survives an UNREADABLE id. On busy
        # forest scenes F3 reads "Targeted Block: x, y, z" cleanly but garbles
        # the id to '?', so parse_looking_at_block returns None (it requires a
        # valid id) and ctx.looking_at is None — losing the fact that the
        # crosshair IS on our target. ctx.targeted_pos recovers just the coords,
        # so we can still confirm we're aimed dead-on the voxel we set out to
        # mine (which the world-map already classified as a log).
        tpos = tuple(ctx.targeted_pos) if ctx.targeted_pos is not None else None
        on_target_raw = (tpos == tuple(self.voxel))

        def _hold(info):                 # camera FROZEN (no look), click held
            self._mining_ticks += 1
            if self._mining_ticks > self.max_ticks:
                return SkillResult(AgentAction(), SkillStatus.FAILED, "timed out mining")
            return SkillResult(AgentAction(interact="attack"), SkillStatus.RUNNING, info)

        def _mine_centred(info):
            # Like _hold, but keeps the crosshair CENTRED on the target voxel
            # while mining. Freezing the camera the instant F3 first flickers
            # "log" can latch it on the block's EDGE; pose jitter then slips the
            # crosshair off and MC RESETS break progress, so it punches forever
            # without breaking (live-observed: 10 s, never broke). A small
            # correction toward the FIXED voxel centre (not chasing jittery F3)
            # holds it on the block; capped so it can't swing onto a neighbour.
            self._mining_ticks += 1
            if self._mining_ticks > self.max_ticks:
                return SkillResult(AgentAction(), SkillStatus.FAILED, "timed out mining")
            e = _eye(ctx.pose)
            if e is None:
                return SkillResult(AgentAction(interact="attack"),
                                   SkillStatus.RUNNING, info)
            c = (self.voxel[0] + 0.5, self.voxel[1] + 0.5, self.voxel[2] + 0.5)
            yaw, pitch = aim_angles(e, c)
            dx, dy, _ = self.aim.step(ctx, yaw, pitch)
            cap = 6                              # ~1 deg/tick: nudge, never swing off
            dx = max(-cap, min(cap, int(dx))); dy = max(-cap, min(cap, int(dy)))
            return SkillResult(AgentAction(interact="attack", look_dx=dx, look_dy=dy),
                               SkillStatus.RUNNING, info)

        # While LATCHED onto a target (mine_log / clear_leaf), keep checking it's
        # in INTERACTION reach. F3's targeted-block ray reaches ~20 blocks — far
        # past the 4.5 we can actually hit — so the crosshair can rest on a log
        # up a ledge or across a gap that we'll only punch air at until the
        # 20s mine timeout (the live "trying to punch a tree barely too far
        # away"). Abandon promptly so the FSM walks closer / skips it.
        if self._mode in ("mine_log", "clear_leaf"):
            _e = _eye(ctx.pose)
            if _e is not None and block_reach_distance(_e, self.voxel) > self.max_reach:
                return SkillResult(AgentAction(), SkillStatus.FAILED,
                                   f"target out of reach while mining ({self.voxel})")

        # ── MINING a log: frozen camera, hold click until the log is gone ──
        if self._mode == "mine_log":
            # "Still on the log" = F3 names a log, OR the id is unreadable but
            # the raw targeted position is still our exact voxel (the log
            # hasn't broken — F3 just can't name it). When it breaks the
            # crosshair falls through and the targeted position changes/clears.
            if self._is_log(la_id) or (la_id is None and on_target_raw):
                self._silent = 0
                return _mine_centred("mining log")   # hold crosshair ON the block
            self._silent += 1
            if self._silent <= 3:        # ride out a brief F3 OCR gap, still frozen
                return _hold("mining log (F3 gap)")
            self.broke = True            # the log we held on is gone -> counted
            return SkillResult(AgentAction(), SkillStatus.DONE, "mined log")

        # ── CLEARING occluding leaves: frozen, hold click; NOT counted.
        # Stays held across leaf→leaf and brief F3 gaps so a leaf WALL doesn't
        # cause spam-clicking; a LOG appearing -> straight into mine_log. ──
        if self._mode == "clear_leaf":
            self._clear_ticks += 1
            if self._clear_ticks > 45:        # leaves never revealing a log -> give up
                return SkillResult(AgentAction(), SkillStatus.FAILED,
                                   "leaves not clearing to a log — abandon")
            if self._is_log(la_id):                       # log behind -> mine it
                self._mode = "mine_log"; self._silent = 0
                return _hold("log behind leaves -> mining")
            if self._is_pass(la_id):                      # still a leaf -> keep clearing
                self._silent = 0
                return _hold("clearing leaf")
            self._silent += 1
            if self._silent <= 3:                         # brief gap -> keep holding
                return _hold("clearing (brief gap)")
            self._mode = "aim"; self._silent = 0          # gone -> re-evaluate

        eye = _eye(ctx.pose)
        if eye is None:
            return SkillResult(AgentAction(), SkillStatus.BLOCKED, "no pose")
        # ACQUISITION timeout — the aim phase (unlike mining) was uncapped, so a
        # mis-located/phantom target or garbled F3 left the bot "aiming" at
        # nothing for ~20s. Bail after ~5s so the FSM re-scans / walks to a real
        # tree instead of staring off to the side doing nothing.
        self._aim_ticks += 1
        if self._aim_ticks > 50:
            return SkillResult(AgentAction(), SkillStatus.FAILED,
                               "couldn't engage a log (aim/F3 unreliable) — abandon")
        # Out of reach? Don't aim/clear-leaves at a block we can't touch (a
        # too-high canopy log) — abandon so the FSM walks closer or skips it.
        if block_reach_distance(eye, self.voxel) > self.max_reach:
            return SkillResult(AgentAction(), SkillStatus.FAILED,
                               f"target out of reach ({self.voxel})")
        if not self._tool_ok:
            hb = ctx.hotbar
            slot = (hb.best_slot_for(self.tool_role) if hb is not None else None)
            self._tool_ok = True
            if slot is not None:
                return SkillResult(AgentAction(hotbar=slot), SkillStatus.RUNNING,
                                   f"select {self.tool_role} slot {slot}")

        # THE MOMENT F3 shows a LOG under the crosshair: LATCH -> freeze + mine.
        # NB: do NOT reset _aim_ticks here — a crosshair that just FLICKERS onto
        # a log (edge-of-reach jitter) would keep resetting it and never time
        # out. _aim_ticks counts TOTAL aim-phase ticks this MineBlock; a real
        # mine sits in mine_log (not aiming) so it never approaches the cap.
        if self._is_log(la_id):
            self._mode = "mine_log"; self._silent = 0
            return _hold("log in crosshair -> mining")

        # Id unreadable, but F3's raw POSITION confirms the crosshair is on the
        # EXACT voxel we set out to mine (the world-map classified it a log, and
        # the _Aimer has us pointed at it). Trust position + that prior
        # classification and mine — otherwise busy-scene id-garble (F3 '?')
        # makes the bot stare at a real log it can't "confirm" and abandon.
        # A READABLE non-log id still abandons below, so builds stay safe.
        if la_id is None and on_target_raw:
            self._mode = "mine_log"; self._silent = 0
            return _hold("on-target by position (id unreadable) -> mining")

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
                self._mode = "clear_leaf"; self._silent = 0; self._clear_ticks = 0
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
                 sprint: bool = True, jump_after: int = 4,
                 arrive_on_column: bool = False,
                 arrive_max_dy: Optional[float] = None):
        self.target = tuple(target)
        self.arrive_dist = arrive_dist
        # Optional VERTICAL gate on arrival: only count as arrived when the feet
        # are within this many blocks of the target's y. Used by drop-collection
        # on slopes/ledges so the bot doesn't "arrive" on a shelf high ABOVE the
        # drop and dwell there — it keeps walking (with avoid_fall off) to
        # descend to the item's level. None = no vertical constraint.
        self.arrive_max_dy = arrive_max_dy
        # When True, "arrived" means the player's ROUNDED (floor) x,z equal the
        # target's x,z — i.e. standing in the target block's column. Used to
        # COLLECT a broken block's drop: walk onto its exact x,z so the item
        # (which falls straight down) is within pickup range.
        self.arrive_on_column = arrive_on_column
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
        # Optional vertical gate: not "arrived" while we're still well above (or
        # below) the target's level — keeps the collector walking down to the
        # drop instead of stopping on a ledge over it.
        dy_ok = (self.arrive_max_dy is None
                 or abs(int(math.floor(p.y)) - self.target[1]) <= self.arrive_max_dy)
        # Arrived when standing in the target's column (rounded x,z match) — the
        # tight goal used for item collection — or within arrive_dist otherwise.
        if dy_ok and self.arrive_on_column and int(math.floor(px)) == self.target[0] \
                and int(math.floor(pz)) == self.target[2]:
            return SkillResult(AgentAction(movement={"forward": False}),
                               SkillStatus.DONE, "arrived on column")
        if dy_ok and dist <= self.arrive_dist:
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


class NavigateTo(Skill):
    """Route to ``goal`` by A* over the WorldMap, FOLLOWING the path with the
    reactive WalkToward (sprint / jump / edge-safe) one waypoint at a time.
    This is the planner layer: A* routes AROUND known gaps/obstacles that the
    straight-line WalkToward alone would fail at. Replans when a segment gets
    stuck/edges out; falls back to a direct reactive walk when there's no map
    or no route. DONE when ``goal`` is within reach (or arrive_dist).

    The A* default here is ``passable`` (unknown = open) because the
    recogniser's map is sparse — that lets it route around the KNOWN
    walls/gaps it has mapped while WalkToward's own edge check guards the
    unknown last metre."""
    name = "navigate_to"

    def __init__(self, goal: Voxel, *, arrive_reach: float = PLAYER_REACH,
                 arrive_dist: Optional[float] = None,
                 unknown_policy: str = "passable", max_replans: int = 5,
                 dimension: Optional[str] = None):
        self.goal = tuple(goal)
        self.arrive_reach = arrive_reach     # done when goal within this 3D reach
        self.arrive_dist = arrive_dist       # ...or within this horizontal dist
        self.cfg = PathfinderConfig(unknown_policy=unknown_policy)
        self.max_replans = max_replans
        self.dimension = dimension
        self.reset()

    def reset(self):
        self._path = []
        self._wp = 0
        self._sub = None
        self._replans = 0

    @staticmethod
    def _feet(pose):
        return (int(math.floor(pose.x)), int(math.floor(pose.y)),
                int(math.floor(pose.z)))

    def _arrived(self, pose) -> bool:
        if self.arrive_dist is not None:
            d = math.hypot(self.goal[0] + 0.5 - pose.x, self.goal[2] + 0.5 - pose.z)
            return d <= self.arrive_dist
        return block_in_reach(pose, self.goal, self.arrive_reach)

    def _goal_candidates(self):
        # The goal voxel (a log) usually isn't standable; A* needs a standable
        # FEET cell. Try the goal, then standable cells beside/below it.
        gx, gy, gz = self.goal
        yield self.goal
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for dy in (0, -1, 1):
                yield (gx + dx, gy + dy, gz + dz)

    def _plan(self, ctx) -> bool:
        feet = self._feet(ctx.pose)
        dim = self.dimension if self.dimension is not None else ctx.dimension
        for g in self._goal_candidates():
            res = find_path(ctx.world_map, feet, g, dimension=dim, config=self.cfg)
            if res and res.waypoints:
                self._path = list(res.waypoints)
                self._wp = 0
                return True
        return False

    def _reactive(self, ctx):
        if self._sub is None or not isinstance(self._sub, WalkToward):
            self._sub = WalkToward(self.goal, arrive_dist=(self.arrive_dist or 1.6))
        return self._sub.tick(ctx)

    def tick(self, ctx: SkillContext) -> SkillResult:
        pose = ctx.pose
        if pose is None:
            return SkillResult(AgentAction(), SkillStatus.BLOCKED, "no pose")
        if self._arrived(pose):
            return SkillResult(AgentAction(movement={"forward": False}),
                               SkillStatus.DONE, "arrived")
        if ctx.world_map is None:                    # no map -> straight-line
            return self._reactive(ctx)

        if not self._path:
            if self._replans > self.max_replans:
                r = self._reactive(ctx)              # routing exhausted -> best effort
                if r.status == SkillStatus.FAILED:
                    return SkillResult(r.action, SkillStatus.FAILED, "no route")
                return r
            self._replans += 1
            if not self._plan(ctx):
                return self._reactive(ctx)           # A* found nothing this tick

        # Advance past waypoints we're already standing on.
        def _hd(wp):
            return math.hypot(wp[0] + 0.5 - pose.x, wp[2] + 0.5 - pose.z)
        while self._wp < len(self._path) - 1 and _hd(self._path[self._wp]) <= 1.3:
            self._wp += 1
            self._sub = None
        wp = self._path[self._wp]
        if self._sub is None or getattr(self._sub, "target", None) != wp:
            self._sub = WalkToward(wp, arrive_dist=1.1, stuck_window=14)
        r = self._sub.tick(ctx)
        if r.status == SkillStatus.DONE:
            self._sub = None
            self._replans = 0                        # progress -> refresh replan budget
            if self._wp >= len(self._path) - 1:
                self._path = []                      # reached path end -> re-eval/replan
                return SkillResult(r.action, SkillStatus.RUNNING, "reached path end")
            self._wp += 1
            return SkillResult(r.action, SkillStatus.RUNNING,
                               f"waypoint {self._wp}/{len(self._path)}")
        if r.status in (SkillStatus.FAILED, SkillStatus.BLOCKED):
            self._path = []; self._sub = None        # segment blocked -> replan around it
            return SkillResult(r.action, SkillStatus.RUNNING, "segment blocked; replanning")
        return r


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
        self._dir = 1               # fell UP first, then DOWN to clear the stump
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
                self._target = (self._target[0], self._target[1] + self._dir,
                                self._target[2])
                self._aim = LookAtVoxel(self._target, tol_deg=3.0)
                self._phase = "aim_up"; self._check_ticks = 0
                return SkillResult(r.action, SkillStatus.RUNNING,
                                   f"mined {self._mined}; looking "
                                   f"{'up' if self._dir > 0 else 'down'}")
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
            if self._dir > 0:
                # Top of the trunk reached. Now fell DOWNWARD from below the
                # start voxel: a stump left below the chop point blocks the bot
                # from standing on the drop column, so the logs we just felled
                # never get collected (the live "chopped N, collected 0" bug).
                self._dir = -1
                self._target = (self.start[0], self.start[1] - 1, self.start[2])
                self._aim = LookAtVoxel(self._target, tol_deg=3.0)
                self._phase = "aim_up"; self._check_ticks = 0
                return SkillResult(AgentAction(), SkillStatus.RUNNING,
                                   "trunk top cleared; felling stump below")
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


_CONFIRMED_SOURCES = ("looking_at", "ray_clear_air")


def find_nearest_block(world_map, origin, match, *, max_radius: int = 48,
                       dimension: Optional[str] = None, exclude=None, pos_ok=None,
                       min_confidence: float = 0.0):
    """Nearest observed block satisfying ``match(block_id) -> bool`` within
    ``max_radius`` (Chebyshev) of ``origin`` voxel. Returns ``(voxel, obs)``
    or ``None``. Used to spot the closest oak log the recogniser has mapped.

    ``min_confidence`` gates BELIEF-map guesses: a voxel is only eligible if it
    was F3-CONFIRMED (``source`` in :data:`_CONFIRMED_SOURCES`) OR its belief
    confidence is at least this. This stops the chopper navigating to a
    low-confidence MISLABEL (grass / leaf_litter guessed as a log) — it would
    aim at it, fail to confirm a log, and waste the aim budget (or, with the
    position fallback, chop the wrong block). 0.0 keeps the old behaviour.
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
        # Confidence gate: trust F3-confirmed voxels unconditionally; require
        # belief-map guesses to clear ``min_confidence`` so a mislabelled
        # grass/leaf doesn't become a chop target.
        if min_confidence > 0.0:
            src = getattr(obs, "source", None)
            if src not in _CONFIRMED_SOURCES \
                    and float(getattr(obs, "confidence", 0.0) or 0.0) < min_confidence:
                continue
        if exclude is not None and tuple(obs.pos) in exclude:
            continue
        if pos_ok is not None and not pos_ok(obs.pos):
            continue
        vx, vy, vz = obs.pos
        d2 = (vx - ox) ** 2 + (vy - oy) ** 2 + (vz - oz) ** 2
        if best_d2 is None or d2 < best_d2:
            best_d2, best = d2, (obs.pos, obs)
    return best


class PlaceBlock(Skill):
    """Place the block from ``tool_role``'s hotbar slot on a valid nearby
    surface, robustly across uneven ground.

    Rather than guess one pitch (too shallow -> F3 sees no block; too steep ->
    placement is too close and MC silently rejects it), it SCANS a set of look
    directions (each ``pitch`` x ``yaw_off``). For each it aims, lets F3 catch
    up, and asks ``can_place_block`` whether a top-face placement is possible.
    On the first placeable view it emits use_item, then VERIFIES the block
    actually appeared under the still-aimed crosshair (F3 ``pos`` == the
    placement voxel, non-air). If MC rejected it, it moves on to the next view.
    DONE only once placement is confirmed, with the voxel in ``placed_at``.

    The forward views (yaw_off 0) are tried first across all pitches, then it
    turns. Defaults cover ~steep-to-shallow on flat-ish ground; tune via
    ``pitches`` / ``yaw_offs``."""
    name = "place_block"

    def __init__(self, tool_role: str = "blocks", *, slot: Optional[int] = None,
                 pitches=(48.0, 42.0, 55.0, 36.0),
                 yaw_offs=(0.0, 60.0, -60.0, 120.0, -120.0, 180.0),
                 max_ticks: int = 900, max_reach: float = PLAYER_REACH,
                 tol_deg: float = 5.0, verify_ticks: int = 9,
                 max_step_backs: int = 3, back_ticks: int = 12,
                 spot_budget: int = 60, prime_back_ticks: int = 11,
                 prime: bool = True):
        self.tool_role = tool_role
        self.prime = bool(prime)         # try "step back + place where you stood"
        self.slot = slot                 # explicit hotbar slot 1-9, overrides role
        # SWEEP DIRECTIONS FIRST at the best pitch (turn away from obstacles
        # like a tree we're standing in), THEN revisit with other pitches —
        # pitch-outer so we cycle through every heading before nodding up/down.
        self._cands = [(float(p), float(y)) for p in pitches for y in yaw_offs]
        self.max_ticks = max_ticks
        self.max_reach = max_reach
        self._tol = tol_deg
        self.verify_ticks = verify_ticks
        self.max_step_backs = int(max_step_backs)
        self.back_ticks = int(back_ticks)
        self.spot_budget = int(spot_budget)
        self.prime_back_ticks = int(prime_back_ticks)
        self.reset()

    def reset(self):
        self._t = 0
        self._idx = 0               # which candidate view we're on
        self._base_yaw = None       # fixed reference so scans don't chase
        self._selected = False
        self._aimed_count = 0
        self._mode = "aim"          # "aim" -> "verify"
        self._verify = 0
        self._aimer = _Aimer(tol_deg=self._tol)
        self.placed_at = None
        self._support = None
        self._step_backs = 0
        self._stepping = False
        self._back_t = 0
        self._spot_ticks = 0        # ticks scanned at the CURRENT spot
        self._escape_yaw = None     # fixed heading to retreat along (set once)
        # PRIME: "step back one block and place where you were standing" — the
        # most reliable placement (that cell is guaranteed solid-below +
        # clear-above), tried once before any scanning.
        self._primed = not getattr(self, "prime", True)
        self._stand = None          # the block under our feet at start
        self._prime_t = 0
        self._prime_aim = None

    def _exhausted_views(self):
        """All views failed at this spot. Step back onto fresh ground and
        re-scan (the bot is usually standing in the cluttered patch it just
        chopped, where the close ground is occupied by stumps/leaves/drops).
        Returns a SkillResult to step back, or None when the budget is spent."""
        if self._step_backs >= self.max_step_backs:
            return None
        self._step_backs += 1
        self._stepping = True
        self._back_t = 0
        self._spot_ticks = 0
        return SkillResult(AgentAction(movement={"backward": True}),
                           SkillStatus.RUNNING,
                           f"place: no spot here — stepping back ({self._step_backs})")

    def _next_view(self) -> bool:
        """Advance to the next candidate; return False if exhausted."""
        self._idx += 1
        self._aimed_count = 0
        self._verify = 0
        self._mode = "aim"
        self._aimer = _Aimer(tol_deg=self._tol)
        return self._idx < len(self._cands)

    def tick(self, ctx: SkillContext) -> SkillResult:
        self._t += 1
        if self._t > self.max_ticks:
            return SkillResult(AgentAction(), SkillStatus.FAILED, "place: timed out")
        pose = ctx.pose
        if pose is None:
            return SkillResult(AgentAction(), SkillStatus.BLOCKED, "place: no pose")
        # 1. hold the block in hand (explicit slot wins over the role lookup)
        if not self._selected:
            slot = self.slot
            if slot is None:
                slot = ctx.hotbar.best_slot_for(self.tool_role) if ctx.hotbar else None
            if slot is None:
                return SkillResult(AgentAction(), SkillStatus.FAILED,
                                   f"place: no '{self.tool_role}' in hotbar")
            self._selected = True
            return SkillResult(AgentAction(hotbar=int(slot)), SkillStatus.RUNNING,
                               f"place: select slot {slot}")
        # Anchor the yaw to the STARTING yaw so each view offset is a fixed
        # target (recomputing from the live yaw each tick makes it run away).
        if self._base_yaw is None:
            self._base_yaw = float(pose.yaw)

        # 1b. PRIME (once, before any scanning): the most reliable placement
        # anywhere is "step back one block and put the table where you were just
        # standing" — that cell is GUARANTEED solid below (you stood on it) and
        # clear above (your body filled it), so it needs no terrain scan and
        # works on cluttered/forest/edge ground that defeats the view scan.
        if not self._primed:
            if self._stand is None:
                self._stand = (int(math.floor(pose.x)),
                               int(math.floor(pose.y)) - 1,
                               int(math.floor(pose.z)))
                self._prime_t = 0
                self._prime_aim = None
            self._prime_t += 1
            if self._prime_t <= self.prime_back_ticks:
                # back up off the stand cell so placing there won't intersect us
                return SkillResult(AgentAction(movement={"backward": True}),
                                   SkillStatus.RUNNING, "place: priming (step back)")
            # Aim PRECISELY at the cell we just stepped off of — at the VOXEL
            # itself (LookAtVoxel computes the exact angle), not a fixed
            # down-forward pitch. The fixed pitch overshot onto the tree trunk
            # we'd been chopping right next to (the live "couldn't place the
            # table" in a forest), whereas the exact aim lands on the grass we
            # stood on, which is closer than the trunk.
            if self._prime_aim is None:
                self._prime_aim = LookAtVoxel(self._stand, tol_deg=self._tol)
            ar = self._prime_aim.tick(ctx)
            if ar.status == SkillStatus.RUNNING \
                    and self._prime_t < self.prime_back_ticks + 30:
                return SkillResult(ar.action, SkillStatus.RUNNING,
                                   "place: priming (aim at stand spot)")
            aimed = (ar.status == SkillStatus.DONE)
            self._primed = True           # only ever prime once; then scan
            place = (can_place_block(pose, ctx.looking_at, ctx.world_map,
                                     getattr(pose, "dimension", None),
                                     self.max_reach, assume_face="up")
                     if aimed else None)
            self._base_yaw = float(pose.yaw)
            if place is not None:
                self.placed_at = place
                self._support = (place[0], place[1] - 1, place[2])
                self._mode = "verify"; self._verify = 0
                return SkillResult(AgentAction(interact="use_item"),
                                   SkillStatus.RUNNING,
                                   f"place: priming at {place}")
            # couldn't place where we stood -> fall through to the view scan

        # STEP-BACK: walk backward onto fresh ground, then re-scan every view
        # from there (escape the cluttered chop patch). A fixed number of ticks
        # of backward, then reset the scan.
        if self._stepping:
            self._back_t += 1
            if self._back_t < self.back_ticks:
                # Retreat along a FIXED heading (locked on the first step-back)
                # so successive step-backs accumulate in ONE direction — out of
                # the tree box — instead of wandering as each re-scan re-faces
                # the bot. Hold that yaw with a gentle look correction while
                # walking backward.
                if self._escape_yaw is None:
                    self._escape_yaw = float(pose.yaw)
                err = ((self._escape_yaw - float(pose.yaw) + 180.0) % 360.0) - 180.0
                ldx = int(max(-50.0, min(50.0, err * (ctx.px_per_deg or 6.5))))
                return SkillResult(
                    AgentAction(movement={"backward": True}, look_dx=ldx),
                    SkillStatus.RUNNING, "place: stepping back to clearer ground")
            self._stepping = False
            self._idx = 0
            self._base_yaw = None
            self._spot_ticks = 0
            self._aimer = _Aimer(tol_deg=self._tol)
            return SkillResult(AgentAction(), SkillStatus.RUNNING,
                               "place: re-scanning from new spot")

        # Per-spot time cap: if we've scanned this spot too long without
        # placing, RELOCATE (step back) instead of grinding every view to the
        # global budget — the spot is just bad (cratered/sloped chop terrain).
        # This guarantees the step-back actually fires (the full 24-view scan is
        # slow enough that the whole place step otherwise times out in one spot).
        if self._mode == "aim":
            self._spot_ticks += 1
            if self._spot_ticks > self.spot_budget:
                r = self._exhausted_views()
                if r is not None:
                    return r
                return SkillResult(AgentAction(), SkillStatus.FAILED,
                                   "place: no placeable ground (stepped back "
                                   f"{self._step_backs}x)")

        # 2b. VERIFY a placement we just attempted: did the block appear under
        # the (still-aimed) crosshair? If so we're done; if MC rejected it
        # (nothing there after a few ticks), try the next view.
        if self._mode == "verify":
            self._verify += 1
            la = ctx.looking_at
            lpos = getattr(la, "pos", None)
            lbid = getattr(la, "block_id", None)
            # Confirm by POSITION: after placing, the crosshair (still aimed at
            # the same view) now hits the new block AT the placement voxel,
            # whereas before it hit the support block below it. Accept EITHER
            # signal — the id-recognised looking_at OR the raw F3 targeted
            # coords — because the placement's IDENTITY is irrelevant here, only
            # that a block now occupies placed_at. The block recogniser routinely
            # MISREADS a freshly-placed table (seen live: as birch_leaves) and
            # F3 OCRs its id to garble, so relying on a correct id would miss a
            # real placement; the raw targeted coords stay clean.
            id_hit = (lpos == self.placed_at and lbid not in (None, AIR_BLOCK))
            raw_hit = (ctx.targeted_pos == self.placed_at)
            if id_hit or raw_hit:
                return SkillResult(AgentAction(), SkillStatus.DONE,
                                   f"placed + confirmed at {self.placed_at}")
            # Re-aim the crosshair ONTO the placed voxel itself. A steep PRIME
            # place aimed down at the grass and put the table on TOP of it, so
            # the crosshair sits on the support BELOW the table and F3 reports
            # the support — we'd otherwise call the (real) placement a failure
            # and walk off, abandoning the table we just put down (live-seen).
            # Lifting the aim onto placed_at lets F3 confirm the table.
            adx = ady = 0
            if self.placed_at is not None:
                e = _eye(pose)
                if e is not None:
                    c = (self.placed_at[0] + 0.5, self.placed_at[1] + 0.5,
                         self.placed_at[2] + 0.5)
                    yaw, pitch = aim_angles(e, c)
                    adx, ady, _ = self._aimer.step(ctx, yaw, pitch)
            if self._verify > self.verify_ticks:
                # genuinely nothing there after re-aiming -> MC rejected it.
                self.placed_at = None
                if not self._next_view():
                    r = self._exhausted_views()
                    if r is not None:
                        return r
                    return SkillResult(AgentAction(), SkillStatus.FAILED,
                                       "place: every view rejected the placement")
                return SkillResult(AgentAction(), SkillStatus.RUNNING,
                                   f"place: rejected, trying view {self._idx}")
            return SkillResult(AgentAction(look_dx=adx, look_dy=ady),
                               SkillStatus.RUNNING, "place: verifying (aim at placed)")

        # 2a. AIM at the current candidate view.
        if self._idx >= len(self._cands):
            return SkillResult(AgentAction(), SkillStatus.FAILED,
                               "place: no valid surface in any view")
        pitch, yaw_off = self._cands[self._idx]
        want_yaw = self._base_yaw + yaw_off
        dx, dy, aimed = self._aimer.step(ctx, want_yaw, pitch)
        if not aimed:
            self._aimed_count = 0
            return SkillResult(AgentAction(look_dx=dx, look_dy=dy),
                               SkillStatus.RUNNING,
                               f"place: aiming view {self._idx}")
        # let F3 'looking at' catch up to the new view before judging
        self._aimed_count += 1
        if self._aimed_count < 2:
            return SkillResult(AgentAction(), SkillStatus.RUNNING, "place: settling")
        # 3. is placing possible from here? (looking down -> top-face place)
        place = can_place_block(pose, ctx.looking_at, ctx.world_map,
                                getattr(pose, "dimension", None), self.max_reach,
                                assume_face="up")
        if place is not None:
            self.placed_at = place
            # the block we placed against (looking down -> directly below the
            # placement voxel); used to fast-detect a rejected placement
            self._support = (place[0], place[1] - 1, place[2])
            self._mode = "verify"
            self._verify = 0
            return SkillResult(AgentAction(interact="use_item"),
                               SkillStatus.RUNNING, f"place: placing at {place}")
        # 4. nothing placeable from this view -> next candidate
        if not self._next_view():
            r = self._exhausted_views()
            if r is not None:
                return r
            return SkillResult(AgentAction(), SkillStatus.FAILED,
                               "place: no valid surface in any view")
        return SkillResult(AgentAction(), SkillStatus.RUNNING,
                           f"place: scanning view {self._idx}")


class BreakLookedAt(Skill):
    """Break the block under the crosshair (hold attack until F3 no longer
    reports it there). Used to reclaim a just-placed block — e.g. the bot
    breaks its OWN crafting table after using it, before moving the crosshair.

    Two modes:
      * id-based (default): break whatever block F3 names under the crosshair;
        ``avoid(block_id)->bool`` refuses protected blocks. Needs a readable id.
      * ``expect_pos`` (a Voxel): break the block at THAT exact position,
        identified by POSITION not id — present while ``looking_at.pos`` OR the
        raw F3 ``targeted_pos`` equals it, gone once neither does (the crosshair
        drops to the block below). This is essential for a freshly-placed
        crafting table, whose id OCRs to garble so ``looking_at`` is None and
        the id-based path wrongly reports "nothing to break"."""
    name = "break_looked_at"

    def __init__(self, *, tool_role=None, max_ticks: int = 160,
                 gone_ticks: int = 3, avoid=None, expect_pos=None):
        self.tool_role = tool_role
        self.max_ticks = max_ticks
        self.gone_ticks = gone_ticks
        self.avoid = avoid
        self.expect_pos = tuple(expect_pos) if expect_pos is not None else None
        self.reset()

    def reset(self):
        self._t = 0
        self._silent = 0
        self._attacked = False
        self._selected = False
        self.broke = False

    def _select_tool(self, ctx):
        if self.tool_role and not self._selected and ctx.hotbar is not None:
            self._selected = True
            slot = ctx.hotbar.best_slot_for(self.tool_role)
            if slot is not None:
                return SkillResult(AgentAction(hotbar=slot), SkillStatus.RUNNING,
                                   f"break: select {self.tool_role}")
        return None

    def tick(self, ctx: SkillContext) -> SkillResult:
        self._t += 1
        la = ctx.looking_at
        lpos = getattr(la, "pos", None)
        bid = getattr(la, "block_id", None) if la is not None else None

        # ── position-based mode: break the exact voxel we placed, by POSITION
        # (its id won't OCR). Present while either signal points at it.
        if self.expect_pos is not None:
            present = (lpos == self.expect_pos or ctx.targeted_pos == self.expect_pos)
            if not present:
                self._silent += 1
                if self._silent >= self.gone_ticks:
                    self.broke = self._attacked
                    return SkillResult(AgentAction(), SkillStatus.DONE,
                                       "broke it" if self._attacked else "nothing to break")
                return SkillResult(AgentAction(interact="attack"),
                                   SkillStatus.RUNNING, "break: confirming gone")
            self._silent = 0
            if self._t > self.max_ticks:
                return SkillResult(AgentAction(), SkillStatus.FAILED, "break: timed out")
            sel = self._select_tool(ctx)
            if sel is not None:
                return sel
            self._attacked = True
            return SkillResult(AgentAction(interact="attack"),
                               SkillStatus.RUNNING, f"breaking block at {self.expect_pos}")

        # ── id-based mode (default).
        # nothing under the crosshair -> it's gone (or never was)
        if bid in (None, AIR_BLOCK):
            self._silent += 1
            if self._silent >= self.gone_ticks:
                self.broke = self._attacked
                return SkillResult(AgentAction(), SkillStatus.DONE,
                                   "broke it" if self._attacked else "nothing to break")
            return SkillResult(AgentAction(interact="attack"),
                               SkillStatus.RUNNING, "break: confirming gone")
        self._silent = 0
        if self.avoid is not None and self.avoid(bid):
            return SkillResult(AgentAction(), SkillStatus.FAILED,
                               f"break: refusing {bid}")
        if self._t > self.max_ticks:
            return SkillResult(AgentAction(), SkillStatus.FAILED, "break: timed out")
        sel = self._select_tool(ctx)
        if sel is not None:
            return sel
        self._attacked = True
        return SkillResult(AgentAction(interact="attack"),
                           SkillStatus.RUNNING, f"breaking {str(bid).split(':')[-1]}")


# Registry of the currently-implemented skills (name -> class). The task
# layer / a future planner can look skills up by name.
SKILLS = {s.name: s for s in (SelectRole, LookAtVoxel, Eat, MineBlock,
                              PillarUp, Bridge, WalkToward, NavigateTo, ChopTrunk,
                              PlaceBlock, BreakLookedAt, SkillSequence)}


__all__ = [
    "SkillStatus", "SkillResult", "SkillContext", "Skill",
    "SelectRole", "LookAtVoxel", "Eat", "MineBlock", "PillarUp", "Bridge",
    "WalkToward", "NavigateTo", "ChopTrunk", "SkillSequence", "find_nearest_block",
    "PlaceBlock", "BreakLookedAt", "can_place_block", "placement_voxel",
    "block_in_reach", "block_reach_distance", "PLAYER_REACH",
    "aim_angles", "norm_angle", "SKILLS",
]
