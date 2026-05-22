# agents/navigation.py
"""
Rule-based navigation agent.

The first non-trivial agent. Uses the F3 OCR (XYZ + yaw) to:

  1. Pick a target XYZ — either supplied at construction time, or set
     automatically to "25 blocks in the direction the player is currently
     facing" the first time decide() sees valid F3 data.
  2. Each tick: compute yaw error to the target, turn the camera toward
     it, and walk forward when the heading is close enough.
  3. Stop when within ``stop_distance`` blocks of the target.
  4. Eat from a designated hotbar slot if hunger drops below a threshold.

This agent exercises the entire pipeline at 20 Hz — capture, processing,
OCR (at 3 Hz), decision, mouse easing, keyboard hold/release, safety
gating — and is the canonical smoke test for "the loop runs in
real-time against live Minecraft".

It is intentionally simple. It does NOT do pathfinding, obstacle
avoidance, jumping, or combat. Drop it on flat ground with no mobs to
see it work; expect it to get stuck on the first wall it hits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from brain.interfaces import AgentAction, BaseAgent
from vision.processing import GameState, ScreenState


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NavigationConfig:
    # Stop when within this many blocks (XZ plane) of the target.
    stop_distance: float = 2.5

    # Don't bother adjusting the camera if heading error is below this.
    yaw_tolerance_deg: float = 6.0

    # Mouse-pixels per degree of yaw at MC's default sensitivity (100 %).
    # Tune this if the agent overshoots / undershoots target yaw.
    mouse_per_degree: float = 6.5

    # Fraction of the total yaw-error correction to emit per tick.
    # Lower = smoother / slower, higher = snappier / more jittery.
    mouse_gain: float = 0.45

    # Cap per-tick mouse delta so we never produce a teleport-style flick.
    max_mouse_dx: int = 140

    # Walk forward only when heading is within this many degrees of target.
    walk_when_aligned_within: float = 35.0

    # Sprint while walking? (Requires Sprint=Hold in MC controls.)
    sprint: bool = False

    # Auto-target: when no explicit target is given, walk N blocks
    # forward from the player's starting heading.
    auto_target_forward_blocks: float = 25.0

    # Eat from this hotbar slot when hunger drops below threshold. Set
    # ``eat_below_hunger`` to 0 to disable. The slot must already hold
    # food in the player's inventory — the agent does not check.
    eat_below_hunger: float = 0.30
    food_slot: int = 9
    eat_hold_ticks: int = 30   # ~1.5 s at 20 Hz; enough for one food item


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class NavigationAgent(BaseAgent):
    """
    Walk to (target_x, target_z) using F3 position + facing.

    If ``target_x`` or ``target_z`` is None, the agent records the
    player's first observed position + yaw and sets the target to
    ``current + facing_vector * auto_target_forward_blocks``.
    """

    name = "navigation"

    def __init__(self,
                 target_x: Optional[float] = None,
                 target_z: Optional[float] = None,
                 config: Optional[NavigationConfig] = None):
        self.cfg = config or NavigationConfig()
        self._target_x = target_x
        self._target_z = target_z
        self._arrived = False
        self._eat_ticks_remaining = 0
        self._tick = 0
        self._last_log_tick = -1000

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._arrived = False
        self._eat_ticks_remaining = 0
        self._tick = 0
        self._last_log_tick = -1000

    # ------------------------------------------------------------------
    # Decide
    # ------------------------------------------------------------------

    def decide(self, state: GameState) -> AgentAction:
        self._tick += 1

        # Bail out if we're not in gameplay.
        if state.screen_state != ScreenState.PLAYING:
            return self._stop_action()

        # Without F3 data we cannot orient ourselves at all.
        f3 = state.f3
        if f3 is None or f3.x is None or f3.yaw is None:
            return self._stop_action()

        # If hunger eating is in progress, keep emitting use_item.
        if self._eat_ticks_remaining > 0:
            self._eat_ticks_remaining -= 1
            return AgentAction(
                movement=self.release_all_movement(),
                interact="use_item",
            )

        # Eat-from-hotbar logic. Only fires if the HUD is clearly rendered
        # (health > 5 %). When every bar reads 0 the player is most likely
        # in CREATIVE mode (Mojang hides all four bars in that state) and
        # we'd otherwise infinite-loop right-clicking food onto the world.
        hud_rendered = state.health > 0.05
        if (self.cfg.eat_below_hunger > 0
                and hud_rendered
                and state.hunger < self.cfg.eat_below_hunger
                and self.cfg.food_slot is not None):
            self._log(f"hunger={state.hunger:.2f} → eating from slot "
                      f"{self.cfg.food_slot} for {self.cfg.eat_hold_ticks} ticks")
            self._eat_ticks_remaining = self.cfg.eat_hold_ticks
            return AgentAction(
                movement=self.release_all_movement(),
                hotbar=self.cfg.food_slot,
                interact="use_item",
            )

        # Auto-target: first valid tick records the heading and projects.
        if self._target_x is None or self._target_z is None:
            yaw_rad = math.radians(f3.yaw)
            # MC convention: yaw 0 = facing +Z (south). The unit forward
            # vector in world space is (-sin(yaw), 0, cos(yaw)).
            fwd_x = -math.sin(yaw_rad)
            fwd_z =  math.cos(yaw_rad)
            self._target_x = f3.x + fwd_x * self.cfg.auto_target_forward_blocks
            self._target_z = f3.z + fwd_z * self.cfg.auto_target_forward_blocks
            self._log(f"auto-target set: ({self._target_x:.1f}, "
                      f"{self._target_z:.1f}) from start ({f3.x:.1f}, "
                      f"{f3.z:.1f}) facing yaw={f3.yaw:.1f}")

        # Distance to target.
        dx = self._target_x - f3.x
        dz = self._target_z - f3.z
        dist = math.hypot(dx, dz)

        if dist < self.cfg.stop_distance:
            if not self._arrived:
                self._log(f"ARRIVED at ({f3.x:.1f}, {f3.z:.1f}) — "
                          f"target ({self._target_x:.1f}, {self._target_z:.1f}), "
                          f"distance={dist:.2f}")
                self._arrived = True
            return self._stop_action()

        # Yaw error.
        desired_yaw = math.degrees(math.atan2(-dx, dz))
        yaw_err = self._normalize_angle(desired_yaw - f3.yaw)

        # Mouse delta.
        if abs(yaw_err) > self.cfg.yaw_tolerance_deg:
            raw = yaw_err * self.cfg.mouse_per_degree * self.cfg.mouse_gain
            mouse_dx = int(max(-self.cfg.max_mouse_dx,
                               min(self.cfg.max_mouse_dx, raw)))
        else:
            mouse_dx = 0

        # Walk forward only if heading is reasonably aligned.
        aligned = abs(yaw_err) < self.cfg.walk_when_aligned_within
        movement = {
            "forward":  aligned,
            "backward": False,
            "left":     False,
            "right":    False,
            "sprint":   aligned and self.cfg.sprint,
        }

        # Periodic progress log (once every 2 s at 20 Hz).
        if self._tick - self._last_log_tick >= 40:
            self._log(
                f"pos=({f3.x:.1f},{f3.z:.1f})  "
                f"target=({self._target_x:.1f},{self._target_z:.1f})  "
                f"dist={dist:5.2f}  yaw_err={yaw_err:+6.1f}°  "
                f"mouse_dx={mouse_dx:+4d}  walking={aligned}"
            )
            self._last_log_tick = self._tick

        return AgentAction(movement=movement, look_dx=mouse_dx)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stop_action(self) -> AgentAction:
        return AgentAction(movement=self.release_all_movement())

    @staticmethod
    def _normalize_angle(deg: float) -> float:
        """Normalise an angle to (-180, 180]."""
        while deg >  180.0: deg -= 360.0
        while deg <= -180.0: deg += 360.0
        return deg

    def _log(self, msg: str) -> None:
        print(f"[agent.navigation] {msg}")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_navigation_agent(settings: dict) -> NavigationAgent:
    """Build a NavigationAgent from the project settings dict."""
    nav_cfg = ((settings or {}).get("agent", {}) or {}).get("navigation", {}) or {}

    cfg = NavigationConfig()
    for key in (
        "stop_distance", "yaw_tolerance_deg", "mouse_per_degree",
        "mouse_gain", "max_mouse_dx", "walk_when_aligned_within",
        "auto_target_forward_blocks", "eat_below_hunger",
        "food_slot", "eat_hold_ticks",
    ):
        if key in nav_cfg and nav_cfg[key] is not None:
            setattr(cfg, key, type(getattr(cfg, key))(nav_cfg[key]))
    if "sprint" in nav_cfg:
        cfg.sprint = bool(nav_cfg["sprint"])

    tx = nav_cfg.get("target_x")
    tz = nav_cfg.get("target_z")
    return NavigationAgent(target_x=tx, target_z=tz, config=cfg)
