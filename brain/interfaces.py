# brain/interfaces.py
"""
Agent contract — what every agent (rule-based, IL, RL, hybrid) must
return per tick, and what the runtime guarantees in exchange.

Why a structured AgentAction (not just a string)
-------------------------------------------------
The original ``_stub_agent`` returned a single ``{"action": ...}`` dict.
That's fine for a *one* action per tick agent, but real agents need to
hold movement keys WHILE turning the camera WHILE selecting a hotbar
slot — all in the same 50 ms tick. A structured AgentAction lets the
runtime dispatch each field independently:

  * ``movement`` dict   -> ActionWrapper.set_movement_state(**dict)
  * ``look_dx/dy``      -> Mouse.track_target / Mouse.flick
  * ``interact``        -> attack / use_item / drop_item
  * ``hotbar``          -> select_hotbar_slot
  * ``inventory_toggle`` -> tap E

Empty fields are *no-ops* — they don't reset prior state. So an agent
holding W only needs to set ``movement={"forward": True}`` once and can
leave it out next tick if W should stay held.

Future agents (PyTorch policies) will produce the same AgentAction, so
the runtime never has to know whether the brain is a hand-written rule
or a 200 M parameter transformer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from vision.processing import GameState


# ---------------------------------------------------------------------------
# AgentAction
# ---------------------------------------------------------------------------

@dataclass
class AgentAction:
    """
    Per-tick agent output.

    Fields are independent and additive: the runtime dispatches whatever
    is set. An empty AgentAction is a true no-op (no key state changes).
    """

    # Movement keys to drive this tick. Keys: forward / backward / left /
    # right / jump / sneak / sprint. Values are bool (True = hold, False =
    # release). Keys not present in the dict are LEFT ALONE — so the
    # runtime keeps holding W from the previous tick if the agent doesn't
    # explicitly release it.
    movement: Dict[str, bool] = field(default_factory=dict)

    # Camera delta this tick, in screen pixels. Positive dx = look right,
    # positive dy = look down. The runtime applies easing via Mouse.
    look_dx: int = 0
    look_dy: int = 0

    # Continuous camera VELOCITY in screen-pixels per second. Used by
    # humanlike-motion agents (e.g. WorldExplorer) that prefer one
    # smooth turn over many per-tick nudges. When set, the runtime
    # routes through ``Mouse.set_velocity`` — a background thread
    # emits sub-pixel-rate micro-motions until the next velocity
    # update or a zero-velocity command. ``look_vx == look_vy == 0``
    # halts continuous motion. ``look_dx``/``look_dy`` still work in
    # parallel (one-shot easing) — agents typically use one OR the
    # other.
    look_vx: float = 0.0
    look_vy: float = 0.0

    # When True, the runtime ALWAYS dispatches ``set_velocity`` even
    # if both velocity components are zero — used by agents that want
    # to halt an in-progress velocity command without having to send
    # any other field. Without this flag, ``(0, 0)`` would short-circuit
    # the dispatcher (since zero velocity is the default no-op) and an
    # earlier non-zero velocity would keep the camera spinning.
    force_velocity: bool = False

    # One-shot interaction. Fires the moment it's set; the agent must
    # re-emit it the next tick if it wants to keep clicking.
    # Valid: "attack" | "use_item" | "drop_item" | None.
    interact: Optional[str] = None

    # Select hotbar slot 1..9. None = don't change selection.
    hotbar: Optional[int] = None

    # Toggle inventory screen (tap E).
    inventory_toggle: bool = False

    # Free-form extras for future agents (e.g. chat output, planner state).
    extras: Dict[str, Any] = field(default_factory=dict)

    def is_noop(self) -> bool:
        return (
            not self.movement
            and self.look_dx == 0 and self.look_dy == 0
            and self.look_vx == 0.0 and self.look_vy == 0.0
            and not self.force_velocity
            and self.interact is None
            and self.hotbar is None
            and not self.inventory_toggle
            and not self.extras
        )


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------

class BaseAgent:
    """
    Common interface for all agents.

    Lifecycle
    ---------
    1. Construct (typically with task-specific kwargs).
    2. ``reset()`` once at episode start. Clears internal state.
    3. ``decide(state)`` once per tick. Must be cheap — runtime budget
       at 20 Hz is ~50 ms, of which capture + processing already burns
       ~10 ms, OCR (when it fires at 3 Hz) burns ~30 ms.
    4. ``shutdown()`` once at episode end. Optional cleanup.

    Sub-classes override ``decide`` and (optionally) ``reset`` / ``shutdown``.
    """

    name: str = "base"

    def reset(self) -> None:
        """Clear per-episode state. Called once before the first decide()."""
        pass

    def shutdown(self) -> None:
        """Called once after the loop stops. Default: nothing."""
        pass

    def decide(self, state: GameState) -> AgentAction:
        raise NotImplementedError(
            f"{type(self).__name__}.decide() must return an AgentAction"
        )

    # ------------------------------------------------------------------
    # Convenience helpers for subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def release_all_movement() -> Dict[str, bool]:
        """Return a movement dict that releases every key."""
        return {
            "forward":  False, "backward": False,
            "left":     False, "right":    False,
            "jump":     False, "sneak":    False, "sprint": False,
        }


__all__ = ["AgentAction", "BaseAgent"]
