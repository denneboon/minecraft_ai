# control/action_wrapper.py
"""
ActionWrapper — the single interface between the AI and the hardware input layer.

The AI never calls Keyboard or Mouse directly. It calls ActionWrapper methods,
which translate high-level discrete/continuous actions into timed key/mouse events.

Design goals:
- Fully serialisable action space (every method maps 1-to-1 to a named action string
  and a small set of numeric arguments). This lets you log, replay, and train on actions.
- All timing/cooldown logic stays here or in keyboard.py/mouse.py — not in the agent.
- Swapping backends (pynput → SendInput, etc.) never requires touching agent code.
- The action space is intentionally minimal. Add actions only when the agent needs them.

Action categories:
  Movement : forward, backward, left, right, jump, sneak, sprint
  Camera   : look(dx, dy) — smooth; flick(dx, dy) — fast
  Combat   : attack, use_item, drop_item
  Hotbar   : slot(1-9), scroll_up, scroll_down
  Inventory: open_inventory, close_screen
  System   : no_op

Usage (from an agent tick):
    wrapper.execute("forward", press=True)
    wrapper.execute("look", dx=45, dy=0)
    wrapper.execute("slot", slot=3)
    wrapper.execute("attack")
    wrapper.execute("no_op")
"""

from __future__ import annotations

from typing import Any

from control.keyboard import Keyboard
from control.mouse import Mouse


# ---------------------------------------------------------------------------
# Discrete action names (the AI's vocabulary)
# ---------------------------------------------------------------------------

MOVEMENT_ACTIONS = {"forward", "backward", "left", "right", "jump", "sneak", "sprint"}
CAMERA_ACTIONS   = {"look", "flick"}
COMBAT_ACTIONS   = {"attack", "use_item", "drop_item"}
HOTBAR_ACTIONS   = {"slot", "scroll_up", "scroll_down"}
UI_ACTIONS       = {"open_inventory", "close_screen"}
META_ACTIONS     = {"no_op"}

ALL_ACTIONS = (
    MOVEMENT_ACTIONS | CAMERA_ACTIONS | COMBAT_ACTIONS |
    HOTBAR_ACTIONS   | UI_ACTIONS     | META_ACTIONS
)


# ---------------------------------------------------------------------------
# ActionWrapper
# ---------------------------------------------------------------------------

class ActionWrapper:
    """
    Translates AI action commands into hardware input calls.

    Parameters
    ----------
    keyboard : Keyboard
        Fully started Keyboard instance.
    mouse : Mouse
        Fully started Mouse instance.
    gate : InputGate, optional
        Shared gate — wrapper will check it before dispatching.
    """

    def __init__(self, keyboard: Keyboard, mouse: Mouse, gate=None):
        self._kb   = keyboard
        self._ms   = mouse
        self._gate = gate
        self._attack_held = False     # continuous left-click (mining/combat)
        self._use_held = False        # continuous right-click (eating)

    def set_attack(self, held: bool) -> None:
        """Hold or release the attack button CONTINUOUSLY (idempotent) —
        mining and sustained combat need a held left-click, not per-tick
        clicks. Call every tick with the desired state. Releasing always
        succeeds (even with the gate closed) so the button never sticks."""
        if held:
            if self._gate and not self._gate.allow():
                return                # gate shut: don't start attacking
            if not self._attack_held:
                self._ms.left_press()
                self._attack_held = True
        else:
            if self._attack_held:
                self._ms.left_release()
                self._attack_held = False

    def set_use(self, held: bool) -> None:
        """Hold or release the use/right button CONTINUOUSLY (idempotent).
        Eating needs a sustained right-click (~1.6 s) — per-tick clicks
        restart the eat each tick. Placing stays a one-shot ``use_item``.
        Releasing always succeeds so the button never sticks."""
        if held:
            if self._gate and not self._gate.allow():
                return
            if not self._use_held:
                self._ms.right_press()
                self._use_held = True
        else:
            if self._use_held:
                self._ms.right_release()
                self._use_held = False

    # ------------------------------------------------------------------
    # Primary entry point — called once per agent tick
    # ------------------------------------------------------------------

    def execute(self, action: str, **kwargs: Any) -> bool:
        """
        Dispatch a named action with optional keyword arguments.

        Returns True if the action was dispatched, False if blocked by gate
        or unrecognised.

        Common kwargs:
          press  (bool)  — for movement actions: True=hold, False=release
          dx, dy (int)   — for look/flick
          slot   (int)   — 1-based hotbar slot number
          duration (float) — override click/tap hold time in seconds
        """
        if self._gate and not self._gate.allow():
            return False

        if action not in ALL_ACTIONS:
            raise ValueError(f"Unknown action: {action!r}. Valid: {sorted(ALL_ACTIONS)}")

        # --- Movement ---
        if action in MOVEMENT_ACTIONS:
            return self._dispatch_movement(action, **kwargs)

        # --- Camera ---
        if action == "look":
            dx = int(kwargs.get("dx", 0))
            dy = int(kwargs.get("dy", 0))
            self._ms.track_target(dx, dy)
            return True

        if action == "flick":
            dx = int(kwargs.get("dx", 0))
            dy = int(kwargs.get("dy", 0))
            self._ms.flick(dx, dy)
            return True

        # --- Combat ---
        if action == "attack":
            dur = kwargs.get("duration")
            self._ms.left_click(duration=dur)
            return True

        if action == "use_item":
            dur = kwargs.get("duration")
            self._ms.right_click(duration=dur)
            return True

        if action == "drop_item":
            self._kb.tap("drop")
            return True

        # --- Hotbar ---
        if action == "slot":
            slot = int(kwargs.get("slot", 1))
            self._kb.select_hotbar_slot(slot)
            return True

        if action == "scroll_up":
            self._ms.scroll_up(int(kwargs.get("amount", 1)))
            return True

        if action == "scroll_down":
            self._ms.scroll_down(int(kwargs.get("amount", 1)))
            return True

        # --- UI ---
        if action == "open_inventory":
            self._kb.tap("inventory")
            return True

        if action == "close_screen":
            self._kb.tap("escape")
            return True

        # --- Meta ---
        if action == "no_op":
            return True

        return False  # should never reach here

    # ------------------------------------------------------------------
    # Convenience: hold/release a set of movement keys each tick
    # ------------------------------------------------------------------

    def set_movement_state(self, **flags: bool) -> None:
        """
        Set any combination of movement keys held/released in one call.

        Example (agent tick):
            wrapper.set_movement_state(forward=True, sprint=True, left=False)

        Only the keys you pass are changed; others stay as they are.
        """
        for action, held in flags.items():
            if action not in MOVEMENT_ACTIONS:
                raise ValueError(f"Not a movement action: {action!r}")
            self._dispatch_movement(action, press=held)

    def release_all_movement(self) -> None:
        """Release every movement key. Call on episode reset or emergency."""
        for action in MOVEMENT_ACTIONS:
            self._dispatch_movement(action, press=False)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _dispatch_movement(self, action: str, press: bool = True, **_: Any) -> bool:
        key_map = {
            "forward":  "move_forward",
            "backward": "move_backward",
            "left":     "move_left",
            "right":    "move_right",
            "jump":     "jump",
            "sneak":    "sneak",
            "sprint":   "sprint",
        }
        key = key_map[action]
        if press:
            self._kb.hold(key)
        else:
            self._kb.release_action(key)
        return True
