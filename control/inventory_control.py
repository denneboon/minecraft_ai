"""
Hotkey-first inventory manipulation.

Moves items inside an open container by HOVERING a slot (absolute OS-cursor
move) and using Minecraft's number-key SWAP hotkey (hover a slot + press
1-9 -> swaps that slot with hotbar slot N). This is the preferred path —
no click-and-drag. Stack-splitting (one item into each of several grid
cells, which number keys can't express) falls back to left-click pick-up +
right-click place-one. Dragging is intentionally NOT used.

Carry-slot policy (per project owner): to shuttle an item via a hotbar
slot, prefer slot 6, and NEVER use a hotbar slot that currently holds the
item it's assigned to (e.g. the axe in slot 2) — only empty or non-role
slots are borrowed.

This is the execution layer under the crafting behaviour; it consumes the
``CraftStep`` plans from ``knowledge/recipes.py``.
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

from vision.inventory_layout import slot_rects

# Hotbar number keys in MC are 1-9. The carry-slot preference order: 6 first
# (project owner's choice), then the other unreserved slots, then the rest.
_CARRY_PREF = [6, 4, 7, 8, 1, 2, 3, 5, 9]


def grid_slot(row: int, col: int, width: int) -> str:
    """The container slot name for grid cell (row, col). The 2x2 inventory
    grid (width=2) and 3x3 table grid (width=3) are both row-major
    craft_in_(row*width + col) — verified against the live slot rects."""
    return f"craft_in_{row * width + col}"


class InventoryController:
    def __init__(self, mouse, keyboard, reader, hotbar, capture, *,
                 ui_scale: int = 2, container: str = "player_inventory",
                 settle: float = 0.16):
        self._m = mouse
        self._kb = keyboard
        self._reader = reader
        self._hb = hotbar
        self._cap = capture
        self._ui = ui_scale
        self.container = container
        self._settle = settle
        self._rects = None                 # slot_name -> SlotRect (last read)

    # ── perception ───────────────────────────────────────────────────
    def read(self):
        """Capture a frame, parse the open container -> InventorySnapshot,
        and cache the slot rects for hovering."""
        frame = self._cap.get_frame()
        self._rects = slot_rects(frame.shape, layout=self.container,
                                 ui_scale=self._ui)
        return self._reader.read(frame, container=self.container)

    def _center(self, slot_name: str) -> Tuple[int, int]:
        if self._rects is None or slot_name not in self._rects:
            self.read()
        r = self._rects[slot_name]
        cx, cy = r.center()
        return int(cx), int(cy)

    # ── primitive actions (hotkey-first) ─────────────────────────────
    def hover(self, slot_name: str) -> None:
        x, y = self._center(slot_name)
        self._m.move_to_screen_xy(x, y)
        time.sleep(self._settle)

    def number_swap(self, slot_name: str, hotbar_n: int) -> None:
        """Swap ``slot_name`` with hotbar slot ``hotbar_n`` (1-9) via the
        vanilla number-key hotkey. The preferred way to move a whole stack."""
        self.hover(slot_name)
        self._kb.tap(str(int(hotbar_n)))
        time.sleep(self._settle)

    def left_click(self, slot_name: str) -> None:
        self.hover(slot_name)
        self._m.left_click()
        time.sleep(self._settle)

    def right_click(self, slot_name: str) -> None:
        self.hover(slot_name)
        self._m.right_click()
        time.sleep(self._settle)

    def shift_left_click(self, slot_name: str) -> None:
        self.hover(slot_name)
        try:
            self._kb.press("shift")
            self._m.left_click()
        finally:
            self._kb.release("shift")
        time.sleep(self._settle)

    # ── container open/close ─────────────────────────────────────────
    def open_inventory(self) -> None:
        self._kb.tap("e"); time.sleep(0.35)

    def close(self) -> None:
        self._kb.tap("escape"); time.sleep(0.25)

    # ── carry-slot selection ─────────────────────────────────────────
    def carry_slot(self, snap) -> int:
        """Choose a hotbar slot (1-9) to shuttle items through. Prefers an
        EMPTY slot (clean 2-swap, no displacement), slot 6 first; never a
        slot holding its assigned role item."""
        roles = getattr(getattr(self._hb, "cfg", None), "slot_roles", {}) or {}
        slots = getattr(snap, "slots", {}) or {}

        def _empty(n: int) -> bool:
            sc = slots.get(f"hotbar_{n - 1}")
            return sc is None or getattr(sc, "item", None) is None

        for n in _CARRY_PREF:                 # empty slot -> cleanest
            if _empty(n):
                return n
        for n in _CARRY_PREF:                 # else a non-role slot
            if roles.get(n) is None:
                return n
        return 6

    # ── compound moves used by crafting ──────────────────────────────
    def move_stack(self, src: str, dst: str, snap) -> int:
        """Move src's whole stack to (empty) dst via a carry hotbar slot —
        two number-key swaps, no drag. Returns the carry slot used."""
        n = self.carry_slot(snap)
        self.number_swap(src, n)              # src stack -> carry
        self.number_swap(dst, n)              # carry -> dst (dst was empty)
        return n

    def distribute_one(self, src: str, cells: List[str]) -> None:
        """Put ONE item from src's stack into each cell. Number keys move a
        whole stack, so split with pick-up + right-click place-one, then
        return the remainder."""
        if not cells:
            return
        self.left_click(src)                  # whole stack onto cursor
        for cell in cells:
            self.right_click(cell)            # drop one
        self.left_click(src)                  # remainder back

    def take_result(self) -> None:
        """Collect the crafting output: shift-click pulls every craftable
        result straight into the inventory."""
        self.shift_left_click("craft_result")
