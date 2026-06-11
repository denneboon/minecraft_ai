#!/usr/bin/env python3
"""Offline self-test for control/inventory_control.py — verifies the
hotkey-first action sequences + the carry-slot policy without a live game
(mock mouse/keyboard/capture/reader; real slot geometry)."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

from control.inventory_control import InventoryController, grid_slot
from vision.inventory_layout import slot_rects

_fails = 0
def ok(m): print(f"  [OK]  {m}")
def bad(m):
    global _fails; _fails += 1; print(f"  [FAIL] {m}")


class _Mouse:
    def __init__(self, log): self.log = log
    def move_to_screen_xy(self, x, y): self.log.append(("move", x, y))
    def left_click(self): self.log.append(("lclick",))
    def right_click(self): self.log.append(("rclick",))

class _Kb:
    def __init__(self, log): self.log = log
    def tap(self, k): self.log.append(("tap", k))
    def press(self, k): self.log.append(("press", k))
    def release(self, k): self.log.append(("release", k))

class _Cap:
    def get_frame(self): return np.zeros((1080, 1920, 3), dtype=np.uint8)

class _Reader:
    def __init__(self, snap): self._snap = snap
    def read(self, frame, container=None): return self._snap


def _snap(hotbar_items):
    """hotbar_items: dict {n(1-9): item_id or None}."""
    slots = {}
    for n in range(1, 10):
        slots[f"hotbar_{n-1}"] = SimpleNamespace(item=hotbar_items.get(n))
    return SimpleNamespace(slots=slots)


def _ctl(log, snap):
    hb = SimpleNamespace(cfg=SimpleNamespace(
        slot_roles={1: "sword", 2: "axe", 3: "pickaxe", 5: "blocks", 9: "food"}))
    return InventoryController(_Mouse(log), _Kb(log), _Reader(snap), hb, _Cap())


def main() -> int:
    print("=" * 56); print(" inventory_control — offline self-test"); print("=" * 56)
    rects = slot_rects((1080, 1920, 3), layout="player_inventory", ui_scale=2)
    center = lambda name: tuple(int(v) for v in rects[name].center())

    # 1. grid_slot row-major mapping (2x2 and 3x3).
    print("\n[1] grid_slot row-major")
    (ok if grid_slot(0, 0, 2) == "craft_in_0" and grid_slot(1, 0, 2) == "craft_in_2"
        and grid_slot(1, 1, 2) == "craft_in_3" else bad)("2x2 grid maps row-major")
    (ok if grid_slot(2, 1, 3) == "craft_in_7" else bad)("3x3 grid maps row-major")

    # 2. carry-slot policy: prefer 6 when empty; never a role-occupied slot.
    print("\n[2] carry-slot policy")
    c = _ctl([], None)
    (ok if c.carry_slot(_snap({})) == 6 else bad)("empty hotbar -> prefer slot 6")
    # 6 occupied (non-role item) but 4 empty -> pick 4 (clean/empty first)
    (ok if c.carry_slot(_snap({6: "minecraft:dirt"})) == 4 else bad)(
        "slot 6 full -> next empty (4)")
    # every slot full: must avoid role-held slots -> a non-role slot (4/6/7/8)
    full = {n: "minecraft:dirt" for n in range(1, 10)}
    full.update({2: "minecraft:iron_axe"})           # axe in its role slot
    pick = c.carry_slot(_snap(full))
    (ok if pick in (4, 6, 7, 8) else bad)(f"full hotbar -> non-role slot ({pick})")
    (ok if pick != 2 else bad)("never borrows the axe's slot (role-held)")

    # 3. move_stack: hover src, tap carry, hover dst, tap carry (no drag).
    print("\n[3] move_stack = two number-key swaps")
    log = []; c = _ctl(log, None)
    sp = _snap({})                                   # 6 empty -> carry 6
    c.move_stack("inv_5", "craft_in_0", sp)
    moves = [e for e in log if e[0] == "move"]
    taps = [e for e in log if e[0] == "tap"]
    (ok if [e[1] for e in taps] == ["6", "6"] else bad)(f"two '6' swaps ({taps})")
    (ok if moves[0][1:] == center("inv_5") and moves[1][1:] == center("craft_in_0")
     else bad)("hovers src then dst (correct slot centers)")
    (ok if not any(e[0] in ("lclick", "rclick") for e in log) else bad)(
        "no clicks (pure hotkey move)")

    # 4. distribute_one: pick up, right-click each cell, return remainder.
    print("\n[4] distribute_one = pickup + place-one per cell")
    log = []; c = _ctl(log, None)
    c.distribute_one("inv_3", ["craft_in_0", "craft_in_1"])
    seq = [e[0] for e in log if e[0] in ("lclick", "rclick")]
    (ok if seq == ["lclick", "rclick", "rclick", "lclick"] else bad)(
        f"pickup, place-one x2, return ({seq})")

    # 5. take_result: shift-held left click on the result.
    print("\n[5] take_result = shift-click result")
    log = []; c = _ctl(log, None)
    c.take_result()
    kinds = [e for e in log if e[0] in ("press", "lclick", "release")]
    (ok if [k[0] for k in kinds] == ["press", "lclick", "release"]
        and kinds[0][1] == "shift" else bad)(f"shift+click ({kinds})")

    # 6. to_hotbar: inv slot -> single number-key swap into a hotbar slot;
    # already-in-hotbar is a no-op that just returns the number.
    print("\n[6] to_hotbar = one swap into the hotbar")
    log = []; c = _ctl(log, None)
    n = c.to_hotbar("inv_8", _snap({}))              # 6 empty -> swap into 6
    taps = [e[1] for e in log if e[0] == "tap"]
    moves = [e for e in log if e[0] == "move"]
    (ok if n == 6 and taps == ["6"] else bad)(f"inv -> hotbar via one '6' swap (n={n}, {taps})")
    (ok if moves and moves[0][1:] == center("inv_8") else bad)("hovers the inv slot")
    log2 = []; c2 = _ctl(log2, None)
    n2 = c2.to_hotbar("hotbar_3", _snap({}))         # already hotbar -> no-op
    (ok if n2 == 4 and not log2 else bad)(f"already in hotbar -> no-op, returns 4 (got {n2})")

    print("\n" + ("ALL INVENTORY-CONTROL TESTS PASSED" if not _fails
                  else f"{_fails} CHECK(S) FAILED"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
