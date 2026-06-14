#!/usr/bin/env python3
"""
Prep a CLEAN scene for an end-to-end make.py test (operator/dev tool).

Sends chat commands to: empty the player's inventory (/clear), remove dropped
items nearby, and clear any stray crafting tables left by prior runs in a box
around the player. Optionally teleport first (``--tp X Y Z``) — e.g. out to a
fresh forest. Cheats must be on.

    python tools/_prep_scene.py
    python tools/_prep_scene.py --tp 3000 70 3000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as M
from utils.focus import activate_minecraft, _find_minecraft_hwnd
from control.keyboard import Keyboard


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tp", nargs=3, type=int, default=None, metavar=("X", "Y", "Z"))
    ap.add_argument("--radius", type=int, default=14)
    ap.add_argument("--give-axe", action="store_true",
                    help="also give a stone axe (default: leave inventory empty)")
    args = ap.parse_args(argv)

    if not _find_minecraft_hwnd():
        print("[prep] Minecraft not running."); return 2
    activate_minecraft(); time.sleep(0.4)
    kb = Keyboard(); kb.start()

    r = args.radius
    cmds = []
    if args.tp:
        cmds.append(f"/tp @s {args.tp[0]} {args.tp[1]} {args.tp[2]}")
    cmds += [
        "/clear @s",
        "/kill @e[type=item,distance=..40]",
        # Remove stray crafting tables from prior runs around us.
        f"/fill ~-{r} ~-4 ~-{r} ~{r} ~6 ~{r} air replace minecraft:crafting_table",
    ]
    if args.give_axe:
        cmds.append("/give @s minecraft:stone_axe 1")
    try:
        for c in cmds:
            print(f"[prep] {c}")
            M._send_chat_message(kb, c)
            time.sleep(0.35)
    finally:
        kb.stop()
    print("[prep] scene clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
