#!/usr/bin/env python3
"""
Wear the BEST armour the bot is carrying.

    python tools/equip_armor.py

Equips the best helmet / chestplate / leggings / boots from the inventory into
the armour slots (by protection tier: netherite > diamond > iron >
chainmail/turtle > gold > leather). Only upgrades — never swaps in a worse or
equal piece. Needs MC running + focused. Panic: Ctrl+Shift+F12.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

from tools.inventory_session import inventory_session
from agents.armor_equip import equip_best_armor


def main(argv=None) -> int:
    with inventory_session("armor") as sess:
        if sess is None:
            return 1
        eq = equip_best_armor(sess.ctl, catalog=sess.cat, log=print)
        if eq:
            print("[armor] equipped: " + ", ".join(
                f"{s}={i.split(':')[-1]}" for s, i in eq.items()))
        else:
            print("[armor] nothing to upgrade (already wearing the best, or none carried)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
