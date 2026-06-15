#!/usr/bin/env python3
"""
Arrange the hotbar: put the BEST item of each role into its reserved slot.

    python tools/arrange_hotbar.py

Best = highest material tier for tools (netherite > diamond > iron > copper >
gold > stone > wood), best food quality for food, and most-count for blocks.
The slots come from hotbar.slot_roles in config/settings.yaml (1=sword 2=pickaxe
3=axe 4=shovel 5=hoe 6=blocks 7=food by default). Needs MC running + focused.
Panic: Ctrl+Shift+F12.
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
from agents.hotbar_arranger import arrange_hotbar


def main(argv=None) -> int:
    with inventory_session("hotbar") as sess:
        if sess is None:
            return 1
        roles = {int(k): str(v) for k, v in
                 ((sess.settings.get("hotbar") or {}).get("slot_roles") or {}).items()}
        xfood = tuple((sess.settings.get("hotbar") or {}).get("extra_food") or ())
        arr = arrange_hotbar(sess.ctl, slot_roles=roles or None, catalog=sess.cat,
                             extra_food=xfood, log=print)
        if arr:
            print("[hotbar] arranged: " + ", ".join(
                f"{r}={i.split(':')[-1]}" for r, i in arr.items()))
        else:
            print("[hotbar] nothing to arrange (no role items found)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
