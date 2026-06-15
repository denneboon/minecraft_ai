#!/usr/bin/env python3
"""Offline self-test for knowledge/mining.py — tool-role-by-block (from the
catalog's mineable/* tags + suffix fallbacks) and the raw-item -> source-block
gather mapping (cobblestone <- stone, etc.)."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

from knowledge.mining import tool_role_for_block, gather_source_for

_fails = 0
def ok(m): print(f"  [OK]  {m}")
def bad(m):
    global _fails; _fails += 1; print(f"  [FAIL] {m}")


class _MockCat:
    """Catalog stand-in mapping a few block ids to their mineable/* tag."""
    _TAGS = {
        "stone": {"mineable/pickaxe"}, "deepslate": {"mineable/pickaxe"},
        "oak_log": {"mineable/axe"}, "dirt": {"mineable/shovel"},
        "oak_leaves": {"mineable/hoe"},
    }
    def block(self, bid):
        return SimpleNamespace(id=bid, tags=self._TAGS.get(bid.split(":")[-1], set()))


def main() -> int:
    print("=" * 56); print(" mining knowledge — offline self-test"); print("=" * 56)
    cat = _MockCat()

    # 1. tool role from catalog mineable/* tags.
    print("\n[1] tool role by mineable tag")
    cases = {
        "minecraft:stone": "pickaxe", "minecraft:deepslate": "pickaxe",
        "minecraft:oak_log": "axe", "minecraft:dirt": "shovel",
        "minecraft:oak_leaves": "hoe",
    }
    for bid, want in cases.items():
        got = tool_role_for_block(bid, cat)
        (ok if got == want else bad)(f"{bid.split(':')[-1]} -> {want} (got {got})")

    # 2. suffix fallback when NO catalog is supplied.
    print("\n[2] suffix fallback (no catalog)")
    fb = {
        "minecraft:cobblestone": "pickaxe", "minecraft:iron_ore": "pickaxe",
        "minecraft:granite": "pickaxe", "minecraft:birch_log": "axe",
        "minecraft:sand": "shovel", "minecraft:gravel": "shovel",
        "minecraft:spruce_leaves": "hoe",
    }
    for bid, want in fb.items():
        got = tool_role_for_block(bid)
        (ok if got == want else bad)(f"{bid.split(':')[-1]} -> {want} (got {got})")
    (ok if tool_role_for_block(None) is None else bad)("None block -> None role")

    # 3. raw item -> (source block, tool).
    print("\n[3] gather source for a raw item")
    src, role = gather_source_for("minecraft:cobblestone", cat)
    (ok if src == "minecraft:stone" and role == "pickaxe" else bad)(
        f"cobblestone <- (stone, pickaxe) (got {src}, {role})")
    src2, role2 = gather_source_for("minecraft:cobbled_deepslate", cat)
    (ok if src2 == "minecraft:deepslate" and role2 == "pickaxe" else bad)(
        f"cobbled_deepslate <- (deepslate, pickaxe) (got {src2}, {role2})")
    # an item with no special source mines the block of its own id.
    src3, role3 = gather_source_for("minecraft:dirt", cat)
    (ok if src3 == "minecraft:dirt" and role3 == "shovel" else bad)(
        f"dirt <- (dirt, shovel) (got {src3}, {role3})")
    (ok if gather_source_for(None) == (None, None) else bad)("None item -> (None, None)")

    print("\n" + ("ALL MINING-KNOWLEDGE TESTS PASSED" if not _fails
                  else f"{_fails} CHECK(S) FAILED"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
