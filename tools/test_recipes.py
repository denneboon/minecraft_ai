#!/usr/bin/env python3
"""Offline self-test for knowledge/recipes.py — parse vanilla recipes,
resolve tag ingredients, know 2x2-vs-table, and plan craft chains."""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

_fails = 0
def ok(m): print(f"  [OK]  {m}")
def bad(m):
    global _fails; _fails += 1; print(f"  [FAIL] {m}")


def main() -> int:
    print("=" * 56); print(" recipes — offline self-test"); print("=" * 56)
    from knowledge.catalog import Catalog
    from vision.mc_assets import MCAssets
    from knowledge import recipes as R
    a = MCAssets.load()
    cat = Catalog.load(a)

    # 1. oak_planks: shapeless, 1 log -> 4 planks, fits 2x2, tag resolved.
    print("\n[1] oak_planks (shapeless, tag ingredient)")
    rp = R.recipe_for("minecraft:oak_planks", a, cat)
    (ok if rp and not rp.shaped and rp.result_count == 4 else bad)(
        f"shapeless, makes 4 ({rp and rp.result_count})")
    (ok if rp and rp.fits_2x2 else bad)("fits 2x2 (inventory)")
    opts = rp.placements()[0][1] if rp and rp.placements() else []
    (ok if "minecraft:oak_log" in opts else bad)(
        f"#oak_logs tag resolved to oak_log ({opts[:3]})")

    # 2. stick: shaped, 2 planks vertical, fits 2x2.
    print("\n[2] stick (shaped 1x2)")
    rs = R.recipe_for("minecraft:stick", a, cat)
    (ok if rs and rs.shaped and rs.height == 2 and rs.width == 1 else bad)(
        f"shaped 1 wide x 2 tall ({rs and (rs.width, rs.height)})")
    (ok if rs and rs.fits_2x2 and rs.result_count == 4 else bad)(
        "fits 2x2, makes 4 sticks")

    # 3. crafting_table: 2x2 planks, fits 2x2.
    print("\n[3] crafting_table (shaped 2x2)")
    rc = R.recipe_for("minecraft:crafting_table", a, cat)
    (ok if rc and rc.width == 2 and rc.height == 2 and rc.fits_2x2 else bad)(
        "2x2 planks, fits inventory grid")
    (ok if rc and len(rc.cells) == 4 else bad)(f"4 filled cells ({rc and len(rc.cells)})")

    # 4. wooden_pickaxe: 3x3, needs a table.
    print("\n[4] wooden_pickaxe (shaped 3x3 -> needs table)")
    rw = R.recipe_for("minecraft:wooden_pickaxe", a, cat)
    (ok if rw and rw.width == 3 and not rw.fits_2x2 else bad)(
        f"3 wide -> needs crafting table ({rw and (rw.width, rw.height)})")

    # 5. plan_step: oak_log on hand -> place into a cell.
    print("\n[5] plan_step picks a concrete held ingredient")
    st = R.plan_step("minecraft:oak_planks", {"minecraft:oak_log": 3}, a, cat)
    (ok if st and list(st.cell_items.values()) == ["minecraft:oak_log"] else bad)(
        f"places oak_log in the grid ({st and st.cell_items})")
    (ok if st and not st.needs_table else bad)("planks don't need a table")
    none = R.plan_step("minecraft:oak_planks", {"minecraft:cobblestone": 9}, a, cat)
    (ok if none is None else bad)("can't craft planks without logs -> None")

    # 6. plan_craft chains logs -> planks -> sticks.
    print("\n[6] plan_craft chains dependencies")
    chain = R.plan_craft("minecraft:stick", {"minecraft:oak_log": 1}, a, cat)
    (ok if chain and [s.result_id for s in chain]
        == ["minecraft:oak_planks", "minecraft:stick"] else bad)(
        f"1 log -> planks -> sticks ({chain and [s.result_id.split(':')[-1] for s in chain]})")
    tbl = R.plan_craft("minecraft:crafting_table", {"minecraft:oak_log": 1}, a, cat)
    (ok if tbl and tbl[-1].result_id == "minecraft:crafting_table" else bad)(
        f"1 log -> planks -> table ({tbl and len(tbl)} steps)")
    (ok if R.plan_craft("minecraft:stick", {"minecraft:cobblestone": 9}, a, cat) is None
     else bad)("no wood -> can't plan sticks -> None")

    print("\n" + ("ALL RECIPE TESTS PASSED" if not _fails
                  else f"{_fails} CHECK(S) FAILED"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
