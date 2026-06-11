#!/usr/bin/env python3
"""
Offline self-test for the hotbar slot-role system (no Minecraft needed):
knowledge/item_roles.py classification + control/hotbar.py manager + the
settings-driven builder.
"""
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


def test_item_roles() -> None:
    print("\n[1] item_role classification")
    from vision.mc_assets import MCAssets
    from knowledge.catalog import Catalog
    from knowledge.item_roles import item_role
    cat = Catalog.load(MCAssets.load())
    cases = {
        "minecraft:diamond_sword": "sword",
        "minecraft:netherite_sword": "sword",
        "minecraft:iron_axe": "axe",
        "minecraft:stone_pickaxe": "pickaxe",
        "minecraft:wooden_shovel": "shovel",
        "minecraft:diamond_hoe": "hoe",
        "minecraft:cooked_beef": "food",
        "minecraft:bread": "food",
        "minecraft:apple": "food",
        "minecraft:oak_log": "blocks",
        "minecraft:cobblestone": "blocks",
        "minecraft:dirt": "blocks",
        "minecraft:diamond_helmet": "armor",
        "minecraft:stick": None,
        "minecraft:iron_ingot": None,
        None: None,
    }
    for iid, want in cases.items():
        got = item_role(iid, cat)
        (ok if got == want else bad)(f"{str(iid):28} -> {got} (want {want})")
    # Suffix fallback works WITHOUT a catalog.
    if item_role("minecraft:golden_axe", None) == "axe":
        ok("suffix fallback (no catalog): golden_axe -> axe")
    else:
        bad("suffix fallback failed without catalog")
    # extra_food extension.
    from knowledge.item_roles import item_role as ir
    if ir("minecraft:weird_snack", None, extra_food=["weird_snack"]) == "food":
        ok("extra_food extension works")
    else:
        bad("extra_food extension failed")


def test_hotbar_manager() -> None:
    print("\n[2] HotbarManager slot selection")
    from vision.mc_assets import MCAssets
    from knowledge.catalog import Catalog
    from control.hotbar import HotbarManager, HotbarConfig
    cat = Catalog.load(MCAssets.load())
    cfg = HotbarConfig(slot_roles={1: "sword", 2: "axe", 3: "pickaxe",
                                   5: "blocks", 9: "food"})
    hb = HotbarManager(cfg, catalog=cat)

    # Tidy hotbar: each reserved slot holds the right role.
    hb.update([
        "minecraft:diamond_sword",   # 1 sword
        "minecraft:iron_axe",        # 2 axe
        "minecraft:stone_pickaxe",   # 3 pickaxe
        None,                        # 4
        "minecraft:oak_planks",      # 5 blocks
        None, None, None,            # 6-8
        "minecraft:cooked_beef",     # 9 food
    ])
    for role, want in [("sword", 1), ("axe", 2), ("pickaxe", 3),
                       ("blocks", 5), ("food", 9)]:
        got = hb.slot_for_role(role)
        (ok if got == want else bad)(f"tidy: slot_for_role({role}) -> {got} (want {want})")
    if not hb.mis_stocked():
        ok("tidy hotbar: nothing mis-stocked")
    else:
        bad(f"tidy hotbar reported mis-stock: {hb.mis_stocked()}")
    if hb.food_slot() == 9 and hb.axe_slot() == 2 and hb.blocks_slot() == 5:
        ok("convenience accessors (food/axe/blocks) correct")
    else:
        bad("convenience accessors wrong")

    # Untidy: axe is in slot 7 (not its reserved slot 2), slot 2 empty.
    hb.update([
        "minecraft:diamond_sword", None, "minecraft:stone_pickaxe", None,
        "minecraft:cobblestone", None, "minecraft:iron_axe", None,
        "minecraft:bread",
    ])
    if hb.slot_for_role("axe") == 7:
        ok("untidy: finds axe in slot 7 (fallback past empty reserved slot)")
    else:
        bad(f"untidy axe fallback wrong: {hb.slot_for_role('axe')}")
    ms = dict((s, r) for s, r, _ in hb.mis_stocked())
    if ms.get(2) == "axe":
        ok("mis_stocked flags empty reserved axe slot (2)")
    else:
        bad(f"mis_stocked missed slot 2: {hb.mis_stocked()}")

    # Missing role entirely -> None (don't crash a behaviour).
    hb.update([None] * 9)
    if hb.food_slot() is None and hb.axe_slot() is None:
        ok("empty hotbar: roles resolve to None (safe)")
    else:
        bad("empty hotbar should give None")


def test_builder() -> None:
    print("\n[3] build_hotbar_manager from settings")
    from control.hotbar import build_hotbar_manager
    settings = {"hotbar": {"slot_roles": {1: "sword", 2: "axe", 3: "pickaxe",
                                          5: "blocks", 9: "food"},
                           "extra_food": ["space_ration"]}}
    hb = build_hotbar_manager(settings)      # no catalog -> suffix/curated only
    if hb.assigned_slot("axe") == 2 and hb.assigned_slot("food") == 9:
        ok("settings slot_roles parsed (axe=2, food=9)")
    else:
        bad("settings slot_roles parse wrong")
    # Bad config entries are ignored, not crash.
    hb2 = build_hotbar_manager({"hotbar": {"slot_roles": {0: "axe", 99: "food",
                                                          2: "not_a_role"}}})
    if hb2.assigned_slot("axe") is None and hb2.assigned_slot("food") is None:
        ok("invalid slot/role entries rejected safely")
    else:
        bad("invalid config not rejected")
    # The real project settings load + build cleanly.
    try:
        from config import load_settings
        real = build_hotbar_manager(load_settings())
        ok(f"real settings.yaml builds (slot_roles={dict(sorted(real.cfg.slot_roles.items()))})")
    except Exception as e:
        bad(f"real settings build failed: {e!r}")


def main() -> int:
    print("=" * 60)
    print(" Hotbar slot-role system — offline self-test")
    print("=" * 60)
    test_item_roles()
    test_hotbar_manager()
    test_builder()
    print("\n" + ("ALL HOTBAR TESTS PASSED" if not _fails
                  else f"{_fails} CHECK(S) FAILED"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
