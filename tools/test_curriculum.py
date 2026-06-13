#!/usr/bin/env python3
"""
Offline tests for the curriculum trainer's safety/labelling LOGIC (no MC).

Covers the pure helpers that decide what is safe to sample + how a commanded
weather is labelled, since those are the corruption-prevention guarantees:
  * biome_precip — coarse precip class (for snow labelling / dry-biome skip)
  * is_corrupted_view — underwater/lava global colour-cast detector
  * resolve_label — only trust a rain/thunder label when actually confirmed

Run: python tools/test_curriculum.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.train_curriculum import (
    biome_precip, is_corrupted_view, resolve_label, station_plan,
)


def _ok(m): print(f"  [OK]  {m}")
def _fail(m): print(f"  [FAIL] {m}")


def _frame(rgb, h=120, w=200):
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[:] = rgb
    return a


def test_biome_precip() -> bool:
    print("\n[1] biome_precip")
    cases = {
        "minecraft:snowy_plains": "snow", "minecraft:jagged_peaks": "snow",
        "minecraft:frozen_river": "snow", "minecraft:grove": "snow",
        "minecraft:desert": "none", "minecraft:savanna": "none",
        "minecraft:badlands": "none", "minecraft:the_end": "none",
        "minecraft:plains": "rain", "minecraft:dark_forest": "rain",
        "minecraft:jungle": "rain", "minecraft:swamp": "rain",
    }
    ok = True
    for b, want in cases.items():
        got = biome_precip(b)
        if got != want:
            ok = False; _fail(f"{b} -> {got} (want {want})")
    none_ok = biome_precip(None) is None
    (_ok if ok else _fail)("biome ids map to the right precip class")
    (_ok if none_ok else _fail)("unknown/None biome -> None")
    return ok and none_ok


def test_corrupted_view() -> bool:
    print("\n[2] is_corrupted_view (submerged / lava cast)")
    underwater = is_corrupted_view(_frame((40, 75, 140)))
    lava = is_corrupted_view(_frame((185, 95, 50)))
    normal = is_corrupted_view(_frame((95, 115, 80)))     # grassy/brown scene
    # A real outdoor view: bright blue sky fills the UPPER frame, terrain the
    # lower. The detector samples only the lower-centre band, so this must NOT
    # trip (the bug that made it re-teleport forever).
    sky = _frame((70, 130, 70))                           # green terrain
    sky[: int(sky.shape[0] * 0.55)] = (90, 140, 215)      # bright sky on top
    sky_ok = not is_corrupted_view(sky)
    (_ok if underwater else _fail)("underwater blue cast -> corrupted")
    (_ok if lava else _fail)("lava orange cast -> corrupted")
    (_ok if not normal else _fail)("a normal terrain frame -> NOT corrupted")
    (_ok if sky_ok else _fail)("bright sky filling the upper frame -> NOT corrupted")
    return underwater and lava and (not normal) and sky_ok


def test_resolve_label() -> bool:
    print("\n[3] resolve_label (only trust confirmed precipitation)")
    checks = [
        ("clear", "clear", "minecraft:desert", "clear"),
        ("rain", "rain", "minecraft:plains", "rain"),
        # rain biome + NO subtitle (Show Subtitles off) must STILL label rain —
        # the biome is enough; this is the bug that skipped all rain before.
        ("rain", "clear", "minecraft:plains", "rain"),
        ("rain", "clear", "minecraft:snowy_plains", "snow"),
        ("rain", "clear", "minecraft:desert", None),
        ("rain", "clear", None, None),                 # unknown + no caption -> skip
        ("rain", "rain", None, "rain"),                # unknown biome, caption confirms
        ("thunder", "clear", "minecraft:plains", "thunder"),
        ("thunder", "thunder", "minecraft:plains", "thunder"),
        ("thunder", "clear", "minecraft:snowy_plains", "snow"),
        ("thunder", "clear", "minecraft:desert", None),
    ]
    ok = True
    for wcmd, sub, biome, want in checks:
        got = resolve_label(wcmd, sub, biome)
        tag = f"{wcmd}+sub={sub}+{str(biome).split(':')[-1]} -> {got}"
        if got == want:
            _ok(tag)
        else:
            ok = False; _fail(tag + f" (want {want})")
    plan_ok = (len(station_plan()) >= 3
               and all(len(t) == 2 for t in station_plan()))
    (_ok if plan_ok else _fail)("station_plan is a list of (weather,time)")
    return ok and plan_ok


def main() -> int:
    results = [test_biome_precip(), test_corrupted_view(), test_resolve_label()]
    n = sum(1 for r in results if r)
    print(f"\n[curriculum] {n}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
