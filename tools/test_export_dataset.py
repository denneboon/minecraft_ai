#!/usr/bin/env python3
"""Offline self-test for tools/export_dataset.py — episodes -> flat
(observation, action) rows, with correct filtering + encoding."""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

import tools.export_dataset as ed

_fails = 0
def ok(m): print(f"  [OK]  {m}")
def bad(m):
    global _fails; _fails += 1; print(f"  [FAIL] {m}")


def main() -> int:
    print("=" * 56); print(" export_dataset — offline self-test"); print("=" * 56)
    tmp = tempfile.mkdtemp(prefix="ds_test_")
    ep = os.path.join(tmp, "treechop_t.jsonl")
    recs = [
        {"type": "meta", "agent": "treechop", "stamp": "t", "schema": 1},
        # usable: chopping a log under the crosshair
        {"k": 0, "t": 0.1, "p": [-12.0, 65.0, 45.0, 90.0, 3.0],
         "look": ["minecraft:birch_log", [-17, 66, 45]], "hud": [1.0, 0.8],
         "scr": "playing", "a": {"int": "attack", "slot": 2},
         "ag": {"state": "chop", "logs": 0}},
        # usable: walking, no target block
        {"k": 1, "t": 0.2, "p": [-12.0, 65.0, 45.0, 90.0, 0.0], "scr": "playing",
         "a": {"mv": {"forward": True, "sprint": True}, "ld": [7, 0]},
         "ag": {"state": "approach"}},
        # NOT usable: gated
        {"k": 2, "t": 0.3, "p": [-12.0, 65.0, 45.0, 0, 0], "scr": "playing",
         "gated": True, "a": {"int": "attack"}},
        # NOT usable: not playing
        {"k": 3, "t": 0.4, "scr": "menu", "a": {}},
        {"type": "summary", "ticks": 4},
    ]
    with open(ep, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    ed.OUT_DIR = os.path.join(tmp, "datasets")
    csv_path, m = ed.export([ep], "t")

    (ok if m["rows"] == 2 else bad)(f"only PLAYING+dispatched rows kept ({m['rows']}/2)")
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    (ok if len(rows) == 2 else bad)(f"csv has 2 data rows ({len(rows)})")

    r0 = rows[0]
    (ok if r0["o_look_cat"] == "log" else bad)(f"birch_log -> look_cat 'log' ({r0['o_look_cat']})")
    (ok if r0["a_interact"] == "attack" and r0["a_slot"] == "2" else bad)(
        f"action attack + slot 2 ({r0['a_interact']}, {r0['a_slot']})")
    (ok if r0["o_look_dx"] == "-5.0" and r0["o_look_dz"] == "0.0" else bad)(
        f"relative offset to looked-at voxel ({r0['o_look_dx']},{r0['o_look_dz']})")
    (ok if float(r0["o_look_dist"]) > 0 else bad)(f"look_dist computed ({r0['o_look_dist']})")

    r1 = rows[1]
    (ok if r1["a_forward"] == "1" and r1["a_sprint"] == "1" else bad)("movement holds encoded")
    (ok if r1["a_look_dx"] == "7" else bad)(f"action look delta encoded ({r1['a_look_dx']})")
    (ok if r1["o_look_cat"] == "none" else bad)("no targeted block -> look_cat 'none'")

    (ok if m["interact_dist"].get("attack") == 1 and "chop" in m["state_dist"] else bad)(
        f"manifest distributions ({m['interact_dist']}, {m['state_dist']})")
    (ok if os.path.exists(os.path.join(ed.OUT_DIR, "t_manifest.json")) else bad)(
        "manifest written")

    print("\n" + ("ALL EXPORT TESTS PASSED" if not _fails
                  else f"{_fails} CHECK(S) FAILED"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
