#!/usr/bin/env python3
"""
Turn recorded episodes (data/episodes/*.jsonl) into a flat, ML-ready
(observation -> action) dataset for imitation learning — the first concrete
brick of the rule-skill -> learned-policy transition the architecture was
built for.

Each usable tick (PLAYING + dispatched) becomes one row:
  observation : what the bot SAW   — view angles, the F3 targeted block
                (category + relative offset + distance), HUD, behaviour state
  action      : what the bot DID   — movement holds, look delta, interact,
                hotbar slot
Stdlib only (csv + json). Writes data/datasets/<name>.csv + a manifest.

    python tools/export_dataset.py                 # all episodes -> one csv
    python tools/export_dataset.py <episode.jsonl> # just one
    python tools/export_dataset.py --out mydata    # name the output
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EP_DIR = os.path.join(ROOT, "data", "episodes")
OUT_DIR = os.path.join(ROOT, "data", "datasets")

_LOG_SUF = ("_log", "_wood", "_stem", "_hyphae")
_INTERACT = {None: "none", "attack": "attack", "use_hold": "use_hold",
             "use_item": "use_item", "drop_item": "drop"}

OBS_COLS = ["o_yaw", "o_pitch", "o_look_cat", "o_look_dx", "o_look_dy",
            "o_look_dz", "o_look_dist", "o_health", "o_hunger", "o_state"]
ACT_COLS = ["a_forward", "a_back", "a_left", "a_right", "a_jump", "a_sprint",
            "a_sneak", "a_look_dx", "a_look_dy", "a_interact", "a_slot"]


def _block_cat(bid):
    if bid is None:
        return "none"
    if bid == "minecraft:air":
        return "air"
    if bid.endswith(_LOG_SUF):
        return "log"
    if bid.endswith("_leaves"):
        return "leaves"
    return "other"


def _row(rec):
    """One (obs, act) row from a tick record, or None if not usable."""
    if rec.get("scr") not in (None, "playing") or rec.get("gated"):
        return None
    p = rec.get("p")              # [x,y,z,yaw,pitch]
    look = rec.get("look")        # [block_id, [x,y,z]] | None
    hud = rec.get("hud") or [None, None]
    a = rec.get("a") or {}
    ag = rec.get("ag") or {}
    mv = a.get("mv") or {}
    ld = a.get("ld") or [0, 0]

    look_cat, ldx, ldy, ldz, ldist = "none", "", "", "", -1.0
    if look and look[1] is not None and p is not None:
        bx, by, bz = look[1]
        ldx, ldy, ldz = bx - p[0], by - p[1], bz - p[2]
        ldist = round(math.sqrt(ldx * ldx + ldy * ldy + ldz * ldz), 3)
        ldx, ldy, ldz = round(ldx, 2), round(ldy, 2), round(ldz, 2)
    if look:
        look_cat = _block_cat(look[0])

    obs = {
        "o_yaw": (p[3] if p else ""), "o_pitch": (p[4] if p else ""),
        "o_look_cat": look_cat, "o_look_dx": ldx, "o_look_dy": ldy,
        "o_look_dz": ldz, "o_look_dist": ldist,
        "o_health": (hud[0] if hud[0] is not None else ""),
        "o_hunger": (hud[1] if hud[1] is not None else ""),
        "o_state": ag.get("state", ""),
    }
    act = {
        "a_forward": int(bool(mv.get("forward"))),
        "a_back": int(bool(mv.get("backward"))),
        "a_left": int(bool(mv.get("left"))),
        "a_right": int(bool(mv.get("right"))),
        "a_jump": int(bool(mv.get("jump"))),
        "a_sprint": int(bool(mv.get("sprint"))),
        "a_sneak": int(bool(mv.get("sneak"))),
        "a_look_dx": int(ld[0]), "a_look_dy": int(ld[1]),
        "a_interact": _INTERACT.get(a.get("int"), a.get("int") or "none"),
        "a_slot": int(a.get("slot") or 0),
    }
    return obs, act


def export(files, out_name):
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, out_name + ".csv")
    n_rows = 0
    n_eps = 0
    interacts = Counter()
    states = Counter()
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["episode"] + OBS_COLS + ACT_COLS)
        for path in files:
            ep = os.path.basename(path)
            used = False
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if "k" not in rec:
                        continue
                    row = _row(rec)
                    if row is None:
                        continue
                    obs, act = row
                    w.writerow([ep] + [obs[c] for c in OBS_COLS]
                               + [act[c] for c in ACT_COLS])
                    n_rows += 1
                    used = True
                    interacts[act["a_interact"]] += 1
                    states[obs["o_state"] or "?"] += 1
            if used:
                n_eps += 1
    manifest = {
        "rows": n_rows, "episodes": n_eps,
        "obs_cols": OBS_COLS, "act_cols": ACT_COLS,
        "categoricals": {
            "o_look_cat": ["none", "air", "log", "leaves", "other"],
            "o_state": sorted(states),
            "a_interact": sorted(interacts),
        },
        "interact_dist": dict(interacts.most_common()),
        "state_dist": dict(states.most_common()),
        "csv": os.path.basename(csv_path),
    }
    with open(os.path.join(OUT_DIR, out_name + "_manifest.json"), "w",
              encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    return csv_path, manifest


def main(argv=None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    out_name = "dataset"
    if "--out" in argv:
        i = argv.index("--out")
        out_name = argv[i + 1]
        del argv[i:i + 2]
    if argv:
        files = argv
    else:
        files = sorted(glob.glob(os.path.join(EP_DIR, "*.jsonl")))
    if not files:
        print(f"No episodes in {EP_DIR}. Run an agent first.")
        return 1
    csv_path, m = export(files, out_name)
    print(f"Exported {m['rows']} rows from {m['episodes']} episode(s) -> {csv_path}")
    print(f"  interact labels: {m['interact_dist']}")
    print(f"  behaviour states: {m['state_dist']}")
    print(f"  manifest: {os.path.join(OUT_DIR, out_name + '_manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
