#!/usr/bin/env python3
"""
Scrub the untrustworthy `weather` label out of the world-sample sidecars.

The weather DETECTOR is a heuristic that mislabels (e.g. it tagged a clear
world ~64% "rain"), and with `doWeatherCycle` off rain is impossible anyway —
so any stored `weather` value is noise that would poison the fusion model.
This sets every sidecar's `weather` to null ("missing"), which is honest:
the fusion feature then ignores it (presence flag 0) instead of learning a
wrong categorical.

Idempotent + safe (only rewrites sidecars that still carry a weather value).
Run it on ANY machine whose sample store predates the weather fix — in
particular, `sync.py import-samples` MERGES by filename, so re-importing does
NOT overwrite an already-present sidecar; run this to clean it in place.

    python tools/scrub_weather.py            # scrub the default world store
    python tools/scrub_weather.py --dry-run  # report how many would change
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_STORE = ROOT / "data" / "training" / "world_samples"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=str, default=str(DEFAULT_STORE),
                    help="sample-store root (default: data/training/world_samples)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.store)
    if not root.is_dir():
        print(f"[scrub] no store at {root}"); return 1
    sidecars = list(root.rglob("*.json"))
    sidecars = [p for p in sidecars if p.name != "_manifest.json"]
    changed = 0
    for p in sidecars:
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if m.get("weather") is not None:
            changed += 1
            if not args.dry_run:
                m["weather"] = None
                m["weather_scrubbed"] = True
                p.write_text(json.dumps(m, sort_keys=True), encoding="utf-8")
    verb = "would scrub" if args.dry_run else "scrubbed"
    print(f"[scrub] {verb} weather from {changed}/{len(sidecars)} sidecars "
          f"under {root}")
    if changed and not args.dry_run:
        print("[scrub] weather is now 'missing' for training — honest. "
              "(trust_weather stays off until the detector is validated.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
