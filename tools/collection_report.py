#!/usr/bin/env python3
"""
Fast, no-train coverage report for the world sample store.

`diagnose_recognizer.py` trains a model to show where it FAILS (minutes);
this just reads the store metadata and shows, in seconds, what you HAVE and
what's MISSING — ideal for a quick progress check during a multi-machine
collection run, and for deciding what the curriculum should target next.

Reports: totals + rich-context coverage, per-block counts, the distribution of
weather / time-of-day / biome / face, and the thinnest (block x weather) and
(block x time) cells with a concrete "collect next" list.

    python tools/collection_report.py
    python tools/collection_report.py --min-cell 8 --top 25
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

from vision.world.sample_store import build_world_sample_store

_COND = ("weather", "time_of_day", "biome", "face")


def _val(md, key):
    if not md:
        return None
    v = md.get(key)
    return None if v is None else str(v).split(":")[-1]


def _dist(samples, key):
    c = Counter()
    for s in samples:
        v = _val(getattr(s, "metadata", None), key)
        if v is not None:
            c[v] += 1
    return c


def _bar(n, total, width=24):
    fill = int(round(width * n / total)) if total else 0
    return "#" * fill + "." * (width - fill)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-cell", type=int, default=8,
                    help="a (block x condition) cell below this is 'thin'")
    ap.add_argument("--top", type=int, default=20,
                    help="how many 'collect next' cells to list")
    args = ap.parse_args(argv)

    store = build_world_sample_store()
    alls = store.load_all(with_metadata=True)
    n = len(alls)
    print("=" * 66)
    print(" Sample store — coverage report")
    print("=" * 66)
    if not n:
        print("  store is empty — run train_curriculum."); return 0

    rich = [s for s in alls if _val(getattr(s, "metadata", None), "biome")]
    labelled_w = [s for s in alls if _val(getattr(s, "metadata", None), "weather")]
    labelled_t = [s for s in alls if _val(getattr(s, "metadata", None), "time_of_day")]
    blocks = Counter(s.block_id for s in alls)
    print(f"  samples: {n}  |  blocks: {len(blocks)}  |  rich-context "
          f"(biome present): {len(rich)} ({len(rich)/n:.0%})")
    print(f"  weather-labelled: {len(labelled_w)} ({len(labelled_w)/n:.0%})  |  "
          f"time-labelled: {len(labelled_t)} ({len(labelled_t)/n:.0%})")

    print(f"\n  ── per block ──")
    for b, c in blocks.most_common():
        print(f"     {b.split(':')[-1]:22} {c:5d}  {_bar(c, blocks.most_common(1)[0][1])}")

    for key in _COND:
        d = _dist(alls, key)
        if not d:
            continue
        tot = sum(d.values())
        print(f"\n  ── {key} ({tot} labelled) ──")
        for v, c in d.most_common():
            print(f"     {v:22} {c:5d}  {_bar(c, d.most_common(1)[0][1])}")

    # Gap matrices: (block x weather) and (block x time), thin cells first.
    for cond in ("weather", "time_of_day"):
        cells = defaultdict(int)
        bset, vset = set(), set()
        for s in alls:
            md = getattr(s, "metadata", None)
            v = _val(md, cond)
            if v is None:
                continue
            b = s.block_id.split(":")[-1]
            cells[(b, v)] += 1
            bset.add(b); vset.add(v)
        if not vset:
            continue
        thin = []
        for b in sorted(bset):
            for v in sorted(vset):
                cnt = cells.get((b, v), 0)
                if cnt < args.min_cell:
                    thin.append((cnt, f"{b}/{v}"))
        thin.sort()
        print(f"\n  ── collect-next: thinnest (block x {cond}) cells "
              f"(< {args.min_cell}) ──")
        if not thin:
            print("     (all cells well covered)")
        else:
            shown = thin[: args.top]
            for i in range(0, len(shown), 3):
                print("     " + "   ".join(f"{lbl}={cnt}"
                                           for cnt, lbl in shown[i:i + 3]))
            if len(thin) > args.top:
                print(f"     … and {len(thin) - args.top} more thin cells")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
