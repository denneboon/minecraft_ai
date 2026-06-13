#!/usr/bin/env python3
"""
Per-CONDITION diagnostic for the context-fusion recogniser.

`eval_recognizer.py` benchmarks the CNN/NN overall; this answers the questions
that matter for a CONTEXT-conditioned model trained from a roaming curriculum:

  * Where does it fail — by weather, time-of-day, biome, and viewed face?
    (A model that's 0.66 on oak_log overall might be 0.9 in clear/day and 0.3
    in rain/night — that tells you to collect oak-log-in-rain, not just "more
    oak_log".)
  * What's MISSING — which (block x weather) and (block x time) cells have few
    or zero TRAINING samples? Those gaps are exactly what the curriculum should
    target next, and they cap how well any condition can ever score.

Deterministic for a fixed store (seeded split + training), so it A/Bs cleanly.
Offline — no Minecraft.

    python tools/diagnose_recognizer.py
    python tools/diagnose_recognizer.py --min-per-block 10 --visual-width 48
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

from tools.eval_recognizer import _split, _short
from vision.world.sample_store import build_world_sample_store
from vision.world.context_features import ContextFeatureSet
from vision.world.context_fusion import (
    ContextFusionRecognizer, ContextFusionConfig, _TORCH_OK,
)

# Context keys we slice accuracy by + how to read them from a sample's metadata.
_COND_KEYS = ("weather", "time_of_day", "biome", "face")


def _cond(md, key):
    """Read a condition value from sample metadata, normalised to a short
    label; '(missing)' when absent so gaps are visible."""
    if not md:
        return "(missing)"
    v = md.get(key)
    if v is None:
        return "(missing)"
    return str(v).split(":")[-1]


def _slice_table(rows, title):
    print(f"\n  ── accuracy by {title} ──")
    print(f"     {title:18} {'acc':>6}  {'n':>5}")
    for val, (corr, tot) in sorted(rows.items(), key=lambda kv: -kv[1][1]):
        if tot:
            print(f"     {val:18} {corr/tot:6.2f}  {tot:5d}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-per-block", type=int, default=8)
    ap.add_argument("--visual-width", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gap-threshold", type=int, default=6,
                    help="flag (block x condition) train cells below this count")
    args = ap.parse_args(argv)

    print("=" * 68)
    print(" Context-fusion recogniser — per-condition diagnostic")
    print("=" * 68)
    if not _TORCH_OK:
        print("  PyTorch unavailable."); return 1

    store = build_world_sample_store()
    alls = store.load_all(with_metadata=True)
    train, test, classes = _split(alls, args.min_per_block, seed=args.seed)
    print(f"  store: {len(alls)} samples; train {len(train)} / test {len(test)} "
          f"over {len(classes)} blocks (>= {args.min_per_block} each)")
    if len(classes) < 2 or not test:
        print("  not enough data — collect more (train_curriculum)."); return 0

    fs = ContextFeatureSet()
    cfg = ContextFusionConfig(epochs=args.epochs, visual_width=args.visual_width)
    rec = ContextFusionRecognizer(store, config=cfg, feature_set=fs)
    print(f"  training fusion (width={args.visual_width}, epochs={args.epochs})…")
    rec.train_now(samples=train)
    print(f"  {rec.status()}")

    # Evaluate held-out, accumulating overall, per-block, per-condition.
    per_block = defaultdict(lambda: [0, 0])              # bid -> [correct, total]
    cond_acc = {k: defaultdict(lambda: [0, 0]) for k in _COND_KEYS}
    confusion = defaultdict(Counter)
    block_cond = defaultdict(lambda: [0, 0])             # (bid, weather) cell
    n_correct = 0
    for s in test:
        md = getattr(s, "metadata", None)
        guess, _ = rec.classify(s.rgb, metadata=md)
        ok = int(guess == s.block_id)
        n_correct += ok
        per_block[s.block_id][0] += ok; per_block[s.block_id][1] += 1
        confusion[s.block_id][_short(guess) if guess else "ABSTAIN"] += 1
        for k in _COND_KEYS:
            cell = cond_acc[k][_cond(md, k)]
            cell[0] += ok; cell[1] += 1
        bc = block_cond[(_short(s.block_id), _cond(md, "weather"))]
        bc[0] += ok; bc[1] += 1

    acc = n_correct / len(test)
    print(f"\n  OVERALL held-out accuracy: {acc:.3f}  ({n_correct}/{len(test)})")

    print(f"\n  ── per block (worst first) ──")
    print(f"     {'block':20} {'acc':>6}  {'n':>4}  confused-with")
    for bid, (c, t) in sorted(per_block.items(), key=lambda kv: kv[1][0]/max(1, kv[1][1])):
        others = ", ".join(f"{g}×{n}" for g, n in confusion[bid].most_common()
                           if g != _short(bid))[:46]
        print(f"     {_short(bid):20} {c/max(1,t):6.2f}  {t:4d}  {others}")

    for k in _COND_KEYS:
        _slice_table(cond_acc[k], k)

    # Coverage gaps from the TRAIN split — what the curriculum should target.
    train_cells = defaultdict(int)
    seen_blocks, seen_weather = set(), set()
    for s in train:
        md = getattr(s, "metadata", None)
        b, w = _short(s.block_id), _cond(md, "weather")
        train_cells[(b, w)] += 1
        seen_blocks.add(b); seen_weather.add(w)
    print(f"\n  ── thin/empty TRAIN cells (block x weather, < {args.gap_threshold}) ──")
    gaps = []
    for b in sorted(seen_blocks):
        for w in sorted(seen_weather):
            n = train_cells.get((b, w), 0)
            if n < args.gap_threshold:
                gaps.append(f"{b}/{w}={n}")
    if gaps:
        for i in range(0, len(gaps), 4):
            print("     " + "   ".join(gaps[i:i + 4]))
    else:
        print("     (none — every block x weather cell is well covered)")
    print("=" * 68)
    print("  -> collect the thin cells (train_curriculum reaches biomes by /weather"
          " + roaming); re-run to confirm the gap closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
