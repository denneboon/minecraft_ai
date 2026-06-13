#!/usr/bin/env python3
"""
Train the context-FUSION block recogniser (vision/world/context_fusion.py) —
visual patch + as many context inputs as the sidecars carry (face, biome,
light, weather, neighbours, distance, ...). Built to run on a big PC with the
rich data the new perception pipeline collects.

    python tools/train_fusion_model.py                  # train + held-out eval
    python tools/train_fusion_model.py --epochs 120 --visual-width 48
    python tools/train_fusion_model.py --ab             # also train visual-only
                                                        # to MEASURE the context lift
    python tools/train_fusion_model.py --out data/calibration/block_fusion.pt

Deterministic split (seeded) so a code/feature change is A/B-comparable, just
like tools/eval_recognizer.py. Reports per-block accuracy + how many test
samples actually had rich context (so the numbers are honest about coverage).
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

import numpy as np
from collections import Counter, defaultdict

from tools.eval_recognizer import _split            # reuse the seeded split
from vision.world.sample_store import build_world_sample_store
from vision.world.context_features import ContextFeatureSet
from vision.world.context_fusion import (
    ContextFusionRecognizer, ContextFusionConfig, _TORCH_OK,
)


def _evaluate(rec, test, *, use_context: bool):
    correct = defaultdict(int); total = defaultdict(int)
    n_ctx = 0
    for s in test:
        md = getattr(s, "metadata", None)
        if md:
            n_ctx += 1
        guess, _ = rec.classify(s.rgb, metadata=(md if use_context else None))
        total[s.block_id] += 1
        if guess == s.block_id:
            correct[s.block_id] += 1
    n = len(test)
    acc = sum(correct.values()) / n if n else 0.0
    return acc, correct, total, n_ctx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--visual-width", type=int, default=32)
    ap.add_argument("--embed-dim", type=int, default=96)
    ap.add_argument("--min-per-block", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default=None,
                    help="model output path (default: data/calibration/block_fusion.pt)")
    ap.add_argument("--ab", action="store_true",
                    help="also evaluate the SAME model with context zeroed, to "
                         "measure how much the context inputs add")
    args = ap.parse_args(argv)

    print("=" * 64)
    print(" Context-fusion block recogniser — train + held-out eval")
    print("=" * 64)
    if not _TORCH_OK:
        print(" PyTorch unavailable — cannot train."); return 1

    store = build_world_sample_store()
    alls = store.load_all(with_metadata=True)
    n_rich = sum(1 for s in alls if s.metadata)
    train, test, classes = _split(alls, args.min_per_block, seed=args.seed)
    fs = ContextFeatureSet()
    print(f"  store: {len(alls)} samples ({n_rich} with rich context); "
          f"split -> train {len(train)}, test {len(test)} over {len(classes)} blocks")
    print(f"  context inputs ({len(fs.names())}): {', '.join(fs.names())}")
    print(f"  context vector width: {fs.total_dim}")
    if len(classes) < 2 or not test:
        print("  not enough data (collect more via train_overnight)."); return 0

    cfg = ContextFusionConfig(
        epochs=args.epochs, visual_width=args.visual_width, embed_dim=args.embed_dim,
        model_path=args.out or ContextFusionConfig().model_path)
    rec = ContextFusionRecognizer(store, config=cfg, feature_set=fs)
    print(f"  training (epochs={args.epochs}, visual_width={args.visual_width}, "
          f"embed_dim={args.embed_dim})…")
    rec.train_now(samples=train)
    print(f"  {rec.status()}")

    acc, correct, total, n_ctx = _evaluate(rec, test, use_context=True)
    print(f"\n  ── FUSION (with context) ── acc={acc:.3f}  "
          f"({n_ctx}/{len(test)} test samples had rich context)")
    for bid in classes:
        t = total.get(bid, 0)
        if t:
            print(f"     {bid.split(':')[-1]:20} {correct.get(bid,0)/t:5.2f}  "
                  f"({correct.get(bid,0)}/{t})")

    if args.ab:
        acc0, _, _, _ = _evaluate(rec, test, use_context=False)
        lift = acc - acc0
        print(f"\n  ── A/B ── visual-only acc={acc0:.3f} | with-context acc={acc:.3f} "
              f"| context lift = {lift:+.3f}")
        print("  (lift grows as more samples carry rich context — keep collecting.)")

    print("=" * 64)
    print(f"  saved -> {cfg.model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
