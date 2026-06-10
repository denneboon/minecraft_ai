#!/usr/bin/env python3
"""
Reproducible quality benchmark for the world block recogniser.

Unlike ``tools/test_cnn_recognizer.py`` (a pass/fail smoke test), this is a
DIAGNOSTIC: it trains the CNN deterministically on a seeded train split of
the live F3-labelled sample store, evaluates on the held-out split, and
prints a per-class accuracy table + a confusion matrix for BOTH the CNN
and the raw-pixel NN, clean and augmented.

Why it exists: you can't make "the best model" without measuring it. The
store grows every self-teaching lesson, so absolute numbers move as data
accrues — but for a FIXED store this report is byte-reproducible
(training is seeded), so you can A/B a code change against the same store
and see exactly which blocks improved or regressed, and what they get
confused with.

    python tools/eval_recognizer.py                 # min 6 samples/block
    python tools/eval_recognizer.py --min-per-block 10 --augment
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

import numpy as np

from vision.world.sample_store import build_world_sample_store
from vision.world.cnn_recognizer import (
    CNNBlockRecognizer, CNNBlockRecognizerConfig, _TORCH_OK, _augment,
)
from vision.world.sample_recognizer import (
    SampleBlockRecognizer, SampleBlockRecognizerConfig,
)


def _split(samples, min_per_block, train_frac=0.7, seed=7):
    by = defaultdict(list)
    for s in samples:
        by[s.block_id].append(s)
    rng = np.random.default_rng(seed)
    train, test, classes = [], [], []
    for bid, lst in sorted(by.items()):
        if len(lst) < min_per_block:
            continue
        classes.append(bid)
        idx = rng.permutation(len(lst))
        cut = max(1, int(len(lst) * train_frac))
        train += [lst[i] for i in idx[:cut]]
        test += [lst[i] for i in idx[cut:]]
    return train, test, classes


def _short(bid):
    return bid.split(":")[-1] if bid else "ABSTAIN"


def _evaluate(rec, test, classes, augment):
    """Return (per_class_correct, per_class_total, confusion, acc, cov)."""
    rng = np.random.default_rng(99)
    correct = defaultdict(int)
    total = defaultdict(int)
    confusion = defaultdict(Counter)     # truth -> Counter(guess|ABSTAIN)
    n_correct = n_decided = 0
    for s in test:
        patch = _augment(s.rgb, rng) if augment else s.rgb
        guess, conf = rec.classify(patch)
        total[s.block_id] += 1
        confusion[s.block_id][guess or "ABSTAIN"] += 1
        if guess is not None:
            n_decided += 1
            if guess == s.block_id:
                n_correct += 1
                correct[s.block_id] += 1
    n = len(test)
    acc = n_correct / n_decided if n_decided else 0.0
    cov = n_decided / n if n else 0.0
    return correct, total, confusion, acc, cov


def _print_report(name, rec, test, classes, augment):
    correct, total, confusion, acc, cov = _evaluate(rec, test, classes, augment)
    tag = "augmented" if augment else "clean"
    print(f"\n  ── {name} [{tag}] ── decided-acc={acc:.2f} coverage={cov:.2f}")
    print(f"     {'block':22} {'acc':>6}  confused-with (count)")
    for bid in classes:
        t = total.get(bid, 0)
        if not t:
            continue
        a = correct.get(bid, 0) / t
        others = ", ".join(f"{_short(g)}×{c}"
                           for g, c in confusion[bid].most_common()
                           if g != bid)
        print(f"     {_short(bid):22} {a:6.2f}  ({correct.get(bid,0)}/{t})"
              f"  {others}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-per-block", type=int, default=6,
                    help="min samples for a block to enter the benchmark")
    ap.add_argument("--augment", action="store_true",
                    help="also show the augmented (lighting/biome/angle) eval")
    args = ap.parse_args(argv)

    print("=" * 64)
    print(" World block recogniser — quality benchmark (deterministic)")
    print("=" * 64)
    if not _TORCH_OK:
        print("  PyTorch unavailable — CNN cannot be evaluated.")
        return 1

    store = build_world_sample_store()
    alls = store.load_all()
    train, test, classes = _split(alls, args.min_per_block)
    print(f"  store: {len(alls)} samples; split -> train {len(train)}, "
          f"test {len(test)} over {len(classes)} blocks "
          f"(>= {args.min_per_block} samples each)")
    if len(classes) < 2 or not test:
        print("  not enough data for a benchmark "
              "(collect more via tools/learn_world_live.py).")
        return 0

    # Raw-pixel NN (train-split only).
    nn = SampleBlockRecognizer(store, config=SampleBlockRecognizerConfig())
    nn._samples = list(train); nn._rebuild_tensor()

    # CNN trained from scratch on the train split (deterministic).
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), "_eval_block_cnn.pt")
    if os.path.exists(tmp):
        os.remove(tmp)
    cnn = CNNBlockRecognizer(
        store, config=CNNBlockRecognizerConfig(epochs=60, model_path=tmp),
        auto_train=False)
    cnn._model = None; cnn._trained = False; cnn._texture_proto = {}
    cnn._samples = list(train)
    cnn.train_now()
    print(f"  {cnn.status()}")

    for aug in ([False, True] if args.augment else [False]):
        _print_report("raw-pixel NN", nn, test, classes, aug)
        _print_report("CNN", cnn, test, classes, aug)
    print("=" * 64)
    print("  (deterministic for a fixed store; re-run after a code change "
          "to A/B per-block.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
