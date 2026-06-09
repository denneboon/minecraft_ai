#!/usr/bin/env python3
"""
Offline accuracy test for the self-teaching CNN block recogniser.

Validates that the learned CNN embedding (vision/world/cnn_recognizer.py)
beats the raw-pixel L1 nearest-neighbour (vision/world/sample_recognizer.py)
on HELD-OUT real captures — especially under photometric/geometric jitter
that simulates the day/night, biome-tint, distance and angle variation the
recogniser meets in-world (which is exactly where raw-pixel matching
fails). No Minecraft needed — trains and evaluates on the existing
data/training/world_samples store.

Run: python tools/test_cnn_recognizer.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from vision.world.sample_store import build_world_sample_store
from vision.world.sample_recognizer import (
    SampleBlockRecognizer, SampleBlockRecognizerConfig,
)
from vision.world.cnn_recognizer import (
    CNNBlockRecognizer, CNNBlockRecognizerConfig, _TORCH_OK, _augment,
)

_fail = 0


def ok(msg):
    print(f"  [ok] {msg}")


def bad(msg):
    global _fail
    _fail += 1
    print(f"  [FAIL] {msg}")


def _split(samples, min_per_block=10, train_frac=0.7, seed=7):
    """Per-block train/test split over blocks with enough samples."""
    by = defaultdict(list)
    for s in samples:
        by[s.block_id].append(s)
    rng = np.random.default_rng(seed)
    train, test = [], []
    for bid, lst in by.items():
        if len(lst) < min_per_block:
            continue
        idx = rng.permutation(len(lst))
        k = int(round(len(lst) * train_frac))
        train += [lst[i] for i in idx[:k]]
        test += [lst[i] for i in idx[k:]]
    return train, test, sorted({s.block_id for s in train})


def _eval(recognizer, test, augment=False, seed=99):
    rng = np.random.default_rng(seed)
    correct = abstain = wrong = 0
    for s in test:
        patch = _augment(s.rgb, rng) if augment else s.rgb
        pred, conf = recognizer.classify(patch)
        if pred is None:
            abstain += 1
        elif pred == s.block_id:
            correct += 1
        else:
            wrong += 1
    n = len(test)
    # Accuracy over decisions made (abstain isn't wrong — perception falls
    # back), plus coverage.
    decided = correct + wrong
    acc = correct / decided if decided else 0.0
    cov = decided / n if n else 0.0
    return acc, cov, correct, wrong, abstain, n


def main():
    print("=" * 64)
    print(" CNN block recogniser — offline accuracy vs raw-pixel NN")
    print("=" * 64)
    if not _TORCH_OK:
        print("  [skip] PyTorch not available — CNN recogniser inactive")
        return 0

    store = build_world_sample_store()
    alls = store.load_all()
    train, test, classes = _split(alls)
    print(f"  data: {len(alls)} samples; split -> train {len(train)}, "
          f"test {len(test)}, over {len(classes)} blocks (>=10 samples each)")
    if len(classes) < 3 or not test:
        print("  [skip] not enough world_samples on disk for a split "
              "(data/training/ is gitignored / local-only)")
        return 0

    # --- Raw-pixel NN baseline (trained on the train split) ---
    nn = SampleBlockRecognizer(store, config=SampleBlockRecognizerConfig())
    nn._samples = list(train); nn._rebuild_tensor()      # restrict to train

    # --- CNN (trained on the same train split) ---
    # Use a throwaway model path so the test never clobbers the runtime
    # model (data/calibration/block_cnn.pt). Start from scratch (ignore any
    # pre-trained checkpoint) so the accuracy numbers reflect this split.
    import tempfile, os
    tmp_model = os.path.join(tempfile.gettempdir(), "_test_block_cnn.pt")
    if os.path.exists(tmp_model):
        os.remove(tmp_model)
    cnn = CNNBlockRecognizer(
        store, config=CNNBlockRecognizerConfig(epochs=50, model_path=tmp_model),
        auto_train=False)
    cnn._model = None; cnn._trained = False; cnn._texture_proto = {}
    cnn._samples = list(train)
    print("  training CNN (background-style, synchronous here)…")
    cnn.train_now()
    print(f"  {cnn.status()}")
    (ok if cnn.available() else bad)("CNN trained and available")

    # --- Evaluate both, clean and augmented ---
    for label, aug in (("clean", False), ("augmented", True)):
        nn_acc, nn_cov, *_ = _eval(nn, test, augment=aug)
        cnn_acc, cnn_cov, c, w, ab, n = _eval(cnn, test, augment=aug)
        print(f"\n  [{label}] held-out test ({n} patches)")
        print(f"    raw-pixel NN : acc={nn_acc:.2f}  coverage={nn_cov:.2f}")
        print(f"    CNN          : acc={cnn_acc:.2f}  coverage={cnn_cov:.2f}  "
              f"(correct={c} wrong={w} abstain={ab})")
        # The CNN should be at least competitive clean, and clearly better
        # (acc and/or coverage) under augmentation.
        if aug:
            better = (cnn_acc >= nn_acc - 0.02) and (
                cnn_cov > nn_cov + 0.05 or cnn_acc > nn_acc + 0.05)
            (ok if better else bad)(
                f"CNN beats raw-NN under augmentation "
                f"(acc {cnn_acc:.2f} vs {nn_acc:.2f}, cov {cnn_cov:.2f} vs {nn_cov:.2f})")
        else:
            (ok if cnn_acc >= 0.7 else bad)(
                f"CNN clean accuracy reasonable ({cnn_acc:.2f} >= 0.70)")

    print("=" * 64)
    if _fail:
        print(f" {_fail} CHECK(S) FAILED")
        return 1
    print(" ALL CNN RECOGNISER TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
