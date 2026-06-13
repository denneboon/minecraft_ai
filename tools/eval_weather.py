#!/usr/bin/env python3
"""
Weather-detector validation harness — measures the REAL accuracy of the
nearest-centroid weather classifier (vision/weather.py) on a held-out split of
the command-labelled samples, so we can decide whether weather is trustworthy
enough to feed the recogniser as a TRAINING LABEL (see
WorldPerceptionConfig.trust_weather).

    python tools/eval_weather.py                  # held-out confusion + verdict
    python tools/eval_weather.py --seed 11 --min-conf 0.75

Offline + reproducible (uses the stored feature vectors in
data/training/weather_samples.json — no Minecraft needed). It rebuilds the
exact production centroids from the TRAIN split and runs the production
classifier on the held-out TEST split, then prints a confusion matrix, overall
accuracy, and — crucially — accuracy among only the CONFIDENT predictions
(conf >= --min-conf), which is what trust_weather actually gates on.

Verdict guidance: weather should stay UNTRUSTED unless confident-accuracy is
high (>= ~0.9) AND confident-coverage is non-trivial. A detector that's only
seen clear+rain in one biome/time will look fine here yet fail live — so also
collect diverse samples (tools/collect_weather.py across biomes + times) before
trusting it.
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

from vision.weather import (
    WeatherDetector, WeatherDetectorConfig, FEATURE_KEYS, _load_samples,
)


def _split(samples_by_state, frac, seed):
    rng = np.random.default_rng(seed)
    train, test = defaultdict(list), []
    for state, lst in samples_by_state.items():
        idx = rng.permutation(len(lst))
        cut = max(1, int(len(lst) * frac))
        train[state] = [lst[i] for i in idx[:cut]]
        for i in idx[cut:]:
            test.append((state, lst[i]))
    return train, test


def _fit_centroids(det, train):
    """Rebuild production centroids from a TRAIN subset (mirrors
    WeatherDetector.reload_training)."""
    usable = {s: v for s, v in train.items() if len(v) >= 1}
    pooled = np.array([det._vec(d) for v in usable.values() for d in v],
                      dtype=np.float32)
    det._feat_mean = pooled.mean(axis=0)
    det._feat_std = pooled.std(axis=0)
    det._feat_std[det._feat_std < 1e-6] = 1.0
    det._centroids = {}
    for state, samples in usable.items():
        mat = np.array([det._vec(d) for d in samples], dtype=np.float32)
        det._centroids[state] = ((mat - det._feat_mean) / det._feat_std).mean(axis=0)
    det._trained = True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--min-conf", type=float, default=0.75,
                    help="confidence threshold trust_weather would gate on")
    args = ap.parse_args(argv)

    print("=" * 64)
    print(" Weather detector — held-out validation")
    print("=" * 64)
    det = WeatherDetector()
    data = _load_samples(det._store_path)
    counts = {s: len(v) for s, v in data.items() if v}
    print(f"  labelled samples: {counts or '(none)'}")
    states = [s for s, v in data.items() if len(v) >= 4]
    if len(states) < 2:
        print("  NOT ENOUGH labelled data for a real test (need >=2 states, "
              ">=4 each). Collect with tools/collect_weather.py, then re-run.")
        print("  -> keep trust_weather OFF.")
        return 0

    data = {s: data[s] for s in states}
    train, test = _split(data, args.train_frac, args.seed)
    _fit_centroids(det, train)

    confusion = defaultdict(Counter)
    n = correct = 0
    conf_n = conf_correct = 0
    for true_state, feat in test:
        pred, conf = det._classify_trained(feat)
        confusion[true_state][pred] += 1
        n += 1
        correct += (pred == true_state)
        if conf >= args.min_conf:
            conf_n += 1
            conf_correct += (pred == true_state)

    acc = correct / n if n else 0.0
    print(f"\n  held-out: {n} samples over {len(states)} states "
          f"(train/test {args.train_frac:.0%})")
    print(f"  {'true':10} -> predictions")
    for s in states:
        row = ", ".join(f"{p}x{c}" for p, c in confusion[s].most_common())
        print(f"     {s:10} -> {row or '(none)'}")
    print(f"\n  overall accuracy:        {acc:.2f}  ({correct}/{n})")
    if conf_n:
        print(f"  confident (>= {args.min_conf:.2f}) acc: "
              f"{conf_correct/conf_n:.2f}  ({conf_correct}/{conf_n}; "
              f"covers {conf_n/n:.0%} of samples)")
    else:
        print(f"  no predictions reached conf >= {args.min_conf:.2f}")

    print("-" * 64)
    trustworthy = (conf_n and conf_correct / conf_n >= 0.9 and conf_n / n >= 0.5
                   and len(states) >= 2)
    if trustworthy:
        print("  VERDICT: confident-accuracy is high — weather is a candidate to "
              "TRUST. But verify across BIOMES + times first (this set may be "
              "narrow); then set vision.world.trust_weather: true.")
    else:
        print("  VERDICT: NOT trustworthy yet — keep trust_weather OFF. Collect "
              "more DIVERSE labelled data (tools/collect_weather.py across biomes "
              "+ day/night, incl. snow/thunder) and re-run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
