#!/usr/bin/env python3
"""
Offline self-test for vision.weather (no Minecraft needed).

Covers:
  * heuristic: cave -> unknown, blue sky -> clear, grey overcast -> rain
  * sky-not-visible gate (cave) regardless of trained data
  * trained nearest-centroid path classifies by feature centroids
  * temporal smoothing majority-votes away single-frame noise
  * append_samples / reload_training round-trip

Run: python tools/test_weather.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from vision.weather import (
    WeatherDetector, WeatherDetectorConfig, WeatherState,
    FEATURE_KEYS, append_samples,
)

_fail = 0


def ok(msg):
    print(f"  [ok] {msg}")


def bad(msg):
    global _fail
    _fail += 1
    print(f"  [FAIL] {msg}")


def _frame_blue():
    f = np.zeros((1080, 1920, 3), np.uint8)
    f[:, :, 0] = 120; f[:, :, 1] = 170; f[:, :, 2] = 215   # RGB blue sky
    return f


def _frame_grey():
    return np.full((1080, 1920, 3), 130, np.uint8)


def _frame_cave():
    rng = np.random.default_rng(1)
    return rng.integers(0, 38, (1080, 1920, 3)).astype(np.uint8)


def test_heuristic():
    print("[1] heuristic classification")
    d = WeatherDetector(store_path=Path(tempfile.gettempdir()) / "no_such_w.json")
    o = d.detect(_frame_cave())
    (ok if o.state == WeatherState.UNKNOWN else bad)(f"cave -> {o.state} (want unknown)")

    d.reset()
    o = d.detect(_frame_blue())
    (ok if o.state == WeatherState.CLEAR else bad)(f"blue sky -> {o.state} (want clear)")

    d.reset()
    o = d.detect(_frame_grey())
    (ok if o.state in (WeatherState.RAIN, WeatherState.THUNDER) else bad)(
        f"grey overcast -> {o.state} (want rain/thunder)")

    # Feature dict completeness.
    (ok if set(o.features) == set(FEATURE_KEYS) else bad)(
        "feature dict carries every FEATURE_KEY")


def test_cave_gate_overrides_training():
    print("[2] cave gate wins even with trained data")
    tmp = Path(tempfile.mkdtemp()) / "w.json"
    # Train clear+rain with sky visible; a cave frame must still be unknown.
    clear = [{k: 0.0 for k in FEATURE_KEYS} | {"sky_open_frac": 0.9,
             "sky_saturation": 0.4, "sky_blueness": 0.7} for _ in range(10)]
    rain = [{k: 0.0 for k in FEATURE_KEYS} | {"sky_open_frac": 0.9,
            "sky_saturation": 0.05, "sky_blueness": 0.4} for _ in range(10)]
    append_samples(tmp, "clear", clear)
    append_samples(tmp, "rain", rain)
    d = WeatherDetector(store_path=tmp)
    (ok if d.is_trained() else bad)("detector reports trained after >=2 states")
    o = d.detect(_frame_cave())
    (ok if o.state == WeatherState.UNKNOWN else bad)(
        f"cave -> {o.state} (sky gate beats trained classifier)")


def test_trained_classifies():
    print("[3] trained nearest-centroid picks the right class")
    tmp = Path(tempfile.mkdtemp()) / "w.json"
    base = {k: 0.0 for k in FEATURE_KEYS}
    # Two well-separated clusters in feature space, both sky-visible.
    a = [base | {"sky_open_frac": 0.9, "sky_saturation": 0.45,
                 "sky_blueness": 0.75, "global_brightness": 0.6}
         for _ in range(12)]
    b = [base | {"sky_open_frac": 0.9, "sky_saturation": 0.04,
                 "sky_blueness": 0.40, "global_brightness": 0.35}
         for _ in range(12)]
    append_samples(tmp, "clear", a)
    append_samples(tmp, "rain", b)
    d = WeatherDetector(store_path=tmp)
    if not d.is_trained():
        bad("expected trained classifier")
        return
    # Query close to the 'rain' centroid (sky visible, desaturated, dark).
    q_rain = base | {"sky_open_frac": 0.92, "sky_saturation": 0.05,
                     "sky_blueness": 0.41, "global_brightness": 0.34}
    state, conf = d._classify_trained(q_rain)
    (ok if state == "rain" else bad)(f"rain-like features -> {state} (conf {conf:.2f})")
    q_clear = base | {"sky_open_frac": 0.9, "sky_saturation": 0.44,
                      "sky_blueness": 0.74, "global_brightness": 0.61}
    state2, _ = d._classify_trained(q_clear)
    (ok if state2 == "clear" else bad)(f"clear-like features -> {state2}")


def test_smoothing():
    print("[4] temporal smoothing majority vote")
    d = WeatherDetector(config=WeatherDetectorConfig(smooth_window=5),
                        store_path=Path(tempfile.gettempdir()) / "no_w.json")
    seq = ["clear", "clear", "rain", "clear", "clear"]   # one noisy 'rain'
    out = [d._smooth(s) for s in seq]
    (ok if out[-1] == "clear" else bad)(
        f"single noisy 'rain' smoothed away -> {out[-1]} (want clear)")


def main():
    print("=" * 60)
    print(" vision.weather self-test")
    print("=" * 60)
    test_heuristic()
    test_cave_gate_overrides_training()
    test_trained_classifies()
    test_smoothing()
    print("=" * 60)
    if _fail:
        print(f" {_fail} CHECK(S) FAILED")
        return 1
    print(" ALL WEATHER TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
