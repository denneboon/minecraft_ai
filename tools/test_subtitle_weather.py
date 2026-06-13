#!/usr/bin/env python3
"""
Offline tests for the subtitle weather reader + commanded environment labels.

Covers (NO Minecraft needed):
  1. The ``time_of_day`` context feature encodes/degrades correctly and is in
     the signature.
  2. SubtitleWeatherReader keyword + time-window logic: a caption -> the right
     state at high confidence; absence -> clear; a sighting persists for
     ``window_sec`` then expires. (OCR itself is stubbed — we test the logic.)
  3. Perception's commanded override (set_environment): a /weather or /time
     label becomes the trusted sample label with NO detection, and clears back
     to the live readers. Plus the trust rules per source (command + subtitle
     trusted; raw sky heuristic still gated by trust_weather).

Run: python tools/test_subtitle_weather.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _ok(msg): print(f"  [OK]  {msg}")
def _fail(msg): print(f"  [FAIL] {msg}")


def test_time_of_day_feature() -> bool:
    print("\n[1] time_of_day context feature")
    from vision.world.context_features import ContextFeatureSet
    s = ContextFeatureSet()
    ok_sig = "time_of_day:5" in s.signature
    (_ok if ok_sig else _fail)("time_of_day is in the feature signature")
    # present -> the right one-hot slot is set + presence flag 1
    v_night = s.encode({"time_of_day": "night"})
    v_missing = s.encode({})
    # locate the time_of_day chunk by re-deriving offsets
    off = 0
    tod_present_idx = None
    for f in s.features:
        if f.name == "time_of_day":
            tod_slice = v_night[off:off + f.dim]
            tod_present_idx = off + f.dim
            break
        off += f.dim + 1
    present_set = (tod_present_idx is not None
                   and v_night[tod_present_idx] == 1.0
                   and float(tod_slice.sum()) == 1.0)
    missing_off = v_missing[tod_present_idx] == 0.0 if tod_present_idx else False
    (_ok if present_set else _fail)("a commanded time_of_day encodes one-hot + present")
    (_ok if missing_off else _fail)("a missing time_of_day -> presence flag 0")
    return ok_sig and present_set and bool(missing_off)


def test_subtitle_reader_logic() -> bool:
    print("\n[2] SubtitleWeatherReader keyword + window logic")
    from vision.subtitle_weather import (SubtitleWeatherReader,
                                          SubtitleWeatherConfig)
    templates = {"A": (np.ones((8, 6), dtype=np.uint8) * 255)}
    cfg = SubtitleWeatherConfig(window_sec=90.0)
    r = SubtitleWeatherReader(templates=templates, config=cfg)
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    caption = {"text": ""}
    r._read_region_text = lambda f: caption["text"]   # stub the OCR

    caption["text"] = "rain falls"
    a = r.read(frame, now=0.0)
    caption["text"] = "thunder rumbles\nrain falls"
    b = r.read(frame, now=1.0)
    caption["text"] = ""
    c = r.read(frame, now=2.0)            # no caption now -> but rain seen @0..1
    d = r.read(frame, now=200.0)          # >window since any caption -> clear

    rain_ok = a.state == "rain" and a.source == "subtitle" and a.confidence >= 0.9
    thunder_ok = b.state == "thunder" and b.confidence >= 0.9
    window_hold = c.state == "thunder"    # thunder@1.0 still within 90s window
    expired = d.state == "clear" and d.confidence >= 0.8
    (_ok if rain_ok else _fail)("'rain falls' -> rain @ high confidence, subtitle src")
    (_ok if thunder_ok else _fail)("'thunder rumbles' -> thunder")
    (_ok if window_hold else _fail)("a sighting persists within window_sec")
    (_ok if expired else _fail)("after window_sec with no caption -> clear")
    return rain_ok and thunder_ok and window_hold and expired


def test_commanded_override() -> bool:
    print("\n[3] Commanded environment override (set_environment)")
    from vision.world.perception import WorldPerception, WorldPerceptionConfig
    from vision.world.map import WorldMap
    from vision.world.types import WeatherObservation
    cfg = WorldPerceptionConfig()           # trust_weather defaults False
    wp = WorldPerception(config=cfg, world_map=WorldMap())

    # Commanded weather/time -> trusted label with NO detection.
    wp.set_environment(weather="snow", time_of_day="night")
    cmd_w = wp._trusted_weather() == "snow"
    cmd_t = wp._time_of_day() == "night"
    (_ok if cmd_w else _fail)("commanded weather='snow' is the trusted label")
    (_ok if cmd_t else _fail)("commanded time_of_day='night' is the trusted label")

    # Subtitle source is trusted when confident, even with trust_weather OFF.
    wp.set_environment(clear=True)
    wp._last_weather = WeatherObservation(state="rain", confidence=0.95,
                                          sky_visible=1.0, source="subtitle")
    sub_ok = wp._trusted_weather() == "rain"
    (_ok if sub_ok else _fail)("a confident subtitle verdict is trusted (no gate)")

    # Raw sky heuristic stays gated by trust_weather (default OFF -> missing).
    wp._last_weather = WeatherObservation(state="rain", confidence=0.95,
                                          sky_visible=1.0, source="heuristic")
    heur_gated = wp._trusted_weather() is None
    (_ok if heur_gated else _fail)("an unvalidated sky-heuristic verdict is NOT a label")

    # clear=True drops both overrides.
    wp.set_environment(weather="rain", time_of_day="day")
    wp.set_environment(clear=True)
    cleared = wp._env_weather is None and wp._env_time_of_day is None
    (_ok if cleared else _fail)("clear=True returns to the live readers")
    return cmd_w and cmd_t and sub_ok and heur_gated and cleared


def main() -> int:
    results = [
        test_time_of_day_feature(),
        test_subtitle_reader_logic(),
        test_commanded_override(),
    ]
    n_pass = sum(1 for r in results if r)
    print(f"\n[subtitle_weather] {n_pass}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
