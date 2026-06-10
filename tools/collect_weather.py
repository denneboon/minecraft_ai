#!/usr/bin/env python3
"""
Collect command-labelled weather training samples.

Drives Minecraft's ``/weather`` command via chat, waits for the weather
to settle, captures frames, extracts ``vision.weather`` feature vectors,
and appends them to the on-disk weather sample store. After enough
samples per state the :class:`vision.weather.WeatherDetector` switches
from its heuristic to the trained nearest-centroid classifier
automatically.

Requirements
------------
* Minecraft running, focused, with cheats / commands ENABLED.
* For ``snow``: stand in a cold (snowy) biome and collect under the
  ``rain`` command — MC renders precipitation as snow there. Use
  ``--label snow --command rain``.

Examples
--------
    # Cycle clear / rain / thunder (default), 20 frames each:
    python tools/collect_weather.py

    # Snow: stand in a snowy biome first, then:
    python tools/collect_weather.py --label snow --command rain --frames 24

    # Just clear, more frames:
    python tools/collect_weather.py --states clear --frames 40
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as M
from utils.focus import activate_minecraft, _find_minecraft_hwnd
from vision.capture import Capture, CaptureConfig
from vision.weather import (
    append_samples, build_weather_detector, default_weather_store_path,
)

# label -> the /weather argument that produces it. Snow has no command
# of its own (it's rain in a cold biome), so it defaults to "rain".
_DEFAULT_COMMAND = {
    "clear": "clear",
    "rain": "rain",
    "thunder": "thunder",
    "snow": "rain",
}


def collect_one(keyboard, capture, detector, *, label, command,
                frames, settle, period, mouse=None, pan_px=0) -> int:
    print(f"\n[collect] === {label!r} (via '/weather {command}') ===")
    M._send_chat_message(keyboard, f"/weather {command}")
    print(f"[collect] waiting {settle:.0f}s for weather to settle…")
    time.sleep(settle)

    # Pitch UP so the sky fills the view, giving clean sky samples
    # regardless of where the player was looking. MC clamps pitch at
    # -90 (straight up), so an over-move just lands looking up.
    if mouse is not None:
        try:
            mouse.move(0, -260)
            time.sleep(0.3)
        except Exception:
            pass

    # Reset the detector's temporal state so the first frame's motion
    # feature isn't contaminated by the pre-settle frame.
    detector.reset()
    feats = []
    for i in range(frames):
        frame = capture.get_frame()
        f = detector.extract_features(frame)
        feats.append(f)
        # Pan the view a little between captures so the samples span
        # multiple horizontal directions (sun side / shade side / over
        # different terrain) — the diversity the trained classifier
        # needs to generalise beyond one static viewpoint. Alternate
        # direction so we sweep back and forth around the start yaw.
        if mouse is not None and pan_px:
            step = pan_px if (i // 4) % 2 == 0 else -pan_px
            try:
                mouse.move(step, 0)
            except Exception:
                pass
        if i % 5 == 0:
            print(f"[collect]   frame {i+1}/{frames}  "
                  f"sky_open={f['sky_open_frac']:.2f} "
                  f"bright={f['sky_brightness']:.2f} "
                  f"sat={f['sky_saturation']:.2f} "
                  f"blue={f['sky_blueness']:.2f} "
                  f"edge={f['air_edge_density']:.3f} "
                  f"white={f['air_whiteness']:.3f}")
        time.sleep(period)
    total = append_samples(default_weather_store_path(), label, feats)
    print(f"[collect] saved {len(feats)} '{label}' samples (total now {total})")
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--states", default="clear,rain",
                    help="Comma-separated weather labels to cycle through. "
                         "(thunder is intentionally omitted — visually it's "
                         "just darker rain and lightning is brief/rare; add it "
                         "explicitly only if you want a separate label.)")
    ap.add_argument("--label", default=None,
                    help="Collect a single label (overrides --states). Use "
                         "with --command (e.g. --label snow --command rain).")
    ap.add_argument("--command", default=None,
                    help="The /weather argument to send for --label.")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--settle", type=float, default=10.0,
                    help="Seconds to wait after the command before capturing. "
                         "MC fades weather in/out over several seconds, so "
                         "give the transition time to finish or early frames "
                         "will be mislabelled mid-fade.")
    ap.add_argument("--period", type=float, default=0.12,
                    help="Seconds between captured frames.")
    ap.add_argument("--pan-px", type=int, default=45,
                    help="Mouse yaw step between frames so samples span "
                         "directions (0 = static capture). Aim at the sky "
                         "first; panning keeps it roughly in view.")
    args = ap.parse_args(argv)

    if not _find_minecraft_hwnd():
        print("[collect][ERROR] Minecraft is not running.")
        return 2

    activate_minecraft()
    time.sleep(0.4)
    hwnd = _find_minecraft_hwnd()[0][0]
    capture = Capture(CaptureConfig(hwnd=hwnd, threaded=True))
    capture.start()
    for _ in range(5):
        capture.get_frame(); time.sleep(0.05)

    # Gate-less keyboard + mouse so the tool can type and pan without the
    # safety gate (this is an operator-run data-collection utility).
    from control.keyboard import Keyboard
    from control.mouse import Mouse
    keyboard = Keyboard()
    keyboard.start()
    mouse = Mouse()
    mouse.start()

    settings = M._load_yaml(M.SETTINGS_PATH)
    detector = build_weather_detector(settings)

    def _do(label, command):
        collect_one(keyboard, capture, detector, label=label, command=command,
                    frames=args.frames, settle=args.settle, period=args.period,
                    mouse=mouse, pan_px=args.pan_px)

    try:
        if args.label:
            _do(args.label, args.command or _DEFAULT_COMMAND.get(args.label, args.label))
        else:
            for label in [s.strip() for s in args.states.split(",") if s.strip()]:
                _do(label, _DEFAULT_COMMAND.get(label, label))
        # Restore clear weather so we don't leave the world raining.
        M._send_chat_message(keyboard, "/weather clear")
        time.sleep(0.3)
    finally:
        mouse.stop()
        keyboard.stop()
        capture.stop()

    # Rebuild + report the trained classifier.
    detector.reload_training()
    s = detector.stats()
    print(f"\n[collect] store: {s['store']}")
    print(f"[collect] samples per state: {s['states']}")
    print(f"[collect] trained classifier active: {s['trained']}")
    if not s["trained"]:
        print("[collect] (need >=2 states each with the min sample floor "
              "before the trained path activates; collect more.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
