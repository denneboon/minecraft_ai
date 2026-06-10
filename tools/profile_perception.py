#!/usr/bin/env python3
"""
cProfile the real perception+OCR path on live frames to find the
function-level hotspots. Activates Minecraft, mirrors main.py cadence
(F3 ~3 Hz, perception every tick), runs under cProfile, prints the top
functions by cumulative + total time. Requires Minecraft running.

Run:  python tools/profile_perception.py [n_ticks]
"""
from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as M
from utils.focus import activate_minecraft
from vision.processing import ScreenState


def run(n_ticks: int):
    settings = M._load_yaml(M.SETTINGS_PATH)
    capture = M.build_capture(settings)
    processor = M.build_processor(settings)
    f3_reader = M.build_f3_reader(settings)
    world = M._maybe_build_world_perception(settings)
    activate_minecraft()
    time.sleep(0.4)
    capture.start()
    for _ in range(5):
        capture.get_frame()

    f3_interval = 1.0 / 3.0
    last_f3 = 0.0
    f3_info = None
    for _ in range(n_ticks):
        frame = capture.get_frame()
        state = processor.process(frame)
        now = time.perf_counter()
        if now - last_f3 >= f3_interval and state.screen_state == ScreenState.PLAYING:
            f3_info = f3_reader.read(frame)
            last_f3 = time.perf_counter()
        if world is not None:
            world.update(frame, f3_info)
    capture.stop()


def main(argv=None):
    n = int(argv[0]) if argv else 60
    pr = cProfile.Profile()
    pr.enable()
    run(n)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print("\n================ TOP BY CUMULATIVE ================")
    # Filter to project frames only for readability.
    for line in s.getvalue().splitlines():
        if ("minecraft_ai" in line or "perception" in line or "screen_ray" in line
                or "inverse_renderer" in line or "sample_recognizer" in line
                or "glyph_ocr" in line or "ncalls" in line or "function calls" in line):
            print(line)
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats("tottime")
    ps2.print_stats(25)
    print("\n================ TOP BY TOTTIME ===================")
    for line in s2.getvalue().splitlines()[:40]:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
