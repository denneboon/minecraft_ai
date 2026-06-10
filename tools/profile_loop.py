#!/usr/bin/env python3
"""
Profile the live agent loop, stage by stage, to find where per-tick time
goes. Mirrors main.py's real cadence: capture + process + perception every
tick, F3 OCR at ~3 Hz. Requires Minecraft running.

Run:
    python tools/profile_loop.py [n_ticks]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as M  # reuse the exact subsystem builders the live loop uses
from vision.ocr import F3ReaderWorker
from vision.pose_filter import PoseFilter
from utils.focus import activate_minecraft


def _stats(name, xs):
    if not xs:
        print(f"  {name:22s}  (no samples)")
        return
    xs = sorted(xs)
    n = len(xs)
    mean = sum(xs) / n
    p50 = xs[n // 2]
    p95 = xs[min(n - 1, int(n * 0.95))]
    mx = xs[-1]
    print(f"  {name:22s}  mean={mean*1000:7.2f}ms  p50={p50*1000:7.2f}ms  "
          f"p95={p95*1000:7.2f}ms  max={mx*1000:7.2f}ms  n={n}")


def main(argv=None):
    n_ticks = int(argv[0]) if argv else 80

    settings = M._load_yaml(M.SETTINGS_PATH)
    capture = M.build_capture(settings)
    processor = M.build_processor(settings)
    f3_reader = M.build_f3_reader(settings)
    world = M._maybe_build_world_perception(settings)
    activate_minecraft()
    time.sleep(0.4)
    capture.start()
    # Mirror the live loop: threaded OCR worker supplies the latest pose.
    f3_worker = F3ReaderWorker(
        f3_reader, capture, pose_filter=PoseFilter(),
        interval_sec=float(M._get(settings, "vision.ocr.read_interval_sec", 0.12)))
    f3_worker.start()

    t_capture, t_process, t_world, t_total = [], [], [], []

    print("[profile] warming up…")
    for _ in range(10):
        capture.get_frame()
        time.sleep(0.05)

    print(f"[profile] running {n_ticks} ticks…")
    for _ in range(n_ticks):
        t0 = time.perf_counter()

        c0 = time.perf_counter()
        frame = capture.get_frame()
        c1 = time.perf_counter()
        t_capture.append(c1 - c0)

        processor.process(frame)   # timed; result unused in the profiler
        c2 = time.perf_counter()
        t_process.append(c2 - c1)

        f3_info = f3_worker.latest()

        if world is not None:
            w0 = time.perf_counter()
            try:
                world.update(frame, f3_info)
            except Exception as e:
                print(f"[profile][WARN] world.update raised: {e!r}")
            t_world.append(time.perf_counter() - w0)

        t_total.append(time.perf_counter() - t0)
        # pace ~ control loop
        time.sleep(max(0.0, 0.02 - (time.perf_counter() - t0)))

    f3_worker.stop()
    capture.stop()

    print("\n[profile] per-stage timing (over the run, F3 OCR is off-thread):")
    _stats("capture.get_frame", t_capture)
    _stats("processor.process", t_process)
    _stats("world.update", t_world)
    _stats("TOTAL / tick (ex-sleep)", t_total)
    print(f"[profile] f3 worker reads: {f3_worker.reads()}")
    total_mean = (sum(t_total) / len(t_total)) if t_total else 0.0
    print(f"\n[profile] effective rate ≈ {1.0/total_mean:.1f} Hz "
          f"(mean {total_mean*1000:.1f} ms/tick)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
