#!/usr/bin/env python3
"""
Run the project's offline self-test suite in one command.

These are the ``tools/test_*.py`` runners that need NO running Minecraft
(synthetic frames / fakes). The live tools (test_mouse_camera,
test_mouse_visual, test_inventory, test_world_perception_live) are
excluded because they drive a real game window.

Usage
-----
    python tools/run_tests.py            # run all, summarise
    python tools/run_tests.py -k weather # only tests whose name matches

Exit code is non-zero if any suite failed — suitable for CI / a pre-push
check. (pytest isn't required; each suite is a self-contained script.)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The self-contained offline suites, in a sensible order (fast/structural
# first). Keep this list in sync when adding new offline test tools.
OFFLINE_TESTS = [
    "test_ocr_f3",
    "test_cnn_recognizer",
    "test_world_perception",
    "test_world_explorer_offline",
    "test_pathfind",
    "test_walker",
    "test_inventory_synthetic",
    "test_weather",
    "test_script_runner",
    "test_hotbar",
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", "--filter", default=None,
                    help="Only run suites whose name contains this substring.")
    args = ap.parse_args(argv)

    tests = [t for t in OFFLINE_TESTS
             if args.filter is None or args.filter in t]
    if not tests:
        print(f"[run_tests] no offline suite matches {args.filter!r}.")
        return 2

    results = []
    print(f"[run_tests] running {len(tests)} offline suite(s)…\n")
    for name in tests:
        path = ROOT / "tools" / f"{name}.py"
        if not path.is_file():
            print(f"  MISSING  {name} (no {path.name})")
            results.append((name, None, 0.0))
            continue
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, str(path)],
                              capture_output=True, text=True)
        dt = time.perf_counter() - t0
        passed = proc.returncode == 0
        results.append((name, passed, dt))
        status = "PASS" if passed else "FAIL"
        print(f"  {status:4}  {name:34} {dt:6.2f}s")
        if not passed:
            # Show the tail of the failing output so the failure is visible.
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-12:]
            for line in tail:
                print(f"        | {line}")

    n_pass = sum(1 for _, p, _ in results if p)
    n_fail = sum(1 for _, p, _ in results if p is False)
    n_miss = sum(1 for _, p, _ in results if p is None)
    total_t = sum(dt for _, _, dt in results)
    print(f"\n[run_tests] {n_pass} passed, {n_fail} failed, {n_miss} missing "
          f"in {total_t:.1f}s")
    return 0 if (n_fail == 0 and n_miss == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
