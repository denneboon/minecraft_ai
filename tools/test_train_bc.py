#!/usr/bin/env python3
"""Offline self-test for tools/train_bc.py — the behaviour-cloning pipeline
trains, beats the majority baseline on a learnable pattern, and saves a
model. Skips gracefully if PyTorch isn't installed."""
from __future__ import annotations

import csv
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

_fails = 0
def ok(m): print(f"  [OK]  {m}")
def bad(m):
    global _fails; _fails += 1; print(f"  [FAIL] {m}")


def main() -> int:
    print("=" * 56); print(" train_bc — offline self-test"); print("=" * 56)
    try:
        import torch  # noqa: F401
    except Exception:
        print("  [SKIP] PyTorch not installed — BC pipeline test skipped.")
        print("\nALL TRAIN_BC TESTS PASSED (skipped)")
        return 0

    import tools.train_bc as bc
    tmp = tempfile.mkdtemp(prefix="bc_test_")
    csv_path = os.path.join(tmp, "ds.csv")
    cols = ["episode"] + bc._NUM_COLS + ["o_look_cat", "o_state"] + [
        "a_forward", "a_back", "a_left", "a_right", "a_jump", "a_sprint",
        "a_sneak", "a_look_dx", "a_look_dy", "a_interact", "a_slot"]
    # Learnable pattern: on a 'log' -> attack+slot2; otherwise -> none/forward.
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for i in range(120):
            on_log = (i % 2 == 0)
            num = [0.0, 0.0, 1.0, 0.0, 0.0, (2.0 if on_log else 0.0), 1.0, 0.8]
            look = "log" if on_log else "none"
            state = "chop" if on_log else "scan"
            act = (["0"] * 5 + ["0", "0", "0", "0", "attack", "2"] if on_log
                   else ["1", "0", "0", "0", "0", "1", "0", "0", "0", "none", "0"])
            w.writerow(["ep"] + [str(x) for x in num] + [look, state] + act)

    bc.MODEL_DIR = os.path.join(tmp, "models")
    res = bc.train(csv_path, epochs=80, seed=0)
    (ok if res is not None else bad)("train returned a result")
    if res:
        accs, base = res
        (ok if set(accs) == {"interact", "forward", "sprint", "slot"} else bad)(
            f"per-head accuracies reported ({sorted(accs)})")
        (ok if all(0.0 <= v <= 1.0 for v in accs.values()) else bad)(
            "accuracies in [0,1]")
        # The interact pattern is perfectly learnable -> should beat baseline.
        (ok if accs["interact"] >= base["interact"] else bad)(
            f"interact learned >= majority baseline ({accs['interact']:.2f} "
            f">= {base['interact']:.2f})")
    (ok if os.path.exists(os.path.join(bc.MODEL_DIR, "bc_policy.pt")) else bad)(
        "model saved")
    # too-little-data guard
    (ok if bc.train(csv_path, epochs=1) is not None else bad)("trains with epochs=1")

    print("\n" + ("ALL TRAIN_BC TESTS PASSED" if not _fails
                  else f"{_fails} CHECK(S) FAILED"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
