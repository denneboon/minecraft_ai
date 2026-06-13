#!/usr/bin/env python3
"""
Environment health check - run this first on a fresh machine (e.g. the big
PC). Verifies the things that silently break a setup: missing deps, a CPU-only
PyTorch when a GPU is expected, an empty (gitignored) data folder, missing MC
assets. Prints a PASS / WARN / FAIL per check and a one-line verdict.

    python tools/doctor.py

Exit code 0 if nothing FAILED (WARNs are fine), 1 otherwise.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_fail = 0
_warn = 0


def _p(status, msg, detail=""):
    global _fail, _warn
    if status == "FAIL":
        _fail += 1
    elif status == "WARN":
        _warn += 1
    tag = {"PASS": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[status]
    print(f"  {tag} {msg}" + (f"  - {detail}" if detail else ""))


def main() -> int:
    print("=" * 64)
    print(" minecraft_ai - environment doctor")
    print("=" * 64)

    # Python
    v = sys.version_info
    _p("PASS" if v[:2] == (3, 11) else "WARN",
       f"Python {v.major}.{v.minor}.{v.micro}",
       "" if v[:2] == (3, 11) else "project is tested on 3.11")

    # Core deps
    for mod, label in [("numpy", "numpy"), ("cv2", "opencv-python"),
                       ("yaml", "PyYAML"), ("PIL", "Pillow")]:
        try:
            importlib.import_module(mod)
            _p("PASS", f"dep {label}")
        except Exception as e:
            _p("FAIL", f"dep {label} missing", repr(e))

    # PyTorch + CUDA (the reason to use this PC)
    try:
        import torch
        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else "-"
        _p("PASS" if cuda else "WARN",
           f"PyTorch {torch.__version__}",
           f"CUDA available: {cuda} ({name})"
           if cuda else "CPU-only - install the CUDA build to train on the GPU")
    except Exception as e:
        _p("FAIL", "PyTorch missing", repr(e))

    # Data (gitignored - must be copied/regenerated)
    samples = ROOT / "data" / "training" / "world_samples"
    n_png = len(list(samples.rglob("*.png"))) if samples.is_dir() else 0
    _p("PASS" if n_png > 0 else "WARN",
       f"sample store: {n_png} samples",
       "" if n_png else "empty - copy data/training/world_samples from the other machine")

    assets = ROOT / "data" / "mc_assets"
    has_assets = assets.is_dir() and any(assets.iterdir())
    _p("PASS" if has_assets else "WARN",
       "MC assets (data/mc_assets)",
       "" if has_assets else "missing - copy it, or it auto-extracts from the MC jar on first run")

    for m in ("block_cnn.pt", "block_fusion.pt"):
        f = ROOT / "data" / "calibration" / m
        _p("PASS" if f.is_file() else "WARN", f"model {m}",
           "" if f.is_file() else "not present (trainable here)")

    # Can the perception stack import?
    try:
        import vision.world.context_fusion  # noqa
        import vision.world.perception      # noqa
        _p("PASS", "perception + fusion modules import")
    except Exception as e:
        _p("FAIL", "perception import failed", repr(e))

    print("-" * 64)
    verdict = "FAIL" if _fail else ("WARN" if _warn else "PASS")
    print(f"  verdict: {verdict}  ({_fail} fail, {_warn} warn)")
    if _fail == 0 and _warn == 0:
        print("  ready - run: python tools/run_tests.py  then train.")
    elif _fail == 0:
        print("  usable - address WARNs (data/GPU) for full capability.")
    else:
        print("  fix the FAILs above before proceeding.")
    return 1 if _fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
