#!/usr/bin/env python3
"""
Snapshot / restore the block-recogniser model — rollback safety.

The recogniser retrains itself continuously, so a stretch of bad data
(e.g. a vine-heavy or night-darkened view) can quietly degrade the live
model with no way back. This keeps timestamped copies of
``data/calibration/block_cnn.pt`` tagged with the benchmark score at
snapshot time, so you can always roll back to the best model.

    python tools/model_snapshot.py save  --note "after clean day run"
    python tools/model_snapshot.py list
    python tools/model_snapshot.py restore best        # highest clean-acc
    python tools/model_snapshot.py restore <filename>

Pairs with tools/eval_recognizer.py --log: `save` reads the latest logged
benchmark score so a snapshot is labelled with how good it was.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vision.world.cnn_recognizer import CNNBlockRecognizerConfig
from vision.world.metrics import default_metrics_root

MODEL_PATH = Path(CNNBlockRecognizerConfig().model_path)
SNAP_DIR = MODEL_PATH.parent / "model_snapshots"
KEEP = 12          # rotate: keep the most recent N snapshots


def _latest_benchmark() -> dict:
    p = default_metrics_root() / "benchmark_history.jsonl"
    if not p.is_file():
        return {}
    last = ""
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            last = line
    try:
        return json.loads(last) if last else {}
    except Exception:
        return {}


def _snapshots():
    if not SNAP_DIR.is_dir():
        return []
    out = []
    for pt in sorted(SNAP_DIR.glob("block_cnn_*.pt")):
        meta = {}
        j = pt.with_suffix(".json")
        if j.is_file():
            try:
                meta = json.loads(j.read_text(encoding="utf-8"))
            except Exception:
                pass
        out.append((pt, meta))
    return out


def _model_vocab(pt: Path) -> int:
    """How many block classes a snapshot model actually knows (its real
    vocabulary), read straight from the checkpoint — authoritative even when
    the sidecar benchmark is missing or stale. 0 on any error."""
    try:
        import torch
        ck = torch.load(pt, map_location="cpu", weights_only=False)
        return len(ck.get("classes") or [])
    except Exception:
        return 0


def cmd_save(note: str) -> int:
    if not MODEL_PATH.is_file():
        print(f"[snapshot] no model at {MODEL_PATH} — train one first.")
        return 1
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    dst = SNAP_DIR / f"block_cnn_{ts}.pt"
    shutil.copy2(MODEL_PATH, dst)
    bench = _latest_benchmark()
    meta = {
        "ts": ts, "ts_unix": round(time.time(), 1),
        "source": str(MODEL_PATH), "note": note,
        "benchmark_clean_acc": bench.get("cnn_clean_acc"),
        "benchmark_aug_acc": bench.get("cnn_aug_acc"),
        "benchmark_n_samples": bench.get("n_samples"),
        "benchmark_n_blocks": bench.get("n_blocks"),
        "n_vocab": _model_vocab(dst),     # the model's REAL vocabulary size
        "benchmark_ts": bench.get("ts"),
    }
    dst.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[snapshot] saved {dst.name}  "
          f"(clean acc {meta['benchmark_clean_acc']})")
    # Rotate.
    snaps = _snapshots()
    if len(snaps) > KEEP:
        for pt, _ in snaps[:len(snaps) - KEEP]:
            pt.unlink(missing_ok=True)
            pt.with_suffix(".json").unlink(missing_ok=True)
        print(f"[snapshot] rotated; kept newest {KEEP}")
    return 0


def cmd_list() -> int:
    snaps = _snapshots()
    if not snaps:
        print(f"[snapshot] none yet in {SNAP_DIR}")
        return 0
    print(f"=== model snapshots ({len(snaps)}) in {SNAP_DIR} ===")
    for pt, m in snaps:
        acc = m.get("benchmark_clean_acc")
        accs = f"{acc:.0%}" if isinstance(acc, (int, float)) else "  ?"
        nv = m.get("n_vocab") or _model_vocab(pt)
        print(f"  {pt.name}  clean={accs}  vocab={nv}  "
              f"n={m.get('benchmark_n_samples','?')}  {m.get('note','')}")
    return 0


def cmd_restore(which: str) -> int:
    snaps = _snapshots()
    if not snaps:
        print("[snapshot] none to restore.")
        return 1
    if which == "best":
        # COVERAGE-AWARE best: raw clean-acc alone wrongly favours a SMALL
        # vocabulary — fewer/easier classes confuse less, so a 5-block 0.93
        # model beats a 9-block 0.89 one the agent actually needs (a real
        # footgun that happened). Among snapshots within 5% of the top
        # accuracy, pick the one that knows the MOST blocks.
        def _acc(m):
            a = m.get("benchmark_clean_acc")
            return a if isinstance(a, (int, float)) else None
        accs = [a for a in (_acc(m) for _, m in snaps) if a is not None]
        top = max(accs) if accs else None
        if top is None:
            target = snaps[-1][0]         # no benchmarks -> newest
            print(f"[snapshot] no benchmarked snapshots; newest: {target.name}")
        else:
            near = [(pt, m) for pt, m in snaps
                    if _acc(m) is not None and _acc(m) >= top - 0.05]
            near.sort(key=lambda pm: (pm[1].get("n_vocab") or _model_vocab(pm[0]),
                                      _acc(pm[1])), reverse=True)
            target, tm = near[0]
            tv = tm.get("n_vocab") or _model_vocab(target)
            print(f"[snapshot] best (coverage-aware): {target.name} "
                  f"(clean {_acc(tm):.0%}, {tv} blocks; vs top acc {top:.0%})")
    else:
        match = [pt for pt, _ in snaps if pt.name == which or pt.stem == which]
        if not match:
            print(f"[snapshot] no snapshot named {which!r}. Use 'list'.")
            return 1
        target = match[0]
    # Back up the current live model first (so restore is itself reversible).
    if MODEL_PATH.is_file():
        shutil.copy2(MODEL_PATH, MODEL_PATH.with_suffix(".pt.prerestore"))
    shutil.copy2(target, MODEL_PATH)
    print(f"[snapshot] restored {target.name} -> {MODEL_PATH} "
          f"(previous saved as block_cnn.pt.prerestore)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("save"); sp.add_argument("--note", default="")
    sub.add_parser("list")
    rp = sub.add_parser("restore"); rp.add_argument("which")
    args = ap.parse_args(argv)
    if args.cmd == "save":
        return cmd_save(args.note)
    if args.cmd == "list":
        return cmd_list()
    if args.cmd == "restore":
        return cmd_restore(args.which)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
