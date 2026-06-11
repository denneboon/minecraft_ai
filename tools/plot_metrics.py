#!/usr/bin/env python3
"""
Graph the block recogniser's self-teaching progress.

Reads the metrics history written by ``tools/learn_world_live.py``
(``data/metrics/sessions.jsonl`` + the per-session ``*_steps.csv``) and
renders PNG charts so you can SEE the AI getting better over time:

  * accuracy + coverage per session (long-term progress)
  * total sample count per session (how much it has learned from)
  * latest session's per-block accuracy (which blocks it's good/bad at)
  * latest session's within-run learning curve (running accuracy)

    python tools/plot_metrics.py                 # writes PNGs to data/metrics/
    python tools/plot_metrics.py --show          # also open them

No Minecraft needed — pure post-processing of the logged metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vision.world.metrics import default_metrics_root


def _load_sessions(root: Path):
    hist = root / "sessions.jsonl"
    if not hist.is_file():
        return []
    out = []
    for line in hist.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    # Oldest -> newest by start time.
    out.sort(key=lambda s: s.get("start_ts_unix", 0))
    return out


def _load_steps(root: Path, session_id: str):
    p = root / f"{session_id}_steps.csv"
    if not p.is_file():
        return []
    with p.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", action="store_true", help="open the charts after saving")
    args = ap.parse_args(argv)

    root = default_metrics_root()
    sessions = _load_sessions(root)
    if not sessions:
        print(f"[plot] no metrics yet at {root / 'sessions.jsonl'}. "
              f"Run a session: python tools/learn_world_live.py --steps 150")
        return 0

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(1, len(sessions) + 1))
    acc = [s.get("accuracy", 0.0) for s in sessions]
    cov = [s.get("coverage", 0.0) for s in sessions]
    samples = [s.get("samples_after", 0) for s in sessions]
    hits = [s.get("hits", 0) for s in sessions]
    decided = [s.get("decided", 0) for s in sessions]
    saved = []

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(xs, acc, "o-", label="decided accuracy", color="#2a9d2a")
    ax.plot(xs, cov, "s--", label="coverage", color="#3070d0")
    ax.set_ylim(0, 1.02); ax.set_xlabel("session #"); ax.set_ylabel("fraction")
    ax.set_title(f"Recogniser accuracy & coverage over {len(sessions)} session(s)")
    ax.grid(True, alpha=0.3); ax.legend()
    for x, a, n, d in zip(xs, acc, hits, decided):
        ax.annotate(f"{a:.0%}\n{n}/{d}", (x, a), fontsize=7,
                    ha="center", va="bottom")
    p = root / "progress_accuracy.png"; fig.tight_layout(); fig.savefig(p, dpi=110)
    saved.append(p); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs, samples, "o-", color="#b05020")
    ax.set_xlabel("session #"); ax.set_ylabel("samples in store")
    ax.set_title("Training samples accumulated"); ax.grid(True, alpha=0.3)
    p = root / "progress_samples.png"; fig.tight_layout(); fig.savefig(p, dpi=110)
    saved.append(p); plt.close(fig)

    # Latest session per-block accuracy.
    latest = sessions[-1]
    pb = latest.get("per_block", {})
    if pb:
        names = sorted(pb, key=lambda b: pb[b]["seen"], reverse=True)
        accs = [pb[b]["hits"] / pb[b]["seen"] if pb[b]["seen"] else 0 for b in names]
        seen = [pb[b]["seen"] for b in names]
        fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(names))))
        short = [n.split(":")[-1] for n in names]
        bars = ax.barh(short, accs, color="#2a9d2a")
        ax.set_xlim(0, 1.02); ax.invert_yaxis()
        ax.set_xlabel("accuracy"); ax.set_title("Latest session — per-block accuracy")
        for b, a, n in zip(bars, accs, seen):
            ax.text(min(a + 0.01, 0.9), b.get_y() + b.get_height() / 2,
                    f"{a:.0%} (n={n})", va="center", fontsize=8)
        p = root / "progress_per_block.png"; fig.tight_layout(); fig.savefig(p, dpi=110)
        saved.append(p); plt.close(fig)

    # Latest within-session learning curve.
    steps = _load_steps(root, latest.get("session_id", ""))
    if steps:
        sx = [int(r["step"]) for r in steps]
        racc = [float(r["running_acc"]) for r in steps]
        ns = [int(r["samples"]) for r in steps]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(sx, racc, "-", color="#2a9d2a", label="running accuracy")
        ax.set_ylim(0, 1.02); ax.set_xlabel("step"); ax.set_ylabel("running accuracy")
        ax2 = ax.twinx(); ax2.plot(sx, ns, "--", color="#888", label="samples")
        ax2.set_ylabel("samples")
        ax.set_title(f"Within-session learning curve ({latest.get('session_id','')})")
        ax.grid(True, alpha=0.3)
        p = root / "progress_learning_curve.png"; fig.tight_layout(); fig.savefig(p, dpi=110)
        saved.append(p); plt.close(fig)

    # Benchmark history (deterministic held-out accuracy over time) — the
    # signal that catches regressions live-accuracy hides.
    bench_p = root / "benchmark_history.jsonl"
    if bench_p.is_file():
        bench = []
        for line in bench_p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    bench.append(json.loads(line))
                except Exception:
                    pass
        if bench:
            bx = list(range(1, len(bench) + 1))
            cacc = [b.get("cnn_clean_acc", 0.0) for b in bench]
            aacc = [b.get("cnn_aug_acc", 0.0) for b in bench]
            nsmp = [b.get("n_samples", 0) for b in bench]
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.plot(bx, cacc, "o-", color="#2a9d2a", label="CNN benchmark (clean)")
            ax.plot(bx, aacc, "s--", color="#b05020", label="CNN benchmark (augmented)")
            ax.set_ylim(0, 1.02); ax.set_xlabel("benchmark run #")
            ax.set_ylabel("decided accuracy")
            ax.set_title("Held-out benchmark accuracy over time (regression watch)")
            ax.grid(True, alpha=0.3); ax.legend(loc="lower left")
            ax2 = ax.twinx(); ax2.plot(bx, nsmp, ":", color="#888", lw=1)
            ax2.set_ylabel("samples in store")
            p = root / "progress_benchmark.png"; fig.tight_layout()
            fig.savefig(p, dpi=110); saved.append(p); plt.close(fig)

    print(f"[plot] {len(sessions)} session(s) -> wrote {len(saved)} chart(s):")
    for p in saved:
        print(f"    {p}")
    if args.show:
        for p in saved:
            try:
                os.startfile(str(p))     # Windows
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
