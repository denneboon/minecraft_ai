#!/usr/bin/env python3
"""
Dataset health for the world block recogniser — what it knows, what to
collect next.

The recogniser only gets better with more VARIED F3-labelled samples,
and the recurring bottleneck is that a self-teaching run sees whatever
blocks happen to be around (usually a lot of grass). This tool reports
the live sample store's composition so you know, at a glance, which
blocks are solidly learned, which are thin, and which to go look at next.

Tiers (by sample count per block):
  * READY    (>= 12)  — enough to recognise reliably across conditions
  * LEARNING (4..11)  — trainable but thin; more samples will firm it up
  * STARVED  (1..3)   — barely seen; the model can't trust it yet

    python tools/sample_coverage.py            # text report
    python tools/sample_coverage.py --plot     # + a bar chart in data/metrics/

No Minecraft needed — pure read of data/training/world_samples/.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

from vision.world.sample_store import build_world_sample_store

READY_MIN = 12
LEARNING_MIN = 4


def _tier(n: int) -> str:
    if n >= READY_MIN:
        return "READY"
    if n >= LEARNING_MIN:
        return "LEARNING"
    return "STARVED"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true",
                    help="also write a samples-per-block bar chart to data/metrics/")
    args = ap.parse_args(argv)

    store = build_world_sample_store()
    manifest = store.manifest()                 # {block_id: count}
    if not manifest:
        print("[coverage] sample store is empty. Collect some: "
              "python tools/learn_world_live.py --steps 150")
        return 0

    items = sorted(manifest.items(), key=lambda kv: (-kv[1], kv[0]))
    total = sum(manifest.values())
    tiers = {"READY": [], "LEARNING": [], "STARVED": []}
    for bid, n in items:
        tiers[_tier(n)].append((bid, n))

    print("=" * 60)
    print(f" Recogniser dataset coverage — {total} samples / "
          f"{len(manifest)} blocks")
    print("=" * 60)
    for tier, emoji in (("READY", "[v]"), ("LEARNING", "[~]"), ("STARVED", "[!]")):
        rows = tiers[tier]
        print(f"\n {emoji} {tier} ({len(rows)})"
              + (f"  — need {LEARNING_MIN}+ to train, {READY_MIN}+ to trust"
                 if tier == "STARVED" else ""))
        for bid, n in rows:
            bar = "#" * min(40, n)
            print(f"    {bid.split(':')[-1]:24} {n:3d}  {bar}")

    # Actionable guidance.
    print("\n" + "-" * 60)
    need = [b.split(':')[-1] for b, n in items if n < READY_MIN]
    if need:
        print(" COLLECT NEXT (look at these to firm them up): "
              + ", ".join(need))
    print(f" Model-ready blocks: {len(tiers['READY'])}  |  "
          f"trainable: {len(tiers['READY']) + len(tiers['LEARNING'])}  |  "
          f"starved: {len(tiers['STARVED'])}")
    print(" Tip: stand somewhere with variety (stone, ores, logs, planks, "
          "sand, terracotta) and run learn_world_live to grow the thin ones.")
    print("=" * 60)

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from vision.world.metrics import default_metrics_root
            names = [b.split(":")[-1] for b, _ in items]
            counts = [n for _, n in items]
            colors = {"READY": "#2a9d2a", "LEARNING": "#d0a020",
                      "STARVED": "#c03020"}
            bar_colors = [colors[_tier(n)] for n in counts]
            fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(names))))
            ax.barh(names, counts, color=bar_colors)
            ax.invert_yaxis()
            ax.axvline(LEARNING_MIN, ls="--", color="#888", lw=1)
            ax.axvline(READY_MIN, ls="--", color="#444", lw=1)
            ax.set_xlabel("samples"); ax.set_title(
                f"Dataset coverage — {total} samples / {len(manifest)} blocks "
                f"(dashed: trainable {LEARNING_MIN}, ready {READY_MIN})")
            root = default_metrics_root(); root.mkdir(parents=True, exist_ok=True)
            p = root / "dataset_coverage.png"
            fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
            print(f"[coverage] chart -> {p}")
        except Exception as e:
            print(f"[coverage] plot skipped: {e!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
