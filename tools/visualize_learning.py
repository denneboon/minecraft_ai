#!/usr/bin/env python3
"""
"Watch it learn" — animate the context-fusion recogniser's embedding space
organising itself over training (the YouTube-style neural-net learning clip).

Each epoch we train one pass, then project the held-out samples' 96-d fused
embeddings to 2-D and snapshot them. Early on they're a random blob; as the net
learns, same-block points pull into tight clusters. A live accuracy curve grows
alongside.

    python tools/visualize_learning.py                 # -> data/learning.gif (+ .mp4 if ffmpeg)
    python tools/visualize_learning.py --blocks 12 --samples 3000 --epochs 50
"""
from __future__ import annotations
import os, sys, argparse
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import torch
import torch.nn.functional as F
from vision.world.context_fusion import ContextFusionRecognizer, _FusionNet
from vision.world.sample_store import build_world_sample_store


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks", type=int, default=10, help="top-N most common blocks to show")
    ap.add_argument("--samples", type=int, default=2500, help="cap total samples (speed)")
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/learning")
    args = ap.parse_args(argv)

    store = build_world_sample_store()
    cf = ContextFusionRecognizer(store)        # for .features / .cfg / ._prep_patch
    samples = list(store.load_all(with_metadata=True))   # metadata = the context!
    if not samples:
        print("[viz] sample store empty"); return 2

    # Pick the top-N most-populated blocks (clear, distinct clusters) + cap size.
    counts = Counter(s.block_id for s in samples)
    top = [b for b, _ in counts.most_common(args.blocks)]
    sel = [s for s in samples if s.block_id in top]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(sel)
    sel = sel[: args.samples]
    classes = sorted({s.block_id for s in sel})
    cls_idx = {b: i for i, b in enumerate(classes)}
    print(f"[viz] {len(sel)} samples over {len(classes)} blocks: "
          + ", ".join(c.split(':')[-1] for c in classes))

    # Pre-encode everything ONCE (no augmentation -> stable per-epoch snapshots).
    torch.manual_seed(args.seed)
    ctx = torch.from_numpy(np.stack([cf.features.encode(s.metadata)
                                     for s in sel]).astype(np.float32))
    patch = torch.stack([cf._prep_patch(s.rgb) for s in sel])
    y = torch.from_numpy(np.array([cls_idx[s.block_id] for s in sel], dtype=np.int64))

    n = len(sel); idx = np.arange(n); rng.shuffle(idx)
    ntest = max(40, n // 4)
    te, tr = idx[:ntest], idx[ntest:]
    p_tr, c_tr, y_tr = patch[tr], ctx[tr], y[tr]
    p_te, c_te, y_te = patch[te], ctx[te], y[te]

    model = _FusionNet(cf.cfg, ctx.shape[1], len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=cf.cfg.lr, weight_decay=cf.cfg.weight_decay)

    snaps_emb, accs = [], []                   # per-epoch test embeddings + test acc
    bs = 64
    for ep in range(args.epochs):
        model.train()
        order = np.arange(len(tr)); rng.shuffle(order)
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            opt.zero_grad()
            loss = F.cross_entropy(model(p_tr[b], c_tr[b]), y_tr[b])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            emb = model.embed(p_te, c_te).numpy()
            acc = (model(p_te, c_te).argmax(1) == y_te).float().mean().item()
        snaps_emb.append(emb); accs.append(acc)
        print(f"  epoch {ep+1:2d}/{args.epochs}  test acc {acc:.3f}")

    # PCA fit on the FINAL (most-organised) embeddings; project every epoch onto
    # those axes so the camera is fixed and we watch the clusters form.
    Ef = snaps_emb[-1]; mu = Ef.mean(0)
    _, _, Vt = np.linalg.svd(Ef - mu, full_matrices=False)
    comp = Vt[:2].T

    def proj(E):
        return (E - mu) @ comp

    all2d = [proj(E) for E in snaps_emb]
    flat = np.concatenate(all2d)
    xlo, xhi = flat[:, 0].min(), flat[:, 0].max()
    ylo, yhi = flat[:, 1].min(), flat[:, 1].max()
    mx = (xhi - xlo) * 0.08 + 1e-3; my = (yhi - ylo) * 0.08 + 1e-3
    y_te_np = y_te.numpy()
    cmap = plt.get_cmap("tab10" if len(classes) <= 10 else "tab20")
    colors = [cmap(i % cmap.N) for i in range(len(classes))]

    fig, (axE, axA) = plt.subplots(1, 2, figsize=(14, 7),
                                   gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle("Context-Fusion Recogniser — learning to separate blocks",
                 fontsize=15, weight="bold")

    def update(f):
        axE.clear(); axA.clear()
        P = all2d[f]
        for i, cname in enumerate(classes):
            m = y_te_np == i
            axE.scatter(P[m, 0], P[m, 1], s=14, color=colors[i],
                        label=cname.split(":")[-1], alpha=0.8, edgecolors="none")
        axE.set_xlim(xlo - mx, xhi + mx); axE.set_ylim(ylo - my, yhi + my)
        axE.set_xticks([]); axE.set_yticks([])
        axE.set_title(f"held-out embedding space (2-D PCA)\nepoch {f+1}/{len(all2d)}"
                      f"   ·   test accuracy {accs[f]:.0%}", fontsize=11)
        axE.legend(loc="upper right", fontsize=7, framealpha=0.85, ncol=2)
        axA.plot(range(1, f + 2), accs[:f + 1], color="#2b6cb0", lw=2.2)
        axA.scatter([f + 1], [accs[f]], color="#c53030", zorder=5)
        axA.set_xlim(1, len(all2d)); axA.set_ylim(0, 1)
        axA.set_xlabel("epoch"); axA.set_ylabel("held-out accuracy")
        axA.set_title("learning curve", fontsize=11)
        axA.grid(alpha=0.3)
        return axE, axA

    frames = len(all2d)
    anim = FuncAnimation(fig, update, frames=frames, interval=180, blit=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    gif = args.out + ".gif"
    anim.save(gif, writer=PillowWriter(fps=6))
    print(f"[viz] saved {gif}")
    try:
        mp4 = args.out + ".mp4"
        anim.save(mp4, writer="ffmpeg", fps=10, dpi=130)
        print(f"[viz] saved {mp4}")
    except Exception as e:
        print(f"[viz] mp4 skipped (no ffmpeg?): {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
