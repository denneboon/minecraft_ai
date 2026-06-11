#!/usr/bin/env python3
"""
Behaviour-cloning baseline — train a small policy to imitate the rule-based
agents from their recorded episodes. This is the capstone of the data
track (capture -> export -> TRAIN): it closes the loop the architecture was
built for, where a learned policy can later replace a rule-based Skill
behind the same AgentAction interface.

It reads the (observation -> action) CSV from tools/export_dataset.py and
trains a tiny multi-head MLP (PyTorch, CPU) to predict the agent's discrete
action (interact / forward / sprint / hotbar slot) from the observation.
Reports per-head validation accuracy vs a majority-class baseline (so you
can see whether it learned anything beyond "always do the common thing"),
and saves the model + a feature/label spec to data/models/.

    python tools/export_dataset.py        # produce data/datasets/dataset.csv
    python tools/train_bc.py              # train on it
    python tools/train_bc.py --dataset data/datasets/dataset.csv --epochs 60

NOTE: this is a baseline + the full logging->export->train pipeline; the
learned model is NOT swapped into an agent yet (the rule-based bots are
better until far more data is collected). It exists so every recorded run
becomes training signal and the learned-policy path is ready.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

DS_DIR = os.path.join(ROOT, "data", "datasets")
MODEL_DIR = os.path.join(ROOT, "data", "models")

_NUM_COLS = ["o_yaw", "o_pitch", "o_look_dx", "o_look_dy", "o_look_dz",
             "o_look_dist", "o_health", "o_hunger"]
_LOOK_CATS = ["none", "air", "log", "leaves", "other"]
_INTERACTS = ["none", "attack", "use_hold", "use_item", "drop"]


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_rows(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    states = sorted({r.get("o_state", "") for r in rows})
    return rows, states


def featurize(rows, states):
    import numpy as np
    state_idx = {s: i for i, s in enumerate(states)}
    look_idx = {c: i for i, c in enumerate(_LOOK_CATS)}
    X, Yi, Yf, Ys, Yslot = [], [], [], [], []
    for r in rows:
        num = [_f(r.get(c)) for c in _NUM_COLS]
        lc = [0.0] * len(_LOOK_CATS)
        lc[look_idx.get(r.get("o_look_cat", "none"), 0)] = 1.0
        st = [0.0] * len(states)
        st[state_idx.get(r.get("o_state", ""), 0)] = 1.0
        X.append(num + lc + st)
        Yi.append(_INTERACTS.index(r["a_interact"]) if r.get("a_interact") in _INTERACTS else 0)
        Yf.append(int(_f(r.get("a_forward"))))
        Ys.append(int(_f(r.get("a_sprint"))))
        Yslot.append(min(9, max(0, int(_f(r.get("a_slot"))))))
    X = np.array(X, dtype="float32")
    # standardise numeric block (first len(_NUM_COLS) cols)
    n = len(_NUM_COLS)
    mu = X[:, :n].mean(0); sd = X[:, :n].std(0) + 1e-6
    X[:, :n] = (X[:, :n] - mu) / sd
    return (X, np.array(Yi), np.array(Yf), np.array(Ys), np.array(Yslot),
            {"num_mean": mu.tolist(), "num_std": sd.tolist(), "states": states})


def _majority_acc(y, mask):
    import numpy as np
    if mask.sum() == 0:
        return 0.0
    vals, counts = np.unique(y[mask], return_counts=True)
    return float(counts.max() / mask.sum())


def train(path, epochs=60, seed=0):
    import numpy as np
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    rows, states = load_rows(path)
    if len(rows) < 8:
        print(f"Not enough rows to train ({len(rows)}). Collect more episodes.")
        return None
    X, Yi, Yf, Ys, Yslot, norm = featurize(rows, states)
    nfeat = X.shape[1]
    idx = np.random.permutation(len(X))
    cut = max(1, int(0.8 * len(X)))
    tr, va = idx[:cut], idx[cut:]
    if len(va) == 0:
        va = tr

    t = lambda a: torch.tensor(a)
    Xt = t(X)

    class Net(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(d, 64), nn.ReLU(),
                                       nn.Linear(64, 64), nn.ReLU())
            self.h_int = nn.Linear(64, len(_INTERACTS))
            self.h_fwd = nn.Linear(64, 2)
            self.h_spr = nn.Linear(64, 2)
            self.h_slot = nn.Linear(64, 10)

        def forward(self, x):
            z = self.trunk(x)
            return self.h_int(z), self.h_fwd(z), self.h_spr(z), self.h_slot(z)

    net = Net(nfeat)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    Yi_t, Yf_t, Ys_t, Yslot_t = t(Yi), t(Yf), t(Ys), t(Yslot)
    tri = torch.tensor(tr, dtype=torch.long)
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        oi, of, os_, osl = net(Xt[tri])
        loss = (ce(oi, Yi_t[tri]) + ce(of, Yf_t[tri])
                + ce(os_, Ys_t[tri]) + ce(osl, Yslot_t[tri]))
        loss.backward(); opt.step()

    net.eval()
    with torch.no_grad():
        vay = torch.tensor(va, dtype=torch.long)
        oi, of, os_, osl = net(Xt[vay])
        def acc(logits, y): return float((logits.argmax(1) == y[vay]).float().mean())
        accs = {"interact": acc(oi, Yi_t), "forward": acc(of, Yf_t),
                "sprint": acc(os_, Ys_t), "slot": acc(osl, Yslot_t)}
    vamask = np.zeros(len(X), bool); vamask[va] = True
    base = {"interact": _majority_acc(Yi, vamask), "forward": _majority_acc(Yf, vamask),
            "sprint": _majority_acc(Ys, vamask), "slot": _majority_acc(Yslot, vamask)}

    os.makedirs(MODEL_DIR, exist_ok=True)
    out = os.path.join(MODEL_DIR, "bc_policy.pt")
    torch.save({"state_dict": net.state_dict(), "nfeat": nfeat,
                "num_cols": _NUM_COLS, "look_cats": _LOOK_CATS,
                "interacts": _INTERACTS, "norm": norm, "rows": len(rows)}, out)
    print(f"trained on {len(rows)} rows ({len(tr)} train / {len(va)} val), "
          f"{nfeat} features")
    for k in accs:
        print(f"  {k:9} val acc {accs[k]:.2f}  (majority baseline {base[k]:.2f})")
    print(f"  model -> {out}")
    return accs, base


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args(argv)
    try:
        import torch  # noqa: F401
    except Exception:
        print("PyTorch not available — install it (pip install -e .[ml]) to train.")
        return 1
    path = args.dataset
    if not path:
        cands = sorted(glob.glob(os.path.join(DS_DIR, "*.csv")))
        if not cands:
            print(f"No dataset in {DS_DIR}. Run tools/export_dataset.py first.")
            return 1
        path = cands[-1]
    train(path, epochs=args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
