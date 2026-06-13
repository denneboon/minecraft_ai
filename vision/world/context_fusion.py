"""
Context-fusion block recogniser — the "best possible" model the user asked
for: fuse the VISUAL patch with as many CONTEXT inputs as possible
(environment, geometry, the confirmed-neighbour map) and stay futureproof on
BOTH axes — inputs and vocabulary.

Architecture (two branches -> fuse -> metric-learning embedding):

    patch (HxWx3) ─► visual encoder (conv) ─► visual_emb ─┐
                                                          ├─► fuse MLP ─► emb (L2)
    context vector (ContextFeatureSet) ─► context MLP ───►┘                │
                                                                           ▼
                                            nearest class PROTOTYPE (cosine) + kNN

Futureproofing:
  * INPUTS  — the context width comes from ``ContextFeatureSet``; appending a
    feature there just widens the context branch (retrain), no code change.
    Missing context degrades to zeros, so with NO context this reduces to the
    visual recogniser (drop-in compatible).
  * VOCABULARY — classification is nearest-PROTOTYPE over the embedding (not a
    fixed softmax), so a new block is just a new prototype: no architecture
    change. A throwaway head trains the embedding via cross-entropy.
  * VISUAL CAPACITY — the encoder ``width``/``depth``/``embed_dim`` are config,
    so the same code trains a tiny CPU net or a big-PC model.
  * CHECKPOINT — records the feature-set signature + config, so a load detects
    a feature-set mismatch instead of silently mis-feeding the net.

Graceful no-torch import (mirrors cnn_recognizer): the module imports fine and
exposes ``_TORCH_OK``; the recogniser is only constructed when torch is present.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from vision.world.context_features import ContextFeatureSet

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except Exception:  # pragma: no cover - torch optional
    _TORCH_OK = False

_DEFAULT_MODEL_PATH = os.path.join("data", "calibration", "block_fusion.pt")


@dataclass
class ContextFusionConfig:
    input_size: int = 32          # patch upsampled to this
    visual_width: int = 32        # base conv channels (x2,x4,x8 deeper)
    visual_emb: int = 128         # visual branch output
    context_hidden: int = 64
    context_emb: int = 48
    embed_dim: int = 96           # fused embedding (the metric space)
    # Confidence gating (cosine-based, [0,1]) — same shape as cnn_recognizer.
    min_cosine: float = 0.55
    margin_min: float = 0.06
    top_k: int = 5
    # Training.
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1.5e-3
    weight_decay: float = 1e-4
    min_samples_per_block: int = 4
    min_blocks_to_train: int = 2
    max_train_samples: int = 20000
    augment: bool = True
    model_path: str = _DEFAULT_MODEL_PATH
    seed: int = 1234


if _TORCH_OK:

    class _VisualEncoder(nn.Module):
        """Configurable conv tower patch -> visual_emb (L2-normalised). Width
        scales the channel count so the same code is a tiny CPU net or a big
        model on a GPU."""

        def __init__(self, width: int, emb: int):
            super().__init__()
            w = width
            self.features = nn.Sequential(
                nn.Conv2d(3, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(w, w * 2, 3, padding=1), nn.BatchNorm2d(w * 2), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(w * 2, w * 4, 3, padding=1), nn.BatchNorm2d(w * 4), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(w * 4, w * 8, 3, padding=1), nn.BatchNorm2d(w * 8), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.proj = nn.Linear(w * 8, emb)

        def forward(self, x):
            h = self.features(x).flatten(1)
            return F.normalize(self.proj(h), dim=1)

    class _FusionNet(nn.Module):
        """Visual + context -> fused L2 embedding (+ a throwaway class head used
        only during training to shape the embedding)."""

        def __init__(self, cfg: "ContextFusionConfig", ctx_dim: int, n_classes: int):
            super().__init__()
            self.visual = _VisualEncoder(cfg.visual_width, cfg.visual_emb)
            self.context = nn.Sequential(
                nn.Linear(ctx_dim, cfg.context_hidden), nn.ReLU(inplace=True),
                nn.Linear(cfg.context_hidden, cfg.context_emb), nn.ReLU(inplace=True),
            )
            self.fuse = nn.Sequential(
                nn.Linear(cfg.visual_emb + cfg.context_emb, cfg.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(cfg.embed_dim, cfg.embed_dim),
            )
            self.head = nn.Linear(cfg.embed_dim, n_classes)

        def embed(self, patch, ctx):
            v = self.visual(patch)
            c = self.context(ctx)
            e = self.fuse(torch.cat([v, c], dim=1))
            return F.normalize(e, dim=1)

        def forward(self, patch, ctx):
            return self.head(self.embed(patch, ctx))


class ContextFusionRecognizer:
    """Train + classify with the fusion model. Mirrors CNNBlockRecognizer's
    surface (``classify``/``train_now``/``status``/save+load) but ``classify``
    also accepts the sample ``metadata`` so it can use context; with
    ``metadata=None`` it falls back to a zero context vector (visual-only),
    so it's a drop-in for the visual recogniser."""

    def __init__(self, sample_store=None,
                 config: Optional[ContextFusionConfig] = None,
                 feature_set: Optional[ContextFeatureSet] = None):
        self.cfg = config or ContextFusionConfig()
        self.features = feature_set or ContextFeatureSet()
        self.store = sample_store
        # Use the GPU when present (the whole point of training on a big PC);
        # falls back to CPU transparently. Set once at construction.
        self._device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        if _TORCH_OK else None)
        self._model = None
        self._classes: List[str] = []
        self._proto: Dict[str, np.ndarray] = {}      # class -> mean embedding
        self._emb: Optional[np.ndarray] = None        # (N, D) per-sample
        self._emb_ids: List[str] = []
        self._trained = False
        self._note = ""

    # ── helpers ──────────────────────────────────────────────────────────
    def _ctx_dim(self) -> int:
        return self.features.total_dim

    def _prep_patch(self, rgb: np.ndarray):
        import cv2
        p = cv2.resize(rgb[..., :3], (self.cfg.input_size, self.cfg.input_size),
                       interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(p.astype(np.float32) / 255.0).permute(2, 0, 1)
        return t

    # ── training ─────────────────────────────────────────────────────────
    def train_now(self, samples: Optional[List] = None) -> bool:
        """Train on ``samples`` (each must expose ``.rgb``, ``.block_id`` and
        optionally ``.metadata``); falls back to the bound store's
        ``load_all()``. Returns True if a model was trained."""
        if not _TORCH_OK:
            self._note = "torch unavailable"
            return False
        if samples is None and self.store is not None:
            samples = self.store.load_all()
        samples = list(samples or [])
        from collections import Counter
        counts = Counter(s.block_id for s in samples)
        classes = sorted(b for b, c in counts.items()
                         if c >= self.cfg.min_samples_per_block)
        if len(classes) < self.cfg.min_blocks_to_train:
            self._note = f"need >= {self.cfg.min_blocks_to_train} trainable blocks"
            return False
        cls_idx = {b: i for i, b in enumerate(classes)}
        sel = [s for s in samples if s.block_id in cls_idx][: self.cfg.max_train_samples]

        torch.manual_seed(self.cfg.seed)
        rng = np.random.default_rng(self.cfg.seed)
        from vision.world.cnn_recognizer import _augment  # reuse the same aug

        ctx_dim = self._ctx_dim()
        model = _FusionNet(self.cfg, ctx_dim, len(classes)).to(self._device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.cfg.lr,
                               weight_decay=self.cfg.weight_decay)
        # Pre-encode context once per sample (cheap, deterministic).
        meta = [getattr(s, "metadata", None) for s in sel]
        ctx_all = np.stack([self.features.encode(m) for m in meta]).astype(np.float32)
        y_all = np.array([cls_idx[s.block_id] for s in sel], dtype=np.int64)

        n = len(sel)
        idx = np.arange(n)
        for _ in range(self.cfg.epochs):
            rng.shuffle(idx)
            for i in range(0, n, self.cfg.batch_size):
                bi = idx[i: i + self.cfg.batch_size]
                patches = []
                for j in bi:
                    rgb = _augment(sel[j].rgb, rng) if self.cfg.augment else sel[j].rgb
                    patches.append(self._prep_patch(rgb))
                pb = torch.stack(patches).to(self._device)
                cb = torch.from_numpy(ctx_all[bi]).to(self._device)
                yb = torch.from_numpy(y_all[bi]).to(self._device)
                opt.zero_grad()
                loss = F.cross_entropy(model(pb, cb), yb)
                loss.backward()
                opt.step()

        # Build prototypes (mean embedding per class) for vocabulary-free
        # nearest-prototype inference.
        model.eval()
        embs = []
        with torch.no_grad():
            for i in range(0, n, 256):
                pb = torch.stack([self._prep_patch(sel[j].rgb)
                                  for j in range(i, min(i + 256, n))]).to(self._device)
                cb = torch.from_numpy(ctx_all[i: i + 256]).to(self._device)
                embs.append(model.embed(pb, cb).cpu().numpy())
        E = np.concatenate(embs).astype(np.float32)
        self._emb = E
        self._emb_ids = [sel[j].block_id for j in range(n)]
        proto = {}
        for b in classes:
            m = E[[k for k in range(n) if self._emb_ids[k] == b]]
            v = m.mean(axis=0)
            nrm = np.linalg.norm(v)
            if np.isfinite(nrm) and nrm > 1e-6:
                proto[b] = (v / nrm).astype(np.float32)
        self._model = model
        self._classes = classes
        self._proto = proto
        self._trained = True
        self._note = (f"fused {len(classes)} blocks on {n} samples "
                      f"(ctx_dim={ctx_dim}, emb={self.cfg.embed_dim})")
        self._save()
        return True

    # ── inference ────────────────────────────────────────────────────────
    def classify(self, patch_rgb: np.ndarray,
                 metadata: Optional[Dict] = None) -> Tuple[Optional[str], float]:
        if not (_TORCH_OK and self._trained and self._model is not None and self._proto):
            return None, 0.0
        with torch.no_grad():
            pb = self._prep_patch(patch_rgb).unsqueeze(0).to(self._device)
            cb = torch.from_numpy(
                self.features.encode(metadata).reshape(1, -1)).to(self._device)
            e = self._model.embed(pb, cb).cpu().numpy()[0]
        # cosine to each prototype
        best_id, best, second = None, -1.0, -1.0
        for bid, p in self._proto.items():
            cos = float(np.dot(e, p))
            if cos > best:
                second = best; best = cos; best_id = bid
            elif cos > second:
                second = cos
        if best < self.cfg.min_cosine or (best - max(second, 0.0)) < self.cfg.margin_min:
            return None, max(0.0, best)
        return best_id, best

    def status(self) -> str:
        dev = str(self._device) if self._device is not None else "none"
        return (f"fusion(trained={self._trained}, blocks={len(self._classes)}, "
                f"ctx={self._ctx_dim()}, device={dev}, {self._note})")

    # ── persistence (versioned) ──────────────────────────────────────────
    def _save(self) -> None:
        if not (_TORCH_OK and self._model is not None):
            return
        try:
            os.makedirs(os.path.dirname(self.cfg.model_path), exist_ok=True)
            torch.save({
                "version": 1,
                "state_dict": self._model.state_dict(),
                "classes": list(self._classes),
                "proto": {k: v.tolist() for k, v in self._proto.items()},
                "feature_signature": self.features.signature,
                "cfg": self.cfg.__dict__,
            }, self.cfg.model_path)
        except Exception as e:
            self._note = f"save-error: {e!r}"

    def load(self, path: Optional[str] = None) -> bool:
        if not _TORCH_OK:
            return False
        path = path or self.cfg.model_path
        if not os.path.isfile(path):
            return False
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            if ck.get("feature_signature") != self.features.signature:
                self._note = ("feature-set mismatch: model was trained with a "
                              "different context set; retrain")
                return False
            classes = ck["classes"]
            model = _FusionNet(self.cfg, self._ctx_dim(), len(classes))
            model.load_state_dict(ck["state_dict"])
            model.to(self._device); model.eval()
            self._model = model
            self._classes = classes
            self._proto = {k: np.asarray(v, dtype=np.float32)
                           for k, v in (ck.get("proto") or {}).items()}
            self._trained = True
            return True
        except Exception as e:
            self._note = f"load-error: {e!r}"
            return False


__all__ = ["ContextFusionRecognizer", "ContextFusionConfig", "_TORCH_OK"]
