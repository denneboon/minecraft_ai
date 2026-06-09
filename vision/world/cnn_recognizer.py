# vision/world/cnn_recognizer.py
"""
Self-teaching CNN block recogniser.

This is the learned successor to :class:`vision.world.sample_recognizer.
SampleBlockRecognizer` (raw-pixel L1 nearest-neighbour). It is a DROP-IN
replacement: same ``classify(patch_rgb) -> (block_id, confidence)`` and
``reload`` / ``reload_incremental`` surface, so it slots into
``HybridBlockClassifier`` / ``WorldPerception`` at the same seam.

Why a CNN instead of raw-pixel NN
---------------------------------
Raw 24x24 L1 distance is brittle: the SAME block looks very different at
dawn vs noon vs in shadow, at 2 m vs 12 m, head-on vs glancing, and under
biome grass/foliage tints. Pixel distance conflates all of that with the
block identity, so it confuses look-alikes and fails off the exact
capture conditions. A small convolutional network learns texture/colour
FEATURES that are far more invariant to those nuisances — and we train it
purely on the F3-auto-labelled samples the perception layer already
collects, so it teaches ITSELF with zero manual labelling.

Design: embedding + nearest-prototype (metric learning)
-------------------------------------------------------
* A tiny CNN maps a patch -> a 64-d L2-normalised EMBEDDING.
* Each block class has a PROTOTYPE = the mean embedding of its samples.
* ``classify`` embeds the query and picks the nearest prototype by cosine
  similarity; confidence comes from the margin to the runner-up class
  plus k-NN agreement among individual sample embeddings.
* A NEW block just adds a prototype — NO architecture change, no softmax
  resize. This is what makes it futureproof for an open, growing
  vocabulary discovered online.

Self-training loop
------------------
* The embedding net is trained with a throwaway classification head over
  whatever blocks currently have enough samples (cross-entropy), with
  heavy AUGMENTATION (brightness/contrast/hue/scale/rotation/flip/noise)
  so it generalises past the exact capture conditions.
* Training runs in a BACKGROUND thread so it never blocks the perception
  tick. When it finishes, the new weights + prototypes are swapped in
  atomically.
* ``reload_incremental`` (called by perception after each F3 confirm)
  cheaply embeds just the new samples and updates the prototypes/index;
  a full net retrain is triggered only every ``retrain_every_n`` new
  samples (or first time enough data exists), so the system keeps
  improving without thrashing the CPU.

Graceful degradation: if PyTorch is unavailable the recogniser disables
itself (``classify`` returns ``(None, 0.0)``) and the hybrid falls back
to the sample-NN / colour baseline, so importing this module can never
break a torch-less environment.
"""

from __future__ import annotations

import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from vision.world.sample_store import (
    SAMPLE_SIZE,
    StoredWorldSample,
    WorldSampleStore,
)

# ── Optional torch ─────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except Exception:                       # pragma: no cover - env without torch
    torch = None                        # type: ignore
    nn = None                           # type: ignore
    F = None                            # type: ignore
    _TORCH_OK = False


_DEFAULT_MODEL_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "data", "calibration", "block_cnn.pt",
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CNNBlockRecognizerConfig:
    embed_dim: int = 64
    input_size: int = 32          # patches are upsampled to this before the net
    # Confidence gating (cosine-sim based, [0,1]).
    min_cosine: float = 0.55      # best prototype cosine must clear this
    margin_min: float = 0.06      # cosine gap best vs best-different class
    top_k: int = 5                # k-NN agreement vote over sample embeddings
    min_confidence: float = 0.10
    # Training.
    min_blocks_to_train: int = 2          # need >= this many distinct blocks
    min_samples_per_block: int = 4        # a block needs >= this to be trainable
    retrain_every_n: int = 25             # new samples between background retrains
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1.5e-3
    weight_decay: float = 1e-4
    augment: bool = True
    model_path: str = _DEFAULT_MODEL_PATH
    # Keep CPU training bounded: hard cap on samples fed per retrain
    # (balanced across classes). Prevents a huge store from making a
    # retrain take minutes.
    max_train_samples: int = 6000


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

if _TORCH_OK:

    class _SmallBlockCNN(nn.Module):
        """Tiny conv net: patch -> L2-normalised embedding.

        ~0.1M params; a single 32x32 forward is well under a millisecond
        on CPU, and a batch of a few hundred (a whole-frame patch grid)
        is still tens of ms — fast enough for the perception sweep."""

        def __init__(self, embed_dim: int = 64):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),                                       # 32 -> 16
                nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),                                       # 16 -> 8
                nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),                                       # 8 -> 4
                nn.Conv2d(96, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),                              # -> 128 x 1 x 1
            )
            self.embed = nn.Linear(128, embed_dim)

        def forward(self, x):
            h = self.features(x).flatten(1)
            e = self.embed(h)
            return F.normalize(e, dim=1)        # unit-length embedding


# ---------------------------------------------------------------------------
# Augmentation (numpy/cv2 — no torchvision dependency)
# ---------------------------------------------------------------------------

def _augment(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random photometric + small geometric jitter on a HxWx3 uint8 patch.

    Simulates the real-world nuisances that break raw-pixel matching:
    day/night & shadow (brightness/contrast), biome tint (hue/sat),
    viewing angle/distance (rotate/scale), and sensor noise. The block
    IDENTITY is preserved, so this multiplies the effective training set
    and forces the net to key on texture rather than exact pixels."""
    import cv2
    img = rgb.astype(np.float32)
    h, w = img.shape[:2]

    # Geometric: small rotation + scale about centre, reflect border.
    if rng.random() < 0.8:
        ang = float(rng.uniform(-12.0, 12.0))
        scl = float(rng.uniform(0.85, 1.18))
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, scl)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.5:
        img = img[:, ::-1, :]                       # horizontal flip

    # Photometric: brightness, contrast.
    if rng.random() < 0.9:
        img *= float(rng.uniform(0.6, 1.4))         # brightness
    if rng.random() < 0.7:
        m = img.mean()
        img = (img - m) * float(rng.uniform(0.75, 1.3)) + m   # contrast

    # Hue / saturation jitter (biome tint) via HSV.
    if rng.random() < 0.6:
        img = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + float(rng.uniform(-10, 10))) % 180.0
        hsv[..., 1] *= float(rng.uniform(0.7, 1.3))
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)

    # Mild gaussian noise.
    if rng.random() < 0.4:
        img += rng.normal(0.0, 6.0, img.shape).astype(np.float32)

    return np.clip(img, 0, 255).astype(np.uint8)


def _to_tensor_batch(patches: List[np.ndarray], size: int):
    """Stack HxWx3 uint8 patches -> (B,3,size,size) float tensor in [0,1]."""
    import cv2
    arr = np.empty((len(patches), size, size, 3), dtype=np.float32)
    for i, p in enumerate(patches):
        if p.shape[0] != size or p.shape[1] != size:
            p = cv2.resize(p, (size, size), interpolation=cv2.INTER_AREA)
        arr[i] = p.astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    return t


# ---------------------------------------------------------------------------
# Recogniser
# ---------------------------------------------------------------------------

class CNNBlockRecognizer:
    """Learned embedding + nearest-prototype block recogniser.

    Mirrors :class:`SampleBlockRecognizer`'s public surface so it can be
    swapped in directly. ``available()`` reports whether torch is present
    AND a model has been trained — callers should treat an unavailable
    recogniser as 'no opinion' and fall back."""

    def __init__(self,
                 store: WorldSampleStore,
                 *,
                 config: Optional[CNNBlockRecognizerConfig] = None,
                 auto_train: bool = True):
        self._store = store
        self.cfg = config or CNNBlockRecognizerConfig()
        self._lock = threading.RLock()
        self._model = None                       # _SmallBlockCNN | None
        self._proto: Dict[str, np.ndarray] = {}  # block_id -> mean embedding
        self._emb: Optional[np.ndarray] = None   # (N, D) per-sample embeddings
        self._emb_ids: List[str] = []
        self._samples: List[StoredWorldSample] = []
        self._trained = False
        self._training = False
        self._saves_since_retrain = 0
        self._auto_train = auto_train
        self._train_thread: Optional[threading.Thread] = None
        self._epoch_note = ""
        if _TORCH_OK:
            try:
                torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
            except Exception:
                pass
            self._try_load_model()
        self.reload()

    # ── Availability ───────────────────────────────────────────────

    def available(self) -> bool:
        return _TORCH_OK and self._trained and len(self._proto) >= 2

    # ── Public API (matches SampleBlockRecognizer) ─────────────────

    def reload(self) -> None:
        with self._lock:
            self._samples = self._store.load_all()
        self._rebuild_index()
        self._maybe_kickoff_training(reason="reload")

    def reload_incremental(self) -> None:
        with self._lock:
            known = {s.path for s in self._samples}
        new = self._store.load_all(skip_paths=known)
        if not new:
            return
        with self._lock:
            self._samples.extend(new)
            self._saves_since_retrain += len(new)
        # Cheap: embed just the new samples and extend the index/prototypes
        # so the recogniser reflects them immediately, without a retrain.
        if self._model is not None and self._trained:
            self._embed_into_index(new)
        self._maybe_kickoff_training(reason="incremental")

    def sample_count(self) -> int:
        return len(self._samples)

    def block_count(self) -> int:
        return len(set(s.block_id for s in self._samples))

    def template_count(self) -> int:
        return len(self._proto)

    def count_for(self, block_id: str) -> int:
        return sum(1 for s in self._samples if s.block_id == block_id)

    def classify(self, patch_rgb: np.ndarray) -> Tuple[Optional[str], float]:
        """``(block_id, confidence)`` or ``(None, conf)`` if unsure."""
        if not self.available() or patch_rgb is None or patch_rgb.size == 0:
            return None, 0.0
        rgb = patch_rgb
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[..., :3]
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return None, 0.0
        emb = self._embed_one(rgb)
        if emb is None:
            return None, 0.0
        with self._lock:
            proto = self._proto
            emb_mat = self._emb
            emb_ids = self._emb_ids
        if len(proto) < 2:
            return None, 0.0

        # Nearest prototype by cosine (embeddings are unit-norm, so cosine
        # is just a dot product).
        ids = list(proto.keys())
        P = np.stack([proto[k] for k in ids], axis=0)       # (C, D)
        sims = P @ emb                                       # (C,)
        order = np.argsort(-sims)
        best_id = ids[int(order[0])]
        best_sim = float(sims[int(order[0])])
        second_sim = float(sims[int(order[1])]) if len(order) > 1 else -1.0
        margin = best_sim - second_sim

        if best_sim < self.cfg.min_cosine or margin < self.cfg.margin_min:
            return None, max(0.0, best_sim)

        # k-NN agreement over individual sample embeddings (extra robustness:
        # a prototype can be dragged by an outlier; the neighbours vote).
        agreement = 1.0
        if emb_mat is not None and len(emb_ids) >= self.cfg.top_k:
            ssims = emb_mat @ emb                            # (N,)
            knn = np.argsort(-ssims)[: self.cfg.top_k]
            votes = Counter(emb_ids[int(i)] for i in knn)
            agreement = votes[best_id] / float(self.cfg.top_k)

        norm_margin = min(1.0, margin / 0.30)
        conf = max(0.0, min(1.0, 0.5 * agreement + 0.3 * norm_margin
                            + 0.2 * max(0.0, (best_sim - self.cfg.min_cosine)
                                        / (1.0 - self.cfg.min_cosine))))
        if conf < self.cfg.min_confidence:
            return None, conf
        return best_id, conf

    def status(self) -> str:
        return (f"cnn(trained={self._trained}, training={self._training}, "
                f"blocks={len(self._proto)}, samples={len(self._samples)}"
                f"{', ' + self._epoch_note if self._epoch_note else ''})")

    # ── Embedding helpers ──────────────────────────────────────────

    def _embed_one(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        if self._model is None:
            return None
        try:
            with torch.no_grad():
                t = _to_tensor_batch([rgb], self.cfg.input_size)
                with self._lock:
                    self._model.eval()
                    e = self._model(t).cpu().numpy()[0]
            return e.astype(np.float32)
        except Exception:
            return None

    def _embed_batch(self, patches: List[np.ndarray]) -> Optional[np.ndarray]:
        if self._model is None or not patches:
            return None
        try:
            out = []
            with torch.no_grad():
                self._model.eval()
                for i in range(0, len(patches), 256):
                    t = _to_tensor_batch(patches[i:i + 256], self.cfg.input_size)
                    out.append(self._model(t).cpu().numpy())
            return np.concatenate(out, axis=0).astype(np.float32)
        except Exception:
            return None

    def _rebuild_index(self) -> None:
        """Re-embed ALL samples and recompute prototypes (after a retrain
        or a fresh model load)."""
        if self._model is None or not self._trained or not self._samples:
            with self._lock:
                self._emb = None
                self._emb_ids = []
                self._proto = {}
            return
        with self._lock:
            samples = list(self._samples)
        emb = self._embed_batch([s.rgb for s in samples])
        if emb is None:
            return
        ids = [s.block_id for s in samples]
        proto: Dict[str, np.ndarray] = {}
        for bid in set(ids):
            mask = np.array([i == bid for i in ids])
            m = emb[mask].mean(axis=0)
            n = np.linalg.norm(m)
            proto[bid] = (m / n).astype(np.float32) if n > 1e-6 else m.astype(np.float32)
        with self._lock:
            self._emb = emb
            self._emb_ids = ids
            self._proto = proto

    def _embed_into_index(self, new: List[StoredWorldSample]) -> None:
        """Append new sample embeddings and refresh affected prototypes —
        O(new), used on the incremental hot-reload path."""
        emb = self._embed_batch([s.rgb for s in new])
        if emb is None:
            return
        ids = [s.block_id for s in new]
        with self._lock:
            self._emb = emb if self._emb is None else np.concatenate([self._emb, emb], axis=0)
            self._emb_ids.extend(ids)
            # Recompute prototypes for the affected ids from the full index.
            all_emb, all_ids = self._emb, self._emb_ids
            for bid in set(ids):
                mask = np.array([i == bid for i in all_ids])
                m = all_emb[mask].mean(axis=0)
                n = np.linalg.norm(m)
                self._proto[bid] = (m / n).astype(np.float32) if n > 1e-6 else m.astype(np.float32)

    # ── Training ───────────────────────────────────────────────────

    def _trainable_classes(self) -> List[str]:
        counts = Counter(s.block_id for s in self._samples)
        return [b for b, c in counts.items()
                if c >= self.cfg.min_samples_per_block]

    def _maybe_kickoff_training(self, *, reason: str) -> None:
        if not (_TORCH_OK and self._auto_train) or self._training:
            return
        classes = self._trainable_classes()
        if len(classes) < self.cfg.min_blocks_to_train:
            return
        # Train if we've never trained, or enough new samples accumulated.
        need = (not self._trained
                or self._saves_since_retrain >= self.cfg.retrain_every_n)
        if not need:
            return
        self._saves_since_retrain = 0
        self._train_thread = threading.Thread(
            target=self._train_worker, args=(reason,),
            name="block-cnn-train", daemon=True)
        self._training = True
        self._train_thread.start()

    def _train_worker(self, reason: str) -> None:
        try:
            self._train_blocking()
        except Exception as e:                          # never crash perception
            self._epoch_note = f"train-error: {e!r}"
        finally:
            self._training = False

    def train_now(self) -> bool:
        """Synchronous train (used by tools/tests). Returns True if trained."""
        if not _TORCH_OK:
            return False
        if len(self._trainable_classes()) < self.cfg.min_blocks_to_train:
            return False
        self._training = True
        try:
            self._train_blocking()
            return self._trained
        finally:
            self._training = False

    def _train_blocking(self) -> None:
        cfg = self.cfg
        with self._lock:
            samples = list(self._samples)
        classes = sorted(set(
            b for b, c in Counter(s.block_id for s in samples).items()
            if c >= cfg.min_samples_per_block))
        if len(classes) < cfg.min_blocks_to_train:
            return
        cls_idx = {b: i for i, b in enumerate(classes)}
        # Class-balanced sample list, capped for bounded CPU time.
        by_cls: Dict[str, List[np.ndarray]] = {b: [] for b in classes}
        for s in samples:
            if s.block_id in by_cls:
                by_cls[s.block_id].append(s.rgb)
        per_cap = max(1, cfg.max_train_samples // max(1, len(classes)))
        rng = np.random.default_rng(1234)
        train_imgs: List[np.ndarray] = []
        train_lbls: List[int] = []
        for b in classes:
            imgs = by_cls[b]
            if len(imgs) > per_cap:
                idx = rng.choice(len(imgs), per_cap, replace=False)
                imgs = [imgs[i] for i in idx]
            for im in imgs:
                train_imgs.append(im)
                train_lbls.append(cls_idx[b])
        n = len(train_imgs)
        if n < cfg.batch_size:
            cfg_bs = max(2, n)
        else:
            cfg_bs = cfg.batch_size

        model = _SmallBlockCNN(cfg.embed_dim)
        # A throwaway classification head trains the embedding via
        # cross-entropy; we keep only the embedding afterwards.
        head = nn.Linear(cfg.embed_dim, len(classes))
        params = list(model.parameters()) + list(head.parameters())
        opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        lbls = np.asarray(train_lbls, dtype=np.int64)
        t0 = time.perf_counter()
        model.train(); head.train()
        for epoch in range(cfg.epochs):
            perm = rng.permutation(n)
            total = 0.0
            for bi in range(0, n, cfg_bs):
                sel = perm[bi:bi + cfg_bs]
                batch = []
                for j in sel:
                    im = train_imgs[j]
                    batch.append(_augment(im, rng) if cfg.augment else im)
                x = _to_tensor_batch(batch, cfg.input_size)
                y = torch.from_numpy(lbls[sel])
                opt.zero_grad()
                emb = model(x)
                logits = head(emb)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(sel)
            self._epoch_note = (f"epoch {epoch + 1}/{cfg.epochs} "
                                f"loss={total / n:.3f} ({len(classes)} blocks)")
        dt = time.perf_counter() - t0

        # Publish the trained embedding + rebuild prototypes/index.
        model.eval()
        with self._lock:
            self._model = model
            self._trained = True
        self._rebuild_index()
        self._epoch_note = (f"trained {len(classes)} blocks on {n} samples "
                            f"in {dt:.1f}s")
        self._save_model(classes)

    # ── Persistence ────────────────────────────────────────────────

    def _save_model(self, classes: List[str]) -> None:
        if self._model is None:
            return
        try:
            os.makedirs(os.path.dirname(self.cfg.model_path), exist_ok=True)
            torch.save({
                "state_dict": self._model.state_dict(),
                "embed_dim": self.cfg.embed_dim,
                "input_size": self.cfg.input_size,
                "classes": classes,
                "version": 1,
            }, self.cfg.model_path)
        except Exception as e:
            self._epoch_note = f"save-error: {e!r}"

    def _try_load_model(self) -> None:
        path = self.cfg.model_path
        if not (path and os.path.isfile(path)):
            return
        try:
            # weights_only=False: our own checkpoint carries the class
            # list (plain str) alongside the state_dict; it's a file we
            # wrote, so this is safe and silences the future-default warning.
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model = _SmallBlockCNN(int(ckpt.get("embed_dim", self.cfg.embed_dim)))
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self._model = model
            self._trained = True
        except Exception as e:
            self._epoch_note = f"load-error: {e!r}"
            self._model = None
            self._trained = False


# ---------------------------------------------------------------------------
# Tiered classifier: CNN -> sample-NN -> colour baseline
# ---------------------------------------------------------------------------

class TieredBlockClassifier:
    """Try the learned CNN first, then the raw-pixel sample-NN, then the
    colour-signature baseline — returning the first confident answer.

    Implements the ``BlockClassifierProtocol`` surface
    (``classify`` / ``template_count`` / ``reload_samples``) so it drops
    into ``WorldPerception`` exactly like ``HybridBlockClassifier``.

    Rationale for the order: the CNN is the most robust once trained, but
    early on (few samples / not yet trained) it reports ``available()``
    False and is skipped; the sample-NN covers the near-identical-capture
    case the CNN might still be unsure on; the baseline guarantees an
    answer so perception's curiosity queue always has something to chew
    on. All three improve as the F3-labelled sample store grows."""

    def __init__(self, cnn, sample_recognizer, baseline_classifier):
        self.cnn = cnn
        self.sample = sample_recognizer
        self.baseline = baseline_classifier

    def classify(self, patch_rgb) -> Tuple[Optional[str], float]:
        if self.cnn is not None and self.cnn.available():
            bid, conf = self.cnn.classify(patch_rgb)
            if bid is not None and conf >= self.cnn.cfg.min_confidence:
                return bid, conf
        if self.sample is not None:
            bid, conf = self.sample.classify(patch_rgb)
            if bid is not None and conf >= self.sample.cfg.min_confidence:
                return bid, conf
        return self.baseline.classify(patch_rgb)

    def template_count(self) -> int:
        n = self.baseline.template_count()
        if self.cnn is not None:
            n += self.cnn.template_count()
        if self.sample is not None:
            n += self.sample.block_count()
        return n

    def reload_samples(self) -> None:
        """Refresh both learned tiers after new F3-labelled samples land.
        The CNN's incremental reload also kicks off a background retrain
        once enough new data has accumulated."""
        if self.cnn is not None:
            try:
                self.cnn.reload_incremental()
            except Exception:
                pass
        if self.sample is not None:
            try:
                self.sample.reload_incremental()
            except Exception:
                pass

    def status(self) -> str:
        s = self.cnn.status() if self.cnn is not None else "cnn(off)"
        return s


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_cnn_block_recognizer(
        store: Optional[WorldSampleStore] = None,
        *,
        config: Optional[CNNBlockRecognizerConfig] = None,
        auto_train: bool = True,
        ) -> CNNBlockRecognizer:
    if store is None:
        from vision.world.sample_store import build_world_sample_store
        store = build_world_sample_store()
    return CNNBlockRecognizer(store, config=config, auto_train=auto_train)


__all__ = [
    "CNNBlockRecognizer",
    "CNNBlockRecognizerConfig",
    "build_cnn_block_recognizer",
]
