#!/usr/bin/env python3
"""
Offline self-test for the context-fusion recogniser + futureproof feature set
(vision/world/context_fusion.py, context_features.py). Synthetic distinct
blocks so it trains fast on CPU. Skips cleanly if torch is unavailable.

Checks: the feature set encodes + handles missing/new values; the fusion model
trains, classifies, and degrades gracefully to visual-only when context is
absent; checkpoints round-trip; and a feature-set MISMATCH on load is detected
(not silently mis-fed).
"""
from __future__ import annotations

import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

import numpy as np
from types import SimpleNamespace

from vision.world.context_features import ContextFeatureSet, ContextFeature

_fail = 0
def ok(m): print(f"  [OK]  {m}")
def bad(m):
    global _fail; _fail += 1; print(f"  [BAD] {m}")


def _sample(color, bid, meta=None):
    rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    rgb[:] = color
    rgb = (rgb.astype(np.int16) + np.random.randint(-12, 12, rgb.shape)).clip(0, 255).astype(np.uint8)
    return SimpleNamespace(rgb=rgb, block_id=bid, metadata=meta)


def test_features():
    print("[1] futureproof feature set")
    fs = ContextFeatureSet()
    rich = {"face": "south", "biome": "minecraft:forest", "sky_brightness": 176,
            "weather": "clear", "pose": {"y": 63, "dimension": "minecraft:overworld"},
            "neighbors": {"py": "minecraft:air", "px": "minecraft:oak_log"}}
    v = fs.encode(rich); ve = fs.encode({}); vn = fs.encode(None)
    (ok if v.shape[0] == fs.total_dim else bad)(f"encodes to fixed width {fs.total_dim}")
    (ok if int(np.count_nonzero(ve)) == 0 else bad)("missing context -> all zeros (graceful)")
    (ok if np.array_equal(ve, vn) else bad)("None == empty metadata")
    vnew = fs.encode({**rich, "biome": "minecraft:some_future_biome_2030"})
    (ok if not np.array_equal(vnew, v) else bad)("never-seen biome hashes in (no code change)")


def test_fusion_train_classify():
    print("\n[2] fusion model trains + classifies + graceful + round-trips")
    from vision.world.context_fusion import ContextFusionRecognizer, ContextFusionConfig, _TORCH_OK
    if not _TORCH_OK:
        print("  (torch unavailable — skipping)"); return
    np.random.seed(0)
    COLORS = {"minecraft:red_block": (200, 30, 30),
              "minecraft:green_block": (30, 180, 60),
              "minecraft:blue_block": (40, 60, 200)}
    metas = {"minecraft:red_block": {"face": "top", "biome": "minecraft:desert"},
             "minecraft:green_block": {"face": "side", "biome": "minecraft:forest"},
             "minecraft:blue_block": {"face": "top", "biome": "minecraft:ocean"}}
    train = [_sample(c, b, metas[b]) for b, c in COLORS.items() for _ in range(20)]
    tmp = os.path.join(tempfile.gettempdir(), "_test_fusion.pt")
    if os.path.exists(tmp):
        os.remove(tmp)
    cfg = ContextFusionConfig(epochs=12, visual_width=8, embed_dim=32,
                              min_cosine=0.2, margin_min=0.0, model_path=tmp)
    rec = ContextFusionRecognizer(config=cfg)
    trained = rec.train_now(samples=train)
    (ok if trained else bad)(f"trains ({rec.status()})")

    # classify the training distribution (with context)
    hits = sum(rec.classify(_sample(c, b, metas[b]).rgb, metadata=metas[b])[0] == b
               for b, c in COLORS.items() for _ in range(10))
    (ok if hits >= 25 else bad)(f"classifies distinct blocks ({hits}/30 with context)")

    # graceful: classify with NO context must not crash + still mostly works
    g, _ = rec.classify(_sample(COLORS["minecraft:red_block"], "x").rgb, metadata=None)
    (ok if g is not None else bad)("classifies with metadata=None (visual-only, no crash)")

    # checkpoint round-trips
    rec2 = ContextFusionRecognizer(config=cfg)
    (ok if rec2.load(tmp) else bad)("checkpoint loads")
    g2, _ = rec2.classify(_sample(COLORS["minecraft:green_block"], "x").rgb,
                          metadata=metas["minecraft:green_block"])
    (ok if g2 == "minecraft:green_block" else bad)(f"loaded model classifies ({g2})")

    # feature-set MISMATCH is detected (not silently mis-fed)
    fs_diff = ContextFeatureSet(ContextFeatureSet().features[:-1])  # drop one feature
    rec3 = ContextFusionRecognizer(config=cfg, feature_set=fs_diff)
    (ok if not rec3.load(tmp) else bad)("feature-set mismatch on load is rejected")


def main() -> int:
    print("=" * 60); print(" context-fusion recogniser self-test"); print("=" * 60)
    test_features()
    test_fusion_train_classify()
    print("=" * 60)
    print(f" {_fail} CHECK(S) FAILED" if _fail else " ALL FUSION TESTS PASSED")
    return 1 if _fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
