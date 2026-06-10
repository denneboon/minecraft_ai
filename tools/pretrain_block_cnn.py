#!/usr/bin/env python3
"""
Pre-train the block-recognition CNN from the GAME TEXTURES.

Solves the cold-start problem: a fresh world has no F3-collected samples,
so the learned recogniser is blind until the player has held F3 on many
blocks. Here we bootstrap it from Minecraft's own block textures —
generating augmented synthetic patches (lighting / hue / scale / rotation
/ blur / noise simulating how a block actually renders in-world) and
training the embedding on them. The resulting per-block prototypes become
the recogniser's cold-start set (it can NAME 100s-1000s of blocks before
any real capture), and the embedding is the WARM-START for later online
fine-tuning on the player's real F3-labelled samples.

This writes the same model file the runtime recogniser loads
(data/calibration/block_cnn.pt), with a ``texture_proto`` cold-start map.
Online F3 training later warm-starts from this embedding and overrides
prototypes per-block as real samples arrive.

Run:
    python tools/pretrain_block_cnn.py                 # defaults
    python tools/pretrain_block_cnn.py --patches 16 --epochs 25
    python tools/pretrain_block_cnn.py --max-blocks 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from vision.world.sample_store import SAMPLE_SIZE, StoredWorldSample
from vision.world.cnn_recognizer import (
    CNNBlockRecognizer, CNNBlockRecognizerConfig, _TORCH_OK, _augment,
)

_FACE_SUFFIXES = (
    "_top", "_side", "_bottom", "_front", "_back",
    "_inner", "_inner_top", "_inner_bottom",
    "_side_overlay", "_top_overlay", "_outside", "_inside",
    "_pivot", "_round",
)


def _block_id_for_stem(stem: str) -> str:
    base = stem
    for suf in sorted(_FACE_SUFFIXES, key=len, reverse=True):
        if stem.endswith(suf):
            base = stem[: -len(suf)]
            break
    return f"minecraft:{base}" if base else f"minecraft:{stem}"


def _usable_texture(tex: np.ndarray) -> bool:
    """Skip overlays / transparent / near-blank textures — they aren't a
    solid block face and would pollute the embedding."""
    if tex is None or tex.ndim != 3 or tex.shape[2] < 3:
        return False
    rgb = tex[..., :3].astype(np.float32)
    # Mostly-transparent (if alpha present) or near-zero-variance overlays.
    if tex.shape[2] == 4:
        if (tex[..., 3] > 16).mean() < 0.5:
            return False
    if rgb.std() < 6.0:
        return False
    return True


def _gather_textures(assets, max_blocks: int):
    """block_id -> list of ``(face_stem, RGB array)`` (collapsed by base).

    Keeps the face stem so the sample generator knows WHICH face each
    texture is — needed to tint ``grass_block_top`` (green) without
    tinting ``grass_block_side`` (dirt)."""
    by_block = {}
    for stem in assets.list_block_textures():
        if stem.endswith("_overlay"):
            continue
        tex = assets.block_texture(stem)
        if not _usable_texture(tex):
            continue
        rgb = tex[..., :3] if tex.shape[2] == 4 else tex
        bid = _block_id_for_stem(stem)
        by_block.setdefault(bid, []).append((stem, np.ascontiguousarray(rgb)))
    if max_blocks and len(by_block) > max_blocks:
        # Keep a deterministic subset (sorted) so reruns are stable.
        keep = sorted(by_block)[:max_blocks]
        by_block = {k: by_block[k] for k in keep}
    return by_block


def _face_is_tinted(block_base: str, face_stem: str) -> bool:
    """Does Minecraft apply a biome/fixed tint to THIS face of THIS block?

    Plants / leaves / vines are single-texture and fully tinted. The one
    common partial case is ``grass_block``: only the TOP (and the side
    grass overlay, which we skip) is tinted; the ``_side`` face is plain
    dirt and must stay untinted."""
    from vision.world.biome_tint import is_tintable
    if not is_tintable(block_base):
        return False
    if block_base == "grass_block":
        return face_stem == "grass_block_top"
    return True


def _make_samples(by_block, patches_per_block: int, seed: int = 0,
                  tinted: bool = True):
    import cv2
    from vision.world.biome_tint import tints_for_block, apply_tint
    rng = np.random.default_rng(seed)
    samples = []
    for bid, faces in by_block.items():
        block_base = bid.split(":", 1)[-1]
        tints = tints_for_block(block_base) if tinted else []
        for k in range(patches_per_block):
            face_stem, face = faces[rng.integers(len(faces))]
            patch = cv2.resize(face, (SAMPLE_SIZE, SAMPLE_SIZE),
                               interpolation=cv2.INTER_NEAREST)
            # Apply a random biome tint so grass/foliage prototypes are
            # GREEN like the real render, not grey like the raw atlas.
            # One untinted patch per block is kept too (k == 0) for
            # robustness to odd lighting / shaders that wash out tint.
            if tints and k > 0 and _face_is_tinted(block_base, face_stem):
                _, tint = tints[int(rng.integers(len(tints)))]
                patch = apply_tint(patch, tint)
            patch = _augment(patch, rng)        # render-condition jitter
            samples.append(StoredWorldSample(
                block_id=bid, path=Path(f"_tex/{bid}/{k}"), rgb=patch))
    rng.shuffle(samples)
    return samples


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patches", type=int, default=14,
                    help="Augmented synthetic patches generated per block.")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--max-blocks", type=int, default=0,
                    help="Cap distinct blocks (0 = all block-face textures).")
    args = ap.parse_args(argv)

    print("=" * 64)
    print(" Pre-train block CNN from game textures")
    print("=" * 64)
    if not _TORCH_OK:
        print(" PyTorch not available — cannot pre-train.")
        return 1

    from vision.mc_assets import MCAssets
    assets = MCAssets.load()
    by_block = _gather_textures(assets, args.max_blocks)
    print(f"  textures: {len(by_block)} blocks (block-face, de-overlayed)")
    if len(by_block) < 2:
        print("  not enough block textures found.")
        return 1
    samples = _make_samples(by_block, args.patches)
    print(f"  synthetic patches: {len(samples)} "
          f"({args.patches}/block, augmented)")

    cfg = CNNBlockRecognizerConfig(
        epochs=args.epochs,
        min_samples_per_block=max(2, args.patches // 2),
        max_train_samples=max(20000, len(samples)),
    )
    rec = CNNBlockRecognizer(_empty_store(), config=cfg, auto_train=False)
    print(f"  training embedding ({args.epochs} epochs)…")
    okp = rec.pretrain_textures(samples)
    print(f"  {rec.status()}")
    if not okp:
        print("  pre-train FAILED.")
        return 1
    print(f"  saved cold-start model + {len(rec._texture_proto)} texture "
          f"prototypes -> {cfg.model_path}")
    print("=" * 64)
    print(" DONE — runtime recogniser will load this on next start.")
    return 0


def _empty_store():
    from vision.world.sample_store import build_world_sample_store
    return build_world_sample_store()


if __name__ == "__main__":
    raise SystemExit(main())
