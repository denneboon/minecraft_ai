#!/usr/bin/env python3
"""
Two-machine sync helper — move the gitignored runtime data between the
laptop (collects samples + runs the live bot) and the big PC (GPU-trains).
Code travels by git; THIS moves what git can't: the sample store and the
trained models. It packages each into a single zip so you can transfer it by
ANY method (LAN share, USB, Google Drive) — no network assumptions.

Typical loop:
    # laptop, after a collection session:
    python tools/sync.py export-samples         # -> data/_sync/world_samples.zip
    # ...move that one zip to the PC, then on the PC:
    python tools/sync.py import-samples data/_sync/world_samples.zip   # MERGES

    # PC, after GPU training:
    python tools/sync.py export-model           # -> data/_sync/models.zip
    # ...move it to the laptop, then on the laptop:
    python tools/sync.py import-model data/_sync/models.zip            # swaps in

Samples IMPORT is a content-safe MERGE: the store is content-addressed
(sha1-named files), so importing only ADDS files the target doesn't have and
never clobbers locally-collected samples. Models IMPORT overwrites the .pt
files (the freshly-trained one is the one you want), backing up the old one.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "data" / "training" / "world_samples"
CALIB_DIR = ROOT / "data" / "calibration"
SYNC_DIR = ROOT / "data" / "_sync"
# Model files worth syncing PC -> laptop (skip the big snapshots dir by default).
MODEL_FILES = ["block_fusion.pt", "block_cnn.pt"]


def _zip_dir(src: Path, dst_zip: Path, arc_root: str) -> int:
    n = 0
    dst_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dst_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in src.rglob("*"):
            if p.is_file():
                z.write(p, Path(arc_root) / p.relative_to(src))
                n += 1
    return n


def export_samples() -> int:
    if not SAMPLES_DIR.is_dir():
        print(f"[sync] no sample store at {SAMPLES_DIR}"); return 1
    out = SYNC_DIR / "world_samples.zip"
    n = _zip_dir(SAMPLES_DIR, out, "world_samples")
    print(f"[sync] exported {n} files -> {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print("[sync] move this zip to the other machine, then: "
          "python tools/sync.py import-samples <zip>")
    return 0


def import_samples(zip_path: str) -> int:
    zp = Path(zip_path)
    if not zp.is_file():
        print(f"[sync] no zip at {zp}"); return 1
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    added = skipped = 0
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zp) as z:
            z.extractall(td)
        base = Path(td) / "world_samples"
        if not base.is_dir():            # tolerate a zip made without the prefix
            base = Path(td)
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            dst = SAMPLES_DIR / p.relative_to(base)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():             # content-addressed name -> same content
                skipped += 1
            else:
                shutil.copy2(p, dst); added += 1
    # Drop the manifest so it rebuilds from the merged tree on next load.
    (SAMPLES_DIR / "_manifest.json").unlink(missing_ok=True)
    print(f"[sync] merged samples: +{added} new, {skipped} already present. "
          f"Manifest will rebuild on next load.")
    return 0


def export_model() -> int:
    out = SYNC_DIR / "models.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    present = [f for f in MODEL_FILES if (CALIB_DIR / f).is_file()]
    if not present:
        print(f"[sync] no model files {MODEL_FILES} under {CALIB_DIR}"); return 1
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in present:
            z.write(CALIB_DIR / f, f)
    print(f"[sync] exported {present} -> {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print("[sync] move it to the other machine, then: "
          "python tools/sync.py import-model <zip>")
    return 0


def import_model(zip_path: str) -> int:
    zp = Path(zip_path)
    if not zp.is_file():
        print(f"[sync] no zip at {zp}"); return 1
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    swapped = []
    with zipfile.ZipFile(zp) as z:
        for name in z.namelist():
            base = Path(name).name
            dst = CALIB_DIR / base
            if dst.exists():             # back up the old model before swapping
                shutil.copy2(dst, dst.with_suffix(dst.suffix + ".prev"))
            with z.open(name) as src, open(dst, "wb") as f:
                shutil.copyfileobj(src, f)
            swapped.append(base)
    print(f"[sync] imported model(s): {swapped} (old kept as *.prev)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export-samples")
    p_is = sub.add_parser("import-samples"); p_is.add_argument("zip")
    sub.add_parser("export-model")
    p_im = sub.add_parser("import-model"); p_im.add_argument("zip")
    args = ap.parse_args(argv)
    if args.cmd == "export-samples":
        return export_samples()
    if args.cmd == "import-samples":
        return import_samples(args.zip)
    if args.cmd == "export-model":
        return export_model()
    if args.cmd == "import-model":
        return import_model(args.zip)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
