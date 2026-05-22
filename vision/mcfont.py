# vision/mcfont.py
"""
Minecraft default ASCII font extractor + cache.

Minecraft renders F3 (and all default UI text) using a small bitmap font
stored at ``assets/minecraft/textures/font/ascii.png`` inside the game JAR.
The bitmap is a 128×128 atlas of 16×16 cells of 8×8 px glyphs. The character
layout for each cell is defined in ``assets/minecraft/font/include/default.json``.

This module:
  - Finds an installed MC JAR on the user's system.
  - Extracts the ASCII font atlas.
  - Trims each glyph to its actual rendered width (Mojang's renderer does
    the same — the right-most non-empty column + 1 is the glyph width).
  - Caches the resulting templates as a compressed ``.npz`` under
    ``data/calibration/mc_font.npz`` so subsequent runs don't need to
    open the JAR.

The templates are stored at GUI scale 1; callers should upscale with
``INTER_NEAREST`` to match the current ``capture.ui_scale``.

This is the precondition for ``vision/glyph_ocr.py``, which does
pixel-perfect template matching on captured F3 text — far more reliable
than running Tesseract LSTM on a bitmap font.
"""

from __future__ import annotations

import glob
import io
import json
import os
import platform
import re
import zipfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image
except ImportError as e:
    raise RuntimeError(
        "Pillow is required to extract MC font glyphs. "
        "Install with: pip install Pillow"
    ) from e


# ---------------------------------------------------------------------------
# JAR discovery
# ---------------------------------------------------------------------------

_PURE_VERSION_RE = re.compile(r"^(\d+)(?:\.\d+)*$")
_VERSION_RUN_RE  = re.compile(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?")

# Mod-loader jars live in the same versions/ directories but never contain
# the vanilla ASCII font assets. Skip them so we don't try to extract from
# an empty jar before falling through to a real client jar.
_LOADER_TOKENS = ("fabric", "forge", "neoforge", "optifine", "quilt",
                  "liteloader", "rift", "loader")


def _is_loader_name(name: str) -> bool:
    n = name.lower()
    return any(tok in n for tok in _LOADER_TOKENS)


def _version_key(version: str) -> Tuple:
    """
    Natural-order key for Minecraft version strings.

    Pure versions ("1.21.11") rank above decorated ones ("1.8.9-OptiFine_...")
    so the official client jar wins ties. Numeric components within the
    version are compared as integers (so "1.21.11" > "1.21.8").
    """
    name = version.replace("minecraft-", "").replace("-client", "").strip()
    pure_rank = 0 if _PURE_VERSION_RE.match(name) else 1

    m = _VERSION_RUN_RE.search(name)
    if m:
        nums = tuple(int(g) for g in m.groups() if g is not None)
    else:
        nums = (0,)
    # Sort tuple: (pure_first, major, minor, patch). Reverse-sorted later.
    return (-pure_rank,) + nums


def _launcher_library_bases() -> List[str]:
    """
    Directories used by custom launchers (Prism, MultiMC, PolyMC, ATLauncher)
    that share a Maven-style library cache. The Mojang client jar lives at
    ``<base>/com/mojang/minecraft/<version>/minecraft-<version>-client.jar``.
    """
    system = platform.system().lower()
    candidates: List[str] = []
    if system == "windows":
        for root_env in (r"%APPDATA%", r"%LOCALAPPDATA%"):
            root = os.path.expandvars(root_env)
            for launcher in ("PrismLauncher", "MultiMC", "PolyMC", "ATLauncher"):
                candidates.append(os.path.join(root, launcher, "libraries"))
    elif system == "darwin":
        home = os.path.expanduser("~")
        for launcher in ("PrismLauncher", "MultiMC", "PolyMC", "ATLauncher"):
            candidates.append(os.path.join(
                home, "Library", "Application Support", launcher, "libraries"))
    else:  # linux / other unix
        for base in (
            os.path.expanduser("~/.local/share"),
            os.path.expanduser("~/.config"),
        ):
            for launcher in ("PrismLauncher", "MultiMC", "PolyMC", "ATLauncher"):
                candidates.append(os.path.join(base, launcher, "libraries"))
    return candidates


def find_mc_jars() -> List[str]:
    """
    Locate every Minecraft client JAR on the local system. Searches:

      * The official launcher    (%APPDATA%/.minecraft/versions/<v>/<v>.jar)
      * Prism Launcher           (%APPDATA%/PrismLauncher/libraries/com/mojang/minecraft/<v>/...)
      * MultiMC / PolyMC / ATLauncher (same library structure as Prism)

    Returns paths sorted newest-first by natural version order.
    """
    found: List[Tuple[Tuple, str]] = []

    # --- Official Mojang launcher ---
    if platform.system().lower() == "windows":
        official_bases = [
            os.path.expandvars(r"%APPDATA%\.minecraft\versions"),
            os.path.expandvars(r"%USERPROFILE%\.minecraft\versions"),
        ]
    elif platform.system().lower() == "darwin":
        official_bases = [
            os.path.expanduser("~/Library/Application Support/minecraft/versions"),
        ]
    else:
        official_bases = [os.path.expanduser("~/.minecraft/versions")]

    for base in official_bases:
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            if _is_loader_name(name):
                continue
            jar = os.path.join(base, name, f"{name}.jar")
            if os.path.isfile(jar):
                found.append((_version_key(name), jar))

    # --- Custom launchers (Prism / MultiMC / PolyMC / ATLauncher) ---
    for lib_base in _launcher_library_bases():
        mojang_root = os.path.join(lib_base, "com", "mojang", "minecraft")
        if not os.path.isdir(mojang_root):
            continue
        for version in os.listdir(mojang_root):
            if _is_loader_name(version):
                continue
            pattern = os.path.join(mojang_root, version,
                                   f"minecraft-{version}-client.jar")
            if os.path.isfile(pattern):
                found.append((_version_key(version), pattern))
            else:
                # Fall back to any client jar in this version dir.
                for jar in glob.glob(os.path.join(mojang_root, version, "*client*.jar")):
                    found.append((_version_key(version), jar))

    # Deduplicate (same version may appear in multiple launchers — keep all
    # paths but unique them) and sort newest-first.
    seen = set()
    unique: List[Tuple[Tuple, str]] = []
    for key, path in found:
        if path in seen:
            continue
        seen.add(path)
        unique.append((key, path))
    unique.sort(key=lambda kp: kp[0], reverse=True)
    return [path for _, path in unique]


# ---------------------------------------------------------------------------
# Font extraction from a single JAR
# ---------------------------------------------------------------------------

# Paths Mojang has used for the ASCII atlas over the years. Newer first.
_FONT_PNG_PATHS = (
    "assets/minecraft/textures/font/ascii.png",
    "assets/minecraft/font/ascii.png",
)

_FONT_META_PATHS = (
    "assets/minecraft/font/include/default.json",
    "assets/minecraft/font/default.json",
)


def _read_zip_first(z: zipfile.ZipFile, names) -> bytes:
    for n in names:
        try:
            return z.read(n)
        except KeyError:
            continue
    raise FileNotFoundError(f"None of these paths exist in jar: {names}")


def extract_font_templates(jar_path: str) -> Dict[str, np.ndarray]:
    """
    Extract Minecraft's default ASCII font glyph bitmaps from a JAR.

    Returns
    -------
    dict mapping each character → uint8 2D array of shape (cell_h, glyph_width).
    Pixel value 255 = inked, 0 = empty. At Minecraft's native GUI scale 1.
    """
    with zipfile.ZipFile(jar_path) as z:
        png_bytes  = _read_zip_first(z, _FONT_PNG_PATHS)
        meta_bytes = _read_zip_first(z, _FONT_META_PATHS)

    img = Image.open(io.BytesIO(png_bytes))
    arr = np.array(img.convert("LA"))  # (H, W, 2): luma + alpha
    # A pixel is "inked" if its alpha channel is non-zero. The font uses
    # alpha (not luminance) to define glyph shape.
    inked = (arr[:, :, 1] > 0).astype(np.uint8) * 255

    meta = json.loads(meta_bytes)
    chars_rows: Optional[List[str]] = None
    for provider in meta.get("providers", []):
        if provider.get("type") != "bitmap":
            continue
        f = str(provider.get("file", ""))
        if f.endswith("ascii.png"):
            chars_rows = list(provider["chars"])
            break
    if chars_rows is None:
        raise RuntimeError(
            f"No ascii.png provider in {meta_bytes!r}. "
            "Has Mojang changed the font layout?"
        )

    grid_rows = len(chars_rows)
    grid_cols = max(len(row) for row in chars_rows)
    H, W      = inked.shape
    cell_h    = H // grid_rows
    cell_w    = W // grid_cols
    if cell_h == 0 or cell_w == 0:
        raise RuntimeError(
            f"Atlas shape {inked.shape} incompatible with declared grid "
            f"{grid_rows}×{grid_cols}"
        )

    templates: Dict[str, np.ndarray] = {}
    for r, row in enumerate(chars_rows):
        for c, ch in enumerate(row):
            if ord(ch) == 0:
                continue
            cell = inked[r * cell_h : (r + 1) * cell_h,
                         c * cell_w : (c + 1) * cell_w]
            col_sums = cell.sum(axis=0)
            nz = np.where(col_sums > 0)[0]
            if len(nz) == 0:
                continue  # space and similar are handled separately
            width = int(nz[-1]) + 1
            templates[ch] = cell[:, :width].copy()

    return templates


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_font_npz(templates: Dict[str, np.ndarray], path: str) -> None:
    """Save templates to a single compressed ``.npz`` file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    chars = sorted(templates.keys())
    keys  = "".join(chars)
    save_dict = {f"g_{ord(ch):03d}": templates[ch] for ch in chars}
    save_dict["__chars__"] = np.frombuffer(keys.encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(path, **save_dict)


def load_font_npz(path: str) -> Dict[str, np.ndarray]:
    """Load templates from a ``.npz`` previously written by ``save_font_npz``."""
    data = np.load(path, allow_pickle=False)
    chars = bytes(data["__chars__"]).decode("utf-8")
    return {ch: data[f"g_{ord(ch):03d}"] for ch in chars}


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def upscale_template(tpl: np.ndarray, scale: int) -> np.ndarray:
    """Integer ``INTER_NEAREST`` upscale — mandatory for bitmap fonts."""
    if scale <= 1:
        return tpl
    h, w = tpl.shape[:2]
    return cv2.resize(tpl, (w * scale, h * scale),
                      interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# Bootstrap helper — does the right thing automatically
# ---------------------------------------------------------------------------

def ensure_font_cache(cache_path: str,
                      jar_paths: Optional[List[str]] = None,
                      *,
                      force: bool = False) -> Dict[str, np.ndarray]:
    """
    Return MC font templates, extracting + caching them if necessary.

    Parameters
    ----------
    cache_path : path to the ``.npz`` cache file.
    jar_paths  : explicit list of JARs to try (newest first). If None,
                 ``find_mc_jars()`` is used.
    force      : if True, re-extract even if the cache exists.
    """
    if not force and os.path.isfile(cache_path):
        try:
            return load_font_npz(cache_path)
        except Exception:
            # Cache is corrupt — fall through to re-extraction.
            pass

    if jar_paths is None:
        jar_paths = find_mc_jars()

    errors: List[str] = []
    for jar in jar_paths:
        try:
            templates = extract_font_templates(jar)
            save_font_npz(templates, cache_path)
            return templates
        except Exception as e:
            errors.append(f"  - {jar}: {type(e).__name__}: {e}")

    detail = "\n".join(errors) if errors else "  (no jars found at all)"
    raise RuntimeError(
        "Could not extract Minecraft's default font.\n"
        "Tried the following jars:\n" + detail + "\n"
        "If Minecraft Java Edition is installed somewhere unusual, "
        "pass jar_paths= explicitly to ensure_font_cache()."
    )


# ---------------------------------------------------------------------------
# CLI — re-extract on demand (e.g. after a Minecraft update)
# ---------------------------------------------------------------------------

def _cli() -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Extract Minecraft's default ASCII font for glyph OCR."
    )
    p.add_argument("--jar", action="append", default=None,
                   help="Path to a Minecraft .jar (may be passed multiple times).")
    p.add_argument("--cache",
                   default=os.path.join("data", "calibration", "mc_font.npz"),
                   help="Output .npz cache file.")
    p.add_argument("--force", action="store_true",
                   help="Re-extract even if the cache already exists.")
    p.add_argument("--list", action="store_true",
                   help="List candidate Minecraft jars and exit.")
    args = p.parse_args()

    if args.list:
        for j in find_mc_jars():
            print(j)
        return 0

    templates = ensure_font_cache(args.cache, jar_paths=args.jar, force=args.force)
    print(f"  Extracted {len(templates)} glyphs -> {args.cache}")
    sample = "".join(sorted(c for c in templates if c.isprintable()))[:80]
    print(f"  Glyph sample: {sample}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
