# vision/block_icons.py
"""
Render the isometric block icons that Minecraft draws inside inventory
slots for block-form items (cobblestone, dirt, oak_log, …).

Why we need this
----------------
``vision/inventory.py``'s template matcher works by comparing each
inventory slot's pixels against the 16×16 RGBA item textures extracted
from the game jar. That works perfectly for items that are *drawn*
directly from their item.png (apple, diamond, sword) — but blocks are
NOT drawn from a flat texture. Minecraft renders the block model in 3D
at a fixed isometric viewpoint and rasterises that into the 16×16 slot.
Without an isometric render to match against, the recognizer either
confuses every "blocky" texture or falls through to the wrong template.

What this module does
---------------------
* Resolves a block id to its model JSON via ``models/block/<id>.json``,
  walking the parent chain to gather the texture variables.
* Picks the up / north (front) / east (right) face textures.
* Composites them into a 16×16 RGBA icon using three affine warps that
  match Mojang's vanilla rotation (≈30° around X, ≈45° around Y) — the
  same "three-rhombus hexagon" outline you see in every vanilla
  inventory.
* Applies Mojang's per-face shading (top 100 %, left 80 %, right 60 %)
  so the shading matches what the game actually draws.
* Caches every rendered icon under
  ``data/calibration/block_icons/<version>/<id>.png`` so the second run
  is instant. Cache invalidates when the assets cache's ``meta.json``
  reports a newer Minecraft version.

What we don't (yet) do
----------------------
* Per-biome tinting (grass top, foliage). The blocks render correctly
  shape-wise, just without a biome tint; in inventory the player sees
  the *default* tint anyway (grass = green ≈ 123/189/110), which we
  apply when the model carries a tintindex.
* Non-cube models (cross/saplings, torch, fence, slab, stairs). These
  fall back to "no isometric icon" — the recognizer will skip them and
  rely on whatever other template wins.
* Animated textures (water/lava/portal etc.). We take the first frame.

The renderer is fully deterministic, so the cached PNGs are
bit-reproducible for the same Minecraft version.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Geometry — the three visible face quads in a 16×16 icon
# ---------------------------------------------------------------------------
#
# Coordinates were measured from screenshot crops of vanilla inventory icons
# (cobblestone, oak_planks, grass_block) at GUI scale 4. They reproduce
# Mojang's vanilla "rhombus hexagon" silhouette to within one pixel:
#
#         (8, 0)
#        /      \
#    (0,4)      (16,4)
#      |  \   /  |
#      |   (8,8)  |
#    (0,12)    (16,12)
#        \      /
#        (8,16)
#
# Each face is a parallelogram, so a single affine map (3 points → 3 points)
# is enough to project a 16×16 texture onto it. The mapping below sets:
#   texture (0, 0)   →  face's TOP-LEFT  corner
#   texture (15, 0)  →  face's TOP-RIGHT corner
#   texture (0, 15)  →  face's BOTTOM-LEFT corner
# which matches Mojang's "face top = texture top" convention.

_ICON_SIZE = 16


@dataclass(frozen=True)
class _FaceQuad:
    name: str                                 # "up" / "north" / "east"
    tl: Tuple[float, float]                   # icon-space top-left
    tr: Tuple[float, float]                   # icon-space top-right
    bl: Tuple[float, float]                   # icon-space bottom-left
    shade: float                              # multiply colour by this
    # Some face textures are read "rotated" relative to the icon — e.g.
    # for cube_column.end the top texture is unrotated, but for tinted
    # blocks the orientation needs to match the per-face rotation field.
    rotate_quarters: int = 0


# Mojang's vanilla shading factors (BlockRenderDispatcher.FACE_BRIGHTNESS):
#   up         = 1.0
#   down       = 0.5  (never visible in inventory)
#   north/south = 0.8
#   east/west   = 0.6
#
# Empirically (cross-referenced against captured furnace icons, where
# the "front" texture is mapped to NORTH and clearly appears on the
# LEFT of the inventory icon), MC's inventory rotation makes:
#   up    → top half of the icon (diamond)
#   north → LEFT side of the icon  (shade 0.8 — brighter)
#   west  → RIGHT side of the icon (shade 0.6 — darker)
#
# Geometry: the captured icons show the cube as ~14 px wide × ~16 px
# tall hexagon centred horizontally. The valley where all 3 faces meet
# sits at (8, 7), not the icon centre (8, 8). Earlier guesses used a
# full-16-px-wide cube which was off-by-one to two pixels on every
# horizontal edge — enough MAE per pixel to push correct iso matches
# well above the recogniser's score cap.
#
# Vertex positions derived from projecting the 8 cube corners under
# rotation [X=30°, Y=135°], orthographic projection, scale ≈ 10.16
# (so the cube spans 16 px vertically, ~14 px horizontally).
_FACE_QUADS: Tuple[_FaceQuad, ...] = (
    # UP face — diamond on top. Texture-to-icon mapping respects MC's
    # standard UV convention for UP (U=+X, V=+Z).
    #   texture (0,0)  → block (-X, +Y, -Z) → icon (8, 7)
    #   texture (16,0) → block (+X, +Y, -Z) → icon (1, 4)
    #   texture (0,16) → block (-X, +Y, +Z) → icon (15, 4)
    _FaceQuad("up",    tl=(8, 7),  tr=(1, 4),  bl=(15, 4),  shade=1.00),
    # NORTH face — parallelogram on LEFT. UV convention: U=-X, V=-Y.
    #   texture (0,0)  → block (+X, +Y, -Z) → icon (1, 4)
    #   texture (16,0) → block (-X, +Y, -Z) → icon (8, 7)
    #   texture (0,16) → block (+X, -Y, -Z) → icon (1, 12)
    _FaceQuad("north", tl=(1, 4),  tr=(8, 7),  bl=(1, 12),  shade=0.80),
    # WEST face — parallelogram on RIGHT. UV convention: U=-Z, V=-Y.
    #   texture (0,0)  → block (-X, +Y, +Z) → icon (15, 4)
    #   texture (16,0) → block (-X, +Y, -Z) → icon (8, 7)
    #   texture (0,16) → block (-X, -Y, +Z) → icon (15, 12)
    _FaceQuad("west",  tl=(15, 4), tr=(8, 7),  bl=(15, 12), shade=0.60),
)

# Mapping from "I want to draw this face" → "which model-face texture
# should I pull". For cube models (down/up/north/south/east/west) we
# use the exact face's texture. For cube_column we map sides → "side",
# top/bottom → "end". cube_orientable maps "front" to NORTH so we
# include "front" in the north lookup chain. Looked up in this order;
# first hit wins.
_FACE_TEXTURE_KEYS: Dict[str, Tuple[str, ...]] = {
    "up":    ("up",    "top",  "end",  "side", "all"),
    "north": ("north", "front", "side", "all"),
    "west":  ("west",  "side", "all"),
}

# Default biome tints applied to faces with ``tintindex`` (grass top,
# foliage, water, etc.). Values picked from Mojang's biome JSONs for the
# Plains biome — a sensible "neutral default" the inventory uses too.
_DEFAULT_TINTS: Dict[str, Tuple[int, int, int]] = {
    "grass":   (124, 189,  107),
    "foliage": (119, 171,  47),
    "water":   ( 63, 118,  228),
}


# ---------------------------------------------------------------------------
# Model resolver
# ---------------------------------------------------------------------------

@dataclass
class _Model:
    """Flattened block model with all texture variables resolved."""
    block_id: str                                 # "cobblestone"
    textures: Dict[str, str]                      # {"all": "block/cobblestone", ...}
    parents: Tuple[str, ...]                      # chain, root-first
    tintindex_faces: frozenset                    # which face names use tint
    raw: dict                                     # the leaf JSON


class ModelResolver:
    """
    Walk a block model's ``parent`` chain and resolve every ``#var``
    texture reference to a concrete texture path
    (``minecraft:block/cobblestone`` → ``block/cobblestone``).
    """

    def __init__(self, assets):
        # MCAssets — used to load block model JSONs lazily.
        self.assets = assets

    # ------------------------------------------------------------------

    def resolve(self, block_id: str) -> Optional[_Model]:
        leaf = self.assets.block_model(block_id)
        if leaf is None:
            return None
        chain: List[dict] = [leaf]
        seen = {block_id}

        # Walk parents until no further parent or we hit a generic root.
        current = leaf
        parents: List[str] = []
        while True:
            p = current.get("parent")
            if not p:
                break
            p_id = p.split(":", 1)[-1]
            # block/foo → foo. Item parents and the meta "block/block" /
            # "block/cube" / "block/cube_all" / "block/cube_column" all
            # carry geometry but no NEW texture vars; we just need to
            # know we hit them so we don't loop.
            if p_id.startswith("block/"):
                p_id = p_id[len("block/"):]
            elif p_id.startswith("item/"):
                break
            if p_id in seen:
                break
            parents.append(p_id)
            parent_json = self.assets.block_model(p_id)
            if parent_json is None:
                break
            chain.append(parent_json)
            seen.add(p_id)
            current = parent_json

        # Merge textures root-first so the leaf wins.
        merged: Dict[str, str] = {}
        for j in reversed(chain):
            t = j.get("textures") or {}
            for k, v in t.items():
                merged[k] = str(v)

        # Resolve "#var" references inside the merged texture table.
        resolved = self._resolve_vars(merged)

        # Collect the set of face names that request a biome tint.
        tinted = set()
        for el in (leaf.get("elements") or []):
            for face_name, face in (el.get("faces") or {}).items():
                if "tintindex" in face:
                    tinted.add(face_name)

        return _Model(
            block_id=block_id,
            textures=resolved,
            parents=tuple(parents),
            tintindex_faces=frozenset(tinted),
            raw=leaf,
        )

    def is_full_cube(self, model: _Model) -> bool:
        """
        Return True iff the resolved model's geometry is a single
        0,0,0 → 16,16,16 element with all six faces drawn.

        Walks the parent chain leaf-first looking for the first JSON
        that declares ``elements`` — that's the one that determines the
        shape (parents earlier in the chain don't override it; later
        children do, but if a child doesn't declare elements at all it
        inherits its parent's shape).
        """
        # Try the leaf first, then walk parents.
        chain_jsons: List[dict] = [model.raw]
        for p in model.parents:
            j = self.assets.block_model(p)
            if j is not None:
                chain_jsons.append(j)

        for j in chain_jsons:
            elements = j.get("elements")
            if not elements:
                continue
            if len(elements) != 1:
                return False
            el = elements[0]
            frm = el.get("from") or []
            to  = el.get("to")   or []
            if (list(frm) != [0, 0, 0] or list(to) != [16, 16, 16]):
                return False
            faces = el.get("faces") or {}
            # Need at least up + (north or side) + (east or side) — the
            # three faces our isometric projection draws. Missing faces
            # would punch holes in the icon.
            need = {"up", "north", "east"}
            if not need.issubset(set(faces.keys())):
                return False
            return True
        return False

    @staticmethod
    def _resolve_vars(textures: Dict[str, str]) -> Dict[str, str]:
        """
        ``"side": "#all"`` should become whatever ``all`` is. Resolve
        these chains iteratively, with a sane recursion cap.
        """
        out = dict(textures)
        for _ in range(8):
            changed = False
            for k, v in out.items():
                if isinstance(v, str) and v.startswith("#"):
                    target = v[1:]
                    if target in out and out[target] != v:
                        out[k] = out[target]
                        changed = True
            if not changed:
                break
        # Strip the "minecraft:" namespace and the "block/" prefix so we
        # have a name we can hand to assets.block_texture(name).
        cleaned: Dict[str, str] = {}
        for k, v in out.items():
            if not isinstance(v, str) or v.startswith("#"):
                continue
            v = v.split(":", 1)[-1]
            if v.startswith("block/"):
                v = v[len("block/"):]
            cleaned[k] = v
        return cleaned


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class BlockIconRenderer:
    """
    Render the inventory isometric icon for a single block id.

    Build once per process (loads the assets index lazily) and call
    ``render(block_id)`` for each block you want. The result is a
    16×16 RGBA uint8 image with transparent background.
    """

    def __init__(self, assets):
        self.assets = assets
        self._resolver = ModelResolver(assets)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def render(self, block_id: str) -> Optional[np.ndarray]:
        model = self._resolver.resolve(block_id)
        if model is None:
            return None

        # Skip anything that isn't a full 0,0,0→16,16,16 cube — stairs,
        # slabs, fences, doors, walls, crosses all need their own
        # geometry to render correctly, and faking them as full cubes
        # would put a cobblestone_stairs icon side-by-side with the
        # cobblestone icon and break tie-breaking in the recognizer.
        if not self._resolver.is_full_cube(model):
            return None

        faces_rgba: List[Tuple[_FaceQuad, np.ndarray]] = []
        for quad in _FACE_QUADS:
            tex_name = self._pick_texture_for_face(model, quad.name)
            if tex_name is None:
                # If we can't find textures for even one face, this isn't a
                # cube-shaped model and we shouldn't render an iso icon.
                return None
            tex = self._load_texture_16(tex_name)
            if tex is None:
                return None
            if quad.name in model.tintindex_faces:
                tex = _apply_tint(tex, _default_tint_for(model.block_id))
            faces_rgba.append((quad, tex))

        # Composite back-to-front: top, then left, then right. Order
        # matters for the seams along the (8,8) crease — drawing top
        # first means a few sub-pixel rounding errors at the seam end
        # up under the side faces' rendered pixels, which is what MC
        # does too.
        out = np.zeros((_ICON_SIZE, _ICON_SIZE, 4), dtype=np.uint8)
        for quad, tex in faces_rgba:
            patch = _project_face(tex, quad)
            out = _alpha_over(out, patch)
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _pick_texture_for_face(self, model: _Model, face: str) -> Optional[str]:
        for k in _FACE_TEXTURE_KEYS[face]:
            if k in model.textures:
                return model.textures[k]
        return None

    def _load_texture_16(self, name: str) -> Optional[np.ndarray]:
        """Load a block texture and normalise to 16×16 RGBA."""
        tex = self.assets.block_texture(name)
        if tex is None:
            return None
        return _normalize_16(tex)


# ---------------------------------------------------------------------------
# Render pipeline helpers
# ---------------------------------------------------------------------------

def _project_face(tex_rgba: np.ndarray, quad: _FaceQuad) -> np.ndarray:
    """
    Warp a 16×16 RGBA texture onto its parallelogram quad inside a
    16×16 RGBA canvas, then apply the per-face shade.
    """
    if quad.rotate_quarters:
        tex_rgba = np.rot90(tex_rgba, k=quad.rotate_quarters)

    src = np.float32([[0, 0], [_ICON_SIZE - 1, 0], [0, _ICON_SIZE - 1]])
    dst = np.float32([quad.tl, quad.tr, quad.bl])
    M = cv2.getAffineTransform(src, dst)

    patch = cv2.warpAffine(
        tex_rgba, M, (_ICON_SIZE, _ICON_SIZE),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    # Apply face shade to RGB only — alpha is untouched.
    if quad.shade != 1.0:
        rgb = patch[..., :3].astype(np.int32)
        rgb = (rgb * quad.shade).astype(np.int32).clip(0, 255).astype(np.uint8)
        patch[..., :3] = rgb
    return patch


def _alpha_over(base: np.ndarray, top: np.ndarray) -> np.ndarray:
    """
    Standard "src-over" alpha composite of ``top`` onto ``base``. Both
    must be 16×16×4 uint8.
    """
    a_top  = top[..., 3:4].astype(np.float32) / 255.0
    a_base = base[..., 3:4].astype(np.float32) / 255.0
    out_a  = a_top + a_base * (1.0 - a_top)

    # Avoid divide-by-zero where output alpha is 0.
    denom = np.where(out_a > 0, out_a, 1.0)
    rgb_top  = top[..., :3].astype(np.float32)
    rgb_base = base[..., :3].astype(np.float32)
    out_rgb  = (rgb_top * a_top + rgb_base * a_base * (1.0 - a_top)) / denom

    out = np.zeros_like(base)
    out[..., :3] = out_rgb.clip(0, 255).astype(np.uint8)
    out[..., 3]  = (out_a[..., 0] * 255).clip(0, 255).astype(np.uint8)
    return out


def _normalize_16(arr: np.ndarray) -> Optional[np.ndarray]:
    """Coerce a loaded texture into a 16×16 RGBA uint8 array."""
    if arr is None or arr.size == 0:
        return None
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGBA)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2RGBA)
    elif arr.ndim != 3 or arr.shape[2] != 4:
        return None

    h, w = arr.shape[:2]
    # Strip animation atlases — Mojang stores animated textures as
    # vertical strips of N square frames. Take frame 0 only.
    if h != w:
        if w == 0 or h % w != 0:
            return None
        arr = arr[:w]
    if arr.shape[0] != _ICON_SIZE:
        arr = cv2.resize(arr, (_ICON_SIZE, _ICON_SIZE),
                         interpolation=cv2.INTER_AREA)
    return arr.astype(np.uint8)


def _apply_tint(tex_rgba: np.ndarray,
                tint_rgb: Tuple[int, int, int]) -> np.ndarray:
    """
    Multiply every pixel's RGB by ``tint_rgb / 255``. Alpha untouched.
    Matches Mojang's vertex-colour tint pass.
    """
    out = tex_rgba.copy()
    tr, tg, tb = tint_rgb
    f = np.array([tr / 255.0, tg / 255.0, tb / 255.0], dtype=np.float32)
    out[..., :3] = (out[..., :3].astype(np.float32) * f
                    ).clip(0, 255).astype(np.uint8)
    return out


def _default_tint_for(block_id: str) -> Tuple[int, int, int]:
    """Pick a sensible biome tint based on the block name."""
    if "grass" in block_id:
        return _DEFAULT_TINTS["grass"]
    if "leaves" in block_id or "vine" in block_id:
        return _DEFAULT_TINTS["foliage"]
    if block_id in ("water", "water_still", "water_flow"):
        return _DEFAULT_TINTS["water"]
    return _DEFAULT_TINTS["foliage"]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@dataclass
class BlockIconCache:
    """
    Disk + memory cache for rendered icons.

    Layout
    ------
    ``cache_dir / <version> / <block_id>.png``
    ``cache_dir / <version> / _manifest.json``  (which icons rendered OK)
    """
    cache_dir: Path
    version: str
    _mem: Dict[str, Optional[np.ndarray]] = field(default_factory=dict, repr=False)

    @property
    def version_dir(self) -> Path:
        return self.cache_dir / self.version

    @property
    def manifest_path(self) -> Path:
        return self.version_dir / "_manifest.json"

    # ------------------------------------------------------------------

    def get(self, block_id: str) -> Optional[np.ndarray]:
        if block_id in self._mem:
            return self._mem[block_id]
        p = self.version_dir / f"{block_id}.png"
        if p.is_file():
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim == 3 and img.shape[2] == 4:
                rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                self._mem[block_id] = rgba
                return rgba
        return None

    def put(self, block_id: str, rgba: Optional[np.ndarray]) -> None:
        self._mem[block_id] = rgba
        if rgba is None:
            return
        self.version_dir.mkdir(parents=True, exist_ok=True)
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(self.version_dir / f"{block_id}.png"), bgra)

    def write_manifest(self, rendered_ids: List[str],
                       skipped_ids: List[str]) -> None:
        self.version_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps({
            "version":   self.version,
            "rendered":  sorted(rendered_ids),
            "skipped":   sorted(skipped_ids),
            "generated": int(time.time()),
        }, indent=2), encoding="utf-8")

    def load_manifest(self) -> Optional[dict]:
        if not self.manifest_path.is_file():
            return None
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def ensure_block_icon_cache(
    assets,
    *,
    cache_root: Optional[Path] = None,
    force: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> BlockIconCache:
    """
    Render an isometric inventory icon for every block we can resolve a
    model for, caching to disk.

    Skips silently when a block's model isn't cube-shaped (we have no
    geometry for cross saplings, fences, slabs, …); the recognizer can
    still fall back to other templates (item.png if one exists).

    Returns the populated ``BlockIconCache``. Subsequent calls with the
    same arguments are near-instant.
    """
    if cache_root is None:
        cache_root = Path(assets.root).resolve().parent.parent.parent \
            / "data" / "calibration" / "block_icons"
    version = assets.version
    cache = BlockIconCache(cache_dir=Path(cache_root), version=version)

    manifest = cache.load_manifest()
    # We used to drive icon discovery from ``assets.list_block_textures()``
    # but switched to the model registry — every block has a model JSON,
    # whereas texture files exist for non-cube block parts too. Texture
    # listing remains available via MCAssets if a future caller needs it.
    model_dir = Path(assets.root) / "models" / "block"
    if model_dir.is_dir():
        block_models = sorted(p.stem for p in model_dir.iterdir()
                              if p.suffix == ".json")
    else:
        block_models = []

    if (not force and manifest
            and manifest.get("version") == version
            and len(manifest.get("rendered") or []) > 0):
        return cache

    renderer = BlockIconRenderer(assets)
    rendered: List[str] = []
    skipped:  List[str] = []
    total = len(block_models)
    for idx, name in enumerate(block_models):
        # Skip model variants — only render canonical block names. A
        # variant model like "oak_door_top_left" is not an inventory
        # item and re-rendering it adds noise to the template DB.
        if _is_variant_model_name(name):
            continue
        try:
            icon = renderer.render(name)
        except Exception as e:
            icon = None
            # Surface the first render failure — corrupted texture
            # PNG / broken model JSON / missing parent in the asset
            # cache. The skipped block falls back to flat icon
            # template, degrading recogniser quality silently.
            # First-failure WARN; subsequent failures cached but
            # silent so a 1200-block scan doesn't print 50 lines.
            if not getattr(ensure_block_icon_cache,
                           "_render_warn_emitted", False):
                ensure_block_icon_cache._render_warn_emitted = True
                print(f"[block_icons][WARN] render({name!r}) raised "
                      f"{e!r}. The block falls back to its flat-icon "
                      f"template. Further render failures silenced.")
        cache.put(name, icon)
        if icon is None:
            skipped.append(name)
        else:
            rendered.append(name)
        if progress is not None and (idx % 50 == 0 or idx == total - 1):
            progress(idx + 1, total)

    cache.write_manifest(rendered, skipped)
    return cache


_VARIANT_SUFFIXES = (
    "_top", "_bottom", "_side", "_inner", "_outer",
    "_top_left", "_top_right", "_bottom_left", "_bottom_right",
    "_open", "_closed", "_post", "_horizontal", "_inventory",
    "_pressed", "_unpressed", "_lit", "_unlit", "_on", "_off",
    "_north", "_south", "_east", "_west", "_up", "_down",
    "_outer_stairs", "_inner_stairs", "_stage0", "_stage1",
    "_stage2", "_stage3", "_stage4", "_stage5", "_stage6", "_stage7",
    "_age_0", "_age_1", "_age_2", "_age_3", "_age_4", "_age_5",
    "_age_6", "_age_7",
)


def _is_variant_model_name(name: str) -> bool:
    """
    Return True for model files that are render variants (oak_door_top_left,
    cobblestone_wall_post, …) rather than the canonical inventory model.

    Heuristic: ends with one of the known variant suffixes. The canonical
    model file (e.g. ``oak_door``) does NOT end in these.
    """
    return any(name.endswith(s) for s in _VARIANT_SUFFIXES)


def list_rendered_icons(cache: BlockIconCache) -> List[str]:
    """Return the block ids that have a cached icon on disk."""
    if not cache.version_dir.is_dir():
        return []
    return sorted(p.stem for p in cache.version_dir.iterdir()
                  if p.suffix == ".png" and not p.stem.startswith("_"))


__all__ = [
    "BlockIconRenderer", "BlockIconCache",
    "ensure_block_icon_cache", "list_rendered_icons",
]
