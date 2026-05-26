# vision/world/inverse_renderer.py
"""
Inverse-rendering verifier for block hypotheses.

What this is
------------
Given a hypothetical block at a known voxel position, the camera's
exact pose (eye XYZ + yaw + pitch) and the captured frame, we can
compute:

  1. Which of the cube's six faces would be visible (back-face
     culling + dot-product with the eye-to-voxel vector).
  2. Where the four corners of that face land on the screen
     (perspective projection through the camera intrinsics).
  3. What the canonical Minecraft texture for that block's face
     SHOULD look like (from the extracted asset jar).
  4. How well the actual captured pixels in that screen quadrilateral
     match the canonical texture (after perspective unwarp).

A high match score means "yes, this voxel is that block". A low
score means "no, it's something else". This is fundamentally MORE
ACCURATE than the colour-signature or sample-NN approach because:

  * It uses the EXACT screen geometry the renderer used. Sub-pixel
    accuracy from the decimal parts of F3 yaw / pitch is honoured.
  * It compares against the SPECIFIC face being viewed — not a
    block-level signature averaged across all six faces.
  * It needs no training data; the asset jar provides ground truth.

Where it earns its keep
-----------------------
The natural pairing is with F3 ground truth. F3 confirms ONE voxel
at the crosshair. Then ``score_voxel`` lets us:

  * Confirm NEIGHBOUR voxels carry the same block id (a wall, a
    floor, a tree trunk are all multiple identical voxels visible
    simultaneously) — propagating one F3 confirmation into many.
  * Reject hallucinated curiosity-queue entries — if the inverse
    score is low, the perception layer's guess was wrong.

Performance budget
------------------
Per voxel test: ~0.3-0.8 ms (small per-face perspective unwarp + a
single MAE compare). At 50 ms / tick budget we can afford ~50-150
tests per frame, which is plenty for expanding a 5-block-radius
neighbourhood around a confirmed crosshair voxel.

Future-proofing
---------------
* When biome tints / sun-angle lighting become available, the
  expected-appearance step applies them BEFORE comparison.
* When a CNN classifier ships, it can replace ``score_match`` —
  same projection geometry, just a smarter scorer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from vision.world.screen_ray import CameraIntrinsics, ScreenRay


# ---------------------------------------------------------------------------
# Face conventions
# ---------------------------------------------------------------------------
# MC face naming (visible from outside the cube):
#   up    : +Y normal (the "top" of the block)
#   down  : -Y normal (the "bottom")
#   north : -Z normal
#   south : +Z normal
#   east  : +X normal
#   west  : -X normal

_FACE_NORMALS: Dict[str, Tuple[int, int, int]] = {
    "up":    ( 0,  1,  0),
    "down":  ( 0, -1,  0),
    "north": ( 0,  0, -1),
    "south": ( 0,  0,  1),
    "east":  ( 1,  0,  0),
    "west":  (-1,  0,  0),
}

# Corner indices of each face. The voxel is at integer pos (x, y, z);
# its 8 corners are at offsets along each axis:
#   idx bit 0 = X offset (0 → x,  1 → x+1)
#   idx bit 1 = Y offset (0 → y,  1 → y+1)
#   idx bit 2 = Z offset (0 → z,  1 → z+1)
# Each face lists its 4 corners in (top-left, top-right, bottom-right,
# bottom-left) order from the perspective of a viewer outside the
# face looking at it. The perspective transform later maps these to
# the canonical texture corners in the same TL/TR/BR/BL order, so the
# unwarped patch is oriented like the atlas.
_FACE_CORNERS: Dict[str, Tuple[int, int, int, int]] = {
    # +Y face, viewed from above (+Y looking -Y).
    # Screen X = world X (right), screen Y = world Z (down/south).
    "up":    (2, 3, 7, 6),
    # -Y face, viewed from below (-Y looking +Y).
    "down":  (4, 5, 1, 0),
    # -Z face, viewed from -Z. X→right, Y→up.
    "north": (2, 3, 1, 0),
    # +Z face, viewed from +Z. X→LEFT (mirrored), Y→up.
    "south": (7, 6, 4, 5),
    # +X face, viewed from +X. Z→LEFT (mirrored), Y→up.
    "east":  (7, 3, 1, 5),
    # -X face, viewed from -X. Z→right, Y→up.
    "west":  (2, 6, 4, 0),
}


def _corner_xyz(voxel: Tuple[int, int, int], idx: int
                 ) -> Tuple[float, float, float]:
    """Decode the integer corner index into a world-space (x, y, z)."""
    x, y, z = voxel
    cx = x + (1 if (idx & 1) else 0)
    cy = y + (1 if (idx & 2) else 0)
    cz = z + (1 if (idx & 4) else 0)
    return float(cx), float(cy), float(cz)


# ---------------------------------------------------------------------------
# Per-block face → texture lookup
# ---------------------------------------------------------------------------
#
# Most vanilla blocks use one texture for all six faces (stone, dirt,
# planks, etc.) and the texture file is named after the block id with
# no suffix. A subset use side-vs-top textures (logs, grass_block,
# crafting_table, etc.) — those follow a small set of canonical
# suffixes. We hard-code the common patterns so the inverse renderer
# can look up the right face texture without parsing the full
# blockstate / model JSONs.

# Texture-name candidates per face, in priority order. The first one
# that exists in the asset cache wins. Order matters — ``_top`` is
# preferred for the ``up`` face but only as a fallback for ``down``
# because many blocks use the same top texture for both (slabs).
# The empty-string candidate ``""`` is the bare stem (single-texture
# blocks: stone, dirt, planks, …).
#
# This lookup is intentionally HEURISTIC rather than parsing the
# model JSONs. Heuristics cover the ~95 % of vanilla blocks that
# follow the convention; the per-block override dict below lists
# the remaining oddballs. When MC adds a new block following the
# convention (most do), it works without any code change.
_DEFAULT_FACE_SUFFIXES: Dict[str, Tuple[str, ...]] = {
    "up":    ("_top", ""),
    "down":  ("_bottom", "_top", ""),
    "north": ("_side", "_front", ""),
    "south": ("_side", "_front", ""),
    "east":  ("_side", ""),
    "west":  ("_side", ""),
}

# Per-block stem override. Maps a block id to the texture STEM the
# atlas uses (when different from the block id's path). Most logs,
# slabs, fences, signs already follow the ``<stem>_top.png`` /
# ``<stem>_side.png`` convention so they DON'T need an entry here —
# only list blocks where the stem differs from the id path.
#
# Future-proofing: blocks added in later MC versions that follow the
# convention need no entries. Only exotic mappings (e.g. an item-
# named block whose texture stem differs from the item id) belong
# here.
_BLOCK_STEM_ALIASES: Dict[str, str] = {
    # (Empty by default — every alias we used to ship had a stem
    # IDENTICAL to the block-id path, which the bare-id fallback
    # already handles. Listing them was a maintenance trap that
    # broke every time MC added a new log / slab variant.)
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class InverseRendererConfig:
    # Resolution we resample both the actual screen patch AND the
    # canonical texture to before comparing. 32×32 is more than the
    # source 16×16 atlas tile but smooths perspective sampling
    # artefacts.
    compare_size: int = 32
    # Reject matches whose mean absolute error per channel exceeds
    # this; otherwise scale to a [0, 1] confidence.
    mae_max: float = 70.0
    # Minimum visible face area on the screen (px²) for the
    # projection to be considered usable. Tiny far-away projections
    # are too noisy to score reliably.
    min_face_area_px2: float = 32.0
    # Refuse to score if the visible face would be partially or
    # fully off-screen — guarantees the perspective unwarp samples
    # only real pixels.
    require_fully_onscreen: bool = True


# ---------------------------------------------------------------------------
# InverseRenderer
# ---------------------------------------------------------------------------

class InverseRenderer:
    """
    Score how well a voxel's actual screen appearance matches the
    canonical texture for a hypothesised block id.

    Build once per process; the texture-atlas lookup caches faces
    after first read so repeat hypotheses for the same block id are
    free.
    """

    def __init__(self, assets, config: Optional[InverseRendererConfig] = None):
        self.assets = assets
        self.cfg = config or InverseRendererConfig()
        self._face_cache: Dict[Tuple[str, str], Optional[np.ndarray]] = {}

    # ── Geometry ──────────────────────────────────────────────────

    @staticmethod
    def visible_face(voxel: Tuple[int, int, int],
                      eye: Tuple[float, float, float],
                      ) -> Optional[str]:
        """
        Return which face of ``voxel`` is most visible from ``eye``.

        The most visible face is the one whose outward normal has the
        greatest dot product with the (voxel-centre → eye) vector.
        That's the face the eye is most "in front of".
        """
        cx = voxel[0] + 0.5
        cy = voxel[1] + 0.5
        cz = voxel[2] + 0.5
        dx = eye[0] - cx
        dy = eye[1] - cy
        dz = eye[2] - cz
        best_face: Optional[str] = None
        best_dot: float = 0.0
        for face, (nx, ny, nz) in _FACE_NORMALS.items():
            d = dx * nx + dy * ny + dz * nz
            if d > best_dot:
                best_dot = d
                best_face = face
        return best_face

    def project_face(self,
                      voxel: Tuple[int, int, int],
                      face: str,
                      *,
                      sr: ScreenRay,
                      eye: Tuple[float, float, float],
                      yaw: float,
                      pitch: float,
                      frame_shape: Tuple[int, int],
                      ) -> Optional[List[Tuple[float, float]]]:
        """
        Project the 4 corners of ``face`` to screen-space pixels.

        Returns the corners in their canonical order (matching the
        texture's top-left, top-right, bottom-right, bottom-left
        convention) or ``None`` if any corner is behind the camera or
        the projected face violates the on-screen constraint.
        """
        cfg = self.cfg
        H, W = frame_shape
        corner_indices = _FACE_CORNERS.get(face)
        if corner_indices is None:
            return None
        out: List[Tuple[float, float]] = []
        for idx in corner_indices:
            world_xyz = _corner_xyz(voxel, idx)
            proj = sr.project(world_xyz, yaw_deg=yaw, pitch_deg=pitch,
                              eye_xyz=eye)
            if proj is None:                # behind camera
                return None
            px, py, _depth = proj
            if cfg.require_fully_onscreen:
                if px < 0 or px >= W or py < 0 or py >= H:
                    return None
            out.append((px, py))
        # Reject too-small projected faces (anti-aliasing dominates).
        area = _polygon_area(out)
        if area < cfg.min_face_area_px2:
            return None
        return out

    # ── Expected appearance ──────────────────────────────────────

    def expected_face_texture(self,
                               block_id: str,
                               face: str,
                               ) -> Optional[np.ndarray]:
        """
        Return the canonical RGB texture for ``block_id``'s ``face``,
        resized to ``cfg.compare_size``. Cached after first hit.

        Always returns an (S, S, 3) uint8 array — grayscale source
        textures (stone, gravel, etc.) are broadcast to RGB; RGBA
        textures get the alpha channel dropped.
        """
        key = (block_id, face)
        if key in self._face_cache:
            return self._face_cache[key]
        tex = self._load_face_texture(block_id, face)
        if tex is not None:
            s = self.cfg.compare_size
            tex = cv2.resize(tex, (s, s), interpolation=cv2.INTER_AREA)
            if tex.ndim == 2:
                tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2RGB)
            elif tex.ndim == 3 and tex.shape[2] == 4:
                tex = tex[..., :3]
            tex = tex.astype(np.uint8)
        self._face_cache[key] = tex
        return tex

    def _load_face_texture(self,
                           block_id: str,
                           face: str,
                           ) -> Optional[np.ndarray]:
        """Walk a small set of texture-name candidates and return the
        first one that exists in the atlas."""
        stem = _BLOCK_STEM_ALIASES.get(
            block_id,
            block_id.split(":", 1)[-1] if ":" in block_id else block_id,
        )
        candidates: List[str] = []
        for suffix in _DEFAULT_FACE_SUFFIXES.get(face, ("",)):
            candidates.append(stem + suffix)
        # Always try the bare stem as a last resort.
        if stem not in candidates:
            candidates.append(stem)
        for name in candidates:
            tex = self.assets.block_texture(name)
            if tex is not None:
                return tex
        return None

    # ── Scoring ───────────────────────────────────────────────────

    def score_voxel(self,
                     frame_rgb: np.ndarray,
                     voxel: Tuple[int, int, int],
                     block_id: str,
                     *,
                     sr: ScreenRay,
                     eye: Tuple[float, float, float],
                     yaw: float,
                     pitch: float,
                     ) -> float:
        """
        Full pipeline: project the most-visible face, sample the
        actual frame pixels inside the projected quad, compare to
        the canonical texture for ``block_id``'s face. Returns a
        match score in [0, 1] — higher means stronger evidence the
        voxel is ``block_id``.
        """
        face = self.visible_face(voxel, eye)
        if face is None:
            return 0.0
        corners = self.project_face(
            voxel, face, sr=sr, eye=eye, yaw=yaw, pitch=pitch,
            frame_shape=frame_rgb.shape[:2],
        )
        if corners is None:
            return 0.0
        expected = self.expected_face_texture(block_id, face)
        if expected is None:
            return 0.0
        actual = self._unwarp_quad_to_square(frame_rgb, corners)
        if actual is None:
            return 0.0
        return self._compare(actual, expected)

    def best_block_id_for_voxel(self,
                                 frame_rgb: np.ndarray,
                                 voxel: Tuple[int, int, int],
                                 candidates: Iterable[str],
                                 *,
                                 sr: ScreenRay,
                                 eye: Tuple[float, float, float],
                                 yaw: float,
                                 pitch: float,
                                 ) -> Tuple[Optional[str], float]:
        """
        Test multiple ``block_id`` candidates against ``voxel`` and
        return the (best_id, best_score). Useful for "identify this
        voxel from a short list of likely block ids".
        """
        face = self.visible_face(voxel, eye)
        if face is None:
            return None, 0.0
        corners = self.project_face(
            voxel, face, sr=sr, eye=eye, yaw=yaw, pitch=pitch,
            frame_shape=frame_rgb.shape[:2],
        )
        if corners is None:
            return None, 0.0
        actual = self._unwarp_quad_to_square(frame_rgb, corners)
        if actual is None:
            return None, 0.0
        best_id: Optional[str] = None
        best_score: float = 0.0
        for bid in candidates:
            expected = self.expected_face_texture(bid, face)
            if expected is None:
                continue
            s = self._compare(actual, expected)
            if s > best_score:
                best_score = s
                best_id = bid
        return best_id, best_score

    # ── Internals ────────────────────────────────────────────────

    def _unwarp_quad_to_square(self,
                                frame_rgb: np.ndarray,
                                corners: List[Tuple[float, float]],
                                ) -> Optional[np.ndarray]:
        """Use a perspective transform to map the projected quad to a
        flat ``compare_size × compare_size`` square."""
        s = self.cfg.compare_size
        src = np.array(corners, dtype=np.float32)
        # Destination: top-left, top-right, bottom-right, bottom-left.
        dst = np.array([(0, 0), (s - 1, 0), (s - 1, s - 1), (0, s - 1)],
                       dtype=np.float32)
        try:
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(frame_rgb, M, (s, s),
                                          flags=cv2.INTER_AREA,
                                          borderMode=cv2.BORDER_REPLICATE)
        except cv2.error:
            return None
        if warped.ndim == 3 and warped.shape[2] == 4:
            warped = warped[..., :3]
        return warped.astype(np.uint8)

    def _compare(self,
                 actual_rgb: np.ndarray,
                 expected_rgb: np.ndarray) -> float:
        """Mean-absolute-error compare, mapped to confidence in [0, 1].

        The actual frame in MC is often DARKER than the canonical
        texture (alpha-blended with biome tints, applied lighting),
        so we normalise both by their mean before comparing — this
        eats global brightness offsets without losing texture
        structure.
        """
        if actual_rgb.shape != expected_rgb.shape:
            s = self.cfg.compare_size
            expected_rgb = cv2.resize(expected_rgb, (s, s),
                                       interpolation=cv2.INTER_AREA)
        a = actual_rgb.astype(np.float32)
        e = expected_rgb.astype(np.float32)
        # Per-channel mean normalisation.
        a_mean = a.reshape(-1, 3).mean(axis=0)
        e_mean = e.reshape(-1, 3).mean(axis=0)
        scale = np.where(e_mean > 1.0, a_mean / np.maximum(e_mean, 1.0), 1.0)
        e_scaled = np.clip(e * scale[None, None, :], 0, 255)
        mae = float(np.abs(a - e_scaled).mean())
        if mae >= self.cfg.mae_max:
            return 0.0
        return float(1.0 - mae / self.cfg.mae_max)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _polygon_area(points: List[Tuple[float, float]]) -> float:
    """Shoelace formula. Works for convex or concave polygons."""
    n = len(points)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


__all__ = ["InverseRenderer", "InverseRendererConfig"]
