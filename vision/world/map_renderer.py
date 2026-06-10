# vision/world/map_renderer.py
"""
Top-down rendering of the AI's :class:`WorldMap`.

What this is for
----------------
Letting a human (or the AI's future planner) *see* the mental map the
perception layer is building. The WorldMap is a sparse dict keyed by
``(x, y, z)`` integer block coords; this module flattens the most
recently observed layer into a 2-D top-down image with:

* a centered player position (◇),
* a yaw arrow showing where the camera is looking,
* one coloured pixel per known block, with colour derived from the
  block id (vanilla blocks use a curated palette; everything else
  uses a deterministic hash so the same block id always gets the
  same colour),
* a faint translucent overlay marking the crosshair-targeted voxel,
* a header strip showing pose + sample-store stats.

Why per-Y layered, not isometric
--------------------------------
A top-down view is the most useful first-cut visualisation: it shows
nav-relevant geometry (where the holes are, where the trees are) at a
glance, with zero perspective math. An isometric / 3-D view is a nice
follow-on — its renderer can sit beside this module and consume the
same WorldMap.

This module never mutates anything; it only reads from the WorldMap.
Safe to call from background threads as long as the map isn't being
written concurrently.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from vision.world.map import WorldMap
from vision.world.types import PlayerPose


# ---------------------------------------------------------------------------
# Block → colour palette
# ---------------------------------------------------------------------------
# A small curated table for the blocks the AI is most likely to see.
# Everything not in the table falls through to a hash-based colour, so
# any modded block id still gets a stable colour even if it's ugly.

_PALETTE: Dict[str, Tuple[int, int, int]] = {
    # Surfaces
    "minecraft:grass_block":       ( 90, 170,  60),
    "minecraft:short_grass":       (110, 180,  70),
    "minecraft:tall_grass":        (110, 180,  70),
    "minecraft:fern":              ( 95, 165,  80),
    "minecraft:dirt":              (110,  85,  55),
    "minecraft:dirt_path":         (140, 115,  70),
    "minecraft:coarse_dirt":       (105,  80,  50),
    "minecraft:rooted_dirt":       (135, 100,  55),
    "minecraft:podzol":            ( 95,  65,  35),
    "minecraft:mycelium":          (115,  85,  90),
    "minecraft:moss_block":        ( 85, 135,  55),
    "minecraft:moss_carpet":       ( 90, 140,  60),
    "minecraft:sand":              (220, 205, 160),
    "minecraft:red_sand":          (190, 110,  60),
    "minecraft:gravel":            (135, 130, 125),
    "minecraft:clay":              (155, 165, 180),
    "minecraft:snow":              (245, 248, 252),
    "minecraft:snow_block":        (245, 248, 252),
    "minecraft:packed_ice":        (200, 220, 245),
    "minecraft:ice":               (170, 200, 235),
    # Stone family
    "minecraft:stone":             (125, 125, 125),
    "minecraft:cobblestone":       (110, 110, 110),
    "minecraft:andesite":          (135, 135, 135),
    "minecraft:diorite":           (200, 200, 200),
    "minecraft:granite":           (175, 110,  85),
    "minecraft:deepslate":         ( 70,  70,  75),
    "minecraft:tuff":              (110, 105, 100),
    "minecraft:calcite":           (225, 225, 220),
    "minecraft:smooth_stone":      (155, 155, 155),
    "minecraft:smooth_stone_slab": (155, 155, 155),
    "minecraft:smooth_stone_slab_side": (155, 155, 155),
    "minecraft:bedrock":           ( 40,  40,  40),
    # Ores
    "minecraft:coal_ore":          ( 90,  90,  90),
    "minecraft:iron_ore":          (160, 140, 110),
    "minecraft:gold_ore":          (200, 175,  80),
    "minecraft:diamond_ore":       (140, 200, 215),
    "minecraft:redstone_ore":      (180,  60,  60),
    "minecraft:lapis_ore":         ( 80, 110, 165),
    "minecraft:copper_ore":        (180, 120,  90),
    "minecraft:emerald_ore":       ( 80, 180, 100),
    # Wood
    "minecraft:oak_log":           (105,  85,  55),
    "minecraft:spruce_log":        ( 65,  45,  25),
    "minecraft:birch_log":         (215, 210, 195),
    "minecraft:jungle_log":        (155, 115,  75),
    "minecraft:acacia_log":        (170,  85,  50),
    "minecraft:dark_oak_log":      ( 55,  35,  20),
    "minecraft:mangrove_log":      (115,  60,  55),
    "minecraft:cherry_log":        (215, 165, 165),
    "minecraft:oak_leaves":        ( 60, 130,  35),
    "minecraft:spruce_leaves":     ( 50, 110,  60),
    "minecraft:birch_leaves":      (135, 175,  90),
    "minecraft:jungle_leaves":     ( 55, 165,  35),
    "minecraft:acacia_leaves":     (110, 145,  35),
    "minecraft:dark_oak_leaves":   ( 50, 110,  35),
    # Liquids
    "minecraft:water":             ( 65,  90, 200),
    "minecraft:lava":              (235, 100,  35),
    # Common nether / end
    "minecraft:netherrack":        (110,  50,  45),
    "minecraft:soul_sand":         ( 95,  75,  60),
    "minecraft:end_stone":         (225, 220, 175),
    "minecraft:obsidian":          ( 35,  20,  55),
}

# Special placeholder for cells the WorldMap holds but with no id.
_UNKNOWN_COLOR = (180, 180, 180)

# Lazy-loaded asset cache + computed-from-texture colours. We compute
# the dominant colour for any block id on first request, then cache
# it — so any new MC version's blocks (pale_oak_log, firefly_bush,
# pitcher_plant, …) get a meaningful colour from their own texture
# instead of a hash gibberish.
_ASSETS_CACHE = None
_COMPUTED_COLOUR_CACHE: dict = {}


def _load_assets():
    global _ASSETS_CACHE
    if _ASSETS_CACHE is None:
        try:
            from vision.mc_assets import MCAssets
            _ASSETS_CACHE = MCAssets.load()
        except Exception:
            _ASSETS_CACHE = False   # sentinel: asset cache unavailable
    return _ASSETS_CACHE if _ASSETS_CACHE else None


def _colour_from_texture(block_id: str) -> Optional[Tuple[int, int, int]]:
    """
    Compute a representative RGB triple from the texture atlas.

    We try a small set of candidate texture stems (the bare id, then
    common multi-face suffixes like ``_top``, ``_side``), and pick
    the first that exists. The colour is the mean RGB over the
    OPAQUE pixels (alpha > 16) of the texture — opaque-mean ignores
    transparent borders on plants / vines / chains so the result is
    closer to "what this block looks like" than "what its bounding
    box looks like".
    """
    cached = _COMPUTED_COLOUR_CACHE.get(block_id)
    if cached is not None:
        return cached
    assets = _load_assets()
    if assets is None:
        return None
    stem = block_id.split(":", 1)[-1] if ":" in block_id else block_id
    # Try variants in priority order. Top first because plants /
    # logs read better top-down than from the side.
    for suffix in ("_top", "", "_side", "_front", "_end", "_still"):
        tex = assets.block_texture(stem + suffix)
        if tex is None:
            continue
        try:
            colour = _opaque_mean_rgb(tex)
        except Exception:
            colour = None
        if colour is not None:
            _COMPUTED_COLOUR_CACHE[block_id] = colour
            return colour
    _COMPUTED_COLOUR_CACHE[block_id] = None
    return None


def _opaque_mean_rgb(tex) -> Optional[Tuple[int, int, int]]:
    """Return the per-channel mean RGB over opaque texture pixels."""
    if tex is None or tex.size == 0:
        return None
    if tex.ndim == 2:
        # Grayscale texture (stone, gravel) — fake an RGB.
        v = int(tex.mean())
        return (v, v, v)
    if tex.ndim != 3:
        return None
    if tex.shape[2] >= 4:
        rgb = tex[..., :3]
        alpha = tex[..., 3]
        mask = alpha > 16
        if not mask.any():
            return None
        opaque = rgb[mask]
        m = opaque.mean(axis=0)
    else:
        m = tex.reshape(-1, 3).mean(axis=0)
    return (int(m[0]), int(m[1]), int(m[2]))


def _color_for_block(block_id: Optional[str]) -> Tuple[int, int, int]:
    """
    Resolve a block id to an RGB triple.

    Priority:
      1. Curated palette (``_PALETTE``) — hand-tuned for visually
         common vanilla blocks. Stable across asset versions.
      2. Texture-derived dominant colour — works for ANY block in
         the extracted asset jar, including future MC versions and
         (eventually) modded blocks.
      3. SHA-1 hash fallback — deterministic colour for ids whose
         texture isn't in the cache (modded blocks, missing assets).
    """
    if not block_id:
        return _UNKNOWN_COLOR
    if block_id in _PALETTE:
        return _PALETTE[block_id]
    auto = _colour_from_texture(block_id)
    if auto is not None:
        return auto
    # Hash-based fallback. Tuned to skew away from pure grey so it's
    # visually distinct from "unknown" cells. RGB derived from the
    # SHA-1 of the block id keeps the same block stable across runs.
    h = hashlib.sha1(block_id.encode("utf-8")).digest()
    r = 60 + (h[0] % 180)
    g = 60 + (h[1] % 180)
    b = 60 + (h[2] % 180)
    return (r, g, b)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

@dataclass
class MapRenderConfig:
    """Knobs for the top-down view."""
    # Output canvas size in pixels (square map area; header adds height).
    canvas_size_px: int = 540
    # World blocks visible across the canvas at zoom 1.0.
    blocks_across:  int = 64
    # Multiplier — >1 zooms in (fewer blocks shown bigger).
    zoom:           float = 1.0
    # Extra height for the pose/stat header strip.
    header_px:      int = 64
    # Background colour of the world area.
    background:     Tuple[int, int, int] = (24, 26, 30)
    # Grid line colour (subtle).
    grid:           Tuple[int, int, int] = (45, 50, 58)
    # Major-grid spacing (every N blocks).
    grid_step:      int = 16
    # Player marker colour + crosshair-target colour.
    player_color:   Tuple[int, int, int] = (255, 240, 100)
    target_color:   Tuple[int, int, int] = (255,  90,  90)
    # Vertical-slice depth around the player's feet (blocks). Cells
    # outside ``[feet_y - depth_down, feet_y + depth_up]`` are not
    # drawn, which keeps the view from being dominated by far-away
    # high blocks (e.g. the sky-island in a survival world).
    depth_up:       int = 8
    depth_down:     int = 24
    # When the same XZ cell has multiple Y observations we pick the
    # one closest to feet_y; cells further away fade.
    fade_per_block: float = 0.04


def _project_world_to_canvas(world_x: float, world_z: float,
                              center_x: float, center_z: float,
                              px_per_block: float,
                              canvas_size_px: int,
                              header_px: int
                              ) -> Tuple[int, int]:
    """Map a world (x, z) to a canvas pixel. World +Z is south, which
    we draw downward (i.e. canvas +y = world +z)."""
    cx = canvas_size_px * 0.5
    cy = header_px + canvas_size_px * 0.5
    px = int(round(cx + (world_x - center_x) * px_per_block))
    py = int(round(cy + (world_z - center_z) * px_per_block))
    return px, py


class WorldMapRenderer:
    """
    Renders a :class:`WorldMap` as a top-down RGB image.

    Construct once, call :meth:`render` whenever you want a fresh
    image. Stateless across renders — pass the latest pose every time.
    """

    def __init__(self, config: Optional[MapRenderConfig] = None):
        self.cfg = config or MapRenderConfig()

    # ── Public API ────────────────────────────────────────────────

    def render(self,
               world_map: WorldMap,
               pose: Optional[PlayerPose],
               *,
               target_voxel: Optional[Tuple[int, int, int]] = None,
               extra_lines: Optional[List[str]] = None,
               ) -> np.ndarray:
        """
        Return an ``(H, W, 3)`` uint8 RGB image of the world map.

        ``extra_lines`` are appended to the header — used by the live
        viewer to show stats like sample count and FPS.
        """
        cfg = self.cfg
        W = cfg.canvas_size_px
        H = cfg.canvas_size_px + cfg.header_px

        # Canvas + background.
        img = np.full((H, W, 3), cfg.background, dtype=np.uint8)

        # Pose defaults so the renderer always produces SOMETHING.
        if pose is not None:
            cx, cz = float(pose.x), float(pose.z)
            feet_y = int(pose.feet_block()[1])
        else:
            cx, cz, feet_y = 0.0, 0.0, 64

        px_per_block = max(1.0, (W / max(1, cfg.blocks_across)) * cfg.zoom)

        # 1. Grid.
        self._draw_grid(img, cx, cz, px_per_block)

        # 2. Blocks.
        self._draw_blocks(img, world_map, pose, cx, cz, feet_y, px_per_block)

        # 3. Crosshair-target voxel marker.
        if target_voxel is not None:
            self._draw_target(img, target_voxel, cx, cz, px_per_block)

        # 4. Player marker + yaw arrow.
        if pose is not None:
            self._draw_player(img, pose, cx, cz, px_per_block)

        # 5. Header text.
        self._draw_header(img, pose, target_voxel, extra_lines)

        return img

    # ── Drawing helpers ────────────────────────────────────────────

    def _draw_grid(self,
                   img: np.ndarray,
                   cx: float, cz: float,
                   px_per_block: float) -> None:
        cfg = self.cfg
        W = cfg.canvas_size_px
        canvas_top = cfg.header_px
        # Vertical lines (constant world x).
        # Find the integer x at the left edge.
        left_world_x = cx - (W * 0.5) / px_per_block
        first_x = int(math.floor(left_world_x / cfg.grid_step) * cfg.grid_step)
        x = first_x
        while True:
            px, _ = _project_world_to_canvas(x, cz, cx, cz, px_per_block,
                                              W, canvas_top)
            if px > W:
                break
            if 0 <= px < W:
                cv2.line(img, (px, canvas_top),
                         (px, canvas_top + W - 1),
                         cfg.grid, 1, lineType=cv2.LINE_AA)
            x += cfg.grid_step

        # Horizontal lines (constant world z).
        top_world_z = cz - (W * 0.5) / px_per_block
        first_z = int(math.floor(top_world_z / cfg.grid_step) * cfg.grid_step)
        z = first_z
        while True:
            _, py = _project_world_to_canvas(cx, z, cx, cz, px_per_block,
                                              W, canvas_top)
            if py > canvas_top + W:
                break
            if canvas_top <= py < canvas_top + W:
                cv2.line(img, (0, py), (W - 1, py),
                         cfg.grid, 1, lineType=cv2.LINE_AA)
            z += cfg.grid_step

    def _draw_blocks(self,
                     img: np.ndarray,
                     world_map: WorldMap,
                     pose: Optional[PlayerPose],
                     cx: float, cz: float,
                     feet_y: int,
                     px_per_block: float) -> None:
        cfg = self.cfg
        W = cfg.canvas_size_px
        canvas_top = cfg.header_px

        # Visible world-coord rectangle.
        half_blocks = (W * 0.5) / px_per_block
        x_min = math.floor(cx - half_blocks - 1)
        x_max = math.ceil(cx + half_blocks + 1)
        z_min = math.floor(cz - half_blocks - 1)
        z_max = math.ceil(cz + half_blocks + 1)
        y_min = feet_y - cfg.depth_down
        y_max = feet_y + cfg.depth_up

        dim = (pose.dimension if pose is not None
               else world_map.current_dimension())

        # Track best-Y per (x, z) so two stacked observations don't
        # overpaint each other randomly — we want the topmost solid
        # block in the depth window.
        best_at_xz: Dict[Tuple[int, int], Tuple[int, str]] = {}
        # Single pass over the whole sparse dimension store. Cheap
        # while block counts stay in the thousands.
        for obs in world_map.iter_blocks(dimension=dim):
            x, y, z = obs.pos
            if x < x_min or x > x_max or z < z_min or z > z_max:
                continue
            if y < y_min or y > y_max:
                continue
            if obs.block_id is None:
                continue
            prev = best_at_xz.get((x, z))
            # Prefer the highest y in the window (top-down view).
            if prev is None or y > prev[0]:
                best_at_xz[(x, z)] = (y, obs.block_id)

        # Splat each (x, z) → coloured square.
        side = max(1, int(round(px_per_block)))
        for (x, z), (y, block_id) in best_at_xz.items():
            px, py = _project_world_to_canvas(x + 0.5, z + 0.5,
                                               cx, cz, px_per_block,
                                               W, canvas_top)
            color = _color_for_block(block_id)
            # Fade for blocks far from feet_y so depth is visible.
            dy = abs(y - feet_y)
            fade = max(0.45, 1.0 - dy * cfg.fade_per_block)
            color = (int(color[0] * fade),
                     int(color[1] * fade),
                     int(color[2] * fade))
            x0 = px - side // 2
            y0 = py - side // 2
            x1 = x0 + side
            y1 = y0 + side
            x0 = max(0, x0); y0 = max(canvas_top, y0)
            x1 = min(W, x1); y1 = min(canvas_top + W, y1)
            if x1 > x0 and y1 > y0:
                img[y0:y1, x0:x1] = color

    def _draw_target(self,
                     img: np.ndarray,
                     voxel: Tuple[int, int, int],
                     cx: float, cz: float,
                     px_per_block: float) -> None:
        cfg = self.cfg
        W = cfg.canvas_size_px
        canvas_top = cfg.header_px
        px, py = _project_world_to_canvas(voxel[0] + 0.5,
                                           voxel[2] + 0.5,
                                           cx, cz, px_per_block,
                                           W, canvas_top)
        if 0 <= px < W and canvas_top <= py < canvas_top + W:
            side = max(3, int(round(px_per_block)) + 2)
            cv2.rectangle(img,
                          (px - side // 2, py - side // 2),
                          (px + side // 2, py + side // 2),
                          cfg.target_color, 1, lineType=cv2.LINE_AA)

    def _draw_player(self,
                     img: np.ndarray,
                     pose: PlayerPose,
                     cx: float, cz: float,
                     px_per_block: float) -> None:
        cfg = self.cfg
        W = cfg.canvas_size_px
        canvas_top = cfg.header_px
        px, py = _project_world_to_canvas(pose.x, pose.z,
                                           cx, cz, px_per_block,
                                           W, canvas_top)

        # Player dot.
        cv2.circle(img, (px, py), 5, cfg.player_color, -1,
                   lineType=cv2.LINE_AA)

        # Yaw arrow.
        # MC yaw: 0 = +Z (south, screen-down); CW from above.
        # Vector (dx, dz) = (-sin(yaw), cos(yaw)).
        yaw_rad = math.radians(pose.yaw)
        dx = -math.sin(yaw_rad)
        dz = math.cos(yaw_rad)
        # Arrow tip 14 pixels in front of player.
        length_px = 18
        tip_x = int(px + dx * length_px)
        tip_y = int(py + dz * length_px)
        cv2.arrowedLine(img, (px, py), (tip_x, tip_y),
                        cfg.player_color, 2, line_type=cv2.LINE_AA,
                        tipLength=0.35)

    def _draw_header(self,
                     img: np.ndarray,
                     pose: Optional[PlayerPose],
                     target_voxel: Optional[Tuple[int, int, int]],
                     extra_lines: Optional[List[str]]) -> None:
        cfg = self.cfg
        # Header background.
        img[: cfg.header_px] = (15, 17, 20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        white = (235, 235, 235)
        dim   = (160, 165, 175)
        cv2.putText(img, "World Map", (10, 22), font, 0.62, white, 1,
                    cv2.LINE_AA)
        if pose is not None:
            pose_line = (f"X={pose.x:7.1f}  Y={pose.y:6.1f}  Z={pose.z:7.1f}"
                         f"   yaw={pose.yaw:6.1f}   pitch={pose.pitch:5.1f}"
                         f"   dim={pose.dimension.split(':')[-1]}")
        else:
            pose_line = "no pose (F3 not visible)"
        cv2.putText(img, pose_line, (10, 42), font, 0.42, dim, 1,
                    cv2.LINE_AA)

        third_line_parts: List[str] = []
        if target_voxel is not None:
            third_line_parts.append(f"target={target_voxel}")
        if extra_lines:
            third_line_parts.extend(extra_lines)
        if third_line_parts:
            cv2.putText(img, "   ".join(third_line_parts), (10, 60),
                        font, 0.42, dim, 1, cv2.LINE_AA)


__all__ = [
    "WorldMapRenderer",
    "MapRenderConfig",
]
