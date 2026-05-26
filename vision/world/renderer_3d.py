# vision/world/renderer_3d.py
"""
Isometric 3D rendering of the AI's :class:`WorldMap`.

This is the canonical "look what the AI thinks the world looks like"
view. Unlike the top-down ``map_renderer`` which collapses every
voxel column to a single coloured cell, this renderer shows each
known voxel as a proper isometric cube — three visible faces, light
+ dark shading, depth ordering — so the user (and any future debug
tool) can directly compare the AI's mental 3D model to the actual
game view.

What's drawn
------------
* Solid voxels: coloured isometric cubes (top face + two side faces
  with different shading). Colour is the block-id palette from
  :mod:`vision.world.map_renderer`.
* Air voxels: omitted (free space carved by the F3-sightline air
  marker is a NEGATIVE signal — useful internally but visually
  cluttering if drawn).
* The player: a small humanoid sprite centred on the player's voxel.
* The targeted voxel: highlighted with a thicker outline.
* A faint ground-plane grid for orientation.
* A header strip with pose + counts + the currently-pinned block id.

Why isometric and not first-person
----------------------------------
A first-person re-projection from the player's pose would be the most
"immersive" debug view, but it requires a depth buffer and faces the
same monocular-depth problem as the perception layer itself. An
isometric view sidesteps both: it's purely geometric, every known
voxel projects to a unique 2-D position, and depth ordering is
trivial (further voxels behind nearer ones, painter's algorithm by
``x + y + z``).

Performance budget
------------------
At the default zoom + 12-block render radius, the renderer touches
roughly 25 × 25 × 25 ≈ 15 000 voxels — but in practice the AI has
only seen a few hundred, so each tick produces ≤ 1 ms of work after
the WorldMap query. Suitable for the live viewer loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from vision.world.map import WorldMap, AIR_BLOCK
from vision.world.map_renderer import _color_for_block
from vision.world.types import PlayerPose


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class IsoRenderConfig:
    canvas_w_px:    int   = 720
    canvas_h_px:    int   = 540
    header_h_px:    int   = 60

    # Pixels per world block at zoom 1.0. The cube side is then this
    # value × cos(30°) for the projected width and × sin(30°) for the
    # projected height. Bumped up so even a small map (one cube)
    # is clearly visible on screen.
    base_px_per_block: float = 18.0
    zoom:           float = 1.0

    # Render only voxels within this Chebyshev radius of the player.
    # 12 blocks is enough to see the immediate surroundings without
    # over-cluttering the canvas.
    render_radius_blocks: int = 12

    # Vertical layers above + below the player to include. The top
    # layer is also clamped to ``render_radius_blocks`` so a tall
    # cliff doesn't fill the screen.
    layers_above: int = 6
    layers_below: int = 12

    # Colours.
    background:     Tuple[int, int, int] = (24, 26, 30)
    ground_grid:    Tuple[int, int, int] = (40, 46, 56)
    edge_dark:      Tuple[int, int, int] = (15, 17, 20)
    # The player is rendered as a SOFT BLUE humanoid so it can never
    # be mistaken for a yellow-toned block (sand, sandstone, etc.).
    player_color:   Tuple[int, int, int] = ( 95, 175, 255)
    target_color:   Tuple[int, int, int] = (255,  90,  90)
    # Distinct outline for F3-confirmed blocks (highest confidence)
    # so the user can immediately tell which voxels are ground truth
    # vs sample-NN-confidence committed.
    confirmed_outline: Tuple[int, int, int] = (255, 220,  90)


# ---------------------------------------------------------------------------
# Isometric projection helpers
# ---------------------------------------------------------------------------

# Standard isometric: rotate world such that
#   screen_x = (world_x - world_z) * cos(30°)
#   screen_y = (world_x + world_z) * sin(30°) - world_y
# This gives "true" isometric (equal foreshortening on all three axes).
_COS30 = math.cos(math.radians(30.0))
_SIN30 = math.sin(math.radians(30.0))


def _project_iso(world_xyz: Tuple[float, float, float],
                 center_xyz: Tuple[float, float, float],
                 px_per_block: float,
                 origin_px: Tuple[float, float],
                 ) -> Tuple[float, float]:
    """Project a world (x, y, z) to a canvas (px, py)."""
    cx, cy, cz = center_xyz
    wx = world_xyz[0] - cx
    wy = world_xyz[1] - cy
    wz = world_xyz[2] - cz
    sx = (wx - wz) * _COS30 * px_per_block
    sy = ((wx + wz) * _SIN30 - wy) * px_per_block
    return origin_px[0] + sx, origin_px[1] + sy


def _shade(color: Tuple[int, int, int], factor: float
           ) -> Tuple[int, int, int]:
    """Multiply each channel by ``factor``, clamped to [0, 255]."""
    return (int(max(0, min(255, color[0] * factor))),
            int(max(0, min(255, color[1] * factor))),
            int(max(0, min(255, color[2] * factor))))


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class IsoWorldRenderer:
    """
    Render a :class:`WorldMap` to an isometric voxel image.

    Stateless across calls; pass the latest pose every render.
    """

    # Lighting factors for the three visible cube faces. Tuned so the
    # contrast reads cleanly without making any face appear black.
    _TOP_FACE_LIGHT   = 1.10
    _RIGHT_FACE_LIGHT = 0.78
    _LEFT_FACE_LIGHT  = 0.55

    def __init__(self, config: Optional[IsoRenderConfig] = None):
        self.cfg = config or IsoRenderConfig()

    # ── Public API ────────────────────────────────────────────────

    def render(self,
               world_map: WorldMap,
               pose: Optional[PlayerPose],
               *,
               target_voxel: Optional[Tuple[int, int, int]] = None,
               extra_lines: Optional[List[str]] = None,
               ) -> np.ndarray:
        cfg = self.cfg
        W = cfg.canvas_w_px
        H = cfg.canvas_h_px + cfg.header_h_px
        img = np.full((H, W, 3), cfg.background, dtype=np.uint8)

        if pose is not None:
            center = (float(pose.x),
                      float(pose.feet_block()[1]),
                      float(pose.z))
            dim = pose.dimension
        else:
            center = (0.0, 64.0, 0.0)
            dim = world_map.current_dimension()

        px_per_block = cfg.base_px_per_block * cfg.zoom
        # Place the origin at canvas centre but biased UP so taller
        # terrain has room.
        origin_px = (W * 0.5,
                     cfg.header_h_px + cfg.canvas_h_px * 0.55)

        # 1. Faint ground-plane grid centred on the player's feet.
        self._draw_ground_grid(img, center, px_per_block, origin_px)

        # 2. Voxels (painter's algorithm).
        self._draw_voxels(img, world_map, pose, center, px_per_block,
                          origin_px, target_voxel=target_voxel,
                          dimension=dim)

        # 3. Player marker (on top of voxels at the player's voxel
        # position).
        if pose is not None:
            self._draw_player(img, pose, center, px_per_block, origin_px)

        # 4. Header.
        self._draw_header(img, pose, target_voxel, extra_lines, world_map)

        return img

    # ── Internals ─────────────────────────────────────────────────

    def _draw_ground_grid(self,
                          img: np.ndarray,
                          center: Tuple[float, float, float],
                          px_per_block: float,
                          origin_px: Tuple[float, float]) -> None:
        cfg = self.cfg
        feet_y = center[1]
        r = cfg.render_radius_blocks
        # Two families of parallel lines on the y = feet plane.
        for k in range(-r, r + 1, 2):
            p0 = _project_iso((center[0] + k, feet_y, center[2] - r),
                              center, px_per_block, origin_px)
            p1 = _project_iso((center[0] + k, feet_y, center[2] + r),
                              center, px_per_block, origin_px)
            cv2.line(img, (int(p0[0]), int(p0[1])),
                     (int(p1[0]), int(p1[1])),
                     cfg.ground_grid, 1, lineType=cv2.LINE_AA)
            p0 = _project_iso((center[0] - r, feet_y, center[2] + k),
                              center, px_per_block, origin_px)
            p1 = _project_iso((center[0] + r, feet_y, center[2] + k),
                              center, px_per_block, origin_px)
            cv2.line(img, (int(p0[0]), int(p0[1])),
                     (int(p1[0]), int(p1[1])),
                     cfg.ground_grid, 1, lineType=cv2.LINE_AA)

    def _draw_voxels(self,
                     img: np.ndarray,
                     world_map: WorldMap,
                     pose: Optional[PlayerPose],
                     center: Tuple[float, float, float],
                     px_per_block: float,
                     origin_px: Tuple[float, float],
                     *,
                     target_voxel: Optional[Tuple[int, int, int]],
                     dimension: str) -> None:
        cfg = self.cfg
        cx = int(math.floor(center[0]))
        cy = int(math.floor(center[1]))
        cz = int(math.floor(center[2]))
        r = cfg.render_radius_blocks

        # Painter's algorithm: a voxel at (x, y, z) is occluded by
        # voxels with strictly larger x+y+z (further from the viewer
        # in our isometric setup). Sort by x+y+z ascending and draw
        # back-to-front so nearer cubes paint over further ones.
        candidates: List[Tuple[int, int, int]] = []
        for obs in world_map.iter_solid_blocks(dimension=dimension):
            x, y, z = obs.pos
            if abs(x - cx) > r or abs(z - cz) > r:
                continue
            if not (cy - cfg.layers_below <= y <= cy + cfg.layers_above):
                continue
            candidates.append(obs.pos)
        # Sort by depth: smaller x+y+z = further away → drawn first.
        candidates.sort(key=lambda p: (-p[0] - p[2] + p[1]))

        for pos in candidates:
            obs = world_map.get_block(pos, dimension=dimension)
            if obs is None or not obs.block_id:
                continue
            base = _color_for_block(obs.block_id)
            top   = _shade(base, self._TOP_FACE_LIGHT)
            right = _shade(base, self._RIGHT_FACE_LIGHT)
            left  = _shade(base, self._LEFT_FACE_LIGHT)
            is_target    = (pos == target_voxel)
            is_confirmed = (obs.source == "looking_at"
                             or obs.source == "manual")
            self._draw_cube(img, pos, center, px_per_block, origin_px,
                             top=top, right=right, left=left,
                             outline_target=is_target,
                             outline_confirmed=is_confirmed)

    def _draw_cube(self,
                   img: np.ndarray,
                   voxel: Tuple[int, int, int],
                   center: Tuple[float, float, float],
                   px_per_block: float,
                   origin_px: Tuple[float, float],
                   *,
                   top: Tuple[int, int, int],
                   right: Tuple[int, int, int],
                   left: Tuple[int, int, int],
                   outline_target: bool,
                   outline_confirmed: bool = False) -> None:
        x, y, z = voxel
        # Eight corners of the unit cube at integer voxel pos.
        # Convention: y is "up" (so y is the higher cube face).
        c = [
            _project_iso((x,     y,     z),     center, px_per_block, origin_px),
            _project_iso((x + 1, y,     z),     center, px_per_block, origin_px),
            _project_iso((x + 1, y,     z + 1), center, px_per_block, origin_px),
            _project_iso((x,     y,     z + 1), center, px_per_block, origin_px),
            _project_iso((x,     y + 1, z),     center, px_per_block, origin_px),
            _project_iso((x + 1, y + 1, z),     center, px_per_block, origin_px),
            _project_iso((x + 1, y + 1, z + 1), center, px_per_block, origin_px),
            _project_iso((x,     y + 1, z + 1), center, px_per_block, origin_px),
        ]
        # Map to int tuples for cv2.
        ip = [(int(round(p[0])), int(round(p[1]))) for p in c]

        # Visible faces in our convention (positive x = forward-right,
        # positive z = forward-left, positive y = up):
        #   top face : corners 4,5,6,7
        #   right face (+x): 1,2,6,5
        #   left face  (+z): 3,2,6,7
        top_face   = np.array([ip[4], ip[5], ip[6], ip[7]], dtype=np.int32)
        right_face = np.array([ip[1], ip[2], ip[6], ip[5]], dtype=np.int32)
        left_face  = np.array([ip[3], ip[2], ip[6], ip[7]], dtype=np.int32)

        cv2.fillConvexPoly(img, left_face,  left,  lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(img, right_face, right, lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(img, top_face,   top,   lineType=cv2.LINE_AA)

        # Outline priority: red target box > gold confirmed outline >
        # plain dark edge. Thickness scales similarly.
        if outline_target:
            edge = self.cfg.target_color
            thickness = 2
        elif outline_confirmed:
            edge = self.cfg.confirmed_outline
            thickness = 2
        else:
            edge = self.cfg.edge_dark
            thickness = 1
        # Three visible edge silhouettes.
        cv2.polylines(img, [top_face],   True, edge, thickness, cv2.LINE_AA)
        cv2.polylines(img, [right_face], True, edge, thickness, cv2.LINE_AA)
        cv2.polylines(img, [left_face],  True, edge, thickness, cv2.LINE_AA)

    def _draw_player(self,
                     img: np.ndarray,
                     pose: PlayerPose,
                     center: Tuple[float, float, float],
                     px_per_block: float,
                     origin_px: Tuple[float, float]) -> None:
        cfg = self.cfg
        # Player feet position projected to canvas.
        feet = _project_iso((pose.x, pose.y, pose.z),
                             center, px_per_block, origin_px)
        head = _project_iso((pose.x, pose.y + 1.8, pose.z),
                             center, px_per_block, origin_px)
        # Body as a vertical line + head circle, yaw arrow on ground.
        cv2.line(img,
                 (int(feet[0]), int(feet[1])),
                 (int(head[0]), int(head[1])),
                 cfg.player_color, 3, lineType=cv2.LINE_AA)
        cv2.circle(img, (int(head[0]), int(head[1])),
                   max(3, int(px_per_block * 0.35)),
                   cfg.player_color, -1, lineType=cv2.LINE_AA)

        # Yaw arrow on the ground plane.
        yaw_rad = math.radians(pose.yaw)
        dx = -math.sin(yaw_rad)
        dz = math.cos(yaw_rad)
        tip_world = (pose.x + dx * 2.0, pose.y, pose.z + dz * 2.0)
        tip = _project_iso(tip_world, center, px_per_block, origin_px)
        cv2.arrowedLine(img,
                        (int(feet[0]), int(feet[1])),
                        (int(tip[0]),  int(tip[1])),
                        cfg.player_color, 2, line_type=cv2.LINE_AA,
                        tipLength=0.35)

    def _draw_header(self,
                     img: np.ndarray,
                     pose: Optional[PlayerPose],
                     target_voxel: Optional[Tuple[int, int, int]],
                     extra_lines: Optional[List[str]],
                     world_map: WorldMap) -> None:
        cfg = self.cfg
        img[: cfg.header_h_px] = (15, 17, 20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        white = (235, 235, 235)
        dim   = (160, 165, 175)
        # cv2's Hershey fonts don't have a glyph for the em-dash, so
        # we use an ASCII hyphen here to avoid the renderer's `???`
        # fallback in the output PNG.
        cv2.putText(img, "World Map - isometric (3D, F3-only)", (10, 22),
                    font, 0.58, white, 1, cv2.LINE_AA)
        if pose is not None:
            pose_line = (f"X={pose.x:7.1f}  Y={pose.y:6.1f}  Z={pose.z:7.1f}"
                         f"   yaw={pose.yaw:6.1f}   pitch={pose.pitch:5.1f}"
                         f"   dim={pose.dimension.split(':')[-1]}")
        else:
            pose_line = "no pose"
        cv2.putText(img, pose_line, (10, 42), font, 0.42, dim, 1,
                    cv2.LINE_AA)
        if extra_lines:
            cv2.putText(img, "   ".join(extra_lines), (10, 58),
                        font, 0.42, dim, 1, cv2.LINE_AA)
        # Mini-legend top right — gives the user the colour code
        # for what's on the map.
        legend_x = max(20, img.shape[1] - 360)
        cv2.putText(img,
                    "player=blue  F3-confirmed=gold outline  target=red box",
                    (legend_x, 22), font, 0.40, dim, 1, cv2.LINE_AA)


__all__ = ["IsoWorldRenderer", "IsoRenderConfig"]
