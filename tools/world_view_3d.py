# tools/world_view_3d.py
"""
Interactive 3D viewer for a saved WorldMap.

Loads a compact-JSON world dump (``data/calibration/world_map_*.json``)
or any other JSON in the ``minecraft_ai_world_v1`` format and lets you
orbit, pan, zoom around the scene with the mouse.

Controls
--------
  Left-mouse drag    : orbit (yaw + pitch around the target)
  Right-mouse drag   : pan (move target across the view plane)
  Mouse wheel        : dolly zoom (distance to target)
  W / S              : forward / backward zoom (alt for wheel)
  A / D              : pan target left / right (in world XZ)
  Q / E              : pan target down / up (Y axis)
  R                  : reset view to fit all voxels
  F                  : focus on player position (if saved)
  H                  : toggle the help overlay
  ESC                : quit

Run
---
  # latest saved map under data/calibration/
  python tools/world_view_3d.py

  # explicit map
  python tools/world_view_3d.py --json path/to/world_map_<ts>.json

The renderer is pure NumPy + OpenCV — no GPU, no extra dependencies.
For typical session sizes (~500-5000 voxels) it manages ~10-30 FPS on
the user's hardware, which is plenty for inspection. Sort + draw is a
simple painter's algorithm with back-face culling.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.world.map_renderer import _color_for_block


# ---------------------------------------------------------------------------
# WorldMap JSON loader
# ---------------------------------------------------------------------------


@dataclass
class LoadedWorld:
    voxels: np.ndarray              # (N, 3) int — block coordinates
    block_ids: List[str]            # len N — palette-resolved ids
    palette: List[str]              # the unique block ids
    palette_colors: np.ndarray      # (P, 3) uint8 — RGB per palette idx
    palette_idx: np.ndarray         # (N,) int — index into palette per voxel
    dimension: Optional[str]
    player_pos: Optional[Tuple[float, float, float]] = None

    @property
    def n(self) -> int:
        return self.voxels.shape[0]


def _resolve_default_path() -> Optional[Path]:
    """If no path is supplied, pick the most recently-modified
    world_map_*.json under data/calibration/ — typical interactive
    workflow."""
    cal = ROOT / "data" / "calibration"
    if not cal.is_dir():
        return None
    candidates = sorted(cal.glob("world_map_*.json"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True)
    return candidates[0] if candidates else None


# Blocks the AI marks as carved AIR. They aren't real geometry — render
# them as a translucent "ghost" so the cleared-space volume is visible
# without obscuring solid blocks.
AIR_BLOCK_IDS = frozenset({"minecraft:_carved_air", "minecraft:air"})


def load_compact_json(path: Path,
                      *,
                      include_air: bool = False,
                      ) -> LoadedWorld:
    """Parse a ``minecraft_ai_world_v1`` JSON dump into a LoadedWorld."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    fmt = d.get("format")
    if fmt != "minecraft_ai_world_v1":
        print(f"[viewer][WARN] unexpected format tag {fmt!r} "
              f"(expected minecraft_ai_world_v1); attempting to parse "
              f"anyway", file=sys.stderr)

    palette: List[str] = list(d.get("palette") or [])
    blocks: List[List[int]] = list(d.get("blocks") or [])

    # Drop carved-air voxels by default — they aren't solid geometry.
    air_indices = {i for i, bid in enumerate(palette) if bid in AIR_BLOCK_IDS}
    if include_air:
        air_indices = set()

    voxel_list: List[Tuple[int, int, int]] = []
    palette_idx_list: List[int] = []
    for row in blocks:
        if len(row) < 4:
            continue
        x, y, z, pi = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        if pi in air_indices:
            continue
        voxel_list.append((x, y, z))
        palette_idx_list.append(pi)

    if not voxel_list:
        raise ValueError(f"No solid blocks in {path}")

    voxels = np.array(voxel_list, dtype=np.int32)
    palette_idx = np.array(palette_idx_list, dtype=np.int32)

    # Resolve palette colors once. RGB triples in 0..255 (uint8). We
    # translate to BGR at draw time via cv2 conventions.
    palette_colors = np.zeros((len(palette), 3), dtype=np.uint8)
    for i, bid in enumerate(palette):
        if bid is None:
            palette_colors[i] = (160, 160, 160)
        else:
            rgb = _color_for_block(bid)
            palette_colors[i] = (rgb[0], rgb[1], rgb[2])

    block_ids = [palette[pi] if pi < len(palette) else "?"
                 for pi in palette_idx_list]

    # If a sibling .json exists that records the last-pose (some future
    # exporter may emit this; today we just leave it empty), pull it
    # so 'F' can focus on the player. Today the player_pos stays None.
    player_pos: Optional[Tuple[float, float, float]] = None
    pose = d.get("player_pose")
    if isinstance(pose, dict) and all(k in pose for k in ("x", "y", "z")):
        player_pos = (float(pose["x"]), float(pose["y"]), float(pose["z"]))

    return LoadedWorld(
        voxels=voxels,
        block_ids=block_ids,
        palette=palette,
        palette_colors=palette_colors,
        palette_idx=palette_idx,
        dimension=d.get("dimension"),
        player_pos=player_pos,
    )


# ---------------------------------------------------------------------------
# Camera + projection
# ---------------------------------------------------------------------------


@dataclass
class Camera:
    target: np.ndarray = None     # (3,) float, world-space focus point
    yaw_deg: float = 35.0         # rotation around Y
    pitch_deg: float = -25.0      # rotation up/down (negative = looking down)
    radius: float = 30.0          # distance from target to camera
    fov_deg: float = 60.0         # vertical field of view

    def __post_init__(self):
        if self.target is None:
            self.target = np.array([0.0, 64.0, 0.0], dtype=np.float64)
        else:
            self.target = np.asarray(self.target, dtype=np.float64)

    def position(self) -> np.ndarray:
        yaw_r = math.radians(self.yaw_deg)
        pit_r = math.radians(self.pitch_deg)
        cos_p = math.cos(pit_r)
        x = self.radius * cos_p * math.sin(yaw_r)
        y = self.radius * math.sin(pit_r)
        z = self.radius * cos_p * math.cos(yaw_r)
        return self.target + np.array([x, y, z], dtype=np.float64)

    def view_axes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (right, up, forward) unit vectors in world space.
        ``forward`` points from camera toward target."""
        cam = self.position()
        fwd = self.target - cam
        n = np.linalg.norm(fwd)
        if n < 1e-9:
            fwd = np.array([0.0, 0.0, 1.0])
        else:
            fwd = fwd / n
        up_world = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, up_world)
        rn = np.linalg.norm(right)
        if rn < 1e-9:
            # Camera looking straight down/up — pick an arbitrary right.
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / rn
        up = np.cross(right, fwd)
        return right, up, fwd


def _project_points(points: np.ndarray,
                    cam: Camera,
                    canvas_w: int,
                    canvas_h: int,
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perspective-project an array of world points to screen space.

    Returns ``(screen_xy, depth)`` where ``screen_xy`` is (N, 2) in
    pixels and ``depth`` is (N,) — distance from camera along the
    forward axis. Points behind the camera get depth < 0.
    """
    cam_pos = cam.position()
    right, up, fwd = cam.view_axes()

    rel = points - cam_pos
    # Camera-space: (x = right, y = up, z = forward).
    cam_x = rel @ right
    cam_y = rel @ up
    cam_z = rel @ fwd

    fov_rad = math.radians(cam.fov_deg)
    focal = (canvas_h * 0.5) / math.tan(fov_rad * 0.5)

    # Guard against z=0; the painter's-algorithm caller will cull
    # anything with depth <= 0 anyway, but the projection must not NaN.
    eps = 1e-3
    z_safe = np.where(np.abs(cam_z) < eps, eps, cam_z)

    sx = canvas_w * 0.5 + focal * cam_x / z_safe
    sy = canvas_h * 0.5 - focal * cam_y / z_safe

    return np.stack([sx, sy], axis=1), cam_z


# ---------------------------------------------------------------------------
# Voxel face geometry
# ---------------------------------------------------------------------------

# Each face: (outward normal, 4 corner offsets in unit-cube space).
# Order of corners is consistent CCW seen from OUTSIDE the cube so
# back-face culling via the sign of (normal . view) is unambiguous.
_FACES: List[Tuple[np.ndarray, np.ndarray, str]] = [
    # +Y (top)
    (np.array([0, 1, 0]),
     np.array([[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=np.float64),
     "top"),
    # -Y (bottom)
    (np.array([0, -1, 0]),
     np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0], [1, 0, 1]], dtype=np.float64),
     "bottom"),
    # +X (east)
    (np.array([1, 0, 0]),
     np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]], dtype=np.float64),
     "east"),
    # -X (west)
    (np.array([-1, 0, 0]),
     np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=np.float64),
     "west"),
    # +Z (south)
    (np.array([0, 0, 1]),
     np.array([[1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=np.float64),
     "south"),
    # -Z (north)
    (np.array([0, 0, -1]),
     np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float64),
     "north"),
]


# Per-face brightness multiplier — top brightest, sides medium, bottom
# darkest. Matches the existing iso renderer's lighting feel.
_FACE_LIGHT: Dict[str, float] = {
    "top":    1.00,
    "north":  0.82,
    "south":  0.82,
    "east":   0.74,
    "west":   0.74,
    "bottom": 0.55,
}


def _build_face_database(world: LoadedWorld) -> Dict[str, np.ndarray]:
    """
    Pre-compute every face's world-space corners + per-face metadata
    once at load time. The render loop just transforms + sorts them.

    Returns a dict with:
      corners  : (F, 4, 3) float — face-corner world positions
      normals  : (F, 3) float — outward unit normals
      colors   : (F, 3) uint8 — base RGB per face (after lighting)
      depths   : (F,) float — scratch buffer reused each frame
    """
    n_voxels = world.n
    n_faces = n_voxels * 6
    corners = np.zeros((n_faces, 4, 3), dtype=np.float64)
    normals = np.zeros((n_faces, 3), dtype=np.float64)
    colors  = np.zeros((n_faces, 3), dtype=np.uint8)
    face_idx = 0
    voxels_f = world.voxels.astype(np.float64)
    for face_normal, face_offsets, face_name in _FACES:
        light = _FACE_LIGHT[face_name]
        # Broadcast: (N, 1, 3) + (4, 3) -> (N, 4, 3)
        corners_for_face = (voxels_f[:, None, :]
                             + face_offsets[None, :, :])
        block_color = world.palette_colors[world.palette_idx]
        lit = np.clip(block_color.astype(np.float32) * light, 0, 255).astype(np.uint8)
        corners[face_idx:face_idx + n_voxels] = corners_for_face
        normals[face_idx:face_idx + n_voxels] = face_normal
        colors[face_idx:face_idx + n_voxels] = lit
        face_idx += n_voxels

    return {
        "corners":  corners,
        "normals":  normals,
        "colors":   colors,
    }


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

# Background gradient endpoints (BGR, since OpenCV draws in BGR).
_BG_TOP_BGR    = np.array([ 80,  60,  50], dtype=np.uint8)
_BG_BOTTOM_BGR = np.array([ 35,  25,  20], dtype=np.uint8)


def _make_background(h: int, w: int) -> np.ndarray:
    """Vertical gradient — cheap, gives the scene a horizon."""
    col = np.linspace(_BG_TOP_BGR, _BG_BOTTOM_BGR, h).astype(np.uint8)
    return np.broadcast_to(col[:, None, :], (h, w, 3)).copy()


def render(world: LoadedWorld,
           faces: Dict[str, np.ndarray],
           cam: Camera,
           canvas_w: int,
           canvas_h: int,
           *,
           show_outline: bool = True,
           ) -> np.ndarray:
    """Draw one frame and return an HxWx3 BGR image."""
    canvas = _make_background(canvas_h, canvas_w)

    corners = faces["corners"]   # (F, 4, 3)
    normals = faces["normals"]   # (F, 3)
    colors  = faces["colors"]    # (F, 3)
    F = corners.shape[0]

    # Project all face-corners in one batch for speed.
    flat = corners.reshape(-1, 3)
    screen_xy, depth = _project_points(flat, cam, canvas_w, canvas_h)
    screen_xy = screen_xy.reshape(F, 4, 2)
    depth     = depth.reshape(F, 4)

    # Per-face centroid depth — used for painter's sort.
    centroid_depth = depth.mean(axis=1)

    # Visibility filters:
    # 1. ALL corners must be in front of camera (depth > 0). Drop
    #    faces with any negative-z corner so they don't blow up the
    #    projection.
    all_front = (depth > 0.0).all(axis=1)

    # 2. Back-face cull. A face is back-facing if (normal . view_dir)
    #    >= 0, where view_dir points FROM camera to face. For a unit
    #    cube face the centroid offset from voxel center is along the
    #    face normal; equivalently, dot(normal, face_centroid -
    #    camera_pos) > 0 means facing away.
    cam_pos = cam.position()
    centroids = corners.mean(axis=1)                 # (F, 3)
    to_face = centroids - cam_pos
    facing_away = (normals * to_face).sum(axis=1) > 0
    visible = all_front & (~facing_away)

    if not visible.any():
        return canvas

    # Painter's algorithm: draw far-back faces first.
    idx_visible = np.where(visible)[0]
    order = idx_visible[np.argsort(-centroid_depth[idx_visible])]

    # Pre-clamp pixel coordinates to a sane range to dodge int32
    # overflow in cv2 when a near-camera face projects beyond the
    # canvas. ±1e6 is way past any reasonable monitor.
    screen_xy_clamped = np.clip(screen_xy, -1e6, 1e6)

    outline_color = (10, 10, 10)
    for fi in order:
        pts = screen_xy_clamped[fi].astype(np.int32)
        col = colors[fi]
        # cv2 wants BGR; our palette is RGB.
        bgr = (int(col[2]), int(col[1]), int(col[0]))
        cv2.fillConvexPoly(canvas, pts, bgr, lineType=cv2.LINE_AA)
        if show_outline:
            cv2.polylines(canvas, [pts], isClosed=True, color=outline_color,
                          thickness=1, lineType=cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Help overlay
# ---------------------------------------------------------------------------


def _draw_help(canvas: np.ndarray, cam: Camera, world: LoadedWorld,
               fps: float, show_help: bool) -> None:
    h, w = canvas.shape[:2]
    # Tight stats line — always shown.
    stats = (f"vox={world.n}  yaw={cam.yaw_deg:+6.1f}  "
             f"pitch={cam.pitch_deg:+6.1f}  r={cam.radius:5.1f}  "
             f"target=({cam.target[0]:+.1f}, {cam.target[1]:+.1f}, "
             f"{cam.target[2]:+.1f})  fps={fps:4.1f}")
    cv2.putText(canvas, stats, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (230, 230, 230), 1, cv2.LINE_AA)
    if not show_help:
        cv2.putText(canvas, "H for help",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 180, 180), 1, cv2.LINE_AA)
        return

    # Help panel.
    lines = [
        "Mouse:   L-drag orbit  |  R-drag pan  |  wheel zoom",
        "Keys:    W/S zoom  |  A/D pan X  |  Q/E pan Y",
        "         R reset  |  F focus player  |  H toggle help",
        "         O outlines  |  ESC quit",
    ]
    y = 20
    for line in lines:
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (220, 220, 220), 1, cv2.LINE_AA)
        y += 16


# ---------------------------------------------------------------------------
# Mouse callback state
# ---------------------------------------------------------------------------


class _MouseState:
    def __init__(self) -> None:
        self.left_down = False
        self.right_down = False
        self.last_x = 0
        self.last_y = 0
        # Sensitivity tuning. Mouse-pixel deltas multiply these into
        # camera state.
        self.orbit_sens   = 0.4     # deg per pixel
        self.pan_sens     = 0.04    # world units per pixel (scaled by radius)
        self.zoom_sens    = 1.15    # multiplicative factor per scroll notch


# ---------------------------------------------------------------------------
# Main viewer loop
# ---------------------------------------------------------------------------


def _fit_camera_to_world(cam: Camera, world: LoadedWorld) -> None:
    """Set camera target to the world centroid and radius to encompass
    the bounding sphere of all voxels."""
    if world.n == 0:
        return
    mins = world.voxels.min(axis=0)
    maxs = world.voxels.max(axis=0)
    centre = (mins + maxs) / 2.0 + 0.5
    cam.target = centre.astype(np.float64)
    extent = (maxs - mins).max() + 2.0
    cam.radius = max(8.0, float(extent) * 1.4)


def run_viewer(world: LoadedWorld,
               *,
               canvas_w: int = 1280,
               canvas_h: int = 800,
               window_name: str = "WorldMap 3D",
               ) -> None:
    """Open the OpenCV window and run the event loop until ESC."""
    cam = Camera()
    _fit_camera_to_world(cam, world)
    faces = _build_face_database(world)
    print(f"[viewer] loaded {world.n} voxels, "
          f"{faces['corners'].shape[0]} faces. "
          f"Window: {canvas_w}x{canvas_h}. Press H for help.")

    mouse = _MouseState()
    dirty = [True]   # re-render flag, manipulated by the mouse callback
    show_help = [True]
    show_outline = [True]

    def on_mouse(event, x, y, flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse.left_down = True
            mouse.last_x, mouse.last_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            mouse.left_down = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            mouse.right_down = True
            mouse.last_x, mouse.last_y = x, y
        elif event == cv2.EVENT_RBUTTONUP:
            mouse.right_down = False
        elif event == cv2.EVENT_MOUSEMOVE:
            dx = x - mouse.last_x
            dy = y - mouse.last_y
            if mouse.left_down:
                cam.yaw_deg   = (cam.yaw_deg - dx * mouse.orbit_sens) % 360.0
                cam.pitch_deg = max(-89.0, min(89.0,
                                                cam.pitch_deg + dy * mouse.orbit_sens))
                dirty[0] = True
            elif mouse.right_down:
                # Pan: shift target along the camera's right/up axes,
                # scaled by radius so pan-speed feels constant as you
                # zoom in / out.
                right, up, _ = cam.view_axes()
                world_dx = (-dx) * mouse.pan_sens * (cam.radius / 30.0)
                world_dy = ( dy) * mouse.pan_sens * (cam.radius / 30.0)
                cam.target = cam.target + right * world_dx + up * world_dy
                dirty[0] = True
            mouse.last_x, mouse.last_y = x, y
        elif event == cv2.EVENT_MOUSEWHEEL:
            # On Windows OpenCV packs scroll delta into flags >> 16 (signed).
            # +ve = wheel up, -ve = wheel down. We zoom in on +ve.
            notches = int(np.sign(flags >> 16 if flags else 0))
            if notches > 0:
                cam.radius = max(2.0, cam.radius / mouse.zoom_sens)
            elif notches < 0:
                cam.radius = min(500.0, cam.radius * mouse.zoom_sens)
            dirty[0] = True

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    last_render_t = time.perf_counter()
    fps_smoothed = 0.0
    canvas = None
    try:
        while True:
            if dirty[0] or canvas is None:
                t0 = time.perf_counter()
                canvas = render(world, faces, cam, canvas_w, canvas_h,
                                show_outline=show_outline[0])
                dt = time.perf_counter() - t0
                inst_fps = 1.0 / dt if dt > 1e-6 else 999.0
                # EMA so the on-screen fps doesn't jitter.
                fps_smoothed = 0.85 * fps_smoothed + 0.15 * inst_fps
                dirty[0] = False

            display = canvas.copy()
            _draw_help(display, cam, world, fps_smoothed, show_help[0])
            cv2.imshow(window_name, display)

            # Window-close (X button) handling. cv2 doesn't raise on
            # close, and ``waitKey`` keeps returning 0xFFFF, so without
            # this check the process would spin forever after the user
            # closes the window. ``WND_PROP_VISIBLE`` flips to 0 on
            # close; some cv2 builds return -1 if the property is
            # unavailable — accept either as "still visible".
            try:
                vis = cv2.getWindowProperty(window_name,
                                             cv2.WND_PROP_VISIBLE)
            except cv2.error:
                vis = 1.0
            if vis is not None and vis < 1.0:
                break

            key = cv2.waitKey(16) & 0xFFFF
            if key == 0xFFFF:    # no key — keep idling at ~60 Hz
                # But still re-display the canvas so the fps counter
                # updates and the mouse callback can drive re-renders.
                if time.perf_counter() - last_render_t > 0.25:
                    dirty[0] = True
                    last_render_t = time.perf_counter()
                continue
            if key == 27:     # ESC only — Q is bound to pan-Y-down to
                              # match the help overlay.
                break
            elif key in (ord('h'), ord('H')):
                show_help[0] = not show_help[0]
            elif key in (ord('o'), ord('O')):
                show_outline[0] = not show_outline[0]
                dirty[0] = True
            elif key in (ord('r'), ord('R')):
                _fit_camera_to_world(cam, world)
                dirty[0] = True
            elif key in (ord('f'), ord('F')):
                if world.player_pos is not None:
                    cam.target = np.array(world.player_pos, dtype=np.float64)
                    dirty[0] = True
                else:
                    print("[viewer] no player_pos recorded in this JSON; "
                          "F has nothing to focus on.")
            elif key in (ord('w'), ord('W')):
                cam.radius = max(2.0, cam.radius / 1.1)
                dirty[0] = True
            elif key in (ord('s'), ord('S')):
                cam.radius = min(500.0, cam.radius * 1.1)
                dirty[0] = True
            elif key in (ord('a'), ord('A')):
                right, _up, _fwd = cam.view_axes()
                cam.target = cam.target - right * (cam.radius * 0.05)
                dirty[0] = True
            elif key in (ord('d'), ord('D')):
                right, _up, _fwd = cam.view_axes()
                cam.target = cam.target + right * (cam.radius * 0.05)
                dirty[0] = True
            elif key in (ord('e'), ord('E')):
                cam.target = cam.target + np.array([0, cam.radius * 0.05, 0])
                dirty[0] = True
            elif key in (ord('q'), ord('Q')):
                # Q is bound to pan-Y-DOWN, matching the help overlay
                # ``Q/E pan Y``. Earlier draft conflated Q with quit;
                # ESC is the only quit binding now.
                cam.target = cam.target - np.array([0, cam.radius * 0.05, 0])
                dirty[0] = True

            # Idle re-render about every 250 ms so the FPS counter ticks
            # even when the user isn't interacting. Keeps the UI from
            # looking frozen.
            if time.perf_counter() - last_render_t > 0.25:
                dirty[0] = True
                last_render_t = time.perf_counter()
    finally:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive 3D viewer for a saved WorldMap JSON.",
    )
    parser.add_argument("--json", type=str, default=None,
                        help="Path to a world_map_*.json file. If omitted, "
                             "the latest under data/calibration/ is loaded.")
    parser.add_argument("--air", action="store_true",
                        help="Include carved-air voxels in the render. "
                             "Off by default — they bury the geometry.")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    args = parser.parse_args()

    if args.json:
        path = Path(args.json)
    else:
        path = _resolve_default_path()
        if path is None:
            print("[viewer] no world_map_*.json found under "
                  "data/calibration/. Pass --json explicitly.",
                  file=sys.stderr)
            return 2
        print(f"[viewer] auto-selected newest: {path}")

    if not path.is_file():
        print(f"[viewer] not a file: {path}", file=sys.stderr)
        return 2

    try:
        world = load_compact_json(path, include_air=args.air)
    except Exception as e:
        print(f"[viewer] failed to load {path}: {e}", file=sys.stderr)
        return 3

    print(f"[viewer] {world.n} solid voxels  "
          f"palette={len(world.palette)}  dim={world.dimension}")
    run_viewer(world, canvas_w=args.width, canvas_h=args.height)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
