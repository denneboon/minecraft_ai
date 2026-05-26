# vision/world/screen_ray.py
"""
Camera intrinsics and screen ↔ world geometry.

Given the player's pose (XYZ + yaw + pitch) and the rendered frame's
size + horizontal FOV, this module converts:

  * screen pixel  → world ray         (``unproject``)
  * world point   → screen pixel      (``project``)
  * world ray     → stream of voxels  (``voxel_walk``)

Minecraft camera conventions (verified against the F3 readout)
--------------------------------------------------------------
* World axes: +X = east, +Y = up, +Z = south.
* Yaw 0 = looking +Z (south). Yaw increases clockwise from above.
* Pitch 0 = level, -90 = up, +90 = down.
* The camera origin is the player's eye, which is 1.62 blocks above
  feet for the default standing pose.
* The horizontal FOV is the diagonal-ish value in MC's settings.
  Modern MC interprets ``options.fov`` as the *vertical* FOV when the
  aspect ratio is wider than the reference 4:3 — but for our 16:9
  capture and the user's setting of 90°, treating it as the
  *horizontal* FOV is what reproduces the visible cone, so we keep
  that convention here.

Why a separate module
---------------------
The agent loop will eventually project candidate path-points onto the
screen to know where to look ("aim the crosshair at *that* block over
there"), and the perception layer projects screen patches into the
world to know what's where. Keeping the math here means both
directions stay consistent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------

@dataclass
class CameraIntrinsics:
    """
    Camera intrinsics derived from frame size + horizontal FOV.

    The focal-length-equivalent values ``fx`` and ``fy`` are in pixel
    units (so a pinhole projection ``u = fx * X / Z + cx`` works
    without further scaling).
    """

    width:  int
    height: int
    h_fov_deg: float

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def fx(self) -> float:
        return self.cx / math.tan(math.radians(self.h_fov_deg) / 2.0)

    @property
    def fy(self) -> float:
        # Square pixels — fy equals fx in pixel space; the vertical FOV
        # is derived from the aspect ratio, not stored.
        return self.fx

    @property
    def v_fov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.cy / self.fy))

    @classmethod
    def from_frame(cls, width: int, height: int,
                   h_fov_deg: float = 90.0) -> "CameraIntrinsics":
        return cls(width=int(width), height=int(height),
                   h_fov_deg=float(h_fov_deg))


# ---------------------------------------------------------------------------
# Screen ↔ world ray geometry
# ---------------------------------------------------------------------------

@dataclass
class ScreenRay:
    """
    Wraps a :class:`CameraIntrinsics` and provides projection helpers
    parameterised by the player's pose at a single point in time.
    """

    intrinsics: CameraIntrinsics

    # ── Direction vector from yaw + pitch ──────────────────────────

    @staticmethod
    def forward_vector(yaw_deg: float, pitch_deg: float) -> Tuple[float, float, float]:
        """
        Unit forward vector in world coords for the given yaw + pitch.

        Derived from MC's camera convention:
            forward_x = -sin(yaw) * cos(pitch)
            forward_y = -sin(pitch)
            forward_z =  cos(yaw) * cos(pitch)
        """
        ry = math.radians(yaw_deg)
        rp = math.radians(pitch_deg)
        cp = math.cos(rp)
        return (-math.sin(ry) * cp,
                -math.sin(rp),
                 math.cos(ry) * cp)

    @staticmethod
    def right_vector(yaw_deg: float) -> Tuple[float, float, float]:
        """Unit right vector (perpendicular to forward, in the XZ plane)."""
        ry = math.radians(yaw_deg)
        # When yaw = 0 (facing +Z), right = +X (east). Forward = +Z, so
        # right = forward × world_up = (+Z) × (+Y) = (+X). The formula
        # below reproduces that for any yaw.
        return (math.cos(ry), 0.0, math.sin(ry))

    @staticmethod
    def up_vector(yaw_deg: float, pitch_deg: float) -> Tuple[float, float, float]:
        """Unit up vector for the current camera frame (perpendicular
        to both right and forward)."""
        fx, fy, fz = ScreenRay.forward_vector(yaw_deg, pitch_deg)
        rx, ry_, rz = ScreenRay.right_vector(yaw_deg)
        # up = right × forward (right-handed convention)
        ux = ry_ * fz - rz * fy
        uy = rz * fx - rx * fz
        uz = rx * fy - ry_ * fx
        n = math.sqrt(ux * ux + uy * uy + uz * uz)
        if n < 1e-9:
            return (0.0, 1.0, 0.0)
        return (ux / n, uy / n, uz / n)

    # ── Unproject: screen pixel → world ray ────────────────────────

    def unproject(self,
                  px: float,
                  py: float,
                  *,
                  yaw_deg: float,
                  pitch_deg: float,
                  eye_xyz: Tuple[float, float, float],
                  ) -> Tuple[Tuple[float, float, float],
                             Tuple[float, float, float]]:
        """
        Return (origin, direction) of the world-space ray that the
        pixel ``(px, py)`` lies on.

        ``direction`` is unit-length.
        """
        # Camera-space ray: (X, Y, -1) where X, Y are the pixel offset
        # from principal point divided by focal length. The minus on Z
        # follows OpenCV: positive Z is INTO the screen.
        intr = self.intrinsics
        x_cam = (px - intr.cx) / intr.fx
        y_cam = -(py - intr.cy) / intr.fy        # flip — screen Y is down

        # Compose right / up / forward into a rotation that takes
        # camera-space directions to world directions.
        f = self.forward_vector(yaw_deg, pitch_deg)
        r = self.right_vector(yaw_deg)
        u = self.up_vector(yaw_deg, pitch_deg)

        dx = x_cam * r[0] + y_cam * u[0] + f[0]
        dy = x_cam * r[1] + y_cam * u[1] + f[1]
        dz = x_cam * r[2] + y_cam * u[2] + f[2]
        n = math.sqrt(dx * dx + dy * dy + dz * dz)
        if n < 1e-9:
            return eye_xyz, f
        return eye_xyz, (dx / n, dy / n, dz / n)

    # ── Project: world point → screen pixel ────────────────────────

    def project(self,
                world_xyz: Tuple[float, float, float],
                *,
                yaw_deg: float,
                pitch_deg: float,
                eye_xyz: Tuple[float, float, float],
                ) -> Optional[Tuple[float, float, float]]:
        """
        Project a world point onto the screen. Returns ``(px, py,
        depth)`` or ``None`` if the point is behind the camera.

        ``depth`` is the distance along the camera's forward axis
        (positive = in front of the camera).
        """
        dx = world_xyz[0] - eye_xyz[0]
        dy = world_xyz[1] - eye_xyz[1]
        dz = world_xyz[2] - eye_xyz[2]

        f = self.forward_vector(yaw_deg, pitch_deg)
        r = self.right_vector(yaw_deg)
        u = self.up_vector(yaw_deg, pitch_deg)

        # Transform world delta into camera coords.
        x_cam = dx * r[0] + dy * r[1] + dz * r[2]
        y_cam = dx * u[0] + dy * u[1] + dz * u[2]
        z_cam = dx * f[0] + dy * f[1] + dz * f[2]
        if z_cam <= 1e-3:
            return None

        intr = self.intrinsics
        px = intr.fx * (x_cam / z_cam) + intr.cx
        py = -intr.fy * (y_cam / z_cam) + intr.cy
        return px, py, z_cam

    # ── Voxel-stepping along a world ray (3-D DDA) ─────────────────

    @staticmethod
    def voxel_walk(origin: Tuple[float, float, float],
                   direction: Tuple[float, float, float],
                   max_distance: float = 8.0,
                   ) -> Iterable[Tuple[int, int, int]]:
        """
        Yield the sequence of integer block coords a ray would enter,
        starting from ``origin`` and travelling along (unit) ``direction``
        for at most ``max_distance`` world units.

        Implements Amanatides & Woo's classic 3-D DDA — the same
        algorithm Mojang uses for raytracing the targeted block (with
        a default reach of ~5 blocks in survival).
        """
        ox, oy, oz = origin
        dx, dy, dz = direction
        if max_distance <= 0.0 or (dx == 0.0 and dy == 0.0 and dz == 0.0):
            return

        # Starting voxel.
        ix, iy, iz = math.floor(ox), math.floor(oy), math.floor(oz)
        step_x = 1 if dx > 0 else -1 if dx < 0 else 0
        step_y = 1 if dy > 0 else -1 if dy < 0 else 0
        step_z = 1 if dz > 0 else -1 if dz < 0 else 0

        inf = float("inf")
        # Distance to next voxel boundary along each axis.
        def _next_t(o: float, d: float, i: int, step: int) -> float:
            if step == 0:
                return inf
            boundary = (i + 1) if step > 0 else i
            return (boundary - o) / d
        t_max_x = _next_t(ox, dx, ix, step_x)
        t_max_y = _next_t(oy, dy, iy, step_y)
        t_max_z = _next_t(oz, dz, iz, step_z)
        # Distance between successive voxel boundaries along each axis.
        t_delta_x = inf if step_x == 0 else 1.0 / abs(dx)
        t_delta_y = inf if step_y == 0 else 1.0 / abs(dy)
        t_delta_z = inf if step_z == 0 else 1.0 / abs(dz)

        yield (ix, iy, iz)
        travelled = 0.0
        while travelled < max_distance:
            if t_max_x < t_max_y:
                if t_max_x < t_max_z:
                    ix += step_x
                    travelled = t_max_x
                    t_max_x += t_delta_x
                else:
                    iz += step_z
                    travelled = t_max_z
                    t_max_z += t_delta_z
            else:
                if t_max_y < t_max_z:
                    iy += step_y
                    travelled = t_max_y
                    t_max_y += t_delta_y
                else:
                    iz += step_z
                    travelled = t_max_z
                    t_max_z += t_delta_z
            if travelled > max_distance:
                break
            yield (ix, iy, iz)


__all__ = ["CameraIntrinsics", "ScreenRay"]
