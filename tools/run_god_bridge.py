#!/usr/bin/env python3
"""
Adaptive god-bridge runner — diagonal back-strafe variant.

What real god-bridging actually is
----------------------------------
The player stands centred on a block, faces 45° off-axis (toward a
block CORNER), looks down ~+66° so the crosshair lands on the SIDE
face of the block-under-feet from above-diagonally. They then
back-strafe (hold S+A or S+D — NOT W) so they shuffle away from the
corner WHILE the crosshair stays aimed at the corner of the block
under their feet. Each right-click hits a side face and places a new
block at the same level, diagonally adjacent. The player slides onto
that block as it appears. Repeat → diagonal bridge at floor level.

This is a back-strafe placement pattern, NOT a forward sprint-and-jump
pattern. Earlier "sprint+W+space" variants of this script were just
building a stair-step upward and not god-bridging at all.

What this runner does
---------------------
1. Records frames + input + F3 pose.
2. Reads F3 pose live. Aligns yaw to nearest 45° corner-offset and
   pitch to the target (default +66°). Optionally aligns the player's
   xz position to the centre of the block they're standing on.
3. Holds S + (A or D), fires N right-clicks at a rhythmic cadence,
   optionally taps space every --jump-every-ms.
4. Releases keys, pads, writes ``meta.json``.

Usage
-----
    python tools/run_god_bridge.py
    python tools/run_god_bridge.py --strafe right --places 60
    python tools/run_god_bridge.py --no-align --jump-every-ms 800
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cv2  # noqa: E402
import main as M  # noqa: E402
from collect_demo import (   # noqa: E402
    InputRecorder, FrameWriter, _mc_foreground,
    F3_OCR_INTERVAL_SEC,
)
from control.input_gate import InputGate  # noqa: E402
from utils.focus import _find_minecraft_hwnd, activate_minecraft  # noqa: E402
from vision.capture import Capture, CaptureConfig  # noqa: E402
from vision.ocr import build_f3_reader  # noqa: E402


_PITCH_CHUNK_PX = 100         # don't exceed this in one move — single
                              # large vertical SendInput deltas drop.
_PITCH_CHUNK_SLEEP = 0.025
_PITCH_READ_SETTLE = 0.18
_PITCH_CLAMP_MARGIN = 0.5     # within this many deg of -90 counts as
                              # clamped (F3 OCR is rounded to 0.1°).


def _read_pitch(capture, f3) -> Optional[float]:
    if f3 is None:
        return None
    try:
        info = f3.read(capture.get_frame())
        return None if info.pitch is None else float(info.pitch)
    except Exception:
        return None


def _push_pitch(mouse, dy_px: int) -> None:
    """Push pitch by dy_px in safe chunks (positive = down)."""
    if dy_px == 0:
        return
    sign = 1 if dy_px > 0 else -1
    remaining = abs(dy_px)
    while remaining > 0:
        step = min(_PITCH_CHUNK_PX, remaining)
        mouse.move(0, sign * step)
        remaining -= step
        if remaining > 0:
            time.sleep(_PITCH_CHUNK_SLEEP)


def _push_yaw(mouse, dx_px: int) -> None:
    """Push yaw by dx_px in safe chunks (positive = clockwise = +yaw in MC)."""
    if dx_px == 0:
        return
    sign = 1 if dx_px > 0 else -1
    remaining = abs(dx_px)
    while remaining > 0:
        step = min(_PITCH_CHUNK_PX, remaining)
        mouse.move(sign * step, 0)
        remaining -= step
        if remaining > 0:
            time.sleep(_PITCH_CHUNK_SLEEP)


def calibrate_pitch_rate(capture, f3, mouse,
                         *, test_px: int = 300) -> Optional[float]:
    """Push +test_px DOWN, measure pitch delta, push back. Return px/°.

    Stays well clear of the ±90° clamp by working from a neutral pose;
    caller is responsible for being in [-60, +60] before invoking."""
    _, p0, _ = _retry_pose(capture, f3)
    if p0 is None:
        print("[calibrate_pitch] p0 unreadable")
        return None
    _push_pitch(mouse, +test_px)
    time.sleep(_PITCH_READ_SETTLE)
    _, p1, _ = _retry_pose(capture, f3)
    _push_pitch(mouse, -test_px)
    time.sleep(_PITCH_READ_SETTLE)
    if p1 is None:
        print(f"[calibrate_pitch] p1 unreadable (p0 was {p0:+.2f})")
        return None
    delta = abs(p1 - p0)
    if delta < 2.0:
        print(f"[calibrate_pitch] tiny delta: {p0:+.2f} -> {p1:+.2f} "
              f"after pushing {test_px} px (clamp hit?)")
        return None
    return test_px / delta


def _recenter_pitch_toward_zero(capture, f3, mouse, *,
                                 rate_hint: float = 5.0,
                                 max_attempts: int = 4,
                                 target_window_deg: float = 30.0) -> None:
    """Push pitch toward 0° in conservative bites until |pitch| < window.

    Used before calibration so the calibration push doesn't smash into
    the ±90° clamp. Conservative rate prevents overshoot."""
    for _ in range(max_attempts):
        _, p, _ = _retry_pose(capture, f3)
        if p is None or abs(p) <= target_window_deg:
            return
        # Push at most half the remaining distance per attempt.
        correction_deg = -0.6 * p
        dy_px = int(round(correction_deg * rate_hint))
        _push_pitch(mouse, dy_px)
        time.sleep(_PITCH_READ_SETTLE)


def calibrate_yaw_rate(capture, f3, mouse,
                       *, test_px: int = 300) -> Optional[float]:
    """Push +test_px right, measure yaw delta, push back. Return px/°.

    Prints a specific diagnostic on failure: an unreadable pose is a
    different problem (F3 / biome) from a readable-but-unmoving pose
    (the camera isn't accepting input — MC paused, a menu is open, or
    the window isn't truly focused; SendInput then only moves the free
    cursor). The two have completely different fixes, so distinguish
    them instead of a generic 'calibration failed'."""
    y0, _, _ = _retry_pose(capture, f3)
    if y0 is None:
        print("[calibrate_yaw] pose unreadable before push (F3 off / "
              "biome too bright?).")
        return None
    _push_yaw(mouse, +test_px)
    time.sleep(_PITCH_READ_SETTLE)
    y1, _, _ = _retry_pose(capture, f3)
    _push_yaw(mouse, -test_px)
    time.sleep(_PITCH_READ_SETTLE)
    if y1 is None:
        print("[calibrate_yaw] pose unreadable after push.")
        return None
    delta = abs(_norm_yaw_delta(y1 - y0))
    if delta < 2.0:
        print(f"[calibrate_yaw] camera DID NOT MOVE: yaw {y0:+.1f}° → "
              f"{y1:+.1f}° after a {test_px}px push. The pose reads fine "
              f"but mouse input isn't rotating the view — Minecraft is "
              f"PAUSED, a menu/inventory is open, or the window isn't "
              f"truly focused. Click into the game world (so the cursor "
              f"is grabbed and the crosshair shows) and re-run.")
        return None
    return test_px / delta


def _rederive_rate(rate: float, px_pushed: int, deg_moved: float) -> float:
    """EMA the px/° rate toward what we just OBSERVED (px_pushed /
    deg_moved), but only if the observation is plausible. This makes a
    bad initial calibration self-correct over a few refine steps — the
    single-probe calibrate_*_rate can be off (esp. with OS mouse
    acceleration on), but each precise-set step measures the *actual*
    response and converges the rate to the truth."""
    if abs(deg_moved) < 0.2:
        return rate  # too small to measure reliably (OCR is 0.1°)
    obs = abs(px_pushed) / abs(deg_moved)
    if 1.0 <= obs <= 60.0:
        return 0.5 * rate + 0.5 * obs
    return rate


def set_pitch_to(capture, f3, mouse, target_deg: float,
                 px_per_deg_pitch: float,
                 *, verify_and_refine: bool = True,
                 tolerance_deg: float = 0.2,
                 max_iterations: int = 12) -> Optional[float]:
    """Computed pitch set + iterative refinement until residual ≤
    tolerance_deg or max_iterations reached. F3 reads with 0.1° precision
    so 0.2° is the tightest meaningful target. The px/° rate is
    re-derived from each step's observed motion so a wrong initial
    calibration self-corrects instead of biasing every correction."""
    rate = px_per_deg_pitch if px_per_deg_pitch > 0 else 6.0
    cur = _retry_pitch(capture, f3)
    if cur is None:
        return None
    delta_deg = target_deg - cur
    dy_px = int(round(delta_deg * rate))
    _push_pitch(mouse, dy_px)
    time.sleep(_PITCH_READ_SETTLE)
    new = _retry_pitch(capture, f3)
    if not verify_and_refine or new is None:
        return new
    rate = _rederive_rate(rate, dy_px, new - cur)
    cur = new
    for _ in range(max_iterations):
        residual = target_deg - cur
        if abs(residual) <= tolerance_deg:
            return cur
        dy_px = int(round(residual * rate))
        if dy_px == 0:
            # Sub-pixel residual: nudge by the minimum 1 px in the right
            # direction rather than giving up above tolerance.
            dy_px = 1 if residual > 0 else -1
        _push_pitch(mouse, dy_px)
        time.sleep(_PITCH_READ_SETTLE)
        new = _retry_pitch(capture, f3)
        if new is None:
            return None
        rate = _rederive_rate(rate, dy_px, new - cur)
        cur = new
    return cur


def set_yaw_to(capture, f3, mouse, target_deg: float,
               px_per_deg_yaw: float,
               *, verify_and_refine: bool = True,
               tolerance_deg: float = 0.2,
               max_iterations: int = 12) -> Optional[float]:
    """Computed yaw set + iterative refinement until residual ≤
    tolerance_deg. The px/° rate is re-derived from each step's observed
    motion so a wrong initial calibration self-corrects."""
    rate = px_per_deg_yaw if px_per_deg_yaw > 0 else 6.0
    cur = _retry_yaw(capture, f3)
    if cur is None:
        return None
    delta_deg = _norm_yaw_delta(target_deg - cur)
    dx_px = int(round(delta_deg * rate))
    _push_yaw(mouse, dx_px)
    time.sleep(_PITCH_READ_SETTLE)
    new = _retry_yaw(capture, f3)
    if not verify_and_refine or new is None:
        return new
    rate = _rederive_rate(rate, dx_px, _norm_yaw_delta(new - cur))
    cur = new
    for _ in range(max_iterations):
        residual = _norm_yaw_delta(target_deg - cur)
        if abs(residual) <= tolerance_deg:
            return cur
        dx_px = int(round(residual * rate))
        if dx_px == 0:
            dx_px = 1 if residual > 0 else -1
        _push_yaw(mouse, dx_px)
        time.sleep(_PITCH_READ_SETTLE)
        new = _retry_yaw(capture, f3)
        if new is None:
            return None
        rate = _rederive_rate(rate, dx_px, _norm_yaw_delta(new - cur))
        cur = new
    return cur


def walk_until_stop(capture, f3, keyboard, *,
                    keys: Tuple[str, ...] = ("s",),
                    max_seconds: float = 2.5,
                    poll_interval: float = 0.10,
                    stop_threshold_blocks: float = 0.015,
                    stable_polls: int = 3) -> Optional[Tuple[float, float]]:
    """Press ``keys`` and wait until F3 xyz stops changing (player has
    reached an edge or wall), then release ``keys``. DOES NOT manage
    sneak — caller is responsible for shift state so the player stays
    edge-protected through any subsequent mouse moves.

    Returns the final (x, z), or None on bad pose."""
    for k in keys:
        try: keyboard.press(k)
        except Exception: pass
    try:
        last_xz = None
        stable_count = 0
        deadline = time.perf_counter() + max_seconds
        while time.perf_counter() < deadline:
            time.sleep(poll_interval)
            _, _, xyz = _retry_pose(capture, f3, attempts=2, delay=0.04)
            if xyz is None:
                continue
            xz = (xyz[0], xyz[2])
            if last_xz is not None:
                dx = abs(xz[0] - last_xz[0])
                dz = abs(xz[1] - last_xz[1])
                if dx < stop_threshold_blocks and dz < stop_threshold_blocks:
                    stable_count += 1
                    if stable_count >= stable_polls:
                        return xz
                else:
                    stable_count = 0
            last_xz = xz
        return last_xz
    finally:
        for k in reversed(keys):
            try: keyboard.release(k)
            except Exception: pass


def _wait_until_still(capture, f3, *,
                      max_seconds: float = 1.5,
                      poll_interval: float = 0.08,
                      stop_threshold_blocks: float = 0.01,
                      stable_polls: int = 3,
                      label: str = "settle") -> bool:
    """Poll F3 xyz until the player has stopped moving for ``stable_polls``
    consecutive polls. Presses NO keys — it just waits out residual
    momentum so the next precise step starts from a dead stop. Returns
    True if it confirmed a stop, False on timeout/bad pose.

    This is the user's ``stopping to make sure nothing is moving``
    checkpoint between phases — even a few cm/s of leftover drift while
    we release sneak or start the bridge is enough to slide the player
    off the 1-wide column."""
    last_xz = None
    stable = 0
    deadline = time.perf_counter() + max_seconds
    while time.perf_counter() < deadline:
        time.sleep(poll_interval)
        _, _, xyz = _retry_pose(capture, f3, attempts=2, delay=0.04)
        if xyz is None:
            continue
        xz = (xyz[0], xyz[2])
        if last_xz is not None:
            if (abs(xz[0] - last_xz[0]) < stop_threshold_blocks
                    and abs(xz[1] - last_xz[1]) < stop_threshold_blocks):
                stable += 1
                if stable >= stable_polls:
                    return True
            else:
                stable = 0
        last_xz = xz
    return False


def _read_pose(capture, f3) -> Tuple[Optional[float], Optional[float],
                                     Optional[Tuple[float, float, float]]]:
    """Return (yaw, pitch, xyz) — any field may be None on bad OCR.

    F3 OCR is run concurrently by this (main) thread and the recorder
    background thread. They read DIFFERENT capture frames at slightly
    different instants, so an occasional single-frame flake makes one
    thread return None while the other reads cleanly. A lock attached to
    the reader (``f3._read_lock``, set up in ``main``) serialises the two
    so they at least read the same frame era and never contend on the
    reader's internal state."""
    if f3 is None:
        return (None, None, None)
    lock = getattr(f3, "_read_lock", None)
    try:
        if lock is not None:
            lock.acquire()
        info = f3.read(capture.get_frame())
        return (
            None if info.yaw is None else float(info.yaw),
            None if info.pitch is None else float(info.pitch),
            info.position(),
        )
    except Exception:
        return (None, None, None)
    finally:
        if lock is not None:
            lock.release()


def _norm_yaw_delta(d: float) -> float:
    """Shortest signed angle delta in (-180, 180]."""
    while d > 180.0:
        d -= 360.0
    while d <= -180.0:
        d += 360.0
    return d


def _nearest_corner_yaw(yaw: float) -> float:
    """Snap yaw to nearest 45°-offset (45, 135, -45/315, -135/225).

    MC yaw is in (-180, 180]. Corner yaws are the four diagonals — they
    point at block corners, which is exactly what god-bridge needs."""
    # Targets: ..., -135, -45, 45, 135, 225, 315, ...
    # = 90·k + 45 for integer k.
    k = round((yaw - 45.0) / 90.0)
    target = 90.0 * k + 45.0
    # Normalise into (-180, 180].
    while target > 180.0:
        target -= 360.0
    while target <= -180.0:
        target += 360.0
    return target


def _converge_yaw_to(capture, f3, mouse, target_deg: float,
                     px_per_deg_hint: float,
                     *, max_corrections: int = 4,
                     tolerance_deg: float = 1.5) -> Optional[float]:
    """Push yaw toward target, re-deriving rate from observed motion."""
    rate = px_per_deg_hint if px_per_deg_hint > 0 else 6.0
    yaw, _, _ = _read_pose(capture, f3)
    if yaw is None:
        return None
    for _ in range(max_corrections):
        residual = _norm_yaw_delta(target_deg - yaw)
        if abs(residual) <= tolerance_deg:
            break
        delta_px = int(round(residual * rate))
        max_step = int(round(abs(residual) * rate * 1.05)) + 50
        delta_px = max(-max_step, min(max_step, delta_px))
        _push_yaw(mouse, delta_px)
        time.sleep(_PITCH_READ_SETTLE)
        new_yaw, _, _ = _read_pose(capture, f3)
        if new_yaw is None:
            continue
        moved = _norm_yaw_delta(new_yaw - yaw)
        if abs(moved) >= 1.0:
            obs = abs(delta_px) / abs(moved)
            if 1.5 <= obs <= 30.0:
                rate = 0.5 * rate + 0.5 * obs
        yaw = new_yaw
    return yaw


def _align_to_block_centre(capture, f3, keyboard, *,
                           tolerance_blocks: float = 0.08,
                           max_iterations: int = 8,
                           tap_ms: int = 45) -> Optional[Tuple[float, float]]:
    """Tap WASD to drift the player onto the centre of their current
    block. Reads F3 xz and uses the current yaw to convert world-frame
    offset into local-frame (forward/right) keystrokes.

    Returns the final (offset_x, offset_z) magnitudes, or None on
    unreadable pose. Caller decides what to do on failure."""
    import math
    last_offset = (0.0, 0.0)
    for _ in range(max_iterations):
        yaw, _, xyz = _read_pose(capture, f3)
        if xyz is None or yaw is None:
            return None
        # Block centre: half-block above the integer corner.
        cx = math.floor(xyz[0]) + 0.5
        cz = math.floor(xyz[2]) + 0.5
        off_x = cx - xyz[0]
        off_z = cz - xyz[2]
        last_offset = (off_x, off_z)
        if abs(off_x) < tolerance_blocks and abs(off_z) < tolerance_blocks:
            return last_offset
        # MC yaw 0 = south (+Z). Verified: forward = (-sin, cos).
        # Player facing south, their "right" (D-key world direction) is
        # WEST (-X), not east. So right = (-cos, -sin), NOT (cos, sin).
        yaw_rad = math.radians(yaw)
        fwd_x, fwd_z = -math.sin(yaw_rad),  math.cos(yaw_rad)
        rgt_x, rgt_z = -math.cos(yaw_rad), -math.sin(yaw_rad)
        local_fwd = off_x * fwd_x + off_z * fwd_z   # +ve → press W
        local_rgt = off_x * rgt_x + off_z * rgt_z   # +ve → press D
        # Pick the larger axis to tap this iteration (smaller corrections
        # next time). Keep tap short — ~0.2 blocks at walk speed.
        dur = tap_ms / 1000.0
        if abs(local_fwd) >= abs(local_rgt):
            key = "w" if local_fwd > 0 else "s"
        else:
            key = "d" if local_rgt > 0 else "a"
        try:
            keyboard.tap(key, dur)
        except Exception:
            return last_offset
        time.sleep(0.18)  # let MC apply the tap before re-reading
    return last_offset


def _on_solid_ground(xyz: Tuple[float, float, float]) -> bool:
    """The player is standing on a block iff their y is at an integer
    (within rounding) — feet rest on a block top. Mid-air / mid-jump
    gives a fractional y."""
    if xyz is None:
        return False
    return abs(xyz[1] - round(xyz[1])) < 0.02


def _place_starter_block(capture, f3, mouse,
                         original_pitch: float,
                         px_per_deg_pitch: float) -> Optional[float]:
    """Look straight down (+90°), click once to place a block at feet,
    then return to ``original_pitch``. Returns the final pitch reading
    or None on OCR failure."""
    # Push down to clamp.
    _push_pitch(mouse, int(round(80.0 * max(px_per_deg_pitch, 4.0))))
    time.sleep(_PITCH_READ_SETTLE)
    try:
        mouse.right_click()
    except Exception:
        pass
    time.sleep(0.20)
    # Return to original pitch via convergence (we don't trust the
    # clamp pose, so re-converge).
    return _converge_pitch_to(
        capture, f3, mouse,
        target_deg=original_pitch,
        px_per_deg_hint=px_per_deg_pitch,
    )


def _retry_pose(capture, f3, attempts: int = 18, delay: float = 0.05):
    """F3 OCR can drop out for up to ~1 s at certain views (e.g. looking
    straight down at a low-contrast block over bright sand), then
    recover on its own. Retry persistently across that whole window
    before giving up — a transient dropout must NOT abort a precise-aim
    step when the pose becomes readable again a beat later. ~18×0.05 s
    plus the per-read time spans ~1.5-2 s, which covers the dropouts
    observed in recordings; only a GENUINELY unreadable pose (F3 off,
    window occluded, game paused) exhausts all attempts."""
    for _ in range(attempts):
        y, p, xyz = _read_pose(capture, f3)
        if xyz is not None and y is not None and p is not None:
            return y, p, xyz
        time.sleep(delay)
    return _read_pose(capture, f3)


def _retry_yaw(capture, f3, attempts: int = 18,
               delay: float = 0.05) -> Optional[float]:
    """Retry until YAW reads, ignoring whether xyz/pitch are available.

    The precise-yaw step only needs the yaw angle, and the Facing line's
    angle pair parses even when the XYZ line (position) is OCR-garbled
    over bright terrain. Requiring the full (yaw,pitch,xyz) triple here
    would discard a perfectly good yaw read just because position was
    momentarily unreadable — which is exactly what aborted bridges on
    bright-sand views."""
    for _ in range(attempts):
        y, _, _ = _read_pose(capture, f3)
        if y is not None:
            return y
        time.sleep(delay)
    return _read_pose(capture, f3)[0]


def _retry_pitch(capture, f3, attempts: int = 18,
                 delay: float = 0.05) -> Optional[float]:
    """Retry until PITCH reads (see :func:`_retry_yaw`). The precise-pitch
    step only needs the pitch angle."""
    for _ in range(attempts):
        _, p, _ = _read_pose(capture, f3)
        if p is not None:
            return p
        time.sleep(delay)
    return _read_pose(capture, f3)[1]


def pillar_up(capture, f3, mouse, keyboard,
              count: int = 1,
              *,
              px_per_deg_pitch_hint: float = 9.0,
              jump_to_click_ms: int = 220,
              landing_settle_ms: int = 400,
              skip_pitch_setup: bool = False) -> int:
    """Pillar-up / MLG primitive — for each iteration: look straight
    down, jump, click while airborne to place a block where the player
    just stood. The player lands one block higher. Repeat.

    Returns the number of blocks the player actually ascended.

    Common in Minecraft for: MLG (clutch fall onto a placed block),
    starting a god-bridge from an isolated platform, or just gaining
    height in a clearing without scaffolding.

    Pose-read failure does NOT abort — the jump + click still fire (so
    a flaky F3 OCR doesn't silently turn pillar-up into a no-op). The
    ascent check is best-effort."""
    # Look straight down (+90°). Push a large delta so we hit the +90°
    # clamp from any starting pitch. 220° at the hint rate is enough
    # from -90° all the way past +90°.
    if not skip_pitch_setup:
        big_dy = int(round(220.0 * max(px_per_deg_pitch_hint, 4.0)))
        _push_pitch(mouse, big_dy)
        time.sleep(_PITCH_READ_SETTLE)
    successful = 0
    consecutive_misses = 0
    for i in range(int(count)):
        _, _, xyz_before = _retry_pose(capture, f3)
        y_before = xyz_before[1] if xyz_before is not None else None
        # ALWAYS fire jump + click — F3 unreadable is no reason to skip.
        jumped = False
        try:
            keyboard.tap("space", 0.05)
            jumped = True
        except Exception as e:
            print(f"[pillar_up] iteration {i+1}: jump tap raised {e!r}")
        time.sleep(jump_to_click_ms / 1000.0)
        clicked = False
        try:
            mouse.right_click()
            clicked = True
        except Exception as e:
            print(f"[pillar_up] iteration {i+1}: click raised {e!r}")
        time.sleep(landing_settle_ms / 1000.0)
        _, _, xyz_after = _retry_pose(capture, f3)
        y_after = xyz_after[1] if xyz_after is not None else None
        if y_before is not None and y_after is not None:
            dy = y_after - y_before
            if dy >= 0.5:
                successful += 1
                consecutive_misses = 0
                print(f"[pillar_up] {i+1}/{count}: y {y_before:.2f} → "
                      f"{y_after:.2f}  ✓")
            else:
                # No ascent. NOT a block-count problem (the player may be
                # in creative = infinite blocks); the cause is an
                # off-target / obstructed click — the look-down ray hit a
                # spot where placing didn't put a block under the feet
                # (e.g. standing on a 1-wide bridge remnant rather than a
                # clean column top). Re-assert look-down and try the next
                # iteration; only give up after two misses in a row.
                consecutive_misses += 1
                print(f"[pillar_up] {i+1}/{count}: y {y_before:.2f} → "
                      f"{y_after:.2f}  (no ascent — off-target/obstructed "
                      f"click, not a block-count issue; "
                      f"{'retrying' if consecutive_misses < 2 else 'stopping'})")
                if consecutive_misses >= 2:
                    break
                if not skip_pitch_setup:
                    big_dy = int(round(220.0 * max(px_per_deg_pitch_hint, 4.0)))
                    _push_pitch(mouse, big_dy)   # re-assert straight-down
                    time.sleep(_PITCH_READ_SETTLE)
        else:
            print(f"[pillar_up] {i+1}/{count}: jumped={jumped} "
                  f"clicked={clicked}, pose unreadable "
                  f"(y_before={y_before}, y_after={y_after}); "
                  f"continuing on faith.")
            # Count as successful since the action fired; only break
            # the loop if BOTH actions failed (rare).
            if jumped and clicked:
                successful += 1
    return successful


def _target_corner_offset(yaw_deg: float, strafe: str,
                          edge_inset: float = 0.20) -> Tuple[float, float]:
    """Pick the corner of the current block to stand at before bridging.

    The bridge extends in the direction the keystroke combo drives the
    player in WORLD frame. We want to start at the corner OPPOSITE that
    motion direction — at yaw 45° + S+A motion is +X (east), so we
    want to start near the EAST edge so as motion carries the player
    east they cross the block edge onto the first placed block.

    Returns (offset_x, offset_z) in [-0.4, +0.4] from block centre.
    """
    import math
    y_rad = math.radians(yaw_deg)
    # forward unit = (-sin, cos); left = (cos, sin); right = -left
    fwd_x, fwd_z = -math.sin(y_rad),  math.cos(y_rad)
    left_x, left_z = math.cos(y_rad),  math.sin(y_rad)
    # back = -forward; A = left; D = right
    back_x, back_z = -fwd_x, -fwd_z
    if strafe == "left":
        mx = back_x + left_x
        mz = back_z + left_z
    else:
        mx = back_x - left_x
        mz = back_z - left_z
    mag = math.hypot(mx, mz)
    if mag < 1e-3:
        return (0.0, 0.0)
    return (edge_inset * mx / mag, edge_inset * mz / mag)


def _align_to_block_pos(capture, f3, keyboard, *,
                        target_offset: Tuple[float, float] = (0.0, 0.0),
                        sneak: bool = True,
                        tolerance_blocks: float = 0.08,
                        max_iterations: int = 12,
                        tap_ms: int = 40) -> Optional[Tuple[float, float]]:
    """Walk WASD until the player is at ``block_centre + target_offset``.

    Holds Shift (sneak) by default so the player can sit at the edge
    of the block without falling off. Returns the final (x, z) residual.
    """
    import math
    last_residual = (0.0, 0.0)
    if sneak:
        try: keyboard.press("shift")
        except Exception: pass
    try:
        for _ in range(max_iterations):
            yaw, _, xyz = _read_pose(capture, f3)
            if xyz is None or yaw is None:
                return None
            cx = math.floor(xyz[0]) + 0.5 + target_offset[0]
            cz = math.floor(xyz[2]) + 0.5 + target_offset[1]
            off_x = cx - xyz[0]
            off_z = cz - xyz[2]
            last_residual = (off_x, off_z)
            if abs(off_x) < tolerance_blocks and abs(off_z) < tolerance_blocks:
                return last_residual
            yaw_rad = math.radians(yaw)
            fwd_x, fwd_z = -math.sin(yaw_rad),  math.cos(yaw_rad)
            rgt_x, rgt_z = -math.cos(yaw_rad), -math.sin(yaw_rad)
            local_fwd = off_x * fwd_x + off_z * fwd_z
            local_rgt = off_x * rgt_x + off_z * rgt_z
            dur = tap_ms / 1000.0
            if abs(local_fwd) >= abs(local_rgt):
                key = "w" if local_fwd > 0 else "s"
            else:
                key = "d" if local_rgt > 0 else "a"
            try:
                keyboard.tap(key, dur)
            except Exception:
                return last_residual
            time.sleep(0.18)
    finally:
        if sneak:
            try: keyboard.release("shift")
            except Exception: pass
    return last_residual


def _clamp_up(capture, f3, mouse, *, max_chunks: int = 25) -> Tuple[float, int]:
    """Push UP until pitch stops decreasing (= clamp at -90°).

    Returns (observed_clamp_pitch, total_px_pushed). The clamp value is
    the LAST read pitch, which should be ~-90 ± OCR roundoff."""
    pushed = 0
    last_pitch = _read_pitch(capture, f3)
    stable_at_clamp = 0
    for _ in range(max_chunks):
        _push_pitch(mouse, -_PITCH_CHUNK_PX)
        pushed += _PITCH_CHUNK_PX
        time.sleep(_PITCH_READ_SETTLE)
        p = _read_pitch(capture, f3)
        if p is None:
            continue
        # Stop once we're hard against the clamp (≤ -89.5°) for 2 in a row.
        if p <= -90.0 + _PITCH_CLAMP_MARGIN:
            stable_at_clamp += 1
            last_pitch = p
            if stable_at_clamp >= 2:
                return (p, pushed)
        else:
            stable_at_clamp = 0
            last_pitch = p
    return (last_pitch if last_pitch is not None else -90.0, pushed)


def _converge_pitch_to(capture, f3, mouse, target_deg: float,
                       px_per_deg_hint: float,
                       *, max_corrections: int = 4,
                       tolerance_deg: float = 1.5) -> Optional[float]:
    """Iteratively push toward target_deg, halving the residual each step.

    px_per_deg_hint comes from the up-clamp calibration; we re-derive it
    from the first push so a hint that's only a rough guess still works."""
    rate = px_per_deg_hint if px_per_deg_hint > 0 else 6.0
    cur = _read_pitch(capture, f3)
    if cur is None:
        return None

    for i in range(max_corrections):
        residual = target_deg - cur
        if abs(residual) <= tolerance_deg:
            break
        delta_px = int(round(residual * rate))
        # Cap any single correction so we don't blow past target on a
        # high-sensitivity rig.
        max_step = int(round(abs(residual) * rate * 1.05)) + 50
        delta_px = max(-max_step, min(max_step, delta_px))
        _push_pitch(mouse, delta_px)
        time.sleep(_PITCH_READ_SETTLE)
        new = _read_pitch(capture, f3)
        if new is None:
            continue
        # Re-derive rate from observed motion (px_pushed / deg_moved).
        moved = new - cur
        if abs(moved) >= 1.0:
            rate_obs = abs(delta_px) / abs(moved)
            # EMA toward the observed rate, only if plausible.
            if 1.5 <= rate_obs <= 30.0:
                rate = 0.5 * rate + 0.5 * rate_obs
        cur = new
    return cur


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-pitch", type=float, default=75.8,
                    help="Final pitch in degrees (positive = looking "
                         "down). 75.8° is the user-specified god-bridge "
                         "angle — at yaw 45° this puts the crosshair on "
                         "the diagonal corner of the block-under-feet.")
    ap.add_argument("--places", type=int, default=900,
                    help="Max right-clicks (the bridge stops at this many "
                         "or --max-seconds). Each click is one placement "
                         "ATTEMPT; MC places at most one block per tick, so "
                         "extra clicks aren't wasted — they're STALL "
                         "INSURANCE (see --place-cooldown-ms). 900 ≈ a long "
                         "bridge at max click density; raise for more.")
    ap.add_argument("--place-cooldown-ms", type=int, default=0,
                    help="Sleep between clicks during the bridge. Default 0 "
                         "= click as fast as possible. NOTE: human god-bridge "
                         "guides warn that 30+ CPS 'glitches' placement — but "
                         "that's about butterfly/drag CLICKING MECHANICS on a "
                         "physical mouse; our clean discrete SendInput clicks "
                         "do NOT glitch, and empirically faster is strictly "
                         "better for this bot (17 CPS fell at ~2 blocks; ~50 "
                         "CPS sustained ~34). Keep this at/near 0.")
    ap.add_argument("--click-hold-ms", type=int, default=4,
                    help="How long each right-click is held during the "
                         "bridge. Very short (4 ms) + cooldown 0 + the lifted "
                         "mouse rate-limit → ~100+ CPS, the max placement "
                         "density (every tick covered many times over).")
    ap.add_argument("--move-pulse-ms", type=int, default=0,
                    help="No-sneak bridge ONLY. 0 = hold S+strafe "
                         "continuously (fast, but the player outruns the "
                         "placement and walks off the bridge's leading "
                         "edge after ~10 blocks over a void). >0 = STEPPED "
                         "movement: each cycle places a block then pulses "
                         "S+strafe for this many ms (~90 ≈ a third of a "
                         "block) so the player advances in small steps the "
                         "placement always stays ahead of — a reliable "
                         "no-sneak bridge. Ignored when --keep-sneak.")
    ap.add_argument("--move-gap-ms", type=int, default=40,
                    help="Stepped mode (--move-pulse-ms>0): pause after "
                         "each movement pulse to let the block place and "
                         "the player settle before the next step.")
    ap.add_argument("--strafe", type=str, default="left",
                    choices=("left", "right"),
                    help="Which back-strafe direction to hold. left=S+A "
                         "(bridge curves to player's right of facing); "
                         "right=S+D (curves to the left).")
    ap.add_argument("--keep-sneak", action="store_true",
                    help="Hold Shift through the bridge loop too "
                         "(scaffold-bridge mode — slower but very safe). "
                         "Default releases Shift before the bridge so "
                         "walk speed catches up with placements.")
    ap.add_argument("--drift-check-every", type=int, default=4,
                    help="During the bridge, every N clicks, re-read F3 "
                         "and correct yaw drift back to target. 0 disables. "
                         "Catches the player drifting off-axis due to "
                         "tiny yaw error accumulating each frame.")
    ap.add_argument("--jump-every-ms", type=int, default=-1,
                    help="Tap space (jump) every N ms during the bridge. "
                         "-1 = AUTO (default): for a no-sneak bridge, jump "
                         "every --jump-every-blocks blocks; for --keep-sneak "
                         "never (sneak holds the edge, no jump needed). "
                         "0 = force off. >0 = explicit interval. Jumping is "
                         "THE ninja-bridge technique for outrunning the "
                         "tick-rate placement cap: the jump arc lifts the "
                         "player off the leading edge for a beat so the "
                         "block under the next step places in time.")
    ap.add_argument("--jump-every-blocks", type=float, default=8.0,
                    help="AUTO jump cadence in blocks (converted to a time "
                         "interval via vanilla walk speed ≈4.3 b/s). ~8 is "
                         "the common god-bridger cadence; lower if the "
                         "player still outruns the bridge before jumping.")
    ap.add_argument("--pillar-up", type=int, default=3,
                    help="Pillar-up N blocks before bridging — gets the "
                         "player onto an isolated column so the bridge "
                         "isn't biased by pre-existing terrain. Each "
                         "iteration: jump + look-down + click. Set to 0 "
                         "to skip (assume already isolated).")
    ap.add_argument("--corner-tap-ms", type=int, default=160,
                    help="Duration of the single strafe-key TAP at the "
                         "corner (sneak held) that seats the player in the "
                         "bridge direction before precise alignment.")
    ap.add_argument("--settle-polls", type=int, default=3,
                    help="Consecutive stable F3 polls required for a "
                         "'stopped moving' confirmation at each settle "
                         "checkpoint (corner tap, pre-unsneak, post-unsneak).")
    ap.add_argument("--settle-timeout-sec", type=float, default=1.5,
                    help="Max seconds to wait for motion to stop at a "
                         "settle checkpoint before giving up and continuing.")
    ap.add_argument("--pre-bridge-settle-ms", type=int, default=250,
                    help="Fixed pause after releasing sneak and before the "
                         "bridge starts — lets the crouch-release micro-"
                         "shift die out 'to be sure' (user's recipe).")
    ap.add_argument("--align-tolerance-deg", type=float, default=0.2,
                    help="Max allowed residual (degrees) for the precise "
                         "yaw AND pitch set. 0.2° is essentially exact (≈one "
                         "mouse-pixel step) and is the tightest RELIABLY "
                         "achievable target: the minimum move is ~0.15°/px, "
                         "so a target that doesn't fall on the pixel grid "
                         "can't be hit tighter than that and a stricter gate "
                         "just aborts spuriously. The actual god-bridge "
                         "failure mode was the player outrunning placement, "
                         "not a sub-0.2° aim error. Still aborts on a "
                         "genuinely wrong aim (>0.2°).")
    ap.add_argument("--no-yaw-align", action="store_true",
                    help="Skip the snap-to-nearest-45°-corner yaw step. "
                         "Use this if you've manually aimed.")
    ap.add_argument("--no-pos-align", action="store_true",
                    help="Skip the sneak-to-corner alignment step.")
    ap.add_argument("--place-starter", action="store_true",
                    help="Before the bridge loop, look straight down and "
                         "place one block to seed the bridge (helps if "
                         "the player is mid-air or on a half-block).")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--downscale", type=float, default=0.5)
    ap.add_argument("--countdown", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--pad-after-sec", type=float, default=0.5)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args(argv)

    if not (-89.0 < args.target_pitch < 89.0):
        print(f"[run_god_bridge][ERROR] target pitch must be in (-89,+89); "
              f"got {args.target_pitch}")
        return 2

    wins = _find_minecraft_hwnd()
    if not wins:
        print("[run_god_bridge][ERROR] Minecraft isn't running.")
        return 2
    hwnd = wins[0][0]

    settings = M._load_yaml(M.SETTINGS_PATH)
    keymap_flat = M._flatten_keymap_for_keyboard(M._load_json(M.KEYMAP_PATH))

    capture = Capture(CaptureConfig(
        hwnd=hwnd, threaded=True,
        max_fps=float(args.fps) * 2,
        track_window_each_frame=True,
        name="rec_god_bridge",
    ))
    gate = InputGate()
    safety = M.build_safety(settings, gate=gate)
    keyboard = M.build_keyboard(settings, keymap_flat, gate=gate)
    mouse = M.build_mouse(settings, gate=gate)
    # Lift the mouse event rate-limit so the bridge can click as fast as
    # the OS allows. The default (240 events/s = ~120 CPS cap, but the
    # per-event sleeps also serialise the click loop) throttles bridge
    # placement; our clean SendInput clicks don't glitch like a physical
    # mouse, and empirically more clicks = a longer bridge.
    try:
        mouse.cfg.max_events_per_sec = 2000
    except Exception:
        pass
    try:
        f3_reader = build_f3_reader(settings)
        # build_f3_reader defaults the glyph OCR to the "adaptive"
        # bright-background binariser, which reads F3 text on
        # bright/low-contrast biomes (desert, snow, daytime) where the
        # translucent panel is darker than the surrounding terrain. No
        # per-run threshold tweaking is needed anymore.
        # Shared lock so the main thread (_read_pose) and the recorder
        # thread never run the OCR concurrently on different frames.
        f3_reader._read_lock = threading.Lock()
    except Exception as e:
        print(f"[run_god_bridge][ERROR] F3 reader required for adaptive "
              f"pitch: {e}")
        return 2

    sid = (Path(args.output).name if args.output
           else f"adaptive_god_bridge_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    out_dir = Path(args.output) if args.output else ROOT / "data" / "raw" / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    events_path = out_dir / "events.jsonl"

    print(f"[run_god_bridge] target_pitch={args.target_pitch:+.1f}°  "
          f"places={args.places}  strafe={args.strafe}  "
          f"jump_every_ms={args.jump_every_ms}")
    print(f"[run_god_bridge] output → {out_dir}")

    activate_minecraft()
    safety.start(); keyboard.start(); mouse.start(); capture.start()
    time.sleep(0.25)

    input_rec = InputRecorder(); input_rec.start()
    writer = FrameWriter(frames_dir); writer.start()

    for n in range(max(0, args.countdown), 0, -1):
        print(f"[run_god_bridge] starting in {n}…"); time.sleep(1.0)

    def _f3_on() -> bool:
        try:
            info = f3_reader.read(capture.get_frame())
            # Any parsed field means the debug overlay is up. Don't gate
            # on yaw/pitch alone — on a momentary Facing-line garble that
            # would read False even with the panel clearly visible (XYZ
            # still parsing), and we'd toggle F3 *off* trying to "fix" it.
            return (info.yaw is not None or info.pitch is not None
                    or info.position() is not None)
        except Exception:
            return False

    # Up to 4 F3 toggle attempts with longer settle. The MC F3 panel
    # is a toggle, so worst-case ON→OFF→ON→OFF→ON parity matters; 4
    # attempts cover ON-start-with-OCR-glitch + OFF-start scenarios.
    if safety.allow_input() and not _f3_on():
        for attempt in range(4):
            keyboard.tap("f3", 0.08); time.sleep(0.55)
            if _f3_on():
                print(f"[run_god_bridge] F3 ON after tap {attempt+1}")
                break
        else:
            print("[run_god_bridge][WARN] F3 panel didn't come on after 4 "
                  "taps — adaptive pitch can't run. Toggle F3 manually "
                  "(press F3) so the debug overlay shows, then re-run.")

    if not safety.allow_input():
        print("[run_god_bridge][ERROR] Minecraft isn't focused — gate closed.")
        safety.stop(); keyboard.stop(); mouse.stop(); capture.stop()
        return 2

    # ── Recorder loop ────────────────────────────────────────────────
    stop_evt = threading.Event()
    # When set, the recorder SKIPS its F3 OCR. The glyph scan is Python-
    # level and holds the GIL ~150 ms out of every ~250 ms recorder
    # cycle (measured), which starves the tight bridge click loop and
    # drops placements. Over flat ground a dropped placement just steps
    # the player down to the floor (cosmetic), but over a VOID a single
    # missed placement = a fall. So we pause the recorder's OCR for the
    # duration of the bridge click loop. Frame capture for replay
    # continues; the drift check uses the cached pose from just before.
    bridge_active = threading.Event()
    full_w = full_h = out_w = out_h = 0
    last_yaw = last_pitch = None
    last_xyz: Optional[Tuple[float, float, float]] = None
    last_f3_ts = 0.0
    frame_idx = 0
    start_perf = time.perf_counter()
    start_unix = time.time()
    events_file = open(events_path, "w", encoding="utf-8", buffering=1)

    def record_loop():
        nonlocal full_w, full_h, out_w, out_h
        nonlocal last_yaw, last_pitch, last_xyz, last_f3_ts, frame_idx
        tick_period = 1.0 / max(1.0, args.fps)
        while not stop_evt.is_set():
            t0 = time.perf_counter()
            try:
                frame_full = capture.get_frame()
            except Exception:
                time.sleep(0.05); continue
            if not full_w:
                full_h, full_w = frame_full.shape[:2]
                if args.downscale != 1.0:
                    out_w = max(1, int(round(full_w * args.downscale)))
                    out_h = max(1, int(round(full_h * args.downscale)))
                else:
                    out_w, out_h = full_w, full_h
            # Fully pause recorder OCR during the bridge: the jump is
            # pure-time (constant walk speed) so no live position is
            # needed, and the glyph scan's GIL hold would otherwise stall
            # the tight click loop and drop placements. Clean placement
            # is what sustains the bridge. Full-quality reads otherwise.
            if (not bridge_active.is_set()
                    and (t0 - last_f3_ts) >= F3_OCR_INTERVAL_SEC):
                try:
                    _rl = getattr(f3_reader, "_read_lock", None)
                    if _rl is not None:
                        with _rl:
                            info = f3_reader.read(frame_full)
                    else:
                        info = f3_reader.read(frame_full)
                    if info.yaw is not None: last_yaw = float(info.yaw)
                    if info.pitch is not None: last_pitch = float(info.pitch)
                    if info.position() is not None: last_xyz = info.position()
                except Exception:
                    pass
                last_f3_ts = t0
            if (out_w, out_h) != (full_w, full_h):
                frame = cv2.resize(frame_full, (out_w, out_h),
                                   interpolation=cv2.INTER_AREA)
            else:
                frame = frame_full
            frame_idx += 1
            writer.enqueue(frame_idx, frame)
            snap = input_rec.snapshot()
            elapsed = t0 - start_perf
            event = {
                "frame": frame_idx, "ts": round(elapsed, 4),
                "keys": snap.keys, "buttons": snap.buttons,
                "scroll_dy": snap.scroll_dy,
                "mc_focused": _mc_foreground(hwnd),
                "yaw": round(last_yaw, 2) if last_yaw is not None else None,
                "pitch": round(last_pitch, 2) if last_pitch is not None else None,
                "xyz": ([round(c, 3) for c in last_xyz] if last_xyz else None),
            }
            events_file.write(json.dumps(event) + "\n")
            dt = time.perf_counter() - t0
            if dt < tick_period:
                time.sleep(tick_period - dt)

    rec_thread = threading.Thread(target=record_loop, name="rec-loop", daemon=True)
    rec_thread.start()

    # ── The actual god-bridge work ────────────────────────────────────
    pose_log = {
        "yaw_initial": None,
        "pitch_initial": None,
        "xyz_initial": None,
        "px_per_deg_pitch": None,
        "px_per_deg_yaw": None,
        "yaw_target": None,
        "yaw_final": None,
        "pitch_final": None,
        "target_pitch": args.target_pitch,
        "pillared_up": 0,
        "walk_to_corner_final_xz": None,
        "on_solid_ground_initial": None,
        "starter_block_placed": False,
        "places_fired": 0,
    }
    aborted = False
    try:
        # 1. Read initial pose.
        y0, p0, xyz0 = _retry_pose(capture, f3_reader)
        pose_log["yaw_initial"] = y0
        pose_log["pitch_initial"] = p0
        pose_log["xyz_initial"] = list(xyz0) if xyz0 else None
        if p0 is None or y0 is None:
            print("[run_god_bridge][ERROR] F3 pose unreadable. Likely the "
                  "biome background defeats the F3 panel detector. Move "
                  "to darker terrain or toggle F3 manually and re-run.")
            aborted = True
            raise StopIteration
        print(f"[run_god_bridge] start pose: yaw={y0:+.2f}° pitch={p0:+.2f}° "
              f"xyz={xyz0}")
        pose_log["on_solid_ground_initial"] = _on_solid_ground(xyz0)

        # 2. Calibrate px/° for BOTH yaw and pitch axes.
        # Yaw FIRST (it never clamps so the calibration is always
        # reliable). Then use the yaw rate to safely recenter pitch
        # away from the ±90° clamp before calibrating pitch.
        rate_yaw = calibrate_yaw_rate(capture, f3_reader, mouse,
                                      test_px=300)
        if rate_yaw is None:
            print("[run_god_bridge][ERROR] yaw calibration failed; "
                  "aborting.")
            aborted = True
            raise StopIteration
        # Recenter pitch using yaw rate as a hint (axes are usually the
        # same in MC). Then calibrate pitch from the neutral pose.
        _recenter_pitch_toward_zero(capture, f3_reader, mouse,
                                    rate_hint=rate_yaw,
                                    target_window_deg=30.0)
        rate_pitch = calibrate_pitch_rate(capture, f3_reader, mouse,
                                          test_px=300)
        if rate_pitch is None:
            print(f"[run_god_bridge][WARN] pitch calibration failed; "
                  f"falling back to yaw rate {rate_yaw:.2f} for pitch.")
            rate_pitch = rate_yaw
        pose_log["px_per_deg_pitch"] = round(rate_pitch, 3)
        pose_log["px_per_deg_yaw"]   = round(rate_yaw, 3)
        print(f"[run_god_bridge] calibrated: pitch={rate_pitch:.2f} px/°, "
              f"yaw={rate_yaw:.2f} px/°")

        # 3. Pillar up to isolate the player on a fresh column.
        if args.pillar_up > 0:
            n = pillar_up(capture, f3_reader, mouse, keyboard,
                          count=args.pillar_up,
                          px_per_deg_pitch_hint=rate_pitch)
            pose_log["pillared_up"] = n
            if n < args.pillar_up:
                print(f"[run_god_bridge][WARN] only pillared "
                      f"{n}/{args.pillar_up} — continuing anyway.")

        # 4. Press SHIFT (sneak) and hold it CONTINUOUSLY through every
        # subsequent step. Releasing sneak even briefly while the
        # player is at a block edge lets gravity tip them off before
        # the precise yaw/pitch finishes. Sneak stays held until the
        # very end of the bridge loop.
        sneak_held = not args.no_pos_align
        if sneak_held:
            try: keyboard.press("shift")
            except Exception: pass

        # 5. Pre-align: rough yaw + moderate pitch, then S until stopped.
        if not args.no_pos_align:
            cur_yaw_now, _, _ = _retry_pose(capture, f3_reader)
            rough_target_yaw = _nearest_corner_yaw(
                cur_yaw_now if cur_yaw_now is not None else y0)
            set_yaw_to(capture, f3_reader, mouse,
                       target_deg=rough_target_yaw,
                       px_per_deg_yaw=rate_yaw,
                       verify_and_refine=False)
            set_pitch_to(capture, f3_reader, mouse,
                         target_deg=45.0,
                         px_per_deg_pitch=rate_pitch,
                         verify_and_refine=False)
            print(f"[run_god_bridge] walking back (S, sneak held) "
                  f"until stopped against the corner…")
            final_xz = walk_until_stop(
                capture, f3_reader, keyboard,
                keys=("s",),
                max_seconds=2.5, stop_threshold_blocks=0.015)
            pose_log["walk_to_corner_final_xz"] = (
                list(final_xz) if final_xz is not None else None)
            if final_xz is not None:
                print(f"[run_god_bridge] stopped at xz=({final_xz[0]:.3f}, "
                      f"{final_xz[1]:.3f})")

            # 5b. FIRST wait out the momentum from the S walk-back —
            # releasing S doesn't stop the player instantly, they coast
            # for a few ticks. Tapping the strafe key while still
            # coasting on the S axis goes nowhere useful, so settle to a
            # dead stop FIRST. THEN single TAP of the strafe key (sneak
            # STILL held) to seat the player at the corner in the bridge
            # direction, and settle again. (User's recipe: wait until
            # not moving, then tap left/right, then wait again.)
            pre_tap_stop = _wait_until_still(
                capture, f3_reader,
                max_seconds=args.settle_timeout_sec,
                stable_polls=args.settle_polls)
            strafe_key0 = "a" if args.strafe == "left" else "d"
            print(f"[run_god_bridge] S-momentum settled "
                  f"(confirmed_stop={pre_tap_stop}); corner tap: "
                  f"{strafe_key0.upper()} for {args.corner_tap_ms}ms "
                  f"(sneak held), then settling…")
            try:
                keyboard.tap(strafe_key0, args.corner_tap_ms / 1000.0)
            except Exception:
                pass
            stilled = _wait_until_still(
                capture, f3_reader,
                max_seconds=args.settle_timeout_sec,
                stable_polls=args.settle_polls)
            print(f"[run_god_bridge] settled after corner tap "
                  f"(confirmed_stop={stilled})")

        # 6. PRECISE yaw + pitch (sneak still held — player stays on
        # the block while the mouse moves take effect). Both axes MUST
        # land within ``align_tolerance_deg`` of target — a god-bridge
        # depends on the crosshair hitting the exact block corner, and
        # an off-by-1° aim places against the wrong face (or nothing).
        # If either axis can't be made perfect, we ABORT rather than
        # bridge with a wrong aim and silently fail / fall.
        tol = float(args.align_tolerance_deg)
        if not args.no_yaw_align:
            cur_yaw_now, _, _ = _retry_pose(capture, f3_reader)
            target_yaw = _nearest_corner_yaw(
                cur_yaw_now if cur_yaw_now is not None else y0)
            pose_log["yaw_target"] = target_yaw
            y_final = set_yaw_to(
                capture, f3_reader, mouse,
                target_deg=target_yaw,
                px_per_deg_yaw=rate_yaw,
                verify_and_refine=True, tolerance_deg=tol,
            )
            pose_log["yaw_final"] = y_final
            yaw_res = (abs(_norm_yaw_delta(target_yaw - y_final))
                       if y_final is not None else None)
            print(f"[run_god_bridge] precise yaw: "
                  f"{(y_final if y_final is not None else float('nan')):+.2f}° "
                  f"(target {target_yaw:+.1f}°, residual "
                  f"{('%.2f°' % yaw_res) if yaw_res is not None else 'N/A'})")
            if yaw_res is None or yaw_res > tol:
                print(f"[run_god_bridge][ERROR] yaw did not converge to "
                      f"within {tol:.2f}° of {target_yaw:+.1f}° "
                      f"(residual {('%.2f°' % yaw_res) if yaw_res is not None else 'unreadable'}). "
                      f"Aborting before the bridge — refusing to bridge "
                      f"with a wrong aim.")
                aborted = True
                raise StopIteration
        p_final = set_pitch_to(
            capture, f3_reader, mouse,
            target_deg=args.target_pitch,
            px_per_deg_pitch=rate_pitch,
            verify_and_refine=True, tolerance_deg=tol,
        )
        pose_log["pitch_final"] = p_final
        pitch_res = (abs(args.target_pitch - p_final)
                     if p_final is not None else None)
        print(f"[run_god_bridge] precise pitch: "
              f"{(p_final if p_final is not None else float('nan')):+.2f}° "
              f"(target {args.target_pitch:+.1f}°, residual "
              f"{('%.2f°' % pitch_res) if pitch_res is not None else 'N/A'})")
        if pitch_res is None or pitch_res > tol:
            print(f"[run_god_bridge][ERROR] pitch did not converge to "
                  f"within {tol:.2f}° of {args.target_pitch:+.1f}° "
                  f"(residual {('%.2f°' % pitch_res) if pitch_res is not None else 'unreadable'}). "
                  f"Aborting before the bridge — refusing to bridge with "
                  f"a wrong aim.")
            aborted = True
            raise StopIteration
        time.sleep(0.20)

        # 6b. AIM SNAPSHOT — full-res frame of the aimed state (sneak
        # still held, before any bridge motion) plus a zoom on the
        # crosshair. This is the ground truth for "is the aim actually
        # on the block corner?" — the F3 numbers can read on-target
        # while the crosshair still misses the placeable face, which is
        # exactly the failure mode that drops the player off the pillar.
        try:
            import cv2 as _cv2
            aim_full = capture.get_frame()
            _cv2.imwrite(str(out_dir / "aim_snapshot.png"),
                         _cv2.cvtColor(aim_full, _cv2.COLOR_RGB2BGR))
            fh_, fw_ = aim_full.shape[:2]
            cy, cx = fh_ // 2, fw_ // 2
            half = 130
            zoom = aim_full[max(0, cy - half):cy + half,
                            max(0, cx - half):cx + half]
            zoom = _cv2.resize(zoom, (zoom.shape[1] * 3, zoom.shape[0] * 3),
                               interpolation=_cv2.INTER_NEAREST)
            _cv2.imwrite(str(out_dir / "aim_crosshair_zoom.png"),
                         _cv2.cvtColor(zoom, _cv2.COLOR_RGB2BGR))
            print(f"[run_god_bridge] aim snapshot → {out_dir/'aim_snapshot.png'}")
        except Exception as _e:
            print(f"[run_god_bridge][WARN] aim snapshot failed: {_e!r}")

        # 6. Optionally seed a starter block (mostly for mid-air starts).
        if args.place_starter or (xyz0 is not None and not _on_solid_ground(xyz0)):
            print("[run_god_bridge] placing starter block under feet…")
            new_pitch = _place_starter_block(
                capture, f3_reader, mouse,
                original_pitch=args.target_pitch,
                px_per_deg_pitch=rate_pitch,
            )
            pose_log["starter_block_placed"] = True
            if new_pitch is not None:
                pose_log["pitch_final"] = new_pitch
            time.sleep(0.15)

        # 7. Settle (confirm nothing is moving) → release sneak → settle
        # again for a fraction of a second → THEN bridge. This is the
        # user's recipe: after the precise aim, make sure the player is
        # dead-still before un-sneaking, then pause briefly after
        # un-sneaking so any sneak-release micro-shift dies out before
        # the bridge motion begins. A real god-bridge does NOT hold
        # shift during the bridge (that's a scaffold/shift-bridge);
        # --keep-sneak keeps it held for the safe/slow variant.
        back_key = "s"
        strafe_key = "a" if args.strafe == "left" else "d"
        # Pre-unsneak settle — confirm the precise aim left us motionless.
        _wait_until_still(capture, f3_reader,
                          max_seconds=args.settle_timeout_sec,
                          stable_polls=args.settle_polls)
        if sneak_held and not args.keep_sneak:
            try: keyboard.release("shift")
            except Exception: pass
            sneak_held = False
            # Post-unsneak settle — a fixed fraction of a second "to be
            # sure" (the player can micro-shift as the crouch offset
            # releases), then re-confirm stillness before committing.
            time.sleep(max(0.0, args.pre_bridge_settle_ms / 1000.0))
            _wait_until_still(capture, f3_reader,
                              max_seconds=args.settle_timeout_sec,
                              stable_polls=args.settle_polls)
        # Press S and the strafe key on the SAME tick. Pressing them via
        # two normal press() calls puts ~8 ms (the keyboard rate-limit)
        # between them, so one axis engages a tick before the other and
        # the player moves straight (building a wrong / double-wide path)
        # before the diagonal starts. press_together emits both key-downs
        # back-to-back with no gap.
        # Stepped mode pulses the keys each cycle, so don't latch them on
        # here. Continuous mode holds S+strafe for the whole loop.
        stepped = (args.move_pulse_ms > 0) and not args.keep_sneak
        if not stepped:
            keyboard.press_together(back_key, strafe_key)
        bridge_active.set()   # free the click loop from recorder-OCR GIL stalls

        # 8. Click in rhythm; periodic yaw drift correction; optional jump.
        deadline = time.perf_counter() + float(args.max_seconds)
        cooldown = max(0.0, args.place_cooldown_ms / 1000.0)
        move_pulse = max(0.0, args.move_pulse_ms / 1000.0)
        move_gap = max(0.0, args.move_gap_ms / 1000.0)
        # Jump enable. AUTO (-1): jump on a no-sneak bridge, never with
        # sneak. 0 forces off; >0 forces on (explicit).
        #
        # TIMING (the precise lever per the god-bridge community): the
        # mouse is held DEAD STILL after aligning (no jitter), so the
        # player walks at a CONSTANT vanilla speed (~4.317 b/s) — which
        # means "8 blocks" is a precise TIME, with no need for position
        # reads (those stall the click loop and OCR-glitch). The pitch
        # geometry lets you place ~8 blocks, then you MUST jump RIGHT
        # BEFORE the 8th — a jump that lands AT/after block 8 already
        # missed it and you fall. So we fire at (jump_every_blocks - lead)
        # blocks of travel, re-anchored at each jump, where lead puts the
        # jump just before the 8th.
        _WALK_BPS = 4.317                       # vanilla walking speed
        if args.jump_every_ms < 0:
            jump_enabled = not args.keep_sneak
        elif args.jump_every_ms == 0:
            jump_enabled = False
        else:
            jump_enabled = True
        jump_lead_blocks = 1.0                  # fire ~1 block before the Nth
        if args.jump_every_ms > 0:
            jump_period = args.jump_every_ms / 1000.0   # explicit override
        else:
            jump_period = max(0.3, (args.jump_every_blocks - jump_lead_blocks)
                              / _WALK_BPS)
        if jump_enabled:
            print(f"[run_god_bridge] jump: every {jump_period:.3f}s "
                  f"(≈{args.jump_every_blocks - jump_lead_blocks:.1f} blocks at "
                  f"{_WALK_BPS} b/s) — fires RIGHT BEFORE block "
                  f"{args.jump_every_blocks:.0f}, mouse held still")
        drift_check_n = max(0, int(args.drift_check_every))
        # Target yaw / pitch we converged to in step 6.
        bridge_target_yaw = pose_log.get("yaw_target") or 0.0
        bridge_target_pitch = args.target_pitch
        last_jump_ts = time.perf_counter()
        jumps_fired = 0
        # Track sideways drift via xz: at the chosen yaw + strafe,
        # motion should be along one cardinal axis. Anything off-axis
        # is yaw error compounding.
        fired = 0
        yaw_corrections = 0
        max_off_axis_blocks = 0.0
        first_bridge_xyz = None
        for click_i in range(int(args.places)):
            if time.perf_counter() > deadline:
                print(f"[run_god_bridge][WARN] hit max-seconds; "
                      f"fired {fired}/{args.places}")
                break
            if not safety.allow_input():
                print(f"[run_god_bridge][WARN] gate closed "
                      f"(focus lost?); fired {fired}/{args.places}")
                break
            if jump_enabled and (
                    time.perf_counter() - last_jump_ts >= jump_period):
                try: keyboard.tap("space", 0.05)
                except Exception: pass
                last_jump_ts = time.perf_counter()
                jumps_fired += 1
            mouse.right_click(duration=max(0.001, args.click_hold_ms / 1000.0))
            fired += 1
            if stepped:
                # Place-then-step: the block just placed is behind/under
                # the next position; pulse S+strafe a fraction of a block
                # onto it, then release and let it settle. The player
                # never gets ahead of the placement, so they can't walk
                # off the leading edge over a void.
                keyboard.press_together(back_key, strafe_key)
                time.sleep(move_pulse)
                keyboard.release(strafe_key)
                keyboard.release(back_key)
                if move_gap > 0:
                    time.sleep(move_gap)
            elif cooldown > 0:
                time.sleep(cooldown)
            # Periodic yaw drift check — and report off-axis xz drift.
            # CRITICAL: read the recorder thread's CACHED pose (a free
            # variable load), NOT a fresh _retry_pose. A blocking OCR
            # here stalled the click loop ~100 ms every few clicks —
            # that gap in placement is exactly what let the player walk
            # off the edge after a handful of blocks. The recorder re-
            # reads at ~8 Hz in the background, so this is ≤~120 ms
            # stale, which is plenty fresh for a periodic drift nudge.
            if drift_check_n > 0 and (click_i + 1) % drift_check_n == 0:
                cur_yaw = last_yaw
                cur_xyz = last_xyz
                if cur_xyz is not None and first_bridge_xyz is None:
                    first_bridge_xyz = cur_xyz
                if (cur_yaw is not None
                        and abs(_norm_yaw_delta(cur_yaw - bridge_target_yaw))
                            > 0.7):
                    residual = _norm_yaw_delta(bridge_target_yaw - cur_yaw)
                    dx_px = int(round(residual * rate_yaw))
                    if dx_px != 0:
                        _push_yaw(mouse, dx_px)
                        yaw_corrections += 1
                # Off-axis drift: bridge should advance along the world
                # axis dictated by yaw+strafe. Compute perpendicular
                # offset from the initial bridge xyz.
                if cur_xyz is not None and first_bridge_xyz is not None:
                    # Expected motion unit vector (matches target_corner).
                    y_rad = math.radians(bridge_target_yaw)
                    fwd_x, fwd_z = -math.sin(y_rad), math.cos(y_rad)
                    left_x, left_z = math.cos(y_rad), math.sin(y_rad)
                    if args.strafe == "left":
                        mx = -fwd_x + left_x; mz = -fwd_z + left_z
                    else:
                        mx = -fwd_x - left_x; mz = -fwd_z - left_z
                    mag = math.hypot(mx, mz)
                    if mag > 1e-3:
                        mx /= mag; mz /= mag
                    # Perpendicular = (-mz, mx).
                    px_, pz_ = -mz, mx
                    dx = cur_xyz[0] - first_bridge_xyz[0]
                    dz = cur_xyz[2] - first_bridge_xyz[2]
                    off_axis = abs(dx * px_ + dz * pz_)
                    max_off_axis_blocks = max(max_off_axis_blocks, off_axis)
        bridge_active.clear()   # resume recorder pose logging
        pose_log["places_fired"] = fired
        pose_log["jumps_fired"] = jumps_fired
        pose_log["yaw_corrections"] = yaw_corrections
        pose_log["max_off_axis_drift_blocks"] = round(max_off_axis_blocks, 3)

        # 9. Release strafe — then, in sneak-held mode, WAIT until the
        # player has fully coasted to a stop BEFORE releasing shift.
        # Releasing shift while still sliding (residual momentum from the
        # bridge) lets the player slide off the last block and fall — the
        # bridge holds the whole way then drops at the very end. Sneak
        # still protects the edge during this settle.
        keyboard.release(back_key)
        keyboard.release(strafe_key)
        if sneak_held:
            stopped = _wait_until_still(
                capture, f3_reader,
                max_seconds=args.settle_timeout_sec,
                stable_polls=args.settle_polls)
            print(f"[run_god_bridge] post-bridge settle before un-sneak "
                  f"(confirmed_stop={stopped})")
            keyboard.release("shift")
            sneak_held = False
    except StopIteration:
        pass  # graceful abort path used by the F3 unreadable case
    finally:
        bridge_active.clear()   # ensure recorder OCR resumes on any exit
        try:
            for k in ("w", "a", "s", "d", "control", "ctrl", "shift", "space"):
                try: keyboard.release(k)
                except Exception: pass
        except Exception:
            pass
        time.sleep(max(0.0, args.pad_after_sec))
        stop_evt.set()
        rec_thread.join(timeout=2.0)
        try:
            events_file.flush(); events_file.close()
        except Exception: pass
        writer.stop(); writer.join(timeout=5.0)
        input_rec.stop()
        for sub in (keyboard, mouse):
            try: sub.stop()
            except Exception: pass
        safety.stop(); capture.stop()

    duration = time.perf_counter() - start_perf
    meta = {
        "session_id": sid,
        "tool": "run_god_bridge",
        "start_ts_unix": start_unix,
        "duration_sec": round(duration, 3),
        "fps_target": args.fps,
        "fps_realised": round(frame_idx / max(duration, 1e-6), 2),
        "capture_resolution": [full_w, full_h] if full_w else None,
        "frame_resolution": [out_w, out_h] if out_w else None,
        "downscale": args.downscale,
        "frame_count": frame_idx,
        "frames_written": writer.written,
        "frames_dropped": writer.dropped,
        "mc_hwnd": hwnd,
        "aborted": aborted,
        "args": vars(args),
        "pose_log": pose_log,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2),
                                       encoding="utf-8")
    print(f"\n[run_god_bridge] done: {frame_idx} frames over {duration:.1f}s")
    print(f"[run_god_bridge] replay with:\n"
          f"    python tools/replay_demo.py {out_dir}")
    return 0 if not aborted else 1


if __name__ == "__main__":
    raise SystemExit(main())
