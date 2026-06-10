#!/usr/bin/env python3
"""
Measure mouse px-per-degree against F3 ground truth — using the SAME
raw ``mouse.move(dx, dy)`` calls the script runner uses (the macros'
``move``/``MouseMove`` ops). This is what god-bridge yaw oscillation
relies on, so if a macro looks too wide / too narrow this tells you the
real conversion to use.

What it does
------------
1. Read F3 yaw + pitch (BEFORE).
2. For each test delta:
   * push ``mouse.move(+dx, 0)`` (or ``(0, +dy)``)
   * settle, read pose
   * push the inverse to return — settle, read pose
   * compute ``px/deg = |dx| / |delta_yaw|`` (wrap-safe)
3. Average across deltas; warn if the runs disagree a lot (a sign the
   in-game sensitivity slider isn't where you think it is).
4. Compare against the persisted calibrator
   (``data/calibration/mouse_calibration.json``).

Run
---
    python tools/test_px_per_deg.py
    python tools/test_px_per_deg.py --yaw-px 100,200,300,400 --pitch-px 80,160
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.focus import _find_minecraft_hwnd, activate_minecraft  # noqa: E402
from vision.capture import Capture, CaptureConfig                 # noqa: E402
from vision.ocr import build_f3_reader                            # noqa: E402
from control.mouse import Mouse, MouseConfig                      # noqa: E402
from control.input_gate import InputGate                          # noqa: E402
import main as M                                                  # noqa: E402


def _norm_yaw(d: float) -> float:
    while d > 180.0:  d -= 360.0
    while d < -180.0: d += 360.0
    return d


def _read_pose(capture, f3) -> Tuple[Optional[float], Optional[float]]:
    try:
        info = f3.read(capture.get_frame())
        return (info.yaw, info.pitch)
    except Exception:
        return (None, None)


_CHUNK_PX = 100  # ≥ ~100 px in one SendInput vertical delta is unreliable
                  # (silently dropped in Windows raw input on this rig);
                  # so push every test as chunks of this size, same as the
                  # macros do.


def _push(mouse: Mouse, axis: str, px: int, chunk_sleep: float = 0.02) -> None:
    """Chunked push so single-large-delta drops don't bias the measurement."""
    remaining = px
    step_sign = 1 if px >= 0 else -1
    while remaining != 0:
        step = step_sign * min(_CHUNK_PX, abs(remaining))
        if axis == "yaw":
            mouse.move(step, 0)
        else:
            mouse.move(0, step)
        remaining -= step
        if remaining != 0:
            time.sleep(chunk_sleep)


def _measure_one(mouse: Mouse, capture, f3, *,
                 axis: str, px: int, settle: float) -> Optional[float]:
    """Push ``+px`` on ``axis`` ('yaw'|'pitch'), then ``-px`` back. Return
    the observed ``px / deg`` from the forward push, or None on bad read.

    Pitch tests pre-recenter to ~0° (sequence of small UP/DOWN bursts) so a
    large delta doesn't hit the ±90° clamp and silently truncate, which
    would inflate px/deg. Yaw wraps cleanly so no recenter is needed."""
    if axis == "pitch":
        # Re-center toward 0° pitch: 0 → ±90 clamp is a hard wall. Push
        # toward 0 in chunks; F3 OCR confirms when we're close enough.
        for _ in range(6):
            _, p = _read_pose(capture, f3)
            if p is None or abs(p) < 4.0:
                break
            # 1 deg ≈ ~6 px on this rig — small correction.
            corr = int(round(-p * 6.0))
            corr = max(-200, min(200, corr))
            _push(mouse, "pitch", corr)
            time.sleep(0.25)

    y0, p0 = _read_pose(capture, f3)
    if (axis == "yaw" and y0 is None) or (axis == "pitch" and p0 is None):
        return None
    _push(mouse, axis, px)
    time.sleep(settle)
    y1, p1 = _read_pose(capture, f3)

    # Return to roughly the starting pose so the next test isn't biased.
    _push(mouse, axis, -px)
    time.sleep(settle)

    if axis == "yaw":
        if y1 is None: return None
        d = abs(_norm_yaw(y1 - y0))
    else:
        if p1 is None: return None
        d = abs(p1 - p0)
        # Clamp guard: if we asked for ≥30° but moved <2/3 of that, the
        # clamp probably bit us mid-push — discard.
        expected = abs(px) / 6.0
        if expected >= 30.0 and d < 0.66 * expected:
            return None
    if d < 0.5:
        return None
    return abs(px) / d


def _persisted_px_per_deg() -> Tuple[Optional[float], Optional[float]]:
    """Return (yaw_pxpd, pitch_pxpd) from the on-disk calibrator, if any."""
    try:
        import json
        path = ROOT / "data" / "calibration" / "mouse_calibration.json"
        if not path.is_file():
            return (None, None)
        data = json.loads(path.read_text(encoding="utf-8"))
        return (data.get("px_per_deg_yaw"), data.get("px_per_deg_pitch"))
    except Exception:
        return (None, None)


def _parse_ints(s: str) -> List[int]:
    return [int(x) for x in s.replace(" ", "").split(",") if x]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaw-px", type=str, default="120,240,360,480",
                    help="Comma list of horizontal push sizes (px).")
    ap.add_argument("--pitch-px", type=str, default="80,160,240",
                    help="Comma list of vertical push sizes (px). Pitch "
                         "clamps at ±90, so keep these moderate.")
    ap.add_argument("--settle", type=float, default=0.45,
                    help="Seconds to wait after each push for F3 to update.")
    ap.add_argument("--countdown", type=int, default=2)
    args = ap.parse_args(argv)

    wins = _find_minecraft_hwnd()
    if not wins:
        print("[px/deg][ERROR] Minecraft (javaw.exe) isn't running.")
        return 2
    hwnd = wins[0][0]

    settings = M._load_yaml(M.SETTINGS_PATH)

    activate_minecraft()
    time.sleep(0.3)

    capture = Capture(CaptureConfig(hwnd=hwnd, threaded=True, max_fps=60,
                                    name="pxpd_capture"))
    capture.start()
    gate = InputGate(); gate.set_allowed(True)
    mouse = Mouse(config=MouseConfig(), gate=gate)
    mouse.start()
    f3 = build_f3_reader(settings)
    print(f"[px/deg] F3 backend: {f3.backend}")

    # Warm the threaded capture.
    for _ in range(3):
        capture.get_frame(); time.sleep(0.05)

    y, p = _read_pose(capture, f3)
    if y is None or p is None:
        print("[px/deg][WARN] couldn't read yaw/pitch from F3. Make sure F3 "
              "is ON in Minecraft (the always-on debug lines are enough).")
    print(f"[px/deg] starting pose: yaw={y} pitch={p}")

    for n in range(args.countdown, 0, -1):
        print(f"[px/deg] measuring in {n}…"); time.sleep(1.0)

    yaw_results:   List[Tuple[int, float]] = []
    pitch_results: List[Tuple[int, float]] = []

    try:
        for dx in _parse_ints(args.yaw_px):
            r = _measure_one(mouse, capture, f3, axis="yaw",
                             px=dx, settle=args.settle)
            if r is None:
                print(f"  yaw  +{dx:>4} px  -> (no usable F3 read)")
            else:
                yaw_results.append((dx, r))
                print(f"  yaw  +{dx:>4} px  ->  {r:6.2f} px/deg")

        for dy in _parse_ints(args.pitch_px):
            r = _measure_one(mouse, capture, f3, axis="pitch",
                             px=dy, settle=args.settle)
            if r is None:
                print(f"  pitch +{dy:>4} px  -> (no usable F3 read)")
            else:
                pitch_results.append((dy, r))
                print(f"  pitch +{dy:>4} px  ->  {r:6.2f} px/deg")
    finally:
        try: mouse.stop()
        except Exception: pass
        try: capture.stop()
        except Exception: pass

    def _summarise(label, results):
        if not results:
            print(f"\n  {label}: no usable measurements.")
            return None
        vals = [v for _, v in results]
        m = sum(vals) / len(vals)
        spread = max(vals) - min(vals)
        print(f"\n  {label}: mean = {m:.2f} px/deg "
              f"(min {min(vals):.2f}, max {max(vals):.2f}, "
              f"spread {spread:.2f})")
        if spread > 0.5 and m > 0 and spread / m > 0.10:
            print(f"  {label}: > 10% spread — sensitivity slider may have "
                  "moved between runs, or motion is hitting the pitch clamp.")
        return m

    print("\n" + "=" * 56)
    yaw_mean   = _summarise("yaw  ", yaw_results)
    pitch_mean = _summarise("pitch", pitch_results)

    cal_yaw, cal_pitch = _persisted_px_per_deg()
    print("\n  persisted calibrator "
          "(data/calibration/mouse_calibration.json):")
    print(f"    yaw   px/deg: {cal_yaw}")
    print(f"    pitch px/deg: {cal_pitch}")

    if yaw_mean is not None and cal_yaw is not None and cal_yaw > 0:
        drift = (yaw_mean - cal_yaw) / cal_yaw * 100.0
        flag = "  <-- update the macro's yaw deltas" if abs(drift) > 15 else ""
        print(f"\n  measured yaw vs persisted: {drift:+.1f}%{flag}")
    if pitch_mean is not None and cal_pitch is not None and cal_pitch > 0:
        drift = (pitch_mean - cal_pitch) / cal_pitch * 100.0
        flag = "  <-- update the macro's pitch delta" if abs(drift) > 15 else ""
        print(f"  measured pitch vs persisted: {drift:+.1f}%{flag}")

    if yaw_mean is not None:
        # Useful conversion right at the prompt.
        deg_for_22 = 22.0 / yaw_mean
        deg_for_44 = 44.0 / yaw_mean
        print("\n  god_bridge yaw deltas in degrees at the measured rate:")
        print(f"    22 px  ≈ {deg_for_22:.1f}°    "
              f"44 px ≈ {deg_for_44:.1f}°")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
