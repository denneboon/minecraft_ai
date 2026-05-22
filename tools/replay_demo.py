#!/usr/bin/env python3
# tools/replay_demo.py
"""
Replay a recorded demo session with an input overlay.

Usage
-----
::

    python tools/replay_demo.py data/raw/<session-id>
    python tools/replay_demo.py data/raw/<session-id> --start 200
    python tools/replay_demo.py data/raw/<session-id> --speed 2.0

Why this exists
---------------
A replay viewer is the cheapest way to verify a demo recording is
actually useful for training. It also doubles as a debug tool: when an
agent does something weird in a live run we can save the captured
frames + events and step through them here to see exactly what its
perception was.

Controls
--------
* SPACE      — play / pause
* → / d      — step one frame forward (auto-pauses)
* ← / a      — step one frame back   (auto-pauses)
* + / =      — speed up   (×1.5)
* −          — slow down  (÷1.5)
* r          — restart from the beginning
* q / ESC    — quit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


def _load_session(session_dir: Path):
    meta_path   = session_dir / "meta.json"
    events_path = session_dir / "events.jsonl"
    if not meta_path.is_file():
        raise FileNotFoundError(f"No meta.json at {meta_path}")
    if not events_path.is_file():
        raise FileNotFoundError(f"No events.jsonl at {events_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    events = [
        json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return meta, events


def _draw_overlay(canvas: np.ndarray,
                  meta: Dict,
                  event: Dict,
                  *,
                  paused: bool,
                  speed: float) -> None:
    """
    Mutates ``canvas`` in place to add a translucent debug overlay
    showing the recorded input state for this frame.
    """
    h, w = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Semi-transparent panel along the top so text is readable on any
    # background.
    panel_h = 110
    panel = canvas[:panel_h, :].copy()
    cv2.rectangle(panel, (0, 0), (w, panel_h), (0, 0, 0), -1)
    canvas[:panel_h, :] = cv2.addWeighted(canvas[:panel_h, :], 0.35,
                                          panel, 0.65, 0)

    # Header line.
    state = "PAUSED" if paused else f"PLAY ×{speed:.2f}"
    header = (
        f"frame {event['frame']:>6d} / {meta.get('frame_count','?'):<6}  "
        f"t={event.get('ts', 0):6.2f}s   "
        f"[{state}]   "
        f"focused={event.get('mc_focused', '?')}"
    )
    cv2.putText(canvas, header, (10, 22), font, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)

    # Keys held.
    keys = event.get("keys") or []
    key_str = " ".join(keys) if keys else "—"
    cv2.putText(canvas, f"keys: {key_str}", (10, 46),
                font, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    # Mouse buttons + scroll.
    btns = event.get("buttons") or {}
    btn_bits = []
    for name in ("left", "right", "middle"):
        if btns.get(name):
            btn_bits.append(name.upper())
    btn_str = " + ".join(btn_bits) if btn_bits else "—"
    scroll = event.get("scroll_dy", 0) or 0
    cv2.putText(canvas, f"mouse: {btn_str}    scroll_dy={scroll:+d}",
                (10, 70), font, 0.55, (0, 200, 255), 1, cv2.LINE_AA)

    # F3 data.
    yaw   = event.get("yaw")
    pitch = event.get("pitch")
    xyz   = event.get("xyz")
    f3_bits = []
    if xyz is not None:
        f3_bits.append(f"xyz=({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f})")
    if yaw is not None:
        f3_bits.append(f"yaw={yaw:+6.1f}")
    if pitch is not None:
        f3_bits.append(f"pitch={pitch:+6.1f}")
    f3_str = "   ".join(f3_bits) if f3_bits else "(F3 unavailable)"
    cv2.putText(canvas, f"f3: {f3_str}", (10, 94),
                font, 0.55, (255, 255, 0), 1, cv2.LINE_AA)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Replay a recorded demo session.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("session", type=str,
                   help="Path to a session directory under data/raw/")
    p.add_argument("--start", type=int, default=1,
                   help="Frame index to start at (1-based).")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Initial playback speed multiplier.")
    args = p.parse_args(argv)

    session_dir = Path(args.session).resolve()
    if not session_dir.is_dir():
        print(f"[replay] Not a directory: {session_dir}")
        return 2

    try:
        meta, events = _load_session(session_dir)
    except Exception as e:
        print(f"[replay] Failed to load session: {e}")
        return 2
    if not events:
        print("[replay] events.jsonl is empty; nothing to play.")
        return 2

    frames_dir = session_dir / "frames"
    has_frames = frames_dir.is_dir() and any(frames_dir.iterdir())
    if not has_frames:
        print("[replay] Session was recorded with --no-frames. "
              "Overlay-only playback (no images).")

    fps_target = float(meta.get("fps_realised") or meta.get("fps_target") or 20)
    speed = max(0.05, float(args.speed))
    paused = False
    i = max(0, min(len(events) - 1, args.start - 1))

    window = f"replay  —  {session_dir.name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window,
                     meta.get("frame_resolution", [960, 564])[0],
                     meta.get("frame_resolution", [960, 564])[1])

    print(f"[replay] {len(events)} events  "
          f"{meta.get('duration_sec', 0):.1f}s  "
          f"@ {fps_target:.1f} Hz")
    print("[replay] SPACE play/pause   ←/→ step   +/- speed   r restart   q quit")

    blank: Optional[np.ndarray] = None
    if not has_frames:
        size = meta.get("frame_resolution", [960, 564])
        blank = np.zeros((size[1], size[0], 3), dtype=np.uint8)

    while True:
        event = events[i]

        # Locate the frame for this event. Frames are written 1-indexed
        # to match the event's frame field.
        if has_frames:
            path = frames_dir / f"{event['frame']:09d}.png"
            frame = cv2.imread(str(path)) if path.is_file() else None
            if frame is None:
                # Fall back to a blank canvas so the run keeps going.
                size = meta.get("frame_resolution", [960, 564])
                frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                cv2.putText(frame, f"<missing {path.name}>",
                            (20, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2, cv2.LINE_AA)
        else:
            frame = blank.copy()  # type: ignore[union-attr]

        _draw_overlay(frame, meta, event, paused=paused, speed=speed)
        cv2.imshow(window, frame)

        # Wait. When paused, we block until a key; when playing, we
        # honour fps_target / speed to feel like real time.
        if paused:
            wait_ms = 0
        else:
            wait_ms = max(1, int(1000.0 / (fps_target * speed)))
        key = cv2.waitKey(wait_ms) & 0xFF

        if key in (ord("q"), 27):           # q / ESC
            break
        elif key == ord(" "):
            paused = not paused
        elif key in (ord("a"), 81):         # 81 = left arrow on some
            paused = True
            i = max(0, i - 1)
            continue
        elif key in (ord("d"), 83):         # 83 = right arrow
            paused = True
            i = min(len(events) - 1, i + 1)
            continue
        elif key in (ord("+"), ord("=")):
            speed = min(16.0, speed * 1.5)
            print(f"[replay] speed: ×{speed:.2f}")
        elif key == ord("-"):
            speed = max(0.05, speed / 1.5)
            print(f"[replay] speed: ×{speed:.2f}")
        elif key == ord("r"):
            i = 0
            print("[replay] restart")
            continue

        if not paused:
            i += 1
            if i >= len(events):
                i = len(events) - 1
                paused = True
                print("[replay] reached end — paused")

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
