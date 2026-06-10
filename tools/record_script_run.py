#!/usr/bin/env python3
"""
Run a macro / script AND record the gameplay at the same time, so the
result can be replayed visually with ``tools/replay_demo.py``.

The recorder pulls live frames from ``vision.capture``, snapshots held
keys + mouse-button state via pynput hooks (which DO observe the
macro's synthesised SendInput events on Windows), and reads F3 yaw /
pitch / xyz at ~5 Hz. The script runner drives input through the same
gated keyboard / mouse the agents use, so focus-loss auto-stop and
Ctrl+Shift+F12 still apply.

Output layout matches ``scripts/collect_demo.py`` so the existing
``tools/replay_demo.py`` viewer works unchanged.

Usage
-----
    python tools/record_script_run.py scripts/macros/god_bridge.ahk
    python tools/replay_demo.py data/raw/<session-id>     # visual playback
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Make sibling helpers in scripts/collect_demo.py importable so we don't
# duplicate the recorder/writer/key-normalisation code.
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
from control.script_runner import (   # noqa: E402
    ScriptRunner, ScriptRunnerConfig, load_script,
)
from utils.focus import _find_minecraft_hwnd, activate_minecraft  # noqa: E402
from vision.capture import Capture, CaptureConfig  # noqa: E402
from vision.ocr import build_f3_reader  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("script", help="Macro/script file (.ahk/.json/.mcs/.txt).")
    ap.add_argument("--fps", type=float, default=20.0,
                    help="Recorder frame rate (capture is independent).")
    ap.add_argument("--downscale", type=float, default=0.5,
                    help="Save-frame downscale factor (1.0 = native).")
    ap.add_argument("--countdown", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=120.0,
                    help="Wall-clock safety cap on the script.")
    ap.add_argument("--output", type=str, default=None,
                    help="Output dir. Default: data/raw/script_<stem>_<ts>/")
    ap.add_argument("--pad-after-sec", type=float, default=0.5,
                    help="Keep recording this much after the script ends "
                         "so you see the released-keys / settle frames.")
    args = ap.parse_args(argv)

    # ── Resolve & parse the script ────────────────────────────────────
    try:
        name, ops = load_script(args.script)
    except Exception as e:
        print(f"[record_script_run][ERROR] could not load {args.script!r}: {e}")
        return 2

    # ── Need Minecraft running ───────────────────────────────────────
    wins = _find_minecraft_hwnd()
    if not wins:
        print("[record_script_run][ERROR] Minecraft (javaw.exe) isn't running.")
        return 2
    hwnd = wins[0][0]

    # ── Wire the standard agent stack (focus, gate, kb, mouse, capture, f3) ──
    settings = M._load_yaml(M.SETTINGS_PATH)
    keymap_flat = M._flatten_keymap_for_keyboard(M._load_json(M.KEYMAP_PATH))

    capture = Capture(CaptureConfig(
        hwnd=hwnd, threaded=True,
        max_fps=float(args.fps) * 2,    # let mss grab faster than we sample
        track_window_each_frame=True,
        name="rec_script_capture",
    ))

    gate = InputGate()
    safety = M.build_safety(settings, gate=gate)
    keyboard = M.build_keyboard(settings, keymap_flat, gate=gate)
    mouse = M.build_mouse(settings, gate=gate)
    try:
        f3_reader = build_f3_reader(settings)
    except Exception as e:
        print(f"[record_script_run][WARN] F3 reader unavailable: {e}; "
              "yaw/pitch/xyz will be null in events.")
        f3_reader = None

    # Output dir.
    stem = Path(args.script).stem
    sid = (Path(args.output).name if args.output
           else f"script_{stem}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    out_dir = Path(args.output) if args.output else ROOT / "data" / "raw" / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    events_path = out_dir / "events.jsonl"

    print(f"[record_script_run] script={name!r}  ops={len(ops)}")
    print(f"[record_script_run] output → {out_dir}")

    # ── Start subsystems ─────────────────────────────────────────────
    activate_minecraft()
    safety.start()
    keyboard.start()
    mouse.start()
    capture.start()
    time.sleep(0.25)

    input_rec = InputRecorder()
    input_rec.start()
    writer = FrameWriter(frames_dir)
    writer.start()

    # Countdown FIRST so the safety controller has time to detect focus
    # and open the input gate — otherwise the F3 toggle tap below is
    # silently dropped by the closed gate.
    for n in range(max(0, args.countdown), 0, -1):
        print(f"[record_script_run] starting in {n}…")
        time.sleep(1.0)

    # Auto-toggle F3 ON so the recorder captures yaw/pitch/xyz. Without
    # it events show ``yaw/pitch=null`` and there's no ground truth for
    # the macro's actual in-game effect — the whole point of recording.
    # Detection via the F3 OCR itself (does it actually return a pose?)
    # is more robust than ``main._ensure_f3_panel_on``'s top-left
    # brightness check, which gives false negatives in bright biomes.
    def _f3_on() -> bool:
        if f3_reader is None:
            return True
        try:
            info = f3_reader.read(capture.get_frame())
            return info.yaw is not None or info.pitch is not None
        except Exception:
            return False

    if safety.allow_input() and not _f3_on():
        for attempt in range(2):
            keyboard.tap("f3", 0.05)
            time.sleep(0.35)
            if _f3_on():
                print(f"[record_script_run] F3 ON after tap {attempt+1}")
                break
        else:
            print("[record_script_run][WARN] F3 panel didn't come on after 2 "
                  "taps — pose data will be null. Toggle F3 manually if needed.")

    if not safety.allow_input():
        print("[record_script_run][WARN] Minecraft isn't focused — the script "
              "will abort at the closed gate. Click into MC and try again.")

    # ── Recorder loop (background thread) ────────────────────────────
    stop_evt = threading.Event()
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
                time.sleep(0.05)
                continue
            if not full_w:
                full_h, full_w = frame_full.shape[:2]
                if args.downscale != 1.0:
                    out_w = max(1, int(round(full_w * args.downscale)))
                    out_h = max(1, int(round(full_h * args.downscale)))
                else:
                    out_w, out_h = full_w, full_h

            if (f3_reader is not None
                    and (t0 - last_f3_ts) >= F3_OCR_INTERVAL_SEC):
                try:
                    info = f3_reader.read(frame_full)
                    if info.yaw is not None:
                        last_yaw = float(info.yaw)
                    if info.pitch is not None:
                        last_pitch = float(info.pitch)
                    if info.position() is not None:
                        last_xyz = info.position()
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
                "frame": frame_idx,
                "ts": round(elapsed, 4),
                "keys": snap.keys,
                "buttons": snap.buttons,
                "scroll_dy": snap.scroll_dy,
                "mc_focused": _mc_foreground(hwnd),
                "yaw": round(last_yaw, 2) if last_yaw is not None else None,
                "pitch": round(last_pitch, 2) if last_pitch is not None else None,
                "xyz": ([round(c, 3) for c in last_xyz]
                        if last_xyz else None),
            }
            events_file.write(json.dumps(event) + "\n")

            dt = time.perf_counter() - t0
            if dt < tick_period:
                time.sleep(tick_period - dt)

    rec_thread = threading.Thread(target=record_loop, name="rec-loop",
                                  daemon=True)
    rec_thread.start()

    # ── Drive the script in the main thread ──────────────────────────
    runner = ScriptRunner(
        keyboard, mouse, gate=gate,
        config=ScriptRunnerConfig(max_runtime_sec=float(args.max_seconds)),
    )
    completed = False
    try:
        completed = runner.run(ops, name=name)
    finally:
        # Capture the release-frames so the replay shows the script
        # actually letting go of keys.
        time.sleep(max(0.0, args.pad_after_sec))
        stop_evt.set()
        rec_thread.join(timeout=2.0)
        try:
            events_file.flush(); events_file.close()
        except Exception:
            pass

        writer.stop(); writer.join(timeout=5.0)
        input_rec.stop()
        for sub in (keyboard, mouse):
            try: sub.stop()
            except Exception: pass
        safety.stop()
        capture.stop()

    duration = time.perf_counter() - start_perf
    meta = {
        "session_id": sid,
        "script_path": str(args.script),
        "script_name": name,
        "script_ops": len(ops),
        "script_completed": completed,
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
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2),
                                       encoding="utf-8")

    print(f"\n[record_script_run] {'completed' if completed else 'ABORTED'}: "
          f"{frame_idx} frames over {duration:.1f}s "
          f"({meta['fps_realised']:.1f} Hz realised)")
    print(f"[record_script_run] replay with:\n"
          f"    python tools/replay_demo.py {out_dir}")
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
