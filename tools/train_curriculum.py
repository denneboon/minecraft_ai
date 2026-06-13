#!/usr/bin/env python3
r"""
Autonomous CURRICULUM trainer — diverse self-teaching across biomes, weather,
and time, driven entirely by Minecraft commands (cheats must be ON).

The old `train_overnight.py` only ever saw wherever the bot happened to stand,
so the dataset was narrow. This director instead:

  1. ROAMS the world with /spreadplayers (lands on the surface) so it gathers
     blocks from many biomes, not one spot.
  2. At each stop, cycles WEATHER and TIME via /weather + /time and labels every
     sample with that GROUND TRUTH (perception.set_environment) — so each
     condition trains cleanly + separately, no detection guesswork.
  3. NEVER poisons the set with a bad view:
       * sample capture is SUPPRESSED during every teleport/transit;
       * a submerged/lava-tinted frame (global colour cast) suppresses capture
         and triggers an immediate re-roam — so underwater junk never lands AND
         the bot leaves the water before it can drown (the one death risk on
         PEACEFUL, where there are no mobs and no hunger);
       * a "rain"/"thunder" label is only trusted when the SUBTITLE reader
         confirms precipitation is actually falling (so commanding rain in a
         desert — where nothing falls — or a snowy biome — where it's silent —
         doesn't mislabel a clear-looking scene). Snow is labelled when a snowy
         biome is confirmed from F3 (teleport there + /weather rain -> silent
         snow -> labelled "snow").

NO potion effects are used — the world is PEACEFUL (no mob/hunger death), so
the only way to die is drowning while sitting still, which the submerged-guard
re-roam prevents.

It self-teaches the live CNN in the background (like train_overnight); the
diverse samples it banks are what the GPU box later trains the fusion model on.

Safety / unattended (same contract as train_overnight):
  * Only acts while MC is FOREGROUND; on focus loss it stops controlling and
    exits if you stay away ~15s (never grabs focus back).
  * Panic stop any time: Ctrl+Shift+F12 (also Ctrl+Shift+X / End / Pause).
  * Restores /weather clear + /time set day on exit.

Usage:
    python tools/train_curriculum.py                      # ~8h, roam+cycle
    python tools/train_curriculum.py --minutes 120 --segment-sec 70
    python tools/train_curriculum.py --center 0 0 --range 6000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

import main as M
from utils.focus import _find_minecraft_hwnd, activate_minecraft
from control.input_gate import InputGate
from vision.capture import Capture, CaptureConfig
from vision.ocr import build_f3_reader
from vision.world import build_world_perception
from vision.world.sample_store import (
    WorldSampleStore, WorldSampleStoreConfig, default_world_sample_root,
)

# Panic poll — identical combo to make.py / train_overnight.
try:
    import ctypes
    _USER32 = ctypes.windll.user32
except Exception:
    _USER32 = None
_VK_CONTROL, _VK_SHIFT, _VK_X, _VK_END, _VK_PAUSE, _VK_F12 = \
    0x11, 0x10, 0x58, 0x23, 0x13, 0x7B


def _panic() -> bool:
    if _USER32 is None:
        return False
    try:
        g = _USER32.GetAsyncKeyState
        d = lambda vk: (g(vk) & 0x8000) != 0
        cs = d(_VK_CONTROL) and d(_VK_SHIFT)
        return (cs and (d(_VK_F12) or d(_VK_X))) or d(_VK_END) or d(_VK_PAUSE)
    except Exception:
        return False


# ── Pure, testable helpers ──────────────────────────────────────────────────

def biome_precip(biome):
    """Coarse precipitation type for a biome id/name: 'snow' | 'rain' | 'none'
    | None(unknown). Used ONLY to label snow (which is silent) and to skip
    commanding rain where nothing would fall; rain itself is confirmed live by
    the subtitle reader, so a misclass here can't mislabel a scene."""
    if not biome:
        return None
    b = str(biome).split(":")[-1].lower()
    if any(s in b for s in ("snow", "frozen", "ice", "grove", "peaks")):
        return "snow"
    if any(s in b for s in ("desert", "savanna", "badlands", "nether",
                            "basalt", "crimson", "warped", "soul", "the_end",
                            "end_")):
        return "none"
    return "rain"


def is_corrupted_view(frame) -> bool:
    """True if the view has the underwater (blue/teal) or lava (orange)
    full-screen overlay — NOT a clean look at a block, must not be sampled.

    Checks ONLY the LOWER-CENTRE band — the ground/terrain zone below the
    horizon and above the hotbar. In a normal scene that's solid terrain
    (green/brown/grey), never blue; the submerged overlay tints it (and the
    whole screen) blue-teal, lava tints it orange. Sampling here — not the
    whole frame — is what stops a bright BLUE SKY (which fills the UPPER frame
    of any normal outdoor view) from false-positiving as 'underwater'."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return False
    h, w = frame.shape[:2]
    y0, y1 = int(h * 0.60), int(h * 0.88)     # below horizon, above hotbar
    x0, x1 = int(w * 0.32), int(w * 0.68)     # centre (clear of the hand)
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return False
    patch = patch.astype(np.float32)
    r, g, b = (float(patch[..., 0].mean()), float(patch[..., 1].mean()),
               float(patch[..., 2].mean()))
    submerged = (b > r * 1.5 and b > g * 1.15 and b > 55)   # teal: blue>green>red
    # Lava overlay is BLUE-STARVED orange (b≈10-30); ORANGE TERRAIN (badlands,
    # red sand, terracotta) keeps blue ≈75-90, so a hard blue cap separates
    # them. Without this the bot fled every orange biome and banked nothing
    # there (PC finding). (b<45 + very red-dominant = only a true full-lava
    # view; the rare case of standing in lava on peaceful.)
    lava = (r > 130 and b < 45 and r > b * 2.5 and r > g * 1.4)
    return bool(submerged or lava)


def resolve_label(weather_cmd, subtitle_state, biome):
    """Decide the TRUSTED weather label for a commanded state. The BIOME (read
    reliably from F3) is the primary signal: on the surface, commanding rain in
    a rain biome rains, in a snowy biome falls as SILENT snow, and in a dry
    biome (desert/savanna/badlands) nothing falls. The subtitle reader is an
    extra confirmation, used to decide UNKNOWN biomes (and it's fine if "Show
    Subtitles" is off — biome alone covers known biomes). Returns the label, or
    None = "don't sample this segment" (nothing would fall / can't tell)."""
    if weather_cmd == "clear":
        return "clear"            # clear looks clear in every biome
    if weather_cmd not in ("rain", "thunder"):
        return None
    precip = biome_precip(biome)
    if precip == "snow":
        return "snow"             # snowy biome: silent snow (thunder or not)
    if precip == "none":
        return None               # dry biome: nothing falls -> never mislabel
    if precip == "rain":
        return "thunder" if weather_cmd == "thunder" else "rain"
    # Unknown biome: fall back to the subtitle confirmation.
    if subtitle_state in ("rain", "thunder"):
        return "thunder" if weather_cmd == "thunder" else "rain"
    return None


def station_plan():
    """The (weather_command, time_label) segments to run at each stop. Clear is
    always safe; rain/thunder are attempted but only kept if confirmed."""
    return [
        ("clear", "day"), ("rain", "day"), ("thunder", "day"),
        ("clear", "night"), ("rain", "night"),
    ]


# ── Live director ────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=480.0)
    ap.add_argument("--checkpoint-min", type=float, default=15.0)
    ap.add_argument("--segment-sec", type=float, default=70.0,
                    help="seconds of sampling per (weather,time) segment")
    ap.add_argument("--center", type=int, nargs=2, default=[0, 0],
                    metavar=("X", "Z"), help="/spreadplayers centre")
    ap.add_argument("--range", type=int, default=30000,
                    help="/spreadplayers max range from centre. Wide by default: "
                         "the spawn belt (~6k) is all grass/dirt/leaves/logs; "
                         "reaching badlands/deserts/snow needs tens of thousands "
                         "of blocks (PC finding).")
    ap.add_argument("--pan", type=int, default=34, help="yaw mouse-move per step")
    ap.add_argument("--settle", type=float, default=0.22)
    ap.add_argument("--per-block-cap", type=int, default=400)
    ap.add_argument("--no-roam", action="store_true",
                    help="stay put (only cycle weather/time here)")
    ap.add_argument("--idle-exit-sec", type=float, default=180.0,
                    help="end the run after MC stays UNFOCUSED this long. While "
                         "unfocused it just pauses (never grabs focus back); a "
                         "brief tab-away blip resumes. Generous (3 min) so the "
                         "startup hand-off + quick checks don't kill a long "
                         "unattended run.")
    ap.add_argument("--time-day", default="noon",
                    help="/time set arg used for the 'day' label (default noon)")
    ap.add_argument("--time-night", default="midnight",
                    help="/time set arg used for the 'night' label")
    args = ap.parse_args(argv)

    wins = _find_minecraft_hwnd()
    if not wins:
        print("[curriculum] Minecraft window not found — is it running?")
        return 2
    hwnd = wins[0][0]
    activate_minecraft()
    time.sleep(0.5)

    settings = M._load_yaml(M.SETTINGS_PATH)
    gate = InputGate()
    safety = M.build_safety(settings, gate=gate)
    mouse = M.build_mouse(settings, gate=gate)
    capture = Capture(CaptureConfig(hwnd=hwnd, threaded=True))
    safety.start(); mouse.start(); capture.start()
    f3_reader = build_f3_reader(settings)
    wp = build_world_perception(settings)

    keymap_flat = M._flatten_keymap_for_keyboard(M._load_json(M.KEYMAP_PATH))
    keyboard = M.build_keyboard(settings, keymap_flat, gate=gate)
    try:
        keyboard.start()
    except Exception:
        pass

    big_store = WorldSampleStore(
        default_world_sample_root(),
        config=WorldSampleStoreConfig(max_samples_per_block=args.per_block_cap))
    wp.sample_store = big_store
    cnn = getattr(wp.block_classifier, "cnn", None)
    sample_nn = getattr(wp.block_classifier, "sample", None)
    if cnn is not None:
        cnn._store = big_store
    if sample_nn is not None:
        sample_nn._store = big_store

    from vision.world.metrics import default_metrics_root
    metrics_root = default_metrics_root(); metrics_root.mkdir(parents=True, exist_ok=True)
    status_path = metrics_root / "curriculum_status.json"
    logf = open(metrics_root / f"curriculum_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log",
                "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        try:
            logf.write(line + "\n"); logf.flush()
        except Exception:
            pass

    def cmd(text, wait=0.18):
        M._send_chat_message(keyboard, text)
        time.sleep(wait)

    def focus_ok():
        return safety.allow_input() and not _panic()

    start = time.time()
    deadline = start + args.minutes * 60.0
    ckpt_every = args.checkpoint_min * 60.0
    next_ckpt = start + ckpt_every
    start_total = big_store.total_samples()
    station = 0
    seg_done = 0
    seg_skipped = 0
    biome_hits = {}

    def write_status(extra=None):
        try:
            st = {"elapsed_min": round((time.time() - start) / 60.0, 1),
                  "remaining_min": round(max(0.0, deadline - time.time()) / 60.0, 1),
                  "stations": station, "segments": seg_done,
                  "segments_skipped": seg_skipped,
                  "samples_total": big_store.total_samples(),
                  "blocks": big_store.block_count(),
                  "biomes_seen": biome_hits,
                  "recognizer": cnn.status() if cnn is not None else "",
                  "gate_open": bool(safety.allow_input())}
            if extra:
                st.update(extra)
            status_path.write_text(json.dumps(st, indent=2), encoding="utf-8")
        except Exception:
            pass

    def wait_focus(reason=""):
        """Block while MC is unfocused; return False if the run should END
        (you stayed away ~15s) or PANIC."""
        if _panic():
            return False
        if safety.allow_input():
            return True
        lost = time.time()
        log(f"MC lost focus{(' ('+reason+')') if reason else ''} — stopped "
            f"controlling (not grabbing focus). Exiting if away "
            f"~{args.idle_exit_sec:.0f}s.")
        while not safety.allow_input():
            if _panic() or time.time() - lost > args.idle_exit_sec:
                return False
            write_status({"paused": True}); time.sleep(1.0)
        log("MC focused again — resuming.")
        return True

    def roam_and_verify():
        """Teleport to a fresh surface spot and confirm it's a clean, on-land,
        not-submerged view. Returns (ok, biome). Sampling stays SUPPRESSED
        throughout."""
        wp.set_suppress_sampling(True)
        biome = None
        for attempt in range(6):
            if not wait_focus("roam"):
                return False, None
            if not args.no_roam:
                cx, cz = args.center
                cmd(f"/spreadplayers {cx} {cz} 0 {args.range} false @s", wait=0.4)
            # Let chunks LOAD before judging: right after a teleport the client
            # shows sky/fog while terrain streams in, so an early frame is
            # all-sky — not a real view. Wait, settle, THEN judge.
            time.sleep(2.5)
            # Settle ~4s so any fall finishes + terrain renders; read the biome
            # whenever F3 surfaces it. NOTE: acceptance does NOT require a numeric
            # pose — on some displays the F3 XYZ lines garble on bright terrain
            # while the biome line still reads, and demanding a stable pose.y
            # wrongly rejected those biomes (PC finding). A time-based settle +
            # the not-submerged check (both display-robust) is enough.
            t0 = time.time()
            while time.time() - t0 < 4.0:
                frame = capture.get_frame()
                f3 = f3_reader.read(frame)
                wp.update(frame, f3)             # pose/map only (suppressed)
                biome = getattr(f3, "biome", None) or biome
                time.sleep(0.2)
            # Judge submersion (settled + loaded), 2-of-3 fresh frames, so a
            # loading-screen sky frame can't read as 'underwater'.
            time.sleep(0.3)
            bad = sum(int(is_corrupted_view(capture.get_frame()))
                      for _ in range(3)) >= 2
            if not bad:
                if biome:
                    biome_hits[biome.split(":")[-1]] = \
                        biome_hits.get(biome.split(":")[-1], 0) + 1
                return True, biome
            log(f"roam attempt {attempt+1}: submerged/lava view — retrying")
            if args.no_roam:
                return False, biome          # current spot is bad, can't move
        return False, biome

    def sample_segment(label_weather, label_time, seconds):
        """Sweep the camera collecting samples labelled (weather,time) for
        `seconds`. Returns "done", "stop" (panic/focus → end run), or "reroam"
        (the view went corrupted/submerged → leave this spot before drowning)."""
        nonlocal next_ckpt
        wp.set_environment(weather=label_weather, time_of_day=label_time)
        wp.set_suppress_sampling(False)
        t0 = time.time()
        step = 0
        while time.time() - t0 < seconds and time.time() < deadline:
            if not wait_focus("segment"):
                return "stop"
            try:
                dy = int(20 * np.sin(step * 0.4))
                mouse.move(args.pan, dy)
                time.sleep(args.settle)
                frame = capture.get_frame()
                if is_corrupted_view(frame):
                    # The spot turned submerged/lava — don't sit (would drown
                    # on peaceful) and don't bank junk: bail and re-roam.
                    wp.set_suppress_sampling(True)
                    log("  segment view went submerged/lava — leaving to re-roam")
                    return "reroam"
                f3 = f3_reader.read(frame)
                wp.update(frame, f3)                 # collects + self-teaches
                step += 1
            except Exception as e:
                log(f"segment step error: {e!r}")
                time.sleep(0.2)
            if time.time() >= next_ckpt:
                _checkpoint()
        return "done"

    def _checkpoint():
        nonlocal next_ckpt
        comp = big_store.manifest()
        top = ", ".join(f"{k.split(':')[-1]}={v}"
                        for k, v in sorted(comp.items(), key=lambda x: -x[1])[:8])
        log(f"--- checkpoint | station {station}, {seg_done} segments "
            f"({seg_skipped} skipped) | store={big_store.total_samples()}"
            f"/{big_store.block_count()} | biomes={len(biome_hits)} | "
            f"{cnn.status() if cnn else ''}")
        log(f"    blocks: {top}")
        log(f"    biomes: {biome_hits}")
        write_status()
        next_ckpt += ckpt_every

    log(f"=== CURRICULUM start: {args.minutes:.0f} min, roam={'off' if args.no_roam else 'on'} "
        f"(center={tuple(args.center)} range={args.range}), segment={args.segment_sec:.0f}s ===")
    log(f"store={big_store.total_samples()} samples / {big_store.block_count()} blocks")
    log("NO potion effects (peaceful world); the only death risk is drowning "
        "while still, prevented by the submerged-guard re-roam.")
    if not wait_focus("startup"):
        log("never got focus — aborting."); return 0
    cmd("/difficulty peaceful", wait=0.3)   # guarantee no-mob / no-hunger

    try:
        while time.time() < deadline:
            if not wait_focus("loop"):
                break
            station += 1
            ok, biome = roam_and_verify()
            if not ok:
                if _panic() or not safety.allow_input():
                    break
                log(f"station {station}: couldn't find a safe spot — skipping")
                continue
            log(f"station {station}: biome={biome or 'unknown'} "
                f"(precip={biome_precip(biome)})")
            for wcmd, tlabel in station_plan():
                if time.time() >= deadline or not wait_focus("plan"):
                    break
                tset = args.time_day if tlabel == "day" else args.time_night
                cmd(f"/time set {tset}", wait=0.3)
                cmd(f"/weather {wcmd}", wait=5.0)     # let it fully fade in
                sub = "clear"
                reader = getattr(wp, "subtitle_weather_reader", None)
                if reader is not None and wcmd != "clear":
                    seen = []
                    t0 = time.time()
                    while time.time() - t0 < 4.0:
                        seen.append(reader.read(capture.get_frame(),
                                                now=time.perf_counter()).state)
                        time.sleep(0.3)
                    sub = ("thunder" if "thunder" in seen
                           else "rain" if "rain" in seen else "clear")
                label = resolve_label(wcmd, sub, biome)
                if label is None:
                    seg_skipped += 1
                    log(f"  segment {wcmd}/{tlabel}: SKIP "
                        f"(subtitle={sub}, biome precip={biome_precip(biome)} "
                        f"— nothing to label safely)")
                    continue
                log(f"  segment {wcmd}/{tlabel} -> label weather={label}, "
                    f"time={tlabel} (subtitle={sub})")
                res = sample_segment(label, tlabel, args.segment_sec)
                wp.set_suppress_sampling(True)
                if res == "done":
                    seg_done += 1
                elif res == "reroam":
                    break                       # spot went bad → next station
                else:                           # "stop": panic / focus-exit
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        log("interrupted.")
    finally:
        try:
            wp.set_suppress_sampling(True)
            wp.set_environment(clear=True)
            if safety.allow_input():
                cmd("/weather clear", wait=0.2)
                cmd(f"/time set {args.time_day}", wait=0.2)
        except Exception:
            pass
        try:
            mouse.release_all() if hasattr(mouse, "release_all") else None
        except Exception:
            pass
        for stop in (keyboard.stop if keyboard else None, capture.stop, safety.stop):
            try:
                stop() if stop else None
            except Exception:
                pass
        end_total = big_store.total_samples()
        log(f"=== DONE: {station} stations, {seg_done} segments "
            f"({seg_skipped} skipped), {len(biome_hits)} biomes. "
            f"samples {start_total} -> {end_total} (+{end_total - start_total}). ===")
        log(f"    biomes: {biome_hits}")
        write_status({"finished": True})
        try:
            logf.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
