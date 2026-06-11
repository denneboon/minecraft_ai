from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any, Dict, Optional

# Reconfigure stdout/stderr to UTF-8 BEFORE any other module imports so
# that every print across the codebase can emit Unicode (arrows, degrees,
# ⏎ marks in the F3 dump, etc.) without crashing the Windows default
# cp1252 console with UnicodeEncodeError. Python 3.7+ guarantees
# ``reconfigure`` on the underlying io.TextIOWrapper. ``errors="replace"``
# is the belt-and-suspenders fallback if a TTY somehow rejects UTF-8 —
# offending chars become "?" rather than tearing down the agent loop.
for _stream in (sys.stdout, sys.stderr):
    _reconfig = getattr(_stream, "reconfigure", None)
    if _reconfig is not None:
        try:
            _reconfig(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass

from control.input_gate import InputGate

try:
    import yaml
except Exception:
    yaml = None
import json

from control.safety import Safety, SafetyConfig
from control.keyboard import Keyboard, KeyboardConfig
from control.mouse import Mouse, MouseConfig
from control.action_wrapper import ActionWrapper
from brain.episode_logger import build_episode_logger
from vision.capture import Capture, CaptureConfig
from vision.processing import build_processor, FrameProcessor, ScreenState
from vision.ocr import build_f3_reader, F3Reader, F3ReaderWorker
from vision.pose_filter import PoseFilter
from utils.focus import activate_minecraft, _find_minecraft_hwnd

from vision.menu_detect import MenuDetector, MenuDetection, build_menu_detector
from vision.mcfont import ensure_font_cache

# World perception is optional and off by default. We import lazily
# inside _maybe_build_world_perception so a missing assets cache
# doesn't break the rest of the agent.

from brain.interfaces import AgentAction, BaseAgent
from agents import build_agent, available_agents

try:
    from vision.calibration import calibrate as run_calibration
except Exception:
    run_calibration = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(ROOT, "config")
SETTINGS_PATH = os.path.join(CONFIG_DIR, "settings.yaml")
KEYMAP_PATH = os.path.join(CONFIG_DIR, "keymap.json")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path) or yaml is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) or {}


def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur = d
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _flatten_keymap_for_keyboard(raw: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    movement  = raw.get("movement", {})
    hotbar    = raw.get("hotbar", {})
    combat    = raw.get("combat", {})
    inventory = raw.get("inventory", {})

    if movement:
        out.update({
            "move_forward":  movement.get("forward",  "w"),
            "move_left":     movement.get("left",     "a"),
            "move_backward": movement.get("backward", "s"),
            "move_right":    movement.get("right",    "d"),
            "jump":          movement.get("jump",     "space"),
            "sneak":         movement.get("sneak",    "shift"),
            "sprint":        movement.get("sprint",   "control"),
        })

    slots = hotbar.get("slots") if isinstance(hotbar.get("slots"), (list, tuple)) else None
    if slots:
        out["hotbar_slots"] = slots

    if combat:
        out.update({
            "attack": combat.get("attack", "mouse_left"),
            "use":    combat.get("use",    "mouse_right"),
            "drop":   combat.get("drop",   "q"),
        })

    if inventory:
        out["inventory"] = inventory.get("open_inventory", "e")

    return out


# ---------------------------------------------------------------------------
# Subsystem builders
# ---------------------------------------------------------------------------

def build_safety(settings: Dict[str, Any], gate=None) -> Safety:
    title_query = _get(settings, "capture.window_title_query", "Minecraft")
    sc = SafetyConfig(
        minecraft_title_query=str(title_query),
        check_focus_interval=float(_get(settings, "safety.check_focus_interval", 0.075)),
        emergency_hotkey=tuple(_get(settings, "safety.emergency_hotkey",
                                    ["<ctrl>", "<shift>", "x"])),
        log_actions=bool(_get(settings, "safety.log_actions", True)),
        log_focus_events=bool(_get(settings, "safety.log_focus_events", True)),
        allow_run_without_focus=bool(_get(settings, "safety.allow_run_without_focus", False)),
        auto_stop_on_focus_loss=bool(_get(settings, "safety.auto_stop_on_focus_loss", True)),
        print_status_table=bool(_get(settings, "safety.print_status_table", True)),
    )
    sc.assume_focused_when_unknown = bool(_get(settings, "safety.assume_focused_when_unknown", False))
    return Safety(sc, gate=gate)


def build_keyboard(
    settings: Dict[str, Any],
    keymap_flat: Dict[str, Any],
    gate=None,
) -> Keyboard:
    kc = KeyboardConfig(
        allow_repeat_press=bool(_get(settings, "control.keyboard.allow_repeat_press", False)),
        strict_state=bool(_get(settings, "control.keyboard.strict_state", True)),
        tap_default=float(_get(settings, "control.keyboard.tap_default", 0.06)),
        max_actions_per_sec=int(_get(settings, "control.keyboard.max_actions_per_sec", 120)),
        debounce_ms=int(_get(settings, "control.keyboard.debounce_ms", 10)),
        sprint_mode=str(_get(settings, "control.keyboard.sprint_mode", "hold")),
        sneak_mode=str(_get(settings, "control.keyboard.sneak_mode", "hold")),
        autorun_key=str(keymap_flat.get("move_forward", "w")),
        sprint_key=str(keymap_flat.get("sprint", "control")),
        sneak_key=str(keymap_flat.get("sneak", "shift")),
        # 55 ms > 50 ms (1 MC tick) to guarantee each hotbar press lands on a
        # different tick even under OS scheduler jitter.
        hotbar_cooldown_ms=int(_get(settings, "control.keyboard.hotbar_cooldown_ms", 55)),
    )
    return Keyboard(config=kc, keymap=keymap_flat, gate=gate)


def build_mouse(settings: Dict[str, Any], gate=None) -> Mouse:
    defaults = MouseConfig()
    mc = MouseConfig(
        move_duration_ms=int(_get(settings, "control.mouse.move_duration_ms", 40)),
        flick_multiplier=float(_get(settings, "control.mouse.flick_multiplier", 0.35)),
        curve_steps=int(_get(settings, "control.mouse.curve_steps", 10)),
        default_click_duration=float(_get(settings, "control.mouse.default_click_duration", 0.05)),
        max_events_per_sec=int(_get(settings, "control.mouse.max_events_per_sec", 240)),
        enable_scroll=bool(_get(settings, "control.mouse.enable_scroll", True)),
        # Humanlike absolute reach (menus / inventory hovers). Defaults
        # are picked to feel like a deliberate UI move; users can tune
        # via ``control.mouse.screen_move.*`` in settings.yaml. Each
        # ``_get`` falls back to the MouseConfig dataclass default so
        # we don't restate magic numbers here.
        screen_move_min_ms=int(_get(
            settings, "control.mouse.screen_move.min_ms",
            defaults.screen_move_min_ms)),
        screen_move_max_ms=int(_get(
            settings, "control.mouse.screen_move.max_ms",
            defaults.screen_move_max_ms)),
        screen_move_ms_per_px=float(_get(
            settings, "control.mouse.screen_move.ms_per_px",
            defaults.screen_move_ms_per_px)),
        screen_move_step_hz=int(_get(
            settings, "control.mouse.screen_move.step_hz",
            defaults.screen_move_step_hz)),
        screen_move_curvature=float(_get(
            settings, "control.mouse.screen_move.curvature",
            defaults.screen_move_curvature)),
        screen_move_jitter_px=float(_get(
            settings, "control.mouse.screen_move.jitter_px",
            defaults.screen_move_jitter_px)),
    )
    return Mouse(config=mc, gate=gate)


def _maybe_build_world_perception(settings: Dict[str, Any]):
    """
    Build a ``vision.world.WorldPerception`` if enabled in settings.

    Returns the perception instance, or ``None`` when the feature flag
    is off or the build failed (missing assets cache, etc.). World
    perception runs alongside the existing pipeline — never blocks
    capture / HUD / OCR / action dispatch.
    """
    enabled = bool(_get(settings, "vision.world.enabled", False))
    if not enabled:
        return None
    try:
        from vision.world import build_world_perception
        wp = build_world_perception(settings)
        print(f"[MAIN] WorldPerception enabled "
              f"({wp.block_classifier.template_count()} block signatures).")
        return wp
    except Exception as e:
        print(f"[MAIN][WARN] WorldPerception disabled — could not build: {e}")
        return None


def build_capture(settings: Dict[str, Any]) -> Capture:
    hwnd = None
    try:
        wins = _find_minecraft_hwnd()
        if wins:
            hwnd = wins[0][0]
    except Exception:
        pass

    fps = _get(settings, "capture.fps_limit", 60)
    cc = CaptureConfig(
        hwnd=hwnd,
        window_title_query=str(_get(settings, "capture.window_title_query", "Minecraft")),
        downscale=float(_get(settings, "capture.downscale", 1.0)),
        max_fps=float(fps) if fps else None,
        track_window_each_frame=bool(_get(settings, "capture.track_window_each_frame", True)),
        strict_window_find=bool(_get(settings, "capture.strict_window_find", False)),
        clamp_to_monitor=True,
        use_client_area=bool(_get(settings, "capture.use_client_area", True)),
        # Background grabber keeps the ~16-30 ms screen grab off the
        # agent loop's critical path. On for the live loop by default;
        # set ``capture.threaded: false`` to force the old synchronous
        # grab (useful when debugging capture timing).
        threaded=bool(_get(settings, "capture.threaded", True)),
        name="minecraft_capture",
    )
    return Capture(config=cc)


# ---------------------------------------------------------------------------
# Menu / pause detection helpers
# ---------------------------------------------------------------------------

def _detect_open_menu(menu_detector: "MenuDetector",
                      capture) -> "MenuDetection":
    """
    One-shot: capture a frame and OCR the upper-screen for menu text.
    Returns the structured MenuDetection (menu name, matched keyword,
    raw recognised text). Falls through to a False detection on any
    capture/OCR error so callers can keep moving.
    """
    from vision.menu_detect import MenuDetection
    try:
        frame = capture.get_frame()
    except Exception:
        return MenuDetection(open=False)
    try:
        return menu_detector.detect(frame)
    except Exception:
        return MenuDetection(open=False)



def _hud_regions_look_empty(settings: Dict[str, Any]) -> bool:
    hr = _get(settings, "vision.hud_regions", {}) or {}
    if not isinstance(hr, dict) or not hr:
        return True
    def zeros(v):
        return isinstance(v, (list, tuple)) and all(int(x) == 0 for x in v)
    vals = [v for v in hr.values() if isinstance(v, (list, tuple))]
    return bool(vals) and all(zeros(v) for v in vals)


def _maybe_focus_and_unpause(
    settings: Dict[str, Any],
    safety: Safety,
    keyboard: Keyboard,
    capture: Capture,
    menu_detector: "MenuDetector",
) -> None:
    """
    On startup, if a menu is open, take Escape-based action to return
    to gameplay.

    The detector OCRs the upper portion of the captured frame and
    looks for menu-button text ("Back to Game", "Save and Quit to
    Title", "Advancements", "Statistics", "Options", etc.). When the
    *pause* menu is detected, we tap Escape to dismiss it. We repeat
    up to ``unpause_max_attempts`` times in case the first Escape
    didn't land — and log clearly which menu (if any) we saw and
    which keyword matched, so future agents can interact with these
    menus directly.

    If a non-pause menu is open (Advancements / Statistics / Options
    standalone screens), Escape STILL exits them in vanilla MC, so
    the same dismiss logic works for them too.
    """
    dev = settings.get("developer", {}) if isinstance(settings, dict) else {}
    if not bool(dev.get("ensure_unpaused", True)):
        return
    if not safety.allow_input():
        print("[MAIN][WARN] Skipped auto-unpause — safety gate closed at startup.")
        return

    max_attempts     = int(dev.get("unpause_max_attempts", 3))
    unpause_delay_ms = int(dev.get("unpause_delay_ms", 250))

    for attempt in range(1, max_attempts + 1):
        det = _detect_open_menu(menu_detector, capture)
        if not det.open:
            if attempt > 1:
                print(f"[MAIN] Menu dismissed after {attempt - 1} Escape tap(s).")
            return
        print(f"[MAIN] {det.menu!r} menu detected (matched '{det.matched_keyword}')"
              f" — sending Escape (attempt {attempt}/{max_attempts}).")
        keyboard.tap("escape", 0.05)
        time.sleep(unpause_delay_ms / 1000.0)

    final = _detect_open_menu(menu_detector, capture)
    if final.open:
        print(f"[MAIN][WARN] Menu still open after {max_attempts} Escape attempts: "
              f"{final.menu!r} (matched '{final.matched_keyword}'). "
              "Click into Minecraft and verify gameplay state.")


# ---------------------------------------------------------------------------
# Test sequence (smoke-test every subsystem before the real agent runs)
# ---------------------------------------------------------------------------

def _run_test_sequence(mouse: Mouse, keyboard: Keyboard, safety: Safety) -> None:
    print("[TEST] Mouse: smooth track right then back…")
    mouse.track_target(120, 0);    safety.notify_action("mouse.track_target(+120,0)")
    time.sleep(0.3)
    mouse.track_target(-120, 0);   safety.notify_action("mouse.track_target(-120,0)")
    time.sleep(0.3)

    print("[TEST] Mouse: flick up…")
    mouse.flick(0, -80);           safety.notify_action("mouse.flick(0,-80)")
    time.sleep(0.2)

    print("[TEST] Mouse: clicks…")
    mouse.left_click();            safety.notify_action("mouse.left_click()")
    time.sleep(0.15)
    mouse.right_click();           safety.notify_action("mouse.right_click()")
    time.sleep(0.15)
    mouse.simultaneous_click(0.04); safety.notify_action("mouse.simultaneous_click()")
    time.sleep(0.2)

    print("[TEST] Mouse: scroll…")
    mouse.scroll_up(1);            safety.notify_action("mouse.scroll_up(1)")
    time.sleep(0.1)
    mouse.scroll_down(1);          safety.notify_action("mouse.scroll_down(1)")
    time.sleep(0.1)

    print("[TEST] Hotbar: selecting slots in non-sequential order…")
    for slot in [4, 1, 7, 9, 2, 8, 3, 6, 5]:
        print(f"[TEST]   → slot {slot}")
        safety.notify_action(f"hotbar_slot_{slot}")
        keyboard.select_hotbar_slot(slot)

    print("[TEST] All tests complete. Holding 1 s…")
    time.sleep(1.0)


# ---------------------------------------------------------------------------
# F3 panel state management
# ---------------------------------------------------------------------------

def _panel_top_left_brightness(frame) -> float:
    """Mean greyscale brightness of the top-left 60×600 region — used
    to tell whether MC's F3 panel is currently rendered (dark overlay
    pulls the mean down to ~55-60) or gameplay is showing through
    (no overlay → mean ~70+ in most biomes)."""
    if frame is None or frame.size == 0:
        return 0.0
    region = frame[:60, :600]
    return float(region.mean()) if region.size else 0.0


def _ensure_f3_panel_on(capture, keyboard,
                       *,
                       panel_threshold: int = 68,
                       max_tries: int = 3) -> bool:
    """
    Make sure MC's F3 debug panel is rendered. Returns True if the
    panel was already on (caller should NOT toggle it off at shutdown),
    False if WE turned it on (caller should toggle off on cleanup).

    Why this is non-trivial: in MC 1.21.x the F3 key behaves as a
    TOGGLE — one keydown event flips the panel state. So a blind
    ``keyboard.press("f3")`` is a coin flip. We sample the top-left
    brightness, classify (dark panel ≈ 55-65, no panel ≈ 70+), and
    only tap when the panel is missing. If the first tap doesn't
    register (focus race / window switch), we tap again up to
    ``max_tries`` times.
    """
    try:
        frame0 = capture.get_frame()
    except Exception as e:
        print(f"[MAIN][WARN] F3-panel check: capture failed ({e})")
        return False
    brightness = _panel_top_left_brightness(frame0)
    if brightness <= panel_threshold:
        print(f"[MAIN] F3 panel already on (top-left mean={brightness:.1f})")
        return True

    print(f"[MAIN] F3 panel off (top-left mean={brightness:.1f}); tapping F3...")
    for attempt in range(1, max_tries + 1):
        keyboard.tap("f3", 0.05)
        time.sleep(0.25)
        try:
            frame_now = capture.get_frame()
        except Exception as e:
            print(f"[MAIN][WARN] F3-panel verify: capture failed ({e})")
            return False
        new_brightness = _panel_top_left_brightness(frame_now)
        if new_brightness <= panel_threshold:
            print(f"[MAIN] F3 panel ON after {attempt} tap(s) "
                  f"(top-left mean={new_brightness:.1f})")
            return False
        print(f"[MAIN][WARN] F3 panel still off after tap {attempt}/{max_tries} "
              f"(mean={new_brightness:.1f}); retrying...")
    print(f"[MAIN][WARN] F3 panel did NOT come on after {max_tries} taps. "
          f"Agent will run without F3 — pose / Targeted Block reads will be "
          f"degraded. Try toggling F3 manually before starting the agent.")
    return False


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def _send_chat_message(keyboard: Keyboard, text: str,
                       open_delay: float = 0.10,
                       send_delay: float = 0.05) -> None:
    """
    Open the in-game chat (T), type ``text``, press Enter.

    Used to bracket the agent loop with visible markers so the user
    knows when it's safe to tab away (after START) and when the run is
    done (after STOP). Best-effort — silently no-ops if anything goes
    wrong, because failing to chat should never abort a run.

    Timing: 100 ms wait for chat to open, 6 ms per typed char, 50 ms
    before pressing Enter. ~400 ms total for a 30-char message — half
    of the original 800 ms while still leaving MC enough time to
    receive each keystroke reliably. Lower these if MC starts dropping
    characters.
    """
    try:
        keyboard.tap("t", 0.03)
        time.sleep(open_delay)
        keyboard.type_text(text)
        time.sleep(send_delay)
        keyboard.tap("enter", 0.03)
    except Exception as e:
        print(f"[MAIN][WARN] chat message failed: {e}")


def _dispatch_action(
    action: AgentAction,
    actions: ActionWrapper,
    mouse: Mouse,
    keyboard: Keyboard,
) -> None:
    """
    Translate one ``AgentAction`` into hardware calls. The fields are
    independent — we dispatch movement, look, hotbar, and one-shot
    interactions in that fixed order so a tick that does "select slot 3
    and attack with it" works correctly.
    """
    # Movement (continuous holds/releases)
    if action.movement:
        actions.set_movement_state(**action.movement)

    # Hotbar selection (one-shot, but tick-safe in keyboard.py)
    if action.hotbar is not None:
        try:
            keyboard.select_hotbar_slot(int(action.hotbar))
        except Exception as e:
            print(f"[AGENT][WARN] hotbar({action.hotbar}) failed: {e}")

    # Inventory toggle
    if action.inventory_toggle:
        actions.execute("open_inventory")

    # Camera. Two paths:
    #   * Continuous velocity (look_vx/vy in px/sec): forwards to the
    #     mouse's background velocity worker for genuinely smooth
    #     motion — no per-tick gaps. Even ``set_velocity(0, 0)`` is
    #     dispatched so the worker stops when the agent commands a
    #     halt.
    #   * One-shot delta (look_dx/dy in px): the legacy eased motion.
    # An agent can use either; humanlike agents like WorldExplorer
    # should prefer the velocity API.
    has_velocity_field = (action.look_vx != 0.0 or action.look_vy != 0.0
                          or action.force_velocity)
    if has_velocity_field:
        # NaN / inf / absurd-magnitude guard. A buggy agent (broken
        # P-controller divide-by-zero, calibrator returning 0 px/deg,
        # uncapped velocity command, etc.) can hand us NaN, inf, or
        # 1e308. Without this guard those values pass through to the
        # mouse worker which then emits uncontrolled motion until the
        # value wraps. Clamp to a generous but finite range: 8000 px/s
        # at ~1.9 px/deg = 4200°/s, well past any humanlike rotation
        # rate, but bounded enough that a single bad value can't
        # produce a teleport.
        vx = action.look_vx
        vy = action.look_vy
        if not math.isfinite(vx):
            vx = 0.0
        if not math.isfinite(vy):
            vy = 0.0
        _VEL_CAP = 8000.0
        if abs(vx) > _VEL_CAP:
            vx = math.copysign(_VEL_CAP, vx)
        if abs(vy) > _VEL_CAP:
            vy = math.copysign(_VEL_CAP, vy)
        try:
            mouse.set_velocity(float(vx), float(vy))
        except Exception as e:
            print(f"[AGENT][WARN] mouse.set_velocity failed: {e}")
    if action.look_dx or action.look_dy:
        # Same NaN / inf / absurd-magnitude guard for the one-shot
        # path. ``int()`` on NaN raises ValueError; on +inf returns
        # OverflowError. Either way main would have crashed; clamp first.
        dx = action.look_dx
        dy = action.look_dy
        if not math.isfinite(dx):
            dx = 0.0
        if not math.isfinite(dy):
            dy = 0.0
        _PX_CAP = 20000  # ~10000° at calibrated px/deg — way past anything sane
        dx = max(-_PX_CAP, min(_PX_CAP, dx))
        dy = max(-_PX_CAP, min(_PX_CAP, dy))
        mouse.track_target(int(dx), int(dy))

    # Attack + use_hold are CONTINUOUS holds while commanded (mining /
    # eating) — set every tick so they release when the agent stops.
    # use_item (placing) and drop stay one-shot.
    actions.set_attack(action.interact == "attack")
    actions.set_use(action.interact == "use_hold")
    if action.interact == "use_item":
        actions.execute("use_item")
    elif action.interact == "drop_item":
        actions.execute("drop_item")


def _run_agent_loop(
    capture: Capture,
    processor: FrameProcessor,
    f3_reader: F3Reader,
    actions: ActionWrapper,
    mouse: Mouse,
    keyboard: Keyboard,
    safety: Safety,
    agent: BaseAgent,
    settings: Dict[str, Any],
    max_runtime_sec: float,
    world_perception: Optional[Any] = None,
    f3_worker: Optional[F3ReaderWorker] = None,
) -> None:
    """
    Main perception → decision → action loop.

    One iteration:
      1. Grab a frame from Capture.
      2. Process it into a GameState (HUD signals, screen state, model frame).
      3. Every ~3 Hz: run F3Reader to populate state.f3 with XYZ + facing.
      4. Pass the GameState to the agent.
      5. Dispatch the returned AgentAction onto the keyboard / mouse.

    Stops on Ctrl+C, on emergency hotkey, or after ``max_runtime_sec``.
    Always releases held movement keys on exit.
    """
    tick_rate   = float(_get(settings, "agent.tick_rate", 20))
    min_dt      = 1.0 / tick_rate
    f3_interval = 1.0 / 3.0   # 3 Hz OCR is plenty for position tracking

    print(f"[AGENT] Starting '{agent.name}' loop at {tick_rate} ticks/sec "
          f"(max {max_runtime_sec:.0f}s). Ctrl+Shift+F12 = emergency stop.")
    agent.reset()
    # Pose filter: rejects physically-impossible OCR reads
    # (teleporting hundreds of blocks per frame, pitch > 90°, etc.)
    # so a single garbled tick can't poison the perception layer.
    pose_filter = PoseFilter()
    # Episode logging: record (perception, action, outcome) every tick for
    # analytics + future learning. Best-effort; never disturbs the loop.
    episode = build_episode_logger(settings, agent.name, ROOT)
    if episode.enabled and episode.path:
        print(f"[AGENT] Episode log → {episode.path}")

    start_ts       = time.perf_counter()
    last_tick      = start_ts
    last_f3_ts     = 0.0
    last_resync_ts = start_ts
    prev_screen    = None
    prev_gate      = False
    last_f3        = None       # most recent good F3Info — carried across ticks
    resync_interval_sec = 1.0   # heartbeat re-emit of held keys
    # Profiling: log realised tick rate once per second so we can see
    # if we're hitting the target rate (default 20 Hz) or slipping.
    profile_ticks  = 0
    profile_window_start = start_ts
    total_ticks    = 0   # monotonic counter for the shutdown summary
    # Throttled error log: each distinct exception repr is printed once
    # so a recurring failure is visible without 20 Hz spam.
    _world_perception_errors_seen: set = set()

    try:
        while True:
            # Wall-clock safety stop.
            if time.perf_counter() - start_ts > max_runtime_sec:
                print(f"[AGENT] Max runtime ({max_runtime_sec:.0f}s) reached — stopping.")
                break

            # --- Capture ---
            try:
                frame = capture.get_frame()
            except Exception as e:
                print(f"[AGENT][WARN] Capture failed: {e}")
                time.sleep(0.1)
                continue

            # --- Process frame → GameState ---
            state = processor.process(frame)

            # --- F3 OCR (throttled to 3 Hz) ---
            # The OCR is intentionally rate-limited because it costs
            # ~30 ms per call; the rest of the loop runs at 20 Hz. But
            # the agent needs F3 data EVERY tick (otherwise it falls
            # into the "f3 is None" branch and releases W). Carry the
            # last good F3Info across ticks so state.f3 is always
            # populated for the agent — staleness is at most ~330 ms
            # which is fine for navigation.
            if f3_worker is not None:
                # Threaded path: the OCR worker reads + pose-filters on
                # its own thread; we just grab the latest good pose. This
                # keeps the ~70 ms OCR entirely off the control loop.
                state.f3 = f3_worker.latest()
            else:
                # Synchronous fallback (capture.threaded == false): OCR
                # inline, throttled to 3 Hz, carrying the last good pose
                # across the cheaper ticks in between.
                now = time.perf_counter()
                if (now - last_f3_ts >= f3_interval
                        and state.screen_state == ScreenState.PLAYING):
                    try:
                        fresh = f3_reader.read(frame)
                        # Validate the raw read against physical priors.
                        # Outlier rejection prevents a single misread from
                        # teleporting the perception eye 100 blocks and
                        # polluting the curiosity queue.
                        fresh = pose_filter.accept(fresh, now=now)
                        if fresh is not None and fresh.x is not None:
                            last_f3 = fresh
                        state.f3 = fresh
                    except Exception as e:
                        print(f"[AGENT][WARN] F3 OCR failed: {e}")
                    last_f3_ts = time.perf_counter()
                if state.f3 is None:
                    state.f3 = last_f3

            # --- World perception (optional) ---
            # Runs alongside the existing pipeline — adds a WorldFrame
            # to state.world that downstream agents can consume. Failures
            # never abort the loop; perception is best-effort.
            if world_perception is not None:
                try:
                    state.world = world_perception.update(frame, state.f3)
                except Exception as e:
                    # Throttled WARN: print once per error message so a
                    # recurring failure surfaces, but doesn't spam at
                    # 20 Hz. Silent-swallow is dangerous — the prior
                    # gate only printed when ``keyboard.cfg.verbose``
                    # was on, which meant perception crashes flew
                    # under the radar of every normal run.
                    key = repr(e)
                    if key not in _world_perception_errors_seen:
                        _world_perception_errors_seen.add(key)
                        print(f"[AGENT][WARN] world perception failed: {e}")
                        if keyboard.cfg.verbose:
                            import traceback as _tb
                            _tb.print_exc()

            # --- Log screen-state transitions ---
            if state.screen_state != prev_screen:
                print(f"[AGENT] screen_state: {prev_screen} → {state.screen_state}")
                prev_screen = state.screen_state

            # --- Decision ---
            try:
                decision = agent.decide(state)
            except Exception as e:
                print(f"[AGENT][WARN] agent.decide failed: {e}")
                decision = AgentAction(movement=BaseAgent.release_all_movement())

            # --- Key-state resync ---
            # Windows synthesises WM_KEYUP for held keys whenever MC
            # briefly loses foreground (e.g. when the safety status
            # table renders to the console). pynput's view of "what's
            # pressed" doesn't see those keyups, so the fast-path
            # silently stops re-emitting forward/sprint and the player
            # tap-walks. Two safeguards:
            #   1. On gate transition False→True (focus regained), force
            #      a re-emit of every held key.
            #   2. Once per second as a heartbeat, do the same.
            gate_now = safety.allow_input()
            if keyboard.cfg.verbose and gate_now != prev_gate:
                print(f"[GATE]  {prev_gate} → {gate_now}")
            now_resync = time.perf_counter()
            need_resync = (
                (gate_now and not prev_gate)
                or (gate_now and now_resync - last_resync_ts >= resync_interval_sec)
            )
            if need_resync:
                try:
                    keyboard.resync_pressed_keys()
                except Exception:
                    pass
                last_resync_ts = now_resync
            prev_gate = gate_now

            # --- Dispatch (gated by safety + screen state) ---
            dispatch_ok = gate_now and state.screen_state == ScreenState.PLAYING
            if keyboard.cfg.verbose:
                # Snapshot one line per tick: gate / screen / movement /
                # look / interact. ``look_dx/dy`` is the one-shot easing
                # path; ``look_vx/vy`` is the continuous velocity path
                # used by the WorldExplorer. Velocity-mode agents
                # otherwise leave dx/dy at 0 — log both so verbose
                # runs don't silently miss what the agent is doing.
                mv = decision.movement
                print(f"[TICK]  gate={gate_now}  screen={state.screen_state}  "
                      f"dispatch={dispatch_ok}  movement={mv}  "
                      f"look_d=({decision.look_dx},{decision.look_dy})  "
                      f"look_v=({decision.look_vx:+.0f},{decision.look_vy:+.0f})  "
                      f"interact={decision.interact}  hotbar={decision.hotbar}")
            if dispatch_ok:
                try:
                    _dispatch_action(decision, actions, mouse, keyboard)
                except Exception as e:
                    print(f"[AGENT][WARN] dispatch failed: {e}")
            else:
                # Not allowed to act — be sure no movement keys or the
                # attack button are stuck.
                if keyboard.cfg.verbose:
                    print(f"[DISPATCH] BLOCKED — releasing all movement "
                          f"(gate={gate_now}, screen={state.screen_state})")
                try:
                    actions.release_all_movement()
                    actions.set_attack(False)
                    actions.set_use(False)
                except Exception:
                    pass

            # --- Episode logging (best-effort) ---
            episode.record(total_ticks, time.perf_counter() - start_ts,
                           state, decision, agent=agent, dispatched=dispatch_ok)

            # --- Tick pacing + rate log ---
            now = time.perf_counter()
            elapsed = now - last_tick
            if elapsed < min_dt:
                time.sleep(min_dt - elapsed)
            last_tick = time.perf_counter()

            profile_ticks += 1
            total_ticks   += 1
            if last_tick - profile_window_start >= 1.0:
                rate = profile_ticks / (last_tick - profile_window_start)
                if keyboard.cfg.verbose:
                    print(f"[RATE]  {rate:5.1f} ticks/sec "
                          f"(target {tick_rate:.0f}, "
                          f"held={sorted(keyboard._pressed)})")
                profile_ticks = 0
                profile_window_start = last_tick
    except KeyboardInterrupt:
        print("[AGENT] Ctrl+C received — stopping.")
    finally:
        # Halt the velocity worker IMMEDIATELY when the loop ends.
        # Without this, the last commanded velocity (e.g. mid-sweep
        # yaw rate of +68 px/s) keeps emitting motion in the
        # background until ``mouse.stop()`` runs — and ``mouse.stop()``
        # runs AFTER the multi-second world-map dump, so the cursor
        # visibly drifts for the entire duration of the shutdown
        # sequence. ``set_velocity(0, 0)`` updates the worker's
        # in-memory vx/vy to zero on the next loop iteration (~4 ms),
        # stopping motion well before the user sees the "Loop ended"
        # message.
        try:
            mouse.set_velocity(0.0, 0.0)
        except Exception:
            pass
        try:
            actions.set_attack(False)      # never leave the mouse held down
            actions.set_use(False)
        except Exception:
            pass
        # ALWAYS release every movement key on exit so we don't strand
        # the player walking forward into lava after the loop ends.
        try:
            actions.release_all_movement()
        except Exception:
            pass
        try:
            agent.shutdown()
        except Exception:
            pass
        # One-line summary: total wall time, total ticks, realised rate.
        # Always shown (regardless of --debug) — useful for noticing
        # if a frame-processing change tanked the loop rate.
        wall = time.perf_counter() - start_ts
        actual_rate = (total_ticks / wall) if wall > 0 else 0.0
        print(f"[AGENT] Loop ended after {wall:.1f}s "
              f"({total_ticks} ticks, {actual_rate:.1f} Hz actual / "
              f"{tick_rate:.0f} Hz target).")
        try:
            summ = {"wall_s": round(wall, 1), "hz": round(actual_rate, 1)}
            if hasattr(agent, "telemetry"):
                try: summ["final"] = agent.telemetry()
                except Exception: pass
            episode.close(summ)
        except Exception:
            pass
        # OCR throughput — the high-value perception (pose / looking-at).
        if f3_worker is not None:
            try:
                n_ocr = f3_worker.reads()
                print(f"[AGENT] F3 OCR reads: {n_ocr} "
                      f"({n_ocr / wall:.1f} Hz) over the run.")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the Minecraft AI agent loop or a hardware smoke test.",
    )
    p.add_argument(
        "--agent",
        type=str,
        default=None,
        help=(
            "Agent name to run. Use 'none' (or '--test') to skip the "
            "agent loop and run the legacy mouse/hotbar smoke test. "
            f"Available agents: {', '.join(available_agents()) or '(none registered)'}"
        ),
    )
    p.add_argument(
        "--test", action="store_true",
        help="Run the legacy hardware smoke test instead of an agent.",
    )
    p.add_argument(
        "--script", type=str, default=None,
        help="Play a recorded macro/script file instead of an agent. "
             "Supports AutoHotkey (.ahk), Macro/Keybind-Mod (.txt), a "
             "clean op list (.json), or the simple line DSL (.mcs). "
             "e.g. --script scripts/macros/bridge.ahk. --duration caps "
             "the run (wall-clock safety cap).",
    )
    p.add_argument(
        "--duration", type=float, default=None,
        help="Max seconds to run the agent loop before auto-stopping. "
             "Default reads agent.max_runtime_sec from settings (or 60).",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Verbose log every keyboard press/release, gate transition, "
             "dispatch decision, and resync. Useful for diagnosing why a "
             "key isn't being held continuously.",
    )
    return p


def _require_minecraft_running() -> int:
    """
    Refuse to start if Minecraft (javaw.exe) isn't already running.

    This avoids the failure mode where the bot launches, finds no
    window, falls back to whatever stale capture region was cached, and
    sends inputs into the desktop. Returns 0 if MC is up, else a
    non-zero exit code.
    """
    try:
        wins = _find_minecraft_hwnd()
    except Exception as e:
        print(f"[MAIN][ERROR] Could not enumerate windows: {e}")
        return 2
    if not wins:
        print("[MAIN][ERROR] Minecraft (javaw.exe) is not running.")
        print("              Start your Minecraft AI instance in Prism Launcher")
        print("              (or any Java Edition launcher) and try again.")
        return 2
    hwnd, rect = wins[0]
    w, h = rect[2] - rect[0], rect[3] - rect[1]
    print(f"[MAIN] Found Minecraft window hwnd={hwnd} size={w}x{h}")
    return 0


def main(argv: Optional[list] = None) -> int:
    args = _build_cli().parse_args(argv)

    # ── Pre-flight: refuse to launch if MC isn't running ──────────────
    if (rc := _require_minecraft_running()) != 0:
        return rc

    print("[MAIN] Loading configuration…")
    settings    = _load_yaml(SETTINGS_PATH)
    keymap_raw  = _load_json(KEYMAP_PATH)
    keymap_flat = _flatten_keymap_for_keyboard(keymap_raw)

    try:
        if run_calibration and _hud_regions_look_empty(settings):
            print("[MAIN] HUD regions empty → running one-time calibration…")
            run_calibration(save=True)
            settings = _load_yaml(SETTINGS_PATH)
    except Exception as e:
        print(f"[MAIN][WARN] Auto-calibration skipped: {e}")

    # ── Resolve the run mode ──────────────────────────────────────────
    # CLI > settings > default. "none" disables the agent loop.
    if args.script:
        agent_name = None      # macro playback — no agent / perception
    elif args.test:
        agent_name = None
    elif args.agent:
        agent_name = None if args.agent.lower() == "none" else args.agent
    else:
        configured = str(_get(settings, "agent.default_agent", "") or "").strip()
        if configured.lower() in ("", "none", "test", "smoketest"):
            agent_name = None
        elif configured in available_agents():
            agent_name = configured
        else:
            print(f"[MAIN][WARN] agent.default_agent='{configured}' is not a "
                  f"registered agent. Falling back to smoke test. "
                  f"Available: {available_agents()}")
            agent_name = None

    max_runtime = float(
        args.duration if args.duration is not None
        else _get(settings, "agent.max_runtime_sec", 60)
    )

    # ── Construct subsystems ──────────────────────────────────────────
    gate      = InputGate()
    safety    = build_safety(settings, gate=gate)
    keyboard  = build_keyboard(settings, keymap_flat, gate=gate)
    mouse     = build_mouse(settings, gate=gate)
    capture   = build_capture(settings)
    processor = build_processor(settings)
    f3_reader = build_f3_reader(settings)

    # Optional: ask the F3 OCR to dump every line crop to disk for
    # post-mortem analysis. Enabled via the ``MCAI_F3_DUMP_DIR``
    # environment variable so it can be turned on from the shell
    # without editing settings.yaml. Each call adds a handful of PNG
    # files, so leave this off in normal use.
    f3_dump_dir = os.environ.get("MCAI_F3_DUMP_DIR")
    if f3_dump_dir:
        os.makedirs(f3_dump_dir, exist_ok=True)
        f3_reader.set_debug_dump_dir(f3_dump_dir)
        print(f"[MAIN] F3 OCR debug dump → {f3_dump_dir}")
    actions   = ActionWrapper(keyboard=keyboard, mouse=mouse, gate=gate)

    # Background F3 OCR worker. Enabled when capture is threaded (the
    # worker reads frames from a second thread, which the synchronous
    # Capture would refuse) and ``vision.ocr.threaded`` is on. It runs
    # F3Reader + PoseFilter off the control loop so the ~70 ms read
    # never stalls a tick. Falls back to inline OCR when disabled.
    f3_worker: Optional[F3ReaderWorker] = None
    if (not args.script
            and bool(_get(settings, "capture.threaded", True))
            and bool(_get(settings, "vision.ocr.threaded", True))):
        f3_worker = F3ReaderWorker(
            f3_reader, capture,
            pose_filter=PoseFilter(),
            interval_sec=float(_get(settings, "vision.ocr.read_interval_sec", 0.12)),
        )

    # Optional world perception layer. Off unless vision.world.enabled
    # is set in settings.yaml. Built up-front so the heavy classifier
    # init doesn't land inside the first agent tick. Skipped entirely in
    # macro-playback mode (it's blind input replay — no perception needed).
    world_perception = None if args.script else _maybe_build_world_perception(settings)

    # Wire --debug → keyboard verbose flag. (Mouse verbose can be
    # added the same way later if we ever need to debug camera issues.)
    debug_verbose = bool(args.debug or _get(settings, "debug.verbose_inputs", False))
    if debug_verbose:
        keyboard.cfg.verbose = True
        print("[MAIN] DEBUG verbose-inputs ON")

    # Menu detector — shares the cached MC font templates with F3 OCR.
    # Used for the auto-unpause path and future menu interactions.
    try:
        font_templates = ensure_font_cache(
            os.path.join(ROOT, "data", "calibration", "mc_font.npz"),
        )
        menu_detector = build_menu_detector(settings, font_templates)
    except Exception as e:
        print(f"[MAIN][WARN] Could not build MenuDetector: {e}. "
              "Auto-unpause will be skipped.")
        menu_detector = None

    agent: Optional[BaseAgent] = None
    if agent_name:
        try:
            agent = build_agent(agent_name, settings)
            print(f"[MAIN] Agent: {agent_name}")
            # Some agents (currently the WorldExplorer) want a direct
            # reference to the WorldPerception so they can read the
            # curiosity queue + confirmed-voxel set.
            if (hasattr(agent, "attach_perception")
                    and world_perception is not None):
                try:
                    agent.attach_perception(world_perception)
                    print("[MAIN] Attached WorldPerception to agent.")
                except Exception as e:
                    print(f"[MAIN][WARN] attach_perception failed: {e}")
        except Exception as e:
            print(f"[MAIN][WARN] build_agent({agent_name!r}) failed: {e}. "
                  f"Falling back to smoke test.")
            agent = None

    def emergency():
        # Force-release every held key/button BEFORE we tear capture
        # down. Both ``emergency_stop`` calls bypass the input gate so
        # they fire even when focus has already been lost.
        for sub in (keyboard, mouse):
            try:
                sub.emergency_stop()
            except Exception:
                pass
        # Stop the OCR worker BEFORE capture — otherwise its next
        # ``get_frame`` would re-spawn the capture grab thread we're
        # trying to tear down.
        if f3_worker is not None:
            try:
                f3_worker.stop()
            except Exception:
                pass
        for sub in (keyboard, mouse, capture):
            try:
                sub.stop()
            except Exception:
                pass
    safety.set_emergency_callback(emergency)

    print("[MAIN] Starting subsystems…")
    activate_minecraft()
    safety.start()
    keyboard.start()
    mouse.start()
    capture.start()
    time.sleep(0.25)
    if f3_worker is not None:
        f3_worker.start()
        print("[MAIN] F3 OCR worker started (threaded).")
    if menu_detector is not None:
        _maybe_focus_and_unpause(settings, safety, keyboard, capture, menu_detector)

    try:
        frame = capture.get_frame()
        h, w = frame.shape[:2]
        print(f"[MAIN] Capture OK: {w}x{h} px")
        safety.notify_action(f"capture_frame({w}x{h})")

        # Compare capture size against the calibration snapshot in
        # settings. A small mismatch is normal — DPI scaling, hidden
        # title bars, and Windows DWM border changes can each shift
        # the client rect by 30 px or so. Only HUD-dependent reads
        # (health/hunger/hotbar) actually break when the offset is
        # large; F3 OCR sits at the top-left and works regardless,
        # and world perception lazy-rebuilds its screen-ray from the
        # actual capture size each tick.
        #
        # So: WARN on mismatch, do NOT disable the agent. The agent
        # whose features actually need pixel-accurate HUD coords
        # (currently only NavigationAgent's food-eating) can handle
        # missing HUD reads gracefully (health defaults to 0 → eat
        # never triggers). The world-explorer doesn't read HUD at all.
        tol_px = int(_get(settings, "safety.capture_size_tolerance_px", 64))
        calib_res = _get(settings, "vision_calibration.last.resolution") or [None, None]
        if (calib_res and len(calib_res) == 2
                and calib_res[0] and calib_res[1]):
            cw, ch = int(calib_res[0]), int(calib_res[1])
            dw, dh = abs(w - cw), abs(h - ch)
            if dw > tol_px or dh > tol_px:
                print(f"[MAIN][WARN] Capture size {w}x{h} differs from "
                      f"calibration {cw}x{ch} by ({dw}, {dh}) px "
                      f"(tolerance {tol_px}). HUD-dependent reads may "
                      f"be off — re-run calibration if health / hunger "
                      f"detection misbehaves. The agent will still run.")
    except Exception as e:
        print(f"[MAIN][WARN] Could not capture a frame: {e}")

    # Always clean up, even on Ctrl+C or exceptions.
    chat_announced = False
    try:
        if args.script:
            # Macro/script playback mode — run a recorded input script
            # (.ahk / .txt / .json / .mcs) through the gated keyboard +
            # mouse instead of an agent. Focus-loss auto-stop and the
            # emergency hotkey still apply via the same gate.
            from control.script_runner import (
                ScriptRunner, ScriptRunnerConfig, load_script,
            )
            try:
                name, ops = load_script(args.script)
            except Exception as e:
                print(f"[MAIN][ERROR] could not load script {args.script!r}: {e}")
                return 2
            if safety.allow_input():
                _send_chat_message(keyboard, f"[bot] running macro {name}")
                chat_announced = True
                time.sleep(0.3)
            # Finite-loop macros end on their own; this wall-clock cap is
            # only a safety backstop. Use --duration to override; default
            # generous so a long macro isn't cut off by the agent's
            # (short) default runtime.
            script_cap = float(args.duration) if args.duration is not None else 300.0
            runner = ScriptRunner(
                keyboard, mouse, gate=gate,
                config=ScriptRunnerConfig(max_runtime_sec=script_cap),
            )
            runner.run(ops, name=name)
        elif agent is None:
            _run_test_sequence(mouse, keyboard, safety)
        else:
            print(f"[MAIN] Agent mode '{agent.name}'. "
                  f"Press Ctrl+C or Ctrl+Shift+F12 to stop.")
            # Announce in chat so the user knows when it's safe to tab
            # away. Only do this if the safety gate currently allows
            # input — otherwise the chat sequence would silently no-op.
            if safety.allow_input():
                _send_chat_message(
                    keyboard,
                    f"[bot] {agent.name} starting — {int(max_runtime)}s",
                )
                chat_announced = True
                time.sleep(0.3)  # let the chat line settle on screen

            # Ensure the F3 panel is ON before the agent loop starts.
            # Without it the agent can't parse pose (Facing line is
            # in-panel only) or read Targeted Block reliably (the
            # dark panel background is what makes the OCR work on
            # busy biomes).
            #
            # IMPORTANT: in MC 1.21.x the F3 key behaves as a TOGGLE,
            # not a hold-to-show. ``keyboard.press("f3")`` sends one
            # keydown, which MC interprets as a tap → toggles the
            # panel ON or OFF depending on its current state. To make
            # this deterministic we sample a frame, classify the
            # top-left brightness (dark panel ≈ 50-60, gameplay-only
            # ≈ 70+), and tap F3 only when the panel isn't already on.
            # After tapping, we re-sample and tap again if the panel
            # is still off — covers the unlucky case where MC missed
            # the first keypress (focus race at startup, etc.).
            # Auto F3-toggle is OPT-IN. MC 1.21.x treats F3 as a toggle
            # (one tap flips panel state), and detecting "is the panel
            # already on?" reliably across biomes is hard — the dark
            # translucent overlay only drops the top-left brightness
            # by ~15-20 vs the gameplay underneath, and a bright biome
            # (jungle leaves, ocean, snow) can hit that range without
            # any panel. Wrong detection → we toggle the panel OFF
            # mid-run, killing OCR until the next tap.
            #
            # In practice the agent works fine on the always-visible
            # debug lines (``looking_at_block: always``, etc.) alone.
            # If you want the dark panel for cleaner OCR, set
            # ``agent.auto_hold_f3: true`` AND make sure your starting
            # MC view has a uniformly bright top-left so the
            # threshold discriminator stays accurate.
            hold_f3 = bool(_get(settings, "agent.auto_hold_f3", False))
            f3_was_already_on = False
            if hold_f3 and safety.allow_input():
                try:
                    # Make double-sure MC has the keyboard focus right
                    # now. The startup ``activate_minecraft()`` ran
                    # earlier, but anything between then and here
                    # (status-table prints, capture grabs, the chat
                    # announce sequence) could have caused Windows to
                    # briefly pull focus away. A focus race is the
                    # most likely reason the 3 retry taps "didn't
                    # register" in earlier runs — the key events went
                    # to the terminal instead of MC.
                    activate_minecraft()
                    time.sleep(0.15)
                    f3_was_already_on = _ensure_f3_panel_on(
                        capture, keyboard,
                        panel_threshold=int(_get(
                            settings, "agent.f3_panel_brightness_threshold", 68)),
                    )
                except Exception as e:
                    print(f"[MAIN][WARN] could not toggle F3 panel: {e}")

            try:
                _run_agent_loop(
                    capture=capture, processor=processor, f3_reader=f3_reader,
                    actions=actions, mouse=mouse, keyboard=keyboard,
                    safety=safety, agent=agent, settings=settings,
                    max_runtime_sec=max_runtime,
                    world_perception=world_perception,
                    f3_worker=f3_worker,
                )
            finally:
                # Restore the user's F3 state if WE turned it on. If
                # the user already had it on at startup, leave it on.
                if hold_f3 and not f3_was_already_on:
                    try:
                        keyboard.tap("f3", 0.05)
                    except Exception:
                        pass
    finally:
        # Stop the OCR worker FIRST — before the world-map dump (it no
        # longer needs fresh poses) and crucially before capture stops,
        # so its next ``get_frame`` can't re-spawn the grab thread.
        if f3_worker is not None:
            try:
                f3_worker.stop()
            except Exception:
                pass
        # Dump the WorldMap (and a rendered iso 3D snapshot) so the
        # user can inspect what the perception layer built during the
        # run. Best-effort — never block shutdown on render errors.
        if world_perception is not None:
            try:
                import os as _os
                from datetime import datetime as _dt
                out_dir = _os.path.join(ROOT, "data", "calibration")
                _os.makedirs(out_dir, exist_ok=True)
                ts = _dt.now().strftime("%Y%m%d_%H%M%S")
                # Compact JSON (palette-deduped, sparse coords).
                # Drop-in replacement for the old verbose dump — ~8x
                # smaller for typical session sizes.
                try:
                    from vision.world import write_compact_json, write_schematic
                    json_path = _os.path.join(
                        out_dir, f"world_map_{ts}.json"
                    )
                    write_compact_json(world_perception.world_map, json_path)
                    # Sponge .schem: open in Amulet Editor (free,
                    # standalone, full game textures) or import into
                    # MC via Litematica / WorldEdit.
                    schem_path = _os.path.join(
                        out_dir, f"world_map_{ts}.schem"
                    )
                    write_schematic(world_perception.world_map, schem_path)
                    print("[MAIN] World map exported:")
                    print(f"         compact JSON -> {json_path}")
                    print(f"         schematic    -> {schem_path}")
                    print("         Open the .schem in Amulet Editor "
                          "(https://amuletmc.com) or Litematica.")
                except Exception as e:
                    print(f"[MAIN][WARN] compact export failed: {e}")
                    world_perception.world_map.dump_json(
                        _os.path.join(out_dir, f"world_map_{ts}.json")
                    )
                try:
                    import cv2 as _cv2
                    from vision.world import (
                        IsoWorldRenderer, IsoRenderConfig,
                        WorldMapRenderer, MapRenderConfig,
                    )
                    iso = IsoWorldRenderer(IsoRenderConfig())
                    top = WorldMapRenderer(MapRenderConfig())
                    last_pose = None
                    try:
                        last_pose = world_perception.last_pose()
                    except Exception:
                        last_pose = None
                    stats = world_perception.stats()
                    # Print the commit / correction breakdown so the
                    # user can see per-source contributions. Critical
                    # when ``commit_only_from_looking_at`` is False
                    # and we want to audit vision_patch quality.
                    by_src = stats.get("commits_by_source") or {}
                    by_cor = stats.get("corrections_by_source") or {}
                    if by_src or by_cor:
                        print("[MAIN] Perception commit breakdown:")
                        for k, n in by_src.items():
                            print(f"         commit  {n:5d}  {k}")
                        for k, n in by_cor.items():
                            print(f"         WRONG   {n:5d}  {k}")
                        n_commit = sum(by_src.values())
                        n_correct = sum(by_cor.values())
                        rate = (100.0 * n_correct / max(1, n_commit))
                        print(f"         total commits = {n_commit}; "
                              f"corrections = {n_correct} "
                              f"({rate:.1f}% of commits were wrong)")
                    # Print rejected-vision_patch breakdown — patches
                    # the classifier did predict but the commit gate
                    # turned away (most commonly: predictions for
                    # blocks we haven't trained on yet). This is the
                    # signal that tells us where to grow the dataset.
                    try:
                        curi = world_perception.curiosity_queue() or {}
                    except Exception:
                        curi = {}
                    if curi:
                        from collections import Counter as _Counter
                        cnt = _Counter(v.get("block_id", "?") for v in curi.values())
                        print(f"[MAIN] Curiosity queue ({len(curi)} entries) "
                              f"— blocks the gate rejected:")
                        for k, n in cnt.most_common(12):
                            print(f"         curio   {n:5d}  {k}")
                    # Curiosity-correction breakdown: when F3 confirms a
                    # voxel that was sitting in the curiosity queue under
                    # a DIFFERENT block id, that's a direct measurement of
                    # how often the classifier is wrong (and what it
                    # confuses for what). Steers dataset growth.
                    curio_cor = stats.get("curiosity_corrections") or {}
                    if curio_cor:
                        print("[MAIN] Curiosity corrections "
                              "(guess -> actual):")
                        for k, n in list(curio_cor.items())[:12]:
                            print(f"         curio?  {n:5d}  {k}")
                    samp = (stats.get("sample_store") or {})
                    print(f"[MAIN] Sample store: blocks_known="
                          f"{samp.get('blocks_known', 0)} "
                          f"total_samples={samp.get('total_samples', 0)}")

                    # Persist per-block accuracy stats across sessions.
                    # Each run appends to a JSON log; cumulative counters
                    # let us see whether a given block id has racked up
                    # corrections (sign of a misclassification pattern)
                    # or grown its commit count smoothly over time.
                    # Best-effort — never block shutdown on a write error.
                    try:
                        import json as _json
                        acc_path = _os.path.join(
                            out_dir, "perception_accuracy.json"
                        )
                        try:
                            with open(acc_path, "r", encoding="utf-8") as f:
                                acc = _json.load(f)
                            if not isinstance(acc, dict):
                                acc = {}
                        except (FileNotFoundError, _json.JSONDecodeError):
                            acc = {}
                        cum_commits = dict(acc.get("cumulative_commits", {}))
                        cum_corrs   = dict(acc.get("cumulative_corrections", {}))
                        cum_curio   = dict(acc.get("cumulative_curiosity_corrections", {}))
                        curio_cor   = stats.get("curiosity_corrections") or {}
                        for k, n in by_src.items():
                            kk = str(k)  # tuples come back as strings on rehydrate
                            cum_commits[kk] = int(cum_commits.get(kk, 0)) + int(n)
                        for k, n in by_cor.items():
                            kk = str(k)
                            cum_corrs[kk] = int(cum_corrs.get(kk, 0)) + int(n)
                        for k, n in curio_cor.items():
                            cum_curio[k] = int(cum_curio.get(k, 0)) + int(n)
                        sessions = list(acc.get("sessions", []))
                        sessions.append({
                            "ts": ts,
                            "commits":     {str(k): int(v) for k, v in by_src.items()},
                            "corrections": {str(k): int(v) for k, v in by_cor.items()},
                            "curiosity_corrections": {k: int(v) for k, v in curio_cor.items()},
                            "curiosity_size":  int(stats.get("curiosity_size", 0)),
                            "confirmed_count": int(stats.get("confirmed_count", 0)),
                            "sample_total":    int(samp.get("total_samples", 0)),
                        })
                        # Cap the per-session list so a long-lived
                        # accuracy log doesn't grow without bound.
                        # The cumulative counters carry the long memory.
                        max_sessions = 200
                        if len(sessions) > max_sessions:
                            sessions = sessions[-max_sessions:]
                        acc["cumulative_commits"]                = cum_commits
                        acc["cumulative_corrections"]            = cum_corrs
                        acc["cumulative_curiosity_corrections"]  = cum_curio
                        acc["sessions"]                          = sessions
                        # Atomic write: serialise to a temp file, then
                        # os.replace into place. Prevents a crash /
                        # power loss mid-write from corrupting the
                        # file and silently zeroing the cumulative
                        # counters on the next start (the load path
                        # catches JSONDecodeError but at the cost of
                        # losing all prior session history).
                        tmp_path = acc_path + ".tmp"
                        with open(tmp_path, "w", encoding="utf-8") as f:
                            _json.dump(acc, f, indent=2, sort_keys=True)
                            f.flush()
                            try:
                                _os.fsync(f.fileno())
                            except (OSError, AttributeError):
                                pass
                        _os.replace(tmp_path, acc_path)
                        print(f"[MAIN] Accuracy log updated -> {acc_path}")
                    except Exception as e:
                        print(f"[MAIN][WARN] Could not write accuracy log: {e}")
                    extra = [f"solid={sum(1 for _ in world_perception.world_map.iter_solid_blocks())}",
                             f"curi={stats.get('curiosity_size',0)}",
                             f"conf'd={stats.get('confirmed_count',0)}",
                             f"samp={(stats.get('sample_store') or {}).get('total_samples',0)}"]
                    img_iso = iso.render(world_perception.world_map, last_pose,
                                          extra_lines=extra)
                    img_top = top.render(world_perception.world_map, last_pose,
                                          extra_lines=extra)
                    _cv2.imwrite(_os.path.join(out_dir, f"world_iso_{ts}.png"),
                                  _cv2.cvtColor(img_iso, _cv2.COLOR_RGB2BGR))
                    _cv2.imwrite(_os.path.join(out_dir, f"world_top_{ts}.png"),
                                  _cv2.cvtColor(img_top, _cv2.COLOR_RGB2BGR))
                    print(f"[MAIN] World map snapshots written to {out_dir}/world_*_{ts}.*")
                except Exception as e:
                    print(f"[MAIN][WARN] Could not render map snapshot: {e}")
            except Exception as e:
                print(f"[MAIN][WARN] world_map dump failed: {e}")

        print("[MAIN] Stopping subsystems…")
        try:
            actions.release_all_movement()
        except Exception:
            pass
        # Announce the stop in chat BEFORE we tear subsystems down, so
        # the user reads "stopped" in the game window. Guarded so we
        # don't crash here on shutdown failures.
        if chat_announced and safety.allow_input():
            try:
                _send_chat_message(keyboard, "[bot] stopped — safe to tab")
                time.sleep(0.2)
            except Exception:
                pass
        for sub in (keyboard, mouse, capture):
            try:
                sub.stop()
            except Exception:
                pass
        safety.stop()
        print("[MAIN] Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
