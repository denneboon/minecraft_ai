from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, Optional

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
from vision.capture import Capture, CaptureConfig
from vision.processing import build_processor, FrameProcessor, ScreenState
from vision.ocr import build_f3_reader, F3Reader
from utils.focus import activate_minecraft, _find_minecraft_hwnd

from vision.menu_detect import MenuDetector, MenuDetection, build_menu_detector
from vision.mcfont import ensure_font_cache

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
    setattr(sc, "assume_focused_when_unknown",
            bool(_get(settings, "safety.assume_focused_when_unknown", False)))
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
    mc = MouseConfig(
        move_duration_ms=int(_get(settings, "control.mouse.move_duration_ms", 40)),
        flick_multiplier=float(_get(settings, "control.mouse.flick_multiplier", 0.35)),
        curve_steps=int(_get(settings, "control.mouse.curve_steps", 10)),
        default_click_duration=float(_get(settings, "control.mouse.default_click_duration", 0.05)),
        max_events_per_sec=int(_get(settings, "control.mouse.max_events_per_sec", 240)),
        enable_scroll=bool(_get(settings, "control.mouse.enable_scroll", True)),
    )
    return Mouse(config=mc, gate=gate)


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

    # Camera (smooth easing inside Mouse.track_target)
    if action.look_dx or action.look_dy:
        mouse.track_target(int(action.look_dx), int(action.look_dy))

    # One-shot interactions
    if action.interact == "attack":
        actions.execute("attack")
    elif action.interact == "use_item":
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
            now = time.perf_counter()
            if (now - last_f3_ts >= f3_interval
                    and state.screen_state == ScreenState.PLAYING):
                try:
                    fresh = f3_reader.read(frame)
                    if fresh is not None and fresh.x is not None:
                        last_f3 = fresh
                    state.f3 = fresh
                except Exception as e:
                    print(f"[AGENT][WARN] F3 OCR failed: {e}")
                last_f3_ts = time.perf_counter()
            if state.f3 is None:
                state.f3 = last_f3

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
                # Snapshot one line per tick: gate / screen / movement / look / interact
                mv = decision.movement
                print(f"[TICK]  gate={gate_now}  screen={state.screen_state}  "
                      f"dispatch={dispatch_ok}  movement={mv}  "
                      f"look=({decision.look_dx},{decision.look_dy})  "
                      f"interact={decision.interact}  hotbar={decision.hotbar}")
            if dispatch_ok:
                try:
                    _dispatch_action(decision, actions, mouse, keyboard)
                except Exception as e:
                    print(f"[AGENT][WARN] dispatch failed: {e}")
            else:
                # Not allowed to act — be sure no movement keys are stuck.
                if keyboard.cfg.verbose:
                    print(f"[DISPATCH] BLOCKED — releasing all movement "
                          f"(gate={gate_now}, screen={state.screen_state})")
                try:
                    actions.release_all_movement()
                except Exception:
                    pass

            # --- Tick pacing + rate log ---
            now = time.perf_counter()
            elapsed = now - last_tick
            if elapsed < min_dt:
                time.sleep(min_dt - elapsed)
            last_tick = time.perf_counter()

            profile_ticks += 1
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
        # profile_ticks holds only the in-window count; reconstruct
        # total via tick budget rather than maintaining a second counter.
        approx_total_ticks = int(wall * tick_rate)  # target ticks
        print(f"[AGENT] Loop ended after {wall:.1f}s "
              f"(target {tick_rate:.0f} Hz, ~{approx_total_ticks} ticks).")


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
    if args.test:
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
    actions   = ActionWrapper(keyboard=keyboard, mouse=mouse, gate=gate)

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
        except Exception as e:
            print(f"[MAIN][WARN] build_agent({agent_name!r}) failed: {e}. "
                  f"Falling back to smoke test.")
            agent = None

    def emergency():
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
    if menu_detector is not None:
        _maybe_focus_and_unpause(settings, safety, keyboard, capture, menu_detector)

    try:
        frame = capture.get_frame()
        h, w = frame.shape[:2]
        print(f"[MAIN] Capture OK: {w}x{h} px")
        safety.notify_action(f"capture_frame({w}x{h})")

        # Refuse to run the agent if the window isn't at calibration size.
        # Off by a pixel or two is fine; off by hundreds means HUD reads
        # and F3 OCR will all target gameplay-area pixels.
        calib_res = (
            (_get(settings, "vision_calibration.last.resolution") or [None, None])
        )
        if (agent is not None and calib_res and len(calib_res) == 2
                and calib_res[0] and calib_res[1]):
            cw, ch = int(calib_res[0]), int(calib_res[1])
            if abs(w - cw) > 8 or abs(h - ch) > 8:
                print(f"[MAIN][ERROR] Capture size {w}x{h} differs from "
                      f"calibration {cw}x{ch} by more than 8 px. "
                      f"Maximize the Minecraft window or re-run calibration.")
                agent = None  # Fall back to smoke test instead of moving blind.
    except Exception as e:
        print(f"[MAIN][WARN] Could not capture a frame: {e}")

    # Always clean up, even on Ctrl+C or exceptions.
    chat_announced = False
    try:
        if agent is None:
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
            _run_agent_loop(
                capture=capture, processor=processor, f3_reader=f3_reader,
                actions=actions, mouse=mouse, keyboard=keyboard,
                safety=safety, agent=agent, settings=settings,
                max_runtime_sec=max_runtime,
            )
    finally:
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
