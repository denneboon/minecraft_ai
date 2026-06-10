from __future__ import annotations
import sys, os, time, argparse
from typing import Any, Dict, Optional




VISION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(VISION_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import numpy as np
import cv2
try:
    import yaml
except Exception:
    yaml = None


from vision.capture import Capture, CaptureConfig
from vision.hud import HUDReader, HUDReaderConfig
from utils.focus import _find_minecraft_hwnd
from control.keyboard import Keyboard, KeyboardConfig





SETTINGS_PATH = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
CALIB_DIR = os.path.join(PROJECT_ROOT, "data", "calibration")
os.makedirs(CALIB_DIR, exist_ok=True)





def _load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}

def _save_yaml(path: str, data: Dict[str, Any]) -> None:
    if yaml is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def _deep_get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur





def _hud_layout_by_formula(width: int, height: int, ui_scale: int) -> Dict[str, list]:
    assert ui_scale >= 1, "UI scale must be >= 1"

    sW = width // ui_scale
    sH = height // ui_scale

    HOTBAR_W, HOTBAR_H = 182, 22
    XP_W, XP_H = 182, 5

    ICON_W, ICON_H = 8, 9
    ICONS = 10

    BAR_W=1+ICON_W*ICONS

    Y_OFF_HEALTH_HUNGER = 39
    Y_OFF_ARMOR         = 49
    Y_OFF_XP            = 29

    SLOT_W = 20
    # SLOT_H == SLOT_W for vanilla hotbar slots; only width is read
    # below so we don't declare the height constant. Add it back if
    # a non-square hotbar texture (resource pack) ever needs it.
    NUM_SLOTS = 9

    centerX = sW // 2

    health = [
        centerX - BAR_W - 10,
        sH - Y_OFF_HEALTH_HUNGER,
        BAR_W,
        ICON_H,
    ]

    armor = [
        centerX - BAR_W - 10,
        sH - Y_OFF_ARMOR,
        BAR_W,
        ICON_H,
    ]

    hunger = [
        centerX + 10,
        sH - Y_OFF_HEALTH_HUNGER,
        BAR_W,
        ICON_H,
    ]

    xp_bar = [
        sW // 2 - XP_W // 2,
        sH - Y_OFF_XP,
        XP_W,
        XP_H,
    ]

    hotbar = [
        centerX - HOTBAR_W // 2,
        sH - HOTBAR_H,
        HOTBAR_W,
        HOTBAR_H,
    ]

    crosshair = [sW // 2, sH // 2]

    # Hotbar slots: the texture has a 1px border on each side, then 9 slots
    # of 20 GUI-px each packed with no gap (the slot cell includes its own border).
    # Total: 1 + 9*20 + 1 = 182. Correct.
    BORDER = 1
    slots_left_x = hotbar[0] + BORDER
    slots_y = hotbar[1] + BORDER
    slot_h_inner = HOTBAR_H - 2 * BORDER   # 20 GUI-px tall

    slot_rects_scaled = []
    slot_centers_scaled = []
    x = slots_left_x
    for _ in range(NUM_SLOTS):
        slot_rects_scaled.append([x, slots_y, SLOT_W, slot_h_inner])
        slot_centers_scaled.append([x + SLOT_W // 2, slots_y + slot_h_inner // 2])
        x += SLOT_W  # no gap — slots are packed edge-to-edge

    # 6) Scale-up helper (floor via int())
    def up_rect(r):
        return [int(r[0] * ui_scale), int(r[1] * ui_scale), int(r[2] * ui_scale), int(r[3] * ui_scale)]

    def up_pt(p):
        return [int(p[0] * ui_scale), int(p[1] * ui_scale)]

    # 7) Return all in pixel coordinates (unscaled)
    return {
        "hotbar_region":      up_rect(hotbar),
        "xp_bar":             up_rect(xp_bar),
        "health_bar":         up_rect(health),
        "hunger_bar":         up_rect(hunger),
        "armor_bar":          up_rect(armor),
        "crosshair_center":   up_pt(crosshair),
        "hotbar_slot_rects":  [up_rect(r) for r in slot_rects_scaled],
        "hotbar_slot_centers":[up_pt(c) for c in slot_centers_scaled],
        "meta": {
            "ui_scale": ui_scale,
            "scaled_size": [sW, sH],
            "window_size": [width, height],
        }
    }





def _annotate_preview(bgr_img: np.ndarray, regions: Dict[str, Any], ui_scale: int = 1) -> np.ndarray:
    out = bgr_img.copy()
    yellow = (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    H, W = out.shape[:2]

    def draw_outline_around_rect(inner_rect, label=None):
        x, y, w, h = map(int, inner_rect)

        left   = x - 1
        top    = y - 1
        right  = x + w
        bottom = y + h

        # Clamp drawing endpoints; we allow -1 for left/top so the border hugs edges visually
        left_c   = max(-1, left)
        top_c    = max(-1, top)
        right_c  = min(W, right)
        bottom_c = min(H, bottom)

        cv2.rectangle(out, (left_c, top_c), (right_c, bottom_c), yellow, 1)

        if label:
            tx = max(0, left)   # put label just above left-top (clamped inside image)
            ty = max(0, top - 6)
            cv2.putText(out, label, (tx, ty), font, 0.5, yellow, 1, cv2.LINE_AA)

    def draw_crosshair_outline(center_xy, scale):
        cx, cy = map(int, center_xy)
        s = max(1, int(scale))

        inner_w = 9 * s
        inner_h = 9 * s

        # Center the inner box on (cx, cy)
        left = cx - (inner_w // 2)
        top  = cy - (inner_h // 2)

        draw_outline_around_rect([left-1, top+1, inner_w, inner_h], label="crosshair")

    # --- Iterate over regions ---
    for k, v in regions.items():
        if k == "meta":
            continue

        if k == "crosshair_center":
            draw_crosshair_outline(v, ui_scale)
            continue

        # Skip lists of points (e.g., hotbar_slot_centers)
        if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (list, tuple)) and len(v[0]) == 2:
            continue

        # List of rectangles (e.g., hotbar_slot_rects)
        if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (list, tuple)) and len(v[0]) == 4:
            for rect in v:
                draw_outline_around_rect(rect)
            draw_outline_around_rect(v[0], label=k)
            continue

        # Single rectangle
        if isinstance(v, (list, tuple)) and len(v) == 4:
            draw_outline_around_rect(v, label=k)
            continue

    return out

def calibrate(ui_scale: Optional[int] = None,
              window_title: str = "Minecraft",
              fps_limit: int = 30,
              save: bool = True,
              preview: bool = True) -> Dict[str, Any]:
    print(f"[CALIB] Settings path: {SETTINGS_PATH}")
    settings = _load_yaml(SETTINGS_PATH)


    # --- Correct UI scale loading ---
    ui_settings = _deep_get(settings, "capture.ui_scale", None)

    if ui_scale is not None:
        ui = int(ui_scale)              # CLI flag has priority
    elif ui_settings is not None:
        ui = int(ui_settings)           # YAML value
    else:
        ui = 2                          # last-resort default

    print(f"[CALIB] Using UI scale: {ui}")

    # Find the window first — needed for both maximize and capture.
    wins = _find_minecraft_hwnd()
    if not wins:
        raise RuntimeError("Minecraft window not found (javaw.exe)")
    hwnd, _ = wins[0]

    # Maximize and focus the window so the capture resolution is always
    # windowed-maximized (not whatever size it happened to be before).
    try:
        import win32gui, win32con, win32process, win32api
        import ctypes
        user32 = ctypes.windll.user32

        win32gui.ShowWindow(hwnd, win32con.SW_SHOWMAXIMIZED)

        fg = win32gui.GetForegroundWindow()
        fg_tid, _ = win32process.GetWindowThreadProcessId(fg) if fg else (0, 0)
        target_tid, _ = win32process.GetWindowThreadProcessId(hwnd)
        cur_tid = win32api.GetCurrentThreadId()
        user32.AttachThreadInput(cur_tid, fg_tid, True)
        user32.AttachThreadInput(cur_tid, target_tid, True)
        win32gui.BringWindowToTop(hwnd)
        win32gui.SetForegroundWindow(hwnd)
        win32gui.SetActiveWindow(hwnd)
        win32gui.SetFocus(hwnd)
        user32.AttachThreadInput(cur_tid, fg_tid, False)
        user32.AttachThreadInput(cur_tid, target_tid, False)
    except Exception as e:
        print(f"[CALIB] Warning: could not maximize/focus window: {e}")

    # Wait for the maximize animation to fully settle before capturing.
    time.sleep(0.4)

    cap = Capture(CaptureConfig(
        hwnd=hwnd,
        window_title_query=window_title,
        max_fps=fps_limit,
        track_window_each_frame=True,
        strict_window_find=True,
        use_client_area=True,
        clamp_to_monitor=True,
        name="minecraft_calibration_formula"
    ))
    cap.start()

    # Grab a probe frame first to check whether the pause/inventory screen is open.
    # Only send Escape if it actually looks like a menu — never fire blindly.
    probe = cap.get_frame()
    if probe is not None and probe.size > 0:
        ph, pw = probe.shape[:2]
        # Sample the bottom-centre strip where the hotbar lives.
        hb_y = int(ph * 0.90)
        hb_roi = probe[hb_y:, pw // 4: 3 * pw // 4]
        is_menu = float(hb_roi.std()) < 6.0
        if is_menu:
            print("[CALIB] Pause/menu detected — sending Escape to dismiss.")
            try:
                kb = Keyboard(KeyboardConfig())
                kb.start()
                try:
                    kb.tap("escape", 0.05)
                    time.sleep(0.18)
                finally:
                    kb.stop()
            except Exception as e:
                print(f"[CALIB] Warning: could not send Escape: {e}")
        else:
            print("[CALIB] Game is active — skipping Escape tap.")

    frame_rgb = cap.get_frame()
    cap.stop()

    if frame_rgb is None or frame_rgb.size == 0:
        raise RuntimeError("Capture failed (empty frame)")

    H0, W0 = frame_rgb.shape[:2]

    # Compute the extra pixels Minecraft would have dropped
    extra_w = W0 % ui
    extra_h = H0 % ui

    # Trim only the remainder from the right/bottom (keeps UI scale 2 identical when divisible)
    if extra_w or extra_h:
        Hc = H0 - (extra_h if extra_h else 0)
        Wc = W0 - (extra_w if extra_w else 0)
        frame_rgb = frame_rgb[:Hc, :Wc]
        # Optional debug:
        print(f"[CALIB] Cropped to match GUI scaler: {W0}x{H0} -> {Wc}x{Hc} (ui_scale={ui})")


    H, W = frame_rgb.shape[:2]
    print(f"[CALIB] Captured frame: {W}x{H}")


    regions = _hud_layout_by_formula(W, H, ui)


    if preview:
        out_path = os.path.join(CALIB_DIR, f"hud_preview_{W}x{H}.png")
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        overlay = _annotate_preview(bgr, regions, ui_scale=ui)
        cv2.imwrite(out_path, overlay)
        print(f"[CALIB] Wrote preview: {out_path}")


    # --- SAVE TO YAML ---
    if save and yaml is not None:
        settings = _load_yaml(SETTINGS_PATH)

        vision = settings.setdefault("vision", {})
        hud = vision.setdefault("hud_regions", {})

        # Only persist the classic/flat keys here to preserve your existing schema:
        BASIC_KEYS = {
            "hotbar_region",
            "xp_bar",
            "health_bar",
            "hunger_bar",
            "armor_bar",
            "crosshair_center",
        }

        def _is_point(v):
            return isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)

        def _is_rect(v):
            return isinstance(v, (list, tuple)) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v)

        # 1) Save classic HUD into vision.hud_regions (flat, backward-compatible)
        for k, v in regions.items():
            if k == "meta":
                continue
            if k in BASIC_KEYS:
                if _is_point(v):
                    hud[k] = [int(v[0]), int(v[1])]
                elif _is_rect(v):
                    hud[k] = [int(v[0]), int(v[1]), int(v[2]), int(v[3])]
                else:
                    # Skip anything that isn't a flat rect/point
                    continue

        # 2) Save nested/extended shapes under vision.hud_extended
        extended = vision.setdefault("hud_extended", {})
        for k, v in regions.items():
            if k in BASIC_KEYS or k == "meta":
                continue

            # Allow lists of rects/points; convert numerics to int
            if isinstance(v, (list, tuple)):
                if len(v) > 0 and isinstance(v[0], (list, tuple)):
                    # list of rects or points
                    converted = []
                    for item in v:
                        # item can be point [x,y] or rect [x,y,w,h]
                        converted.append([int(x) for x in item])
                    extended[k] = converted
                elif _is_point(v) or _is_rect(v):
                    extended[k] = [int(x) for x in v]
                else:
                    # scalar or unknown shape
                    try:
                        extended[k] = int(v)  # if it's scalar-like
                    except Exception:
                        # skip unknowns silently
                        pass
            elif isinstance(v, dict):
                # If any dict sneaks in (shouldn't normally), store as-is after int-casting where possible
                safe_dict = {}
                for kk, vv in v.items():
                    if isinstance(vv, (list, tuple)):
                        safe_dict[kk] = [int(x) for x in vv]
                    else:
                        try:
                            safe_dict[kk] = int(vv)
                        except Exception:
                            safe_dict[kk] = vv
                extended[k] = safe_dict
            else:
                try:
                    extended[k] = int(v)
                except Exception:
                    pass

        vc = settings.setdefault("vision_calibration", {})
        vc["last"] = {
            "resolution": [W, H],
            "ui_scale": ui,
            "timestamp": int(time.time()),
            "source": "vision.calibration_formula",
        }

        _save_yaml(SETTINGS_PATH, settings)
        print(f"[CALIB] Updated HUD regions → {SETTINGS_PATH}")

    # ─── Self-verification ─────────────────────────────────────────────
    # Read the just-calibrated HUD positions against the captured frame
    # and print a sanity report. This catches the silent-failure mode
    # where calibration "succeeded" but produced rectangles that fall on
    # gameplay/sky pixels (e.g. because the window resized between Focus
    # and Capture).
    _verify_calibration(frame_rgb, regions, ui)

    return regions


def _verify_calibration(frame_rgb: np.ndarray,
                        regions: Dict[str, Any],
                        ui_scale: int) -> None:
    """
    Read each calibrated HUD region with the production HUDReader and
    print a per-bar status line. Each bar gets one of:
        OK      — readings look plausible
        EMPTY   — region is in the right place but no icons visible
                  (player has zero of that resource — not necessarily a bug)
        WEAK    — readings are very low but non-zero; region may be off
                  by a few pixels.
        MISSING — region rectangle is invalid / empty.
    """
    # Build a HUDReader from just the freshly-calibrated regions.
    flat: Dict[str, Any] = {}
    for k in ("health_bar", "hunger_bar", "armor_bar", "xp_bar", "hotbar_region"):
        if k in regions:
            flat[k] = regions[k]

    reader = HUDReader(
        hud_regions=flat,
        config=HUDReaderConfig(ui_scale=int(ui_scale)),
    )
    snap = reader.read(frame_rgb)

    print("[CALIB] Self-check (against current frame):")
    bars = [
        ("health", snap.health, snap.diagnostics.get("health", {})),
        ("hunger", snap.hunger, snap.diagnostics.get("hunger", {})),
        ("armor",  snap.armor,  snap.diagnostics.get("armor",  {})),
        ("xp_bar", snap.xp_bar, snap.diagnostics.get("xp_bar", {})),
    ]
    any_warn = False
    for name, value, diag in bars:
        ratio = float(diag.get("mask_overall_ratio", 0.0))
        if diag.get("reason") == "missing_roi":
            tag = "MISSING"
            any_warn = True
        elif ratio < 0.005:
            tag = "EMPTY  "
        elif ratio < 0.03 and value < 0.1:
            tag = "WEAK   "
            any_warn = True
        else:
            tag = "OK     "
        print(f"        {tag} {name:<7} value={value:.2f}  "
              f"mask_ratio={ratio:.3f}")
    if any_warn:
        print("[CALIB] One or more bars look off — verify the preview PNG and")
        print("        consider re-running calibration with Minecraft in-game.")



def build_cli():
    p = argparse.ArgumentParser(description="Minecraft HUD calibration (formula-based, pixel-perfect).")
    p.add_argument("--ui-scale", type=int, default=None, help="Override UI scale (default: read from config.capture.ui_scale)")
    p.add_argument("--title", type=str, default="Minecraft", help="Window title hint (unused if hwnd is found)")
    p.add_argument("--fps", type=int, default=30, help="Capture fps cap (just for the calibration grab)")
    p.add_argument("--save", action="store_true", help="Persist results to config/settings.yaml")
    p.add_argument("--no-save", dest="save", action="store_false")
    p.add_argument("--no-preview", dest="preview", action="store_false", help="Skip writing the annotated PNG")
    p.set_defaults(save=True, preview=True)
    return p.parse_args()


def main():
    a = build_cli()
    calibrate(ui_scale=a.ui_scale, window_title=a.title, fps_limit=a.fps, save=a.save, preview=a.preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
