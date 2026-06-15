"""
Shared bootstrap for the small inventory tools (arrange_hotbar / equip_armor).

Each of those tools needs the same thing: focus Minecraft, stand up the
capture + input + inventory-reader + controller stack, confirm the game is
controllable, do ONE inventory operation, then tear every thread back down.
That ~55-line bring-up/teardown was copy-pasted per tool and would silently
drift the moment the controller wiring changed. This context manager owns it
so each tool is just its actual logic.

    from tools.inventory_session import inventory_session

    with inventory_session("armor") as sess:
        if sess is None:
            return 1                       # not found / not controllable (banner shown)
        equip_best_armor(sess.ctl, catalog=sess.cat, log=print)
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import main as M
from utils.focus import _find_minecraft_hwnd, activate_minecraft
from control.input_gate import InputGate
from vision.capture import Capture, CaptureConfig
from vision.inventory import build_inventory_reader
from vision.tooltip import build_tooltip_reader
from agents.inventory_inspector import InventoryInspector, InspectorConfig
from control.hotbar import build_hotbar_manager
from control.inventory_control import InventoryController
from agents.inventory_memory import InventoryMemory
from knowledge.catalog import Catalog
from vision.mc_assets import MCAssets


@dataclass
class InventorySession:
    """The handles a tool needs once Minecraft is focused + controllable."""
    ctl: InventoryController
    cat: Catalog
    settings: dict


@contextmanager
def inventory_session(tag: str = "inv") -> Iterator[Optional[InventorySession]]:
    """Focus MC and yield an :class:`InventorySession`, or ``None`` if MC isn't
    found or isn't controllable (the appropriate banner is printed first). All
    started threads (safety/mouse/kb/capture) are torn down on exit."""
    wins = _find_minecraft_hwnd()
    if not wins:
        print(f"[{tag}] Minecraft not found")
        yield None
        return
    hwnd = wins[0][0]
    activate_minecraft(); time.sleep(0.5)

    settings = M._load_yaml(M.SETTINGS_PATH)
    keymap = M._flatten_keymap_for_keyboard(M._load_json(M.KEYMAP_PATH))
    ui_scale = int((settings.get("capture") or {}).get("ui_scale", 2))

    gate = InputGate()
    safety = M.build_safety(settings, gate=gate)
    mouse = M.build_mouse(settings, gate=gate)
    kb = M.build_keyboard(settings, keymap, gate=gate)
    capture = Capture(CaptureConfig(hwnd=hwnd, use_client_area=True, threaded=True))
    safety.start(); mouse.start(); kb.start(); capture.start(); time.sleep(0.3)

    menu_detector = M.build_menu_detector_default(settings)
    a = MCAssets.load(); cat = Catalog.load(a)
    reader = build_inventory_reader(settings, assets=a)
    hotbar = build_hotbar_manager(settings, catalog=cat)
    tooltip = build_tooltip_reader(settings, assets=a)
    try:
        origin = capture.window_origin()
    except Exception:
        origin = (0, 0)
    inspector = InventoryInspector(
        tooltip, capture, mouse=mouse, gate=gate,
        sample_store=getattr(reader, "sample_store", None),
        config=InspectorConfig(max_resolutions_per_call=24))
    memory = InventoryMemory()
    ctl = InventoryController(mouse, kb, reader, hotbar, capture,
                              ui_scale=ui_scale, window_origin=origin,
                              inspector=inspector, memory=memory)
    try:
        ctrl_ok, reason = M.ensure_controllable(capture, menu_detector, kb, gate)
        if not ctrl_ok:
            M.bot_cannot_start_banner(reason)
            yield None
        else:
            yield InventorySession(ctl=ctl, cat=cat, settings=settings)
    finally:
        try:
            if hasattr(mouse, "release_all"):
                mouse.release_all()
        except Exception:
            pass
        try:
            kb.stop()
        except Exception:
            pass
        capture.stop()
        try:
            safety.stop()
        except Exception:
            pass
