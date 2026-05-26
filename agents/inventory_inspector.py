# agents/inventory_inspector.py
"""
Inventory inspector — Phase 2 of the inventory pipeline.

After the vision-only recognizer in ``vision/inventory.py`` returns its
snapshot, many slots are marked ``unknown`` because Phase-1 confidence
gating intentionally refuses borderline matches. This module resolves
those unknowns by simulating what a human player would do: move the
mouse over the slot, wait briefly for the tooltip to appear, capture
a frame, and OCR the tooltip's ``minecraft:<id>`` line (which is
visible because the user has Advanced Tooltips enabled in MC's Options).

What this isn't
---------------
* This is NOT a recognizer — it doesn't replace ``ItemRecognizer``.
  It's a backstop for low-confidence reads.
* It's NOT free — every hover takes ~150-300 ms of waiting plus a
  capture, so resolving 20 unknowns adds ~5 seconds. That's deliberate;
  Phase-3 (self-training a recognizer on labelled hover screenshots)
  is how we eliminate the hover cost over time.

Design choices
--------------
* **Absolute cursor positioning via win32.** Our ``control.mouse``
  module exposes RELATIVE moves only (dx, dy) because in-game camera
  control uses relative deltas. For hovering over inventory slots we
  need a fixed screen pixel — we use ``SetCursorPos`` directly. MC's
  inventory cursor reads OS cursor position so this works without any
  detection.
* **Cursor restore.** After all hovers complete we put the cursor back
  where the user left it, so they can keep using their mouse.
* **Respect the input gate.** If the safety gate is closed we skip the
  hover entirely — the agent should never move the cursor against the
  user's wishes.
"""

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from vision.inventory import InventorySnapshot, SlotContent
from vision.inventory_layout import SlotRect, slot_rects
from vision.sample_store import SampleStore
from vision.tooltip import TooltipReader, TooltipInfo


# ---------------------------------------------------------------------------
# Absolute cursor helpers — Windows only (the project runs on Windows).
# ---------------------------------------------------------------------------

_user32 = ctypes.windll.user32


def get_cursor_xy() -> Tuple[int, int]:
    """Return current cursor position in screen coordinates."""
    class _POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
    pt = _POINT()
    _user32.GetCursorPos(ctypes.byref(pt))
    return int(pt.x), int(pt.y)


def set_cursor_xy(x: int, y: int) -> None:
    """Teleport the cursor to (x, y) in screen coordinates."""
    _user32.SetCursorPos(int(x), int(y))


# ---------------------------------------------------------------------------
# Inspector
# ---------------------------------------------------------------------------

@dataclass
class InspectorConfig:
    # How long to wait after moving the cursor before grabbing the
    # capture frame. MC's tooltip appears with a small delay (varies by
    # GUI animation settings). 200 ms is safe in practice.
    hover_settle_ms: int = 200

    # If the OCR'd tooltip can't be parsed, retry this many times with
    # a slightly longer settle. Useful when the very first frame after
    # cursor-move catches a half-drawn tooltip.
    max_retries: int = 2

    # Maximum number of slots we'll resolve per call to
    # ``resolve_unknowns``. Acts as a safety cap so a buggy caller
    # can't lock the cursor for minutes.
    max_resolutions_per_call: int = 64

    # Skip slots whose Phase-1 confidence is at or above this — they're
    # already trusted. Only "unknown" slots and ones below this are
    # hovered.
    skip_above_confidence: float = 0.50


@dataclass
class InspectionResult:
    """One slot's hover outcome."""
    slot_name: str
    item_id:   Optional[str]
    display_name: Optional[str]
    durability: Optional[Tuple[int, int]]
    tooltip_raw_lines: List[str]


class InventoryInspector:
    """
    Hover over inventory slots to read their tooltips.

    Construct once per session; call ``resolve_unknowns(snap, ...)`` to
    fill in the unknowns of a snapshot.
    """

    def __init__(self,
                 tooltip_reader: TooltipReader,
                 capture,
                 *,
                 gate=None,
                 sample_store: Optional[SampleStore] = None,
                 config: Optional[InspectorConfig] = None):
        self._tooltip = tooltip_reader
        self._capture = capture
        self._gate    = gate
        self.cfg      = config or InspectorConfig()
        # If provided, every successful hover writes the slot crop
        # (BEFORE the cursor moved over it — taken from the original
        # snapshot's frame, passed in via resolve_unknowns) plus the
        # OCR'd item id into the sample store. That dataset is what
        # SampleRecognizer matches against on subsequent runs, so the
        # hover cost amortises over time.
        self._sample_store = sample_store

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def resolve_slot(self,
                     slot: SlotRect,
                     *,
                     window_origin: Tuple[int, int] = (0, 0),
                     ) -> Optional[InspectionResult]:
        """
        Hover over a single slot, wait for the tooltip, OCR it, return
        the result. ``window_origin`` is the (x, y) of the captured
        window's top-left corner on the desktop — slot rects are in
        FRAME pixel coords, but the cursor uses DESKTOP pixel coords,
        so we offset.

        Returns ``None`` if the gate is closed or no tooltip appeared.
        """
        if self._gate is not None and not self._gate.allow():
            return None

        cx, cy = slot.center()
        desktop_x = window_origin[0] + cx
        desktop_y = window_origin[1] + cy
        set_cursor_xy(desktop_x, desktop_y)

        # First attempt: wait the configured settle time. Subsequent
        # attempts add a small linear backoff (capped) — the tooltip
        # sometimes needs a frame or two more than expected, but
        # multiplying the full settle each time is wasteful and pushes
        # large-inventory inspections into multi-second territory.
        base_s = self.cfg.hover_settle_ms / 1000.0
        retry_bonus_s = max(0.05, base_s / 4.0)
        for attempt in range(1, self.cfg.max_retries + 2):
            wait_s = base_s + retry_bonus_s * (attempt - 1)
            time.sleep(wait_s)
            try:
                frame = self._capture.get_frame()
            except Exception:
                continue
            info = self._tooltip.read(frame, near_xy=(cx, cy))
            if info is not None and info.item_id is not None:
                return InspectionResult(
                    slot_name=slot.name,
                    item_id=info.item_id,
                    display_name=info.display_name,
                    durability=info.durability,
                    tooltip_raw_lines=[l.text for l in info.raw_lines],
                )
        return None

    def resolve_unknowns(self,
                         snap: InventorySnapshot,
                         *,
                         window_origin: Tuple[int, int] = (0, 0),
                         container: Optional[str] = None,
                         restore_cursor: bool = True,
                         pre_hover_frame: Optional[np.ndarray] = None,
                         ) -> Dict[str, InspectionResult]:
        """
        Hover over every slot the recogniser is unsure about and update
        ``snap`` in-place with the tooltip findings.

        Parameters
        ----------
        snap          : InventorySnapshot from the recogniser.
        window_origin : Desktop (x, y) of the captured window's top-left
                        corner. Use ``utils.focus._find_minecraft_hwnd``
                        and ``win32gui.GetWindowRect`` (or capture's
                        client rect) to get it.
        container     : Layout key — if ``None``, taken from
                        ``snap.container``.
        restore_cursor: If True (default) we put the cursor back where
                        we found it after all hovers complete. Set
                        False when chaining further automated input.
        pre_hover_frame : The FRAME used to build ``snap`` (before any
                        hovers happened). Required if we have a
                        sample_store wired up — sample crops are taken
                        from this frame, not from the post-hover one,
                        because the cursor sprite would otherwise
                        overlay the slot and contaminate the sample.
        """
        layout = container or snap.container
        if snap.frame_shape is None:
            return {}

        rects = slot_rects(snap.frame_shape, layout=layout,
                           ui_scale=snap.ui_scale)
        original_cursor = get_cursor_xy() if restore_cursor else None

        results: Dict[str, InspectionResult] = {}
        resolved = 0
        try:
            for name, slot in rects.items():
                if resolved >= self.cfg.max_resolutions_per_call:
                    break
                content = snap.slots.get(name)
                # Skip truly empty slots and slots already identified
                # confidently.
                if content is None:
                    continue
                if content.is_empty:
                    continue
                if (content.item is not None
                        and content.confidence >= self.cfg.skip_above_confidence):
                    continue

                result = self.resolve_slot(slot, window_origin=window_origin)
                if result is None:
                    continue
                results[name] = result
                # Update the snapshot in-place with ground truth.
                snap.slots[name] = SlotContent(
                    item=result.item_id,
                    count=content.count,                 # keep Phase-1 count
                    durability=(result.durability[0] / result.durability[1]
                                if result.durability else content.durability),
                    enchanted=content.enchanted,
                    confidence=1.0,                       # OCR'd id is truth
                    score=0.0,
                    second=content.item,                  # what vision had guessed
                    source="hover",
                )
                # Phase 3: record the pre-hover slot pixels with their
                # OCR'd label so future runs can skip this hover.
                if (self._sample_store is not None
                        and pre_hover_frame is not None
                        and result.item_id):
                    self._save_sample_from_frame(
                        pre_hover_frame, slot, result.item_id)
                resolved += 1
        finally:
            if original_cursor is not None:
                set_cursor_xy(*original_cursor)

        return results


    # ------------------------------------------------------------------
    # Phase-3 sample collection
    # ------------------------------------------------------------------

    def _save_sample_from_frame(self,
                                frame_rgb: np.ndarray,
                                slot: SlotRect,
                                item_id: str) -> None:
        """
        Crop the slot pixels out of the PRE-hover frame and hand them
        to the sample store. Swallows any error — sample collection is
        best-effort and must never break the recogniser.
        """
        try:
            x0, y0 = slot.x, slot.y
            x1 = min(frame_rgb.shape[1], x0 + slot.w)
            y1 = min(frame_rgb.shape[0], y0 + slot.h)
            crop = frame_rgb[y0:y1, x0:x1]
            self._sample_store.save(item_id, crop)
        except Exception:
            pass


__all__ = [
    "InventoryInspector", "InspectorConfig", "InspectionResult",
    "get_cursor_xy", "set_cursor_xy",
]
