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
* **Absolute cursor positioning.** Our ``control.mouse`` module exposes
  RELATIVE moves for the gameplay camera (where MC reads Raw Input with
  the cursor locked) AND eased ABSOLUTE moves via
  :meth:`Mouse.move_to_screen_xy` for menus (where MC reads the OS
  cursor). This inspector always uses the eased absolute path so a
  hover sequence looks like a human reaching for the slot — minimum-
  jerk velocity profile, slight perpendicular bow, per-step wobble.
  The bare-teleport helpers below remain available for callers that
  genuinely want an instant jump (e.g. unit tests).
* **Cursor restore.** After all hovers complete we ease the cursor
  back where the user left it, so they can keep using their mouse.
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
from vision.sample_store import SampleStore, EMPTY_SAMPLE_LABEL
from vision.tooltip import TooltipReader


# ---------------------------------------------------------------------------
# Absolute cursor helpers — Windows only (the project runs on Windows).
#
# These remain as MODULE-LEVEL helpers for callers that want an instant
# teleport (e.g. test fixtures). The inspector itself uses
# ``Mouse.move_to_screen_xy`` for an eased, humanlike reach instead.
# ---------------------------------------------------------------------------

_user32 = ctypes.windll.user32


def _mark_slot_empty(snap: "InventorySnapshot", name: str) -> None:
    """Force a slot in ``snap`` to read as truly EMPTY. ``SlotContent.is_empty``
    is a read-only property (``item is None and score == +inf``), so we can't
    assign it — we set the underlying fields instead."""
    sc = snap.slots.get(name) if getattr(snap, "slots", None) else None
    if sc is not None:
        sc.item = None
        sc.score = float("inf")      # with item=None -> is_empty property True
        sc.confidence = 1.0
        sc.source = "hover-empty"


def get_cursor_xy() -> Tuple[int, int]:
    """Return current cursor position in screen coordinates."""
    class _POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
    pt = _POINT()
    _user32.GetCursorPos(ctypes.byref(pt))
    return int(pt.x), int(pt.y)


def set_cursor_xy(x: int, y: int) -> None:
    """Teleport the cursor to (x, y) in screen coordinates. Prefer
    :meth:`Mouse.move_to_screen_xy` from the runtime — this is a
    no-easing primitive kept for tests and one-off scripts."""
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
                 mouse=None,
                 gate=None,
                 sample_store: Optional[SampleStore] = None,
                 config: Optional[InspectorConfig] = None):
        self._tooltip = tooltip_reader
        self._capture = capture
        # ``mouse`` is the runtime :class:`control.mouse.Mouse`. When
        # provided, hovers and cursor-restores use its eased absolute
        # path (humanlike minimum-jerk reach). When omitted (tests,
        # standalone scripts) we fall back to the bare teleport so
        # nothing breaks — but agents in the main loop should always
        # pass it.
        self._mouse   = mouse
        self._gate    = gate
        self.cfg      = config or InspectorConfig()
        # If provided, every successful hover writes the slot crop
        # (BEFORE the cursor moved over it — taken from the original
        # snapshot's frame, passed in via resolve_unknowns) plus the
        # OCR'd item id into the sample store. That dataset is what
        # SampleRecognizer matches against on subsequent runs, so the
        # hover cost amortises over time.
        self._sample_store = sample_store
        # Slot NAMES confirmed empty by a hover this session (the armour/shield
        # placeholder slots the std-dev detector keeps mis-reading as occupied).
        # We skip re-hovering them while they still read non-item — saved empty
        # samples only enter the NN index on the next process start, so this
        # carries the "stop checking it" win within the current session too. A
        # slot drops out of the set the moment a read shows a real item there.
        self._session_empty: set = set()

    def _move_cursor_to(self, x: int, y: int) -> None:
        """Eased reach to (x, y) if a Mouse instance is wired in,
        otherwise the legacy instant teleport. Either path respects
        the input gate."""
        if self._mouse is not None:
            self._mouse.move_to_screen_xy(int(x), int(y))
        else:
            set_cursor_xy(int(x), int(y))

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
        self._move_cursor_to(desktop_x, desktop_y)

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
                         pre_hover_frame: Optional[np.ndarray] = None,
                         stop_when=None,
                         include_empty: bool = False,
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

        results: Dict[str, InspectionResult] = {}
        resolved = 0
        # NOTE: we deliberately do NOT save/restore the OS cursor around the
        # hover pass. The bot is synthetic — wherever the cursor ends up is
        # fine, and the screen is usually about to close anyway. Easing it back
        # to "where the user left it" was pure wasted motion before every close.

        # Surgical: if the caller already has what it needs from the
        # static read, don't hover anything at all.
        if stop_when is not None and stop_when(snap):
            return results
        for name, slot in rects.items():
            if resolved >= self.cfg.max_resolutions_per_call:
                break
            # Never hover the CRAFTING grid (craft_in_*) or its result
            # (craft_result). Those are the crafter's working area — it places
            # KNOWN items there and reads the result by presence only — so
            # spending a tooltip hover to "identify" them is pure waste (this is
            # the "keeps checking the crafting squares" the operator saw). The
            # ledger likewise tracks only storage (inv_/hotbar_), never the grid.
            if name.startswith("craft_"):
                continue
            content = snap.slots.get(name)
            # Skip truly empty slots and slots already identified
            # confidently. ``include_empty`` overrides the empty skip: the
            # EmptySlotDetector occasionally false-empties a real icon (the
            # iso crafting-table reads as empty), so a targeted search can
            # ask to hover even "empty" slots to rescue/learn them.
            if content is None:
                continue
            if content.is_empty and not include_empty:
                continue
            if (content.item is not None
                    and content.confidence >= self.cfg.skip_above_confidence):
                self._session_empty.discard(name)     # a real item showed up
                continue
            # Already learned empty this session (and still reads non-item) ->
            # don't re-hover it. Mark it empty in the snapshot and move on.
            if name in self._session_empty and content.item is None:
                _mark_slot_empty(snap, name)
                continue

            result = self.resolve_slot(slot, window_origin=window_origin)
            if result is None or not result.item_id:
                # The hover found NOTHING here — the slot is actually empty
                # (e.g. an armour/shield slot whose placeholder art fooled the
                # std-dev empty-detector into thinking it held an item). Record
                # the pre-hover pixels under the EMPTY sentinel so the
                # recogniser learns this slot's empty look and skips the hover
                # next time, instead of re-checking it every read; and remember
                # it for the rest of this session (the NN only picks up the new
                # sample on the next process start).
                if (self._sample_store is not None
                        and pre_hover_frame is not None
                        and content.is_empty is False):
                    self._save_sample_from_frame(
                        pre_hover_frame, slot, EMPTY_SAMPLE_LABEL)
                self._session_empty.add(name)
                _mark_slot_empty(snap, name)
                continue
            self._session_empty.discard(name)         # hover found a real item
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
            # Surgical: stop the moment the caller has enough (e.g. the
            # recipe is now plannable) instead of scanning the rest.
            if stop_when is not None and stop_when(snap):
                break

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
        except Exception as e:
            # Sample save is best-effort but a SYSTEMATIC failure
            # (disk full, sample-store corrupted, sample-store mount
            # offline) silently kills the Phase-3 learning loop. The
            # recogniser would otherwise still answer hovers from the
            # in-memory templates, masking the data-loss for hours.
            # First-failure WARN; subsequent failures silenced to
            # avoid spamming at hover rate.
            if not getattr(self, "_save_warn_emitted", False):
                self._save_warn_emitted = True
                print(f"[inventory_inspector][WARN] sample save for "
                      f"{item_id!r} failed: {e!r}. Phase-3 dataset "
                      f"growth disabled until restart. Further "
                      f"failures silenced.")


__all__ = [
    "InventoryInspector", "InspectorConfig", "InspectionResult",
    "get_cursor_xy", "set_cursor_xy",
]
