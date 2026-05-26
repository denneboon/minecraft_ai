# vision/world/f3_target.py
"""
Parse the F3 overlay's "Targeted Block" / "Looking at block" entry.

Layout has changed across MC versions
-------------------------------------
Older MC versions (pre-1.20) rendered the targeted-block info on a
single line::

    Targeted Block: -7, 64, 134  minecraft:stone

Modern MC (1.20+ / 1.21.x) splits the same information across
several consecutive lines on the LEFT column of the F3 overlay::

    Targeted Block: -7, 64, 134
    minecraft:stone
    snowy: false
    #minecraft:mineable/pickaxe
    #minecraft:base_stone_overworld
    ...

When the player has ``debug_screen_text.looking_at_block`` set to
``always`` (the recommended setting for an AI playing on this
instance) these lines are visible at all times without holding F3.

This module accepts a list of F3 lines (in any order) and returns a
:class:`vision.world.types.LookingAtBlock` if a recognizable target is
present. It works for both layouts: single-line and multi-line.

Future-proofing
---------------
We key off line *contents*, not position. The parser tolerates:
  * different label spellings ("Targeted Block", "Looking at block",
    "Target block"),
  * coords on the label line OR on a separate line,
  * block id on the label line OR on a separate line below it,
  * arbitrary order between the label, the id, and the tag lines.

If MC changes the F3 format again, the per-pattern regexes are the
only thing that needs updating — the orchestration above stays valid.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

from vision.world.types import LookingAtBlock


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
# A block id is ``minecraft:<snake_case>`` (or any lower-case namespace).
# We exclude lines starting with ``#`` here because those are TAG lines
# ("#minecraft:mineable/pickaxe"), not block-id lines.
_RE_BLOCK_ID_LINE = re.compile(
    r"^\s*([a-z][a-z0-9_]*)\s*:\s*([a-z][a-z0-9_/]*)\s*$"
)
_RE_BLOCK_ID_INLINE = re.compile(
    r"(?<![#\w])([a-z][a-z0-9_]*):([a-z][a-z0-9_/]*)\b"
)
_RE_TARGET_COORDS = re.compile(
    r"(-?\d+)\s*[,\s]\s*(-?\d+)\s*[,\s]\s*(-?\d+)"
)
_RE_DIRECTION_STATE = re.compile(r"direction\s*=\s*([a-z]+)", re.IGNORECASE)
# Full and "tolerated-misread" label fragments. Real captures sometimes
# corrupt one character of the label (e.g. "Targeted |?lock:") so we
# also recognise just "targeted " / "looking at " as a label hint when
# the line ALSO contains a coord triple — partial match for one keyword
# plus a numeric triple gives the same false-positive resistance as
# the strict full-label match.
_TARGET_LABELS_STRICT = ("targeted block", "looking at block", "target block")
_TARGET_LABELS_HINT   = ("targeted", "looking at", "target")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_looking_at_block(lines: Iterable[str]) -> Optional[LookingAtBlock]:
    """
    Scan F3 lines and return a :class:`LookingAtBlock` if a targeted
    entry is recognised. Returns ``None`` otherwise.

    Strategy:
      1. Find the index of the line that contains a label.
      2. Pull coords from that line (older format) OR scan the next
         few lines for a coord triple (newer format — though usually
         the coords ARE on the label line, just without the block id).
      3. Pull the block id from the same line, the line below, or
         anywhere downstream — taking the first id-line we find that
         isn't a "#tag" line.
      4. Optionally read ``direction=...`` from any line in that block.
    """
    lst: List[str] = [(s or "").strip() for s in lines]
    label_idx = _find_label_line(lst)
    if label_idx is None:
        return None

    label_line = lst[label_idx]
    coord_pos = _extract_coords(label_line)
    if coord_pos is None:
        # Some layouts may put coords on a line right above or below
        # the label. Check the nearest neighbours.
        for j in (label_idx + 1, label_idx - 1):
            if 0 <= j < len(lst):
                coord_pos = _extract_coords(lst[j])
                if coord_pos is not None:
                    break

    block_id = _extract_block_id_inline(label_line)
    if block_id is None:
        # Search forward for the first non-tag id-line. Cap the lookup
        # at a handful of lines so we don't grab an unrelated id far
        # down the panel.
        for j in range(label_idx + 1, min(label_idx + 12, len(lst))):
            cand = lst[j]
            if not cand:
                continue
            if cand.startswith("#"):
                continue
            bid = _extract_block_id_line(cand)
            if bid is not None:
                block_id = bid
                break

    if block_id is None:
        return None

    face: Optional[str] = None
    for j in range(label_idx, min(label_idx + 16, len(lst))):
        m = _RE_DIRECTION_STATE.search(lst[j])
        if m:
            face = m.group(1).lower()
            break

    if coord_pos is None:
        # No coords resolved — return id-only with reduced confidence.
        return LookingAtBlock(block_id=block_id, pos=(0, 0, 0),
                              face=face, confidence=0.5)

    return LookingAtBlock(block_id=block_id, pos=coord_pos,
                          face=face, confidence=1.0)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _find_label_line(lines: List[str]) -> Optional[int]:
    """Find the index of the line that announces a targeted-block readout.

    Two paths:
      1. Strict: the line contains a full known label ("targeted block",
         "looking at block", "target block").
      2. Tolerant: the line contains a *partial* label keyword AND a
         coord triple. This catches OCR-corrupted captures where a
         single glyph in the label was misread (e.g. "Targeted |?lock:
         -83, 96, -114") but the structural cues are otherwise intact.
    """
    for i, raw in enumerate(lines):
        low = raw.lower()
        if any(label in low for label in _TARGET_LABELS_STRICT):
            return i
    for i, raw in enumerate(lines):
        low = raw.lower()
        if any(hint in low for hint in _TARGET_LABELS_HINT):
            if _RE_TARGET_COORDS.search(raw):
                return i
    return None


def _extract_coords(line: str) -> Optional[Tuple[int, int, int]]:
    if not line:
        return None
    m = _RE_TARGET_COORDS.search(line)
    if not m:
        return None
    try:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


_SUSPICIOUS_PARTIAL_IDS = frozenset({
    # Common partial-glyph misreads — these appear ONLY as suffixes
    # of door / sign / banner texture stems in the asset jar
    # (oak_door_lower, dark_oak_sign, etc.) and the OCR sometimes
    # collapses them to bare tokens that look like block ids.
    "lower", "upper", "top", "bottom", "side",
    "head", "foot", "front", "back", "inner",
    "outer", "double",
})


def _extract_block_id_line(line: str) -> Optional[str]:
    """If ``line`` is exactly an ``ns:id`` token, return ``ns:id``."""
    m = _RE_BLOCK_ID_LINE.match(line)
    if not m:
        return None
    ns  = m.group(1)
    bid = m.group(2)
    # Reject suspicious partial-stem ids that aren't real block names.
    # These come from texture-stem misreads, not legitimate F3 output.
    if bid in _SUSPICIOUS_PARTIAL_IDS:
        return None
    return f"{ns}:{bid}"


def _extract_block_id_inline(line: str) -> Optional[str]:
    """Find a ``ns:id`` token anywhere in a line, skipping ``#`` tags."""
    m = _RE_BLOCK_ID_INLINE.search(line)
    if not m:
        return None
    bid = m.group(2)
    if bid in _SUSPICIOUS_PARTIAL_IDS:
        return None
    return f"{m.group(1)}:{bid}"


__all__ = ["parse_looking_at_block"]
