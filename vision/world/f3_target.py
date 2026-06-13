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
        #
        # We try the STRICT line-form first (``^minecraft:<id>$``) so
        # that a clean line wins, and fall back to the INLINE form
        # (``minecraft:<id>`` anywhere in the line) when the OCR added
        # trailing garble like ``minecraft:stone ?``. The inline form
        # has the same plausibility guards (real-block namespace,
        # min length, property/dimension blacklist) so it doesn't
        # mistakenly accept tag lines or dimension lines.
        for j in range(label_idx + 1, min(label_idx + 12, len(lst))):
            cand = lst[j]
            if not cand:
                continue
            if cand.startswith("#"):
                continue
            bid = _extract_block_id_line(cand)
            if bid is None:
                bid = _extract_block_id_inline(cand)
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
        # No coords resolved. The old code returned a (0, 0, 0)
        # sentinel with confidence=0.5 here, expecting downstream to
        # reject it on the confidence floor — but that produced
        # confusing "[F3] looking_at: id=X pos=(0, 0, 0) conf=0.5"
        # diagnostic lines that LOOKED like phantom origin commits.
        # Returning None makes the parser API honest: we know there's
        # a block, we just can't localise it, so don't pretend we can.
        # Downstream code already handles ``looking_at = None``
        # gracefully (it treats the read as "no target this tick").
        return None

    return LookingAtBlock(block_id=block_id, pos=coord_pos,
                          face=face, confidence=1.0)


def targeted_block_pos(lines: Iterable[str]) -> Optional[Tuple[int, int, int]]:
    """Return just the targeted block's coordinates, regardless of whether the
    block id could be read. F3 prints "Targeted Block: x, y, z" reliably even
    when the id line below it OCRs to garble (proportional font on a busy
    background) — e.g. a freshly-placed crafting table reads as ``?`` but its
    coords are clean. ``parse_looking_at_block`` returns None without an id, so
    this is the position-only fallback for code that just needs to know WHICH
    block the crosshair is on (placement confirmation, not identification)."""
    lst: List[str] = [(s or "").strip() for s in lines]
    label_idx = _find_label_line(lst)
    if label_idx is None:
        return None
    pos = _extract_coords(lst[label_idx])
    if pos is None:
        for j in (label_idx + 1, label_idx - 1):
            if 0 <= j < len(lst):
                pos = _extract_coords(lst[j])
                if pos is not None:
                    break
    return pos


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _find_label_line(lines: List[str]) -> Optional[int]:
    """Find the index of the line that announces a targeted-block readout.

    Three paths, in order of confidence:

      1. **Strict** — the line contains a full known label
         ("targeted block", "looking at block", "target block").
      2. **Partial-label tolerant** — the line contains a partial
         label keyword AND a coord triple. Catches single-glyph
         misreads (``Targeted |?lock: -83, 96, -114``).
      3. **Structural fallback** — when the label is fully garbled
         (busy biome backgrounds eat the whole "Targeted Block:"
         line, leaving ``???????? ?????? ???? ??? ??? ?'????``), we
         look for the line IMMEDIATELY ABOVE a ``minecraft:<id>``
         block-line that's followed by at least one ``#minecraft:``
         tag line. That pattern is unique to the Targeted Block
         section of the F3 panel; no other place in the panel
         emits a tag list right under a ``minecraft:<id>`` row.
         Returning that index lets the rest of the parser try to
         pull the coord triple from it — even if it can only
         partial-match a few digits, that's better than no log at all.
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
    # Structural fallback: pair (minecraft:<id>) + (#minecraft:...).
    # Use the LENIENT inline-id extractor here so we still match when
    # the OCR appends a trailing garble char (``minecraft:stone ?``).
    for i, raw in enumerate(lines):
        bid = _extract_block_id_inline(raw)
        if bid is None:
            continue
        # Skip lines where the id is preceded by ``#`` (tag lines —
        # those should never be claimed as the block-id row).
        stripped = (raw or "").strip()
        if stripped.startswith("#"):
            continue
        # Look forward up to a few lines for a #-tag pattern. That
        # pattern is unique to the Targeted Block section of the
        # F3 panel.
        found_tag = False
        for j in range(i + 1, min(i + 6, len(lines))):
            adj = (lines[j] or "").strip().lower()
            if adj.startswith("#minecraft:") or adj.startswith("#"):
                found_tag = True
                break
        if not found_tag:
            continue
        # The Targeted Block "label" line is whichever line sits
        # immediately before this block-id line. Returning i-1 lets
        # ``parse_looking_at_block`` look there (and at neighbours)
        # for the coord triple. If the label was eaten beyond
        # recognition the coords may also be lost — but the
        # block-id-from-this-line still succeeds and we at least get
        # the identity of the block.
        return max(0, i - 1)
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

# The F3 panel renders block PROPERTY rows (``axis: y``, ``snowy: false``,
# ``east: true``, ``waterlogged: false``, ``power: 0``, ``half: bottom``)
# directly under the ``minecraft:<id>`` row. Each of those lines has the
# same ``key:value`` shape our block-id regex matches, so without a
# guard the parser would happily commit ``east:true`` or ``axis:y`` as
# the player's "Looking at block" entry — that's exactly the
# ``south:t @ (-92, 92, -93)`` bogus log line we just observed.
#
# We block these by NAMESPACE: real block ids are always under the
# ``minecraft:`` namespace (or a modded one), never under property
# names like ``east`` / ``axis`` / ``snowy``. The list mirrors the
# canonical block-state property names from MC 1.20+.
_NON_BLOCK_NAMESPACES = frozenset({
    # Sides used in fence/wall connections, redstone, glass_pane, etc.
    "north", "south", "east", "west", "up", "down",
    # Orientation properties.
    "axis", "facing", "rotation", "orientation",
    # Boolean state properties.
    "snowy", "waterlogged", "powered", "lit", "open", "berries",
    "persistent", "attached", "disarmed", "extended", "hanging",
    "in_wall", "occupied", "triggered", "unstable",
    "drag", "conditional", "has_book", "has_bottle_0", "has_bottle_1",
    "has_bottle_2", "has_record", "inverted", "locked", "short",
    "enabled", "tilt", "sculk_sensor_phase", "vault_state",
    # Position-in-multipart properties.
    "half", "part", "type", "leaves", "mode", "shape", "face",
    "instrument", "attachment", "thickness", "bites",
})

# These NAME COMPONENTS are never the second half of a block id.
# Dimension ids (``minecraft:overworld``, ``minecraft:the_nether``)
# share the ``minecraft:`` namespace with real blocks, so the
# namespace check above doesn't catch them — they're rejected here
# by the BID name instead. Also covers the FC marker that ends MC's
# dimension line (``minecraft:overworld FC: 0`` → after the regex
# strips the ``FC: 0``, we get ``minecraft:overworld``).
_NON_BLOCK_IDS = frozenset({
    "overworld", "the_nether", "the_end", "the_void",
    # Item / entity namespaces sometimes share text in the F3 panel
    # (e.g. damage indicator). Block ids never include these stems.
    "air",  # never targeted; rejecting is safe and prevents a stray
            # OCR misread from committing ``minecraft:air``.
})

# Real vanilla block ids are at least this many chars (the shortest
# normally-targeted block is ``stone`` = 5 chars; ``air`` = 3 is
# excluded above). 4 is a safe floor that rejects OCR truncations
# like ``minecraft:s`` while still accepting every legitimate block.
_MIN_BLOCK_ID_LEN = 4


# ALLOWLIST of valid block namespaces. Block-state PROPERTY rows parse to
# ``<property>:<value>`` (``crafting: false`` -> ns="crafting"), and the old
# denylist of property names was perpetually incomplete — every MC version adds
# properties (``crafting``, ``hinge``, ``trial_spawner_state``, …) that leaked
# their ``key:value`` as a confirmed block id at confidence 1.0. A namespace
# ALLOWLIST can't be outrun: a real block id is always ``minecraft:`` (add
# modded namespaces here if ever needed). This is the structural guard; the
# perception layer's catalog validator is the authoritative downstream gate.
_BLOCK_NAMESPACES = frozenset({"minecraft"})

# Property VALUES that pass the length floor but are never a block-id stem —
# rejected so an OCR bleed that drops the property key (leaving a bare value)
# can't sneak through.
_PROPERTY_VALUE_LITERALS = frozenset({
    "true", "false", "left", "right", "none", "compare", "subtract",
    "active", "inactive", "cooldown", "ejecting", "awake", "dormant",
    "unlit", "small", "large", "tall", "wall_hanging",
})


def _looks_like_block_id(ns: str, bid: str) -> bool:
    """True iff ``ns:bid`` plausibly names a real block."""
    if ns not in _BLOCK_NAMESPACES:        # allowlist: property-key "namespaces" out
        return False
    if bid in _NON_BLOCK_IDS:
        return False
    if bid in _PROPERTY_VALUE_LITERALS:
        return False
    if bid in _SUSPICIOUS_PARTIAL_IDS:
        return False
    if len(bid) < _MIN_BLOCK_ID_LEN:
        return False
    # OCR-garble guard: real MC block ids consist of lowercase letters,
    # digits, and single underscores SEPARATING words — they never
    # start with ``_``, end with ``_``, or contain ``__``. The
    # glyph_ocr backend occasionally reads a trailing character as a
    # stray underscore (e.g. ``vine`` -> ``vine_``); accepting these
    # would pollute the WorldMap with ids no exporter or agent can
    # match. Cheap structural reject saves a lot of pain.
    if bid.startswith("_") or bid.endswith("_") or "__" in bid:
        return False
    return True


def _extract_block_id_line(line: str) -> Optional[str]:
    """If ``line`` is exactly an ``ns:id`` token AND it plausibly names
    a real block, return ``ns:id``. Property rows under the panel
    (``axis: y``, ``snowy: false``, ``east: true``) match the same
    pattern but are rejected by :func:`_looks_like_block_id`."""
    m = _RE_BLOCK_ID_LINE.match(line)
    if not m:
        return None
    ns  = m.group(1)
    bid = m.group(2)
    if not _looks_like_block_id(ns, bid):
        return None
    return f"{ns}:{bid}"


def _extract_block_id_inline(line: str) -> Optional[str]:
    """Find a ``ns:id`` token anywhere in a line, skipping ``#`` tags
    and applying the same plausibility check as the line-form parser."""
    m = _RE_BLOCK_ID_INLINE.search(line)
    if not m:
        return None
    ns  = m.group(1)
    bid = m.group(2)
    if not _looks_like_block_id(ns, bid):
        return None
    return f"{ns}:{bid}"


__all__ = ["parse_looking_at_block"]
