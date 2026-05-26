# vision/world/types.py
"""
Dataclasses shared by every world-perception module.

These types are the wire format between the perception orchestrator,
the WorldMap, and downstream agents. They are deliberately small,
typed, and serialisable so future planners (LLM-based or otherwise)
can consume them without reaching into vision internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Player pose
# ---------------------------------------------------------------------------

@dataclass
class PlayerPose:
    """
    The player's position + orientation, in Minecraft world coords.

    Derived primarily from the F3 overlay (``vision.ocr.F3Info``) but
    promoted here so the perception layer never has to know which OCR
    line provided which field.

    Conventions
    -----------
    * ``x``, ``y``, ``z`` are floats (player's eye-level y is ``y + 1.62``).
    * ``yaw`` follows MC: 0 = facing +Z (south), positive = clockwise
      when viewed from above. Wraps at ±180.
    * ``pitch`` follows MC: 0 = level horizon, -90 = looking straight
      up, +90 = looking straight down.
    """

    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    dimension: str = "minecraft:overworld"
    block_x: Optional[int] = None
    block_y: Optional[int] = None
    block_z: Optional[int] = None
    timestamp: float = 0.0

    @property
    def eye_y(self) -> float:
        """Y coord of the player's eye (1.62 above feet)."""
        return self.y + 1.62

    def feet_block(self) -> Tuple[int, int, int]:
        """Integer block under the player's feet."""
        if (self.block_x is not None
                and self.block_y is not None
                and self.block_z is not None):
            return (self.block_x, self.block_y, self.block_z)
        # Fall back to flooring the float coords.
        import math
        return (int(math.floor(self.x)),
                int(math.floor(self.y)),
                int(math.floor(self.z)))


# ---------------------------------------------------------------------------
# Block observation
# ---------------------------------------------------------------------------

@dataclass
class BlockObservation:
    """
    One block the AI has seen at a known integer-block position.

    Lifecycle
    ---------
    Created by :mod:`vision.world.perception` whenever a block at
    ``pos`` has been identified — either by reading the F3 "Looking
    at block" line, by classifying a patch of the rendered view, or
    by inheriting a neighbour's identity (extrapolation).

    Stored in :class:`vision.world.map.WorldMap` keyed by ``pos``.
    Later observations of the same position overwrite the entry only
    if their ``confidence`` exceeds the stored one OR they come from
    a more authoritative ``source``.
    """

    pos: Tuple[int, int, int]
    block_id: Optional[str]       # ``"minecraft:stone"`` or None for "unknown"
    confidence: float = 0.0        # 0..1
    source: str = "unknown"        # "looking_at" | "vision_patch" | "extrapolation" | "manual"
    last_seen_tick: int = 0        # WorldPerception's tick counter
    dimension: str = "minecraft:overworld"
    # Free-form extra info populated by downstream modules.
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Entity observation (mobs, players, vehicles, projectiles)
# ---------------------------------------------------------------------------

@dataclass
class EntityObservation:
    """
    One entity detected somewhere on screen.

    Many fields are optional because vision-side detection is
    intrinsically uncertain — we might know an entity is visible and
    where on screen it is, without knowing exactly which species it is
    or where in 3-D space it stands. Downstream code should treat
    ``None`` / low-confidence values as "I don't know" and not as zero.
    """

    entity_id: Optional[str]              # ``"minecraft:zombie"`` or None
    screen_box: Tuple[int, int, int, int] # (x, y, w, h) in screen pixels
    world_pos: Optional[Tuple[float, float, float]] = None
    distance_blocks: Optional[float]      = None
    hostile: Optional[bool]               = None
    confidence: float                     = 0.0
    source: str                           = "unknown"
    last_seen_tick: int                   = 0
    dimension: str                        = "minecraft:overworld"
    meta: Dict[str, Any]                  = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Item-drop observation (item entities lying on the ground)
# ---------------------------------------------------------------------------

@dataclass
class ItemDropObservation:
    """
    A dropped item entity on the floor.

    Kept separate from :class:`EntityObservation` because their
    in-world behaviour (despawn timer, magnetism toward the player,
    pickup priority) is qualitatively different from mob behaviour.
    """

    item_id: Optional[str]
    screen_box: Tuple[int, int, int, int]
    world_pos: Optional[Tuple[float, float, float]] = None
    count_estimate: int                   = 1
    confidence: float                     = 0.0
    source: str                           = "unknown"
    last_seen_tick: int                   = 0
    dimension: str                        = "minecraft:overworld"


# ---------------------------------------------------------------------------
# "Looking at block" — when the F3 overlay tells us exactly
# ---------------------------------------------------------------------------

@dataclass
class LookingAtBlock:
    block_id: str
    pos: Tuple[int, int, int]
    face: Optional[str] = None           # "up" | "down" | "north" | "south" | "east" | "west"
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Per-tick world frame returned by WorldPerception.update()
# ---------------------------------------------------------------------------

@dataclass
class WorldFrame:
    """
    A snapshot of what the AI sees *this* tick.

    Contains only freshly-observed data. The accumulated long-term
    memory of every block ever seen lives on the :class:`WorldMap`
    referenced via ``world_map``.
    """

    tick: int
    pose: Optional[PlayerPose] = None
    looking_at: Optional[LookingAtBlock] = None
    visible_blocks: List[BlockObservation] = field(default_factory=list)
    visible_entities: List[EntityObservation] = field(default_factory=list)
    visible_drops: List[ItemDropObservation] = field(default_factory=list)
    # Diagnostics: per-patch best/second scores, timings, etc. Useful
    # for debugging and offline analysis; agents should ignore.
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # Reference back to the persistent map (not copied — same instance
    # the perception layer mutates). None when perception has no map.
    world_map: Optional["object"] = None    # quoted to avoid import cycle


__all__ = [
    "PlayerPose",
    "BlockObservation",
    "EntityObservation",
    "ItemDropObservation",
    "LookingAtBlock",
    "WorldFrame",
]
