# vision/world/pathfind.py
"""
A* pathfinding over the WorldMap voxel grid.

The pathfinder turns "I'm at (x, y, z) and want to be at (gx, gy, gz)"
into a sequence of waypoint cells the agent can walk through. It works
on the WorldMap's KNOWN voxels — solids the agent has observed via F3
or via the inverse-renderer expansion, plus carved-air sightlines.

Move set
--------
We model a humanlike walking agent (no flying, no swimming, no portals):

  * Walk N/S/E/W — same Y, adjacent cell, both feet+head clear, ground
    block directly beneath the destination.
  * Step up — destination one block higher, requires the EXTRA head
    clearance one block above the destination's head (so a jump fits).
  * Drop down — destination 1-3 blocks lower, requires the destination
    to be a regular standable cell. Drops over 3 inflict fall damage
    so the pathfinder treats them as impassable.

We do NOT yet model: bridging, parkour gaps, ladders, vines as climbs,
nor water swimming. Each can land later as a new move type without
disturbing existing call sites.

Unknown voxels
--------------
The WorldMap stores only OBSERVED voxels. Anything else returns
``get_block() == None``. We expose a configurable policy on how to
treat unknown space:

  * ``"ground_only"`` (DEFAULT) — unknown is passable EXCEPT for the
    cell directly under a candidate standable position, which must be a
    known solid. Compromise: lets us walk through air we haven't looked
    at, but won't blindly step (or descend) off into unknowns — the
    walker handles small unexpected drops it meets during execution.
  * ``"passable"`` — treat unknown as air, AND optimistically as ground
    under a step. Lets the pathfinder route through (and descend into)
    unobserved space in open biomes; the walker validates as it goes and
    replans if it hits an actual solid. ``NavigateTo`` uses this.
  * ``"blocking"`` — treat unknown as solid. The pathfinder only routes
    through observed-passable space. Safer indoors / in caves but
    useless until the AI has scanned the area.

The default ``ground_only`` is the conservative middle ground; callers
that want to plan through/into unobserved terrain pass ``passable``.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from vision.world.map import AIR_BLOCK, WorldMap


# ---------------------------------------------------------------------------
# Block classification
# ---------------------------------------------------------------------------

# Non-solid blocks the player can walk through. Centralised here so any
# block id we discover during play that should be passable lands in ONE
# spot rather than smeared across the agent code. The list is
# intentionally conservative — only blocks that genuinely don't block
# walking. Half-slabs, fences, doors etc. need richer logic and stay
# out for now.
_PASSABLE_BLOCK_IDS = frozenset({
    AIR_BLOCK,
    "minecraft:air",
    "minecraft:cave_air",
    "minecraft:void_air",
    # Foliage that doesn't block movement.
    "minecraft:short_grass",
    "minecraft:tall_grass",
    "minecraft:fern",
    "minecraft:large_fern",
    "minecraft:dead_bush",
    "minecraft:vine",
    "minecraft:sugar_cane",
    "minecraft:bamboo",
    "minecraft:bamboo_sapling",
    "minecraft:wheat",
    "minecraft:carrots",
    "minecraft:potatoes",
    "minecraft:beetroots",
    "minecraft:nether_wart",
    "minecraft:sweet_berry_bush",
    "minecraft:kelp",
    "minecraft:kelp_plant",
    "minecraft:seagrass",
    "minecraft:tall_seagrass",
    "minecraft:pink_petals",
    # Decorative blocks that don't impede walking.
    "minecraft:torch",
    "minecraft:wall_torch",
    "minecraft:redstone_torch",
    "minecraft:redstone_wire",
    "minecraft:soul_torch",
    "minecraft:soul_wall_torch",
    "minecraft:lever",
    "minecraft:stone_button",
    "minecraft:oak_button",
    "minecraft:spruce_button",
    "minecraft:birch_button",
    "minecraft:jungle_button",
    "minecraft:acacia_button",
    "minecraft:dark_oak_button",
    "minecraft:mangrove_button",
    "minecraft:cherry_button",
    "minecraft:pale_oak_button",
    "minecraft:bamboo_button",
    "minecraft:crimson_button",
    "minecraft:warped_button",
    "minecraft:tripwire",
    "minecraft:tripwire_hook",
    "minecraft:end_rod",
    "minecraft:lightning_rod",
})

# Blocks that ARE solid (block walking) but kill on contact, or
# blocks the player should NEVER step INTO regardless of solidity.
# The pathfinder treats them as hazards and refuses to plan a step
# whose feet would occupy them. Standing on TOP is generally fine
# (handled via the floor-solid check looking at the cell below).
#
# Categories:
#   * Damage on contact: lava, fire, magma_block (on top), cactus
#   * Damage / debuff in voxel: wither_rose, sweet_berry_bush
#   * Slow + freeze: powder_snow
#   * Teleport / dimension change: portals
#   * Suction: end_gateway / end_portal (kill or teleport)
# This list is intentionally tighter than "every annoying block";
# it's just the ones a humanlike walking agent must AVOID.
_HAZARD_BLOCK_IDS = frozenset({
    # Direct damage on contact.
    "minecraft:lava",
    "minecraft:fire",
    "minecraft:soul_fire",
    "minecraft:campfire",
    "minecraft:soul_campfire",
    "minecraft:cactus",
    # Damage / debuff inside voxel.
    "minecraft:wither_rose",
    "minecraft:sweet_berry_bush",
    # Stand-on-top hazard (damage when standing on top). The floor
    # check below rejects them as ground; included here for the
    # "don't walk into them" case too.
    "minecraft:magma_block",
    # Slow + freeze.
    "minecraft:powder_snow",
    # Portals — teleport / kill.
    "minecraft:nether_portal",
    "minecraft:end_portal",
    "minecraft:end_gateway",
})

# Blocks that are SOLID for ground-truth purposes (you can stand on
# top of them as long as nothing else makes them hazardous) but whose
# top-surface kills / hurts you. The pathfinder refuses to plan a
# step whose FLOOR is one of these — the player would take damage
# just standing there. Different from ``_HAZARD_BLOCK_IDS`` which
# bans the cell as a FOOT target.
_DEADLY_FLOOR_BLOCK_IDS = frozenset({
    "minecraft:magma_block",   # damage while standing
    "minecraft:lava",          # if somehow exposed as floor
    "minecraft:campfire",      # damage while on top
    "minecraft:soul_campfire",
})


def is_passable(block_id: Optional[str]) -> bool:
    """True if the player can walk INTO this voxel (foot or head)."""
    if block_id is None:
        # Unknown — handled by the policy layer in PathfinderConfig.
        # Default to PASSABLE so the pathfinder doesn't refuse to route
        # through open biomes the AI hasn't fully observed yet.
        return True
    if block_id in _HAZARD_BLOCK_IDS:
        return False
    return block_id in _PASSABLE_BLOCK_IDS


def is_solid(block_id: Optional[str]) -> bool:
    """True if this voxel supports a player standing on top of it."""
    if block_id is None:
        return False        # unknown -> no foothold
    if block_id == AIR_BLOCK:
        return False
    return block_id not in _PASSABLE_BLOCK_IDS


def is_hazard(block_id: Optional[str]) -> bool:
    """True if walking INTO this voxel would damage / kill the player."""
    return block_id in _HAZARD_BLOCK_IDS


# ---------------------------------------------------------------------------
# Pathfinder config + result
# ---------------------------------------------------------------------------


@dataclass
class PathfinderConfig:
    # How to treat unknown voxels:
    #   * ``"ground_only"`` (default, recommended) — unknown cells
    #     count as PASSABLE for feet/head, but the ground beneath a
    #     standable cell MUST be observed-solid. This bounds the
    #     search to the perimeter of known terrain while still
    #     allowing the agent to walk through unobserved air over
    #     known ground (e.g. crossing a partly-scanned field).
    #   * ``"passable"`` — both passability AND ground are optimistic.
    #     ANY cell becomes standable; A* must rely on the heuristic
    #     to stay focused. Useful only for short-range plans over
    #     guaranteed-flat terrain.
    # A pure-pessimistic policy was tried ("blocking", which refused
    # to plan through any unobserved cell) but proved unworkable —
    # the agent's own start cell almost always has unobserved
    # neighbours, so the pathfinder ended up refusing every request.
    unknown_policy: str = "ground_only"

    # Max search effort. A* will refuse to explore more than this many
    # cells and return None instead. Prevents a degenerate plan call
    # from freezing the agent loop. 50K nodes ≈ 100 ms on a Ryzen 7.
    # Bumped from 20K after live testing: in ``passable`` mode (every
    # unknown cell is standable) a 7-block Manhattan target can still
    # need 25K+ expansions because each node has up to 20 neighbours
    # and the heuristic is admissible (no tie-breaker preference).
    max_nodes_expanded: int = 50_000

    # Heuristic weight. 1.0 = admissible (optimal). >1 trades
    # optimality for speed (weighted A*). We default to 1.2 because:
    #   * The worlds are small + the move costs are similar enough
    #     that a slight over-estimate barely affects path quality.
    #   * Under ``passable`` policy every cell is standable, so an
    #     admissible heuristic gives A* zero tie-breaker preference
    #     in directions perpendicular to the goal — the search ends
    #     up exploring a thick wedge before goal-direction wins.
    #     A weight > 1 dramatically tightens the search cone.
    # 1.2 keeps paths within ~10 % of optimal in practice.
    heuristic_weight: float = 1.2

    # Max safe fall (blocks). MC takes damage past ~3-4 blocks and
    # kills past ~22. We stay conservative at 3 to avoid any fall
    # damage during pathfinding.
    max_fall_blocks: int = 3

    # Move costs. Walking is the unit; jumps are slightly slower in
    # MC (you cover horizontal distance more slowly mid-jump), and
    # falls are slightly slower than walking due to recovery time.
    cost_walk: float = 1.0
    cost_jump_up: float = 1.2
    cost_drop_1: float = 1.05
    cost_drop_n: float = 1.15      # per additional block dropped


@dataclass
class PathResult:
    waypoints: List[Tuple[int, int, int]]   # start..goal inclusive
    cost: float
    nodes_expanded: int
    reason: str                              # "ok" | "blocked" | "budget"

    @property
    def ok(self) -> bool:
        return self.reason == "ok"

    def __bool__(self) -> bool:
        return self.ok and len(self.waypoints) > 0


# ---------------------------------------------------------------------------
# Standability + move generation
# ---------------------------------------------------------------------------


class _WorldQuery:
    """
    Thin cached lookup over the WorldMap. Repeated A* neighbour
    expansions hammer ``get_block`` at the same voxel many times; a
    dict cache costs ~10 ns per hit vs ~200 ns for the WorldMap lookup
    and removes the dim-store branching from the hot path.

    The cache stays valid for the lifetime of a single pathfind call —
    we don't try to invalidate on world updates, since updates are
    rare relative to the per-tick plan cost.
    """

    __slots__ = ("_world", "_dim", "_cache")

    def __init__(self, world: WorldMap, dimension: Optional[str]):
        self._world = world
        self._dim = dimension
        self._cache: dict = {}

    def get_id(self, pos: Tuple[int, int, int]) -> Optional[str]:
        c = self._cache
        v = c.get(pos)
        if v is not None:
            return v if v != "__none__" else None
        obs = self._world.get_block(pos, dimension=self._dim)
        bid = obs.block_id if obs is not None else None
        c[pos] = bid if bid is not None else "__none__"
        return bid


def _standable(query: _WorldQuery,
               pos: Tuple[int, int, int],
               cfg: PathfinderConfig,
               ) -> bool:
    """
    True if the player can stand at ``pos`` (feet at this voxel, head
    one above, ground directly below).

    The three unknown_policy values change ``what counts as passable``
    and ``what counts as solid ground`` independently — see the
    PathfinderConfig docstring for the truth table.
    """
    policy = cfg.unknown_policy
    # Both supported policies treat unknown feet/head as passable —
    # see PathfinderConfig docstring for why the pure-pessimistic
    # alternative was dropped.
    unknown_is_passable = True
    # Only the most optimistic policy treats unknown floor as solid.
    unknown_is_ground = (policy == "passable")

    x, y, z = pos
    feet_id  = query.get_id((x, y, z))
    head_id  = query.get_id((x, y + 1, z))
    floor_id = query.get_id((x, y - 1, z))

    def passable_under(bid: Optional[str]) -> bool:
        if bid is None:
            return unknown_is_passable
        if bid in _HAZARD_BLOCK_IDS:
            return False
        return bid in _PASSABLE_BLOCK_IDS

    def grounded_under(bid: Optional[str]) -> bool:
        if bid is None:
            return unknown_is_ground
        if bid in _DEADLY_FLOOR_BLOCK_IDS:
            # Standing on top would inflict damage — don't ground here.
            return False
        return is_solid(bid)

    if not passable_under(feet_id):
        return False
    if not passable_under(head_id):
        return False
    if not grounded_under(floor_id):
        return False
    return True


def _neighbours(query: _WorldQuery,
                pos: Tuple[int, int, int],
                cfg: PathfinderConfig,
                ) -> Iterable[Tuple[Tuple[int, int, int], float]]:
    """
    Yield ``(neighbour_pos, step_cost)`` for every standable cell
    reachable in one humanlike move from ``pos``.
    """
    x, y, z = pos
    # 4 cardinal directions. Diagonal moves are NOT modeled because MC's
    # collision pushes the player back when the diagonal corner has a
    # block — diagonals are a recipe for getting stuck.
    horiz = ((1, 0), (-1, 0), (0, 1), (0, -1))

    # Whether unknown blocks are treated as passable when validating
    # in-between cells along a move. Matches ``_standable``'s policy.
    unknown_is_passable = cfg.unknown_policy in ("passable", "ground_only")

    def _passable(bid: Optional[str]) -> bool:
        if bid is None:
            return unknown_is_passable
        if bid in _HAZARD_BLOCK_IDS:
            return False
        return bid in _PASSABLE_BLOCK_IDS

    # Walk + step up + drop down per cardinal. We yield EVERY valid
    # destination per direction (not just one) because jump-up and
    # drop-down lead to DIFFERENT y cells than a same-Y walk — A* must
    # see them all to plan vertical motion through cells that ALSO
    # have a horizontal walk available. The earlier ``continue`` after
    # a successful walk hid jumps + drops entirely whenever the walk
    # also worked, which under ``passable`` policy (every walk
    # succeeds) made the pathfinder unable to change Y at all.
    for dx, dz in horiz:
        nx, nz = x + dx, z + dz

        # 1) Walk (same Y)
        candidate = (nx, y, nz)
        walk_ok = _standable(query, candidate, cfg)
        if walk_ok:
            yield candidate, cfg.cost_walk

        # 2) Step / jump up by 1
        up = (nx, y + 1, nz)
        if _standable(query, up, cfg):
            # Two cells must be air for the jump arc to fit:
            #   * y+2 over the CURRENT cell — head sweeps through it
            #     at the peak of the jump.
            #   * y+1 over the CURRENT cell — already part of the
            #     standable-current check (the head we're jumping
            #     from must be clear); _standable doesn't verify it
            #     for ``pos`` though because we're testing the
            #     DESTINATION here. Defensive re-check.
            ceil_id = query.get_id((x, y + 2, z))
            head_id = query.get_id((x, y + 1, z))
            if _passable(ceil_id) and _passable(head_id):
                yield up, cfg.cost_jump_up

        # 3) Drop down by 1..max_fall_blocks. The player walks off the
        # edge horizontally FIRST (passing through (nx, y, nz) and
        # head (nx, y+1, nz)), then falls through (nx, y-1..landing+1).
        # We must verify EVERY cell in that column is passable —
        # otherwise the player would be blocked before reaching the
        # landing or smash into a ceiling overhang on the way down.
        #
        # Skip the drop loop ONLY when the same-Y walk is grounded by
        # an OBSERVED solid floor — in that case the player physically
        # cannot fall through it, so drops aren't possible. Under
        # ``passable`` policy, walk_ok is often "true via optimistic
        # unknown-as-ground"; in that case the unknown space might
        # really be air, so a drop is just as plausible as a walk
        # and we must give A* both options to plan vertical descent.
        floor_below_id = query.get_id((nx, y - 1, nz))
        floor_is_observed_solid = (floor_below_id is not None
                                    and is_solid(floor_below_id))
        if walk_ok and floor_is_observed_solid:
            continue
        horiz_feet_id = query.get_id((nx, y, nz))
        horiz_head_id = query.get_id((nx, y + 1, nz))
        if not _passable(horiz_feet_id):
            continue
        if not _passable(horiz_head_id):
            continue
        column_blocked = False
        for drop in range(1, cfg.max_fall_blocks + 1):
            cand = (nx, y - drop, nz)
            # The cell at the SAME y as the prospective landing's
            # head (i.e. one above ``cand``) must be passable too —
            # otherwise the player's head clips a block on landing.
            # ``_standable`` already covers this for the landing, but
            # we also need every cell from (nx, y, nz) down to
            # (nx, y-drop+1, nz) to be passable for the fall itself.
            # The previous iteration's ``cand`` is the cell directly
            # above this one — if THAT was not passable (would have
            # already been a landing or a blocking solid) we stop.
            if column_blocked:
                break
            if _standable(query, cand, cfg):
                cost = cfg.cost_drop_1 + max(0, drop - 1) * (
                    cfg.cost_drop_n - cfg.cost_drop_1
                )
                yield cand, cost
                # Once we find a landing, we don't keep dropping past
                # it — the player wouldn't pass through floor.
                break
            # Not standable here. If THIS voxel isn't even passable
            # (it's a solid block) the player would have stopped
            # falling here regardless of standability — mark the
            # column blocked so the next iteration bails.
            this_feet = query.get_id(cand)
            if not _passable(this_feet):
                column_blocked = True


# ---------------------------------------------------------------------------
# A*
# ---------------------------------------------------------------------------


def _heuristic(a: Tuple[int, int, int],
               b: Tuple[int, int, int],
               weight: float) -> float:
    # 3D Manhattan. Admissible: every step changes the sum by exactly
    # one for walks, ≥1 for jump/drop combined with horizontal moves.
    return weight * (abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]))


def find_path(world_map: WorldMap,
              start: Tuple[int, int, int],
              goal: Tuple[int, int, int],
              *,
              dimension: Optional[str] = None,
              config: Optional[PathfinderConfig] = None,
              ) -> PathResult:
    """
    Plan a path of standable voxels from ``start`` to ``goal``.

    ``start`` / ``goal`` are FEET positions (the voxel the player's
    feet occupy). Both must satisfy the same standability rules as
    every waypoint between them — i.e. you can stand there. If the
    goal isn't standable we return a "blocked" PathResult.
    """
    cfg = config or PathfinderConfig()
    query = _WorldQuery(world_map, dimension)

    if start == goal:
        return PathResult(waypoints=[start], cost=0.0,
                          nodes_expanded=0, reason="ok")

    if not _standable(query, goal, cfg):
        return PathResult(waypoints=[], cost=math.inf,
                          nodes_expanded=0, reason="blocked")
    if not _standable(query, start, cfg):
        return PathResult(waypoints=[], cost=math.inf,
                          nodes_expanded=0, reason="blocked")

    # Priority queue: (f, counter, node). counter breaks heapq ties
    # deterministically without comparing tuples.
    counter = 0
    open_heap: list = [(0.0, counter, start)]
    came_from: dict = {start: None}
    g_score: dict = {start: 0.0}
    closed: set = set()
    nodes_expanded = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        # Closed-set check: an admissible heuristic guarantees the
        # first time we pop a node, its g_score is optimal — so any
        # later pop of the same node is a stale duplicate from when
        # we pushed it with a worse g. Without this filter A* keeps
        # popping the same nodes over and over and the
        # ``nodes_expanded`` budget runs out long before the goal.
        if current in closed:
            continue
        closed.add(current)
        nodes_expanded += 1

        if current == goal:
            # Reconstruct path.
            waypoints: List[Tuple[int, int, int]] = []
            cur = current
            while cur is not None:
                waypoints.append(cur)
                cur = came_from[cur]
            waypoints.reverse()
            return PathResult(waypoints=waypoints,
                              cost=g_score[current],
                              nodes_expanded=nodes_expanded,
                              reason="ok")

        if nodes_expanded > cfg.max_nodes_expanded:
            return PathResult(waypoints=[], cost=math.inf,
                              nodes_expanded=nodes_expanded,
                              reason="budget")

        for nb, step_cost in _neighbours(query, current, cfg):
            if nb in closed:
                continue
            tentative = g_score[current] + step_cost
            prev = g_score.get(nb)
            if prev is None or tentative < prev:
                g_score[nb] = tentative
                came_from[nb] = current
                f = tentative + _heuristic(nb, goal, cfg.heuristic_weight)
                counter += 1
                heapq.heappush(open_heap, (f, counter, nb))

    return PathResult(waypoints=[], cost=math.inf,
                      nodes_expanded=nodes_expanded, reason="blocked")


__all__ = [
    "PathfinderConfig",
    "PathResult",
    "find_path",
    "is_passable",
    "is_solid",
    "is_hazard",
]
