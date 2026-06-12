# vision/inventory_layout.py
"""
Slot-rect geometry for every Minecraft container screen.

Why a separate module
---------------------
The player inventory layout used to live as private constants inside
``vision/inventory.py``. That worked while we only ever read the player
inventory, but as soon as the AI needs to interact with a chest, a
furnace, a crafting table, a brewing stand, etc. — each has its own
slot grid drawn on its own background-sprite size — we need a single
source of truth that covers all of them.

What this module does
---------------------
* Defines the GUI-scale-1 (x, y) offsets of every slot for each
  container kind. Numbers come straight out of Mojang's vanilla GUI
  classes (``InventoryScreen``, ``ChestScreen``, ``CraftingScreen``,
  ``FurnaceScreen``, ``BrewingStandScreen``, …), which haven't changed
  in many releases. If they ever do change, only the per-container
  tables below need updating — every downstream consumer (the item
  recognizer, the count OCR, the high-level snapshot) stays untouched.
* Converts a chosen layout + a captured frame's resolution + the
  current GUI scale into a dict of ``SlotRect(x, y, w, h)`` in screen
  pixels, ready to crop.

Containers covered today
------------------------
``player_inventory``   — vanilla survival inventory (E key)
``creative_inventory`` — placeholder; creative has a tab strip we'll
                         add when we need it
``chest_single``       — 9×3 container + 9×3 main + 9 hotbar
``chest_double``       — 9×6 container + 9×3 main + 9 hotbar
``crafting_table``     — 3×3 craft grid + 1 result + 9×3 main + 9 hotbar
``furnace``            — 1 input + 1 fuel + 1 result + 9×3 main + 9 hotbar
``smoker``             — same as furnace
``blast_furnace``      — same as furnace
``brewing_stand``      — 1 ingredient + 1 fuel + 3 potions + 9×3 + 9
``hopper``             — 1×5 + 9×3 + 9
``shulker_box``        — 9×3 + 9×3 + 9 (same as chest_single)
``barrel``             — same as chest_single
``dispenser``          — 3×3 + 9×3 + 9
``dropper``            — same as dispenser

All layouts include the 27 main-inventory slots and 9 hotbar slots that
sit at the bottom of every GUI screen, so a single snapshot can read
everything the player can see.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Geometric primitives
# ---------------------------------------------------------------------------

_SLOT_W_GUI = 16
_SLOT_H_GUI = 16
_SLOT_PITCH_GUI = 18    # 16 + 1 + 1 px border


@dataclass(frozen=True)
class SlotRect:
    """A single slot's rect, in screen pixels."""
    name: str
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def center(self) -> Tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2


# ---------------------------------------------------------------------------
# Per-container layout descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContainerLayout:
    """
    Declarative description of one container screen.

    ``name``               human-readable id (also a stable key)
    ``bg_w_gui, bg_h_gui`` size of the centred background sprite
                           (in GUI px at scale 1)
    ``slot_offsets``       ordered list of (slot_name, (x, y)) at GUI
                           scale 1, relative to the background's top-left
    ``main_top_off``       (x, y) of the top-left main-inventory slot;
                           ``None`` if this container has no player main
                           grid (e.g. some custom screens)
    ``hotbar_off``         (x, y) of the leftmost hotbar slot
    """
    name: str
    bg_w_gui: int
    bg_h_gui: int
    slot_offsets: Tuple[Tuple[str, Tuple[int, int]], ...]
    main_top_off: Optional[Tuple[int, int]] = (8, 84)
    hotbar_off:   Optional[Tuple[int, int]] = (8, 142)


def _grid(prefix: str,
          top_left: Tuple[int, int],
          rows: int,
          cols: int,
          pitch: int = _SLOT_PITCH_GUI,
          ) -> List[Tuple[str, Tuple[int, int]]]:
    """Generate a rows×cols grid of slot offsets named ``prefix_<i>``."""
    x0, y0 = top_left
    out: List[Tuple[str, Tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols):
            out.append((f"{prefix}_{r * cols + c}",
                        (x0 + c * pitch, y0 + r * pitch)))
    return out


# ---------------------------------------------------------------------------
# Layout tables — values from Mojang's GUI .png coordinates
# ---------------------------------------------------------------------------

# Player survival inventory (E):
#   * 4 armour slots (8, 8) → (8, 62), stride 18
#   * 1 off-hand at (77, 62)
#   * 2×2 craft grid at (98, 18)/(116, 18)/(98, 36)/(116, 36)
#   * 1 craft result at (154, 28)
#   * main 3×9 at (8, 84)
#   * hotbar 1×9 at (8, 142)
_PLAYER_INVENTORY = ContainerLayout(
    name="player_inventory",
    bg_w_gui=176, bg_h_gui=166,
    slot_offsets=tuple([
        ("armor_head",   ( 8,  8)),
        ("armor_chest",  ( 8, 26)),
        ("armor_legs",   ( 8, 44)),
        ("armor_feet",   ( 8, 62)),
        ("offhand",      (77, 62)),
        ("craft_in_0",   ( 98, 18)),
        ("craft_in_1",   (116, 18)),
        ("craft_in_2",   ( 98, 36)),
        ("craft_in_3",   (116, 36)),
        ("craft_result", (154, 28)),
    ]),
)

# Single chest / barrel / shulker:
#   * container 3×9 at (8, 18)
#   * main 3×9 at (8, 84)
#   * hotbar 1×9 at (8, 142)
_CHEST_SINGLE = ContainerLayout(
    name="chest_single",
    bg_w_gui=176, bg_h_gui=166,
    slot_offsets=tuple(_grid("chest", (8, 18), 3, 9)),
)

# Double chest:
#   * container 6×9 at (8, 18)
#   * main 3×9 at (8, 138)
#   * hotbar 1×9 at (8, 196)
_CHEST_DOUBLE = ContainerLayout(
    name="chest_double",
    bg_w_gui=176, bg_h_gui=222,
    slot_offsets=tuple(_grid("chest", (8, 18), 6, 9)),
    main_top_off=(8, 138),
    hotbar_off=(8, 196),
)

# Crafting table:
#   * 3×3 input grid at (30, 17), stride 18
#   * 1 result at (124, 35)
#   * main 3×9 at (8, 84)
#   * hotbar 1×9 at (8, 142)
_CRAFTING_TABLE = ContainerLayout(
    name="crafting_table",
    bg_w_gui=176, bg_h_gui=166,
    slot_offsets=tuple(
        _grid("craft_in", (30, 17), 3, 3) + [("craft_result", (124, 35))]
    ),
)

# Furnace / smoker / blast furnace:
#   * input at  (56, 17)
#   * fuel  at  (56, 53)
#   * result at (116, 35)
_FURNACE = ContainerLayout(
    name="furnace",
    bg_w_gui=176, bg_h_gui=166,
    slot_offsets=(
        ("furnace_input",  (56, 17)),
        ("furnace_fuel",   (56, 53)),
        ("furnace_result", (116, 35)),
    ),
)
_SMOKER = ContainerLayout(
    name="smoker",
    bg_w_gui=_FURNACE.bg_w_gui, bg_h_gui=_FURNACE.bg_h_gui,
    slot_offsets=_FURNACE.slot_offsets,
)
_BLAST_FURNACE = ContainerLayout(
    name="blast_furnace",
    bg_w_gui=_FURNACE.bg_w_gui, bg_h_gui=_FURNACE.bg_h_gui,
    slot_offsets=_FURNACE.slot_offsets,
)

# Brewing stand:
#   * ingredient at (79, 17)
#   * fuel       at (17, 17)
#   * potions at (56, 51), (79, 58), (102, 51)
_BREWING_STAND = ContainerLayout(
    name="brewing_stand",
    bg_w_gui=176, bg_h_gui=166,
    slot_offsets=(
        ("brew_ingredient", (79, 17)),
        ("brew_fuel",       (17, 17)),
        ("brew_potion_0",   (56, 51)),
        ("brew_potion_1",   (79, 58)),
        ("brew_potion_2",   (102, 51)),
    ),
)

# Hopper: 1×5 at (44, 20)
_HOPPER = ContainerLayout(
    name="hopper",
    bg_w_gui=176, bg_h_gui=133,
    slot_offsets=tuple(_grid("hopper", (44, 20), 1, 5)),
    main_top_off=(8, 51),
    hotbar_off=(8, 109),
)

# Dispenser / dropper: 3×3 at (62, 17)
_DISPENSER = ContainerLayout(
    name="dispenser",
    bg_w_gui=176, bg_h_gui=166,
    slot_offsets=tuple(_grid("disp", (62, 17), 3, 3)),
)
_DROPPER = ContainerLayout(
    name="dropper",
    bg_w_gui=_DISPENSER.bg_w_gui, bg_h_gui=_DISPENSER.bg_h_gui,
    slot_offsets=_DISPENSER.slot_offsets,
)

# Aliases
_SHULKER_BOX = ContainerLayout(
    name="shulker_box",
    bg_w_gui=_CHEST_SINGLE.bg_w_gui, bg_h_gui=_CHEST_SINGLE.bg_h_gui,
    slot_offsets=_CHEST_SINGLE.slot_offsets,
)
_BARREL = ContainerLayout(
    name="barrel",
    bg_w_gui=_CHEST_SINGLE.bg_w_gui, bg_h_gui=_CHEST_SINGLE.bg_h_gui,
    slot_offsets=_CHEST_SINGLE.slot_offsets,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_LAYOUTS: Dict[str, ContainerLayout] = {
    L.name: L for L in (
        _PLAYER_INVENTORY,
        _CHEST_SINGLE,
        _CHEST_DOUBLE,
        _CRAFTING_TABLE,
        _FURNACE, _SMOKER, _BLAST_FURNACE,
        _BREWING_STAND,
        _HOPPER,
        _DISPENSER, _DROPPER,
        _SHULKER_BOX, _BARREL,
    )
}


def available_layouts() -> List[str]:
    return sorted(_LAYOUTS.keys())


def get_layout(name: str) -> ContainerLayout:
    if name not in _LAYOUTS:
        raise KeyError(f"Unknown container layout {name!r}. "
                       f"Known: {available_layouts()}")
    return _LAYOUTS[name]


# ---------------------------------------------------------------------------
# Slot-rect computation
# ---------------------------------------------------------------------------

def _flatten(layout: ContainerLayout) -> List[Tuple[str, Tuple[int, int]]]:
    """Concatenate the layout's container slots + main + hotbar offsets."""
    out: List[Tuple[str, Tuple[int, int]]] = list(layout.slot_offsets)

    if layout.main_top_off is not None:
        x0, y0 = layout.main_top_off
        for r in range(3):
            for c in range(9):
                out.append((f"inv_{r * 9 + c}",
                            (x0 + c * _SLOT_PITCH_GUI,
                             y0 + r * _SLOT_PITCH_GUI)))

    if layout.hotbar_off is not None:
        hx, hy = layout.hotbar_off
        for i in range(9):
            out.append((f"hotbar_{i}", (hx + i * _SLOT_PITCH_GUI, hy)))

    return out


def slot_rects(frame_shape: Tuple[int, int],
               *,
               layout: str = "player_inventory",
               ui_scale: int = 2) -> Dict[str, SlotRect]:
    """
    Compute the screen-pixel rect of every slot for the given container.

    Parameters
    ----------
    frame_shape : (H, W) or (H, W, C) of the captured frame.
    layout      : key into the layout registry (default: player_inventory).
    ui_scale    : Minecraft's current GUI scale (1..4 in vanilla).
                  Slot pixel dimensions scale by this factor.

    Returns
    -------
    Dict mapping slot-name → ``SlotRect``. The slot names are stable
    across versions and chosen for readability:

      * ``armor_head`` / ``armor_chest`` / ``armor_legs`` / ``armor_feet``
      * ``offhand``
      * ``craft_in_<n>``  (0-indexed; row-major)
      * ``craft_result``
      * ``chest_<n>``     (0-indexed; row-major; chest layouts)
      * ``furnace_input`` / ``furnace_fuel`` / ``furnace_result``
      * ``brew_*``
      * ``hopper_<n>`` / ``disp_<n>``
      * ``inv_<n>``       (0-indexed; 27 main inventory)
      * ``hotbar_<n>``    (0-indexed; 9 hotbar)
    """
    H, W = frame_shape[:2]
    spec = get_layout(layout)
    scale = max(1, int(ui_scale))

    bg_w = spec.bg_w_gui * scale
    bg_h = spec.bg_h_gui * scale
    bg_x = (W - bg_w) // 2
    bg_y = (H - bg_h) // 2
    sw   = _SLOT_W_GUI * scale
    sh   = _SLOT_H_GUI * scale

    out: Dict[str, SlotRect] = {}
    for name, (ox, oy) in _flatten(spec):
        out[name] = SlotRect(
            name=name,
            x=bg_x + ox * scale,
            y=bg_y + oy * scale,
            w=sw,
            h=sh,
        )
    return out


def hud_hotbar_rects(frame_shape: Tuple[int, int],
                     ui_scale: int = 2) -> Dict[str, SlotRect]:
    """Screen rects of the 9 GAMEPLAY HUD hotbar slots (inventory CLOSED) —
    the always-visible bar at the bottom of the screen, so the bot can read
    what it's carrying in the hotbar without opening anything.

    Geometry differs from the inventory's hotbar row: the HUD widget is the
    182×22 ``hotbar`` sprite, centred horizontally and anchored to the screen
    bottom, with a 20-GUI-px slot pitch (not 18). Item i is drawn 3 px in from
    the widget's top-left, then +20 px each, at 16×16. Names ``hotbar_0..8``
    match the inventory layout so a snapshot reads the same downstream."""
    H, W = frame_shape[:2]
    s = max(1, int(ui_scale))
    widget_left = (W - 182 * s) // 2
    top = H - 22 * s + 3 * s            # widget bottom-anchored, item 3px inset
    sw = _SLOT_W_GUI * s
    sh = _SLOT_H_GUI * s
    out: Dict[str, SlotRect] = {}
    for i in range(9):
        out[f"hotbar_{i}"] = SlotRect(
            name=f"hotbar_{i}",
            x=widget_left + (3 + i * 20) * s,
            y=top, w=sw, h=sh,
        )
    return out


def background_rect(frame_shape: Tuple[int, int],
                    *,
                    layout: str = "player_inventory",
                    ui_scale: int = 2) -> Tuple[int, int, int, int]:
    """
    Return (x, y, w, h) of the container's background sprite on-screen.
    Useful for detection (does the centre look like our expected layout?)
    and for menu type inference.
    """
    H, W = frame_shape[:2]
    spec = get_layout(layout)
    scale = max(1, int(ui_scale))
    bg_w = spec.bg_w_gui * scale
    bg_h = spec.bg_h_gui * scale
    bg_x = (W - bg_w) // 2
    bg_y = (H - bg_h) // 2
    return bg_x, bg_y, bg_w, bg_h


# ---------------------------------------------------------------------------
# Slot-group convenience constants
# ---------------------------------------------------------------------------

ARMOR_SLOTS:   Tuple[str, ...] = ("armor_head", "armor_chest",
                                  "armor_legs", "armor_feet")
HOTBAR_SLOTS:  Tuple[str, ...] = tuple(f"hotbar_{i}" for i in range(9))
MAIN_SLOTS:    Tuple[str, ...] = tuple(f"inv_{i}"    for i in range(27))
CRAFT_2x2:     Tuple[str, ...] = tuple(f"craft_in_{i}" for i in range(4))
CRAFT_3x3:     Tuple[str, ...] = tuple(f"craft_in_{i}" for i in range(9))


def slot_pitch_px(ui_scale: int = 2) -> int:
    """The full slot-to-slot stride in screen pixels."""
    return _SLOT_PITCH_GUI * max(1, int(ui_scale))


def slot_size_px(ui_scale: int = 2) -> int:
    """The interior icon size in screen pixels."""
    return _SLOT_W_GUI * max(1, int(ui_scale))


__all__ = [
    "SlotRect", "ContainerLayout",
    "slot_rects", "background_rect",
    "available_layouts", "get_layout",
    "ARMOR_SLOTS", "HOTBAR_SLOTS", "MAIN_SLOTS",
    "CRAFT_2x2", "CRAFT_3x3",
    "slot_pitch_px", "slot_size_px",
]
