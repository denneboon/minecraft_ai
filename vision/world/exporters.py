# vision/world/exporters.py
"""
Serialise the :class:`WorldMap` to formats other tools can read.

Two output formats
------------------
1. **Compact JSON** (``.json``). Palette-deduped, sparse-coordinate
   representation — one entry per known voxel as a 4-tuple
   ``[x, y, z, palette_index]``. Self-documenting, trivial to parse
   from any language, ~40-60 bytes per confirmed voxel.

2. **Sponge Schematic v2** (``.schem``). Gzipped NBT, the modern
   de-facto standard for Minecraft build-sharing. Loadable by:

   * **Amulet Editor** — standalone 3D viewer / editor with full
     game textures. https://amuletmc.com  ← best for offline
     exploration of what the AI mapped.
   * **Litematica** — Fabric/Forge mod, overlays the schematic
     in-game with proper lighting, biome tint, animated textures.
   * WorldEdit, MCEdit Unified, Mineways, Amulet Map Editor.

   Reference for the spec we follow:
   https://github.com/SpongePowered/Schematic-Specification/blob/master/versions/schematic-2.md

Implementation choices
----------------------
* Pure-Python NBT writer + reader (no third-party deps). The writer
  supports every tag type the spec uses plus IntArray / LongArray
  / Float / Double for future schematic versions. The reader is
  used by the test suite to round-trip-verify what we write.
* ``DataVersion`` is configurable and defaults to a recent 1.21.x
  value. It is also OPTIONAL per the Sponge v2 spec — pass
  ``data_version=None`` to omit it entirely if you don't know it.
* ``Offset`` is correctly emitted as ``TAG_Int_Array`` (length 3),
  NOT ``TAG_List<TAG_Int>``. The latter is what an earlier version
  of this module wrote and what most online schematic-format
  tutorials get wrong — Amulet and Litematica silently fall back
  to (0, 0, 0) when they see the wrong tag type.
* Block IDs are emitted as bare ``minecraft:id`` strings (default
  block state). When we eventually carry block-state info in the
  WorldMap, we can switch to the bracketed form ``namespace:id[k=v]``
  without changing the schematic structure.
* Air voxels in the carved-sightline sentinel set are EXCLUDED
  from the schematic by default; only solid blocks contribute to
  the palette + block data. Pass ``include_carved_air=True`` to
  preserve them.
"""

from __future__ import annotations

import gzip
import io
import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from vision.world.map import AIR_BLOCK, WorldMap


# ---------------------------------------------------------------------------
# Compact JSON
# ---------------------------------------------------------------------------

COMPACT_JSON_FORMAT = "minecraft_ai_world_v1"


def world_map_to_compact_dict(world_map: WorldMap,
                              *,
                              dimension: Optional[str] = None,
                              include_carved_air: bool = False,
                              include_metadata: bool = True,
                              ) -> Dict[str, Any]:
    """
    Build a compact dict representation of ``world_map``.

    Per voxel: one entry in ``blocks`` formatted as
    ``[x, y, z, palette_index]``. The palette is a list of unique
    block ids. The carved-air sightline sentinel is excluded
    unless ``include_carved_air`` is True.
    """
    dim = dimension or world_map.current_dimension()
    pal: Dict[str, int] = {}
    rows: List[List[int]] = []
    for obs in world_map.iter_blocks(dimension=dim):
        if obs.block_id is None:
            continue
        if obs.block_id == AIR_BLOCK and not include_carved_air:
            continue
        idx = pal.get(obs.block_id)
        if idx is None:
            idx = len(pal)
            pal[obs.block_id] = idx
        rows.append([int(obs.pos[0]), int(obs.pos[1]),
                     int(obs.pos[2]), int(idx)])

    palette: List[Optional[str]] = [None] * len(pal)
    for bid, idx in pal.items():
        palette[idx] = bid

    out: Dict[str, Any] = {
        "format":    COMPACT_JSON_FORMAT,
        "dimension": dim,
        "palette":   palette,
        "blocks":    rows,
    }
    if include_metadata:
        out["counts"] = {
            "solid":  sum(1 for o in world_map.iter_solid_blocks(dimension=dim)),
            "total":  sum(1 for _ in world_map.iter_blocks(dimension=dim)),
        }
    return out


def write_compact_json(world_map: WorldMap,
                        path: Union[str, Path],
                        *,
                        dimension: Optional[str] = None,
                        ) -> Path:
    """Write the compact JSON. Returns the path written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = world_map_to_compact_dict(world_map, dimension=dimension)
    # ``separators`` minimises whitespace for further size reduction.
    p.write_text(json.dumps(data, separators=(",", ":")),
                 encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# NBT tag-id constants (Java NBT spec)
# ---------------------------------------------------------------------------

TAG_END         = 0
TAG_BYTE        = 1
TAG_SHORT       = 2
TAG_INT         = 3
TAG_LONG        = 4
TAG_FLOAT       = 5
TAG_DOUBLE      = 6
TAG_BYTE_ARRAY  = 7
TAG_STRING      = 8
TAG_LIST        = 9
TAG_COMPOUND    = 10
TAG_INT_ARRAY   = 11
TAG_LONG_ARRAY  = 12


# ---------------------------------------------------------------------------
# NBT value wrappers
# ---------------------------------------------------------------------------
# Python int / str / dict / bytes map onto NBT primitives by default
# (Int / String / Compound / ByteArray). For everything else — Byte,
# Short, Long, Float, Double, IntArray, LongArray — wrap the value in
# the appropriate typed class so the writer knows which tag to emit.

class _TypedTag:
    """Marker base — never instantiated directly."""
    __slots__ = ("value",)
    tag_id: int

    def __init__(self, value: Any) -> None:
        self.value = value


class Byte(_TypedTag):       tag_id = TAG_BYTE
class Short(_TypedTag):      tag_id = TAG_SHORT
class Int(_TypedTag):        tag_id = TAG_INT
class Long(_TypedTag):       tag_id = TAG_LONG
class Float(_TypedTag):      tag_id = TAG_FLOAT
class Double(_TypedTag):     tag_id = TAG_DOUBLE
class ByteArray(_TypedTag):  tag_id = TAG_BYTE_ARRAY
class IntArray(_TypedTag):   tag_id = TAG_INT_ARRAY
class LongArray(_TypedTag):  tag_id = TAG_LONG_ARRAY


# A "typed" list — useful when the list element type isn't obvious
# from its first item (e.g. a list of empty compounds).
class TypedList:
    __slots__ = ("element_tag_id", "items")
    tag_id = TAG_LIST

    def __init__(self, element_tag_id: int, items):
        self.element_tag_id = int(element_tag_id)
        self.items = list(items)


# ---------------------------------------------------------------------------
# NBT writer (Java big-endian)
# ---------------------------------------------------------------------------

class NBTWriter:
    def __init__(self) -> None:
        self.buf = io.BytesIO()

    def write_named(self, name: str, value: Any) -> None:
        """Write a top-level named tag (typical NBT file root)."""
        tag_id, payload = self._classify(value)
        self._write_byte(tag_id)
        self._write_string(name)
        self._write_payload(tag_id, payload)

    def getvalue(self) -> bytes:
        return self.buf.getvalue()

    # ── Type classification ───────────────────────────────────────

    def _classify(self, value: Any) -> Tuple[int, Any]:
        """Return (tag_id, unwrapped_payload)."""
        if isinstance(value, _TypedTag):
            return value.tag_id, value.value
        if isinstance(value, TypedList):
            return TAG_LIST, value
        if isinstance(value, bool):
            return TAG_BYTE, int(value)
        if isinstance(value, int):
            return TAG_INT, value
        if isinstance(value, float):
            return TAG_DOUBLE, value
        if isinstance(value, str):
            return TAG_STRING, value
        if isinstance(value, dict):
            return TAG_COMPOUND, value
        if isinstance(value, (bytes, bytearray, memoryview)):
            return TAG_BYTE_ARRAY, bytes(value)
        if isinstance(value, (list, tuple)):
            return TAG_LIST, list(value)
        raise TypeError(f"Unsupported NBT value: {type(value).__name__}")

    # ── Payload writers ───────────────────────────────────────────

    def _write_payload(self, tag_id: int, payload: Any) -> None:
        if tag_id == TAG_BYTE:        self._write_byte(payload)
        elif tag_id == TAG_SHORT:     self._write_short(payload)
        elif tag_id == TAG_INT:       self._write_int(payload)
        elif tag_id == TAG_LONG:      self._write_long(payload)
        elif tag_id == TAG_FLOAT:     self._write_float(payload)
        elif tag_id == TAG_DOUBLE:    self._write_double(payload)
        elif tag_id == TAG_BYTE_ARRAY:self._write_byte_array(payload)
        elif tag_id == TAG_STRING:    self._write_string(payload)
        elif tag_id == TAG_LIST:      self._write_list(payload)
        elif tag_id == TAG_COMPOUND:  self._write_compound(payload)
        elif tag_id == TAG_INT_ARRAY: self._write_int_array(payload)
        elif tag_id == TAG_LONG_ARRAY:self._write_long_array(payload)
        else:
            raise ValueError(f"unsupported tag id {tag_id}")

    def _write_byte(self, v: int) -> None:
        self.buf.write(struct.pack(">b", _signed8(int(v))))

    def _write_short(self, v: int) -> None:
        self.buf.write(struct.pack(">h", int(v)))

    def _write_int(self, v: int) -> None:
        self.buf.write(struct.pack(">i", int(v)))

    def _write_long(self, v: int) -> None:
        self.buf.write(struct.pack(">q", int(v)))

    def _write_float(self, v: float) -> None:
        self.buf.write(struct.pack(">f", float(v)))

    def _write_double(self, v: float) -> None:
        self.buf.write(struct.pack(">d", float(v)))

    def _write_string(self, s: str) -> None:
        b = s.encode("utf-8")
        self.buf.write(struct.pack(">H", len(b)))
        self.buf.write(b)

    def _write_byte_array(self, b: bytes) -> None:
        self.buf.write(struct.pack(">i", len(b)))
        self.buf.write(b)

    def _write_int_array(self, ints) -> None:
        seq = list(ints)
        self.buf.write(struct.pack(">i", len(seq)))
        for v in seq:
            self.buf.write(struct.pack(">i", int(v)))

    def _write_long_array(self, longs) -> None:
        seq = list(longs)
        self.buf.write(struct.pack(">i", len(seq)))
        for v in seq:
            self.buf.write(struct.pack(">q", int(v)))

    def _write_compound(self, d: Dict[str, Any]) -> None:
        for name, value in d.items():
            tag_id, payload = self._classify(value)
            self._write_byte(tag_id)
            self._write_string(name)
            self._write_payload(tag_id, payload)
        # End-of-compound sentinel.
        self._write_byte(TAG_END)

    def _write_list(self, value: Any) -> None:
        if isinstance(value, TypedList):
            elem_id = value.element_tag_id
            items = value.items
        else:
            items = list(value)
            if not items:
                # Empty list — element type defaults to TAG_BYTE,
                # which the spec allows.
                elem_id = TAG_BYTE
            else:
                first_id, _ = self._classify(items[0])
                elem_id = first_id
        self._write_byte(elem_id)
        self._write_int(len(items))
        for it in items:
            tag_id, payload = self._classify(it)
            if tag_id != elem_id:
                raise ValueError(
                    f"NBT TAG_List items must share tag id; "
                    f"got {tag_id} in a list of {elem_id}"
                )
            self._write_payload(elem_id, payload)


def _signed8(v: int) -> int:
    """Coerce arbitrary int to a signed 8-bit int (-128..127)."""
    v &= 0xFF
    return v - 256 if v >= 128 else v


# ---------------------------------------------------------------------------
# NBT reader (used by tests + tools)
# ---------------------------------------------------------------------------

class NBTReader:
    """Lazy NBT parser. Use ``parse_named`` to read a top-level
    named tag (the typical NBT file layout)."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def parse_named(self) -> Tuple[str, Any]:
        tag_id = self._read_byte_unsigned()
        if tag_id == TAG_END:
            return "", None
        name = self._read_string()
        value = self._read_payload(tag_id)
        return name, value

    # ── Primitives ────────────────────────────────────────────────

    def _read_byte_signed(self) -> int:
        (v,) = struct.unpack_from(">b", self.data, self.pos); self.pos += 1
        return v

    def _read_byte_unsigned(self) -> int:
        (v,) = struct.unpack_from(">B", self.data, self.pos); self.pos += 1
        return v

    def _read_short(self) -> int:
        (v,) = struct.unpack_from(">h", self.data, self.pos); self.pos += 2
        return v

    def _read_int(self) -> int:
        (v,) = struct.unpack_from(">i", self.data, self.pos); self.pos += 4
        return v

    def _read_long(self) -> int:
        (v,) = struct.unpack_from(">q", self.data, self.pos); self.pos += 8
        return v

    def _read_float(self) -> float:
        (v,) = struct.unpack_from(">f", self.data, self.pos); self.pos += 4
        return v

    def _read_double(self) -> float:
        (v,) = struct.unpack_from(">d", self.data, self.pos); self.pos += 8
        return v

    def _read_string(self) -> str:
        n = struct.unpack_from(">H", self.data, self.pos)[0]; self.pos += 2
        s = self.data[self.pos:self.pos + n].decode("utf-8")
        self.pos += n
        return s

    def _read_byte_array(self) -> bytes:
        n = self._read_int()
        b = bytes(self.data[self.pos:self.pos + n])
        self.pos += n
        return b

    def _read_int_array(self) -> List[int]:
        n = self._read_int()
        out = list(struct.unpack_from(f">{n}i", self.data, self.pos))
        self.pos += n * 4
        return out

    def _read_long_array(self) -> List[int]:
        n = self._read_int()
        out = list(struct.unpack_from(f">{n}q", self.data, self.pos))
        self.pos += n * 8
        return out

    def _read_compound(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        while True:
            tag_id = self._read_byte_unsigned()
            if tag_id == TAG_END:
                return out
            name = self._read_string()
            out[name] = self._read_payload(tag_id)

    def _read_list(self) -> List[Any]:
        elem_id = self._read_byte_unsigned()
        n = self._read_int()
        return [self._read_payload(elem_id) for _ in range(n)]

    def _read_payload(self, tag_id: int) -> Any:
        if tag_id == TAG_BYTE:        return self._read_byte_signed()
        if tag_id == TAG_SHORT:       return self._read_short()
        if tag_id == TAG_INT:         return self._read_int()
        if tag_id == TAG_LONG:        return self._read_long()
        if tag_id == TAG_FLOAT:       return self._read_float()
        if tag_id == TAG_DOUBLE:      return self._read_double()
        if tag_id == TAG_BYTE_ARRAY:  return self._read_byte_array()
        if tag_id == TAG_STRING:      return self._read_string()
        if tag_id == TAG_LIST:        return self._read_list()
        if tag_id == TAG_COMPOUND:    return self._read_compound()
        if tag_id == TAG_INT_ARRAY:   return self._read_int_array()
        if tag_id == TAG_LONG_ARRAY:  return self._read_long_array()
        raise ValueError(f"unknown tag id {tag_id} at pos {self.pos}")


def read_nbt(path: Union[str, Path]) -> Tuple[str, Any]:
    """Convenience: gunzip + parse an NBT file. Returns (name, root)."""
    p = Path(path)
    with gzip.open(p, "rb") as f:
        data = f.read()
    return NBTReader(data).parse_named()


# ---------------------------------------------------------------------------
# Varint encoding (Sponge schematic block data)
# ---------------------------------------------------------------------------

def _encode_varint(n: int) -> bytes:
    """Unsigned LEB128, matches Sponge schematic's varint format."""
    if n < 0:
        raise ValueError("varint must be non-negative")
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _decode_varints(data: bytes) -> List[int]:
    """Inverse of :func:`_encode_varint`. Used by tests."""
    out: List[int] = []
    pos = 0
    while pos < len(data):
        shift = 0
        v = 0
        while True:
            b = data[pos]; pos += 1
            v |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        out.append(v)
    return out


# ---------------------------------------------------------------------------
# Block id validation
# ---------------------------------------------------------------------------
# Modern MC accepts block ids matching:
#   ^[a-z0-9_-]+:[a-z0-9_./-]+$  (namespace : path)
# Anything else risks Amulet / Litematica rejecting the file.

_BID_RE = None  # lazy-init


def _valid_block_id(bid: str) -> bool:
    global _BID_RE
    if _BID_RE is None:
        import re as _re
        _BID_RE = _re.compile(r"^[a-z0-9_\-]+:[a-z0-9_./\-]+$")
    return bool(bid) and bool(_BID_RE.match(bid))


# ---------------------------------------------------------------------------
# Default Minecraft data version
# ---------------------------------------------------------------------------
# https://minecraft.wiki/w/Data_version
# Latest stable 1.21.x data versions:
#   1.21    -> 3953
#   1.21.1  -> 3955
#   1.21.2  -> 4080
#   1.21.3  -> 4082
#   1.21.4  -> 4189
#   1.21.5  -> 4324
# Tools that respect DataVersion use it to know which block IDs and
# states are valid; using something older than the target version is
# safe (forward compatibility), but using something NEWER than the
# target may make older tools reject the schematic. 1.21 is a good
# default since the user's Prism instance reports 1.21.x.

DEFAULT_DATA_VERSION = 3953   # MC 1.21


# ---------------------------------------------------------------------------
# Sponge Schematic v2 writer
# ---------------------------------------------------------------------------

def write_schematic(world_map: WorldMap,
                     path: Union[str, Path],
                     *,
                     dimension: Optional[str] = None,
                     data_version: Optional[int] = DEFAULT_DATA_VERSION,
                     include_carved_air: bool = False,
                     metadata_name: str = "MinecraftAI WorldMap",
                     ) -> Path:
    """
    Write the WorldMap as a Sponge Schematic v2 file (``.schem``).

    Spans the tight bounding box of the chosen voxels in
    ``dimension`` (Y up). Voxels inside the box that the AI hasn't
    observed are filled with ``minecraft:air`` (the schematic spec
    requires a value at every index).

    Parameters
    ----------
    dimension
        Which WorldMap dimension to export. Defaults to the map's
        current dimension.
    data_version
        Minecraft DataVersion to record. Pass ``None`` to omit the
        field entirely (Amulet / Litematica fall back to a sensible
        default in that case).
    include_carved_air
        Whether to include the ``minecraft:air`` sentinels that the
        perception layer carves along F3 sightlines. Default False —
        most users only care about solid blocks.
    metadata_name
        Stored in the ``Metadata.Name`` tag. Tools display this as
        the schematic's friendly name.
    """
    dim = dimension or world_map.current_dimension()

    blocks_by_pos: Dict[Tuple[int, int, int], str] = {}
    for obs in world_map.iter_blocks(dimension=dim):
        if obs.block_id is None:
            continue
        if obs.block_id == AIR_BLOCK and not include_carved_air:
            continue
        if not _valid_block_id(obs.block_id):
            # Skip ids that wouldn't round-trip through MC.
            continue
        blocks_by_pos[(int(obs.pos[0]), int(obs.pos[1]),
                       int(obs.pos[2]))] = obs.block_id

    if not blocks_by_pos:
        return _write_schematic_empty(path,
                                       data_version=data_version,
                                       metadata_name=metadata_name)

    xs = [p[0] for p in blocks_by_pos]
    ys = [p[1] for p in blocks_by_pos]
    zs = [p[2] for p in blocks_by_pos]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)
    width  = (x_max - x_min) + 1
    height = (y_max - y_min) + 1
    length = (z_max - z_min) + 1

    # Per spec, every voxel must be assigned an entry from Palette.
    # Air at index 0 is the default fill for unobserved voxels.
    palette: Dict[str, int] = {"minecraft:air": 0}
    for bid in blocks_by_pos.values():
        if bid not in palette:
            palette[bid] = len(palette)

    # BlockData index order: (Y, Z, X) — index = (y*Length + z)*Width + x.
    air_idx = palette["minecraft:air"]
    out = bytearray()
    for y in range(height):
        for z in range(length):
            for x in range(width):
                bid = blocks_by_pos.get((x + x_min, y + y_min, z + z_min))
                idx = palette[bid] if bid is not None else air_idx
                out.extend(_encode_varint(idx))

    # Compose the schematic compound.
    pal_compound = {bid: Int(idx) for bid, idx in palette.items()}
    metadata: Dict[str, Any] = {"Name": metadata_name}

    schem: Dict[str, Any] = {
        "Version":     Int(2),
        "Width":       Short(width),
        "Height":      Short(height),
        "Length":      Short(length),
        # ``Offset`` MUST be a TAG_Int_Array of length 3 per Sponge v2
        # spec; tools silently fall back to (0, 0, 0) if it's not.
        "Offset":      IntArray([x_min, y_min, z_min]),
        "PaletteMax":  Int(len(palette)),
        "Palette":     pal_compound,
        "BlockData":   ByteArray(bytes(out)),
        "Metadata":    metadata,
    }
    if data_version is not None:
        schem["DataVersion"] = Int(int(data_version))

    w = NBTWriter()
    w.write_named("Schematic", schem)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wb") as f:
        f.write(w.getvalue())
    return p


def _write_schematic_empty(path: Union[str, Path],
                            *,
                            data_version: Optional[int] = DEFAULT_DATA_VERSION,
                            metadata_name: str = "MinecraftAI WorldMap (empty)",
                            ) -> Path:
    """1×1×1 air schematic. Useful as a placeholder for empty maps."""
    schem: Dict[str, Any] = {
        "Version":     Int(2),
        "Width":       Short(1),
        "Height":      Short(1),
        "Length":      Short(1),
        "Offset":      IntArray([0, 0, 0]),
        "PaletteMax":  Int(1),
        "Palette":     {"minecraft:air": Int(0)},
        "BlockData":   ByteArray(_encode_varint(0)),
        "Metadata":    {"Name": metadata_name},
    }
    if data_version is not None:
        schem["DataVersion"] = Int(int(data_version))
    w = NBTWriter()
    w.write_named("Schematic", schem)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wb") as f:
        f.write(w.getvalue())
    return p


# ---------------------------------------------------------------------------
# Convenience: read a .schem and return its parsed structure
# ---------------------------------------------------------------------------

def read_schematic(path: Union[str, Path]) -> Dict[str, Any]:
    """Round-trip helper. Returns the parsed Schematic compound."""
    name, root = read_nbt(path)
    if name != "Schematic":
        raise ValueError(f"not a Sponge schematic — root tag is {name!r}")
    return root


__all__ = [
    "COMPACT_JSON_FORMAT", "DEFAULT_DATA_VERSION",
    "world_map_to_compact_dict", "write_compact_json",
    "write_schematic", "read_schematic",
    # NBT helpers (exposed for tests / advanced tooling)
    "Byte", "Short", "Int", "Long", "Float", "Double",
    "ByteArray", "IntArray", "LongArray", "TypedList",
    "NBTWriter", "NBTReader", "read_nbt",
]
