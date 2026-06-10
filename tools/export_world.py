# tools/export_world.py
"""
Export a WorldMap snapshot to formats other tools can read.

The agent loop drops both a compact JSON and a .schem next to each
agent run, but if you want to re-export an older JSON or pick a
specific run from the history, use this CLI.

Recommended downstream tools
----------------------------
* **Amulet Editor** — standalone 3D viewer/editor with proper MC
  textures. https://amuletmc.com — drop the .schem onto its
  window and orbit the camera. Best for offline inspection.
* **Litematica** (Fabric/Forge mod) — overlays the schematic in
  the actual game with full lighting + biome tint + animated
  textures. https://www.curseforge.com/minecraft/mc-mods/litematica

Usage
-----
    python tools/export_world.py                  # latest dump → .schem
    python tools/export_world.py --json some.json --out my.schem
    python tools/export_world.py --list           # list known dumps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _latest_json() -> Path | None:
    """Find the most recent ``world_map_*.json`` in the calibration dir."""
    cal = ROOT / "data" / "calibration"
    files = sorted(cal.glob("world_map_*.json"))
    return files[-1] if files else None


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dict_to_world_map(data: dict):
    """Reconstruct a transient WorldMap from a compact JSON dict so
    the writer can re-emit other formats."""
    from vision.world import WorldMap, BlockObservation
    wm = WorldMap()
    dim = data.get("dimension", "minecraft:overworld")
    wm.set_current_dimension(dim)
    palette = data.get("palette", [])
    fmt = data.get("format")

    if fmt == "minecraft_ai_world_v1":
        rows = data.get("blocks", [])
        for row in rows:
            x, y, z, idx = row
            bid = palette[idx]
            wm.update_block(BlockObservation(
                pos=(int(x), int(y), int(z)),
                block_id=bid, confidence=1.0,
                source="manual", last_seen_tick=0,
                dimension=dim,
            ))
        return wm

    # Legacy dump (the old WorldMap.to_dict format).
    dims = data.get("dimensions", {})
    for d, sub in dims.items():
        for entry in sub.get("blocks", []):
            pos = tuple(entry["pos"])
            wm.update_block(BlockObservation(
                pos=pos, block_id=entry.get("block_id"),
                confidence=float(entry.get("confidence", 1.0)),
                source=entry.get("source", "manual"),
                last_seen_tick=int(entry.get("last_seen_tick", 0)),
                dimension=d,
            ))
    return wm


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", type=Path, default=None,
                   help="Path to the source JSON. Defaults to the "
                        "most recent world_map_*.json.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output .schem path. Defaults to the source "
                        "name with .schem extension.")
    p.add_argument("--list", action="store_true",
                   help="List the world_map_*.json files we know "
                        "about and exit.")
    args = p.parse_args(argv)

    if args.list:
        cal = ROOT / "data" / "calibration"
        files = sorted(cal.glob("world_map_*.json"))
        if not files:
            print("(no world_map_*.json found)")
            return 0
        for f in files:
            print(f)
        return 0

    src = args.json or _latest_json()
    if src is None or not src.is_file():
        print("[ERR] No source JSON found. Pass --json PATH.")
        return 2
    print(f"[..] loading {src}")
    data = _load_json(src)
    wm = _dict_to_world_map(data)
    n = sum(1 for _ in wm.iter_solid_blocks())
    print(f"[..] {n} solid blocks in dimension={wm.current_dimension()}")

    out = args.out or src.with_suffix(".schem")
    from vision.world import write_schematic
    written = write_schematic(wm, out)
    size = written.stat().st_size
    print(f"[ok] wrote {written}  ({size} bytes)")
    print("     open in Amulet Editor (https://amuletmc.com) or "
          "import via Litematica.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
