# config/__init__.py
"""
Helpers for loading the project's YAML / JSON configuration files.

The runtime config (``settings.yaml``) and the in-game settings mirror
(``minecraft.yaml``) live side-by-side here. This module exposes both
plus a tiny ``get`` helper for dotted lookups.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

try:
    import yaml
except ImportError:                     # PyYAML missing — degrade gracefully
    yaml = None                          # type: ignore


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CONFIG_DIR        = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH     = os.path.join(CONFIG_DIR, "settings.yaml")
MINECRAFT_PATH    = os.path.join(CONFIG_DIR, "minecraft.yaml")
KEYMAP_PATH       = os.path.join(CONFIG_DIR, "keymap.json")


# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file into a dict. Returns ``{}`` if missing or PyYAML
    isn't installed — callers are expected to provide their own defaults."""
    if yaml is None or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f) or {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Project-specific accessors
# ---------------------------------------------------------------------------

def load_settings() -> Dict[str, Any]:
    """Load ``config/settings.yaml`` (runtime config)."""
    return load_yaml(SETTINGS_PATH)


def load_minecraft_settings() -> Dict[str, Any]:
    """
    Load ``config/minecraft.yaml`` (mirror of the player's in-game options).

    Returns ``{}`` when the file is missing so callers can fall back to
    sensible defaults without crashing.
    """
    return load_yaml(MINECRAFT_PATH)


def load_keymap() -> Dict[str, Any]:
    """Load ``config/keymap.json``."""
    return load_json(KEYMAP_PATH)


# ---------------------------------------------------------------------------
# Dotted lookup
# ---------------------------------------------------------------------------

def get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Walk a dict by dotted ``path`` and return the value at the leaf, or
    ``default`` if any segment is missing.

    >>> get(cfg, "video.brightness", "moody")
    "bright"
    """
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


__all__ = [
    "CONFIG_DIR", "SETTINGS_PATH", "MINECRAFT_PATH", "KEYMAP_PATH",
    "load_yaml", "load_json",
    "load_settings", "load_minecraft_settings", "load_keymap",
    "get",
]
