# agents/__init__.py
"""
Agent registry. Build an agent by name with ``build_agent("navigation", settings)``.

New agents should:
  1. Live in their own module under ``agents/``.
  2. Subclass ``brain.interfaces.BaseAgent``.
  3. Expose a top-level ``build_<name>_agent(settings)`` factory.
  4. Register here via ``_REGISTRY``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from brain.interfaces import BaseAgent
from agents.navigation import build_navigation_agent
from agents.walker import build_pathwalker_agent
from agents.world_explorer import build_world_explorer_agent


_REGISTRY: Dict[str, Callable[[Dict[str, Any]], BaseAgent]] = {
    "navigation":      build_navigation_agent,
    "pathwalker":      build_pathwalker_agent,
    "world_explorer":  build_world_explorer_agent,
}


def build_agent(name: str, settings: Dict[str, Any]) -> BaseAgent:
    """Construct an agent by name. Raises ValueError for unknown names."""
    factory = _REGISTRY.get(name)
    if factory is None:
        raise ValueError(
            f"Unknown agent {name!r}. Known: {sorted(_REGISTRY)}"
        )
    return factory(settings)


def available_agents() -> list:
    return sorted(_REGISTRY)
