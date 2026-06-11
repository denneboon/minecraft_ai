# brain/episode_logger.py
"""
Episode logging — record every agent run as a stream of
(perception, action, outcome) records for later analysis and, eventually,
learning.

Why this exists
---------------
The architecture's whole point is that a rule-based ``Skill`` can later be
replaced by a *learned* policy behind the same ``AgentAction`` interface.
That swap needs data: what the agent SAW (pose, targeted block, HUD), what
it DID (the AgentAction), and how things turned out (agent telemetry +
events). That data can't be captured retroactively — so we log it on every
run, from now on, cheaply.

Format: one JSONL file per run under ``data/episodes/``. Each line is a
compact per-tick record; the first line is a ``meta`` header and the last
is a ``summary``. Keys are short to keep files small. Logging is
best-effort: a failure here must NEVER disturb the control loop.

This is generic (works for any BaseAgent). An agent may optionally expose a
``telemetry() -> dict`` method; if present, its output is attached to each
record under ``"ag"`` (e.g. the tree-chopper's FSM state + log count),
which turns a raw input/output trace into a *labelled* one.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


def _round(v, n=3):
    try:
        return round(float(v), n)
    except Exception:
        return None


def _pose_of(state) -> Optional[list]:
    """[x, y, z, yaw, pitch] from the world pose or F3, else None."""
    p = None
    world = getattr(state, "world", None)
    if world is not None:
        p = getattr(world, "pose", None)
    if p is None:
        p = getattr(state, "f3", None)
    if p is None:
        return None
    try:
        x, y, z = p.x, p.y, p.z
        if x is None or y is None or z is None:
            return None
        return [_round(x, 2), _round(y, 2), _round(z, 2),
                _round(getattr(p, "yaw", None), 1), _round(getattr(p, "pitch", None), 1)]
    except Exception:
        return None


def _looking_at(state) -> Optional[list]:
    world = getattr(state, "world", None)
    la = getattr(world, "looking_at", None) if world is not None else None
    if la is None:
        return None
    try:
        pos = list(la.pos) if getattr(la, "pos", None) is not None else None
        return [getattr(la, "block_id", None), pos]
    except Exception:
        return None


def _action_of(action) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        mv = {k: bool(v) for k, v in (action.movement or {}).items() if v}
        if mv:
            out["mv"] = mv
        if action.look_dx or action.look_dy:
            out["ld"] = [int(action.look_dx), int(action.look_dy)]
        if action.look_vx or action.look_vy:
            out["lv"] = [_round(action.look_vx, 1), _round(action.look_vy, 1)]
        if action.interact:
            out["int"] = action.interact
        if action.hotbar is not None:
            out["slot"] = int(action.hotbar)
        if getattr(action, "inventory_toggle", False):
            out["inv"] = True
    except Exception:
        pass
    return out


class EpisodeLogger:
    """Append-only JSONL episode recorder. Best-effort: never raises into
    the caller; if anything goes wrong it silently disables itself."""

    def __init__(self, out_dir: str, agent_name: str, *,
                 enabled: bool = True, started_ts: str = ""):
        self.enabled = bool(enabled)
        self._f = None
        self.path = None
        self.n = 0
        if not self.enabled:
            return
        try:
            os.makedirs(out_dir, exist_ok=True)
            stamp = started_ts or time.strftime("%Y%m%d_%H%M%S")
            self.path = os.path.join(out_dir, f"{agent_name}_{stamp}.jsonl")
            self._f = open(self.path, "w", encoding="utf-8")
            self._write({"type": "meta", "agent": agent_name, "stamp": stamp,
                         "schema": 1})
        except Exception:
            self.enabled = False
            self._f = None

    def _write(self, obj: Dict[str, Any]) -> None:
        if self._f is None:
            return
        try:
            self._f.write(json.dumps(obj, separators=(",", ":")) + "\n")
        except Exception:
            self.enabled = False

    def record(self, tick: int, t: float, state, action, agent=None,
               dispatched: bool = True) -> None:
        if not self.enabled or self._f is None:
            return
        try:
            rec: Dict[str, Any] = {"k": int(tick), "t": _round(t, 2)}
            pose = _pose_of(state)
            if pose is not None:
                rec["p"] = pose
            la = _looking_at(state)
            if la is not None:
                rec["look"] = la
            health = getattr(state, "health", None)
            hunger = getattr(state, "hunger", None)
            if health is not None or hunger is not None:
                rec["hud"] = [_round(health, 2), _round(hunger, 2)]
            ss = getattr(state, "screen_state", None)
            if ss is not None:
                rec["scr"] = getattr(ss, "value", str(ss))
            if not dispatched:
                rec["gated"] = True
            act = _action_of(action)
            if act:
                rec["a"] = act
            tele = None
            if agent is not None and hasattr(agent, "telemetry"):
                try:
                    tele = agent.telemetry()
                except Exception:
                    tele = None
            if tele:
                rec["ag"] = tele
            self._write(rec)
            self.n += 1
        except Exception:
            pass

    def event(self, name: str, **data: Any) -> None:
        if not self.enabled:
            return
        rec = {"type": "event", "ev": name}
        rec.update(data)
        self._write(rec)

    def close(self, summary: Optional[Dict[str, Any]] = None) -> None:
        if self._f is None:
            return
        try:
            s = {"type": "summary", "ticks": self.n}
            if summary:
                s.update(summary)
            self._write(s)
            self._f.flush()
            self._f.close()
        except Exception:
            pass
        finally:
            self._f = None


def build_episode_logger(settings: Dict[str, Any], agent_name: str,
                         root: str, started_ts: str = "") -> EpisodeLogger:
    """Construct from settings. Enabled unless ``logging.episodes`` is
    explicitly false. Writes to ``<root>/data/episodes/``."""
    log_cfg = (settings.get("logging", {}) or {}) if isinstance(settings, dict) else {}
    enabled = bool(log_cfg.get("episodes", True))
    out_dir = os.path.join(root, "data", "episodes")
    return EpisodeLogger(out_dir, agent_name, enabled=enabled, started_ts=started_ts)


__all__ = ["EpisodeLogger", "build_episode_logger"]
