#!/usr/bin/env python3
"""Offline self-test for brain/episode_logger.py — records valid JSONL,
labels with agent telemetry, is robust to missing perception, and never
raises into the caller."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.console import ensure_utf8_stdout
ensure_utf8_stdout()

from brain.episode_logger import EpisodeLogger, build_episode_logger
from brain.interfaces import AgentAction

_fails = 0
def ok(m): print(f"  [OK]  {m}")
def bad(m):
    global _fails; _fails += 1; print(f"  [FAIL] {m}")


def _state(playing=True, hunger=0.8):
    pose = SimpleNamespace(x=1.25, y=64.0, z=-3.5, yaw=12.0, pitch=-4.0)
    world = SimpleNamespace(pose=pose,
                            looking_at=SimpleNamespace(pos=(2, 64, -3),
                                                       block_id="minecraft:oak_log"))
    scr = SimpleNamespace(value="playing") if playing else SimpleNamespace(value="menu")
    return SimpleNamespace(f3=pose, world=world, health=1.0, hunger=hunger,
                           screen_state=scr)


class _Agent:
    name = "treechop"
    def telemetry(self): return {"state": "chop", "logs": 3}


def main() -> int:
    print("=" * 56); print(" EpisodeLogger — offline self-test"); print("=" * 56)
    tmp = tempfile.mkdtemp(prefix="ep_test_")

    # 1. Records a normal run -> valid JSONL with meta/records/summary.
    print("\n[1] basic record/close")
    lg = EpisodeLogger(tmp, "treechop", started_ts="testrun")
    ag = _Agent()
    act = AgentAction(movement={"forward": True, "sprint": True}, interact="attack",
                      hotbar=2, look_dx=5)
    for k in range(4):
        lg.record(k, k * 0.1, _state(), act, agent=ag, dispatched=True)
    lg.event("chopped", voxel=[2, 64, -3])
    lg.close({"wall_s": 0.4, "final": ag.telemetry()})   # main merges telemetry like this
    path = lg.path
    lines = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    (ok if lines[0].get("type") == "meta" else bad)("first line is meta")
    recs = [l for l in lines if "k" in l]
    (ok if len(recs) == 4 else bad)(f"4 tick records ({len(recs)})")
    r = recs[0]
    (ok if r.get("p") and r.get("look") and r.get("a") and r.get("ag")
     else bad)(f"record has pose/look/action/agent: {sorted(r)}")
    (ok if r["a"].get("int") == "attack" and r["a"].get("slot") == 2
     else bad)(f"action fields captured: {r['a']}")
    (ok if any(l.get("type") == "event" for l in lines) else bad)("event logged")
    summ = [l for l in lines if l.get("type") == "summary"]
    (ok if summ and summ[0].get("ticks") == 4 and summ[0].get("final") else bad)(
        f"summary with ticks + final telemetry: {summ[:1]}")

    # 2. Robust to missing perception (no pose / no world) — no crash.
    print("\n[2] robust to missing fields")
    lg2 = EpisodeLogger(tmp, "x", started_ts="r2")
    try:
        lg2.record(0, 0.0, SimpleNamespace(f3=None, world=None,
                                           screen_state=None), AgentAction())
        lg2.record(1, 0.1, None, AgentAction())          # state=None
        lg2.close()
        ok("records sparse/None state without raising")
    except Exception as e:
        bad(f"raised on sparse state: {e!r}")

    # 3. Disabled logger is a silent no-op.
    print("\n[3] disabled = no-op")
    lg3 = EpisodeLogger(tmp, "x", enabled=False)
    lg3.record(0, 0.0, _state(), AgentAction()); lg3.close()
    (ok if lg3.path is None else bad)("disabled logger writes nothing")

    # 4. build_episode_logger honours settings.logging.episodes.
    print("\n[4] build from settings")
    on = build_episode_logger({"logging": {"episodes": True}}, "a", tmp)
    off = build_episode_logger({"logging": {"episodes": False}}, "a", tmp)
    (ok if on.enabled and not off.enabled else bad)(
        f"settings toggle (on={on.enabled}, off={off.enabled})")
    on.close(); off.close()

    print("\n" + ("ALL EPISODE-LOGGER TESTS PASSED" if not _fails
                  else f"{_fails} CHECK(S) FAILED"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
