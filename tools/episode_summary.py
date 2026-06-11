#!/usr/bin/env python3
"""
Summarise a recorded episode (or the latest) — quick analytics over the
JSONL the EpisodeLogger writes to data/episodes/.

    python tools/episode_summary.py                # latest episode
    python tools/episode_summary.py <file.jsonl>   # a specific one
    python tools/episode_summary.py --all          # roll up every episode
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EP_DIR = os.path.join(ROOT, "data", "episodes")


def _load(path):
    meta = summary = None
    recs = []
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            t = o.get("type")
            if t == "meta":
                meta = o
            elif t == "summary":
                summary = o
            elif t == "event":
                events.append(o)
            elif "k" in o:
                recs.append(o)
    return meta, recs, events, summary


def summarise(path):
    meta, recs, events, summary = _load(path)
    print(f"\n=== {os.path.basename(path)} ===")
    if meta:
        print(f"agent={meta.get('agent')} stamp={meta.get('stamp')}")
    print(f"ticks={len(recs)} events={len(events)}", end="")
    if summary:
        print(f" wall={summary.get('wall_s')}s hz={summary.get('hz')} "
              f"final={summary.get('final')}")
    else:
        print()
    # Agent-state breakdown (e.g. tree-chop FSM states) + action mix.
    states = Counter(r["ag"]["state"] for r in recs if r.get("ag", {}).get("state"))
    if states:
        tot = sum(states.values())
        print("  state time:", ", ".join(
            f"{k} {100*v//tot}%" for k, v in states.most_common()))
    acts = Counter()
    for r in recs:
        a = r.get("a", {})
        if a.get("int"): acts[a["int"]] += 1
        if a.get("mv", {}).get("forward"): acts["forward"] += 1
        if a.get("ld"): acts["look"] += 1
    if acts:
        print("  actions:", ", ".join(f"{k}={v}" for k, v in acts.most_common()))
    if events:
        evs = Counter(e.get("ev") for e in events)
        print("  events:", ", ".join(f"{k}={v}" for k, v in evs.most_common()))


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if argv and argv[0] == "--all":
        files = sorted(glob.glob(os.path.join(EP_DIR, "*.jsonl")))
        if not files:
            print(f"No episodes in {EP_DIR}"); return 1
        agents = Counter(); ticks = 0
        for p in files:
            meta, recs, _, summary = _load(p)
            agents[(meta or {}).get("agent", "?")] += 1
            ticks += len(recs)
        print(f"{len(files)} episodes, {ticks} total ticks; by agent: "
              f"{dict(agents)}")
        return 0
    if argv:
        path = argv[0]
    else:
        files = sorted(glob.glob(os.path.join(EP_DIR, "*.jsonl")))
        if not files:
            print(f"No episodes in {EP_DIR}. Run an agent first."); return 1
        path = files[-1]
    summarise(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
