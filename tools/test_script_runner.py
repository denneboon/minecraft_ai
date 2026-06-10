#!/usr/bin/env python3
"""
Offline self-test for control.script_runner (no Minecraft needed).

Covers:
  * all four parsers (json / simple DSL / AutoHotkey / Keybind-Mod) turn
    the bridging examples into the same op structure
  * loops nest + carry their count; key/mouse/sleep ops normalise
  * the executor drives a FAKE keyboard/mouse in order, honours loop
    counts, aborts when the gate closes, and releases everything it held

Run: python tools/test_script_runner.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.script_runner import (
    ScriptRunner, ScriptRunnerConfig, load_script,
    parse_ahk, parse_simple, parse_keybind_mod,
)

_fail = 0


def ok(m):
    print(f"  [ok] {m}")


def bad(m):
    global _fail
    _fail += 1
    print(f"  [FAIL] {m}")


# ── Fakes ─────────────────────────────────────────────────────────

class FakeKB:
    def __init__(self):
        self.log = []
        self.held = set()
    def press(self, k):   self.log.append(("press", k)); self.held.add(k)
    def release(self, k): self.log.append(("release", k)); self.held.discard(k)
    def tap(self, k, d=None): self.log.append(("tap", k))


class FakeMouse:
    def __init__(self):
        self.log = []
        self.left = self.right = False
    def left_press(self):    self.log.append(("Ldown",)); self.left = True
    def left_release(self):  self.log.append(("Lup",)); self.left = False
    def right_press(self):   self.log.append(("Rdown",)); self.right = True
    def right_release(self): self.log.append(("Rup",)); self.right = False
    def left_click(self, d=None):  self.log.append(("Lclick",))
    def right_click(self, d=None): self.log.append(("Rclick",))
    def move(self, dx, dy):  self.log.append(("move", dx, dy))
    def scroll_up(self, a):  self.log.append(("scrollU", a))
    def scroll_down(self, a):self.log.append(("scrollD", a))


class FakeGate:
    def __init__(self, allow=True): self._a = allow
    def allow(self): return self._a


def _ops_summary(ops):
    """(kinds list, loop count if any) for structural comparison."""
    kinds = []
    loop_count = None
    for o in ops:
        kinds.append(o["op"])
        if o["op"] == "loop":
            loop_count = o["count"]
    return kinds, loop_count


def test_parsers_agree():
    print("[1] all four formats parse the bridge example consistently")
    files = {
        "json": ROOT / "scripts/macros/bridge.json",
        "ahk": ROOT / "scripts/macros/bridge.ahk",
        "mcs": ROOT / "scripts/macros/bridge.mcs",
        "keybind": ROOT / "scripts/macros/bridge_keybind.txt",
    }
    parsed = {}
    for label, path in files.items():
        name, ops = load_script(str(path))
        parsed[label] = ops
        kinds, lc = _ops_summary(ops)
        has_loop = "loop" in kinds
        (ok if has_loop else bad)(f"{label}: parsed {len(ops)} top-level ops, has loop={has_loop}")
    # Each should contain a loop with a body that does a right-action.
    for label, ops in parsed.items():
        loops = [o for o in ops if o["op"] == "loop"]
        if not loops:
            bad(f"{label}: no loop op")
            continue
        body = loops[0]["ops"]
        kinds = [b["op"] for b in body]
        right_place = any(b.get("button") == "right" for b in body
                          if b["op"] in ("mouse_click", "mouse_down"))
        has_sleep = "sleep" in kinds
        (ok if (right_place and has_sleep) else bad)(
            f"{label}: loop body places (right) + sleeps -> {kinds}")


def test_ahk_send_states():
    print("[2] AHK Send {key down/up} -> key_down/key_up")
    _, ops = parse_ahk("Send {Shift down}\nSend {Shift up}\nSend {s down}")
    kinds = [(o["op"], o.get("key")) for o in ops]
    want = [("key_down", "shift"), ("key_up", "shift"), ("key_down", "s")]
    (ok if kinds == want else bad)(f"send states -> {kinds}")


def test_keybind_mouse_mapping():
    print("[3] Keybind-Mod key.use -> right mouse; do(n)/loop()")
    _, ops = parse_keybind_mod("key(key.use,true)\ndo(3)\nkey(key.use,false)\nloop()")
    first = ops[0]
    (ok if first["op"] == "mouse_down" and first["button"] == "right" else bad)(
        f"key.use true -> {first}")
    loops = [o for o in ops if o["op"] == "loop"]
    (ok if loops and loops[0]["count"] == 3 else bad)(
        f"do(3) -> loop count {loops[0]['count'] if loops else None}")


def test_simple_nested_loop():
    print("[4] simple DSL nested loops + counts")
    text = "loop 2\n  loop 3\n    click right\n  endloop\nendloop"
    _, ops = parse_simple(text)
    outer = ops[0]
    (ok if outer["op"] == "loop" and outer["count"] == 2 else bad)(f"outer loop {outer.get('count')}")
    inner = outer["ops"][0]
    (ok if inner["op"] == "loop" and inner["count"] == 3 else bad)(f"inner loop {inner.get('count')}")


def test_executor_runs_and_counts():
    print("[5] executor drives fakes in order + honours loop count")
    kb, ms, gate = FakeKB(), FakeMouse(), FakeGate(True)
    r = ScriptRunner(kb, ms, gate=gate,
                     config=ScriptRunnerConfig(max_runtime_sec=5),
                     on_status=lambda *_: None)
    _, ops = load_script(str(ROOT / "scripts/macros/bridge.json"))
    done = r.run(ops, name="bridge")
    n_rclick = sum(1 for e in ms.log if e == ("Rclick",))
    (ok if done else bad)("ran to completion")
    (ok if n_rclick == 15 else bad)(f"15 right-clicks emitted (got {n_rclick})")
    (ok if not kb.held and not ms.left and not ms.right else bad)(
        "all keys/buttons released at end")
    # 'shift' and 's' must have been pressed then released.
    (ok if ("press", "shift") in kb.log and ("release", "shift") in kb.log else bad)(
        "shift pressed + released")


def test_gate_abort_releases():
    print("[6] gate closing mid-run aborts and releases held keys")
    kb, ms = FakeKB(), FakeMouse()
    gate = FakeGate(True)
    # Hold a key, then a long loop; flip the gate closed before the loop.
    ops = [
        {"op": "key_down", "key": "w"},
        {"op": "loop", "count": 0, "ops": [{"op": "sleep", "ms": 50}]},
    ]
    r = ScriptRunner(kb, ms, gate=gate,
                     config=ScriptRunnerConfig(max_runtime_sec=5),
                     on_status=lambda *_: None)
    gate._a = False   # gate closed -> first _should_abort aborts immediately
    done = r.run(ops, name="abort")
    (ok if not done else bad)("run reported aborted")
    (ok if not kb.held else bad)("held key released after abort")


def main():
    print("=" * 60)
    print(" control.script_runner self-test")
    print("=" * 60)
    test_parsers_agree()
    test_ahk_send_states()
    test_keybind_mouse_mapping()
    test_simple_nested_loop()
    test_executor_runs_and_counts()
    test_gate_abort_releases()
    print("=" * 60)
    if _fail:
        print(f" {_fail} CHECK(S) FAILED")
        return 1
    print(" ALL SCRIPT-RUNNER TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
