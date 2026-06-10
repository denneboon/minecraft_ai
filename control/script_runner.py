# control/script_runner.py
"""
Run pre-recorded Minecraft macros / scripts through the input layer.

Why this exists
---------------
Some techniques (bridging, ladder-climbing, a fixed crafting-grid click
sequence) are *always the same* keystroke-and-click choreography. Rather
than teach the AI each one, you hand it a macro file and it plays it
back through the same gated keyboard / mouse layer the agents use — so
focus-loss auto-stop and the Ctrl+Shift+F12 emergency hotkey still apply.

Supported file types (what Minecraft macros usually are)
--------------------------------------------------------
* ``.ahk``  — **AutoHotkey**, by far the most common MC input-macro
  format on Windows. We support the linear subset real macros use:
  ``Send {key down/up}``, ``Send key``, ``Sleep``, ``Click[, right|left]``,
  ``Click down/up``, ``MouseMove, dx, dy[, , R]``, and ``Loop[, N] { … }``.
* ``.txt`` (Macro / Keybind Mod, Mumfrey) — the most common in-game macro
  mod. We support ``key(<bind>,true|false)``, ``press(<bind>)``,
  ``wait(<n>[ms])`` and ``do(<n>) … loop()``. MC bind names
  (``key.forward``, ``key.use`` → right-click, ``key.attack`` → left-click…)
  are mapped to real inputs.
* ``.json`` — a clean event/op list (what generic recorders export, and
  the easiest for the AI to author). See :func:`parse_json`.
* ``.mcs`` / ``.macro`` — a tiny human-writable line DSL (``down s``,
  ``click right``, ``sleep 200``, ``loop 20 … endloop``).

All formats parse into ONE recursive op model so the executor and the
safety handling are shared. Unrecognised lines are skipped with a
warning rather than aborting the whole script.

Op model
--------
A script is a list of ops. Each op is a dict ``{"op": <kind>, …}``:

* ``{"op": "key_down", "key": "s"}``  / ``key_up`` / ``{"op":"key_tap","key":"e","ms":60}``
* ``{"op": "mouse_down", "button": "right"}`` / ``mouse_up``
* ``{"op": "mouse_click", "button": "right", "ms": 50}``
* ``{"op": "move", "dx": 0, "dy": 120}``      (relative mouse move)
* ``{"op": "scroll", "amount": -1}``
* ``{"op": "sleep", "ms": 200}``
* ``{"op": "loop", "count": 20, "ops": [ … ]}``  (count <= 0 = repeat
  until the safety gate closes or the runtime cap is hit)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Key / button name normalisation
# ---------------------------------------------------------------------------

# Map the many names macros use for the same physical key onto the names
# the Keyboard backend understands. (Keyboard also has its own aliases,
# but macro files use a wider vocabulary — e.g. AHK's "LShift".)
_KEY_ALIASES: Dict[str, str] = {
    "ctrl": "control", "lctrl": "control", "rctrl": "control_r",
    "control_l": "control", "control_r": "control_r",
    "lcontrol": "control", "rcontrol": "control_r",
    "lshift": "shift", "rshift": "shift_r", "shift_l": "shift",
    "lalt": "alt", "ralt": "alt_r", "alt_l": "alt",
    "return": "enter", "esc": "escape", "spacebar": "space", "spc": "space",
    "del": "delete", "ins": "insert", "pgup": "page_up", "pgdn": "page_down",
}

# Minecraft Macro/Keybind-Mod bind names → either a keyboard action that
# the project keymap resolves, or a ("mouse", button) pair. ``forward`` /
# ``sneak`` etc. are the action names ``control/action_wrapper`` &
# ``main`` feed into the keymap, so the actual bound key is honoured.
_MC_BIND_TO_ACTION: Dict[str, str] = {
    "key.forward": "move_forward", "key.back": "move_backward",
    "key.left": "move_left", "key.right": "move_right",
    "key.jump": "jump", "key.sneak": "sneak", "key.sprint": "sprint",
    "key.inventory": "inventory", "key.drop": "drop",
}
_MC_BIND_TO_MOUSE: Dict[str, str] = {
    "key.attack": "left",      # left-click = attack / break
    "key.use": "right",        # right-click = use / place (bridging!)
    "key.pickitem": "middle",
}


def normalize_key(name: str) -> str:
    k = str(name).strip().lower()
    return _KEY_ALIASES.get(k, k)


def normalize_button(name: str) -> str:
    b = str(name).strip().lower()
    if b in ("lbutton", "left", "leftclick", "lclick", "l", "mouse_left", "m1"):
        return "left"
    if b in ("rbutton", "right", "rightclick", "rclick", "r", "mouse_right", "m2"):
        return "right"
    if b in ("mbutton", "middle", "mclick", "m3"):
        return "middle"
    return b


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ScriptRunnerConfig:
    default_tap_ms: int = 60          # key_tap with no explicit duration
    default_click_ms: int = 50        # mouse_click with no explicit duration
    # Hard wall-clock cap so an infinite ``loop`` (or a runaway file) can
    # never run forever. The gate / emergency hotkey stop it sooner.
    max_runtime_sec: float = 120.0
    # Per-loop iteration cap for ``count <= 0`` (infinite) loops, as a
    # belt-and-suspenders alongside the wall-clock cap.
    max_infinite_iters: int = 1_000_000
    max_loop_depth: int = 16          # nesting guard
    # Sleeps are chunked to this so a long ``sleep`` still aborts promptly
    # when focus is lost / the emergency hotkey fires.
    sleep_chunk_ms: int = 20


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class ScriptRunner:
    """
    Execute a parsed op list through a (gated) Keyboard + Mouse.

    The runner is gate-aware: before every op and during every sleep it
    checks ``gate.allow()`` (the Safety controller closes it on focus
    loss / emergency) and the wall-clock cap, and aborts cleanly —
    releasing every key / button the script was holding.
    """

    def __init__(self,
                 keyboard,
                 mouse,
                 *,
                 gate=None,
                 config: Optional[ScriptRunnerConfig] = None,
                 on_status: Callable[[str], None] = print):
        self._kb = keyboard
        self._ms = mouse
        self._gate = gate
        self.cfg = config or ScriptRunnerConfig()
        self._say = on_status
        self._held_keys: set = set()
        self._held_buttons: set = set()
        self._abort_reason: Optional[str] = None
        self._deadline: float = 0.0

    # ── Public ────────────────────────────────────────────────────

    def run(self, ops: List[Dict[str, Any]], *, name: str = "script") -> bool:
        """Play ``ops``. Returns True if it ran to completion, False if it
        aborted (gate closed / runtime cap / error). Always releases any
        keys/buttons it was holding."""
        self._held_keys.clear()
        self._held_buttons.clear()
        self._abort_reason = None
        self._deadline = time.perf_counter() + self.cfg.max_runtime_sec
        n_ops = _count_ops(ops)
        self._say(f"[script] running {name!r}: {n_ops} ops "
                  f"(max {self.cfg.max_runtime_sec:.0f}s; "
                  f"Ctrl+Shift+F12 = stop)")
        try:
            self._exec(ops, depth=0)
        except Exception as e:
            self._abort_reason = f"error: {e!r}"
        finally:
            self._release_all()
        if self._abort_reason:
            self._say(f"[script] {name!r} stopped early: {self._abort_reason}")
            return False
        self._say(f"[script] {name!r} completed.")
        return True

    # ── Execution ─────────────────────────────────────────────────

    def _exec(self, ops: List[Dict[str, Any]], depth: int) -> None:
        if depth > self.cfg.max_loop_depth:
            self._say(f"[script][WARN] loop nesting > {self.cfg.max_loop_depth}; "
                      "stopping descent.")
            return
        for op in ops:
            if self._should_abort():
                return
            kind = op.get("op")
            if kind == "loop":
                self._run_loop(op, depth)
            elif kind == "key_down":
                self._press(normalize_key(op["key"]))
            elif kind == "key_up":
                self._release(normalize_key(op["key"]))
            elif kind == "key_tap":
                self._tap(normalize_key(op["key"]),
                          int(op.get("ms", self.cfg.default_tap_ms)))
            elif kind == "mouse_down":
                self._mouse_down(normalize_button(op["button"]))
            elif kind == "mouse_up":
                self._mouse_up(normalize_button(op["button"]))
            elif kind == "mouse_click":
                self._mouse_click(normalize_button(op["button"]),
                                  int(op.get("ms", self.cfg.default_click_ms)))
            elif kind == "move":
                self._move(int(op.get("dx", 0)), int(op.get("dy", 0)))
            elif kind == "scroll":
                self._scroll(int(op.get("amount", 0)))
            elif kind == "sleep":
                self._sleep(int(op.get("ms", 0)))
            else:
                self._say(f"[script][WARN] unknown op {kind!r} — skipped.")

    def _run_loop(self, op: Dict[str, Any], depth: int) -> None:
        count = int(op.get("count", 1))
        inner = op.get("ops", [])
        infinite = count <= 0
        i = 0
        while not self._should_abort():
            if not infinite and i >= count:
                break
            if infinite and i >= self.cfg.max_infinite_iters:
                break
            self._exec(inner, depth + 1)
            i += 1

    def _should_abort(self) -> bool:
        if self._abort_reason:
            return True
        if self._gate is not None and not self._gate.allow():
            self._abort_reason = "input gate closed (focus lost / emergency)"
            return True
        if time.perf_counter() > self._deadline:
            self._abort_reason = f"max runtime ({self.cfg.max_runtime_sec:.0f}s) reached"
            return True
        return False

    # ── Primitive dispatch (track held state for clean release) ───

    def _press(self, key: str) -> None:
        self._kb.press(key)
        self._held_keys.add(key)

    def _release(self, key: str) -> None:
        self._kb.release(key)
        self._held_keys.discard(key)

    def _tap(self, key: str, ms: int) -> None:
        self._kb.tap(key, max(0.0, ms / 1000.0))

    def _mouse_down(self, button: str) -> None:
        if button == "left":
            self._ms.left_press()
        elif button == "right":
            self._ms.right_press()
        else:
            self._say(f"[script][WARN] mouse_down {button!r} unsupported.")
            return
        self._held_buttons.add(button)

    def _mouse_up(self, button: str) -> None:
        if button == "left":
            self._ms.left_release()
        elif button == "right":
            self._ms.right_release()
        else:
            return
        self._held_buttons.discard(button)

    def _mouse_click(self, button: str, ms: int) -> None:
        dur = max(0.0, ms / 1000.0)
        if button == "left":
            self._ms.left_click(dur)
        elif button == "right":
            self._ms.right_click(dur)
        else:
            self._say(f"[script][WARN] mouse_click {button!r} unsupported.")

    def _move(self, dx: int, dy: int) -> None:
        if dx or dy:
            self._ms.move(dx, dy)

    def _scroll(self, amount: int) -> None:
        if amount > 0:
            self._ms.scroll_up(amount)
        elif amount < 0:
            self._ms.scroll_down(-amount)

    def _sleep(self, ms: int) -> None:
        if ms <= 0:
            return
        end = time.perf_counter() + ms / 1000.0
        chunk = max(0.001, self.cfg.sleep_chunk_ms / 1000.0)
        while time.perf_counter() < end:
            if self._should_abort():
                return
            time.sleep(min(chunk, max(0.0, end - time.perf_counter())))

    def _release_all(self) -> None:
        """Release everything the script was holding. Releases are never
        gated, so this fires even after focus loss / emergency."""
        for key in list(self._held_keys):
            try:
                self._kb.release(key)
            except Exception:
                pass
        self._held_keys.clear()
        releasers = {"left": self._ms.left_release, "right": self._ms.right_release}
        for button in list(self._held_buttons):
            fn = releasers.get(button)
            if fn is None:
                continue
            try:
                fn()
            except Exception:
                pass
        self._held_buttons.clear()


def _count_ops(ops: List[Dict[str, Any]]) -> int:
    n = 0
    for op in ops:
        n += 1
        if op.get("op") == "loop":
            n += _count_ops(op.get("ops", []))
    return n


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_json(data: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """A clean op list. Accepts either ``[op, …]`` or
    ``{"name": …, "ops": [op, …]}``. Ops match the model documented at
    the top of this module."""
    if isinstance(data, dict):
        name = str(data.get("name", "json-script"))
        ops = data.get("ops", [])
    elif isinstance(data, list):
        name, ops = "json-script", data
    else:
        raise ValueError("JSON script must be a list of ops or an object with 'ops'.")
    return name, _validate_ops(ops)


def _validate_ops(ops: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(ops, list):
        return out
    for op in ops:
        if not isinstance(op, dict) or "op" not in op:
            continue
        if op["op"] == "loop":
            op = dict(op)
            op["ops"] = _validate_ops(op.get("ops", []))
        out.append(op)
    return out


def _unwind(stack: List[List[Dict[str, Any]]],
            loop_counts: List[int]) -> None:
    """Close any loop blocks left open at end-of-parse (a malformed file
    missing an ``endloop`` / ``}`` / ``loop()``). The block's ops are
    folded into its parent as a loop rather than silently dropped. Pops
    ``stack`` and ``loop_counts`` in lockstep so they can't desync."""
    while len(stack) > 1:
        inner = stack.pop()
        count = loop_counts.pop() if loop_counts else 1
        stack[-1].append({"op": "loop", "count": count, "ops": inner})


def parse_simple(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Tiny line DSL. One command per line; ``#`` comments; ``loop N`` …
    ``endloop`` blocks (nestable). Commands::

        down <key> | up <key> | tap <key> [ms]
        click <left|right> [ms] | press <left|right> | release <left|right>
        move <dx> <dy> | scroll <amount> | sleep <ms>
        loop <n>  …  endloop          (n omitted / 0 = infinite)
        name <script name>
    """
    name = "macro"
    root: List[Dict[str, Any]] = []
    stack: List[List[Dict[str, Any]]] = [root]
    loop_counts: List[int] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]
        if cmd == "name":
            name = " ".join(args) or name
        elif cmd in ("down", "press_key", "hold"):
            stack[-1].append({"op": "key_down", "key": args[0]})
        elif cmd in ("up", "release_key"):
            stack[-1].append({"op": "key_up", "key": args[0]})
        elif cmd == "tap":
            o = {"op": "key_tap", "key": args[0]}
            if len(args) > 1:
                o["ms"] = _int(args[1])
            stack[-1].append(o)
        elif cmd == "click":
            o = {"op": "mouse_click", "button": args[0] if args else "left"}
            if len(args) > 1:
                o["ms"] = _int(args[1])
            stack[-1].append(o)
        elif cmd in ("press", "mdown"):
            stack[-1].append({"op": "mouse_down", "button": args[0] if args else "left"})
        elif cmd in ("release", "mup"):
            stack[-1].append({"op": "mouse_up", "button": args[0] if args else "left"})
        elif cmd == "move":
            stack[-1].append({"op": "move",
                              "dx": _int(args[0]) if args else 0,
                              "dy": _int(args[1]) if len(args) > 1 else 0})
        elif cmd == "scroll":
            stack[-1].append({"op": "scroll", "amount": _int(args[0]) if args else 0})
        elif cmd in ("sleep", "wait"):
            stack[-1].append({"op": "sleep", "ms": _int(args[0]) if args else 0})
        elif cmd == "loop":
            inner: List[Dict[str, Any]] = []
            loop_counts.append(_int(args[0]) if args else 0)
            stack.append(inner)
        elif cmd in ("endloop", "end"):
            if len(stack) > 1:
                inner = stack.pop()
                count = loop_counts.pop()
                stack[-1].append({"op": "loop", "count": count, "ops": inner})
        # else: unknown line ignored
    _unwind(stack, loop_counts)
    return name, root


# ── AutoHotkey (.ahk) — the common linear subset ──────────────────

_AHK_SEND_TOKEN = re.compile(r"\{([^}]+)\}|(\S+)")


def parse_ahk(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    root: List[Dict[str, Any]] = []
    stack: List[List[Dict[str, Any]]] = [root]
    loop_counts: List[int] = []
    pending_loop_count: Optional[int] = None   # Loop seen, awaiting '{'

    # Split braces that sit on the end of a directive onto their own line
    # so "Loop, 20 {" and "Click, right" are handled uniformly.
    lines: List[str] = []
    for raw in text.splitlines():
        s = _strip_ahk_comment(raw).strip()
        if not s:
            continue
        # Hotkey labels ("F1::", "$*RButton::") just open a macro body —
        # drop the label, keep anything after "::".
        if "::" in s:
            s = s.split("::", 1)[1].strip()
            if not s:
                continue
        # Pull a trailing/leading brace onto its own token.
        if s.endswith("{") and s != "{":
            lines.append(s[:-1].strip())
            lines.append("{")
            continue
        lines.append(s)

    for s in lines:
        low = s.lower()
        if s == "{":
            inner: List[Dict[str, Any]] = []
            loop_counts.append(pending_loop_count if pending_loop_count is not None else 1)
            pending_loop_count = None
            stack.append(inner)
            continue
        if s == "}":
            if len(stack) > 1:
                inner = stack.pop()
                stack[-1].append({"op": "loop", "count": loop_counts.pop(), "ops": inner})
            continue
        if low.startswith("loop"):
            rest = s[4:].lstrip(", ").strip()
            pending_loop_count = _int(rest) if rest else 0   # 0 = infinite
            continue
        if low.startswith("sleep"):
            stack[-1].append({"op": "sleep", "ms": _int(s[5:].lstrip(", ").strip())})
            continue
        if low.startswith("send"):
            _parse_ahk_send(s[4:].lstrip(", "), stack[-1])
            continue
        if low.startswith("click"):
            _parse_ahk_click(s[5:].lstrip(", "), stack[-1])
            continue
        if low.startswith("mousemove"):
            _parse_ahk_mousemove(s[9:].lstrip(", "), stack[-1])
            continue
        # silently ignore unsupported AHK directives (return, #NoEnv, etc.)
    _unwind(stack, loop_counts)
    return "ahk-macro", root


def _parse_ahk_send(payload: str, out: List[Dict[str, Any]]) -> None:
    """Handle ``Send`` content: ``{key down}``, ``{key up}``, bare keys."""
    for m in _AHK_SEND_TOKEN.finditer(payload):
        brace, bare = m.group(1), m.group(2)
        if brace is not None:
            toks = brace.split()
            key = normalize_key(_ahk_key(toks[0]))
            action = toks[1].lower() if len(toks) > 1 else ""
            if action == "down":
                out.append({"op": "key_down", "key": key})
            elif action == "up":
                out.append({"op": "key_up", "key": key})
            else:
                out.append({"op": "key_tap", "key": key})
        elif bare:
            for ch in bare:
                out.append({"op": "key_tap", "key": normalize_key(ch)})


def _parse_ahk_click(payload: str, out: List[Dict[str, Any]]) -> None:
    """``Click`` | ``Click, right`` | ``Click down`` | ``Click up`` |
    ``Click, X, Y`` (coords ignored — we drive relative)."""
    toks = [t.strip().lower() for t in payload.replace(",", " ").split() if t.strip()]
    button = "left"
    state = None
    for t in toks:
        if t in ("left", "right", "middle"):
            button = t
        elif t in ("down", "up"):
            state = t
        # numbers (coords) and 'rel'/'r' are ignored
    if state == "down":
        out.append({"op": "mouse_down", "button": button})
    elif state == "up":
        out.append({"op": "mouse_up", "button": button})
    else:
        out.append({"op": "mouse_click", "button": button})


def _parse_ahk_mousemove(payload: str, out: List[Dict[str, Any]]) -> None:
    nums = re.findall(r"-?\d+", payload)
    if len(nums) >= 2:
        out.append({"op": "move", "dx": int(nums[0]), "dy": int(nums[1])})


def _ahk_key(name: str) -> str:
    """AHK key tokens → our key names."""
    n = name.strip().lower()
    table = {
        "lbutton": "__mouse_left", "rbutton": "__mouse_right",
        "enter": "enter", "return": "enter", "space": "space",
        "tab": "tab", "escape": "escape", "esc": "escape",
        "shift": "shift", "lshift": "shift", "rshift": "shift_r",
        "ctrl": "control", "control": "control", "lctrl": "control",
        "alt": "alt", "lalt": "alt",
    }
    return table.get(n, n)


def _strip_ahk_comment(line: str) -> str:
    # AHK line comments start with ';' (only when preceded by whitespace
    # or at line start). Good enough for macro files.
    idx = line.find(";")
    if idx == 0:
        return ""
    if idx > 0 and line[idx - 1] in " \t":
        return line[:idx]
    return line


# ── Macro / Keybind Mod (.txt) ────────────────────────────────────

_KEYBIND_CALL = re.compile(r"(\w+)\s*\(\s*([^)]*)\)")


def parse_keybind_mod(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    root: List[Dict[str, Any]] = []
    stack: List[List[Dict[str, Any]]] = [root]
    loop_counts: List[int] = []
    # The mod wraps bodies in $${ … }$$ — strip those markers.
    text = text.replace("$${", " ").replace("}$$", " ")
    for raw in text.splitlines():
        # Strip comments FIRST (before reading calls) so comment prose
        # like "do(n)...loop()" can't be parsed as real statements.
        line = raw.split("#", 1)[0].split("//", 1)[0].strip()
        if not line:
            continue
        # Process every call on the line, left to right (the mod usually
        # has one per line, but be tolerant).
        for m in _KEYBIND_CALL.finditer(line):
            fn = m.group(1).lower()
            argstr = m.group(2).strip()
            args = [a.strip() for a in argstr.split(",")] if argstr else []
            if fn == "key" and args:
                _emit_keybind(args[0], args[1].lower() if len(args) > 1 else "true",
                              stack[-1])
            elif fn == "press" and args:
                _emit_keybind_press(args[0], stack[-1])
            elif fn == "wait":
                stack[-1].append({"op": "sleep", "ms": _parse_wait(args[0] if args else "")})
            elif fn == "do":
                loop_counts.append(_int(args[0]) if args else 0)
                stack.append([])
            elif fn == "loop":
                if len(stack) > 1:
                    inner = stack.pop()
                    stack[-1].append({"op": "loop",
                                      "count": loop_counts.pop(), "ops": inner})
            # echo(), log(), if() … unsupported → ignored
    _unwind(stack, loop_counts)
    return "keybind-macro", root


def _bind_target(name: str):
    n = name.strip().lower()
    if n in _MC_BIND_TO_MOUSE:
        return ("mouse", _MC_BIND_TO_MOUSE[n])
    if n in _MC_BIND_TO_ACTION:
        return ("key", _MC_BIND_TO_ACTION[n])
    # A literal key like "key.keyboard.w" or "w".
    n = n.replace("key.keyboard.", "").replace("key.", "")
    return ("key", normalize_key(n))


def _emit_keybind(name: str, state: str, out: List[Dict[str, Any]]) -> None:
    kind, target = _bind_target(name)
    down = state in ("true", "1", "down", "on")
    if kind == "mouse":
        out.append({"op": "mouse_down" if down else "mouse_up", "button": target})
    else:
        out.append({"op": "key_down" if down else "key_up", "key": target})


def _emit_keybind_press(name: str, out: List[Dict[str, Any]]) -> None:
    kind, target = _bind_target(name)
    if kind == "mouse":
        out.append({"op": "mouse_click", "button": target})
    else:
        out.append({"op": "key_tap", "key": target})


def _parse_wait(s: str) -> int:
    s = s.strip().lower()
    if s.endswith("ms"):
        return _int(s[:-2])
    if s.endswith("s"):
        return int(_float(s[:-1]) * 1000)
    # Bare number in the keybind mod is TICKS (1 tick = 50 ms).
    n = _int(s)
    return n * 50 if n else 0


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_script(path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Load + parse a macro file, choosing the parser by extension
    (with a content sniff for ambiguous ``.txt``)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No script at {p}")
    text = p.read_text(encoding="utf-8", errors="replace")
    ext = p.suffix.lower()
    if ext == ".json":
        return parse_json(json.loads(text))
    if ext == ".ahk":
        return parse_ahk(text)
    if ext in (".mcs", ".macro"):
        return parse_simple(text)
    if ext == ".txt":
        # Disambiguate: Keybind-Mod files are full of "name(args)" calls;
        # otherwise treat as the simple line DSL.
        if _KEYBIND_CALL.search(text) and ("key(" in text or "do(" in text
                                            or "wait(" in text or "press(" in text):
            return parse_keybind_mod(text)
        return parse_simple(text)
    # Last resort: try JSON, then the simple DSL.
    try:
        return parse_json(json.loads(text))
    except Exception:
        return parse_simple(text)


_INT_RE = re.compile(r"-?\d+")


def _int(s: str, default: int = 0) -> int:
    m = _INT_RE.search(str(s))
    return int(m.group()) if m else default


def _float(s: str, default: float = 0.0) -> float:
    try:
        return float(str(s).strip())
    except ValueError:
        return default


__all__ = [
    "ScriptRunner", "ScriptRunnerConfig",
    "load_script", "parse_json", "parse_simple", "parse_ahk",
    "parse_keybind_mod", "normalize_key", "normalize_button",
]
