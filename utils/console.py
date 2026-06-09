# utils/console.py
"""
Console encoding safety for Windows.

Garbled OCR text, F3 dumps, and progress UIs routinely contain non-ASCII
glyphs (─ ∙ … ° ⏎ box-drawing). The default Windows console is cp1252,
which can't encode those, so a bare ``print`` raises ``UnicodeEncodeError``
and can tear down a whole loop. :func:`ensure_utf8_stdout` reconfigures
stdout/stderr to UTF-8 with ``errors="replace"`` so prints never crash —
offending chars degrade to ``?`` instead.

``main.py`` does this inline before any imports; tools/tests should call
``ensure_utf8_stdout()`` at startup to get the same protection.
"""

from __future__ import annotations

import sys


def ensure_utf8_stdout() -> None:
    """Make stdout/stderr tolerate arbitrary Unicode (idempotent, safe)."""
    for stream in (sys.stdout, sys.stderr):
        reconfig = getattr(stream, "reconfigure", None)
        if reconfig is not None:
            try:
                reconfig(encoding="utf-8", errors="replace")
            except (AttributeError, OSError, ValueError):
                pass


__all__ = ["ensure_utf8_stdout"]
