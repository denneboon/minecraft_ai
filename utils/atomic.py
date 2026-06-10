# utils/atomic.py
"""
Atomic file writes that tolerate Windows' transient ``os.replace`` locks.

On Windows ``os.replace(tmp, target)`` can raise ``PermissionError``
(WinError 5, "Access is denied") when the target is momentarily held by
another handle — an antivirus scan, the search indexer, or another
process that just read it. The write itself is fine; a brief retry
almost always succeeds. Several persistence paths in this project do the
same "write tmp → fsync → replace" dance (mouse calibration, weather
samples, the perception accuracy log), so it lives here once.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Union


def atomic_write_text(path: Union[str, Path],
                      text: str,
                      *,
                      encoding: str = "utf-8",
                      retries: int = 5,
                      retry_delay: float = 0.05,
                      fsync: bool = True) -> None:
    """Write ``text`` to ``path`` atomically (tmp file + ``os.replace``),
    retrying the replace on a transient Windows ``PermissionError``.

    Raises the last error if every attempt fails (callers typically wrap
    this best-effort). The temp file is cleaned up on persistent failure
    so a half-written sibling isn't left behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding=encoding) as f:
        f.write(text)
        f.flush()
        if fsync:
            try:
                os.fsync(f.fileno())
            except (OSError, AttributeError):
                pass

    attempts = max(1, retries)
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == attempts - 1:
                # Out of retries — clean up the temp and re-raise so the
                # caller's best-effort handler logs it.
                try:
                    tmp.unlink()
                except OSError:
                    pass
                raise
            time.sleep(retry_delay)


__all__ = ["atomic_write_text"]
