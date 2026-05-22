from __future__ import annotations
from typing import Optional
import threading

class InputGate:
    def __init__(self):
        self._allow = False
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def allow(self) -> bool:
        with self._lock:
            return self._allow

    def set_allowed(self, value: bool) -> None:
        with self._cond:
            self._allow = bool(value)
            self._cond.notify_all()

    def wait_until_allowed(self, timeout: Optional[float] = None) -> bool:
        with self._cond:
            if not self._allow:
                self._cond.wait(timeout=timeout)
            return self._allow