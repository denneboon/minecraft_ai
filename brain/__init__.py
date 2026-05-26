"""
The agent contract — what every agent (rule-based today, ML tomorrow)
must implement, and what the runtime guarantees in exchange.

Public surface:

* :class:`brain.interfaces.BaseAgent` — base class. Override
  :meth:`decide` (and optionally :meth:`reset` / :meth:`shutdown`).
* :class:`brain.interfaces.AgentAction` — structured per-tick output:
  movement keys to hold, look delta / velocity, hotbar slot,
  interactions, free-form ``extras``.

Why this lives in its own package: it has no dependencies on the
vision or control packages, so anything (a test, a notebook, a
training loop) can import it without dragging the OpenCV / pynput
stack along.
"""
