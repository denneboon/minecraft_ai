# vision/world/metrics.py
"""
Self-teaching session metrics — graph-ready performance history.

Every self-teaching run (``tools/learn_world_live.py``) records how the
block recogniser did against F3 ground truth so progress can be tracked
and plotted over time. Two artefacts per session:

* ``data/metrics/sessions.jsonl`` — ONE summary line per session
  (appended). This is the long-term history: accuracy / coverage /
  sample-count / per-block breakdown vs date. Plot it to see the AI
  improve across sessions.
* ``data/metrics/<session_id>_steps.csv`` — one row per classified step
  (the within-session learning curve: running accuracy, confidence,
  sample count as the run progresses).

Both are plain text (JSONL + CSV) so any tool — ``tools/plot_metrics.py``,
Excel, pandas — can graph them with no special reader.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


def default_metrics_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "metrics"


# Result tags for a single classified step.
HIT = "HIT"
MISS = "miss"
ABSTAIN = "abstain"


class SessionMetrics:
    """Accumulates per-step results for one self-teaching session and
    writes a summary + per-step CSV on :meth:`finalize`."""

    def __init__(self, tool: str, session_id: str, start_ts_unix: float,
                 *, root: Optional[Path] = None):
        self.tool = tool
        self.session_id = session_id
        self.start_ts_unix = float(start_ts_unix)
        self.root = Path(root) if root is not None else default_metrics_root()
        self._steps: List[dict] = []
        self._hits = 0
        self._decided = 0          # steps where the recogniser gave a guess
        self._conf_sum = 0.0
        self._per_seen: Counter = Counter()
        self._per_hit: Counter = Counter()

    # ── Recording ──────────────────────────────────────────────────
    def record(self, *, step: int, truth: str, guess: Optional[str],
               conf: float, samples: int) -> str:
        """Record one classified step (one that HAD an F3 target). Returns
        the result tag (HIT / miss / abstain)."""
        self._per_seen[truth] += 1
        if guess is None:
            result = ABSTAIN
        elif guess == truth:
            result = HIT
            self._hits += 1
            self._decided += 1
            self._per_hit[truth] += 1
            self._conf_sum += conf
        else:
            result = MISS
            self._decided += 1
            self._conf_sum += conf
        seen = sum(self._per_seen.values())
        self._steps.append({
            "step": step,
            "truth": truth,
            "guess": guess or "",
            "conf": round(float(conf), 4),
            "result": result,
            "running_acc": round(self._hits / seen, 4) if seen else 0.0,
            "samples": int(samples),
        })
        return result

    # ── Finalise ───────────────────────────────────────────────────
    def summary(self, *, no_target: int, samples_before: int,
                samples_after: int, recognizer_status: str = "",
                end_ts_unix: Optional[float] = None,
                aborted: bool = False) -> dict:
        seen = sum(self._per_seen.values())
        misses = self._decided - self._hits
        abstains = seen - self._decided
        return {
            "session_id": self.session_id,
            "tool": self.tool,
            "start_ts_unix": round(self.start_ts_unix, 3),
            "end_ts_unix": round(float(end_ts_unix), 3) if end_ts_unix else None,
            "aborted": bool(aborted),
            "steps_with_target": seen,
            "no_target": int(no_target),
            "decided": self._decided,
            "hits": self._hits,
            "misses": misses,
            "abstains": abstains,
            # accuracy = hits / decided (of the guesses it committed to)
            "accuracy": round(self._hits / self._decided, 4) if self._decided else 0.0,
            # coverage = decided / seen (how often it ventured a guess)
            "coverage": round(self._decided / seen, 4) if seen else 0.0,
            "mean_conf": round(self._conf_sum / self._decided, 4) if self._decided else 0.0,
            "blocks_seen": len(self._per_seen),
            "samples_before": int(samples_before),
            "samples_after": int(samples_after),
            "samples_added": int(samples_after - samples_before),
            "per_block": {b: {"seen": self._per_seen[b],
                              "hits": self._per_hit.get(b, 0)}
                          for b in sorted(self._per_seen)},
            "recognizer_status": recognizer_status,
        }

    def finalize(self, **summary_kwargs) -> dict:
        """Write the per-step CSV + append the summary line. Best-effort:
        a write failure is reported but never raised into the caller's
        shutdown path."""
        summ = self.summary(**summary_kwargs)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            # Per-step CSV (within-session learning curve).
            if self._steps:
                cols = ["step", "truth", "guess", "conf",
                        "result", "running_acc", "samples"]
                csv_path = self.root / f"{self.session_id}_steps.csv"
                lines = [",".join(cols)]
                for s in self._steps:
                    lines.append(",".join(str(s[c]) for c in cols))
                csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            # Append one summary line to the long-term history.
            hist = self.root / "sessions.jsonl"
            with hist.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(summ, sort_keys=True) + "\n")
        except Exception as e:           # never break shutdown over metrics
            print(f"[metrics][WARN] failed to write session metrics: {e!r}")
        return summ


__all__ = ["SessionMetrics", "default_metrics_root", "HIT", "MISS", "ABSTAIN"]
