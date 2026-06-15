"""
InventoryMemory — an in-session ledger of what the bot is carrying.

Reading the inventory is expensive (open the screen, render, recognise every
slot, sometimes hover to identify). So the bot should NOT re-open just to ask
"do I already have N of X?". This remembers the last full reading and answers
that from memory, only opening to double-check when it's genuinely unsure.

Policy (the owner's intent — don't fetch/craft what you already have, e.g.
stop making a pile of crafting tables):

  * Remembered total  >= wanted             -> HAVE   (don't open, don't craft).
  * Hotbar alone (always on the HUD) >= wanted -> HAVE (don't open).
  * Otherwise (short, or never read)        -> CHECK  (open + re-read, then act
    on the TRUE deficit) — because a remembered low count may be stale, we
    verify before acquiring so we never over-craft on out-of-date numbers.

Counts come from the inventory recogniser's stack-count OCR (reliable now);
presence is always reliable. Keep the ledger in sync across crafts/pickups
with :meth:`note_delta` so a sequence of "ensure N" calls doesn't each re-open.

Pure state — no screen, no mouse — so it is fully offline-testable. The
controller feeds it via :meth:`observe` on every read.
"""
from __future__ import annotations

from typing import Dict, Tuple

# Carried-storage slot prefixes (exclude the craft grid, armour, result).
_HOTBAR = "hotbar_"


def count_regions(snap) -> Tuple[Dict[str, int], Dict[str, int]]:
    """(main_inventory_counts, hotbar_counts) from a snapshot — item id ->
    summed stack count, in the deep inventory (``inv_*``) and the hotbar
    (``hotbar_*``) SEPARATELY. Kept apart so a HUD-only hotbar read can refresh
    the hotbar without clobbering remembered deep-inventory counts."""
    inv: Dict[str, int] = {}
    hotbar: Dict[str, int] = {}
    for name, sc in (getattr(snap, "slots", {}) or {}).items():
        item = getattr(sc, "item", None)
        if not item:
            continue
        n = max(1, int(getattr(sc, "count", 1) or 1))
        if name.startswith(_HOTBAR):
            hotbar[item] = hotbar.get(item, 0) + n
        elif name.startswith("inv_"):
            inv[item] = inv.get(item, 0) + n
    return inv, hotbar


def inventory_counts(snap) -> Dict[str, int]:
    """item id -> total count across the main inventory + hotbar."""
    inv, hotbar = count_regions(snap)
    out = dict(inv)
    for k, v in hotbar.items():
        out[k] = out.get(k, 0) + v
    return out


def read_open_inventory(ctl):
    """Read a snapshot from an already-open InventoryController, tolerating
    controllers whose ``read`` doesn't accept ``stop_when`` (older builds / test
    doubles). Returns the snapshot, or whatever ``read`` returns (may be None).
    Shared by the inventory arrangers so the fallback isn't copy-pasted."""
    try:
        return ctl.read(stop_when=lambda s: False)
    except TypeError:
        return ctl.read()


class InventoryMemory:
    """What the bot believes it is carrying, learned from inventory reads."""

    def __init__(self):
        self._inv: Dict[str, int] = {}        # deep inventory (from a full read)
        self._hotbar: Dict[str, int] = {}     # hotbar (full read OR HUD read)
        self._delta: Dict[str, int] = {}      # tracked changes since last read
        self._complete = False                # ever captured a full inventory?

    # ── learning ─────────────────────────────────────────────────────
    def observe(self, snap, *, complete: bool = True) -> None:
        """Record counts from a snapshot.

        ``complete=True`` (a normal open-inventory read, which sees every slot)
        REPLACES the whole ledger with fresh truth. ``complete=False`` is a
        partial glimpse — currently the gameplay HUD hotbar — which refreshes
        ONLY the hotbar (items may have left it, so it replaces, not merges)
        and leaves the remembered deep inventory untouched. Either way the
        tracked-delta overlay is cleared (we just saw ground truth)."""
        inv, hotbar = count_regions(snap)
        self._hotbar = hotbar                  # hotbar is fully seen by both reads
        if complete:
            self._inv = inv
            self._complete = True
        self._delta = {}

    def note_delta(self, item: str, delta: int) -> None:
        """Keep the ledger in sync after a KNOWN change — crafted ``+k``, used
        or placed ``-k`` — so the next 'do I have enough?' need not re-open.
        Cleared by the next :meth:`observe` (real read wins)."""
        self._delta[item] = self._delta.get(item, 0) + int(delta)

    def forget(self) -> None:
        """Drop all knowledge (e.g. after actions we couldn't track) so the
        next query opens and re-reads."""
        self._inv = {}
        self._hotbar = {}
        self._delta = {}
        self._complete = False

    # ── querying ─────────────────────────────────────────────────────
    def count(self, item: str) -> int:
        return max(0, int(self._inv.get(item, 0)) + int(self._hotbar.get(item, 0))
                   + int(self._delta.get(item, 0)))

    def hotbar_count(self, item: str) -> int:
        return int(self._hotbar.get(item, 0))

    def deficit(self, item: str, n: int) -> int:
        """How many MORE of ``item`` are needed to reach ``n`` (0 if enough)."""
        return max(0, int(n) - self.count(item))

    @property
    def complete(self) -> bool:
        return self._complete

    def assess(self, item: str, n: int) -> Tuple[str, int]:
        """Decide how to satisfy "want ``n`` of ``item``" WITHOUT opening if we
        can. Returns ``(verdict, k)``:

          * ``("have", 0)``  — already carrying >= ``n`` (skip; don't open).
          * ``("check", n)`` — unsure or short -> open, re-read, then craft/
            fetch :meth:`deficit` (the verified shortfall).

        It deliberately never reports a bare "short" from memory: when the
        ledger looks short we VERIFY first, which is what stops BOTH needless
        opening (when we clearly have enough) AND over-acquiring on a stale
        low count."""
        if int(n) <= 0:
            return ("have", 0)
        if self._complete and self.count(item) >= n:
            return ("have", 0)
        if self.hotbar_count(item) >= n:        # HUD is always visible
            return ("have", 0)
        return ("check", int(n))

    # ── category queries (a SET of item ids — e.g. all log types) ─────
    def count_across(self, items) -> int:
        """Total carried across a set of item ids (any log counts toward
        'want 8 logs')."""
        return sum(self.count(i) for i in set(items))

    def hotbar_across(self, items) -> int:
        return sum(self.hotbar_count(i) for i in set(items))

    def deficit_across(self, items, n: int) -> int:
        return max(0, int(n) - self.count_across(items))

    def assess_across(self, items, n: int) -> Tuple[str, int]:
        """Like :meth:`assess` but over a category of acceptable items."""
        items = set(items)
        if int(n) <= 0:
            return ("have", 0)
        if self._complete and self.count_across(items) >= n:
            return ("have", 0)
        if self.hotbar_across(items) >= n:
            return ("have", 0)
        return ("check", int(n))

    def snapshot(self) -> Dict[str, int]:
        """The remembered totals (for logging/debug)."""
        items = set(self._inv) | set(self._hotbar) | set(self._delta)
        return {it: self.count(it) for it in items if self.count(it) > 0}
