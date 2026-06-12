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

# Slots that count as "carried storage" (exclude the craft grid, armour, result).
_STORAGE = ("inv_", "hotbar_")
_HOTBAR = "hotbar_"


def count_regions(snap) -> Tuple[Dict[str, int], Dict[str, int]]:
    """(total_counts, hotbar_counts) from an inventory snapshot — item id ->
    summed stack count, across all storage and across the hotbar alone."""
    total: Dict[str, int] = {}
    hotbar: Dict[str, int] = {}
    for name, sc in (getattr(snap, "slots", {}) or {}).items():
        if not name.startswith(_STORAGE):
            continue
        item = getattr(sc, "item", None)
        if not item:
            continue
        n = max(1, int(getattr(sc, "count", 1) or 1))
        total[item] = total.get(item, 0) + n
        if name.startswith(_HOTBAR):
            hotbar[item] = hotbar.get(item, 0) + n
    return total, hotbar


def inventory_counts(snap) -> Dict[str, int]:
    """item id -> total count across the main inventory + hotbar."""
    return count_regions(snap)[0]


class InventoryMemory:
    """What the bot believes it is carrying, learned from inventory reads."""

    def __init__(self):
        self._counts: Dict[str, int] = {}     # item -> total carried
        self._hotbar: Dict[str, int] = {}     # item -> count in the hotbar
        self._complete = False                # ever captured a full inventory?

    # ── learning ─────────────────────────────────────────────────────
    def observe(self, snap, *, complete: bool = True) -> None:
        """Record counts from an inventory snapshot. A normal open-inventory
        read sees every slot, so ``complete=True`` (the default) REPLACES the
        ledger with the fresh truth. Pass ``complete=False`` for a partial /
        HUD-only glimpse: it's merged in but doesn't claim to be the whole
        picture (so :meth:`assess` still verifies a shortfall)."""
        total, hotbar = count_regions(snap)
        if complete:
            self._counts = total
            self._hotbar = hotbar
            self._complete = True
        else:
            self._counts.update(total)
            self._hotbar.update(hotbar)

    def note_delta(self, item: str, delta: int) -> None:
        """Keep the ledger in sync after a KNOWN change — crafted ``+k``, used
        or placed ``-k`` — so the next 'do I have enough?' need not re-open."""
        self._counts[item] = max(0, self.count(item) + int(delta))
        if delta < 0:                          # spent from somewhere; assume hotbar
            self._hotbar[item] = max(0, self.hotbar_count(item) + int(delta))

    def forget(self) -> None:
        """Drop all knowledge (e.g. after actions we couldn't track) so the
        next query opens and re-reads."""
        self._counts = {}
        self._hotbar = {}
        self._complete = False

    # ── querying ─────────────────────────────────────────────────────
    def count(self, item: str) -> int:
        return int(self._counts.get(item, 0))

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

    def snapshot(self) -> Dict[str, int]:
        """A copy of the remembered totals (for logging/debug)."""
        return dict(self._counts)
