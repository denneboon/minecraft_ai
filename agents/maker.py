"""
Maker — autonomous "make X from scratch".

Ties the pieces together into one goal: given a target item, work out the full
plan (knowledge.recipes.plan_make), GATHER the raw materials it needs, CRAFT the
2x2 intermediates (planks, sticks, a crafting table), and TABLE-CRAFT the final
3x3 recipe — all count-aware (only the shortfall) via the inventory ledger.

The heavyweight, world-touching steps are injected as callbacks so this stays a
thin, offline-testable orchestrator:

  * ``gather_fn(item_id, qty) -> bool`` — acquire ``qty`` of a raw material from
    the world (e.g. chop logs). Returns whether it got enough.
  * ``table_craft_fn(target_id) -> (ok, msg)`` — place a crafting table, open
    it, craft the 3x3 recipe, break the table back.

2x2 crafting uses the injected :class:`Crafter` on the inventory grid; the
ledger (:class:`InventoryMemory`) lets it skip work it doesn't need.
"""
from __future__ import annotations

import time
from typing import Callable, Optional, Tuple

from knowledge.recipes import plan_make
from agents.inventory_memory import inventory_counts

_TABLE = "minecraft:crafting_table"


class Maker:
    def __init__(self, controller, crafter, memory, assets, cat, *,
                 gather_fn: Optional[Callable[[str, int], bool]] = None,
                 table_craft_fn: Optional[Callable[[str], Tuple[bool, str]]] = None,
                 log: Callable[[str], None] = print):
        self.ctl = controller
        self.crafter = crafter
        self.memory = memory
        self.assets = assets
        self.cat = cat
        self.gather_fn = gather_fn
        self.table_craft_fn = table_craft_fn
        self.log = log

    # ------------------------------------------------------------------
    def _read_counts(self) -> dict:
        """Open + full-read the inventory (updates the ledger), return counts.

        The GUI must RENDER and the (threaded) screen capture must deliver a
        fresh POST-open frame before the read is meaningful — a too-early read
        sees the pre-open world and comes back EMPTY, which would wrongly
        trigger a from-scratch re-gather (the live round-2 bug: logs ARE on
        hand but the read missed them). Measured: ~0.6 s settle was still too
        short on this box, ~0.8 s read cleanly. So: a generous first settle,
        then re-read a few times until non-empty before trusting 'empty' as
        real. A genuinely empty inventory just costs the full retry budget once."""
        self.ctl.open_inventory()
        counts: dict = {}
        try:
            time.sleep(0.8)             # GUI render + capture refresh
            for _ in range(3):
                try:
                    snap = self.ctl.read(stop_when=lambda s: False)
                except TypeError:
                    snap = self.ctl.read()
                counts = inventory_counts(snap)
                if counts:
                    break
                time.sleep(0.4)         # let a fresh frame arrive, then retry
        finally:
            self.ctl.close()
        return counts

    # ------------------------------------------------------------------
    def make(self, target_id: str, count: int = 1, *, max_rounds: int = 5
             ) -> Tuple[bool, str]:
        """Make ``count`` of ``target_id`` from scratch. Returns (ok, message).

        Round-based so it stays robust to reality: each round RE-READS and
        RE-PLANS, gathers any raw shortfall (then re-plans — important because
        the species-agnostic gatherer may bring back birch when we'd pencilled
        in oak), and runs the craft chain. Re-planning means the steps always
        match what's actually on hand."""
        short = target_id.split(":")[-1]
        if count <= 0:
            return True, f"need 0 {short}"

        # Cheap HUD glance + ledger first — skip everything if already stocked.
        try:
            if hasattr(self.ctl, "read_hotbar"):
                self.ctl.read_hotbar()
        except Exception:
            pass
        if self.memory.assess(target_id, count)[0] == "have":
            return True, f"already have >={count} {short}"

        last_msg = "no progress"
        prev_state = None
        stuck_rounds = 0
        gather_fails: dict = {}     # raw item -> consecutive rounds it failed
        for rnd in range(max_rounds):
            avail = self._read_counts()
            if avail.get(target_id, 0) >= count:
                return True, f"have {avail.get(target_id, 0)}/{count} {short}"
            # No-progress guard: tolerate a transient hiccup (a craft that
            # failed WITHOUT consuming leaves inventory unchanged, and a retry
            # may well work) but abort if NOTHING changes for two rounds in a
            # row — a genuine blocker (missing ingredient, persistently failing
            # step) won't fix itself, so don't burn the remaining rounds. A
            # partially-done chain DID change inventory, so it keeps retrying.
            state = tuple(sorted(avail.items()))
            if prev_state is not None and state == prev_state:
                stuck_rounds += 1
                if stuck_rounds >= 2:
                    return False, f"stuck — no progress for 2 rounds ({last_msg})"
            else:
                stuck_rounds = 0
            prev_state = state
            plan = plan_make(target_id, count, avail, self.assets, self.cat)
            if plan is None:
                return False, f"no crafting recipe for {short}"
            raw, steps = plan
            raw_summary = ", ".join(f"{k.split(':')[-1]}:{v}" for k, v in raw.items()) or "-"
            self.log(f"[make] round {rnd+1}: have-target=0, gather [{raw_summary}], "
                     f"{len(steps)} craft step(s)")

            # Gather any raw shortfall, then RE-PLAN (loop) from the result.
            if raw:
                if self.gather_fn is None:
                    return False, f"need {raw_summary} but no gather capability"
                progressed = False
                for item, qty in raw.items():
                    self.log(f"[make] gather {qty} {item.split(':')[-1]}")
                    if self.gather_fn(item, qty):
                        progressed = True
                        gather_fails.pop(item, None)
                    else:
                        # Per-item failure counter: a partial gather that keeps
                        # nudging OTHER items' counts changes `state` and resets
                        # stuck_rounds, masking a persistently un-gatherable
                        # item. Abort on one that fails several rounds running.
                        gather_fails[item] = gather_fails.get(item, 0) + 1
                        if gather_fails[item] >= 3:
                            return False, (f"can't gather "
                                           f"{item.split(':')[-1]} "
                                           f"(failed {gather_fails[item]} rounds)")
                if not progressed:
                    return False, f"couldn't gather [{raw_summary}]"
                continue                                  # re-read + re-plan

            # No raw needed -> run the craft chain (2x2 batched; 3x3 via table).
            i = 0
            crafted_ok = True
            made_target = 0          # how many of the TARGET this chain produced
            while i < len(steps):
                if steps[i].needs_table:
                    if self.table_craft_fn is None:
                        return False, "a 3x3 recipe needs a table but no table-craft capability"
                    step = steps[i]
                    res = step.result_id
                    self.log(f"[make] table-craft {res.split(':')[-1]}")
                    ok, msg = self.table_craft_fn(res)
                    last_msg = msg
                    if not ok:
                        crafted_ok = False
                        self.log(f"[make] table-craft {res.split(':')[-1]} failed: {msg}")
                        break
                    i += 1
                    if res == target_id:
                        made_target += step.result_count
                else:
                    self.ctl.open_inventory()
                    try:
                        while i < len(steps) and not steps[i].needs_table:
                            step = steps[i]
                            res = step.result_id
                            ok, msg = self.crafter.craft(res)
                            last_msg = msg
                            self.log(f"[make] craft {res.split(':')[-1]}: "
                                     f"{'OK' if ok else 'FAIL'} ({msg})")
                            if not ok:
                                crafted_ok = False
                                break
                            i += 1
                            if res == target_id:
                                made_target += step.result_count
                    finally:
                        self.ctl.close()
                    if not crafted_ok:
                        break
            # The TARGET was crafted and the crafter VERIFIED its result slot
            # produced output, so trust that — don't depend on a later inventory
            # read RECOGNISING the new item's icon. The recogniser may have no
            # sample for a freshly-made tool (wooden_pickaxe), which would read
            # the target as still-absent and loop us into re-crafting forever
            # (live: pickaxe crafted, then "0/1 after 5 rounds").
            if made_target >= count:
                if self.memory is not None:
                    try:
                        self.memory.note_delta(target_id, made_target)
                    except Exception:
                        pass
                return True, f"made {made_target}x {short}"
            if not crafted_ok:
                # Don't hard-abort on a (possibly transient) craft hiccup —
                # an inventory misread or slot-timing glitch can fail one step.
                # Fall through to the next round: it re-reads the REAL
                # inventory and re-plans, so a partially-done chain resumes
                # from exactly where it is. The no-progress guard at the top
                # aborts if a round genuinely changes nothing, so this can't
                # spin. Bounded by max_rounds either way.
                self.log(f"[make] craft step failed ({last_msg}); re-reading + "
                         f"re-planning (round {rnd+1}/{max_rounds})")
                continue
            # loop: re-read to verify / continue any remaining chain

        avail = self._read_counts()
        have = avail.get(target_id, 0)
        return (have >= count), f"have {have}/{count} {short} after {max_rounds} rounds ({last_msg})"
