# CLAUDE.md — operating guide for Claude working on this repo

Pixel-only Minecraft AI (Java 1.21.x). The bot sees only the game **screen +
F3 debug overlay** and acts through **synthetic keyboard/mouse** — no mod, no
API, no in-game data. `README.md` has the project layout; this file is the
"how to work here safely" guide. Branch: **`world-ai-recognizer`**.

## Golden rules
- **The bot only controls Minecraft when MC is the FOREGROUND window** — input
  is gated to that, and capture is a screen grab. A backgrounded MC = every
  action dropped + a stale frame. If the bot "does nothing" / acts on
  impossible data, FIRST confirm MC is focused (see `tools/diag_capture.py`,
  `tools/diag_control.py`) before debugging behaviour.
- **Keep the offline suite green:** `python tools/run_tests.py` (no MC needed).
  Run it after any change; it's the regression gate.
- **Never commit `data/`** (samples, models, MC assets, logs — gitignored).
- End commit messages with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Commit/push only when the user asks. Panic-stop any live tool with **Ctrl+Shift+F12**.

## The system is a HYBRID (this matters)
- **Perception = real ML.** Block recognition is the bot's main input.
  - `block_cnn.pt` — the live visual recogniser (metric-learning CNN,
    self-trains in the background from F3 labels). `tools/eval_recognizer.py`
    benchmarks it; `tools/train_overnight.py` collects + trains live.
  - `block_fusion.pt` — the context-FUSION recogniser (`vision/world/
    context_fusion.py` + `context_features.py`): visual patch + ~14 context
    inputs (face, biome, light, weather, neighbours, …). Train on a GPU box
    with `tools/train_fusion_model.py`.
  - **F3 OCR** (`vision/ocr.py`, `vision/glyph_ocr.py`) is template matching,
    not ML. It has a per-read time budget — busy/garbled scenes can't blow it
    up (a forest read was 3.7s before the budget; now ~0.25s).
- **Decision-making = hand-coded** FSMs + classical search (A*). `agents/
  treechop.py`, `agents/maker.py`, `agents/skills.py`. Not a learned policy —
  interpretable + debuggable.

## Robust labels + the two maps (don't regress these)
A confirmed block / training label must pass: catalog validity + a garble
gate (F3 `?`-ratio) + **multi-frame agreement** + **crosshair-ray geometry** —
so a one-frame OCR slip never emits a stray id ("never randomly output
sandstone"). `WorldMap.get_confirmed/iter_confirmed` = ground truth (F3
looking-at only); `get_block/iter_blocks` = belief map (incl. CNN guesses).
`wf.looking_at` is the raw read (responsive agent actions); `wf.looking_at_confirmed`
is the gated one (map/training/display).

## ⚠️ Fusion model is VERSION-FROZEN
`context_features.py` `DEFAULT_FEATURES` and the `context_fusion.py` net define
a checkpoint **signature** recorded in `block_fusion.pt`. Changing the feature
set (order/dims) or the architecture **invalidates existing fusion checkpoints**
(load is rejected on signature mismatch). Adding a context input is *meant* to
be one line in `DEFAULT_FEATURES` — but coordinate it with a retrain, and don't
change it casually while another machine is mid-training. Adding new *metadata
keys* in perception is safe (extractors read defensively; unknown keys ignored).
(The `time_of_day` feature was the most recent addition — signature changed, so
re-pull + retrain before relying on `block_fusion.pt` across machines.)

## Weather & time conditioning
Two honest sources, never the old sky-colour guess (it mislabelled a clear
forest ~64% "rain"):
- **Commanded labels (training):** fix a known state in-game (`/weather rain`,
  `/time set night`) and pass `--weather/--time` to `train_overnight.py`, which
  calls `perception.set_environment(...)`. Every sample is labelled with that
  GROUND TRUTH — no detection — so each state trains cleanly + separately. This
  is the reliable path and the one to use for snow (snow isn't its own weather:
  rain falls as snow by biome temperature + altitude, and it's silent — see
  below).
- **Live (occasional):** `vision/subtitle_weather.py` reads MC's subtitle
  captions ("Rain falls"/"Thunder rumbles") via the existing glyph OCR —
  explicit text, position-robust, throttled. A confident `subtitle`/`command`
  verdict is trusted as a label; the raw sky heuristic stays gated behind
  `trust_weather` (default off). **Snow is silent in MC → not caption-readable;
  use the commanded label.**

## Diverse data collection (the curriculum trainer)
`tools/train_curriculum.py` is the autonomous, command-driven way to bank a
DIVERSE dataset (the old `train_overnight.py` only saw one spot). Cheats must be
on; the world is PEACEFUL so there's NO potion-effect use — no mobs/hunger, and
the only death (drowning while still) is prevented by the submerged-guard
re-roam. It **roams** with `/spreadplayers`, sets `/difficulty peaceful` once,
and at each stop cycles `/weather` + `/time`, labelling every sample with that
ground truth via `set_environment`. Corruption guards (the point): sampling is
**suppressed** during transit and on any submerged/lava colour-cast frame
(`is_corrupted_view` → leave immediately, before drowning), and a rain/thunder
label is kept only when the **subtitle reader confirms** precipitation is
actually falling (so commanding rain in a dry desert can't mislabel a clear
scene). Snow IS trained here: land in (or teleport to) a snowy biome, `/weather
rain` → silent snow → labelled "snow". Same focus/panic contract as
`train_overnight`; restores clear/day on exit. `train_overnight.py
--weather/--time` remains the manual single-state path.

## Two-machine workflow
Laptop collects data + runs the live bot; a GPU PC trains. `tools/sync.py`
moves the gitignored data by a single zip (`export-samples`/`import-samples`
content-merge; `export-model`/`import-model`). `tools/doctor.py` is a one-command
env health check (deps, CUDA, data presence) — run it first on a fresh machine.
The GPU build of PyTorch is required for `device=cuda`; the default pip torch is
CPU-only.

## Headline goal + how to run it
`python tools/make.py wooden_pickaxe` — chop logs → craft planks/sticks/table →
place table → 3×3 craft. Needs MC focused. Other live tools in `tools/` (craft,
table_craft, diag_*). More detail in `docs/world_recognizer.md` and
`docs/bot_actions.md`.
