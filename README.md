# minecraft_ai

Pixel-only Minecraft AI. The bot reads the game screen with the same
information a human has — captured frames plus the F3 debug overlay —
and acts through synthetic keyboard / mouse events. No mod, no API, no
in-game data access. Java Edition only, currently 1.21.x.

## Project layout

```
.
├── main.py                 # entry point — runs the agent loop
├── pyproject.toml          # packaging + tool config
├── requirements.txt        # pinned deps (canonical install path)
├── agents/                 # rule-based + future ML agents
├── brain/                  # AgentAction / BaseAgent contract
├── control/                # keyboard, mouse, safety, action wrapper
├── vision/                 # capture, OCR, HUD, world perception
│   └── world/              #   3D voxel map + screen-ray + classifiers
├── knowledge/              # vanilla block / item / entity catalog
├── config/                 # settings.yaml, minecraft.yaml, keymap.json
├── utils/                  # focus / window helpers
├── tools/                  # smoke-tests + dev runners (need running MC)
├── scripts/                # one-off data-collection scripts
├── experiments/            # experiment configs + notes
├── tests/                  # pytest unit tests (run without MC)
├── docs/                   # architecture notes
├── data/                   # runtime artefacts (mostly gitignored)
└── models/                 # trained weights (LFS / out-of-tree)
```

## Quickstart (Windows)

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
# (optional ML / OCR extras)
pip install -e .[ml,ocr]
```

Then launch Minecraft (the Prism instance configured in
`config/minecraft.yaml`) and run:

```powershell
python main.py --agent world_explorer --duration 60
```

`Ctrl+Shift+F12` is the emergency stop — releases every held key,
halts the velocity worker, and tears down the capture loop.

## How it works (one-paragraph version)

Every 50 ms the loop captures a frame, runs HUD + screen-state
extraction, and at ~3 Hz OCRs the F3 overlay to read XYZ / yaw / pitch.
The structured `GameState` goes to the current agent, which returns an
`AgentAction` (movement keys to hold, look velocity, hotbar slot,
interaction). The runtime dispatches each field through the
keyboard / mouse layer with safety gating, then sleeps to the next
tick. The world-perception module builds a sparse voxel map from the
crosshair raycast, the F3 "Looking at block" line, and an inverse-
renderer cross-check.

## Configuration

* `config/settings.yaml` — runtime config (FPS limit, tick rate, safety
  policy, agent tuning, world-perception toggles).
* `config/minecraft.yaml` — mirror of the in-game options (FOV, GUI
  scale, key binds, etc.). When you change a setting in MC, update
  this file too so the perception layer interprets the screen correctly.
* `config/keymap.json` — the action → key map used by the keyboard
  layer; derived from `minecraft.yaml`'s key-binds block.

## Safety

* Refuses to start unless Minecraft is already running.
* Emergency-hotkey (default `Ctrl+Shift+F12`) force-releases every key
  / button and stops the velocity worker bypassing the gate.
* Auto-stops on focus loss (configurable via
  `safety.auto_stop_on_focus_loss`).
* Releases all held keys / buttons on shutdown — including after
  Ctrl+C and after the emergency hotkey.

## Development

Run unit tests (no Minecraft needed):

```powershell
pytest
```

Lint:

```powershell
ruff check .
ruff format --check .
```

Tools that require a running Minecraft instance live in `tools/` (e.g.
`tools/pipeline_test.py`, `tools/test_world_perception_live.py`). They
are runnable scripts, not pytest tests.

## License

MIT — see `LICENSE`.
