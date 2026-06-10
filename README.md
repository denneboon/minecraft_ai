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

## Commands

Every runnable entry point in the project, grouped by what it needs.
Live tools require Minecraft to be **running and focused**; offline tools
run anywhere. All paths are from the repo root.

### Run the agent / play macros (live MC)

| Command | What it does |
| --- | --- |
| `python main.py --agent world_explorer --duration 60` | Run the agent loop for an agent (`--agent <name>`); `--duration` caps wall-clock seconds. |
| `python main.py --test` | Legacy mouse / hotbar hardware smoke test (no agent). |
| `python main.py --script scripts/macros/bridge.ahk --duration 60` | Play a recorded macro (`.ahk` / `.txt` / `.json` / `.mcs`) instead of an agent. |
| `python tools/run_god_bridge.py` | Adaptive god-bridge runner (pillar-up, auto-align yaw/pitch, diagonal back-strafe). Many flags — `--places`, `--strafe {left,right}`, `--target-pitch`, `--max-seconds`, `--keep-sneak`, `--countdown`. |
| `python tools/run_script.py <file>` | Run (or `--inspect`) a single macro/script file. |
| `python tools/learn_world_live.py` | Autonomous self-teaching live test for the world block recogniser. Records graph-ready metrics to `data/metrics/` each run (`--save-patches` for screenshots, `--no-metrics` to disable). |
| `python tools/plot_metrics.py` | Plot the recogniser's self-teaching progress (accuracy/coverage per session, sample growth, per-block accuracy, within-session learning curve) → PNGs in `data/metrics/`. `--show` to open them. |

### Live diagnostics & calibration (live MC)

| Command | What it does |
| --- | --- |
| `python tools/pipeline_test.py` | End-to-end pipeline smoke test on live frames. |
| `python tools/test_world_perception_live.py` | Live world-perception smoke test (real frames). |
| `python tools/test_mouse_camera.py` | Verify the mouse → camera pipeline against F3. |
| `python tools/test_mouse_visual.py` | Mouse-camera test with before/after screenshots. |
| `python tools/test_px_per_deg.py` | Measure mouse px-per-degree against F3 ground truth. |
| `python tools/test_inventory.py` | Run the full inventory pipeline on a captured/loaded frame. |
| `python tools/window_inspector.py` | Inspect the detected Minecraft window geometry. |
| `python tools/world_map_view.py` | Live top-down view of the AI's world map. |

### Offline tests (no MC needed)

| Command | What it does |
| --- | --- |
| `python tools/run_tests.py` | Run **all** offline self-test suites; `-k <substr>` filters. Exit code non-zero on failure (CI-friendly). |
| `pytest` | Run the pytest unit suite in `tests/`. |
| `python tools/test_world_perception.py` | `vision/world/` smoke test (16 sub-tests). |
| `python tools/test_world_explorer_offline.py` | World-explorer + perception pipeline. |
| `python tools/test_cnn_recognizer.py` | Accuracy test for the self-teaching CNN recogniser. |
| `python tools/test_ocr_f3.py` | F3 overlay OCR on hard backgrounds. |
| `python tools/test_pathfind.py` | WorldMap A* pathfinder. |
| `python tools/test_walker.py` | Voxel-path walker. |
| `python tools/test_weather.py` | `vision.weather`. |
| `python tools/test_script_runner.py` | `control.script_runner`. |
| `python tools/test_inventory_synthetic.py` | Inventory pipeline on synthetic frames. |
| `python tools/eval_recognizer.py --augment` | Deterministic recogniser **benchmark**: per-block accuracy + confusion matrix (CNN vs raw-NN, clean + augmented) over the local sample store. Reproducible for a fixed store — A/B a change per-block. |

### Training & data collection

| Command | What it does |
| --- | --- |
| `python tools/pretrain_block_cnn.py` | Pre-train the block-recognition CNN from game textures (warm-start). |
| `python scripts/collect_demo.py` | Record human demonstrations as (frame, input-state) pairs. |
| `python tools/collect_weather.py` | Collect command-labelled weather training samples. |
| `python tools/record_macro.py` | Record live keyboard/mouse input into a replayable macro. |
| `python tools/record_script_run.py` | Run a macro/script while recording gameplay. |

### Viewers, export & profiling

| Command | What it does |
| --- | --- |
| `python tools/replay_demo.py <session_dir>` | Replay a recorded demo with an input overlay. |
| `python tools/world_view_3d.py <map>` | Interactive 3D viewer for a saved WorldMap. |
| `python tools/export_world.py` | Export a WorldMap snapshot (JSON / schematic). |
| `python tools/profile_loop.py` | Profile the live agent loop stage by stage. |
| `python tools/profile_perception.py` | cProfile the perception + OCR path on live frames. |

`Ctrl+Shift+F12` is the global emergency stop for any live command.

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
