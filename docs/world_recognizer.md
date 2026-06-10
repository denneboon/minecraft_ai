# World block recogniser — architecture

The world recogniser is the project's centrepiece: a **self-teaching,
pixel-only block identifier**. It reads the game screen, guesses what
block is at the crosshair, double-checks the guess against the F3
"Targeted Block" overlay (free ground truth), saves the confirmed pixels
as a labelled sample, and retrains itself — with **no hand-labelling and
no game-internal data**. It improves the more it's used.

This doc is the map of that pipeline. Files are under `vision/world/`
unless noted.

---

## 1. The per-tick pipeline (`perception.py :: WorldPerception.update`)

```
frame (RGB)  +  F3Info (OCR'd debug overlay)
      │
      ▼
1. Pose          _pose_from_f3 → PlayerPose (x,y,z,yaw,pitch,dim,eye_y)
      │            (no pose → blank frame, skip the rest)
      ▼
2. F3 target     f3_target.parse_looking_at_block → LookingAtBlock
      │            ├─ distance gate   (reject OCR garbage > f3_target_max_dist)
      │            └─ CATALOG GATE    (reject ids that aren't real blocks —
      │                                tag-with-eaten-#, truncations, garble)
      ▼
3. Commit truth  world_map.update_block(source="looking_at")   ← ground truth
      │
4. Sample        _maybe_save_crosshair_sample
      │            ├─ SPRITE/THIN-CROP GATE (skip cross/carpet/torch/rail/…
      │            │                          — their crop is background, not
      │            │                          the block; would mislabel)
      │            ├─ distance-normalised crop (frame ~1 block face)
      │            ├─ crosshair inpaint mask
      │            └─ WorldSampleStore.save  → hot-reload recogniser
      ▼
5. Patch sweep   _sweep_patches  (grid of patches, gated off under
      │            commit_only_from_looking_at)
      │            ├─ tiered classify_batch (CNN → NN → colour baseline)
      │            ├─ TEMPORAL VOTE per voxel (majority over recent frames)
      │            └─ commit gated guesses as source="vision_patch"
      ▼
   WorldFrame  (pose, looking_at, visible_blocks, curiosity queue, map)
```

The map has an **authority hierarchy** (`map.py`): `manual > looking_at >
ray_clear_air > vision_patch > extrapolation`. F3 truth always wins over
a vision guess, so a wrong guess can never overwrite a confirmed block.

---

## 2. The gates (why the dataset stays clean)

These are the guardrails that keep the self-teaching loop from poisoning
itself — the single biggest determinant of model quality.

| Gate | File | Rejects | Why |
| --- | --- | --- | --- |
| **Catalog** | `perception.py` + `knowledge/catalog.py` | block ids that aren't real blocks | F3 OCR can drop a `#` (a tag reads as a block), truncate (`short_grass`→`short`), or garble. The parser is structural and can't tell; the catalog (built from the asset jar) is authoritative. |
| **Sprite / thin-crop** | `perception.py` (`sprite_block_predicate`) | cross/tinted_cross/crop/carpet/torch/lantern/rail/button/pressure_plate blocks | A crosshair crop of a thin block is dominated by whatever is *behind* it. Saving it as a label mislabels the data **and** drags down the cube classes. Detected from the asset model parent, so new blocks reusing these templates are caught automatically. The map still gets the block from F3 — only the unreliable *training* capture is skipped. |
| **Distance-normalised crop** | `perception.py` (`_apparent_crop_px`) | — (sizes, not rejects) | A block face spans ~`fx/distance` px. A fixed crop captures only a fragment at range; sizing each crop to ~one block face keeps training and inference at the same scale. |
| **Crosshair mask** | `perception.py` (`_mask_crosshair`) | — (inpaints) | The crosshair sits dead-centre on every sample; inpainting it stops the model learning a synthetic `+`. |

---

## 3. The tiered classifier (`cnn_recognizer.py :: TieredBlockClassifier`)

`CNN → sample-NN → colour baseline`, first confident answer wins.

1. **CNN** (`CNNBlockRecognizer`) — the primary. A tiny (~0.1M-param)
   conv net maps a patch to a 64-d L2-normalised embedding; each block is
   a **prototype** = mean embedding of its samples. `classify` = nearest
   prototype by cosine + a k-NN agreement vote. A new block is just a new
   prototype — no architecture change (futureproof for a growing
   vocabulary). Robust to lighting/biome/angle, which is where the raw-NN
   collapses.
2. **Sample-NN** (`sample_recognizer.py`) — raw-pixel L1 nearest
   neighbour. Good on near-identical captures the CNN is unsure on.
3. **Colour baseline** (`block_classifier.py`) — 28-dim colour signature
   over the vanilla atlas (with biome-tint variants). Always has an
   answer, so the curiosity queue is never starved. Its confidence is
   **capped at 0.50** by the tier wrapper — a different, less reliable
   scale that must never masquerade as a learned commit.

### The CNN, in detail
- **Embedding + prototypes**, not a fixed-head classifier — see above.
- **Background training**: retrains every `retrain_every_n` new samples
  on a daemon thread; never blocks the tick; atomic weight swap.
- **Deterministic**: `torch.manual_seed` in `_train` → same samples
  produce the same model (debuggable, reproducible).
- **Warm-start** from the texture pre-training (`pretrain_block_cnn.py`)
  — flat textures don't transfer for *naming*, but they're a good
  embedding initialiser; real prototypes replace texture ones the moment
  real data exists.
- **Bounded memory**: `reload_incremental` resyncs from disk every 60
  calls so evicted samples don't accumulate over long runs.
- **NaN-safe**: degenerate/non-finite prototypes are skipped, not stored.
- **Graceful**: no torch → CNN disables itself, falls through to NN/baseline.

---

## 4. Temporal voting (`temporal_vote.py`)

When the agent dwells on a voxel, its per-frame guesses are a free
ensemble. `TemporalVoter` returns the majority block id over a recent
tick window with agreement-scaled confidence. It **never suppresses a
first sighting** (single-frame behaviour is unchanged), only corrects
transient flips. Purged on dimension change. Gated by
`vision.world.temporal_voting`.

---

## 5. The sample store (`sample_store.py`)

`data/training/world_samples/<block>/<pixel_hash>.png` (+ `.json`
sidecar with pose/distance/weather/rejected-guess). Content-addressed
(dedup), per-block capped (sliding window: evicts oldest past the cap so
a capture-scheme change can refresh stale samples). Gitignored (local).

---

## 6. Measuring it

| Tool | What it shows |
| --- | --- |
| `tools/eval_recognizer.py` | Deterministic per-block accuracy + confusion matrix (CNN vs raw-NN, clean + augmented) over the store. A/B a code change per-block. |
| `tools/sample_coverage.py` | Dataset health: per-block sample counts, READY/LEARNING/STARVED tiers, and what to collect next. |
| `tools/learn_world_live.py` | The live self-teaching run. `--save-patches` writes annotated guess-vs-truth screenshots; records graph-ready metrics to `data/metrics/`. |
| `tools/plot_metrics.py` | Charts the metrics history: accuracy/coverage per session, sample growth, per-block accuracy, within-session learning curve. |
| `vision/world/metrics.py` | `SessionMetrics` — the JSONL + CSV recorder behind the above. |

---

## 7. Key config (`config/settings.yaml → vision.world`)

| Key | Default | Effect |
| --- | --- | --- |
| `use_cnn_recognizer` / `use_sample_recognizer` | true | enable the learned tiers |
| `auto_sample_from_looking_at` | true | collect a sample on every F3 confirm |
| `commit_only_from_looking_at` | false | when true, ONLY F3 truth commits (no vision-patch); the conservative mode |
| `skip_sprite_samples` | true | the sprite/thin-crop gate |
| `temporal_voting` | true | per-voxel vote smoothing |
| `distance_normalized_crops` | true | size crops to apparent block size |
| `min_block_confidence` | 0.25 | vision-patch commit floor |

---

## 8. Known limitations (honest)

- **Data-bound, not code-bound.** Accuracy is excellent on well-sampled
  cubes (grass_block ~98% live, oak_log/sandstone 1.00 on the benchmark)
  and weak on starved ones. The lever is *more varied samples* — see
  `sample_coverage.py`.
- **Sprite blocks** (short_grass, fern, flowers) are deliberately *not*
  learned — their crosshair crop is visually the backing block, so the
  recogniser sees grass and says grass. This is correct visual behaviour
  but reads as a "miss" against F3's game-logic truth.
- **Glancing-angle contamination**: a thin slice of the target against a
  contrasting neighbour can let the neighbour dominate the crop (e.g.
  grass→oak_log when trunk streaks cross a grass crop). Low frequency.

---

## 9. Where things live

```
vision/world/
  perception.py        orchestrator (the pipeline above) + the gates
  cnn_recognizer.py    CNN + TieredBlockClassifier
  sample_recognizer.py raw-pixel NN tier
  block_classifier.py  colour-signature baseline tier
  biome_tint.py        Mojang per-biome tint tables (shared source of truth)
  temporal_vote.py     per-voxel vote smoothing
  sample_store.py      labelled patch store on disk
  f3_target.py         "Targeted Block" OCR parser
  screen_ray.py        camera model / projection / voxel walk
  map.py               sparse 3-D WorldMap + authority hierarchy
  metrics.py           self-teaching session metrics
knowledge/catalog.py   real-block registry (the catalog gate's source)
tools/                 learn_world_live, eval_recognizer, sample_coverage,
                       plot_metrics, pretrain_block_cnn, test_*
```
