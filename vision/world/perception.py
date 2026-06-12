# vision/world/perception.py
"""
WorldPerception — the per-tick orchestrator that ties together:

  * the player pose (from F3 OCR via :class:`vision.ocr.F3Reader`)
  * the rendered frame (from :class:`vision.capture.Capture`)
  * the block classifier (``vision.world.block_classifier``)
  * the entity classifier (``vision.world.entity_classifier``)
  * the persistent :class:`WorldMap`

… and turns them into a :class:`WorldFrame` for the agent.

Pipeline
--------
1. Build a :class:`PlayerPose` from the supplied F3 info. If pose is
   unavailable, return a blank WorldFrame (the agent can still act on
   pixels alone, but nothing world-aware happens this tick).
2. Determine the targeted block:
     * Prefer the "Looking at block" F3 line if it's there
       (high-confidence, exact).
     * Otherwise voxel-walk the crosshair ray and classify the centre
       patch — produces a guess + confidence.
3. Sweep a sparse grid of patches across the gameplay region (avoiding
   the HUD), classify each patch, and project the patch back into the
   world using the camera ray.
4. Run the entity detector on the frame (off by default until a real
   model lands).
5. Update the WorldMap with everything.

Performance budget
------------------
At 20 Hz we have ~50 ms / tick; the rest of the pipeline (capture,
HUD, OCR every 3rd tick) consumes ~10–15 ms. The defaults below were
chosen so a typical update runs in <15 ms on CPU: a 6×4 patch grid is
24 classifications at <0.5 ms each.

If you need higher resolution, bump ``patch_grid_size`` — the
classifier's per-patch cost dominates.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from vision.ocr import F3Info
from vision.world.block_classifier import (
    BlockClassifierProtocol,
    build_block_classifier,
)
from vision.world.entity_classifier import (
    EntityClassifierProtocol,
    build_entity_classifier,
)
from vision.world.f3_target import parse_looking_at_block
from vision.world.temporal_vote import TemporalVoter, TemporalVoteConfig
from vision.world.map import WorldMap
from vision.world.sample_store import (
    WorldSampleStore,
    build_world_sample_store,
)
from vision.world.sample_recognizer import (
    HybridBlockClassifier,
    SampleBlockRecognizer,
)
from vision.world.inverse_renderer import (
    InverseRenderer, InverseRendererConfig,
)
from vision.world.screen_ray import CameraIntrinsics, ScreenRay
from vision.world.types import (
    BlockObservation,
    LookingAtBlock,
    PlayerPose,
    WorldFrame,
)


def _point_in_any_rect(px: int, py: int,
                        rects: Tuple[Tuple[int, int, int, int], ...]
                        ) -> bool:
    """True if (px, py) is inside any of ``rects`` (x, y, w, h)."""
    return any(rx <= px < rx + rw and ry <= py < ry + rh
               for rx, ry, rw, rh in rects)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WorldPerceptionConfig:
    """Tunables for :class:`WorldPerception`."""

    # Horizontal FOV in degrees (must match what MC is rendering).
    h_fov_deg: float = 90.0

    # GUI scale the user is running MC at. Used to size the crosshair
    # mask (vanilla crosshair is 15 GUI px regardless of resolution).
    # ``build_world_perception`` forwards this from
    # ``capture.ui_scale`` so a single setting controls every layer.
    ui_scale: int = 2

    # Patch grid for "look around me" sweeps. (n_cols, n_rows).
    # 24×14 = 336 patches gives roughly one ray per 60×60 pixel block
    # at 1920×1080, dense enough that any visible MC block at typical
    # distance is hit by at least one patch.
    patch_grid_size: Tuple[int, int] = (24, 14)

    # Each patch is this many pixels (square) sampled from the frame.
    # Smaller → faster + noisier; larger → slower + more representative.
    # Only used as a fallback when distance_normalized_crops is off or no
    # depth is known.
    patch_size_px: int = 16

    # ── Distance-normalised cropping ──────────────────────────────
    # A block's apparent on-screen size shrinks with distance: a 1-block
    # face at distance D spans ~fx/D pixels. A FIXED-size crop therefore
    # captures a whole near block but a multi-block jumble of a far one,
    # so the recogniser (trained on ~one-block-face patches) fails at
    # range. When enabled, both the auto-sampler and the patch sweep size
    # their crop to the block's apparent size at its measured distance, so
    # every patch frames ~``crop_block_span`` block-widths regardless of
    # distance — then resize to SAMPLE_SIZE. This is the single biggest
    # lever on in-world recognition accuracy (live-test finding).
    distance_normalized_crops: bool = True
    crop_block_span: float = 1.0      # block-widths to frame per patch
    crop_px_min: int = 24             # floor (crosshair-mask + resolution)
    crop_px_max: int = 140            # ceiling (near blocks fill the crop)

    # How far along the ray (in blocks) to project each patch before
    # giving up. Default matches MC's survival reach + a little for
    # surfaces visible in the distance.
    max_walk_distance: float = 16.0

    # ── Depth-anchored raycasting ─────────────────────────────────
    # When F3 tells us EXACTLY which voxel the crosshair is on, the
    # distance from the eye to that voxel becomes a strong prior for
    # how far away OTHER classified patches probably are. We anchor
    # the sweep at that depth ± a search window so projected patches
    # land on the correct voxel rather than the eye-adjacent one.
    use_f3_depth_anchor: bool = True
    # When no F3 anchor is available, this depth (in blocks) is used
    # as a fallback. Tuned to "what's typically right in front of the
    # player in survival" — a few blocks away.
    default_anchor_depth: float = 4.0
    # ± window around the anchor depth where the surface-finding
    # heuristic tries to land. In blocks. The raycaster picks the
    # voxel inside this window with the strongest classification
    # confidence.
    depth_search_radius:  float = 6.0
    # Carve air along the ray from the eye to each confirmed solid
    # voxel — that volume MUST be transparent or we wouldn't have
    # been able to see through it. This is the AI's self-correction
    # signal for free space.
    carve_air_along_sightlines: bool = True

    # When F3 reads but the "Targeted Block" line is ABSENT, the
    # player is looking at nothing within reach: MC's raytrace went
    # the full ``f3_targeting_reach_blocks`` and found no solid.
    # Every voxel that ray passes through must therefore be air —
    # the strongest free-space signal available short of physically
    # walking through them. ``f3_targeting_reach_blocks`` matches
    # vanilla's ``block_interaction_range`` attribute: 4.5 blocks
    # in survival, ~5 in creative. A slightly conservative 4.5 is
    # safe for both modes.
    carve_air_when_no_target: bool  = True
    f3_targeting_reach_blocks: float = 4.5

    # ── "Absolutely sure" commit gate ─────────────────────────────
    # When ``commit_only_from_looking_at`` is True (the default), the
    # ONLY thing that commits a voxel to the WorldMap is an F3
    # "Targeted Block" confirmation. Vision-patch guesses,
    # sample-NN matches, inverse-renderer cross-validation, and
    # neighbour expansion are all DISABLED. This is the safest
    # mode — the map contains only blocks the AI directly looked at
    # and verified via in-game ground truth. Disable this flag once
    # the smarter paths (cross-validation, expansion) are mature
    # enough to trust.
    commit_only_from_looking_at: bool = True

    # Skip the expensive commit / inverse-render-expansion / patch-sweep
    # work on ticks where the F3 pose timestamp hasn't changed since the
    # last processed tick. With threaded OCR the same pose is handed to
    # ``update`` across several control-loop ticks; re-committing the
    # same voxel and re-expanding the same neighbours each time is pure
    # redundant CPU. Projecting a fresh frame against a STALE pose is
    # also less accurate (pose/frame skew), so skipping is both faster
    # and safer. Map growth then tracks the OCR cadence (~6-8 Hz) rather
    # than the control-loop rate. Set False to force per-tick re-work.
    skip_stale_pose_updates: bool = True

    # Recognise the weather (clear / rain / snow / thunder / unknown)
    # from the frame and attach it to the WorldFrame, so downstream
    # recognition can account for rain/snow changing how blocks look.
    # See ``vision.weather``.
    detect_weather: bool = True
    # Weather changes over MINUTES, so checking it every tick is wasted
    # CPU (and steals GIL time from the far more valuable F3 OCR for
    # pose / looking-at). We re-check at most this often; between checks
    # the last verdict is simply held. The detector itself also holds
    # whenever the sky isn't clearly in view, so a slow cadence is fine.
    weather_check_interval_sec: float = 5.0

    # The legacy gate, kept for back-compat. Active only when
    # ``commit_only_from_looking_at`` is False.
    strict_commit_gate: bool = True
    sample_commit_confidence: float = 0.92

    # ── Inverse-rendering ground-truth expansion ─────────────────
    # When F3 confirms voxel V with block id B, test the 26
    # neighbouring voxels: project each one's visible face to the
    # screen, sample the actual pixels there, compare to the
    # canonical texture for B. Matches above ``expand_score_min``
    # commit to the WorldMap as ``extrapolation`` observations —
    # one F3 confirmation can become 5-25 confirmed voxels for a
    # wall or floor surface. The geometry is exact (sub-pixel
    # accuracy from the F3 yaw/pitch decimals), so this is far
    # more accurate than colour-signature matching.
    use_inverse_renderer:     bool  = True
    expand_score_min:         float = 0.55
    expand_neighbour_radius:  int   = 2     # 1=6 axial neighbours,
                                             # 2 = ±2 in each axis (≈ 124 voxels)
    expand_max_voxels_per_tick: int = 32    # cap CPU per frame

    # Unified wall-clock budget (ms) for ALL the optional heavy work in
    # one ``update`` — inverse-render expansion, curiosity cross-
    # validation, and the patch sweep. Each project+warp+signature
    # ``score_voxel`` is mid-single-digit ms; stacked across 32 expansion
    # voxels + 8 cross-validations + a 24-patch sweep they reach ~150 ms
    # and tank the control-loop rate in block-dense views. Each phase
    # processes its highest-value candidates first and stops when the
    # shared deadline passes — so the control loop holds its rate and
    # the map just grows a little slower in dense scenes. 0 disables the
    # cap (process everything, old behaviour).
    perception_time_budget_ms: float = 30.0

    # Cross-validate curiosity-queue entries against the inverse
    # renderer. If a vision-patch guessed (block_id, voxel) AND the
    # geometric/texture check scores the SAME (block_id, voxel)
    # above this threshold, the two independent signals agree —
    # commit the voxel as "extrapolation" without needing F3
    # confirmation. This lets the map populate from purely visual
    # evidence when F3 looking_at isn't firing (e.g. player on a
    # high platform looking at far terrain).
    inverse_validate_curiosity:    bool  = True
    inverse_validate_score_min:    float = 0.62
    inverse_validate_max_per_tick: int   = 8
    # The sample-NN classifier needs THIS many stored samples on disk
    # before it's allowed to commit anything to the WorldMap by
    # itself. Below this, the NN can match noise to noise and produce
    # large clusters of false-positive cubes around the player.
    # 30 samples ≈ 3-4 distinct block ids; enough variety to discriminate.
    min_samples_for_commit: int = 30
    # Per-block sample floor. Even if the total store has plenty of
    # samples, predictions for a SPECIFIC block id are only trusted if
    # that block has at least this many real labelled patches backing
    # it. Stops the baseline ColourSignatureClassifier (1157 generic
    # block templates) from producing false positives like
    # acacia_leaves / mangrove_leaves / soul_sand when we've only ever
    # captured grass, vines, stone, etc. Unbacked predictions still go
    # to the curiosity queue so the agent can confirm them via F3.
    min_samples_per_block_for_commit: int = 20
    # Cap the curiosity queue so a long session can't grow it
    # unboundedly. Older / lower-confidence entries are evicted first.
    curiosity_queue_max: int = 256

    # Inset from each edge of the frame to skip when sampling the
    # patch grid. The bottom inset is generous because the HUD lives
    # there; the top inset accounts for the "Always" debug-text lines
    # (player_position / section_position) the user has enabled.
    margin_left_px:   int = 80
    margin_right_px:  int = 80
    margin_top_px:    int = 120        # leave room for Always-on F3 lines
    margin_bottom_px: int = 180        # leave room for hotbar + health/hunger

    # Reference resolution for ``margin_*_px`` and ``exclude_rects``.
    # The user calibrates these at a known window size (typically
    # 1920x1129 client-area at GUI scale 2); when the actual captured
    # frame has different dimensions (windowed mode, different
    # monitor, the player resizing MC), the values are auto-scaled by
    # the actual-frame / reference-frame ratio. This keeps the UI
    # exclusion zones lined up with the hotbar / hand / F3 panel
    # without re-calibrating after every window resize.
    frame_ref_resolution: Tuple[int, int] = (1920, 1129)

    # Excluded-rectangle list. Patches whose centre falls inside ANY
    # of these rectangles are skipped during the sweep AND ignored
    # when capturing samples for the recogniser. The HOTBAR + the
    # player's first-person HAND in the bottom-right corner are the
    # canonical entries. Each rect is (x, y, w, h) in screen pixels
    # at ``frame_ref_resolution``; tools/window_inspector.py helps
    # tune these if the user changes GUI scale or item-in-hand
    # animation.
    exclude_rects: Tuple[Tuple[int, int, int, int], ...] = (
        # Hotbar + health/hunger bars: bottom centre + bottom corners.
        (760, 1030, 410, 100),
        # First-person hand: bottom-right ~25 % of the gameplay area.
        (1200,  700, 720, 430),
    )

    # Crosshair raycast settings. Reach in survival is ~4.5 blocks for
    # blocks and ~3.0 for entities; we use a touch above that to also
    # catch the next block beyond what the player could touch.
    crosshair_reach_blocks: float = 5.0

    # Confidence floor below which a vision_patch observation is
    # dropped instead of being written to the WorldMap.
    min_block_confidence: float = 0.25

    # Smooth per-voxel vision-patch guesses over a recent-frame window
    # (majority vote + agreement-scaled confidence). Stabilises commits
    # when the agent dwells on a voxel; a no-op for one-off sightings.
    temporal_voting: bool = True

    # Per-block relative-darkness sample gate. Reject a training sample if
    # it's darker than ``dark_sample_factor`` x the block's running mean
    # brightness (only after ``dark_sample_min_history`` samples seed that
    # mean). Drops night/cave-darkened captures that smear prototypes
    # without culling genuinely-dark blocks. 0.0 disables.
    dark_sample_factor: float = 0.7
    dark_sample_min_history: int = 8

    # Run the entity detector every N ticks (1 = every frame).
    entity_detect_every_n_ticks: int = 4

    # Sanity-cap for the targeted-block distance read out of F3 —
    # used to reject OCR garbage that places a block kilometres
    # away. MC's F3 raytrace reaches further than its interaction
    # reach (typically up to ~20 blocks in 1.20+ with a clear line
    # of sight), so 64 leaves comfortable headroom while still
    # catching the occasional ``-77, 92, -.U?3`` style misread.
    f3_target_max_dist_blocks: float = 64.0

    # ── Self-improvement: auto-sample blocks the F3 overlay names ──
    # When the F3 "Looking at block" line is visible, the perception
    # layer KNOWS which block the player is crosshaired at. We can
    # therefore capture the screen patch under the crosshair and save
    # it as a labelled training sample for the sample-NN recogniser.
    # This is the AI's primary self-improvement signal for block
    # recognition.
    auto_sample_from_looking_at: bool = True

    # Don't keep re-saving the same block from the same exact voxel
    # while the player is staring at it — wait this many ticks between
    # samples from the same (block_id, voxel) tuple.
    sample_cooldown_ticks: int = 6

    # Patch size to sample around the crosshair when collecting. Larger
    # than the perception classifier's ``patch_size_px`` gives the
    # sample store a bit more context (it's stored at SAMPLE_SIZE after
    # downscale either way). 48 = 24 GUI-px at scale 2 — enough block
    # surface around the crosshair for inpainting to fill the cross
    # arms from real texture, not from the cross's own neighbours.
    sample_capture_px: int = 48

    # ── Debugging ────────────────────────────────────────────────
    # When True, every ~N ticks the perception layer dumps the latest
    # F3 raw OCR text plus the parse result. Useful to diagnose
    # "no blocks are being logged" issues without staring at the live
    # game — you can see whether OCR is producing the "Targeted Block"
    # line at all, and if so why parse_looking_at_block isn't matching.
    debug_f3_dump: bool = False
    debug_f3_dump_interval_ticks: int = 100   # 5 s at 20 Hz

    # ── Crosshair masking ─────────────────────────────────────────
    # MC's crosshair sits at the centre of every sample we collect.
    # It inverts the colour of the pixels underneath it (white on
    # dark, dark on bright), which contaminates the sample with a
    # synthetic + shape every future recogniser would learn to match
    # instead of the underlying block texture. We inpaint the cross
    # arms out using cv2.INPAINT_TELEA, propagating the surrounding
    # block texture into the masked region.
    mask_crosshair_in_samples: bool = True
    # Crosshair geometry. Vanilla MC's hud/crosshair.png is 15 GUI px
    # wide × 15 GUI px tall — a 1-px-thick + with arms reaching 7 GUI
    # px each direction from centre. We add a tiny safety margin so
    # inpainting catches the full anti-aliased edge.
    crosshair_arm_half_len_gui: int = 8        # 7 + 1 margin
    crosshair_arm_half_thick_gui: int = 1      # measured 1 GUI px
    # Resource packs / overlays sometimes draw a larger crosshair.
    # If you have one, bump these in settings.yaml.

    # Reload the sample recogniser's in-memory tensor every N saves.
    # Cheap (a few MB at typical sizes) and lets the recogniser pick
    # up the newest samples without waiting for a process restart.
    sample_reload_every_n_saves: int = 3


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class WorldPerception:
    """
    Update a persistent :class:`WorldMap` from screen frames + player
    pose every tick.

    Construction is light — the heavy lift (building the block
    signature library) lives inside the classifier. Calling
    ``update(...)`` is the per-tick hot path.

    Thread safety
    -------------
    Not thread-safe. WorldPerception expects to be called from a
    single owner — the agent loop in ``main.py``.
    """

    def __init__(self,
                 *,
                 config: Optional[WorldPerceptionConfig] = None,
                 block_classifier: Optional[BlockClassifierProtocol] = None,
                 entity_classifier: Optional[EntityClassifierProtocol] = None,
                 world_map: Optional[WorldMap] = None,
                 sample_store: Optional[WorldSampleStore] = None,
                 inverse_renderer: Optional[InverseRenderer] = None,
                 weather_detector: Optional[Any] = None,
                 block_id_validator: Optional[Callable[[str], bool]] = None,
                 sprite_block_predicate: Optional[Callable[[str], bool]] = None):
        self.cfg = config or WorldPerceptionConfig()
        self.block_classifier  = block_classifier
        self.entity_classifier = entity_classifier or build_entity_classifier()
        self.world_map         = world_map or WorldMap()
        self.sample_store      = sample_store
        self.inverse_renderer  = inverse_renderer
        # Optional gate that answers "is this string a real block id?".
        # When supplied (built from the asset catalog), an F3 ``looking_at``
        # read whose id is NOT a known block is rejected before it can
        # commit to the WorldMap OR be saved as a training sample. This
        # is the guard that stops OCR garbage — a tag whose leading ``#``
        # got eaten (``goats_spawnable_on``), a truncation (``short`` from
        # ``short_grass``), or a mangled stem (``azalea_grc``) — from
        # poisoning the self-teaching dataset. ``None`` disables the check
        # (back-compat for the offline tests, which build perception by
        # hand without a catalog).
        self._block_id_validator = block_id_validator
        # Predicate: "is this block a thin sprite (cross/tinted_cross model
        # — short_grass, fern, flowers, saplings, crops, …)?". A crosshair
        # patch of such a block is dominated by whatever is BEHIND the
        # blade (the backing block / a tree trunk), so saving it as a
        # training sample for the sprite MISLABELS it — and that noise
        # measurably drags down the cube classes too (verified: adding
        # foliage samples regressed held-out accuracy). When set, the
        # auto-sampler SKIPS these blocks (they still commit to the
        # WorldMap from F3 ground truth — only the unreliable TRAINING
        # capture is suppressed). ``None`` disables the skip (back-compat).
        self._sprite_block_predicate = sprite_block_predicate
        self._sprite_skips = 0
        # Per-block running mean brightness (for the relative-darkness gate)
        # and the count of samples that have seeded it.
        self._block_brightness: Dict[str, float] = {}
        self._block_brightness_n: Dict[str, int] = {}
        self._dark_skips = 0
        # Per-voxel temporal vote smoother for vision-patch guesses. When
        # the agent looks at the same voxel across frames, voting over the
        # independent guesses is a free ensemble — far more stable than any
        # single noisy frame. Never suppresses a first sighting, so
        # single-frame behaviour (and every offline test) is unchanged.
        self._temporal_voter = TemporalVoter(
            TemporalVoteConfig()) if self.cfg.temporal_voting else None
        # Weather recogniser (optional). Built lazily here when enabled
        # and not injected, so existing call-sites/tests keep working.
        self.weather_detector = weather_detector
        if self.weather_detector is None and self.cfg.detect_weather:
            try:
                from vision.weather import WeatherDetector
                self.weather_detector = WeatherDetector()
            except Exception as e:
                print(f"[perception][WARN] weather detector unavailable: {e!r}")
                self.weather_detector = None
        self._last_weather = None
        self._tick             = 0
        self._screen_ray:      Optional[ScreenRay] = None
        self._last_frame_shape: Optional[Tuple[int, int]] = None
        # (block_id, voxel) -> last tick we sampled it; throttles repeats.
        self._last_sample_at: Dict[Tuple[str, Tuple[int, int, int]], int] = {}
        self._samples_saved: int = 0
        self._samples_saved_since_reload: int = 0
        # Stats useful for the live viewer / diagnostics.
        self._n_corrections: int = 0      # times a sample overrode a stale guess
        # Count of distinct voxels F3 confirmed for the first time during
        # this process lifetime — drives the "[perception] LOGGED ..."
        # heartbeat so the user can see the agent making progress.
        self._n_new_confirmed_this_run: int = 0
        # Per-source commit counters. Lets the user see exactly how many
        # voxels each pipeline stage put into the WorldMap. Crucial when
        # ``commit_only_from_looking_at`` is False: it's the FIRST place
        # you look if the map starts filling with garbage.
        #   ``looking_at`` — F3 ground truth (the conservative baseline).
        #   ``vision_patch`` — sample-NN matches from the patch sweep.
        #   ``extrapolation`` — inverse-renderer expansion of F3 seeds.
        # Each key is a (source, block_id) pair so we can spot a single
        # block id flooding the map from a misclassification.
        self._commits_by_source: Dict[Tuple[str, str], int] = {}
        # Per-(guessed-block, actual-block) curiosity-correction counter.
        # Incremented when F3 confirms a voxel that the patch sweep had
        # ALREADY classified (and parked in the curiosity queue) with a
        # different block id. These never reached the WorldMap — so they
        # don't fall under ``_corrections_by_source`` — but they're the
        # purest signal of "how often is the colour-signature baseline
        # wrong, and what does it confuse for what." Drives dataset
        # priorities: a heavy ``(acacia_leaves -> oak_leaves)`` entry
        # means acacia_leaves should be the next block we collect.
        self._curiosity_corrections: Dict[Tuple[str, str], int] = {}
        # Per-(source, block) correction counter — incremented whenever
        # F3 ground truth contradicts a previously-committed guess.
        # If ``vision_patch`` racks up many corrections for the same id,
        # that's a clear "tighten the confidence threshold" signal.
        self._corrections_by_source: Dict[Tuple[str, str], int] = {}
        # Most recent successfully-parsed pose. Other modules (e.g. the
        # main loop's shutdown handler) read this to render a final
        # map snapshot.
        self._last_pose: Optional[PlayerPose] = None
        # Curiosity queue: voxels classified by vision but NOT confidently
        # enough to commit to the WorldMap. Keyed by voxel so a single
        # noisy voxel can't fill the queue; value is the most recent
        # diagnostic about why it's uncertain. Agents (like
        # WorldExplorerAgent) read this to decide which voxel to aim at
        # next for ground-truth confirmation.
        self._curiosity: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        # Voxels we've already CONFIRMED via F3 — never re-add them to the
        # curiosity queue, even if a noisy patch sweep classifies them
        # again. Prevents the explorer from re-investigating known blocks.
        self._confirmed: set = set()
        # Tracks the F3Info.timestamp of the previously processed read
        # so we can fire the debug dump on EACH fresh OCR cycle instead
        # of on every tick (most ticks reuse the same cached F3Info
        # produced by the main loop). Re-using stale F3 would spam
        # the same dump 7 times per cycle.
        self._last_f3_dump_ts: Optional[float] = None
        # Timestamp of the last F3 read we did the FULL commit/expand
        # work for. Lets ``update`` skip redundant re-work when the same
        # pose is handed in across several control ticks (threaded OCR).
        self._last_processed_ts: Optional[float] = None
        # Per-update wall-clock deadline shared by the heavy phases.
        # Set at the top of each non-skipped update; None = no budget.
        self._tick_deadline: Optional[float] = None
        # Wall-clock of the last weather re-check (throttled — see
        # ``weather_check_interval_sec``). -inf so the first tick checks.
        self._last_weather_check: float = float("-inf")

    # ── Public API ─────────────────────────────────────────────────

    def update(self,
               frame_rgb: np.ndarray,
               f3: Optional[F3Info],
               ) -> WorldFrame:
        """
        Run one tick of world perception.

        Parameters
        ----------
        frame_rgb : full-resolution RGB frame from Capture.
        f3        : the latest F3Info (may be None if OCR didn't fire
                    this tick — perception will still produce a frame
                    with ``pose = None``).
        """
        self._tick += 1
        wf = WorldFrame(tick=self._tick, world_map=self.world_map)

        # Weather recognition — independent of pose (works in a cave →
        # "unknown", or outdoors with F3 off). THROTTLED: weather changes
        # over minutes, so we re-check only every
        # ``weather_check_interval_sec`` and hold the verdict in between.
        # This keeps the main thread (and the GIL) free for the F3 OCR
        # worker, which is the high-value perception (pose / looking-at).
        # Best-effort: a detector failure must never abort perception.
        # The held result rides on every WorldFrame (incl. the pose-None
        # / stale-skip early returns below).
        _now = time.perf_counter()
        if (self.weather_detector is not None and frame_rgb is not None
                and (_now - self._last_weather_check)
                >= self.cfg.weather_check_interval_sec):
            self._last_weather_check = _now
            try:
                # Pass pitch so the detector can skip frames where the
                # camera is pitched down (top band isn't sky). f3.pitch
                # is the raw read; None when pose is unavailable.
                _pitch = getattr(f3, "pitch", None) if f3 is not None else None
                w = self.weather_detector.detect(frame_rgb, pitch=_pitch)
                # Log only on a state TRANSITION so a long run doesn't
                # spam, but the user can see the agent noticing weather.
                if w is not None and (self._last_weather is None
                                      or w.state != self._last_weather.state):
                    prev = self._last_weather.state if self._last_weather else "?"
                    print(f"[weather] {prev} -> {w.state} "
                          f"(conf={w.confidence:.2f}, sky={w.sky_visible:.2f}, "
                          f"src={w.source})")
                self._last_weather = w
            except Exception as e:
                if not getattr(self, "_weather_warn_emitted", False):
                    self._weather_warn_emitted = True
                    print(f"[perception][WARN] weather detect failed: {e!r} "
                          f"(further errors silenced)")
        wf.weather = self._last_weather

        # F3 diagnostic dump must fire BEFORE the pose-None early-out:
        # when pose fails to parse (most common during scan / no-target
        # streaks) the dump is exactly what tells us WHY. Show every
        # parsed field plus the first ~320 chars of raw text. Throttled
        # to one print per fresh F3Info.timestamp so we don't repeat
        # the same dump 7 times across the ticks that reuse the same
        # cached OCR result between 3 Hz reads.
        if self.cfg.debug_f3_dump and f3 is not None:
            ts = getattr(f3, "timestamp", None)
            if ts is not None and ts != self._last_f3_dump_ts:
                self._last_f3_dump_ts = ts

                def _fmt(v, fmt: str = ".2f") -> str:
                    return ("?" if v is None
                            else format(v, fmt) if isinstance(v, (int, float))
                            else str(v))

                print(f"[F3] tick={self._tick} backend={f3.backend}  "
                      f"xyz=({_fmt(f3.x, '.3f')}, {_fmt(f3.y, '.3f')}, "
                      f"{_fmt(f3.z, '.3f')})  "
                      f"yaw={_fmt(f3.yaw, '.1f')} pitch={_fmt(f3.pitch, '.1f')}  "
                      f"facing={_fmt(f3.facing_name, 's')} "
                      f"dim={_fmt(f3.dimension, 's')}  "
                      f"block={_fmt(f3.block_x, 'd')},"
                      f"{_fmt(f3.block_y, 'd')},"
                      f"{_fmt(f3.block_z, 'd')}")
                raw = (f3.raw_text or "").replace("\n", " | ")
                if len(raw) > 1600:
                    raw = raw[:1597] + "..."
                # ascii(): garbled OCR text routinely contains box-drawing
                # / symbol glyphs (─, ∙, …) that a cp1252 Windows console
                # can't encode — a bare print(raw!r) raises UnicodeEncodeError
                # and would crash the whole perception tick. ascii() escapes
                # non-ASCII to \uXXXX so the diagnostic is always printable.
                print(f"[F3]   raw: {ascii(raw)}")

        pose = self._pose_from_f3(f3)
        if pose is None:
            return wf
        wf.pose = pose
        # Detect dimension change BEFORE updating ``_last_pose``. Voxel
        # coords are dimension-specific (nether ↔ overworld map to
        # different scales AND completely separate block states), so
        # any curiosity entry / confirmed-voxel set carried over from
        # the prior dimension is stale by definition. Purge them so
        # the agent doesn't waste cycles aiming at coords that no
        # longer correspond to anything visible.
        prev_dim = (self._last_pose.dimension
                    if self._last_pose is not None else None)
        if (pose.dimension is not None and prev_dim is not None
                and pose.dimension != prev_dim):
            print(f"[perception] DIMENSION CHANGE {prev_dim} -> "
                  f"{pose.dimension}; purging {len(self._curiosity)} "
                  f"curiosity entries and {len(self._confirmed)} "
                  f"confirmed voxels")
            self._curiosity.clear()
            self._confirmed.clear()
            if self._temporal_voter is not None:
                self._temporal_voter.purge()   # voxels differ across dims
        self._last_pose = pose
        if pose.dimension:
            self.world_map.set_current_dimension(pose.dimension)

        # Lazy-init the screen-ray with whichever resolution the
        # capture actually produces. Re-init if the frame size changes.
        # When the frame is None on this tick we DO NOT keep a stale
        # screen-ray around for downstream code — clearing it here is
        # cheap and the next valid frame rebuilds it anyway.
        shape = frame_rgb.shape[:2] if frame_rgb is not None else None
        if shape is None:
            self._screen_ray = None
            self._last_frame_shape = None
        elif shape != self._last_frame_shape:
            intr = CameraIntrinsics.from_frame(
                width=shape[1], height=shape[0],
                h_fov_deg=self.cfg.h_fov_deg,
            )
            self._screen_ray = ScreenRay(intrinsics=intr)
            self._last_frame_shape = shape
        sr = self._screen_ray
        if sr is None or frame_rgb is None:
            return wf

        # Stale-pose skip: with threaded OCR the same F3Info (same
        # timestamp) is handed to ``update`` across several control
        # ticks. Re-committing the same voxel + re-expanding the same
        # neighbours every tick is redundant CPU, and projecting a fresh
        # frame against a stale pose only adds skew. Do the heavy work
        # once per fresh pose; ``wf.pose`` is already set so the agent
        # still gets the current pose this tick.
        f3_ts = getattr(f3, "timestamp", None) if f3 is not None else None
        if (self.cfg.skip_stale_pose_updates
                and f3_ts is not None
                and f3_ts == self._last_processed_ts):
            return wf
        self._last_processed_ts = f3_ts

        # Shared wall-clock deadline for all optional heavy phases this
        # tick (expansion / cross-validation / patch sweep). ``None``
        # when the budget is disabled. Phases process best-first and
        # bail once this passes, bounding the worst-case tick.
        budget_s = max(0.0, float(self.cfg.perception_time_budget_ms) / 1000.0)
        self._tick_deadline = (time.perf_counter() + budget_s) if budget_s else None

        eye = (pose.x, pose.eye_y, pose.z)
        anchor_depth: Optional[float] = None  # filled if we get F3 ground truth
        # When `target_was_rejected` is True we KNOW the player was
        # looking at some block but couldn't parse it cleanly — so
        # we MUST NOT mark the forward ray as air (there's actually
        # a block in it somewhere). Only the genuinely-absent case
        # — F3 reading succeeded but no Targeted-Block line was
        # present — triggers air carving.
        target_was_rejected = False

        # 1. F3 "Looking at block" line — exact, if visible.
        la = None
        # Detect whether MC's F3 panel CONTAINS a Targeted-Block-style
        # line even if we couldn't parse coords from it. Used below to
        # set ``target_was_rejected`` so the forward-ray air-carving
        # path knows the player WAS looking at something — just not
        # parseable — and refuses to mark voxels along that ray as air.
        # Without this, an OCR cycle that reads the label but garbles
        # the coords would mistakenly "see free space" through a wall.
        label_seen_unparsed = False
        if f3 is not None and f3.raw_text:
            la = parse_looking_at_block(f3.raw_text.splitlines())
            if la is None:
                low = f3.raw_text.lower()
                if "targeted" in low or "looking at" in low:
                    label_seen_unparsed = True

        # Diagnostic dump on every FRESH F3 read. The main loop hands
        # the same F3Info to perception across all 20 Hz ticks between
        # 3 Hz OCR calls, so we key off ``f3.timestamp`` to fire the
        # dump exactly once per real OCR cycle.
        #
        # (The numeric-field dump moved earlier in ``update()`` so it
        # fires even when pose fails to parse — we still want to see
        # WHY pose is missing. Here we add the looking_at parse
        # result onto the most recent dump line; the numeric dump
        # above will already have run for this same fresh timestamp.)
        if (self.cfg.debug_f3_dump and f3 is not None
                and getattr(f3, "timestamp", None) is not None):
            parse_str = (
                f"id={la.block_id} pos={la.pos} conf={la.confidence}"
                if la is not None else "<no match>"
            )
            print(f"[F3]   looking_at: {parse_str}")

        if f3 is not None and f3.raw_text:
            # Label was present but the parser couldn't extract coords.
            # Treat as a rejection so the air-carving path stays safe.
            if label_seen_unparsed:
                wf.diagnostics.setdefault("looking_at_rejects", []).append(
                    {"reason": "no_coords"})
                target_was_rejected = True
            # SANITY-check the targeted-block distance against the eye.
            # The old check rejected anything beyond
            # ``crosshair_reach_blocks + 1.5`` (≈ 6.5 blocks) on the
            # assumption that F3 only reports interaction-range
            # targets — but in MC 1.20+ F3's raytrace distance is much
            # larger than the interaction-reach attribute and routinely
            # reports targets 15-20 blocks away when the crosshair has
            # a clear line of sight. That made the agent reject valid
            # F3 reads in jungles and on hilltops.
            #
            # Now we ONLY reject obvious OCR garbage: coordinates more
            # than ``f3_target_max_dist_blocks`` from the eye (default
            # 64). That covers MC's longest plausible raytrace range
            # while still catching parse errors that put the target on
            # the other side of the world.
            if la is not None:
                tcx = la.pos[0] + 0.5
                tcy = la.pos[1] + 0.5
                tcz = la.pos[2] + 0.5
                d = math.sqrt((tcx - eye[0]) ** 2
                              + (tcy - eye[1]) ** 2
                              + (tcz - eye[2]) ** 2)
                max_dist = self.cfg.f3_target_max_dist_blocks
                if d > max_dist:
                    wf.diagnostics.setdefault("looking_at_rejects", []).append(
                        {"reason": "out_of_reach", "pos": list(la.pos),
                         "distance": round(d, 2),
                         "id": la.block_id})
                    target_was_rejected = True
                    la = None
            # Catalog plausibility gate. The F3 parser is structural — it
            # can't tell a real block id from a tag whose ``#`` was eaten
            # by OCR, a truncated stem, or a glyph-mangled name. Cross-check
            # the id against the known-block catalog so garbage never reaches
            # the WorldMap or the sample store (both live under the
            # ``if la is not None`` block below). Unknown ids are dropped
            # like an out-of-reach read.
            if la is not None and self._block_id_validator is not None:
                try:
                    is_known = self._block_id_validator(la.block_id)
                except Exception:
                    is_known = True   # never let a validator bug drop real reads
                if not is_known:
                    wf.diagnostics.setdefault("looking_at_rejects", []).append(
                        {"reason": "unknown_block_id", "pos": list(la.pos),
                         "id": la.block_id})
                    target_was_rejected = True
                    la = None
            if la is not None:
                wf.looking_at = la

                # Curiosity-queue correction signal. The voxel F3 just
                # confirmed may have been parked in the curiosity queue
                # under a different block id (the patch sweep guessed
                # one thing, F3 says another). The entry never reached
                # the WorldMap — the commit gate caught it — but the
                # disagreement is still the cleanest "where is the
                # classifier wrong" signal we have. Tally before
                # removing the entry from the queue so the entry's
                # block_id is available.
                curio_entry = self._curiosity.get(la.pos)
                if (curio_entry is not None
                        and curio_entry.get("block_id") is not None
                        and curio_entry["block_id"] != la.block_id):
                    ck = (curio_entry["block_id"], la.block_id)
                    self._curiosity_corrections[ck] = \
                        self._curiosity_corrections.get(ck, 0) + 1
                    print(f"[perception] CURIO-CORRECTION at {la.pos}: "
                          f"guess={curio_entry['block_id']} -> "
                          f"F3={la.block_id}")

                # Self-improvement signal: the F3 overlay just told us
                # EXACTLY what block is under the crosshair. If we had
                # previously classified this same voxel as something
                # else from vision_patch, that was a wrong guess — bump
                # the correction counter so the diagnostics surface it.
                prev = self.world_map.get_block(la.pos,
                                                 dimension=pose.dimension)
                if (prev is not None
                        and prev.block_id is not None
                        and prev.block_id != la.block_id
                        and prev.source in ("vision_patch", "extrapolation")):
                    self._n_corrections += 1
                    # Tally the corrected (source, wrong-block) so we
                    # can spot a single source repeatedly mislabelling
                    # one block id — e.g. ``("vision_patch", "stone")``
                    # spiking means we should tighten the NN
                    # confidence threshold for stone specifically.
                    ckey = (prev.source, prev.block_id)
                    self._corrections_by_source[ckey] = \
                        self._corrections_by_source.get(ckey, 0) + 1
                    wf.diagnostics.setdefault("corrections", []).append({
                        "pos": list(la.pos),
                        "was": prev.block_id,
                        "now": la.block_id,
                    })
                    # Real-time log so we can SEE the agent
                    # self-correcting during a run.
                    print(f"[perception] CORRECTION at {la.pos}: "
                          f"{prev.source} said {prev.block_id} → F3 says "
                          f"{la.block_id}")

                self._commit(BlockObservation(
                    pos          = la.pos,
                    block_id     = la.block_id,
                    confidence   = la.confidence,
                    source       = "looking_at",
                    last_seen_tick = self._tick,
                    dimension    = pose.dimension,
                    meta         = {"face": la.face} if la.face else {},
                ))
                # Visible LOG line the FIRST time each voxel is
                # confirmed. Repeated confirmations of the same voxel
                # are silent so a long stare doesn't spam the console.
                # The world_map is the authoritative store; this is
                # purely a user-visible heartbeat.
                if la.pos not in self._confirmed:
                    self._n_new_confirmed_this_run += 1
                    short_id = la.block_id.replace("minecraft:", "")
                    print(f"[perception] LOGGED {short_id} @ "
                          f"({la.pos[0]}, {la.pos[1]}, {la.pos[2]})  "
                          f"[#{self._n_new_confirmed_this_run}]")
                # Once F3 confirms it, the voxel leaves the curiosity
                # queue and never re-enters it.
                self._confirmed.add(la.pos)
                self._curiosity.pop(la.pos, None)
                # Inverse-rendering expansion: project neighbour
                # voxels' faces onto the screen, compare actual
                # pixels to the canonical texture for the same
                # block id. DISABLED while
                # ``commit_only_from_looking_at`` is on — neighbour
                # voxels are NOT what the player is "directly
                # looking at", so they don't satisfy the strict
                # guarantee. Will be re-enabled later once the
                # geometric check has been tightened enough to be
                # trusted as ground truth.
                if (not self.cfg.commit_only_from_looking_at
                        and self.cfg.use_inverse_renderer
                        and self.inverse_renderer is not None):
                    n_extra = self._expand_via_inverse_render(
                        frame_rgb=frame_rgb,
                        sr=sr,
                        eye=eye,
                        pose=pose,
                        seed_voxel=la.pos,
                        seed_block_id=la.block_id,
                    )
                    if n_extra:
                        wf.diagnostics["inverse_render_expanded"] = n_extra

                # Carve free space along the F3-confirmed sightline.
                # If MC says the targeted block is at P, every voxel
                # between the eye and P MUST be transparent (we saw
                # through them). This is the strongest "free space"
                # signal the perception layer can collect.
                if self.cfg.carve_air_along_sightlines:
                    cleared = self._air_voxels_along(
                        eye=eye, target_voxel=la.pos, sr=sr, pose=pose,
                    )
                    self.world_map.mark_air_along(
                        cleared,
                        dimension=pose.dimension,
                        confidence=0.95,
                        tick=self._tick,
                    )

                # Anchor depth for the patch sweep below.
                tx, ty, tz = la.pos
                center = (tx + 0.5, ty + 0.5, tz + 0.5)
                anchor_depth = math.sqrt(
                    (center[0] - eye[0]) ** 2
                    + (center[1] - eye[1]) ** 2
                    + (center[2] - eye[2]) ** 2
                )
                wf.diagnostics["anchor_depth"] = round(anchor_depth, 2)

                # Auto-sample the crosshair patch labelled with the
                # F3 ground-truth block id. This is what trains the
                # sample-based recogniser between sessions. We attach
                # a metadata sidecar recording the pose + depth + the
                # classifier guess (if any) that was wrong — useful
                # training context for any downstream ML.
                if (self.cfg.auto_sample_from_looking_at
                        and self.sample_store is not None):
                    rejected = wf.diagnostics.get("corrections", [])
                    last_rejected = (rejected[-1].get("was")
                                      if rejected else None)
                    # Distance from eye to target voxel centre. Reused
                    # by the sample-store metadata sidecar so future ML
                    # can condition on view angle / depth.
                    tcx = la.pos[0] + 0.5
                    tcy = la.pos[1] + 0.5
                    tcz = la.pos[2] + 0.5
                    distance_blocks = round(math.sqrt(
                        (tcx - eye[0]) ** 2 + (tcy - eye[1]) ** 2
                        + (tcz - eye[2]) ** 2
                    ), 3)
                    metadata = {
                        "tick": self._tick,
                        "pose": {
                            "x": pose.x, "y": pose.y, "z": pose.z,
                            "yaw": pose.yaw, "pitch": pose.pitch,
                            "dimension": pose.dimension,
                        },
                        "target_voxel": list(la.pos),
                        "distance_blocks": distance_blocks,
                        "rejected_guess": last_rejected,
                        # Weather under which this block sample was
                        # captured — lets future training condition block
                        # appearance on rain/snow/clear instead of
                        # blending wet + dry views of the same block.
                        "weather": (self._last_weather.state
                                    if self._last_weather is not None else None),
                        "source": "f3_looking_at",
                    }
                    self._maybe_save_crosshair_sample(
                        frame_rgb=frame_rgb,
                        sr=sr,
                        block_id=la.block_id,
                        voxel=la.pos,
                        metadata=metadata,
                    )

        # 1b. NO targeted block this frame → if pose was fresh AND
        # the F3 panel didn't have a Targeted-Block line at all (NOT
        # a rejected-as-garbage one), MC's raytrace went the full
        # block_interaction_range and hit nothing. Every voxel along
        # the forward ray must be air — the strongest free-space
        # deduction we get without an actual block reference.
        #
        # ``target_was_rejected`` guards against the failure mode
        # where OCR DID see "Targeted Block:" but couldn't parse
        # coords / reach-checked out — we mustn't mark air there
        # because there's actually a real block in the ray.
        if (wf.looking_at is None
                and not target_was_rejected
                and f3 is not None and f3.x is not None
                and self.cfg.carve_air_when_no_target):
            forward = sr.forward_vector(pose.yaw, pose.pitch)
            cleared = self._air_voxels_in_ray(
                eye=eye, direction=forward,
                max_distance=self.cfg.f3_targeting_reach_blocks,
            )
            n_cleared = self.world_map.mark_air_along(
                cleared,
                dimension=pose.dimension,
                confidence=0.85,
                tick=self._tick,
            )
            if n_cleared:
                wf.diagnostics["air_voxels_carved_no_target"] = n_cleared

        # Patch sweep only runs when F3 looking_at gave us a depth
        # anchor for this frame. Without one, the default depth
        # would project every classified patch to a phantom shell
        # of cubes around the player. Under
        # ``commit_only_from_looking_at`` the patch sweep is also
        # entirely useless (its outputs can't reach the WorldMap),
        # so we skip it then too — saves CPU.
        skip_patch_sweep = (
            anchor_depth is None
            or self.cfg.commit_only_from_looking_at
        )

        # 2. Crosshair vision-patch (only used when F3 didn't give us
        # the targeted block — F3 is always preferred). Skipped
        # entirely under ``commit_only_from_looking_at`` because the
        # observation it produces is a vision-patch GUESS, which the
        # strict-gate user explicitly does not want in the WorldMap.
        if (self.block_classifier is not None
                and wf.looking_at is None
                and not self.cfg.commit_only_from_looking_at):
            crosshair_obs = self._crosshair_observation(
                frame_rgb, sr, pose, anchor_depth,
            )
            if crosshair_obs is not None and self._should_commit(crosshair_obs):
                wf.visible_blocks.append(crosshair_obs)
                wf.looking_at = LookingAtBlock(
                    block_id   = crosshair_obs.block_id or "unknown",
                    pos        = crosshair_obs.pos,
                    confidence = crosshair_obs.confidence,
                )
                self._commit(crosshair_obs)
                if self.cfg.carve_air_along_sightlines:
                    cleared = self._air_voxels_along(
                        eye=eye, target_voxel=crosshair_obs.pos,
                        sr=sr, pose=pose,
                    )
                    self.world_map.mark_air_along(
                        cleared,
                        dimension=pose.dimension,
                        confidence=0.7,
                        tick=self._tick,
                    )

        # 3. Sweep the patch grid (depth-anchored). SKIPPED when no
        # F3 looking_at gave us a real depth anchor — without one
        # every patch projects to a fictitious "shell" of voxels at
        # the configured default depth, which the strict gate then
        # mis-commits as a hallucinated cluster around the player.
        if self.block_classifier is not None and not skip_patch_sweep:
            n_air_carved = 0
            n_committed  = 0
            n_curious    = 0
            _deadline = getattr(self, "_tick_deadline", None)
            for obs in self._sweep_patches(frame_rgb, sr, pose, anchor_depth):
                # Each yielded patch already paid a classify; stop pulling
                # more once the shared per-update budget is spent.
                if _deadline is not None and time.perf_counter() > _deadline:
                    break
                wf.visible_blocks.append(obs)
                # Strict commit gate: vision-patch guesses ONLY enter
                # the WorldMap when both (a) the sample-NN recogniser
                # gave a confident verdict, and (b) the voxel hasn't
                # been previously F3-confirmed as a different block.
                if self._should_commit(obs):
                    self._commit(obs)
                    n_committed += 1
                    if self.cfg.carve_air_along_sightlines:
                        cleared = self._air_voxels_along(
                            eye=eye, target_voxel=obs.pos,
                            sr=sr, pose=pose,
                        )
                        n_air_carved += self.world_map.mark_air_along(
                            cleared,
                            dimension=pose.dimension,
                            confidence=0.5,
                            tick=self._tick,
                        )
                elif obs.pos not in self._confirmed:
                    # Unconfirmed guess — feed the curiosity queue so the
                    # explorer agent can decide to aim at it later. Skip
                    # voxels that are out of MC's block-reach distance
                    # from the current eye; F3 "Looking at block" can
                    # only confirm reachable voxels, so investigating
                    # out-of-reach ones is wasted work.
                    tcx = obs.pos[0] + 0.5
                    tcy = obs.pos[1] + 0.5
                    tcz = obs.pos[2] + 0.5
                    dist = math.sqrt(
                        (tcx - eye[0]) ** 2
                        + (tcy - eye[1]) ** 2
                        + (tcz - eye[2]) ** 2
                    )
                    if dist > self.cfg.crosshair_reach_blocks:
                        continue
                    n_curious += 1
                    self._curiosity[obs.pos] = {
                        "block_id":   obs.block_id,
                        "confidence": float(obs.confidence),
                        "seen_tick":  self._tick,
                        "screen":     obs.meta.get("screen") if obs.meta else None,
                    }
            self._prune_curiosity()
            if n_air_carved:
                wf.diagnostics["air_voxels_carved"] = n_air_carved
            if n_committed:
                wf.diagnostics["patches_committed"] = n_committed
            if n_curious:
                wf.diagnostics["curiosity_added"]   = n_curious
            wf.diagnostics["curiosity_size"] = len(self._curiosity)
            wf.diagnostics["confirmed_count"] = len(self._confirmed)

        # Cross-validate curiosity-queue entries against the inverse
        # renderer. DISABLED while ``commit_only_from_looking_at``
        # is on — the cross-validator commits voxels the AI wasn't
        # directly looking at, which violates the strict guarantee.
        if (not self.cfg.commit_only_from_looking_at
                and self.cfg.use_inverse_renderer
                and self.cfg.inverse_validate_curiosity
                and self.inverse_renderer is not None):
            n_xv = self._cross_validate_curiosity(
                frame_rgb=frame_rgb, sr=sr, eye=eye, pose=pose,
            )
            if n_xv:
                wf.diagnostics["inverse_cross_validated"] = n_xv

        # 4. Entity detection (rate-limited).
        if self._tick % max(1, self.cfg.entity_detect_every_n_ticks) == 0:
            ents, drops = self.entity_classifier.detect(frame_rgb)
            for i, e in enumerate(ents):
                e.last_seen_tick = self._tick
                e.dimension = pose.dimension
                self.world_map.update_entity(f"e_{self._tick}_{i}", e)
            for i, d in enumerate(drops):
                d.last_seen_tick = self._tick
                d.dimension = pose.dimension
                self.world_map.update_drop(f"d_{self._tick}_{i}", d)
            wf.visible_entities = ents
            wf.visible_drops    = drops

        # 5. Periodic decay so the map doesn't accumulate stale mobs.
        # We do this lazily — once every 30 ticks is plenty.
        if self._tick % 30 == 0:
            self.world_map.decay_entities(self._tick, dimension=pose.dimension)
            self.world_map.decay_drops(self._tick, dimension=pose.dimension)

        return wf

    # ── Pose helpers ───────────────────────────────────────────────

    def _pose_from_f3(self, f3: Optional[F3Info]) -> Optional[PlayerPose]:
        """
        Build a :class:`PlayerPose` from the latest F3 parse.

        Position (x/y/z) is REQUIRED — without an eye position we can't
        evaluate reach distance or project anything to screen space.
        ``yaw`` and ``pitch``, however, are OPTIONAL: when MC's
        ``Facing: <card> (yaw / pitch)`` line gets too OCR-garbled to
        recover the angles (capital ``F`` lost, ``/`` separator eaten),
        we still want to COMMIT the targeted-block reads we DO have —
        those come from the ``Targeted Block:`` line + the
        ``minecraft:<id>`` line, neither of which depends on
        orientation. Missing yaw/pitch fall back to 0.0 so downstream
        code that needs an orientation (forward-vector, patch sweep)
        gets a defined-but-arbitrary value; those code paths are gated
        off in scan-only mode anyway.
        """
        if f3 is None or f3.x is None:
            return None
        return PlayerPose(
            x = float(f3.x),
            y = float(f3.y) if f3.y is not None else 0.0,
            z = float(f3.z) if f3.z is not None else 0.0,
            yaw = float(f3.yaw) if f3.yaw is not None else 0.0,
            pitch = float(f3.pitch) if f3.pitch is not None else 0.0,
            dimension = f3.dimension or "minecraft:overworld",
            block_x = f3.block_x,
            block_y = f3.block_y,
            block_z = f3.block_z,
            timestamp = f3.timestamp,
        )

    # ── Crosshair raycast ─────────────────────────────────────────

    def _crosshair_observation(self,
                               frame_rgb: np.ndarray,
                               sr: ScreenRay,
                               pose: PlayerPose,
                               anchor_depth: Optional[float],
                               ) -> Optional[BlockObservation]:
        intr = sr.intrinsics
        eye = (pose.x, pose.eye_y, pose.z)
        # Patch is sampled at screen centre.
        patch = self._crop_patch(frame_rgb,
                                 int(intr.cx), int(intr.cy),
                                 self.cfg.patch_size_px)
        if patch is None or self.block_classifier is None:
            return None
        block_id, conf = self.block_classifier.classify(patch)
        if block_id is None or conf < self.cfg.min_block_confidence:
            return None
        forward = sr.forward_vector(pose.yaw, pose.pitch)
        target_voxel = self._voxel_at_depth(
            eye=eye, direction=forward,
            target_depth=(anchor_depth if anchor_depth is not None
                          else self.cfg.default_anchor_depth),
            max_distance=self.cfg.crosshair_reach_blocks,
        )
        if target_voxel is None:
            return None
        return BlockObservation(
            pos = target_voxel,
            block_id = block_id,
            confidence = conf,
            source = "vision_patch",
            last_seen_tick = self._tick,
            dimension = pose.dimension,
            meta = {"origin": "crosshair"},
        )

    # ── Depth-aware voxel finder ──────────────────────────────────

    @staticmethod
    def _voxel_at_depth(*,
                        eye: Tuple[float, float, float],
                        direction: Tuple[float, float, float],
                        target_depth: float,
                        max_distance: float,
                        ) -> Optional[Tuple[int, int, int]]:
        """
        Walk along ``direction`` from ``eye`` and pick the voxel whose
        ENTRY-distance from the eye is closest to ``target_depth``
        (clamped to ``max_distance``). Skips the eye-voxel itself.

        Picking by entry-distance (rather than midpoint or exit)
        attributes the patch to the voxel whose visible face the ray
        first touches — i.e. the surface block, not the air just
        beyond it.
        """
        eye_vox = (int(math.floor(eye[0])),
                    int(math.floor(eye[1])),
                    int(math.floor(eye[2])))
        if target_depth <= 0.0:
            target_depth = 1.0
        target_depth = min(target_depth, max_distance)

        best_vox: Optional[Tuple[int, int, int]] = None
        best_err: float = float("inf")
        for travel, vox in WorldPerception._enumerate_voxels(
                eye, direction, max_distance):
            if vox == eye_vox:
                continue
            err = abs(travel - target_depth)
            if err < best_err:
                best_err = err
                best_vox = vox
            if travel > target_depth + 4.0:
                break
        return best_vox

    @staticmethod
    def _enumerate_voxels(eye, direction, max_distance):
        """Yield (entry_distance, voxel) along the ray."""
        # Re-implement voxel walk with explicit entry-distance tracking
        # so the depth picker can see HOW FAR each voxel is from eye.
        ox, oy, oz = eye
        dx, dy, dz = direction
        if max_distance <= 0.0 or (dx == 0.0 and dy == 0.0 and dz == 0.0):
            return
        ix, iy, iz = math.floor(ox), math.floor(oy), math.floor(oz)
        step_x = 1 if dx > 0 else -1 if dx < 0 else 0
        step_y = 1 if dy > 0 else -1 if dy < 0 else 0
        step_z = 1 if dz > 0 else -1 if dz < 0 else 0
        inf = float("inf")
        def _next_t(o, d, i, step):
            if step == 0:
                return inf
            boundary = (i + 1) if step > 0 else i
            return (boundary - o) / d
        t_max_x = _next_t(ox, dx, ix, step_x)
        t_max_y = _next_t(oy, dy, iy, step_y)
        t_max_z = _next_t(oz, dz, iz, step_z)
        t_delta_x = inf if step_x == 0 else 1.0 / abs(dx)
        t_delta_y = inf if step_y == 0 else 1.0 / abs(dy)
        t_delta_z = inf if step_z == 0 else 1.0 / abs(dz)
        yield 0.0, (ix, iy, iz)
        travelled = 0.0
        while travelled < max_distance:
            if t_max_x < t_max_y:
                if t_max_x < t_max_z:
                    ix += step_x;  travelled = t_max_x; t_max_x += t_delta_x
                else:
                    iz += step_z;  travelled = t_max_z; t_max_z += t_delta_z
            else:
                if t_max_y < t_max_z:
                    iy += step_y;  travelled = t_max_y; t_max_y += t_delta_y
                else:
                    iz += step_z;  travelled = t_max_z; t_max_z += t_delta_z
            if travelled > max_distance:
                break
            yield travelled, (ix, iy, iz)

    # ── Inverse-render cross-validation of curiosity entries ─────

    def _cross_validate_curiosity(self,
                                   *,
                                   frame_rgb: np.ndarray,
                                   sr: ScreenRay,
                                   eye: Tuple[float, float, float],
                                   pose: PlayerPose,
                                   ) -> int:
        """
        For each recent curiosity-queue entry, score the voxel
        against the patch sweep's guessed block id via the inverse
        renderer. If they agree, commit. Returns number committed.

        Useful when F3 looking_at isn't firing — gives the map a
        way to populate from purely visual cross-validation.
        """
        ir = self.inverse_renderer
        if ir is None or not self._curiosity:
            return 0
        cap = self.cfg.inverse_validate_max_per_tick
        # Take the freshest entries — those have the latest pose
        # backing them, so the geometric check is most accurate.
        ordered = sorted(self._curiosity.items(),
                         key=lambda kv: -kv[1]["seen_tick"])
        committed = 0
        deadline = getattr(self, "_tick_deadline", None)
        for pos, meta in ordered[:cap]:
            if deadline is not None and time.perf_counter() > deadline:
                break
            block_id = meta.get("block_id")
            if not block_id:
                continue
            # Already trusted by another path?
            if pos in self._confirmed:
                self._curiosity.pop(pos, None)
                continue
            try:
                score = ir.score_voxel(
                    frame_rgb, pos, block_id,
                    sr=sr, eye=eye, yaw=pose.yaw, pitch=pose.pitch,
                )
            except Exception as e:
                # Inverse renderer can raise on exotic block ids
                # whose face textures aren't in the asset cache.
                # Skip silently (single bad id shouldn't kill the
                # tick); surface first occurrence so we know.
                if not getattr(self, "_ir_score_warn_emitted", False):
                    self._ir_score_warn_emitted = True
                    print(f"[perception][WARN] inverse_renderer.score_voxel "
                          f"raised {e!r} for {block_id!r} — further "
                          f"errors silenced")
                continue
            if score < self.cfg.inverse_validate_score_min:
                continue
            self._commit(BlockObservation(
                pos=pos, block_id=block_id,
                confidence=min(0.95, score + meta["confidence"] * 0.3),
                source="extrapolation",
                last_seen_tick=self._tick,
                dimension=pose.dimension,
                meta={"cross_validated": True,
                      "inverse_score": round(score, 3),
                      "patch_conf":    round(float(meta["confidence"]), 3)},
            ))
            self._curiosity.pop(pos, None)
            committed += 1
        return committed

    # ── Inverse-render expansion ──────────────────────────────────

    def _expand_via_inverse_render(self,
                                    *,
                                    frame_rgb: np.ndarray,
                                    sr: ScreenRay,
                                    eye: Tuple[float, float, float],
                                    pose: PlayerPose,
                                    seed_voxel: Tuple[int, int, int],
                                    seed_block_id: str,
                                    ) -> int:
        """
        Test every voxel within a small neighbourhood of the seed
        for matching ``seed_block_id`` via inverse rendering. Commit
        confident matches to the WorldMap as ``extrapolation``.

        Returns the count of new voxels committed.
        """
        cfg = self.cfg
        ir  = self.inverse_renderer
        if ir is None:
            return 0
        r = max(1, cfg.expand_neighbour_radius)
        budget = max(1, cfg.expand_max_voxels_per_tick)
        sx, sy, sz = seed_voxel
        n_committed = 0
        # Iterate neighbours, closest-first, so the budget covers the
        # likeliest candidates if we run out of time.
        candidates: List[Tuple[int, Tuple[int, int, int]]] = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    pos = (sx + dx, sy + dy, sz + dz)
                    if pos in self._confirmed:
                        continue
                    # Existing more-authoritative observation? skip.
                    prev = self.world_map.get_block(pos,
                                                    dimension=pose.dimension)
                    if prev is not None and prev.source in (
                            "manual", "looking_at"):
                        continue
                    chebyshev = max(abs(dx), abs(dy), abs(dz))
                    candidates.append((chebyshev, pos))
        candidates.sort(key=lambda kv: kv[0])

        # Closest-first, bounded by both the voxel budget and the shared
        # per-update wall-clock deadline (see ``perception_time_budget_ms``)
        # so a block-dense scene can't blow the tick.
        deadline = getattr(self, "_tick_deadline", None)
        for _dist, pos in candidates[:budget]:
            if deadline is not None and time.perf_counter() > deadline:
                break
            try:
                score = ir.score_voxel(
                    frame_rgb, pos, seed_block_id,
                    sr=sr, eye=eye, yaw=pose.yaw, pitch=pose.pitch,
                )
            except Exception as e:
                # Exotic block id without an asset-cache face texture
                # — skip without crashing the tick.
                if not getattr(self, "_ir_expand_warn_emitted", False):
                    self._ir_expand_warn_emitted = True
                    print(f"[perception][WARN] inverse_renderer.score_voxel "
                          f"raised {e!r} during F3-seed expansion for "
                          f"{seed_block_id!r} — further errors silenced")
                continue
            if score < cfg.expand_score_min:
                continue
            self._commit(BlockObservation(
                pos=pos, block_id=seed_block_id,
                confidence=score, source="extrapolation",
                last_seen_tick=self._tick,
                dimension=pose.dimension,
                meta={"seed": list(seed_voxel),
                      "score": round(score, 3)},
            ))
            self._curiosity.pop(pos, None)
            n_committed += 1
        return n_committed

    # ── Free-space carving along an arbitrary forward ray ───────

    def _air_voxels_in_ray(self,
                           *,
                           eye: Tuple[float, float, float],
                           direction: Tuple[float, float, float],
                           max_distance: float,
                           ) -> List[Tuple[int, int, int]]:
        """
        Voxels intersected by a ray from ``eye`` in ``direction`` up
        to ``max_distance`` blocks. Used to mark air when MC's F3
        raytrace returned no target — every voxel within MC's
        reach along the player's forward ray must be transparent.
        """
        # We only need the voxel positions here; ``_enumerate_voxels``
        # also yields the cumulative travel distance which other
        # callers consume — discard it with ``_`` to keep this helper
        # cheap and obvious about its intent.
        return [vox for _, vox in self._enumerate_voxels(eye, direction,
                                                          max_distance)]

    # ── Free-space carving ────────────────────────────────────────

    def _air_voxels_along(self,
                          *,
                          eye: Tuple[float, float, float],
                          target_voxel: Tuple[int, int, int],
                          sr: ScreenRay,
                          pose: PlayerPose,
                          ) -> List[Tuple[int, int, int]]:
        """
        Enumerate the voxels between the eye and ``target_voxel`` —
        every one of these must be AIR, since the sightline passed
        through them to reach the target.

        We stop just before the target voxel itself; the caller marks
        the target with its actual block id separately.
        """
        # Aim a ray at the target voxel's centre.
        tx, ty, tz = target_voxel
        target_centre = (tx + 0.5, ty + 0.5, tz + 0.5)
        dx = target_centre[0] - eye[0]
        dy = target_centre[1] - eye[1]
        dz = target_centre[2] - eye[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 1e-3:
            return []
        direction = (dx / dist, dy / dist, dz / dist)
        # Every voxel between eye and target is carved as air, INCLUDING
        # the eye-voxel: it's usually inside the player's body / feet
        # and is always traversable space by definition.
        out: List[Tuple[int, int, int]] = []
        for _travel, vox in self._enumerate_voxels(eye, direction, dist + 0.01):
            if vox == target_voxel:
                break
            out.append(vox)
        return out

    # ── Sample collection (self-improvement) ───────────────────────

    def _maybe_save_crosshair_sample(self,
                                     *,
                                     frame_rgb: np.ndarray,
                                     sr: ScreenRay,
                                     block_id: str,
                                     voxel: Tuple[int, int, int],
                                     metadata: Optional[Dict[str, Any]] = None,
                                     ) -> None:
        """
        Crop a sample patch at the crosshair and hand it to the sample
        store labelled with ``block_id``.

        Throttled so the same (block_id, voxel) only contributes one
        sample per ``sample_cooldown_ticks`` ticks — staring at one
        block for 200 ticks shouldn't write 200 duplicate captures.
        """
        if self.sample_store is None:
            return
        # Sprite/cross blocks: the crosshair crop is dominated by the
        # backing block, not the sprite — an unreliable label. Skip the
        # training capture (the map commit already happened from F3).
        if self._sprite_block_predicate is not None:
            try:
                is_sprite = self._sprite_block_predicate(block_id)
            except Exception:
                is_sprite = False
            if is_sprite:
                self._sprite_skips += 1
                if self._sprite_skips <= 3:
                    print(f"[perception] skip sprite sample {block_id} "
                          f"(cross-model; crosshair crop is background-"
                          f"dominated, would mislabel training data)")
                return
        key = (block_id, voxel)
        last = self._last_sample_at.get(key, -10_000)
        if self._tick - last < self.cfg.sample_cooldown_ticks:
            return
        intr = sr.intrinsics
        # Size the crop to the block's apparent size at its distance so the
        # stored sample frames ~one block face — matching the sweep's
        # distance-normalised crops, so prototypes and queries share scale.
        dist = (metadata or {}).get("distance_blocks")
        cap_px = self._apparent_crop_px(intr, dist)
        patch = self._crop_patch(frame_rgb,
                                 int(intr.cx), int(intr.cy), cap_px)
        if patch is None:
            return
        # Remove the crosshair before persisting — see the long
        # rationale on ``mask_crosshair_in_samples`` in the config
        # dataclass. Failure here MUST NOT abort sample collection,
        # so we fall back to the raw patch on any inpaint error.
        if self.cfg.mask_crosshair_in_samples:
            try:
                patch = self._mask_crosshair(patch)
            except Exception as e:
                if not getattr(self, "_warned_crosshair_mask", False):
                    print(f"[perception][WARN] crosshair masking failed "
                          f"({e!r}); saving raw patch")
                    self._warned_crosshair_mask = True
        # Per-block relative-darkness gate. A block lit by deep night/cave
        # darkness is near-indistinguishable from any other dark block, so
        # such a sample smears the prototype and wrecks daytime accuracy
        # (verified: training on night-darkened grass dropped its bright
        # accuracy from 0.94 to 0.42). We reject a sample only if it's an
        # OUTLIER-dark version of a block we normally see brighter — so
        # genuinely-dark blocks (obsidian, deepslate, coal) keep their
        # samples (their running mean is dark, nothing is an outlier),
        # which a blunt global brightness floor would have wrongly culled.
        if self.cfg.dark_sample_factor > 0.0:
            b = float(patch.mean())
            ema = self._block_brightness.get(block_id)
            n = self._block_brightness_n.get(block_id, 0)
            if (ema is not None and n >= self.cfg.dark_sample_min_history
                    and b < self.cfg.dark_sample_factor * ema):
                self._dark_skips += 1
                if self._dark_skips <= 3:
                    print(f"[perception] skip dark sample {block_id} "
                          f"(brightness {b:.0f} < {self.cfg.dark_sample_factor:.2f}"
                          f"x block mean {ema:.0f}; low-light, would smear "
                          f"the prototype)")
                self._last_sample_at[key] = self._tick
                return
            # Update the running brightness mean for this block.
            self._block_brightness[block_id] = (
                b if ema is None else 0.9 * ema + 0.1 * b)
            self._block_brightness_n[block_id] = n + 1
        path = self.sample_store.save(block_id, patch, metadata=metadata)
        if path is None:
            # Duplicate or cap reached — still update the throttle so we
            # don't try every tick.
            self._last_sample_at[key] = self._tick
            return
        self._last_sample_at[key] = self._tick
        self._samples_saved += 1
        self._samples_saved_since_reload += 1

        # Hot-reload the sample recogniser so the newly-saved patch is
        # available immediately. Reload is cheap (small in-memory
        # tensor); we still throttle to every N saves.
        if (self._samples_saved_since_reload
                >= self.cfg.sample_reload_every_n_saves):
            self._samples_saved_since_reload = 0
            self._reload_sample_recognizer()

    def _reload_sample_recognizer(self) -> None:
        """If the active classifier is a hybrid wrapping a sample
        recogniser, refresh its in-memory tensor.

        First-failure log: a silently-failing reload makes the
        recogniser stale forever. We auto-collect samples every F3
        confirm and rely on the reload to surface new block ids, so a
        silent failure here breaks the entire self-improvement loop —
        the agent thinks it's learning but never recognises the new
        block. Surface the first occurrence.
        """
        cls = self.block_classifier
        if cls is None:
            return
        reload_fn = getattr(cls, "reload_samples", None)
        if callable(reload_fn):
            try:
                reload_fn()
            except Exception as e:
                if not getattr(self, "_reload_warn_emitted", False):
                    self._reload_warn_emitted = True
                    print(f"[perception][WARN] sample recogniser reload "
                          f"failed: {e!r}. The self-improvement loop is "
                          f"now stale until restart.")

    # ── Patch sweep ────────────────────────────────────────────────

    def _sweep_patches(self,
                       frame_rgb: np.ndarray,
                       sr: ScreenRay,
                       pose: PlayerPose,
                       anchor_depth: Optional[float],
                       ) -> List[BlockObservation]:
        """
        Sample a dense grid of patches across the gameplay region,
        classify each, and project them into the world at the
        F3-anchored depth (or the configured fallback). Each surviving
        patch contributes one BlockObservation at the voxel its ray
        crosses at roughly that depth.

        Sky / out-of-bounds patches are filtered by a cheap colour
        test BEFORE classification so the per-patch cost stays bounded.
        """
        if self.block_classifier is None:
            return []
        eye = (pose.x, pose.eye_y, pose.z)
        depth = (anchor_depth if anchor_depth is not None
                 else self.cfg.default_anchor_depth)
        depth = max(1.0, min(depth, self.cfg.max_walk_distance))
        # Distance-normalised crop size for this sweep (all patches share
        # the anchor depth) — frames ~one block face, matching the scale
        # the recogniser's samples were captured at.
        sweep_cap_px = self._apparent_crop_px(sr.intrinsics, depth)

        n_cols, n_rows = self.cfg.patch_grid_size
        if n_cols < 1 or n_rows < 1:
            return []

        H, W = frame_rgb.shape[:2]
        # Resolution-aware scaling: the user calibrates margins and
        # exclude_rects at ``frame_ref_resolution``; if the captured
        # frame is a different size (windowed MC, different monitor,
        # the player resizing the window), scale every pixel value
        # by the actual / reference ratio so the UI exclusion zones
        # still cover the hotbar / hand / F3 panel correctly.
        ref_w, ref_h = self.cfg.frame_ref_resolution
        sx = W / float(ref_w) if ref_w > 0 else 1.0
        sy = H / float(ref_h) if ref_h > 0 else 1.0
        x0 = int(round(self.cfg.margin_left_px   * sx))
        x1 = W - int(round(self.cfg.margin_right_px  * sx))
        y0 = int(round(self.cfg.margin_top_px    * sy))
        y1 = H - int(round(self.cfg.margin_bottom_px * sy))
        if x1 - x0 < self.cfg.patch_size_px or y1 - y0 < self.cfg.patch_size_px:
            return []

        observations: List[BlockObservation] = []
        # De-dupe: many patches may project to the same voxel; keep
        # the one with highest confidence per voxel and drop the rest.
        best_per_voxel: Dict[Tuple[int, int, int],
                              Tuple[float, BlockObservation]] = {}

        excludes = tuple(
            (int(round(rx * sx)), int(round(ry * sy)),
             int(round(rw * sx)), int(round(rh * sy)))
            for (rx, ry, rw, rh) in (self.cfg.exclude_rects or ())
        )
        # Shared per-update wall-clock deadline. The per-patch
        # ``block_classifier.classify`` (sample-NN) is the single most
        # expensive perception op and scales with the sample dataset, so
        # we MUST bound how many patches we classify per tick here —
        # checking the deadline in the CALLER doesn't help because this
        # method builds the whole list eagerly before returning.
        deadline = getattr(self, "_tick_deadline", None)
        # ── Pass 1: collect candidate patches + their screen positions ──
        # We gather first, then classify in ONE batch. With the learned
        # CNN recogniser a single batched forward over the whole grid is
        # ~10-50 ms, vs hundreds of ms for per-patch calls — which is what
        # makes whole-vision recognition real-time. (The colour baseline
        # has no batch path; it's classified per-patch in pass 2.)
        cand_px: List[int] = []
        cand_py: List[int] = []
        cand_patch: List[np.ndarray] = []
        for j in range(n_rows):
            if deadline is not None and time.perf_counter() > deadline:
                break
            for i in range(n_cols):
                if deadline is not None and time.perf_counter() > deadline:
                    break
                px = int(x0 + (i + 0.5) * (x1 - x0) / n_cols)
                py = int(y0 + (j + 0.5) * (y1 - y0) / n_rows)
                # Skip patches landing inside any excluded rectangle —
                # hotbar, hand, future inventory overlays.
                if _point_in_any_rect(px, py, excludes):
                    continue
                # Crop at the distance-normalised size (frames ~one block
                # face at the anchor depth), matching the scale the
                # recogniser's samples were captured at. (A fixed size here
                # mismatched the model's training scale and wrecked accuracy.)
                patch = self._crop_patch(frame_rgb, px, py, sweep_cap_px)
                if patch is None:
                    continue
                # Skip obvious sky / void patches without paying to classify.
                if self._looks_like_sky(patch):
                    continue
                cand_px.append(px)
                cand_py.append(py)
                cand_patch.append(patch)

        # ── Classify (batched if the recogniser supports it) ──
        batch_fn = getattr(self.block_classifier, "classify_batch", None)
        if callable(batch_fn):
            results = batch_fn(cand_patch)
        else:
            results = [self.block_classifier.classify(p) for p in cand_patch]

        # ── Pass 2: project the confident hits into the world ──
        for px, py, (block_id, conf) in zip(cand_px, cand_py, results):
            if block_id is None or conf < self.cfg.min_block_confidence:
                continue
            _, direction = sr.unproject(px, py,
                                        yaw_deg=pose.yaw,
                                        pitch_deg=pose.pitch,
                                        eye_xyz=eye)
            target = self._voxel_at_depth(
                eye=eye, direction=direction,
                target_depth=depth,
                max_distance=self.cfg.max_walk_distance,
            )
            if target is None:
                continue
            obs = BlockObservation(
                pos = target,
                block_id = block_id,
                confidence = conf,
                source = "vision_patch",
                last_seen_tick = self._tick,
                dimension = pose.dimension,
                meta = {"screen": [int(px), int(py)],
                        "depth": round(depth, 2)},
            )
            prev = best_per_voxel.get(target)
            if prev is None or conf > prev[0]:
                best_per_voxel[target] = (conf, obs)

        for _, obs in best_per_voxel.values():
            # Temporal smoothing: fold this voxel's guess into its recent
            # vote history. First sight passes through unchanged; repeated
            # views correct transient flips and scale confidence by
            # agreement. (Map commit still applies its own gates after.)
            if self._temporal_voter is not None:
                vid, vconf, _stable = self._temporal_voter.vote(
                    obs.pos, obs.block_id, obs.confidence, self._tick)
                if vid != obs.block_id or vconf != obs.confidence:
                    obs = replace(obs, block_id=vid, confidence=vconf)
            observations.append(obs)
        return observations

    @staticmethod
    def _looks_like_sky(patch_rgb: np.ndarray) -> bool:
        """Cheap-and-cheerful sky / overworld-void filter.

        Sky in vanilla MC during daytime sits at roughly (120, 160,
        230) — mean B significantly > mean R. Sunset / dawn shifts
        this further toward orange, which is handled separately by
        the broader colour-signature classifier downstream. Night sky
        is uniformly very dark and gets filtered by the existing
        min_block_confidence gate.
        """
        if patch_rgb is None or patch_rgb.size == 0:
            return False
        if patch_rgb.ndim != 3 or patch_rgb.shape[2] < 3:
            return False
        r = float(patch_rgb[..., 0].mean())
        g = float(patch_rgb[..., 1].mean())
        b = float(patch_rgb[..., 2].mean())
        # Bright blue / cyan dominant sky.
        if b > 160 and b > r + 30 and b > g + 10:
            return True
        return False

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _crop_patch(frame: np.ndarray, cx: int, cy: int, size: int
                   ) -> Optional[np.ndarray]:
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        half = size // 2
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(w, x0 + size)
        y1 = min(h, y0 + size)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        return frame[y0:y1, x0:x1]

    def _apparent_crop_px(self, intr, distance: Optional[float]) -> int:
        """Pixel size of a crop framing ~``crop_block_span`` block-widths
        at ``distance`` blocks, given the camera focal length. A 1-block
        face at distance D spans ~fx/D px (pinhole). Clamped to
        [crop_px_min, crop_px_max]. Falls back to ``sample_capture_px``
        when distance-normalisation is off or distance is unknown."""
        if (not self.cfg.distance_normalized_crops or distance is None
                or not math.isfinite(distance) or distance <= 0.0
                or intr is None):
            return self.cfg.sample_capture_px
        fx = float(getattr(intr, "fx", 0.0) or 0.0)
        if fx <= 0.0:
            return self.cfg.sample_capture_px
        px = fx * self.cfg.crop_block_span / max(1.0, float(distance))
        return int(max(self.cfg.crop_px_min, min(self.cfg.crop_px_max, round(px))))

    def _mask_crosshair(self, patch_rgb: np.ndarray) -> np.ndarray:
        """
        Inpaint the crosshair arms out of a centre-cropped patch.

        MC's crosshair (gui/sprites/hud/crosshair.png, 15×15 GUI px)
        sits dead-centre on every sample we collect. It inverts the
        colours of pixels underneath, so neither bright-pixel nor
        edge-based detection works reliably across light/dark biomes.
        A fixed geometric mask of the cross arms is robust regardless
        of background, and ``cv2.INPAINT_TELEA`` propagates the
        surrounding block texture into the masked region — the saved
        patch ends up with continuous block pixels in the centre
        instead of a synthetic + that future recognisers would learn.

        Returns a fresh array; the input is not modified in place.
        """
        h, w = patch_rgb.shape[:2]
        cx, cy = w // 2, h // 2
        scale = max(1, int(self.cfg.ui_scale))
        arm_half_len   = self.cfg.crosshair_arm_half_len_gui   * scale
        arm_half_thick = self.cfg.crosshair_arm_half_thick_gui * scale
        mask = np.zeros((h, w), dtype=np.uint8)
        # Horizontal arm.
        y0 = max(0, cy - arm_half_thick)
        y1 = min(h, cy + arm_half_thick + 1)
        x0 = max(0, cx - arm_half_len)
        x1 = min(w, cx + arm_half_len + 1)
        mask[y0:y1, x0:x1] = 255
        # Vertical arm.
        y0 = max(0, cy - arm_half_len)
        y1 = min(h, cy + arm_half_len + 1)
        x0 = max(0, cx - arm_half_thick)
        x1 = min(w, cx + arm_half_thick + 1)
        mask[y0:y1, x0:x1] = 255
        # cv2.inpaint requires a contiguous uint8 BGR/RGB image; if the
        # caller handed us a non-contiguous slice (view of a larger
        # frame) we must copy it first.
        if not patch_rgb.flags["C_CONTIGUOUS"]:
            patch_rgb = np.ascontiguousarray(patch_rgb)
        return cv2.inpaint(patch_rgb, mask, 3, cv2.INPAINT_TELEA)

    # ── Strict-commit + curiosity queue ────────────────────────────

    def _commit(self, obs: BlockObservation) -> None:
        """
        Persist a BlockObservation to the WorldMap AND tally it in the
        per-source / per-block counter. Every commit path in this module
        funnels through here so the stats are authoritative — every
        voxel in the WorldMap maps 1:1 to an increment in
        ``_commits_by_source`` keyed by ``(source, block_id)``.
        """
        self.world_map.update_block(obs)
        key = (obs.source or "?", obs.block_id or "?")
        self._commits_by_source[key] = self._commits_by_source.get(key, 0) + 1

    def _should_commit(self, obs: BlockObservation) -> bool:
        """
        Decide whether an observation is confident enough to commit
        to the WorldMap.

        With ``commit_only_from_looking_at`` set, the ONLY source
        that commits is F3 ``looking_at`` — everything else gets
        rejected and stays in the curiosity queue until / unless
        the player aims at it directly. This is the conservative
        "the map contains only what the AI was directly looking at"
        guarantee the user asked for.

        When the strict-only flag is off, the legacy gate runs:
        vision-patches commit only with high confidence AND enough
        backing samples; F3 / manual sources always commit.
        """
        if self.cfg.commit_only_from_looking_at:
            return obs.source in ("looking_at", "manual")
        if not self.cfg.strict_commit_gate:
            return True
        if obs.source != "vision_patch":
            return True
        if obs.block_id is None:
            return False
        if obs.confidence < self.cfg.sample_commit_confidence:
            return False
        cls = self.block_classifier
        sample_rec = getattr(cls, "sample", None)
        if sample_rec is None:
            return False
        # Per-block sample-backing check.
        #
        # The previous version checked TOTAL ``sample_count()`` of the
        # store — but the HybridBlockClassifier falls back to the
        # colour-signature baseline (1157 block templates from the MC
        # asset jar) whenever the sample-NN can't confidently name a
        # patch. The baseline ROUTINELY returns block ids the sample
        # store has never seen (`acacia_leaves`, `mangrove_leaves`,
        # `sugar_cane`, etc.). With the old gate those leaked into
        # the WorldMap because the gate only verified the store had
        # ≥30 samples in TOTAL — which it does, just not of THIS
        # block.
        #
        # The fix: require the SPECIFIC block id to have at least
        # ``min_samples_per_block_for_commit`` samples backing it.
        # Predictions for blocks we've never trained on are rejected
        # (they go to the curiosity queue and the agent can aim its
        # crosshair to learn that block via F3).
        # Total-store floor (defence in depth): even if a single
        # block has 20 captures, the NN needs at least 2 distinct
        # classes to discriminate. ``min_samples_for_commit`` keeps
        # the historic "enough data on disk" gate alongside the
        # tighter per-block check below. Use public accessors so
        # a future thread-safe ``reload()`` doesn't expose us to a
        # half-rebuilt internal list.
        try:
            total_samples = sample_rec.sample_count()
            per_block_count = sample_rec.count_for(obs.block_id)
        except AttributeError:
            return False
        if total_samples < self.cfg.min_samples_for_commit:
            return False
        if per_block_count < self.cfg.min_samples_per_block_for_commit:
            return False
        return True

    def _prune_curiosity(self) -> None:
        """Cap the curiosity queue. Evicts oldest seen entries first;
        tiebreak by lowest confidence (those are the least informative)."""
        if len(self._curiosity) <= self.cfg.curiosity_queue_max:
            return
        items = sorted(
            self._curiosity.items(),
            key=lambda kv: (kv[1]["seen_tick"], -kv[1]["confidence"]),
        )
        n_drop = len(self._curiosity) - self.cfg.curiosity_queue_max
        for k, _ in items[:n_drop]:
            self._curiosity.pop(k, None)

    def curiosity_queue(self) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
        """Return a snapshot of the curiosity queue for downstream
        agents. The returned dict is a shallow copy — modifying it
        won't affect the perception layer's state."""
        return dict(self._curiosity)

    def take_curiosity_target(self,
                              *,
                              eye: Tuple[float, float, float],
                              prefer: str = "closest",
                              yaw_deg: Optional[float] = None,
                              ) -> Optional[Tuple[int, int, int]]:
        """
        Pop and return one voxel for the agent to investigate next.

        ``prefer``:
          * ``"closest"`` — nearest voxel to the eye (default).
          * ``"oldest"``  — entry with the smallest ``seen_tick``.
          * ``"lowest_conf"`` — voxel where the classifier was least sure.
          * ``"in_view"``  — closest voxel AHEAD of the current view
            direction (requires ``yaw_deg``). Strongly preferred over
            ``closest`` for the live agent loop: rotating 140° to
            investigate a voxel behind you wastes 5+ seconds of camera
            motion and routinely fails to converge before the
            investigate timeout. A voxel 3 blocks ahead at yaw_err=5°
            is dramatically faster to confirm than one 1 block behind
            at yaw_err=170°, even if the geometric distance is similar.

        Returns ``None`` if the queue is empty.
        """
        if not self._curiosity:
            return None
        if prefer == "oldest":
            pos = min(self._curiosity,
                      key=lambda p: self._curiosity[p]["seen_tick"])
        elif prefer == "lowest_conf":
            pos = min(self._curiosity,
                      key=lambda p: self._curiosity[p]["confidence"])
        elif prefer == "in_view" and yaw_deg is not None:
            # Score = distance + heavy yaw-error penalty. Voxels behind
            # the player score worst even at close range; voxels
            # straight ahead at moderate range score best.
            import math as _math
            yaw_rad = _math.radians(yaw_deg)
            forward_x = -_math.sin(yaw_rad)
            forward_z =  _math.cos(yaw_rad)
            def _s(p):
                dx = p[0] + 0.5 - eye[0]
                dz = p[2] + 0.5 - eye[2]
                horiz = _math.hypot(dx, dz)
                # Cosine of angle between forward and direction-to-voxel.
                # 1.0 = straight ahead, -1.0 = directly behind.
                if horiz < 0.01:
                    cos_ang = 1.0
                else:
                    cos_ang = (forward_x * dx + forward_z * dz) / horiz
                # 0 when straight ahead, 4 when directly behind.
                ahead_penalty = 2.0 * (1.0 - cos_ang)
                dy = p[1] + 0.5 - eye[1]
                dist_sq = dx * dx + dy * dy + dz * dz
                return dist_sq + ahead_penalty * ahead_penalty * 6.0
            pos = min(self._curiosity, key=_s)
        else:  # closest
            def _d(p):
                return ((p[0] + 0.5 - eye[0]) ** 2
                        + (p[1] + 0.5 - eye[1]) ** 2
                        + (p[2] + 0.5 - eye[2]) ** 2)
            pos = min(self._curiosity, key=_d)
        self._curiosity.pop(pos, None)
        return pos

    def confirmed_voxels(self) -> set:
        """Set of voxel positions that F3 has confirmed at least once."""
        return set(self._confirmed)

    def is_confirmed(self, pos: Tuple[int, int, int]) -> bool:
        """O(1) membership check — cheaper than copying ``confirmed_voxels()``
        on the per-tick hot path of an investigating agent."""
        return pos in self._confirmed

    def purge_curiosity_queue(self) -> int:
        """Drop every entry from the curiosity queue and return the
        count purged. Agents call this when the queue is dominated by
        voxels from misread poses — the perception layer will repopulate
        it from current frames within a few ticks."""
        n = len(self._curiosity)
        self._curiosity.clear()
        return n

    def last_pose(self) -> Optional[PlayerPose]:
        """Most recent valid player pose, or None if we've never had one."""
        return self._last_pose

    # ── Diagnostics ────────────────────────────────────────────────

    @property
    def tick(self) -> int:
        return self._tick

    def stats(self) -> Dict[str, Any]:
        # Sort per-source commits + corrections so the shutdown dump
        # reads top-down by volume — easier to spot the troublemakers.
        commits_sorted = sorted(self._commits_by_source.items(),
                                  key=lambda kv: -kv[1])
        corrections_sorted = sorted(self._corrections_by_source.items(),
                                      key=lambda kv: -kv[1])
        curio_corrections_sorted = sorted(
            self._curiosity_corrections.items(), key=lambda kv: -kv[1]
        )
        out: Dict[str, Any] = {
            "tick": self._tick,
            "templates_loaded": (self.block_classifier.template_count()
                                  if self.block_classifier is not None else 0),
            "map": self.world_map.stats(),
            "samples_saved_this_session": self._samples_saved,
            "corrections_seen": self._n_corrections,
            "curiosity_size":  len(self._curiosity),
            "confirmed_count": len(self._confirmed),
            # Per-source / per-block breakdown for the FN/FP audit. A
            # healthy run has many ``looking_at`` commits, a smaller
            # number of ``vision_patch`` commits, and a SMALL count
            # of corrections relative to vision_patch — say <10%.
            "commits_by_source": {
                f"{src}:{bid}": n for (src, bid), n in commits_sorted
            },
            "corrections_by_source": {
                f"{src}:{bid}": n for (src, bid), n in corrections_sorted
            },
            # Curiosity-queue corrections: (guessed_block) -> (actual_block)
            # counts. Diagnostic-only; never committed anywhere. Heavy
            # entries point at which block id the classifier hallucinates
            # most, and which real block it actually was — both useful
            # for dataset prioritisation.
            "curiosity_corrections": {
                f"{guess}->{actual}": n
                for (guess, actual), n in curio_corrections_sorted
            },
        }
        if self.sample_store is not None:
            out["sample_store"] = {
                "blocks_known": self.sample_store.block_count(),
                "total_samples": self.sample_store.total_samples(),
            }
        return out


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_world_perception(settings: Optional[Dict[str, Any]] = None,
                           *,
                           assets=None,
                           ) -> WorldPerception:
    """
    Build a fully wired :class:`WorldPerception` from project settings.

    Honoured keys (all optional):

      ``vision.world.enabled``                : bool
      ``vision.world.h_fov_deg``              : float (default 90)
      ``vision.world.patch_grid_size``        : [cols, rows]
      ``vision.world.patch_size_px``          : int
      ``vision.world.max_walk_distance``      : float
      ``vision.world.crosshair_reach_blocks`` : float
      ``vision.world.min_block_confidence``   : float
      ``vision.world.entity_detect_every_n_ticks`` : int
      ``vision.world.auto_sample_from_looking_at``  : bool  (default True)
      ``vision.world.sample_cooldown_ticks``        : int
      ``vision.world.sample_capture_px``            : int
      ``vision.world.use_sample_recognizer``        : bool  (default True)

    The caller is expected to check ``vision.world.enabled`` *before*
    constructing the perception layer — this function will build it
    regardless so unit tests can use it without faking settings.
    """
    cfg = WorldPerceptionConfig()
    world_cfg = ((settings or {}).get("vision", {})
                                  .get("world", {})) or {}
    if "h_fov_deg" in world_cfg:
        cfg.h_fov_deg = float(world_cfg["h_fov_deg"])
    if "patch_grid_size" in world_cfg:
        v = world_cfg["patch_grid_size"]
        if isinstance(v, (list, tuple)) and len(v) == 2:
            cfg.patch_grid_size = (int(v[0]), int(v[1]))
    for key in ("patch_size_px", "margin_left_px", "margin_right_px",
                "margin_top_px", "margin_bottom_px",
                "entity_detect_every_n_ticks",
                "sample_cooldown_ticks", "sample_capture_px",
                "sample_reload_every_n_saves",
                "curiosity_queue_max",
                "min_samples_for_commit",
                "min_samples_per_block_for_commit",
                "expand_neighbour_radius",
                "expand_max_voxels_per_tick",
                "inverse_validate_max_per_tick",
                "crosshair_arm_half_len_gui",
                "crosshair_arm_half_thick_gui"):
        if key in world_cfg:
            setattr(cfg, key, int(world_cfg[key]))
    # ui_scale lives at top-level capture.* in settings.yaml — pull
    # it from there if the user hasn't specified an override on
    # vision.world directly.
    capture_ui_scale = (settings or {}).get("capture", {}).get("ui_scale")
    if "ui_scale" in world_cfg:
        cfg.ui_scale = int(world_cfg["ui_scale"])
    elif capture_ui_scale is not None:
        cfg.ui_scale = int(capture_ui_scale)
    for key in ("max_walk_distance", "crosshair_reach_blocks",
                "min_block_confidence",
                "default_anchor_depth", "depth_search_radius",
                "sample_commit_confidence",
                "expand_score_min",
                "perception_time_budget_ms",
                "weather_check_interval_sec",
                "inverse_validate_score_min"):
        if key in world_cfg:
            setattr(cfg, key, float(world_cfg[key]))
    if "skip_stale_pose_updates" in world_cfg:
        cfg.skip_stale_pose_updates = bool(world_cfg["skip_stale_pose_updates"])
    if "auto_sample_from_looking_at" in world_cfg:
        cfg.auto_sample_from_looking_at = bool(
            world_cfg["auto_sample_from_looking_at"])
    for bkey in ("strict_commit_gate", "carve_air_along_sightlines",
                 "use_f3_depth_anchor",
                 "use_inverse_renderer",
                 "inverse_validate_curiosity",
                 "commit_only_from_looking_at",
                 "mask_crosshair_in_samples",
                 "detect_weather",
                 "debug_f3_dump"):
        if bkey in world_cfg:
            setattr(cfg, bkey, bool(world_cfg[bkey]))
    # ``exclude_rects`` from YAML: list of [x, y, w, h].
    raw_excl = world_cfg.get("exclude_rects")
    if isinstance(raw_excl, (list, tuple)):
        parsed = []
        for r in raw_excl:
            if isinstance(r, (list, tuple)) and len(r) == 4:
                parsed.append((int(r[0]), int(r[1]),
                                int(r[2]), int(r[3])))
        if parsed:
            cfg.exclude_rects = tuple(parsed)

    # ``frame_ref_resolution`` from YAML: [W, H] pair the rects /
    # margins are calibrated against. Lets the user re-tune at one
    # resolution and trust auto-scaling at the actual capture size.
    raw_ref = world_cfg.get("frame_ref_resolution")
    if isinstance(raw_ref, (list, tuple)) and len(raw_ref) == 2:
        cfg.frame_ref_resolution = (int(raw_ref[0]), int(raw_ref[1]))

    if assets is None:
        from vision.mc_assets import MCAssets
        assets = MCAssets.load()
    baseline_cls = build_block_classifier(settings, assets=assets)

    # Known-block gate for F3 reads. Built from the same asset jar the
    # classifier uses, so it's authoritative for this MC version. Wrapped
    # defensively: if the catalog can't load, perception still runs (the
    # validator is simply absent and no plausibility check is applied).
    block_id_validator: Optional[Callable[[str], bool]] = None
    try:
        from knowledge.catalog import Catalog
        _catalog = Catalog.load(assets)
        block_id_validator = lambda bid: _catalog.block(bid) is not None
    except Exception as e:
        print(f"[world] block-id catalog gate disabled: {e}")

    # Sprite-block predicate: a block whose model is a thin billboard
    # (cross / tinted_cross / crop) rather than a full cube. The crosshair
    # crop of such a block is dominated by the backing block, so it's an
    # unreliable training label — the auto-sampler skips it. Detected from
    # the asset model parent; results cached (block_model parse isn't free).
    # Honour vision.world.skip_sprite_samples (default on).
    sprite_block_predicate: Optional[Callable[[str], bool]] = None
    if bool(world_cfg.get("skip_sprite_samples", True)):
        _sprite_cache: Dict[str, bool] = {}
        # A block has an UNRELIABLE crosshair crop (dominated by the
        # block behind it, so it mislabels training data) when its model
        # is a thin sprite or a thin/flat attachment rather than a solid
        # cube. Matched by substring on the model parent so new blocks
        # that reuse these vanilla templates are caught automatically:
        #   cross/tinted_cross/crop  - grass, ferns, flowers, saplings, wheat…
        #   carpet                   - all carpets / moss carpet
        #   pressure_plate / button  - thin floor/wall plates
        #   rail                     - all rail variants (flat on the floor)
        #   torch / lantern          - thin emissive attachments
        # Substantial PARTIAL cubes (slab, stairs, fence, wall) are NOT
        # skipped — their crop still contains a large representative chunk
        # of the block, so they remain learnable.
        _UNRELIABLE_CROP_HINTS = (
            "cross", "crop", "carpet", "pressure_plate",
            "button", "rail", "torch", "lantern",
        )

        def _is_sprite_block(bid: str) -> bool:
            hit = _sprite_cache.get(bid)
            if hit is not None:
                return hit
            res = False
            try:
                model = assets.block_model(bid.split(":", 1)[-1])
                parent = (model or {}).get("parent") or ""
                short = (parent.split("/")[-1] if parent else "").lower()
                if not short:
                    # No resolvable simple parent model => a multipart /
                    # blockstate-driven THIN block (vine, glass_pane, iron_bars,
                    # ladder, fences, walls, …). Its crosshair crop is a thin
                    # overlay dominated by the backing block and is visually
                    # confusable with leaves/grass — proven net-harmful (vine
                    # collection dragged grass 0.85->0.79 and oak_leaves
                    # 1.00->0.80). Skip it like a sprite; solid cubes always
                    # resolve to a cube/leaves/column parent and are kept.
                    res = True
                else:
                    res = any(h in short for h in _UNRELIABLE_CROP_HINTS)
            except Exception:
                res = False
            _sprite_cache[bid] = res
            return res

        sprite_block_predicate = _is_sprite_block

    # Wrap the baseline with a sample-NN recogniser unless explicitly
    # disabled. The hybrid is harmless when the sample store is empty
    # (it just falls through to the baseline every call).
    use_samples = bool(world_cfg.get("use_sample_recognizer", True))
    use_cnn = bool(world_cfg.get("use_cnn_recognizer", True))
    sample_store: Optional[WorldSampleStore] = None
    block_cls: BlockClassifierProtocol = baseline_cls
    if use_samples:
        sample_store = build_world_sample_store()
        sample_recognizer = SampleBlockRecognizer(sample_store)
        # Prefer the learned CNN (robust to lighting/biome/angle) when
        # torch is present; it self-trains in the background from the same
        # F3-labelled store and falls back to the sample-NN / colour
        # baseline until it has learned enough. Tiered: CNN -> NN -> base.
        cnn_recognizer = None
        if use_cnn:
            try:
                from vision.world.cnn_recognizer import (
                    CNNBlockRecognizer, _TORCH_OK,
                )
                if _TORCH_OK:
                    cnn_recognizer = CNNBlockRecognizer(
                        sample_store, auto_train=True)
            except Exception as e:
                # LOUD: without the CNN the world map can only label the block
                # under the crosshair (F3), not SURVEY the scene — the find
                # step (mapping logs to walk to) is badly degraded. If this
                # ever fires in a real run it must be obvious in the log, not a
                # one-line whisper. (Seen only under a pathological import
                # order; the normal make.py/main.py order loads it fine.)
                print("[world] *** WARNING: CNN block recogniser DISABLED "
                      f"({e}) — falling back to sample-NN; scene-wide block "
                      "recognition (the find step) will be degraded. ***")
                cnn_recognizer = None
        if cnn_recognizer is not None:
            from vision.world.cnn_recognizer import TieredBlockClassifier
            block_cls = TieredBlockClassifier(
                cnn_recognizer, sample_recognizer, baseline_cls)
        else:
            block_cls = HybridBlockClassifier(sample_recognizer, baseline_cls)

    entity_cls = build_entity_classifier(settings)

    # Inverse renderer. Cheap to construct; lookups cache lazily.
    inv_ren: Optional[InverseRenderer] = None
    if cfg.use_inverse_renderer:
        try:
            inv_ren = InverseRenderer(assets, InverseRendererConfig())
        except Exception as e:
            print(f"[world] inverse renderer disabled: {e}")
            inv_ren = None

    # Weather detector honouring vision.weather.* settings (thresholds,
    # smoothing, trained-sample floor). Disabled cleanly if construction
    # fails so perception still runs.
    weather_det = None
    if cfg.detect_weather:
        try:
            from vision.weather import build_weather_detector
            weather_det = build_weather_detector(settings)
        except Exception as e:
            print(f"[world] weather detector disabled: {e}")

    return WorldPerception(
        config=cfg,
        block_classifier=block_cls,
        entity_classifier=entity_cls,
        world_map=WorldMap(),
        sample_store=sample_store,
        inverse_renderer=inv_ren,
        weather_detector=weather_det,
        block_id_validator=block_id_validator,
        sprite_block_predicate=sprite_block_predicate,
    )


__all__ = [
    "WorldPerception",
    "WorldPerceptionConfig",
    "build_world_perception",
]
