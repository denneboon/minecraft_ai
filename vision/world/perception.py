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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
from vision.world.map import WorldMap
from vision.world.sample_store import (
    WorldSampleStore,
    build_world_sample_store,
)
from vision.world.sample_recognizer import (
    HybridBlockClassifier,
    SampleBlockRecognizer,
    SampleBlockRecognizerConfig,
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
    for (rx, ry, rw, rh) in rects:
        if rx <= px < rx + rw and ry <= py < ry + rh:
            return True
    return False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WorldPerceptionConfig:
    """Tunables for :class:`WorldPerception`."""

    # Horizontal FOV in degrees (must match what MC is rendering).
    h_fov_deg: float = 90.0

    # Patch grid for "look around me" sweeps. (n_cols, n_rows).
    # 24×14 = 336 patches gives roughly one ray per 60×60 pixel block
    # at 1920×1080, dense enough that any visible MC block at typical
    # distance is hit by at least one patch.
    patch_grid_size: Tuple[int, int] = (24, 14)

    # Each patch is this many pixels (square) sampled from the frame.
    # Smaller → faster + noisier; larger → slower + more representative.
    patch_size_px: int = 16

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

    # Excluded-rectangle list. Patches whose centre falls inside ANY
    # of these rectangles are skipped during the sweep AND ignored
    # when capturing samples for the recogniser. The HOTBAR + the
    # player's first-person HAND in the bottom-right corner are the
    # canonical entries. Each rect is (x, y, w, h) in screen pixels
    # at the captured 1920x1129 resolution; tools/window_inspector.py
    # helps tune these if the user changes GUI scale or item-in-hand
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

    # Run the entity detector every N ticks (1 = every frame).
    entity_detect_every_n_ticks: int = 4

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
    # downscale either way).
    sample_capture_px: int = 32

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
                 inverse_renderer: Optional[InverseRenderer] = None):
        self.cfg = config or WorldPerceptionConfig()
        self.block_classifier  = block_classifier
        self.entity_classifier = entity_classifier or build_entity_classifier()
        self.world_map         = world_map or WorldMap()
        self.sample_store      = sample_store
        self.inverse_renderer  = inverse_renderer
        self._tick             = 0
        self._screen_ray:      Optional[ScreenRay] = None
        self._last_frame_shape: Optional[Tuple[int, int]] = None
        # (block_id, voxel) -> last tick we sampled it; throttles repeats.
        self._last_sample_at: Dict[Tuple[str, Tuple[int, int, int]], int] = {}
        self._samples_saved: int = 0
        self._samples_saved_since_reload: int = 0
        # Stats useful for the live viewer / diagnostics.
        self._n_corrections: int = 0      # times a sample overrode a stale guess
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

        pose = self._pose_from_f3(f3)
        if pose is None:
            return wf
        wf.pose = pose
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
        if f3 is not None and f3.raw_text:
            la = parse_looking_at_block(f3.raw_text.splitlines())
            # Reject confidence-0.5 reads where the coord parser
            # couldn't extract a target position. The block id alone
            # at position (0,0,0) would pollute the map.
            if la is not None and la.confidence < 1.0:
                wf.diagnostics.setdefault("looking_at_rejects", []).append(
                    {"reason": "no_coords", "id": la.block_id})
                target_was_rejected = True
                la = None
            # Reject targets outside MC's block-reach distance from
            # the eye. F3 never reports unreachable blocks, so any
            # such reading is OCR garbage (wrong coords).
            if la is not None:
                tcx = la.pos[0] + 0.5
                tcy = la.pos[1] + 0.5
                tcz = la.pos[2] + 0.5
                d = math.sqrt((tcx - eye[0]) ** 2
                              + (tcy - eye[1]) ** 2
                              + (tcz - eye[2]) ** 2)
                if d > self.cfg.crosshair_reach_blocks + 1.5:
                    wf.diagnostics.setdefault("looking_at_rejects", []).append(
                        {"reason": "out_of_reach", "pos": list(la.pos),
                         "distance": round(d, 2),
                         "id": la.block_id})
                    target_was_rejected = True
                    la = None
            if la is not None:
                wf.looking_at = la

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
                    wf.diagnostics.setdefault("corrections", []).append({
                        "pos": list(la.pos),
                        "was": prev.block_id,
                        "now": la.block_id,
                    })

                self.world_map.update_block(BlockObservation(
                    pos          = la.pos,
                    block_id     = la.block_id,
                    confidence   = la.confidence,
                    source       = "looking_at",
                    last_seen_tick = self._tick,
                    dimension    = pose.dimension,
                    meta         = {"face": la.face} if la.face else {},
                ))
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
                self.world_map.update_block(crosshair_obs)
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
            for obs in self._sweep_patches(frame_rgb, sr, pose, anchor_depth):
                wf.visible_blocks.append(obs)
                # Strict commit gate: vision-patch guesses ONLY enter
                # the WorldMap when both (a) the sample-NN recogniser
                # gave a confident verdict, and (b) the voxel hasn't
                # been previously F3-confirmed as a different block.
                if self._should_commit(obs):
                    self.world_map.update_block(obs)
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
        if f3 is None or f3.x is None or f3.yaw is None or f3.pitch is None:
            return None
        return PlayerPose(
            x = float(f3.x),
            y = float(f3.y) if f3.y is not None else 0.0,
            z = float(f3.z) if f3.z is not None else 0.0,
            yaw = float(f3.yaw),
            pitch = float(f3.pitch),
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
        for pos, meta in ordered[:cap]:
            block_id = meta.get("block_id")
            if not block_id:
                continue
            # Already trusted by another path?
            if pos in self._confirmed:
                self._curiosity.pop(pos, None)
                continue
            score = ir.score_voxel(
                frame_rgb, pos, block_id,
                sr=sr, eye=eye, yaw=pose.yaw, pitch=pose.pitch,
            )
            if score < self.cfg.inverse_validate_score_min:
                continue
            self.world_map.update_block(BlockObservation(
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

        for _dist, pos in candidates[:budget]:
            score = ir.score_voxel(
                frame_rgb, pos, seed_block_id,
                sr=sr, eye=eye, yaw=pose.yaw, pitch=pose.pitch,
            )
            if score < cfg.expand_score_min:
                continue
            self.world_map.update_block(BlockObservation(
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
        out: List[Tuple[int, int, int]] = []
        for travel, vox in self._enumerate_voxels(eye, direction,
                                                    max_distance):
            out.append(vox)
        return out

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
        key = (block_id, voxel)
        last = self._last_sample_at.get(key, -10_000)
        if self._tick - last < self.cfg.sample_cooldown_ticks:
            return
        intr = sr.intrinsics
        patch = self._crop_patch(frame_rgb,
                                 int(intr.cx), int(intr.cy),
                                 self.cfg.sample_capture_px)
        if patch is None:
            return
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
        recogniser, refresh its in-memory tensor."""
        cls = self.block_classifier
        if cls is None:
            return
        reload_fn = getattr(cls, "reload_samples", None)
        if callable(reload_fn):
            try:
                reload_fn()
            except Exception:
                pass

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

        n_cols, n_rows = self.cfg.patch_grid_size
        if n_cols < 1 or n_rows < 1:
            return []

        H, W = frame_rgb.shape[:2]
        x0 = self.cfg.margin_left_px
        x1 = W - self.cfg.margin_right_px
        y0 = self.cfg.margin_top_px
        y1 = H - self.cfg.margin_bottom_px
        if x1 - x0 < self.cfg.patch_size_px or y1 - y0 < self.cfg.patch_size_px:
            return []

        observations: List[BlockObservation] = []
        # De-dupe: many patches may project to the same voxel; keep
        # the one with highest confidence per voxel and drop the rest.
        best_per_voxel: Dict[Tuple[int, int, int],
                              Tuple[float, BlockObservation]] = {}

        excludes = self.cfg.exclude_rects or ()
        for j in range(n_rows):
            for i in range(n_cols):
                px = int(x0 + (i + 0.5) * (x1 - x0) / n_cols)
                py = int(y0 + (j + 0.5) * (y1 - y0) / n_rows)
                # Skip patches landing inside any excluded rectangle —
                # hotbar, hand, future inventory overlays. This stops
                # the classifier from labelling the player's sword
                # texture as a real-world block.
                if _point_in_any_rect(px, py, excludes):
                    continue
                patch = self._crop_patch(frame_rgb, px, py,
                                         self.cfg.patch_size_px)
                if patch is None:
                    continue
                # Skip obvious sky / void patches without paying for a
                # full classify call: sky pixels are typically uniform
                # blue, mean R ≪ mean B.
                if self._looks_like_sky(patch):
                    continue
                block_id, conf = self.block_classifier.classify(patch)
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

    # ── Strict-commit + curiosity queue ────────────────────────────

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
        n = getattr(sample_rec, "sample_count", lambda: 0)()
        if n < self.cfg.min_samples_for_commit:
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
                              ) -> Optional[Tuple[int, int, int]]:
        """
        Pop and return one voxel for the agent to investigate next.

        ``prefer``:
          * ``"closest"`` — nearest voxel to the eye (default).
          * ``"oldest"``  — entry with the smallest ``seen_tick``.
          * ``"lowest_conf"`` — voxel where the classifier was least sure.

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
        out: Dict[str, Any] = {
            "tick": self._tick,
            "templates_loaded": (self.block_classifier.template_count()
                                  if self.block_classifier is not None else 0),
            "map": self.world_map.stats(),
            "samples_saved_this_session": self._samples_saved,
            "corrections_seen": self._n_corrections,
            "curiosity_size":  len(self._curiosity),
            "confirmed_count": len(self._confirmed),
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
                "expand_neighbour_radius",
                "expand_max_voxels_per_tick",
                "inverse_validate_max_per_tick"):
        if key in world_cfg:
            setattr(cfg, key, int(world_cfg[key]))
    for key in ("max_walk_distance", "crosshair_reach_blocks",
                "min_block_confidence",
                "default_anchor_depth", "depth_search_radius",
                "sample_commit_confidence",
                "expand_score_min",
                "inverse_validate_score_min"):
        if key in world_cfg:
            setattr(cfg, key, float(world_cfg[key]))
    if "auto_sample_from_looking_at" in world_cfg:
        cfg.auto_sample_from_looking_at = bool(
            world_cfg["auto_sample_from_looking_at"])
    for bkey in ("strict_commit_gate", "carve_air_along_sightlines",
                 "use_f3_depth_anchor",
                 "use_inverse_renderer",
                 "inverse_validate_curiosity",
                 "commit_only_from_looking_at"):
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

    if assets is None:
        from vision.mc_assets import MCAssets
        assets = MCAssets.load()
    baseline_cls = build_block_classifier(settings, assets=assets)

    # Wrap the baseline with a sample-NN recogniser unless explicitly
    # disabled. The hybrid is harmless when the sample store is empty
    # (it just falls through to the baseline every call).
    use_samples = bool(world_cfg.get("use_sample_recognizer", True))
    sample_store: Optional[WorldSampleStore] = None
    block_cls: BlockClassifierProtocol = baseline_cls
    if use_samples:
        sample_store = build_world_sample_store()
        sample_recognizer = SampleBlockRecognizer(sample_store)
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

    return WorldPerception(
        config=cfg,
        block_classifier=block_cls,
        entity_classifier=entity_cls,
        world_map=WorldMap(),
        sample_store=sample_store,
        inverse_renderer=inv_ren,
    )


__all__ = [
    "WorldPerception",
    "WorldPerceptionConfig",
    "build_world_perception",
]
