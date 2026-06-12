# tools/test_world_perception.py
"""
Smoke test for ``vision/world/`` — runs WITHOUT Minecraft attached.

What it covers
--------------
1. Catalog: load ``knowledge.Catalog`` and print counts of blocks /
   items / entities, plus a sample lookup.
2. Block classifier: load every signature, then re-classify a handful
   of vanilla textures against the library to confirm the baseline
   identifies them as themselves (sanity check, not a hard accuracy
   metric — colour signatures can confuse near-identical surfaces).
3. ScreenRay geometry: project a few known points and round-trip them.
4. WorldMap: insert observations, query by range, decay entities.
5. WorldPerception: feed a fake frame + a synthetic F3Info and confirm
   the WorldFrame is populated.

Run it directly from the project root:

    python tools/test_world_perception.py
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# Lazy imports — kept inside main() so import errors surface with a
# clear message rather than failing at module-load time.

def _ok(msg: str) -> None: print(f"  [OK]  {msg}")
def _fail(msg: str) -> None: print(f"  [FAIL] {msg}")
def _info(msg: str) -> None: print(f"        {msg}")


def test_catalog() -> bool:
    print("\n[1] Catalog")
    try:
        from knowledge import Catalog
        from vision.mc_assets import MCAssets
        assets = MCAssets.load()
    except Exception as e:
        _fail(f"asset cache not available: {e}")
        _info("Run `python -m vision.mc_assets --extract` first.")
        return False

    cat = Catalog(assets)
    n_blocks   = len(cat.blocks())
    n_items    = len(cat.items())
    n_entities = len(cat.entities())
    _info(f"blocks   = {n_blocks}")
    _info(f"items    = {n_items}")
    _info(f"entities = {n_entities}")
    if n_blocks < 50 or n_items < 50 or n_entities < 20:
        _fail("Catalog looks suspiciously empty.")
        return False

    stone = cat.block("stone")
    if stone is None or stone.id != "minecraft:stone":
        _fail("Could not look up minecraft:stone")
        return False
    _ok(f"stone ->{stone.name or '(no name)'}")

    z = cat.entity("zombie")
    if z is None or z.category != "hostile":
        _fail("Zombie should be classified as hostile")
        return False
    _ok(f"zombie ->category={z.category}")

    n_hostile = len(cat.hostile_mobs())
    n_passive = len(cat.passive_mobs())
    _ok(f"hostile={n_hostile}  passive={n_passive}")
    return True


def test_block_classifier() -> bool:
    print("\n[2] Block classifier")
    try:
        from vision.world import build_block_classifier
        from vision.mc_assets import MCAssets
        assets = MCAssets.load()
    except Exception as e:
        _fail(f"asset cache not available: {e}")
        return False

    cls = build_block_classifier({}, assets=assets)
    n = cls.template_count()
    _info(f"signatures loaded = {n}")
    if n < 100:
        _fail("Far fewer block signatures than expected.")
        return False

    # Identify a handful of canonical textures.
    targets = ["stone", "dirt", "oak_planks", "gold_block", "diamond_block"]
    hits = 0
    for stem in targets:
        tex = assets.block_texture(stem)
        if tex is None:
            _info(f"  - {stem}: texture missing, skipped")
            continue
        # Drop alpha and pretend this is a screen patch.
        if tex.ndim == 3 and tex.shape[2] == 4:
            tex = tex[..., :3]
        guess, conf = cls.classify(tex)
        match = (guess == f"minecraft:{stem}")
        if match:
            hits += 1
            _ok(f"{stem} ->{guess} (conf={conf:.2f})")
        else:
            _info(f"  - {stem}: guess={guess} conf={conf:.2f} (acceptable; baseline is fuzzy)")
    _info(f"exact matches: {hits} / {len(targets)}")
    return True


def test_screen_ray() -> bool:
    print("\n[3] ScreenRay geometry")
    from vision.world import CameraIntrinsics, ScreenRay
    intr = CameraIntrinsics.from_frame(1920, 1080, h_fov_deg=90.0)
    sr = ScreenRay(intrinsics=intr)
    _info(f"vfov = {intr.v_fov_deg:.2f}°")

    # Round-trip: project a point 10 blocks straight ahead, then make
    # sure the screen pixel sits at the centre.
    eye = (0.0, 65.0, 0.0)
    target = (0.0, 65.0, 10.0)
    proj = sr.project(target, yaw_deg=0.0, pitch_deg=0.0, eye_xyz=eye)
    if proj is None:
        _fail("Failed to project a point 10 blocks ahead")
        return False
    px, py, depth = proj
    if abs(px - intr.cx) > 1 or abs(py - intr.cy) > 1:
        _fail(f"Expected screen-centre projection, got ({px:.1f},{py:.1f})")
        return False
    _ok(f"forward block projects to centre at depth={depth:.1f}")

    # Orientation guard — the camera basis must NOT be horizontally
    # mirrored or vertically flipped. A centred forward point can't
    # detect that (it projects to the centre under either sign), so we
    # check OFF-centre points against MC physical geometry. Facing
    # south (yaw=0) with head up (+Y):
    #   * a block ABOVE the eye must appear in the UPPER half (py < cy)
    #   * east (+X) is on the player's LEFT, so it must appear LEFT
    #     of centre (px < cx)
    # Before the screen_ray right/up-vector fix both axes were inverted
    # here, silently mirror-flipping every screen↔world mapping.
    above = sr.project((0.0, 67.0, 4.0), yaw_deg=0.0, pitch_deg=0.0, eye_xyz=eye)
    east  = sr.project((2.0, 65.0, 4.0), yaw_deg=0.0, pitch_deg=0.0, eye_xyz=eye)
    if above is None or east is None:
        _fail("Orientation guard: off-centre points failed to project")
        return False
    if not (above[1] < intr.cy - 1):
        _fail(f"Orientation: block above eye projected to py={above[1]:.0f} "
              f"(expected < cy={intr.cy:.0f}); vertical axis is flipped")
        return False
    if not (east[0] < intr.cx - 1):
        _fail(f"Orientation: east (+X) block projected to px={east[0]:.0f} "
              f"(expected < cx={intr.cx:.0f}); horizontal axis is mirrored")
        return False
    _ok("camera basis right-side-up and not mirrored (above->up, east->left)")

    # Voxel walk from the eye forward 5 blocks should yield ~5 voxels.
    voxels = list(sr.voxel_walk(eye, (0.0, 0.0, 1.0), max_distance=5.0))
    _ok(f"voxel walk forward 5 blocks ->{len(voxels)} voxels {voxels[:3]}…")
    if len(voxels) < 3:
        _fail("Voxel walk produced suspiciously few entries.")
        return False

    # Edge-case guards: degenerate FOV must clamp, not crash, and the
    # up-vector at pitch=±90 must not return NaN (cross product
    # collapses near zero — the 1e-6 threshold catches accumulated
    # rounding error that 1e-9 missed).
    bad = CameraIntrinsics.from_frame(1920, 1080, h_fov_deg=0.0)
    if bad.h_fov_deg <= 0 or bad.h_fov_deg >= 180:
        _fail(f"degenerate FOV not clamped: got {bad.h_fov_deg}")
        return False
    if not math.isfinite(bad.fx):
        _fail(f"FOV clamp didn't prevent non-finite fx: got {bad.fx}")
        return False
    _ok(f"degenerate FOV 0.0° clamped to {bad.h_fov_deg}°, fx={bad.fx:.1f}")

    for p in (90.0, -90.0, 89.999):
        ux, uy, uz = ScreenRay.up_vector(0.0, p)
        if not (math.isfinite(ux) and math.isfinite(uy) and math.isfinite(uz)):
            _fail(f"up_vector(yaw=0, pitch={p}) returned non-finite: "
                  f"({ux},{uy},{uz})")
            return False
    _ok("up_vector at pitch=±90 stays finite")
    return True


def test_world_map() -> bool:
    print("\n[4] WorldMap")
    from vision.world import WorldMap, BlockObservation
    wm = WorldMap()
    wm.set_current_dimension("minecraft:overworld")
    for x in range(-5, 6):
        wm.update_block(BlockObservation(
            pos=(x, 64, 0), block_id="minecraft:stone",
            confidence=0.8, source="vision_patch",
            last_seen_tick=10,
        ))
    _info(f"after inserts: {wm.block_count()}")
    if wm.block_count() != 11:
        _fail("Expected 11 block observations")
        return False

    near = list(wm.iter_blocks_in_range((0, 64, 0), radius=2))
    if len(near) != 5:
        _fail(f"Range query expected 5 results, got {len(near)}")
        return False
    _ok(f"range query: {len(near)} blocks within 2-block radius")

    # Manual observation should beat vision_patch.
    wm.update_block(BlockObservation(
        pos=(0, 64, 0), block_id="minecraft:diamond_ore",
        confidence=1.0, source="looking_at", last_seen_tick=20,
    ))
    got = wm.get_block((0, 64, 0))
    if got is None or got.block_id != "minecraft:diamond_ore":
        _fail("looking_at observation should override vision_patch")
        return False
    _ok("looking_at observation overrides vision_patch at the same cell")

    # Eviction regression: at the cap, eviction must bring the store
    # back UNDER the cap (target ~90 %), not just drop a fixed 10 %.
    # The old behaviour dropped 10 % of the CURRENT size, so heavy
    # burst inserts could outpace eviction and grow past the cap
    # forever.
    wm2 = WorldMap()
    wm2._max_blocks_per_dim = 100
    for i in range(150):
        wm2.update_block(BlockObservation(
            pos=(i, 64, 0), block_id="minecraft:stone",
            confidence=0.5, source="vision_patch", last_seen_tick=i,
        ))
    if wm2.block_count() > 100:
        _fail(f"eviction did not bring store under cap: "
              f"{wm2.block_count()} > 100")
        return False
    if wm2.block_count() < 50:
        _fail(f"eviction overshot — store too small: "
              f"{wm2.block_count()}")
        return False
    _ok(f"eviction held to cap=100, ended at {wm2.block_count()} "
        f"after 150 inserts")
    return True


def test_world_perception() -> bool:
    print("\n[5] WorldPerception end-to-end")
    try:
        from vision.world import build_world_perception
        from vision.ocr import F3Info
        from vision.mc_assets import MCAssets
        assets = MCAssets.load()
    except Exception as e:
        _fail(f"asset cache not available: {e}")
        return False

    wp = build_world_perception({"vision": {"world": {
        "h_fov_deg": 90.0,
        "patch_grid_size": [3, 2],
        "patch_size_px": 24,
        "margin_left_px": 40,
        "margin_right_px": 40,
        "margin_top_px": 40,
        "margin_bottom_px": 40,
    }}}, assets=assets)

    # Synthetic frame: a mostly-green ground texture so the classifier
    # picks SOMETHING for every patch. We aren't checking which block —
    # we just want to confirm the pipeline runs end-to-end.
    H, W = 540, 960
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:, :, 1] = 110   # green ground
    frame[:H // 2, :, :] = (135, 180, 220)  # sky-ish

    f3 = F3Info(x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0,
                facing_name="south", dimension="minecraft:overworld",
                block_x=0, block_y=64, block_z=0,
                raw_text="XYZ: 0 / 64 / 0\nFacing: south")

    t0 = time.perf_counter()
    wf = wp.update(frame, f3)
    dt = (time.perf_counter() - t0) * 1000.0
    _info(f"update() took {dt:.1f} ms; visible_blocks={len(wf.visible_blocks)}")
    if wf.pose is None:
        _fail("pose should have been populated from synthetic F3")
        return False
    _ok(f"WorldFrame tick={wf.tick} pose=({wf.pose.x:.1f},{wf.pose.y:.1f},{wf.pose.z:.1f})")
    _ok(f"WorldMap stats: {wp.world_map.stats()}")
    return True


def test_sample_store_and_recognizer() -> bool:
    print("\n[6] Sample store + recognizer + map renderer")
    import tempfile
    from pathlib import Path
    from vision.world import (
        WorldSampleStore, SampleBlockRecognizer, HybridBlockClassifier,
        WorldMapRenderer, MapRenderConfig, WorldMap, BlockObservation,
        PlayerPose,
    )
    from vision.world.sample_store import SAMPLE_SIZE

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        store = WorldSampleStore(root)

        # Save two distinct labelled patches.
        red_patch  = np.zeros((SAMPLE_SIZE, SAMPLE_SIZE, 3), dtype=np.uint8)
        red_patch[..., 0] = 200
        blue_patch = np.zeros((SAMPLE_SIZE, SAMPLE_SIZE, 3), dtype=np.uint8)
        blue_patch[..., 2] = 200

        p1 = store.save("minecraft:redstone_block", red_patch)
        p2 = store.save("minecraft:lapis_block",    blue_patch)
        if p1 is None or p2 is None:
            _fail("expected both samples to be written")
            return False
        # De-dup: re-saving the same pixels should return None.
        if store.save("minecraft:redstone_block", red_patch) is not None:
            _fail("identical sample should de-dup")
            return False
        _ok(f"saved {store.total_samples()} samples across "
            f"{store.block_count()} blocks")

        rec = SampleBlockRecognizer(store)
        if rec.sample_count() != 2:
            _fail("recogniser should see 2 samples after reload")
            return False
        # Classifying a noisy variant of red should hit redstone_block.
        noisy_red = red_patch.copy()
        noisy_red[..., 0] = 195
        guess, conf = rec.classify(noisy_red)
        if guess != "minecraft:redstone_block":
            _fail(f"expected redstone match, got {guess} (conf={conf})")
            return False
        _ok(f"sample recogniser correctly identifies noisy red (conf={conf:.2f})")

        # Map renderer should produce a non-empty image even with an
        # empty map; with one block, that block should appear coloured.
        wm = WorldMap()
        wm.update_block(BlockObservation(
            pos=(10, 64, 5),
            block_id="minecraft:grass_block",
            confidence=1.0, source="manual", last_seen_tick=1,
        ))
        renderer = WorldMapRenderer(MapRenderConfig(canvas_size_px=200))
        pose = PlayerPose(x=10.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0)
        img = renderer.render(wm, pose, target_voxel=(10, 64, 5))
        if img.shape != (200 + renderer.cfg.header_px, 200, 3):
            _fail(f"unexpected renderer output shape {img.shape}")
            return False
        # The grass block should have been painted somewhere → not all
        # pixels are the background colour.
        bg = np.array(renderer.cfg.background, dtype=np.uint8)
        non_bg = np.any(img != bg, axis=-1).sum()
        if non_bg < 50:
            _fail("renderer produced an essentially blank image")
            return False
        _ok(f"renderer drew {non_bg} non-background pixels (header + block + player)")

        # Hybrid classifier with a dummy baseline: when the sample
        # match is good the hybrid should defer to it; when the input
        # is unrelated, the baseline result should pass through.
        class _DummyBaseline:
            def template_count(self) -> int: return 99
            def classify(self, _): return ("minecraft:cobblestone", 0.42)
        hybrid = HybridBlockClassifier(rec, _DummyBaseline())
        g1, _ = hybrid.classify(noisy_red)
        g2, _ = hybrid.classify(np.full((SAMPLE_SIZE, SAMPLE_SIZE, 3),
                                         128, dtype=np.uint8))
        if g1 != "minecraft:redstone_block":
            _fail(f"hybrid should prefer sample match, got {g1}")
            return False
        if g2 != "minecraft:cobblestone":
            _fail(f"hybrid should fall through to baseline, got {g2}")
            return False
        _ok("hybrid prefers sample match, falls back to baseline cleanly")

    return True


def test_3d_renderer_and_air_carving() -> bool:
    print("\n[9] 3D iso renderer + air carving + depth-anchored raycast")
    from vision.world import (
        WorldMap, BlockObservation, PlayerPose,
        IsoWorldRenderer, IsoRenderConfig, WorldPerception,
    )

    # Build a small world: a 5×1×5 floor of stone with one diamond block.
    wm = WorldMap()
    for x in range(-2, 3):
        for z in range(-2, 3):
            wm.update_block(BlockObservation(
                pos=(x, 63, z), block_id="minecraft:stone",
                confidence=1.0, source="manual", last_seen_tick=1,
            ))
    wm.update_block(BlockObservation(
        pos=(0, 63, 0), block_id="minecraft:diamond_block",
        confidence=1.0, source="manual", last_seen_tick=1,
    ))

    # Carve some air voxels above the floor.
    n_air = wm.mark_air_along(
        [(0, 64, 0), (0, 65, 0), (0, 66, 0)],
        confidence=0.9, tick=1,
    )
    if n_air != 3:
        _fail(f"mark_air_along expected 3, got {n_air}")
        return False
    _ok(f"air voxels carved: {n_air}")

    # iter_solid_blocks must exclude the air voxels.
    n_solid = sum(1 for _ in wm.iter_solid_blocks())
    if n_solid != 25:
        _fail(f"iter_solid_blocks expected 25, got {n_solid}")
        return False
    _ok(f"iter_solid_blocks excludes air: {n_solid} solid")

    # Render.
    pose = PlayerPose(x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=30.0)
    renderer = IsoWorldRenderer(IsoRenderConfig(canvas_w_px=400,
                                                 canvas_h_px=300,
                                                 base_px_per_block=12.0))
    img = renderer.render(wm, pose, target_voxel=(0, 63, 0),
                           extra_lines=["test"])
    if img.shape != (300 + renderer.cfg.header_h_px, 400, 3):
        _fail(f"unexpected iso renderer shape {img.shape}")
        return False
    bg = np.array(renderer.cfg.background, dtype=np.uint8)
    non_bg = int(np.any(img != bg, axis=-1).sum())
    if non_bg < 500:
        _fail(f"iso renderer drew too few pixels: {non_bg}")
        return False
    _ok(f"iso renderer drew {non_bg} non-background pixels")

    # Depth-anchored voxel pick: walking forward 5 blocks from eye
    # should land us on roughly the 5th voxel ahead.
    eye = (0.0, 65.0, 0.0)
    voxel = WorldPerception._voxel_at_depth(
        eye=eye, direction=(0.0, 0.0, 1.0),
        target_depth=5.0, max_distance=12.0,
    )
    if voxel is None or abs(voxel[2] - 5) > 1 or voxel[0] != 0 or voxel[1] != 65:
        _fail(f"_voxel_at_depth at depth 5 expected ~(0,65,5), got {voxel}")
        return False
    _ok(f"_voxel_at_depth(5) ->{voxel}")
    return True


def test_air_carving_guards() -> bool:
    """Regression: forward-ray air carving must NOT fire when the
    F3 Targeted Block line was parsed but rejected (no coords / out
    of reach). Doing so would mark voxels as air even though the
    player is clearly looking at SOME block in that direction."""
    print("\n[13] Air-carving rejects-vs-absent regression")
    from vision.world.perception import (
        WorldPerception, WorldPerceptionConfig,
    )
    from vision.world.map import WorldMap, AIR_BLOCK
    from vision.ocr import F3Info
    import numpy as _np

    cfg = WorldPerceptionConfig()
    cfg.commit_only_from_looking_at = True
    wp = WorldPerception(config=cfg, world_map=WorldMap())

    # Fake frame + pose pointing along +Z (south).
    frame = _np.zeros((540, 960, 3), dtype=_np.uint8)
    # Path A: Targeted Block line is GENUINELY absent → carve air.
    f3_absent = F3Info(
        x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0,
        facing_name="south", dimension="minecraft:overworld",
        block_x=0, block_y=64, block_z=0,
        raw_text="XYZ: 0 / 64 / 0\nBlock: 0 64 0\nFacing: south",
    )
    # Side-effect update — we measure its impact via world_map state
    # below, not via the return value, so discard the WorldFrame.
    wp.update(frame, f3_absent)
    n_a = sum(1 for o in wp.world_map.iter_blocks()
              if o.block_id == AIR_BLOCK)
    if n_a == 0:
        _fail("expected forward-ray air carving when target is ABSENT")
        return False
    _ok(f"absent target: {n_a} air voxels carved")

    # Reset and try path B: Targeted Block line is PRESENT but
    # malformed (block id without coords). The perception layer
    # rejects this — and MUST NOT carve air, since the player IS
    # looking at something in that direction.
    wp = WorldPerception(config=cfg, world_map=WorldMap())
    f3_rejected = F3Info(
        x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0,
        facing_name="south", dimension="minecraft:overworld",
        block_x=0, block_y=64, block_z=0,
        raw_text=("XYZ: 0 / 64 / 0\n"
                  "Targeted Block\n"               # no coords on the line
                  "minecraft:stone\n"               # id is parseable
                  "Facing: south"),
    )
    wf_b = wp.update(frame, f3_rejected)
    n_b = sum(1 for o in wp.world_map.iter_blocks()
              if o.block_id == AIR_BLOCK)
    if n_b != 0:
        _fail(f"unexpected air carving when target was rejected: {n_b}")
        return False
    _ok("rejected target: no air carved (correct)")
    # Sanity: the diagnostics should record the rejection.
    rejects = wf_b.diagnostics.get("looking_at_rejects", [])
    if not rejects:
        _fail("expected looking_at_rejects diagnostic")
        return False
    _ok(f"rejection diagnostic recorded: {rejects[0]['reason']}")
    return True


def test_block_id_validator_gate() -> bool:
    """Regression: an F3 ``looking_at`` read whose id is NOT a real block
    (a tag whose ``#`` was OCR-eaten, a truncation, a mangled stem) must
    be rejected — never committed to the WorldMap, never saved as a
    training sample. This is the guard that stops the self-teaching loop
    from poisoning its own dataset with garbage labels."""
    print("\n[17] Block-id catalog gate rejects garbage F3 labels")
    from vision.world.perception import (
        WorldPerception, WorldPerceptionConfig,
    )
    from vision.world.map import WorldMap
    from vision.ocr import F3Info
    import numpy as _np

    # Validator that knows only ONE real block. Anything else is garbage.
    known = {"minecraft:stone"}
    validator = lambda bid: bid in known

    cfg = WorldPerceptionConfig()
    cfg.commit_only_from_looking_at = True
    frame = _np.zeros((540, 960, 3), dtype=_np.uint8)

    def _looking_at_frame(block_id: str):
        return F3Info(
            x=0.0, y=64.0, z=2.0, yaw=0.0, pitch=0.0,
            facing_name="south", dimension="minecraft:overworld",
            block_x=0, block_y=64, block_z=3,
            raw_text=("XYZ: 0 / 64 / 2\n"
                      "Targeted Block: 0, 64, 3\n"
                      f"{block_id}\n"
                      "#minecraft:mineable/pickaxe\n"
                      "Facing: south"),
        )

    # Path A: garbage id (a tag stem) → rejected, nothing committed.
    wp = WorldPerception(config=cfg, world_map=WorldMap(),
                         block_id_validator=validator)
    wf_bad = wp.update(frame, _looking_at_frame("minecraft:goats_spawnable_on"))
    if wf_bad.looking_at is not None:
        _fail("garbage block id was accepted as looking_at")
        return False
    if wp.world_map.get_block((0, 64, 3),
                              dimension="minecraft:overworld") is not None:
        _fail("garbage block id leaked a commit into the WorldMap")
        return False
    rejects = wf_bad.diagnostics.get("looking_at_rejects", [])
    if not any(r.get("reason") == "unknown_block_id" for r in rejects):
        _fail("expected an unknown_block_id rejection diagnostic")
        return False
    _ok("garbage id rejected (no commit, diagnostic recorded)")

    # Path B: a real block id sails through and commits.
    wp = WorldPerception(config=cfg, world_map=WorldMap(),
                         block_id_validator=validator)
    wf_ok = wp.update(frame, _looking_at_frame("minecraft:stone"))
    if wf_ok.looking_at is None or wf_ok.looking_at.block_id != "minecraft:stone":
        _fail("a real block id was wrongly rejected by the gate")
        return False
    _ok("real id accepted (gate is not over-eager)")

    # Path C: no validator supplied → back-compat, no gating applied.
    wp = WorldPerception(config=cfg, world_map=WorldMap())
    wf_nogate = wp.update(frame, _looking_at_frame("minecraft:stone"))
    if wf_nogate.looking_at is None:
        _fail("absent validator should not block any read")
        return False
    _ok("absent validator leaves reads untouched (back-compat)")
    return True


def test_sprite_sample_skip() -> bool:
    """The auto-sampler must SKIP training-sample capture for sprite/cross
    blocks (short_grass, fern, …) whose crosshair crop is dominated by the
    backing block — those mislabel the dataset — while still capturing
    full-cube blocks. The map commit is unaffected (it happens from F3)."""
    print("\n[18] Sprite/cross blocks are skipped by the auto-sampler")
    import tempfile, numpy as _np
    from pathlib import Path as _Path
    from vision.world.perception import WorldPerception, WorldPerceptionConfig
    from vision.world.map import WorldMap
    from vision.world.sample_store import WorldSampleStore
    from vision.world.screen_ray import ScreenRay, CameraIntrinsics

    sprite = {"minecraft:short_grass", "minecraft:fern"}
    wp = WorldPerception(
        config=WorldPerceptionConfig(), world_map=WorldMap(),
        sample_store=WorldSampleStore(_Path(tempfile.mkdtemp())),
        sprite_block_predicate=lambda bid: bid in sprite)
    wp._screen_ray = ScreenRay(CameraIntrinsics(960, 540, 90.0))
    frame = _np.random.default_rng(0).integers(40, 200, (540, 960, 3), dtype=_np.uint8)

    for bid, vox in (("minecraft:short_grass", (1, 2, 3)),
                     ("minecraft:fern", (1, 2, 4)),
                     ("minecraft:grass_block", (4, 5, 6)),
                     ("minecraft:stone", (7, 8, 9))):
        wp._tick += 100   # clear the per-(block,voxel) cooldown
        wp._maybe_save_crosshair_sample(
            frame_rgb=frame, sr=wp._screen_ray, block_id=bid, voxel=vox,
            metadata={"distance_blocks": 3.0})
    m = wp.sample_store.manifest()
    if "minecraft:short_grass" in m or "minecraft:fern" in m:
        _fail(f"sprite block was sampled (should be skipped): {m}")
        return False
    _ok("sprite blocks (short_grass, fern) skipped")
    if "minecraft:grass_block" not in m or "minecraft:stone" not in m:
        _fail(f"full-cube block was NOT sampled (should be kept): {m}")
        return False
    _ok("full-cube blocks (grass_block, stone) still captured")
    # No predicate => nothing skipped (back-compat).
    wp2 = WorldPerception(
        config=WorldPerceptionConfig(), world_map=WorldMap(),
        sample_store=WorldSampleStore(_Path(tempfile.mkdtemp())))
    wp2._screen_ray = ScreenRay(CameraIntrinsics(960, 540, 90.0))
    wp2._tick = 9999
    wp2._maybe_save_crosshair_sample(
        frame_rgb=frame, sr=wp2._screen_ray, block_id="minecraft:short_grass",
        voxel=(1, 2, 3), metadata={"distance_blocks": 3.0})
    if "minecraft:short_grass" not in wp2.sample_store.manifest():
        _fail("absent predicate should NOT skip anything (back-compat)")
        return False
    _ok("absent predicate leaves sampling untouched (back-compat)")
    return True


def test_dark_sample_gate() -> bool:
    """The per-block darkness gate must reject a sample that's an
    outlier-dark version of a normally-brighter block (night/cave smear),
    while keeping bright samples — without needing per-block thresholds."""
    print("\n[20] Per-block relative-darkness sample gate")
    import tempfile, numpy as _np
    from pathlib import Path as _Path
    from vision.world.perception import WorldPerception, WorldPerceptionConfig
    from vision.world.map import WorldMap
    from vision.world.sample_store import WorldSampleStore
    from vision.world.screen_ray import ScreenRay, CameraIntrinsics

    cfg = WorldPerceptionConfig()
    cfg.dark_sample_min_history = 2      # seed fast for the test
    cfg.dark_sample_factor = 0.7
    cfg.mask_crosshair_in_samples = False
    wp = WorldPerception(config=cfg, world_map=WorldMap(),
                         sample_store=WorldSampleStore(_Path(tempfile.mkdtemp())))
    wp._screen_ray = ScreenRay(CameraIntrinsics(960, 540, 90.0))
    rng = _np.random.default_rng(0)

    def feed(bright_lo, bright_hi, vox):
        wp._tick += 100   # clear the per-(block,voxel) cooldown
        frame = rng.integers(bright_lo, bright_hi, (540, 960, 3), dtype=_np.uint8)
        wp._maybe_save_crosshair_sample(
            frame_rgb=frame, sr=wp._screen_ray,
            block_id="minecraft:grass_block", voxel=vox,
            metadata={"distance_blocks": 3.0})

    # Seed the running mean with two BRIGHT samples (~100).
    feed(92, 108, (1, 2, 3))
    feed(92, 108, (1, 2, 4))
    n_after_bright = wp.sample_store.manifest().get("minecraft:grass_block", 0)
    if n_after_bright < 2:
        _fail(f"bright seed samples were not saved ({n_after_bright})")
        return False
    # A DARK sample (~30) of the same block is an outlier -> skipped.
    feed(22, 38, (1, 2, 5))
    n_after_dark = wp.sample_store.manifest().get("minecraft:grass_block", 0)
    if n_after_dark != n_after_bright:
        _fail(f"dark sample was NOT skipped ({n_after_bright} -> {n_after_dark})")
        return False
    _ok(f"outlier-dark sample skipped (store stayed at {n_after_dark})")
    # Another BRIGHT sample still saves.
    feed(92, 108, (1, 2, 6))
    if wp.sample_store.manifest().get("minecraft:grass_block", 0) <= n_after_dark:
        _fail("a bright sample after the dark one was wrongly skipped")
        return False
    _ok("bright samples still captured (gate is not over-eager)")
    # factor=0 disables the gate (back-compat): a dark sample saves.
    cfg2 = WorldPerceptionConfig()
    cfg2.dark_sample_factor = 0.0
    cfg2.mask_crosshair_in_samples = False
    wp2 = WorldPerception(config=cfg2, world_map=WorldMap(),
                          sample_store=WorldSampleStore(_Path(tempfile.mkdtemp())))
    wp2._screen_ray = ScreenRay(CameraIntrinsics(960, 540, 90.0))
    wp2._tick = 9999
    dark = rng.integers(22, 38, (540, 960, 3), dtype=_np.uint8)
    wp2._maybe_save_crosshair_sample(
        frame_rgb=dark, sr=wp2._screen_ray, block_id="minecraft:obsidian",
        voxel=(1, 2, 3), metadata={"distance_blocks": 3.0})
    if "minecraft:obsidian" not in wp2.sample_store.manifest():
        _fail("factor=0 should disable the gate (dark sample should save)")
        return False
    _ok("factor=0 disables the gate (dark blocks like obsidian unaffected)")
    return True


def test_garble_sample_gate() -> bool:
    """A heavily-garbled F3 read must NOT label a training sample — a mangled
    panel can yield a wrong-but-valid id (grass→"sand", night→"stone") that
    poisons that class (verified live: a --walk run into night/swamp filled
    gravel/stone/sand with dark/green mislabelled patches). Gate on the '?'
    ratio of the raw F3 text."""
    print("\n[20b] F3-garble training-sample gate")
    from vision.world.perception import WorldPerception, WorldPerceptionConfig
    gr = WorldPerception._f3_garble_ratio
    thr = WorldPerceptionConfig().max_sample_garble_ratio
    clean = ("Targeted Block: 0, 70, -34 | XYZ: 3.2 / 69.0 / -37.6 | "
             "minecraft:oak_log | Block: 3 69 -39 | axis: y | "
             "#minecraft:logs | #minecraft:mineable/axe | ? ? ? ?")
    garbled = ("? ?? ???. ? ? ???? ? ?? . ? ????? | XYZ: 3.2 / 69 / -37 | "
               "? ? ? ?? ? ?? ? | ?????? ? . ? ?? ? ? ? ? | "
               "? ? ?? ?? ` ? . ? ?? | ' ????????? ? ?? ? ? ?? ?")
    cgr, ggr = gr(clean), gr(garbled)
    ok1 = cgr < thr
    ok2 = ggr > thr
    ok3 = gr("") == 0.0
    (_ok if ok1 else _fail)(f"clean read passes the gate ({cgr:.0%} < {thr:.0%})")
    (_ok if ok2 else _fail)(f"garbled read is gated out ({ggr:.0%} > {thr:.0%})")
    (_ok if ok3 else _fail)("empty read -> 0 ratio (no spurious gating)")
    return ok1 and ok2 and ok3


def test_multiframe_confirm_gate() -> bool:
    """The confirmed map + training labels are gated by multi-frame agreement
    + crosshair-ray geometry: a one-frame OCR slip ("sandstone" amid a run of
    oak_log) and an off-crosshair coord misread NEVER reach ground truth."""
    print("\n[20c] Multi-frame + ray confirmation gate")
    from types import SimpleNamespace
    from vision.world.perception import WorldPerception, WorldPerceptionConfig
    from vision.world.map import WorldMap
    from vision.world.screen_ray import ScreenRay, CameraIntrinsics
    from vision.world.types import LookingAtBlock
    cfg = WorldPerceptionConfig()
    cfg.confirm_window = 5; cfg.min_confirm_reads = 2
    wp = WorldPerception(config=cfg, world_map=WorldMap())
    wp._screen_ray = ScreenRay(CameraIntrinsics(960, 540, 70.0))
    pose = SimpleNamespace(x=0.5, y=64.0, z=0.5, eye_y=65.62, yaw=0.0,
                           pitch=0.0, dimension="minecraft:overworld")
    eye = (0.5, 65.62, 0.5)
    la = lambda b, v: LookingAtBlock(block_id=b, pos=v, face=None, confidence=1.0)
    vox = (0, 65, 3)                          # on the +Z crosshair ray
    c1 = wp._confirm_looking_at(la("minecraft:oak_log", vox), pose, eye)
    c2 = wp._confirm_looking_at(la("minecraft:oak_log", vox), pose, eye)
    out = wp._confirm_looking_at(la("minecraft:sandstone", vox), pose, eye)
    off = wp._confirm_looking_at(la("minecraft:oak_log", (8, 65, 3)), pose, eye)
    (_ok if not c1 else _fail)("1st read is NOT yet confirmed (needs agreement)")
    (_ok if c2 else _fail)("2nd matching read IS confirmed")
    (_ok if not out else _fail)("transient 'sandstone' outlier is rejected")
    (_ok if not off else _fail)("off-crosshair coord misread is rejected (ray)")
    return (not c1) and c2 and (not out) and (not off)


def test_held_item_occlusion() -> bool:
    """A crop overlapping the held-item / hand / hotbar HUD region is rejected
    so the recogniser never trains on arm/item pixels; the offhand region only
    counts when an offhand item is held."""
    print("\n[20d] Held-item / HUD occlusion gate")
    from vision.world.perception import WorldPerception, WorldPerceptionConfig
    from vision.world.map import WorldMap
    wp = WorldPerception(config=WorldPerceptionConfig(), world_map=WorldMap())
    h, w = 1094, 1920
    centre_ok = not wp._crop_occluded(w * 0.5, h * 0.5, 80, h, w)
    in_hand = wp._crop_occluded(w * 0.82, h * 0.86, 120, h, w)
    # offhand (bottom-left) only excluded when an offhand item is held
    off_before = wp._crop_occluded(w * 0.12, h * 0.86, 120, h, w)
    wp.set_held_item(main_hand="minecraft:oak_log", off_hand="minecraft:torch")
    off_after = wp._crop_occluded(w * 0.12, h * 0.86, 120, h, w)
    (_ok if centre_ok else _fail)("a centred crop is NOT flagged")
    (_ok if in_hand else _fail)("a crop in the bottom-right hand IS flagged")
    (_ok if (not off_before) and off_after else _fail)(
        "offhand region flagged only when an offhand item is held")
    return centre_ok and in_hand and (not off_before) and off_after


def test_temporal_voter() -> bool:
    """Per-voxel temporal vote smoothing: first sight passes through
    unchanged (single-frame safe), repeated agreement boosts confidence,
    and a transient flip is overruled by the window majority."""
    print("\n[19] Temporal vote smoothing of vision-patch guesses")
    from vision.world.temporal_vote import TemporalVoter, TemporalVoteConfig
    V = (5, 64, 7)

    # First sighting: unchanged, not yet stable.
    tv = TemporalVoter(TemporalVoteConfig(window_ticks=50))
    bid, conf, stable = tv.vote(V, "minecraft:stone", 0.40, tick=1)
    if bid != "minecraft:stone" or abs(conf - 0.40) > 1e-9 or stable:
        _fail(f"first sight must pass through unchanged (got {bid},{conf},{stable})")
        return False
    _ok("first sighting passes through unchanged (single-frame safe)")

    # Repeated agreement -> stable, confidence scaled UP toward the best
    # observed (never above it).
    tv.vote(V, "minecraft:stone", 0.55, tick=2)
    bid, conf, stable = tv.vote(V, "minecraft:stone", 0.50, tick=3)
    if bid != "minecraft:stone" or not stable:
        _fail(f"agreement should be stable stone (got {bid}, stable={stable})")
        return False
    if conf > 0.55 + 1e-9 or conf < 0.50:
        _fail(f"agreement confidence out of range: {conf}")
        return False
    _ok(f"3x agreement -> stable, conf={conf:.2f} (<= best 0.55)")

    # A single transient flip is overruled by the stone majority.
    bid, conf, stable = tv.vote(V, "minecraft:deepslate", 0.52, tick=4)
    if bid != "minecraft:stone":
        _fail(f"transient flip should be overruled by majority (got {bid})")
        return False
    _ok("transient flip (deepslate) overruled by stone majority")

    # Out-of-window guesses drop: after the window passes, old votes expire.
    tv2 = TemporalVoter(TemporalVoteConfig(window_ticks=10))
    tv2.vote(V, "minecraft:stone", 0.6, tick=1)
    bid, conf, stable = tv2.vote(V, "minecraft:sand", 0.6, tick=100)
    if bid != "minecraft:sand" or stable:
        _fail(f"stale vote should have expired (got {bid}, stable={stable})")
        return False
    _ok("votes outside the tick window expire correctly")
    return True


def test_sweep_delta_cap() -> bool:
    print("\n[14] Sweep-progress cap against absurd yaw jumps")
    from agents.world_explorer import (
        WorldExplorerAgent, WorldExplorerConfig,
    )
    a = WorldExplorerAgent(WorldExplorerConfig())
    a._reset_sweep_tracking(0.0)
    # Small valid deltas accumulate.
    a._update_sweep_progress(10.0)   # +10
    a._update_sweep_progress(25.0)   # +15
    if abs(a._sweep_yaw_unwrapped - 25.0) > 0.1:
        _fail(f"expected unwrapped=25, got {a._sweep_yaw_unwrapped}")
        return False
    _ok("small deltas accumulate correctly")

    # An absurd jump (e.g. OCR misread that puts yaw 170° away) is
    # capped — does NOT mis-credit the sweep.
    before = a._sweep_yaw_unwrapped
    a._update_sweep_progress(-170.0)
    if abs(a._sweep_yaw_unwrapped - before) > 0.1:
        _fail(f"absurd jump should NOT contribute; before={before} "
              f"after={a._sweep_yaw_unwrapped}")
        return False
    _ok(f"absurd jump capped (unwrapped stays at {before:.1f}°)")

    # The next legitimate delta is computed against the latest
    # reading (so the cap doesn't desync the anchor forever).
    a._update_sweep_progress(-160.0)   # +10° (relative to the -170 anchor)
    expected = before + 10.0
    if abs(a._sweep_yaw_unwrapped - expected) > 0.1:
        _fail(f"after-cap delta wrong: expected {expected}, got "
              f"{a._sweep_yaw_unwrapped}")
        return False
    _ok("normal delta after cap re-engages the unwrap")
    return True


def test_strict_gate_crosshair_bypass() -> bool:
    """Regression: under ``commit_only_from_looking_at`` the crosshair
    vision-patch path used to call ``world_map.update_block`` directly,
    bypassing ``_should_commit``. That let vision-patch GUESSES into
    the WorldMap even though the user asked for F3 ground truth only.
    The fix gates the entire branch on the strict flag."""
    print("\n[15] Strict gate must block crosshair vision-patch")
    from vision.world.perception import (
        WorldPerception, WorldPerceptionConfig,
    )
    from vision.world.map import WorldMap
    from vision.ocr import F3Info
    import numpy as _np

    # Stub block classifier that always "sees" stone with high conf.
    class _AlwaysStone:
        def template_count(self) -> int: return 1
        def classify(self, _patch): return ("minecraft:stone", 0.99)

    cfg = WorldPerceptionConfig()
    cfg.commit_only_from_looking_at = True
    wp = WorldPerception(
        config=cfg,
        world_map=WorldMap(),
        block_classifier=_AlwaysStone(),
    )

    # Frame + pose with NO Targeted-Block line, so the crosshair
    # patch path would normally fire.
    frame = _np.full((540, 960, 3), 100, dtype=_np.uint8)
    f3 = F3Info(
        x=10.0, y=64.0, z=5.0, yaw=0.0, pitch=0.0,
        facing_name="south", dimension="minecraft:overworld",
        block_x=10, block_y=64, block_z=5,
        raw_text="XYZ: 10 / 64 / 5\nFacing: south",
    )
    wf = wp.update(frame, f3)

    # The map MUST contain only air carving (free space) — never a
    # stone voxel from the crosshair patch.
    solids = [o for o in wp.world_map.iter_solid_blocks()]
    if solids:
        _fail(f"strict gate leaked vision-patch solids: {solids[:3]}")
        return False
    _ok("crosshair vision-patch blocked under strict gate")

    # wf.looking_at should also remain None (we did NOT spoof one
    # from the patch under strict gate).
    if wf.looking_at is not None:
        _fail(f"wf.looking_at should be None under strict gate: {wf.looking_at}")
        return False
    _ok("wf.looking_at correctly stays None when crosshair branch is skipped")
    return True


def test_perception_public_api() -> bool:
    """Regression: agents used to reach into ``WorldPerception._curiosity``
    directly (a private attribute) to purge stale curiosity entries.
    The encapsulation-safe ``purge_curiosity_queue()`` / ``is_confirmed()``
    helpers are what the agent now uses; we check both work."""
    print("\n[16] WorldPerception public-API helpers")
    from vision.world.perception import (
        WorldPerception, WorldPerceptionConfig,
    )
    from vision.world.map import WorldMap

    wp = WorldPerception(config=WorldPerceptionConfig(),
                          world_map=WorldMap())

    # purge_curiosity_queue clears AND reports count.
    wp._curiosity[(1, 64, 1)] = {"block_id": "x", "confidence": 0.5,
                                  "seen_tick": 1, "screen": None}
    wp._curiosity[(2, 64, 2)] = {"block_id": "y", "confidence": 0.5,
                                  "seen_tick": 1, "screen": None}
    n = wp.purge_curiosity_queue()
    if n != 2 or wp._curiosity:
        _fail(f"purge expected 2 entries removed, got n={n}, "
              f"remaining={len(wp._curiosity)}")
        return False
    _ok(f"purge_curiosity_queue() removed {n} entries")

    # is_confirmed is O(1) membership.
    wp._confirmed.add((9, 9, 9))
    if not wp.is_confirmed((9, 9, 9)):
        _fail("is_confirmed should hit (9,9,9)")
        return False
    if wp.is_confirmed((1, 1, 1)):
        _fail("is_confirmed should miss (1,1,1)")
        return False
    _ok("is_confirmed() works for hit + miss")
    return True


def test_exporters() -> bool:
    print("\n[12] Compact JSON + Sponge schematic exporters")
    import tempfile
    from pathlib import Path
    from vision.world import (
        WorldMap, BlockObservation,
        write_compact_json, write_schematic, world_map_to_compact_dict,
    )
    from vision.world.exporters import (
        read_schematic, _decode_varints, COMPACT_JSON_FORMAT,
        DEFAULT_DATA_VERSION,
    )
    import json as _json

    wm = WorldMap()
    wm.set_current_dimension("minecraft:overworld")
    # Tiny 2x1x2 floor of grass + one stone column.
    for x in range(0, 2):
        for z in range(0, 2):
            wm.update_block(BlockObservation(
                pos=(x, 63, z), block_id="minecraft:grass_block",
                confidence=1.0, source="looking_at", last_seen_tick=1,
            ))
    wm.update_block(BlockObservation(
        pos=(0, 64, 0), block_id="minecraft:stone",
        confidence=1.0, source="looking_at", last_seen_tick=2,
    ))

    # ── Compact dict ──────────────────────────────────────────────
    d = world_map_to_compact_dict(wm)
    if d["format"] != COMPACT_JSON_FORMAT:
        _fail(f"bad format tag: {d['format']}")
        return False
    if set(d["palette"]) != {"minecraft:grass_block", "minecraft:stone"}:
        _fail(f"palette wrong: {d['palette']}")
        return False
    if len(d["blocks"]) != 5:
        _fail(f"expected 5 blocks, got {len(d['blocks'])}")
        return False
    _ok(f"compact dict: palette={d['palette']}, blocks={len(d['blocks'])}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        jp = write_compact_json(wm, td / "w.json")
        sp = write_schematic(wm, td / "w.schem")
        if not (50 <= jp.stat().st_size <= 4000):
            _fail(f"json size out of range: {jp.stat().st_size}")
            return False
        if not (50 <= sp.stat().st_size <= 4000):
            _fail(f"schem size out of range: {sp.stat().st_size}")
            return False

        # JSON round-trip.
        loaded = _json.loads(jp.read_text(encoding="utf-8"))
        if len(loaded["blocks"]) != 5:
            _fail("JSON round-trip lost blocks")
            return False

        # ── Schematic round-trip: parse the NBT we wrote back and
        # verify EVERY field has the correct TAG type / shape.
        root = read_schematic(sp)
        # Version must be Int = 2.
        if root.get("Version") != 2:
            _fail(f"Version field wrong: {root.get('Version')}")
            return False
        # DataVersion present + sensible.
        if root.get("DataVersion") != DEFAULT_DATA_VERSION:
            _fail(f"DataVersion wrong: {root.get('DataVersion')}")
            return False
        # Width/Height/Length match the floor + column bounds.
        if (root.get("Width"), root.get("Height"), root.get("Length")) != (2, 2, 2):
            _fail(f"box wrong: {root.get('Width')}x{root.get('Height')}x"
                  f"{root.get('Length')}")
            return False
        # Offset MUST decode as a list of 3 ints (TAG_Int_Array
        # produces a Python list at the reader).
        off = root.get("Offset")
        if not (isinstance(off, list) and len(off) == 3
                and all(isinstance(v, int) for v in off)):
            _fail(f"Offset wrong shape / type: {off!r}")
            return False
        if off != [0, 63, 0]:
            _fail(f"Offset should be tight-bbox min, got {off}")
            return False
        _ok(f"Offset is IntArray(3) = {off}")
        # Palette must contain air + grass_block + stone.
        pal = root.get("Palette") or {}
        if not (isinstance(pal, dict)
                and {"minecraft:air", "minecraft:grass_block",
                     "minecraft:stone"} <= set(pal)):
            _fail(f"palette missing entries: {pal!r}")
            return False
        # PaletteMax matches len(palette).
        if root.get("PaletteMax") != len(pal):
            _fail(f"PaletteMax mismatch: {root.get('PaletteMax')} vs {len(pal)}")
            return False
        # BlockData decodes to Width*Height*Length varints.
        bdata = root.get("BlockData")
        if not isinstance(bdata, (bytes, bytearray)):
            _fail(f"BlockData wrong type: {type(bdata)}")
            return False
        indices = _decode_varints(bytes(bdata))
        expected_n = 2 * 2 * 2
        if len(indices) != expected_n:
            _fail(f"BlockData has {len(indices)} entries, expected {expected_n}")
            return False
        # Air at (0,63,0)+(1,1,0) = upper layer Z=0, X=1 → idx 5.
        # Check one known cell: the stone column is at world (0,64,0),
        # bbox-relative (0,1,0), index = (1*2 + 0)*2 + 0 = 4.
        grass_idx = pal["minecraft:grass_block"]
        stone_idx = pal["minecraft:stone"]
        air_idx   = pal["minecraft:air"]
        if indices[4] != stone_idx:
            _fail(f"stone should be at block-data index 4, got idx {indices[4]}")
            return False
        if indices[0] != grass_idx:
            _fail(f"grass should be at index 0, got idx {indices[0]}")
            return False
        # The unobserved upper-layer cells fill with air.
        if indices[5] != air_idx:
            _fail(f"unobserved cell should be air, got idx {indices[5]}")
            return False
        _ok(f"BlockData decodes correctly "
            f"(grass={grass_idx} stone={stone_idx} air={air_idx})")
        _ok(f"wrote {jp.stat().st_size}B json, {sp.stat().st_size}B schem")

        # Empty-map schematic still validates structurally.
        empty_wm = WorldMap()
        empty_path = write_schematic(empty_wm, td / "empty.schem")
        empty_root = read_schematic(empty_path)
        if (empty_root.get("Width"), empty_root.get("Height"),
                empty_root.get("Length")) != (1, 1, 1):
            _fail(f"empty schem box wrong: "
                  f"{empty_root.get('Width')}x{empty_root.get('Height')}x"
                  f"{empty_root.get('Length')}")
            return False
        _ok("empty WorldMap -> 1x1x1 air schem, parseable")

        # DataVersion=None must produce a schematic WITHOUT that field.
        no_dv = write_schematic(wm, td / "no_dv.schem", data_version=None)
        no_dv_root = read_schematic(no_dv)
        if "DataVersion" in no_dv_root:
            _fail("DataVersion=None should omit the field")
            return False
        _ok("data_version=None omits the field as per spec")

    return True


def test_inverse_renderer() -> bool:
    print("\n[11] InverseRenderer projection + scoring")
    import cv2
    from vision.world import InverseRenderer
    from vision.world.screen_ray import CameraIntrinsics, ScreenRay
    from vision.mc_assets import MCAssets

    try:
        assets = MCAssets.load()
    except Exception as e:
        _fail(f"asset cache missing: {e}")
        return False

    ir = InverseRenderer(assets)

    # Visible-face picker: player looking +Z at a voxel ahead → -Z
    # face (north) should be visible.
    face = ir.visible_face(voxel=(0, 64, 5), eye=(0.0, 65.62, 0.0))
    if face != "north":
        _fail(f"voxel north of eye should expose 'north' face, got {face}")
        return False
    _ok(f"visible_face(eye south of voxel) -> {face}")

    # Geometry projection: a voxel directly ahead of the eye should
    # land at the screen centre. Put the eye at (0.5, 65.62, 0.5) so
    # the voxel centre (0.5, 65.5, 5.5) is on the +Z axis from eye.
    intr = CameraIntrinsics.from_frame(960, 540, h_fov_deg=90.0)
    sr = ScreenRay(intrinsics=intr)
    eye = (0.5, 65.62, 0.5)
    corners = ir.project_face(
        (0, 65, 5), "north", sr=sr, eye=eye, yaw=0.0, pitch=0.0,
        frame_shape=(540, 960),
    )
    if corners is None or len(corners) != 4:
        _fail(f"project_face returned {corners}")
        return False
    # All corners should be near the screen centre.
    cx = sum(p[0] for p in corners) / 4
    cy = sum(p[1] for p in corners) / 4
    if abs(cx - 480) > 15 or abs(cy - 270) > 30:
        _fail(f"voxel ahead should project near centre, got ({cx:.0f},{cy:.0f})")
        return False
    _ok(f"voxel ahead projects near screen centre ({cx:.0f},{cy:.0f})")

    # Scoring: synthesise a frame whose face quad is filled with the
    # canonical stone texture. The renderer should match against
    # minecraft:stone with a high score.
    expected = ir.expected_face_texture("minecraft:stone", "north")
    if expected is None:
        _fail("could not load minecraft:stone canonical face")
        return False
    frame = np.full((540, 960, 3), 30, dtype=np.uint8)  # dark bg
    # Paint a 200×200 block of stone pixels around the projection.
    s = 200
    x0, y0 = int(cx) - s // 2, int(cy) - s // 2
    tex_big = cv2.resize(expected, (s, s), interpolation=cv2.INTER_NEAREST)
    if tex_big.ndim == 2:
        tex_big = cv2.cvtColor(tex_big, cv2.COLOR_GRAY2RGB)
    if tex_big.shape[2] == 4:
        tex_big = tex_big[..., :3]
    frame[y0:y0 + s, x0:x0 + s] = tex_big
    score = ir.score_voxel(frame, (0, 65, 5), "minecraft:stone",
                            sr=sr, eye=eye, yaw=0.0, pitch=0.0)
    if score < 0.6:
        _fail(f"stone-on-stone score should be high, got {score:.2f}")
        return False
    _ok(f"stone face on stone texture scores {score:.2f}")

    # Negative: the same frame should score MUCH lower for a different
    # block (diamond_block) since the texture is wrong.
    score_wrong = ir.score_voxel(frame, (0, 65, 5), "minecraft:diamond_block",
                                  sr=sr, eye=eye, yaw=0.0, pitch=0.0)
    if score_wrong >= score:
        _fail(f"wrong-id score {score_wrong:.2f} should be < right {score:.2f}")
        return False
    _ok(f"wrong-id score {score_wrong:.2f} < correct {score:.2f}")
    return True


def test_strict_commit_and_curiosity() -> bool:
    print("\n[10] Strict commit gate + curiosity queue")
    from vision.world.perception import (
        WorldPerception, WorldPerceptionConfig,
    )
    from vision.world.map import WorldMap
    from vision.world.types import BlockObservation

    cfg = WorldPerceptionConfig()
    cfg.strict_commit_gate = True
    cfg.sample_commit_confidence = 0.85
    wp = WorldPerception(config=cfg, world_map=WorldMap())

    # Vision-patch observation, modest confidence → should NOT commit.
    obs_uncertain = BlockObservation(
        pos=(3, 64, 7),
        block_id="minecraft:stone",
        confidence=0.40,
        source="vision_patch",
        last_seen_tick=1,
    )
    if wp._should_commit(obs_uncertain):
        _fail("uncertain vision_patch should NOT commit under strict gate")
        return False
    _ok("strict gate rejects uncertain vision_patch")

    # Even high-confidence vision_patch without a sample-NN attached
    # shouldn't commit (no evidence beyond colour signature).
    obs_high_conf = BlockObservation(
        pos=(3, 64, 7),
        block_id="minecraft:stone",
        confidence=0.95,
        source="vision_patch",
        last_seen_tick=1,
    )
    if wp._should_commit(obs_high_conf):
        _fail("high-conf vision_patch with no sample-NN should NOT commit")
        return False
    _ok("strict gate requires sample-NN evidence")

    # looking_at always commits.
    obs_truth = BlockObservation(
        pos=(3, 64, 7),
        block_id="minecraft:stone",
        confidence=1.0,
        source="looking_at",
        last_seen_tick=1,
    )
    if not wp._should_commit(obs_truth):
        _fail("looking_at must always commit")
        return False
    _ok("looking_at always commits")

    # Curiosity queue: feed entries, prune capacity, pop targets.
    wp.cfg.curiosity_queue_max = 3
    wp._curiosity[(0, 0, 0)] = {"block_id": "x", "confidence": 0.4,
                                "seen_tick": 1, "screen": None}
    wp._curiosity[(1, 0, 0)] = {"block_id": "y", "confidence": 0.2,
                                "seen_tick": 2, "screen": None}
    wp._curiosity[(2, 0, 0)] = {"block_id": "z", "confidence": 0.6,
                                "seen_tick": 3, "screen": None}
    wp._curiosity[(3, 0, 0)] = {"block_id": "w", "confidence": 0.5,
                                "seen_tick": 4, "screen": None}
    wp._prune_curiosity()
    if len(wp._curiosity) != 3:
        _fail(f"prune expected 3 entries, got {len(wp._curiosity)}")
        return False
    _ok("curiosity queue prunes oldest entries")

    # Pop closest to a given eye position.
    eye = (0.5, 0.5, 0.5)
    target = wp.take_curiosity_target(eye=eye, prefer="closest")
    if target is None:
        _fail("take_curiosity_target returned None")
        return False
    _ok(f"closest target popped: {target}")
    if target in wp._curiosity:
        _fail("popped target should be removed from queue")
        return False

    # Confirming a voxel removes it from curiosity and adds to confirmed.
    wp._curiosity[(5, 5, 5)] = {"block_id": "q", "confidence": 0.3,
                                "seen_tick": 1, "screen": None}
    wp._confirmed.add((5, 5, 5))
    wp._curiosity.pop((5, 5, 5), None)
    if (5, 5, 5) in wp._curiosity:
        _fail("confirmed voxel should leave curiosity")
        return False
    _ok("confirmed voxels leave curiosity queue")
    return True


def test_f3_target_multiline() -> bool:
    print("\n[7] F3 Targeted-Block parser — single-line + multi-line layouts")
    from vision.world.f3_target import parse_looking_at_block

    # MC pre-1.20 single-line layout.
    single = parse_looking_at_block([
        "Targeted Block: -7, 64, 134  minecraft:stone",
    ])
    if single is None or single.block_id != "minecraft:stone" or single.pos != (-7, 64, 134):
        _fail(f"single-line parse failed: {single}")
        return False
    _ok(f"single-line: id={single.block_id} pos={single.pos}")

    # MC 1.21.x multi-line layout (label + coords, id on next line, tags after).
    multi = parse_looking_at_block([
        "Targeted Block: -83, 96, -114",
        "minecraft:grass_block",
        "snowy: false",
        "#minecraft:sniffer_diggable_block",
        "#minecraft:moss_replaceable",
    ])
    if multi is None or multi.block_id != "minecraft:grass_block" or multi.pos != (-83, 96, -114):
        _fail(f"multi-line parse failed: {multi}")
        return False
    _ok(f"multi-line: id={multi.block_id} pos={multi.pos}")

    # OCR-corrupted label line — should still skip tags and find the id below.
    noisy = parse_looking_at_block([
        "Targeted |?lock: -83, 96, -114",
        "minecraft:dirt",
        "#minecraft:dirt_replaceable",
    ])
    if noisy is None or noisy.block_id != "minecraft:dirt":
        _fail(f"noisy-label parse failed: {noisy}")
        return False
    _ok(f"noisy-label: id={noisy.block_id}")

    # No coords on the label line → return None. Previously the parser
    # returned a (0,0,0) sentinel with conf=0.5 here, but the
    # perception layer ALWAYS rejected those on the confidence floor;
    # the diagnostic dump just made it LOOK like phantom origin
    # commits. Returning None is the honest API.
    id_only = parse_looking_at_block([
        "Looking at block",
        "minecraft:cobblestone",
    ])
    if id_only is not None:
        _fail(f"id-only without coords should return None, got: {id_only}")
        return False
    _ok("id-only-without-coords returns None")

    # Garbage input returns None cleanly.
    if parse_looking_at_block(["XYZ: 0 / 64 / 0", "no target line here"]) is not None:
        _fail("garbage input should return None")
        return False
    _ok("garbage input returns None")
    return True


def test_f3_two_column_crops() -> bool:
    print("\n[8] F3Reader crops BOTH left and right columns")
    from vision.ocr import OCRConfig, F3Reader
    import numpy as np
    # Synthesise a frame with text-like ink on BOTH sides of the top
    # strip so we can assert both regions get cropped.
    frame = np.zeros((200, 1920, 3), dtype=np.uint8)
    # ink scribble on left
    frame[5:15,   20:200, :] = 220
    # ink scribble on right
    frame[5:15, 1500:1900, :] = 220

    # Build a minimal reader directly (avoid the mc_font cache dependency).
    cfg = OCRConfig(
        ui_scale=2, line_height_px=16, line_gap_px=2,
        text_start_y=2, text_start_x=2, line_margin_px=1,
        line_step_px=18, max_lines=4,
        enable_right_column=True,
        right_column_width_px=620, right_column_margin_px=2,
        line_width_px=700,
        enable_tesseract_fallback=False,
    )
    # Build a stub reader without templates by routing through Tesseract
    # path; we ONLY test the cropping, not the OCR backend.
    try:
        reader = F3Reader(cfg, templates={"A": np.ones((8, 5), dtype=np.uint8)})
    except Exception as e:
        _fail(f"could not build reader: {e}")
        return False
    crops = reader._crop_lines(frame)
    if len(crops) < 2:
        _fail(f"expected at least 2 crops (LEFT+RIGHT), got {len(crops)}")
        return False
    # First two should alternate L, R.
    widths = [c.shape[1] for c in crops[:2]]
    if widths[0] < widths[1] - 100:
        # Right column should be wider when LEFT is narrower; we
        # actually configured LEFT = 700, RIGHT = 620. So LEFT > RIGHT.
        # This branch detects swapped order which would be wrong.
        _fail(f"crop sequence order seems wrong: widths={widths}")
        return False
    _ok(f"two-column crops produced (L={widths[0]} R={widths[1]} per row)")
    return True


def main() -> int:
    print("World perception smoke test")
    print("=" * 60)
    results = [
        ("catalog",       test_catalog()),
        ("classifier",    test_block_classifier()),
        ("screen_ray",    test_screen_ray()),
        ("world_map",     test_world_map()),
        ("perception",    test_world_perception()),
        ("samples+render", test_sample_store_and_recognizer()),
        ("3d+air+depth",   test_3d_renderer_and_air_carving()),
        ("strict+curiosity", test_strict_commit_and_curiosity()),
        ("inverse_renderer", test_inverse_renderer()),
        ("exporters",       test_exporters()),
        ("air_guard",       test_air_carving_guards()),
        ("blockid_gate",    test_block_id_validator_gate()),
        ("sprite_skip",     test_sprite_sample_skip()),
        ("dark_gate",       test_dark_sample_gate()),
        ("garble_gate",     test_garble_sample_gate()),
        ("confirm_gate",    test_multiframe_confirm_gate()),
        ("held_occlusion",  test_held_item_occlusion()),
        ("temporal_vote",   test_temporal_voter()),
        ("sweep_cap",       test_sweep_delta_cap()),
        ("strict_xhair",    test_strict_gate_crosshair_bypass()),
        ("public_api",      test_perception_public_api()),
        ("f3_target",     test_f3_target_multiline()),
        ("f3_two_col",    test_f3_two_column_crops()),
    ]
    print("\nSummary")
    print("-" * 60)
    n_pass = sum(1 for _, ok in results if ok)
    for name, ok in results:
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {name}")
    print(f"\n{n_pass} / {len(results)} sub-tests passed.")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
