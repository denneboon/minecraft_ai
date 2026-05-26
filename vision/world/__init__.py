# vision/world/__init__.py
"""
World perception — the pixel-only AI's mental model of what surrounds
the player.

What lives here
---------------
* :mod:`vision.world.types`        — small dataclasses (observations,
                                     player pose, world frames).
* :mod:`vision.world.map`          — :class:`WorldMap`, the sparse
                                     3-D voxel + entity store.
* :mod:`vision.world.screen_ray`   — camera intrinsics, screen ↔ world
                                     projection, voxel DDA stepping.
* :mod:`vision.world.f3_target`    — parses the "Looking at block"
                                     line out of the F3 overlay.
* :mod:`vision.world.block_classifier`
                                   — turns a screen patch into a block
                                     id guess (pluggable strategy;
                                     ships with a colour-signature
                                     baseline, slot for a NN later).
* :mod:`vision.world.entity_classifier`
                                   — scaffold for mob / player / item
                                     drop detection (NN later).
* :mod:`vision.world.perception`   — :class:`WorldPerception`, the
                                     per-tick orchestrator.

What the agent sees
-------------------
At every tick the perception layer can update the agent with a
:class:`WorldFrame` — what's visible right now plus a reference to a
persistent :class:`WorldMap` of everything the AI has ever seen. The
agent does not need to know whether a block came from the "Looking at
block" F3 line, from patch classification, or from extrapolating a
previously-seen block: it just asks the WorldMap and trusts the
confidence values.

Future-proofing
---------------
Every module here defines a strategy/protocol so it can be swapped:

* Replace ``BlockClassifierProtocol`` with a CNN that wins on
  ``patch → block_id``.
* Replace ``EntityClassifierProtocol`` with YOLO / RT-DETR / a
  custom detector trained on captured mob crops.
* Replace ``WorldMap`` with an octree-backed store for very long
  exploration runs.

None of those changes require touching the agent or the
WorldPerception orchestrator.
"""

from __future__ import annotations

from vision.world.types import (
    BlockObservation,
    EntityObservation,
    ItemDropObservation,
    LookingAtBlock,
    PlayerPose,
    WorldFrame,
)
from vision.world.map import WorldMap
from vision.world.screen_ray import CameraIntrinsics, ScreenRay
from vision.world.block_classifier import (
    BlockClassifierProtocol,
    ColourSignatureBlockClassifier,
    build_block_classifier,
)
from vision.world.entity_classifier import (
    EntityClassifierProtocol,
    NullEntityClassifier,
    build_entity_classifier,
)
from vision.world.sample_store import (
    WorldSampleStore,
    WorldSampleStoreConfig,
    StoredWorldSample,
    build_world_sample_store,
    default_world_sample_root,
)
from vision.world.sample_recognizer import (
    SampleBlockRecognizer,
    SampleBlockRecognizerConfig,
    HybridBlockClassifier,
    build_sample_block_recognizer,
)
from vision.world.map_renderer import (
    WorldMapRenderer,
    MapRenderConfig,
)
from vision.world.renderer_3d import (
    IsoWorldRenderer,
    IsoRenderConfig,
)
from vision.world.inverse_renderer import (
    InverseRenderer,
    InverseRendererConfig,
)
from vision.world.exporters import (
    write_compact_json,
    write_schematic,
    world_map_to_compact_dict,
)
from vision.world.perception import WorldPerception, build_world_perception

__all__ = [
    # types
    "BlockObservation", "EntityObservation", "ItemDropObservation",
    "LookingAtBlock", "PlayerPose", "WorldFrame",
    # map
    "WorldMap",
    # geometry
    "CameraIntrinsics", "ScreenRay",
    # classifiers
    "BlockClassifierProtocol", "ColourSignatureBlockClassifier",
    "build_block_classifier",
    "EntityClassifierProtocol", "NullEntityClassifier",
    "build_entity_classifier",
    # sample-based learning
    "WorldSampleStore", "WorldSampleStoreConfig", "StoredWorldSample",
    "build_world_sample_store", "default_world_sample_root",
    "SampleBlockRecognizer", "SampleBlockRecognizerConfig",
    "HybridBlockClassifier", "build_sample_block_recognizer",
    # visualisation
    "WorldMapRenderer", "MapRenderConfig",
    "IsoWorldRenderer", "IsoRenderConfig",
    "InverseRenderer", "InverseRendererConfig",
    "write_compact_json", "write_schematic", "world_map_to_compact_dict",
    # perception
    "WorldPerception", "build_world_perception",
]
