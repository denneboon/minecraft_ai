# vision/world/entity_classifier.py
"""
Scaffold for mob / player / item-drop detection in the rendered view.

What this is today
------------------
A protocol + a null implementation. The protocol pins the API down so
the rest of the pipeline (WorldPerception, WorldMap, agents) can be
built against it now, even though no real detector exists yet.

What this becomes
-----------------
A small object-detection model (YOLOv8-n or similar, fine-tuned on
captures of vanilla mobs) wrapped in the same ``detect`` method. We
plan to gather training data automatically by hovering over entities
in-game (the same trick the inventory module's SampleStore uses for
items) — every detection gets logged with the F3 "Looking at entity"
line as the ground-truth label.

Until that lands, ``NullEntityClassifier`` reports nothing — agents
that need mob / player awareness can check for ``len(entities) == 0``
and fall back to safer behaviour.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from vision.world.types import EntityObservation, ItemDropObservation


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class EntityClassifierProtocol(Protocol):
    """
    Detector contract: given a frame (and an optional region of
    interest), return everything that looks like a mob, player,
    item drop, or projectile.

    The implementation is free to use any technique (template
    matching, classical CV, a neural network) and may be expensive —
    callers will rate-limit by deciding *how often* to invoke it,
    not how long it takes.
    """

    def detect(self,
               frame_rgb: np.ndarray,
               *,
               roi: Optional[Tuple[int, int, int, int]] = None,
               ) -> Tuple[List[EntityObservation], List[ItemDropObservation]]:
        ...


# ---------------------------------------------------------------------------
# Null implementation
# ---------------------------------------------------------------------------

class NullEntityClassifier:
    """
    The default detector: always returns no entities.

    Useful as a placeholder so the rest of the perception pipeline can
    run end-to-end without a trained model.
    """

    def detect(self,
               frame_rgb: np.ndarray,
               *,
               roi: Optional[Tuple[int, int, int, int]] = None,
               ) -> Tuple[List[EntityObservation], List[ItemDropObservation]]:
        return [], []


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_entity_classifier(settings: Optional[Dict[str, Any]] = None
                           ) -> EntityClassifierProtocol:
    """
    Build the entity classifier configured by ``settings.yaml``.

    Honoured keys:

      ``vision.world.entity_classifier``
          ``null``  (default) — see :class:`NullEntityClassifier`.
          ``yolo``            — placeholder for the future YOLO model.
    """
    cfg = ((settings or {}).get("vision", {})
                            .get("world", {})
                            .get("entity_classifier", "null"))
    if cfg == "yolo":
        print("[world] entity_classifier='yolo' requested but no YOLO "
              "model is bundled yet — falling back to NullEntityClassifier.")
    return NullEntityClassifier()


__all__ = [
    "EntityClassifierProtocol",
    "NullEntityClassifier",
    "build_entity_classifier",
]
