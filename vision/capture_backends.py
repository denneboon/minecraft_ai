from __future__ import annotations
from typing import Dict
import numpy as np

class ICaptureBackend:
    def grab(self, region: Dict[str, int]) -> np.ndarray:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError