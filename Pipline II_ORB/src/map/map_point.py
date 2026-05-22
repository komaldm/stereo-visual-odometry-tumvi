from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class MapPoint:
    id: int
    position: np.ndarray
    descriptor: np.ndarray
    observations: list = field(default_factory=list)
    created_in_keyframe: int = -1
    last_seen_frame: int = -1
    observation_count: int = 1
    consecutive_misses: int = 0
    is_bad: bool = False
    is_new: bool = True