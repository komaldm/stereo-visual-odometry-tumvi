from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PinholeCamera:
    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    def image_size(self) -> tuple[int, int]:
        return (self.width, self.height)