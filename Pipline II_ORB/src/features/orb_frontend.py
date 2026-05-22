from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class FeatureSet:
    keypoints: list
    descriptors: np.ndarray


class ORBFrontend:
    def __init__(
        self,
        nfeatures: int = 2000,
        scaleFactor: float = 1.2,
        nlevels: int = 8,
        edgeThreshold: int = 31,
        fastThreshold: int = 20,
    ):
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=scaleFactor,
            nlevels=nlevels,
            edgeThreshold=edgeThreshold,
            fastThreshold=fastThreshold,
        )

    def detect_and_compute(self, image: np.ndarray) -> FeatureSet:
        keypoints, descriptors = self.orb.detectAndCompute(image, None)

        if descriptors is None:
            descriptors = np.empty((0, 32), dtype=np.uint8)

        return FeatureSet(
            keypoints=keypoints,
            descriptors=descriptors,
        )