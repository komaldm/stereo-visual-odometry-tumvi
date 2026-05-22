from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from src.stereo.disparity import StereoDisparityEstimator, StereoSGBMConfig
from src.stereo.reconstruction import StereoReconstructionResult, reconstruct_from_disparity


@dataclass
class StereoFrameResult:
    left_gray: np.ndarray
    right_gray: np.ndarray
    disparity: np.ndarray
    reconstruction: StereoReconstructionResult


class StereoFrameProcessor:
    def __init__(
        self,
        K_left: np.ndarray,
        baseline_m: float,
        sgbm_config: StereoSGBMConfig | None = None,
        max_depth_m: float = 20.0,
    ):
        self.K_left = K_left
        self.baseline_m = baseline_m
        self.max_depth_m = max_depth_m
        self.disparity_estimator = StereoDisparityEstimator(sgbm_config)

    def process(self, left_img: np.ndarray, right_img: np.ndarray) -> StereoFrameResult:
        left_gray = self._to_gray(left_img)
        right_gray = self._to_gray(right_img)

        disparity = self.disparity_estimator.compute(left_gray, right_gray)

        reconstruction = reconstruct_from_disparity(
            disparity=disparity,
            K_left=self.K_left,
            baseline_m=self.baseline_m,
            max_depth_m=self.max_depth_m,
        )

        return StereoFrameResult(
            left_gray=left_gray,
            right_gray=right_gray,
            disparity=disparity,
            reconstruction=reconstruction,
        )

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)