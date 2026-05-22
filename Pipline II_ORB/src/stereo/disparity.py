from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class StereoSGBMConfig:
    min_disparity: int = 0
    num_disparities: int = 64   # must be divisible by 16
    block_size: int = 7
    p1: int | None = None
    p2: int | None = None
    disp12_max_diff: int = 1
    pre_filter_cap: int = 31
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY


class StereoDisparityEstimator:
    def __init__(self, config: StereoSGBMConfig | None = None):
        self.config = config if config is not None else StereoSGBMConfig()

    def compute(self, left_gray: np.ndarray, right_gray: np.ndarray) -> np.ndarray:
        if left_gray.ndim != 2 or right_gray.ndim != 2:
            raise ValueError("StereoDisparityEstimator expects grayscale images.")

        block_size = self.config.block_size
        p1 = self.config.p1 if self.config.p1 is not None else 8 * 1 * block_size * block_size
        p2 = self.config.p2 if self.config.p2 is not None else 32 * 1 * block_size * block_size

        matcher = cv2.StereoSGBM_create(
            minDisparity=self.config.min_disparity,
            numDisparities=self.config.num_disparities,
            blockSize=block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=self.config.disp12_max_diff,
            preFilterCap=self.config.pre_filter_cap,
            uniquenessRatio=self.config.uniqueness_ratio,
            speckleWindowSize=self.config.speckle_window_size,
            speckleRange=self.config.speckle_range,
            mode=self.config.mode,
        )

        disparity = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        return disparity


def normalize_disparity_for_vis(disparity: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    disp_vis = disparity.copy()

    if valid_mask is None:
        valid_mask = disp_vis > 0.0

    out = np.zeros_like(disp_vis, dtype=np.uint8)
    if np.count_nonzero(valid_mask) == 0:
        return out

    vmin = float(np.min(disp_vis[valid_mask]))
    vmax = float(np.max(disp_vis[valid_mask]))

    if vmax - vmin < 1e-6:
        out[valid_mask] = 255
        return out

    scaled = (disp_vis - vmin) / (vmax - vmin)
    out[valid_mask] = np.clip(255.0 * scaled[valid_mask], 0, 255).astype(np.uint8)
    return out