from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class StereoFeature:
    keypoint: cv2.KeyPoint
    descriptor: np.ndarray
    disparity: float
    point_3d: np.ndarray   # shape (3,), left camera frame


class StereoFeatureExtractor:
    def __init__(
        self,
        K_left: np.ndarray,
        baseline_m: float,
        orb_nfeatures: int = 2000,
        orb_fast_threshold: int = 20,
        orb_min_fast_threshold: int = 7,
        min_disparity: float = 1.0,
        max_depth_m: float = 15.0,
        min_depth_m: float = 0.3,
        disparity_patch_radius: int = 2,
        min_valid_patch_samples: int = 5,
        max_disparity_std: float = 1.5,
        grid_rows: int = 6,
        grid_cols: int = 8,
        max_features_per_cell: int = 40,
        min_valid_features_for_fallback: int = 230,
    ):
        self.K_left = K_left
        self.baseline_m = baseline_m
        self.orb_nfeatures = orb_nfeatures
        self.orb_fast_threshold = orb_fast_threshold
        self.orb_min_fast_threshold = orb_min_fast_threshold
        self.min_disparity = min_disparity
        self.max_depth_m = max_depth_m
        self.min_depth_m = min_depth_m
        self.disparity_patch_radius = disparity_patch_radius
        self.min_valid_patch_samples = min_valid_patch_samples
        self.max_disparity_std = max_disparity_std
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.max_features_per_cell = max_features_per_cell
        self.min_valid_features_for_fallback = min_valid_features_for_fallback

        self.fx = float(K_left[0, 0])
        self.fy = float(K_left[1, 1])
        self.cx = float(K_left[0, 2])
        self.cy = float(K_left[1, 2])

        self.orb = self._make_orb(self.orb_fast_threshold)
        self.orb_fallback = None
        if self.orb_min_fast_threshold < self.orb_fast_threshold:
            self.orb_fallback = self._make_orb(self.orb_min_fast_threshold)

    def _make_orb(self, fast_threshold: int) -> cv2.ORB:
        return cv2.ORB_create(
            nfeatures=self.orb_nfeatures,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            patchSize=31,
            fastThreshold=fast_threshold,
        )

    def _sample_stable_disparity(
        self,
        disparity: np.ndarray,
        u: float,
        v: float,
    ) -> float | None:
        h, w = disparity.shape
        x = int(round(u))
        y = int(round(v))
        r = self.disparity_patch_radius

        if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
            return None

        patch = disparity[y - r : y + r + 1, x - r : x + r + 1].astype(np.float32)
        valid = np.isfinite(patch) & (patch > self.min_disparity)
        if int(np.count_nonzero(valid)) < self.min_valid_patch_samples:
            return None

        vals = patch[valid]
        if float(np.std(vals)) > self.max_disparity_std:
            return None

        return float(np.median(vals))

    def _bucket_valid_features(
        self,
        keypoints: List[cv2.KeyPoint],
        descriptors: np.ndarray,
        points_3d: np.ndarray,
        image_shape: tuple[int, int],
    ) -> tuple[List[cv2.KeyPoint], np.ndarray, np.ndarray, np.ndarray]:
        h, w = image_shape
        cells: dict[tuple[int, int], list[int]] = {}

        for i, kp in enumerate(keypoints):
            x, y = kp.pt
            c = min(self.grid_cols - 1, max(0, int(self.grid_cols * x / max(w, 1))))
            r = min(self.grid_rows - 1, max(0, int(self.grid_rows * y / max(h, 1))))
            cells.setdefault((r, c), []).append(i)

        keep = []
        for idxs in cells.values():
            idxs.sort(key=lambda i: keypoints[i].response, reverse=True)
            keep.extend(idxs[: self.max_features_per_cell])

        keep.sort()
        return (
            [keypoints[i] for i in keep],
            descriptors[keep],
            points_3d[keep],
            np.asarray(keep, dtype=np.int32),
        )

    def _extract_valid_features(
        self,
        orb: cv2.ORB,
        left_gray: np.ndarray,
        disparity: np.ndarray,
    ) -> tuple[List[cv2.KeyPoint], np.ndarray | None, np.ndarray, np.ndarray]:
        keypoints, descriptors = orb.detectAndCompute(left_gray, None)

        if descriptors is None or len(keypoints) == 0:
            return [], None, np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32)

        valid_keypoints = []
        valid_descriptors = []
        valid_points_3d = []
        valid_indices = []

        for i, kp in enumerate(keypoints):
            u, v = kp.pt
            d = self._sample_stable_disparity(disparity, u, v)
            if d is None:
                continue

            z = (self.fx * self.baseline_m) / d
            if z <= self.min_depth_m or z > self.max_depth_m:
                continue

            X = (u - self.cx) * z / self.fx
            Y = (v - self.cy) * z / self.fy

            valid_keypoints.append(kp)
            valid_descriptors.append(descriptors[i])
            valid_points_3d.append([X, Y, z])
            valid_indices.append(i)

        if len(valid_keypoints) == 0:
            return [], None, np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32)

        valid_keypoints, valid_descriptors, valid_points_3d, keep = self._bucket_valid_features(
            valid_keypoints,
            np.asarray(valid_descriptors, dtype=np.uint8),
            np.asarray(valid_points_3d, dtype=np.float32),
            disparity.shape,
        )
        valid_indices = np.asarray(valid_indices, dtype=np.int32)[keep]

        return (
            valid_keypoints,
            valid_descriptors,
            valid_points_3d,
            valid_indices,
        )

    def extract(
        self,
        left_gray: np.ndarray,
        disparity: np.ndarray,
    ) -> tuple[List[cv2.KeyPoint], np.ndarray | None, np.ndarray, np.ndarray]:
        """
        Returns:
            keypoints
            descriptors
            points_3d: (N, 3)
            valid_indices: indices into keypoints/descriptors that survived disparity filtering
        """
        return self._extract_valid_features(self.orb, left_gray, disparity)

    def extract_dense(
        self,
        left_gray: np.ndarray,
        disparity: np.ndarray,
    ) -> tuple[List[cv2.KeyPoint], np.ndarray | None, np.ndarray, np.ndarray]:
        if self.orb_fallback is None:
            return self.extract(left_gray, disparity)
        return self._extract_valid_features(self.orb_fallback, left_gray, disparity)
