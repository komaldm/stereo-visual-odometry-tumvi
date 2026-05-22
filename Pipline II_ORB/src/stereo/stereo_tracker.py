from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class StereoTrackingResult:
    success: bool
    method: str
    matches: List[cv2.DMatch]
    inlier_matches: List[cv2.DMatch]
    object_points: np.ndarray     # (M, 3)
    image_points: np.ndarray      # (M, 2)
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    R: np.ndarray | None
    t: np.ndarray | None


class StereoPnPTracker:
    def __init__(
        self,
        K_left: np.ndarray,
        reprojection_error_px: float = 2.0,
        confidence: float = 0.999,
        min_inliers: int = 25,
        ratio_thresh: float = 0.70,
        mutual_check: bool = True,
        max_tracking_depth_m: float = 12.0,
        grid_rows: int = 4,
        grid_cols: int = 4,
        max_matches_per_cell: int = 12,
        min_3d3d_inliers: int = 15,
        ransac_iterations_3d3d: int = 200,
        ransac_threshold_3d3d_m: float = 0.15,
    ):
        self.K_left = K_left
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        self.reprojection_error_px = reprojection_error_px
        self.confidence = confidence
        self.min_inliers = min_inliers
        self.ratio_thresh = ratio_thresh
        self.mutual_check = mutual_check
        self.max_tracking_depth_m = max_tracking_depth_m
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.max_matches_per_cell = max_matches_per_cell
        self.min_3d3d_inliers = min_3d3d_inliers
        self.ransac_iterations_3d3d = ransac_iterations_3d3d
        self.ransac_threshold_3d3d_m = ransac_threshold_3d3d_m

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def _ratio_filter(self, knn_matches: list[list[cv2.DMatch]]) -> list[cv2.DMatch]:
        good = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_thresh * n.distance:
                good.append(m)
        return good

    def _bucket_matches_by_image_cells(
        self,
        matches: list[cv2.DMatch],
        keypoints: List[cv2.KeyPoint],
    ) -> list[cv2.DMatch]:
        if not matches:
            return []

        xs = [kp.pt[0] for kp in keypoints]
        ys = [kp.pt[1] for kp in keypoints]
        if not xs or not ys:
            return matches

        w = max(xs) + 1.0
        h = max(ys) + 1.0
        cells: dict[tuple[int, int], list[cv2.DMatch]] = {}

        for m in matches:
            x, y = keypoints[m.trainIdx].pt
            c = min(self.grid_cols - 1, max(0, int(self.grid_cols * x / max(w, 1.0))))
            r = min(self.grid_rows - 1, max(0, int(self.grid_rows * y / max(h, 1.0))))
            cells.setdefault((r, c), []).append(m)

        selected = []
        for cell_matches in cells.values():
            cell_matches.sort(key=lambda mm: mm.distance)
            selected.extend(cell_matches[: self.max_matches_per_cell])

        selected.sort(key=lambda mm: (mm.distance, mm.queryIdx, mm.trainIdx))
        return selected

    @staticmethod
    def _estimate_rigid_transform_3d3d(
        src_points: np.ndarray,
        dst_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if len(src_points) < 3 or len(dst_points) < 3:
            return None

        src_centroid = np.mean(src_points, axis=0)
        dst_centroid = np.mean(dst_points, axis=0)

        src_centered = src_points - src_centroid
        dst_centered = dst_points - dst_centroid

        H = src_centered.T @ dst_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1.0
            R = Vt.T @ U.T

        t = dst_centroid - R @ src_centroid
        return R.astype(np.float64), t.astype(np.float64)

    def _estimate_rigid_transform_3d3d_ransac(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        n = len(src_points)
        if n < self.min_3d3d_inliers:
            return None

        rng = np.random.default_rng(0)
        best_inliers = None
        best_count = 0

        for _ in range(self.ransac_iterations_3d3d):
            sample_idx = rng.choice(n, size=3, replace=False)
            estimate = self._estimate_rigid_transform_3d3d(
                src_points[sample_idx],
                dst_points[sample_idx],
            )
            if estimate is None:
                continue
            R, t = estimate
            transformed = (R @ src_points.T).T + t.reshape(1, 3)
            residuals = np.linalg.norm(transformed - dst_points, axis=1)
            inliers = residuals < self.ransac_threshold_3d3d_m
            count = int(np.count_nonzero(inliers))

            if count > best_count:
                best_count = count
                best_inliers = inliers

        if best_inliers is None or best_count < self.min_3d3d_inliers:
            return None

        refined = self._estimate_rigid_transform_3d3d(
            src_points[best_inliers],
            dst_points[best_inliers],
        )
        if refined is None:
            return None

        R, t = refined
        return R, t, best_inliers

    def track(
        self,
        prev_keypoints: List[cv2.KeyPoint],
        prev_descriptors: np.ndarray | None,
        prev_points_3d: np.ndarray,
        curr_keypoints: List[cv2.KeyPoint],
        curr_descriptors: np.ndarray | None,
        curr_points_3d: np.ndarray | None = None,
        reprojection_error_override: float | None = None,
        mutual_check_override: bool | None = None,
        min_inliers_override: int | None = None,
    ) -> StereoTrackingResult:
        _reprojection_error = reprojection_error_override if reprojection_error_override is not None else self.reprojection_error_px
        _mutual_check = mutual_check_override if mutual_check_override is not None else self.mutual_check
        _min_inliers = min_inliers_override if min_inliers_override is not None else self.min_inliers

        if (
            prev_descriptors is None
            or curr_descriptors is None
            or len(prev_keypoints) == 0
            or len(curr_keypoints) == 0
            or len(prev_points_3d) == 0
        ):
            return StereoTrackingResult(
                success=False,
                method="none",
                matches=[],
                inlier_matches=[],
                object_points=np.empty((0, 3), dtype=np.float32),
                image_points=np.empty((0, 2), dtype=np.float32),
                rvec=None,
                tvec=None,
                R=None,
                t=None,
            )

        knn = self.matcher.knnMatch(prev_descriptors, curr_descriptors, k=2)
        good_matches = self._ratio_filter(knn)

        if _mutual_check and len(good_matches) > 0:
            reverse_knn = self.matcher.knnMatch(curr_descriptors, prev_descriptors, k=2)
            reverse_good = self._ratio_filter(reverse_knn)
            reverse_pairs = {(m.queryIdx, m.trainIdx) for m in reverse_good}
            good_matches = [
                m for m in good_matches
                if (m.trainIdx, m.queryIdx) in reverse_pairs
            ]

        best_for_train: dict[int, cv2.DMatch] = {}
        for m in good_matches:
            prev_best = best_for_train.get(m.trainIdx)
            if prev_best is None or m.distance < prev_best.distance:
                best_for_train[m.trainIdx] = m

        good_matches = list(best_for_train.values())
        good_matches = self._bucket_matches_by_image_cells(good_matches, curr_keypoints)

        if len(good_matches) < 6:
            return StereoTrackingResult(
                success=False,
                method="matching",
                matches=good_matches,
                inlier_matches=[],
                object_points=np.empty((0, 3), dtype=np.float32),
                image_points=np.empty((0, 2), dtype=np.float32),
                rvec=None,
                tvec=None,
                R=None,
                t=None,
            )

        object_points = []
        image_points = []
        curr_3d_points = []
        filtered_matches = []

        for m in good_matches:
            if m.queryIdx >= len(prev_points_3d):
                continue
            P = prev_points_3d[m.queryIdx]
            if not np.isfinite(P).all():
                continue
            if P[2] <= 0.0 or P[2] > self.max_tracking_depth_m:
                continue
            if curr_points_3d is not None:
                if m.trainIdx >= len(curr_points_3d):
                    continue
                P_curr = curr_points_3d[m.trainIdx]
                if not np.isfinite(P_curr).all():
                    continue
                if P_curr[2] <= 0.0 or P_curr[2] > self.max_tracking_depth_m:
                    continue
                curr_3d_points.append(P_curr)
            uv = curr_keypoints[m.trainIdx].pt
            object_points.append(P)
            image_points.append(uv)
            filtered_matches.append(m)

        if len(filtered_matches) < 6:
            return StereoTrackingResult(
                success=False,
                method="filtering",
                matches=good_matches,
                inlier_matches=[],
                object_points=np.empty((0, 3), dtype=np.float32),
                image_points=np.empty((0, 2), dtype=np.float32),
                rvec=None,
                tvec=None,
                R=None,
                t=None,
            )

        object_points = np.asarray(object_points, dtype=np.float32)
        image_points = np.asarray(image_points, dtype=np.float32)
        curr_3d_points = np.asarray(curr_3d_points, dtype=np.float32) if len(curr_3d_points) > 0 else np.empty((0, 3), dtype=np.float32)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=self.K_left,
            distCoeffs=self.dist_coeffs,
            reprojectionError=_reprojection_error,
            confidence=self.confidence,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success or inliers is None or len(inliers) < _min_inliers:
            if len(curr_3d_points) == len(object_points) and len(curr_3d_points) >= self.min_3d3d_inliers:
                rigid = self._estimate_rigid_transform_3d3d_ransac(object_points, curr_3d_points)
                if rigid is not None:
                    R_rigid, t_rigid, rigid_inliers = rigid
                    rigid_idx = np.where(rigid_inliers)[0].tolist()
                    rigid_inlier_matches = [filtered_matches[i] for i in rigid_idx]
                    rvec_rigid, _ = cv2.Rodrigues(R_rigid.astype(np.float64))
                    return StereoTrackingResult(
                        success=True,
                        method="3d3d",
                        matches=filtered_matches,
                        inlier_matches=rigid_inlier_matches,
                        object_points=object_points[rigid_inliers],
                        image_points=image_points[rigid_inliers],
                        rvec=rvec_rigid,
                        tvec=t_rigid.reshape(3, 1),
                        R=R_rigid.astype(np.float64),
                        t=t_rigid.astype(np.float64),
                    )

            return StereoTrackingResult(
                success=False,
                method="pnp",
                matches=filtered_matches,
                inlier_matches=[],
                object_points=object_points,
                image_points=image_points,
                rvec=None,
                tvec=None,
                R=None,
                t=None,
            )

        inlier_idx = inliers.ravel().tolist()
        inlier_matches = [filtered_matches[i] for i in inlier_idx]

        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points[inlier_idx],
                image_points[inlier_idx],
                self.K_left,
                self.dist_coeffs,
                rvec,
                tvec,
            )
        except Exception:
            pass

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)

        return StereoTrackingResult(
            success=True,
            method="pnp",
            matches=filtered_matches,
            inlier_matches=inlier_matches,
            object_points=object_points[inlier_idx],
            image_points=image_points[inlier_idx],
            rvec=rvec,
            tvec=tvec,
            R=R,
            t=t,
        )
