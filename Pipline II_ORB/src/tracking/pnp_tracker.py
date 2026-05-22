from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class PnPTrackingResult:
    success: bool
    R: np.ndarray | None
    t: np.ndarray | None
    inlier_count: int
    matched_map_point_ids: list[int]
    image_points: np.ndarray
    object_points: np.ndarray
    inlier_mask: np.ndarray | None
    num_matches_before_ransac: int
    inlier_ratio: float
    candidate_coverage: float = 0.0
    candidate_occupied_cells: int = 0
    candidate_max_cell_fraction: float = 0.0


class PnPTracker:
    def __init__(self, K: np.ndarray, ratio_thresh: float = 0.75):
        self.K = K
        self.ratio_thresh = ratio_thresh
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def _bucket_matches_by_image_cells(
        self,
        matches,
        keypoints,
        grid_rows: int = 4,
        grid_cols: int = 4,
        max_per_cell: int = 10,
    ):
        if not matches:
            return []

        xs = [kp.pt[0] for kp in keypoints]
        ys = [kp.pt[1] for kp in keypoints]
        if not xs or not ys:
            return matches

        w = max(xs) + 1.0
        h = max(ys) + 1.0

        cells = {}
        for m in matches:
            x, y = keypoints[m.trainIdx].pt
            c = min(grid_cols - 1, max(0, int(grid_cols * x / max(w, 1.0))))
            r = min(grid_rows - 1, max(0, int(grid_rows * y / max(h, 1.0))))
            cells.setdefault((r, c), []).append(m)

        selected = []
        for cell_matches in cells.values():
            cell_matches.sort(key=lambda m: m.distance)
            selected.extend(cell_matches[:max_per_cell])

        selected.sort(key=lambda m: m.distance)
        return selected

    def _image_distribution_stats(
        self,
        image_points,
        keypoints,
        image_shape=None,
        grid_rows: int = 4,
        grid_cols: int = 4,
    ):
        if image_points is None or len(image_points) == 0:
            return 0.0, 0, 0.0

        if image_shape is not None:
            h = float(image_shape[0])
            w = float(image_shape[1])
        else:
            xs = [kp.pt[0] for kp in keypoints]
            ys = [kp.pt[1] for kp in keypoints]
            if not xs or not ys:
                return 0.0, 0, 0.0
            w = max(xs) + 1.0
            h = max(ys) + 1.0

        counts = {}
        for pt in image_points:
            x, y = float(pt[0]), float(pt[1])
            c = min(grid_cols - 1, max(0, int(grid_cols * x / max(w, 1.0))))
            r = min(grid_rows - 1, max(0, int(grid_rows * y / max(h, 1.0))))
            counts[(r, c)] = counts.get((r, c), 0) + 1

        occupied_cells = len(counts)
        coverage = float(occupied_cells) / float(grid_rows * grid_cols)
        max_cell_fraction = float(max(counts.values())) / float(len(image_points))
        return coverage, occupied_cells, max_cell_fraction

    def track(
        self,
        keypoints,
        descriptors,
        map_positions,
        map_descriptors,
        map_ids,
        reprojection_error: float = 4.0,
        use_spatial_bucketing: bool = False,
        image_shape=None,
        min_coverage: float | None = None,
        min_occupied_cells: int | None = None,
        max_cell_fraction: float | None = None,
    ) -> PnPTrackingResult:
        if descriptors is None or len(descriptors) == 0 or len(map_descriptors) == 0:
            return PnPTrackingResult(
                success=False,
                R=None,
                t=None,
                inlier_count=0,
                matched_map_point_ids=[],
                image_points=np.empty((0, 2), dtype=np.float32),
                object_points=np.empty((0, 3), dtype=np.float32),
                inlier_mask=None,
                num_matches_before_ransac=0,
                inlier_ratio=0.0,
            )

        knn_matches = self.matcher.knnMatch(map_descriptors, descriptors, k=2)

        ratio_good = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_thresh * n.distance:
                ratio_good.append(m)

        best_for_train = {}
        for m in ratio_good:
            prev = best_for_train.get(m.trainIdx)
            if prev is None or m.distance < prev.distance:
                best_for_train[m.trainIdx] = m

        unique_matches = list(best_for_train.values())
        unique_matches.sort(key=lambda m: (m.queryIdx, m.trainIdx, m.distance))

        if use_spatial_bucketing:
            unique_matches = self._bucket_matches_by_image_cells(
                unique_matches,
                keypoints,
                grid_rows=4,
                grid_cols=4,
                max_per_cell=10,
            )

        matched_map_point_ids = []
        object_points = []
        image_points = []

        for m in unique_matches:
            pid = map_ids[m.queryIdx]
            matched_map_point_ids.append(pid)
            object_points.append(map_positions[m.queryIdx])
            image_points.append(keypoints[m.trainIdx].pt)

        num_matches_before_ransac = len(object_points)
        candidate_coverage, candidate_occupied_cells, candidate_max_cell_fraction = (
            self._image_distribution_stats(
                image_points,
                keypoints,
                image_shape=image_shape,
                grid_rows=4,
                grid_cols=4,
            )
        )

        if len(object_points) < 6:
            return PnPTrackingResult(
                success=False,
                R=None,
                t=None,
                inlier_count=0,
                matched_map_point_ids=[],
                image_points=np.empty((0, 2), dtype=np.float32),
                object_points=np.empty((0, 3), dtype=np.float32),
                inlier_mask=None,
                num_matches_before_ransac=num_matches_before_ransac,
                inlier_ratio=0.0,
                candidate_coverage=candidate_coverage,
                candidate_occupied_cells=candidate_occupied_cells,
                candidate_max_cell_fraction=candidate_max_cell_fraction,
            )

        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        if (
            (min_coverage is not None and candidate_coverage < min_coverage)
            or (min_occupied_cells is not None and candidate_occupied_cells < min_occupied_cells)
            or (max_cell_fraction is not None and candidate_max_cell_fraction > max_cell_fraction)
        ):
            return PnPTrackingResult(
                success=False,
                R=None,
                t=None,
                inlier_count=0,
                matched_map_point_ids=[],
                image_points=image_points,
                object_points=object_points,
                inlier_mask=None,
                num_matches_before_ransac=num_matches_before_ransac,
                inlier_ratio=0.0,
                candidate_coverage=candidate_coverage,
                candidate_occupied_cells=candidate_occupied_cells,
                candidate_max_cell_fraction=candidate_max_cell_fraction,
            )

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=self.K,
            distCoeffs=None,
            reprojectionError=reprojection_error,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success or inliers is None or len(inliers) < 6:
            return PnPTrackingResult(
                success=False,
                R=None,
                t=None,
                inlier_count=0,
                matched_map_point_ids=[],
                image_points=image_points,
                object_points=object_points,
                inlier_mask=None,
                num_matches_before_ransac=num_matches_before_ransac,
                inlier_ratio=0.0,
                candidate_coverage=candidate_coverage,
                candidate_occupied_cells=candidate_occupied_cells,
                candidate_max_cell_fraction=candidate_max_cell_fraction,
            )

        inlier_indices = inliers.flatten()
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points[inlier_indices],
                image_points[inlier_indices],
                self.K,
                None,
                rvec,
                tvec,
            )
        except Exception:
            pass

        # Post-refinement tight filter: remove inliers whose reprojection error
        # exceeds 2.0px after LM refinement — inspired by the optical-flow
        # pipeline's iterative 1.5px outlier rejection pass.
        proj_pts, _ = cv2.projectPoints(
            object_points[inlier_indices], rvec, tvec, self.K, None
        )
        reproj_errs = np.linalg.norm(
            proj_pts.reshape(-1, 2) - image_points[inlier_indices], axis=1
        )
        tight_keep = reproj_errs < 2.0
        if tight_keep.sum() >= 6:
            inlier_indices = inlier_indices[tight_keep]

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)

        inlier_mask = np.zeros(len(object_points), dtype=bool)
        inlier_mask[inlier_indices] = True

        inlier_ids = [pid for pid, keep in zip(matched_map_point_ids, inlier_mask) if keep]

        inlier_ratio = float(len(inlier_indices) / max(num_matches_before_ransac, 1))

        return PnPTrackingResult(
            success=True,
            R=R,
            t=t,
            inlier_count=int(len(inlier_indices)),
            matched_map_point_ids=inlier_ids,
            image_points=image_points[inlier_mask],
            object_points=object_points[inlier_mask],
            inlier_mask=inlier_mask,
            num_matches_before_ransac=num_matches_before_ransac,
            inlier_ratio=inlier_ratio,
            candidate_coverage=candidate_coverage,
            candidate_occupied_cells=candidate_occupied_cells,
            candidate_max_cell_fraction=candidate_max_cell_fraction,
        )
