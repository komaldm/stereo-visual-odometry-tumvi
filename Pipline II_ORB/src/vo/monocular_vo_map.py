from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import cv2
import numpy as np
import time

from src.features.orb_frontend import ORBFrontend
from src.features.matcher import ORBMatcher
from src.geometry.essential import estimate_essential_and_pose
from src.geometry.projection import make_projection_matrix
from src.geometry.transforms import make_transform, invert_transform
from src.geometry.triangulation import (
    triangulate_points,
    filter_triangulated_points,
    filter_triangulated_points_world,
)
from src.map.map_manager import MapManager
from src.tracking.motion_validator import MotionValidator
from src.tracking.pnp_tracker import PnPTracker
from src.vo.fallback_state import FallbackState


@dataclass
class MonocularVOMapResult:
    timestamps_ns: list[int]
    poses_wc: list[np.ndarray]
    keyframe_indices: list[int]


class MonocularVOMap:
    def __init__(
        self,
        K: np.ndarray,
        undistorter,
        frontend: ORBFrontend,
        matcher: ORBMatcher,
        min_pnp_inliers: int = 20,
        keyframe_translation_thresh: float = 0.25,
        # Dataset-tunable parameters
        min_essential_inliers: int = 40,
        max_fallback_frames_before_rebuild: int = 10,
        keyframe_rotation_thresh_deg: float = 5.0,
        min_relocalization_inliers: int = 14,
        min_lost_relocalization_inliers: int = 14,
        min_lost_relocalization_inlier_ratio: float = 0.10,
        local_map_window: int = 80,
        post_relocalization_grace_frames: int = 8,
        recovery_probation_length: int = 5,
        recovery_min_local_inliers: int = 18,
        min_strong_inliers_for_rebuild_anchor: int = 30,
        min_rebuild_baseline: float = 0.75,
        min_rebuild_match_count: int = 50,
        max_cv_fallback_frames: int = 8,
        max_translation_step: float = 2.0,
        max_rotation_step_deg: float = 25.0,
        # Initialization search range
        init_max_offset: int = 60,
    ):
        self.K = K
        self.undistorter = undistorter
        self.frontend = frontend
        self.matcher = matcher
        self.pnp_tracker = PnPTracker(K)
        self.map_manager = MapManager()
        self.fallback_state = FallbackState()
        self.motion_validator = MotionValidator(
            max_translation_step=max_translation_step,
            max_rotation_step_deg=max_rotation_step_deg,
        )

        self.min_pnp_inliers = min_pnp_inliers
        self.min_essential_inliers = min_essential_inliers
        self.max_fallback_frames_before_rebuild = max_fallback_frames_before_rebuild
        self.min_rebuild_points = 12
        self.keyframe_translation_thresh = keyframe_translation_thresh
        self.min_keyframe_gap = 4
        self.keyframe_rotation_thresh_deg = keyframe_rotation_thresh_deg
        self.min_relocalization_inliers = min_relocalization_inliers
        self.min_lost_relocalization_inliers = min_lost_relocalization_inliers
        self.min_lost_relocalization_inlier_ratio = min_lost_relocalization_inlier_ratio
        self.local_map_window = local_map_window
        self.post_relocalization_grace_frames = post_relocalization_grace_frames
        self.last_relocalization_frame = -100000
        self.recovery_mode = False
        self.recovery_frames_remaining = 0
        self.recovery_probation_length = recovery_probation_length
        self.recovery_min_local_inliers = recovery_min_local_inliers
        self.pending_relocalization = False
        self.prev_feat_for_fallback = None
        self.prev_pose_for_fallback = None
        self.prev_dataset_idx_for_fallback = None
        self.last_strong_feat = None
        self.last_strong_pose = None
        self.last_strong_dataset_idx = None
        self.min_strong_inliers_for_rebuild_anchor = min_strong_inliers_for_rebuild_anchor
        self.min_rebuild_baseline = min_rebuild_baseline
        self.min_anchor_frame_gap = 10
        self.min_rebuild_match_count = min_rebuild_match_count
        self.last_canvas = None
        self.recent_metric_steps: list[float] = []
        self.max_recent_metric_steps = 30
        self.default_fallback_step_m = 0.05
        self.keyframes: list[dict] = []
        self.stats = {
            "frames_processed": 0,
            "accepted_pose_count": 0,
            "tracking_lost_events": 0,
            "relocalization_attempts": 0,
            "relocalization_successes": 0,
            "first_tracking_lost_frame": None,
            "current_tracked_streak": 0,
            "longest_tracked_streak": 0,
            "cv_fallback_count": 0,
        }

        # Initialization search range
        self.init_max_offset: int = init_max_offset

        # Constant-velocity fallback state (inspired by optical-flow pipeline's dampened motion model)
        self._cv_delta_T: np.ndarray | None = None
        self._cv_consecutive: int = 0
        self._cv_damp: float = 0.88
        self._cv_max_consecutive: int = max_cv_fallback_frames

    def _rotation_step_deg(self, T_a: np.ndarray, T_b: np.ndarray) -> float:
        R_rel = T_a[:3, :3].T @ T_b[:3, :3]
        trace = np.trace(R_rel)
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    def _load_left_frame(self, frame, image_loader):
        img = image_loader(frame.left_image_path, grayscale=True)
        img_u = self.undistorter.undistort(img)
        feat = self.frontend.detect_and_compute(img_u)
        return img_u, feat

    def _pose_from_camera_extrinsics(self, R_c_w: np.ndarray, t_c_w: np.ndarray) -> np.ndarray:
        return invert_transform(make_transform(R_c_w, t_c_w))

    def _compose_pose_with_relative_camera_motion(
        self,
        T_prev_wc: np.ndarray,
        R_curr_prev: np.ndarray,
        t_curr_prev: np.ndarray,
    ) -> np.ndarray:
        T_curr_prev = make_transform(R_curr_prev, t_curr_prev)
        return T_prev_wc @ invert_transform(T_curr_prev)

    def _make_keyframe_record(
        self,
        frame_idx: int,
        T_wc: np.ndarray,
        feat,
        map_point_ids: list[int] | None = None,
    ) -> dict:
        descriptors = None
        if feat.descriptors is not None:
            descriptors = feat.descriptors.copy()

        return {
            "frame_idx": int(frame_idx),
            "pose": T_wc.copy(),
            "keypoints": list(feat.keypoints),
            "descriptors": descriptors,
            "map_point_ids": list(dict.fromkeys(map_point_ids or [])),
        }

    def _try_track_with_map_source(
        self,
        feat,
        idx: int,
        use_local_map: bool,
    ):
        if use_local_map:
            map_positions, map_descriptors, map_ids = self.map_manager.get_local_positions_and_descriptors(
                current_frame_idx=idx,
                recent_frame_window=self.local_map_window,
            )
            source_name = "local"
            reprojection_error = 4.0
            use_spatial_bucketing = False
        else:
            map_positions, map_descriptors, map_ids = self.map_manager.get_relocalization_positions_and_descriptors(
                max_points=400,
                min_observations=2,
                max_consecutive_misses=40,
            )
            source_name = "global"
            reprojection_error = 2.5
            use_spatial_bucketing = True

        track_result = self.pnp_tracker.track(
            keypoints=feat.keypoints,
            descriptors=feat.descriptors,
            map_positions=map_positions,
            map_descriptors=map_descriptors,
            map_ids=map_ids,
            reprojection_error=reprojection_error,
            use_spatial_bucketing=use_spatial_bucketing,
        )
        return track_result, source_name

    def _try_essential_fallback(
        self,
        prev_feat,
        curr_feat,
        prev_pose_wc: np.ndarray,
        translation_scale_m: float,
    ):
        match_result = self.matcher.match(
            prev_feat.keypoints,
            prev_feat.descriptors,
            curr_feat.keypoints,
            curr_feat.descriptors,
        )

        if len(match_result.good_matches) < 30:
            return None, 0

        try:
            essential_result = estimate_essential_and_pose(
                pts1=match_result.pts1,
                pts2=match_result.pts2,
                K=self.K,
                ransac_prob=0.999,
                ransac_threshold=1.0,
            )
        except Exception:
            return None, 0

        if essential_result.num_inliers < self.min_essential_inliers:
            return None, essential_result.num_inliers

        t_scaled = essential_result.t * float(translation_scale_m)
        T_candidate_wc = self._compose_pose_with_relative_camera_motion(
            prev_pose_wc,
            essential_result.R,
            t_scaled,
        )

        return T_candidate_wc, essential_result.num_inliers

    def _record_metric_step(self, T_prev_wc: np.ndarray, T_curr_wc: np.ndarray) -> None:
        step = float(np.linalg.norm(T_curr_wc[:3, 3] - T_prev_wc[:3, 3]))
        if np.isfinite(step) and 1e-4 < step < 2.0:
            self.recent_metric_steps.append(step)
            if len(self.recent_metric_steps) > self.max_recent_metric_steps:
                self.recent_metric_steps = self.recent_metric_steps[-self.max_recent_metric_steps:]

    def _get_fallback_translation_scale(self) -> float:
        if not self.recent_metric_steps:
            return self.default_fallback_step_m
        scale = float(np.median(self.recent_metric_steps[-10:]))
        return float(np.clip(scale, 0.02, 0.30))

    def _is_rebuild_successful(self, inserted_points: int) -> bool:
        return inserted_points >= self.min_rebuild_points

    def _has_sufficient_rebuild_baseline(
        self,
        T_a_wc: np.ndarray,
        T_b_wc: np.ndarray,
    ) -> bool:
        baseline = np.linalg.norm(T_a_wc[:3, 3] - T_b_wc[:3, 3])
        return baseline >= self.min_rebuild_baseline

    def _should_update_strong_anchor(
        self,
        idx: int,
        T_wc: np.ndarray,
        inlier_count: int,
    ) -> bool:
        if inlier_count < self.min_strong_inliers_for_rebuild_anchor:
            return False

        if self.last_strong_pose is None or self.last_strong_dataset_idx is None:
            return True

        frame_gap = idx - self.last_strong_dataset_idx
        baseline = np.linalg.norm(T_wc[:3, 3] - self.last_strong_pose[:3, 3])

        return frame_gap >= self.min_anchor_frame_gap and baseline >= self.min_rebuild_baseline

    def _count_feature_matches(self, feat_a, feat_b) -> int:
        match_result = self.matcher.match(
            feat_a.keypoints,
            feat_a.descriptors,
            feat_b.keypoints,
            feat_b.descriptors,
        )
        return len(match_result.good_matches)

    def _find_best_initialization_pair(
        self,
        dataset,
        image_loader,
        start_idx: int = 0,
        min_offset: int = 2,
        max_offset: int = 20,
        min_matches: int = 80,
        min_essential_inliers: int = 50,
        min_triangulated: int = 120,
        min_median_parallax_deg: float = 1.0,
    ):
        best = None

        if len(dataset) == 0:
            return None

        frame0 = dataset[start_idx]
        _, feat0 = self._load_left_frame(frame0, image_loader)

        if feat0.descriptors is None or len(feat0.keypoints) < 50:
            return None

        last_idx = min(len(dataset) - 1, start_idx + max_offset)
        P0 = make_projection_matrix(self.K, np.eye(3), np.zeros((3, 1)))

        for idx1 in range(start_idx + min_offset, last_idx + 1):
            frame1 = dataset[idx1]
            _, feat1 = self._load_left_frame(frame1, image_loader)

            if feat1.descriptors is None or len(feat1.keypoints) < 50:
                continue

            match_result = self.matcher.match(
                feat0.keypoints,
                feat0.descriptors,
                feat1.keypoints,
                feat1.descriptors,
            )

            num_matches = len(match_result.good_matches)
            if num_matches < min_matches:
                continue

            try:
                essential_result = estimate_essential_and_pose(
                    pts1=match_result.pts1,
                    pts2=match_result.pts2,
                    K=self.K,
                    ransac_prob=0.999,
                    ransac_threshold=1.0,
                )
            except Exception:
                continue

            num_inliers = essential_result.num_inliers
            if num_inliers < min_essential_inliers:
                continue

            P1 = make_projection_matrix(self.K, essential_result.R, essential_result.t)
            points_3d = triangulate_points(
                P0,
                P1,
                essential_result.pts1_inliers,
                essential_result.pts2_inliers,
            )

            tri_result = filter_triangulated_points(
                points_3d=points_3d,
                pts1=essential_result.pts1_inliers,
                pts2=essential_result.pts2_inliers,
                K=self.K,
                R2=essential_result.R,
                t2=essential_result.t.reshape(3),
                max_depth=80.0,
                max_reproj_error=3.0,
                min_parallax_deg=0.5,
            )

            num_tri = len(tri_result.points_3d)
            if num_tri < min_triangulated:
                continue

            median_parallax_deg = (
                float(np.median(tri_result.parallax_deg))
                if len(tri_result.parallax_deg) > 0
                else 0.0
            )
            if median_parallax_deg < min_median_parallax_deg:
                continue

            train_indices = self._filtered_match_indices(
                match_indices=match_result.train_indices,
                prefilter_mask=essential_result.inlier_mask,
                tri_valid_mask=tri_result.valid_mask,
            )
            num_map_desc = len(train_indices)
            if num_map_desc < 100:
                continue

            # recoverPose returns translation up to scale, so parallax is the useful geometric discriminator.
            score = num_tri + 5.0 * median_parallax_deg + 0.2 * num_inliers

            if best is None or score > best["score"]:
                best = {
                    "idx0": start_idx,
                    "idx1": idx1,
                    "score": score,
                    "num_matches": num_matches,
                    "num_inliers": num_inliers,
                    "num_tri": num_tri,
                    "num_map_desc": num_map_desc,
                    "median_parallax_deg": median_parallax_deg,
                }

        if best is not None:
            print(
                f"[INIT] selected pair ({best['idx0']}, {best['idx1']}): "
                f"matches={best['num_matches']}, "
                f"essential_inliers={best['num_inliers']}, "
                f"triangulated={best['num_tri']}, "
                f"map_desc={best['num_map_desc']}, "
                f"median_parallax_deg={best['median_parallax_deg']:.3f}, "
                f"score={best['score']:.2f}"
            )

        return best

    def _filtered_match_indices(
        self,
        match_indices: np.ndarray,
        tri_valid_mask: np.ndarray,
        prefilter_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if len(match_indices) == 0:
            return np.empty((0,), dtype=np.int32)

        filtered_indices = match_indices
        if prefilter_mask is not None:
            filtered_indices = filtered_indices[prefilter_mask]

        if len(filtered_indices) == 0:
            return np.empty((0,), dtype=np.int32)

        return filtered_indices[tri_valid_mask]

    def _descriptors_from_train_indices(
        self,
        feat_curr,
        train_indices: np.ndarray,
    ):
        if feat_curr.descriptors is None or len(train_indices) == 0:
            return []

        return [feat_curr.descriptors[int(train_idx)] for train_idx in train_indices]

    def _descriptors_from_match_train_indices(
        self,
        feat_curr,
        train_indices: np.ndarray,
    ):
        return self._descriptors_from_train_indices(feat_curr, train_indices)

    def _compute_reprojection_rmse(self, object_points, image_points, T_wc):
        if object_points is None or image_points is None:
            return float("inf")
        if len(object_points) == 0 or len(image_points) == 0:
            return float("inf")

        T_c_w = invert_transform(T_wc)
        R = T_c_w[:3, :3]
        t = T_c_w[:3, 3].reshape(3, 1)

        rvec, _ = cv2.Rodrigues(R)
        proj, _ = cv2.projectPoints(
            object_points.astype(np.float32),
            rvec,
            t.astype(np.float32),
            self.K,
            None,
        )
        proj = proj.reshape(-1, 2)
        err = proj - image_points.astype(np.float32)
        rmse = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
        return rmse

    def _image_coverage_score(self, image_points, image_shape, grid_rows=4, grid_cols=4):
        coverage, _, _ = self._image_distribution_stats(
            image_points,
            image_shape,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )
        return coverage

    def _image_distribution_stats(self, image_points, image_shape, grid_rows=4, grid_cols=4):
        if image_points is None or len(image_points) == 0:
            return 0.0, 0, 0.0

        h, w = image_shape[:2]
        counts = {}

        for pt in image_points:
            x, y = float(pt[0]), float(pt[1])
            c = min(grid_cols - 1, max(0, int(grid_cols * x / max(w, 1))))
            r = min(grid_rows - 1, max(0, int(grid_rows * y / max(h, 1))))
            counts[(r, c)] = counts.get((r, c), 0) + 1

        occupied_cells = len(counts)
        coverage = float(occupied_cells) / float(grid_rows * grid_cols)
        max_cell_fraction = float(max(counts.values())) / float(len(image_points))
        return coverage, occupied_cells, max_cell_fraction

    def _score_track_result(self, result):
        if result is None or not result.success:
            return (-1, -1, -1.0)
        return (
            int(result.inlier_count),
            int(result.num_matches_before_ransac),
            float(result.inlier_ratio),
        )

    def _score_keyframe_similarity(self, feat, keyframe, image_shape, min_matches: int = 20):
        if feat.descriptors is None or keyframe["descriptors"] is None:
            return None

        match_result = self.matcher.match(
            keyframe["keypoints"],
            keyframe["descriptors"],
            feat.keypoints,
            feat.descriptors,
        )

        num_matches = len(match_result.good_matches)
        if num_matches < min_matches:
            return None

        coverage, occupied_cells, max_cell_fraction = self._image_distribution_stats(
            match_result.pts2,
            image_shape,
            grid_rows=4,
            grid_cols=4,
        )
        if coverage < 0.20 or occupied_cells < 4:
            return None

        score = float(num_matches) + 30.0 * coverage
        return {
            "score": score,
            "num_matches": num_matches,
            "coverage": coverage,
            "occupied_cells": occupied_cells,
            "max_cell_fraction": max_cell_fraction,
            "match_result": match_result,
        }

    def _select_relocalization_keyframes(self, feat, image_shape, top_k: int = 5):
        candidates = []

        for keyframe in self.keyframes:
            result = self._score_keyframe_similarity(feat, keyframe, image_shape)
            if result is None:
                continue
            candidates.append((result["score"], keyframe, result))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[:top_k]

    def _best_keyframe_relocalization_candidate(self, feat, idx: int, image_shape, top_k: int = 5):
        keyframe_candidates = self._select_relocalization_keyframes(
            feat,
            image_shape,
            top_k=top_k,
        )

        best_candidate = None

        for _, keyframe, sim_info in keyframe_candidates:
            map_positions, map_descriptors, map_ids = self.map_manager.get_points_for_keyframe_hypothesis(
                keyframe_idx=keyframe["frame_idx"],
                frame_window=40,
                max_points=400,
                min_observations=1,
                max_consecutive_misses=80,
                anchor_point_ids=keyframe.get("map_point_ids"),
            )

            if len(map_ids) < 20:
                continue

            track_result = self.pnp_tracker.track(
                keypoints=feat.keypoints,
                descriptors=feat.descriptors,
                map_positions=map_positions,
                map_descriptors=map_descriptors,
                map_ids=map_ids,
                reprojection_error=2.5,
                use_spatial_bucketing=True,
                image_shape=image_shape,
                min_coverage=0.25,
                min_occupied_cells=4,
                max_cell_fraction=0.65,
            )

            if track_result is None or not track_result.success:
                if track_result is not None and track_result.num_matches_before_ransac > 0:
                    print(
                        f"[RELOCAL-KF] frame={idx} kf={keyframe['frame_idx']} "
                        f"sim_matches={sim_info['num_matches']} "
                        f"rejected_pre_pnp coverage={track_result.candidate_coverage:.3f} "
                        f"cells={track_result.candidate_occupied_cells} "
                        f"max_cell_frac={track_result.candidate_max_cell_fraction:.3f}"
                    )
                continue

            T_candidate = self._pose_from_camera_extrinsics(
                track_result.R,
                track_result.t,
            )
            rmse = self._compute_reprojection_rmse(
                track_result.object_points,
                track_result.image_points,
                T_candidate,
            )
            coverage, occupied_cells, max_cell_fraction = self._image_distribution_stats(
                track_result.image_points,
                image_shape,
                grid_rows=4,
                grid_cols=4,
            )
            if coverage < 0.20 or occupied_cells < 4 or max_cell_fraction > 0.70:
                print(
                    f"[RELOCAL-KF] frame={idx} kf={keyframe['frame_idx']} "
                    f"pnp_inliers={track_result.inlier_count} "
                    f"rejected_post_pnp coverage={coverage:.3f} "
                    f"cells={occupied_cells} "
                    f"max_cell_frac={max_cell_fraction:.3f}"
                )
                continue
            score = (
                5.0 * float(track_result.inlier_count)
                + 15.0 * float(track_result.inlier_ratio)
                + 40.0 * coverage
                - 1.5 * rmse
            )

            print(
                f"[RELOCAL-KF] frame={idx} kf={keyframe['frame_idx']} "
                f"sim_matches={sim_info['num_matches']} "
                f"pnp_inliers={track_result.inlier_count} "
                f"rmse={rmse:.2f} coverage={coverage:.3f} "
                f"cells={occupied_cells} max_cell_frac={max_cell_fraction:.3f}"
            )

            candidate = {
                "source": "global",
                "track_result": track_result,
                "T_candidate": T_candidate,
                "rmse": rmse,
                "coverage": coverage,
                "keyframe_idx": keyframe["frame_idx"],
                "reference_pose": keyframe["pose"],
                "score": score,
            }

            if best_candidate is None or score > best_candidate["score"]:
                best_candidate = candidate

        return best_candidate

    def _best_relocalization_candidate(self, feat, idx, image_shape):
        local_result, _ = self._try_track_with_map_source(
            feat=feat,
            idx=idx,
            use_local_map=True,
        )

        best_global = self._best_keyframe_relocalization_candidate(
            feat=feat,
            idx=idx,
            image_shape=image_shape,
            top_k=5,
        )
        global_result = best_global["track_result"] if best_global is not None else None

        candidates = []
        if local_result is not None and local_result.success:
            candidates.append(("local", local_result, None))
        if global_result is not None and global_result.success:
            candidates.append(("global", global_result, best_global))

        if not candidates:
            return None, None, None

        candidates.sort(key=lambda x: self._score_track_result(x[1]), reverse=True)
        return candidates[0]

    def _accept_lost_mode_relocalization(
        self,
        T_prev_wc: np.ndarray,
        T_candidate_wc: np.ndarray,
        track_result,
        image_shape,
    ) -> tuple[bool, float, float, float, float]:
        if track_result is None or not track_result.success:
            return False, float("inf"), float("inf"), float("inf"), 0.0

        is_valid, trans_step, rot_step_deg = self.motion_validator.is_pose_valid(
            T_prev_wc,
            T_candidate_wc,
        )

        reproj_rmse = self._compute_reprojection_rmse(
            track_result.object_points,
            track_result.image_points,
            T_candidate_wc,
        )
        coverage = self._image_coverage_score(
            track_result.image_points,
            image_shape,
            grid_rows=4,
            grid_cols=4,
        )

        inliers = int(track_result.inlier_count)
        ratio = float(track_result.inlier_ratio)
        matches = int(track_result.num_matches_before_ransac)

        if (
            inliers < self.min_lost_relocalization_inliers
            or ratio < self.min_lost_relocalization_inlier_ratio
        ):
            return False, trans_step, rot_step_deg, reproj_rmse, coverage

        if is_valid and reproj_rmse < 3.5 and coverage >= 0.20:
            return True, trans_step, rot_step_deg, reproj_rmse, coverage

        very_strong = (
            inliers >= 22
            and ratio >= 0.18
            and matches >= 45
            and reproj_rmse < 3.0
            and coverage >= 0.25
        )
        strong = (
            inliers >= 16
            and ratio >= 0.15
            and matches >= 35
            and reproj_rmse < 2.5
            and coverage >= 0.20
        )

        moderate_motion = trans_step < 6.0 and rot_step_deg < 45.0
        mild_motion = trans_step < 4.0 and rot_step_deg < 35.0

        if very_strong and moderate_motion:
            return True, trans_step, rot_step_deg, reproj_rmse, coverage

        if strong and mild_motion:
            return True, trans_step, rot_step_deg, reproj_rmse, coverage

        return False, trans_step, rot_step_deg, reproj_rmse, coverage

    def _accept_absolute_relocalization_reset(
        self,
        T_candidate_wc: np.ndarray,
        track_result,
        image_shape,
        reference_pose_wc: np.ndarray | None = None,
    ) -> tuple[bool, float, float, int, float, float, float]:
        if track_result is None or not track_result.success:
            return False, float("inf"), 0.0, 0, 0.0, float("inf"), float("inf")

        reproj_rmse = self._compute_reprojection_rmse(
            track_result.object_points,
            track_result.image_points,
            T_candidate_wc,
        )
        coverage, occupied_cells, max_cell_fraction = self._image_distribution_stats(
            track_result.image_points,
            image_shape,
            grid_rows=4,
            grid_cols=4,
        )

        inliers = int(track_result.inlier_count)
        ratio = float(track_result.inlier_ratio)
        matches = int(track_result.num_matches_before_ransac)

        strong_geometry = (
            inliers >= 45
            and ratio >= 0.45
            and matches >= 60
            and reproj_rmse < 3.0
        )
        enough_spread = occupied_cells >= 2 and max_cell_fraction <= 0.80

        if not strong_geometry or not enough_spread:
            return False, reproj_rmse, coverage, occupied_cells, max_cell_fraction, float("inf"), float("inf")

        ref_trans = float("inf")
        ref_rot = float("inf")
        if reference_pose_wc is not None:
            _, ref_trans, ref_rot = self.motion_validator.is_pose_valid(
                reference_pose_wc,
                T_candidate_wc,
            )
            if ref_trans > 5.0 or ref_rot > 120.0:
                return False, reproj_rmse, coverage, occupied_cells, max_cell_fraction, ref_trans, ref_rot

        return True, reproj_rmse, coverage, occupied_cells, max_cell_fraction, ref_trans, ref_rot

    def _accept_relocalization_pose(
        self,
        T_prev_wc: np.ndarray,
        T_candidate_wc: np.ndarray,
        inlier_count: int,
        inlier_ratio: float,
        lost_mode: bool,
    ) -> tuple[bool, float, float]:
        is_valid, trans_step, rot_step_deg = self.motion_validator.is_pose_valid(
            T_prev_wc,
            T_candidate_wc,
        )

        if is_valid:
            return True, trans_step, rot_step_deg

        if lost_mode:
            strong = inlier_count >= 40 and inlier_ratio >= 0.25
            moderate_motion = trans_step < 4.0 and rot_step_deg < 35.0
            if strong and moderate_motion:
                return True, trans_step, rot_step_deg
        else:
            strong = inlier_count >= 50 and inlier_ratio >= 0.30
            moderate_motion = trans_step < 3.0 and rot_step_deg < 30.0
            if strong and moderate_motion:
                return True, trans_step, rot_step_deg

        return False, trans_step, rot_step_deg

    def _accept_tracking_candidate(
        self,
        T_prev_wc: np.ndarray,
        T_candidate_wc: np.ndarray,
        track_result,
        image_shape,
    ) -> tuple[bool, float, float, float]:
        is_valid, trans_step, rot_step_deg = self.motion_validator.is_pose_valid(
            T_prev_wc,
            T_candidate_wc,
        )
        reproj_rmse = self._compute_reprojection_rmse(
            track_result.object_points,
            track_result.image_points,
            T_candidate_wc,
        )

        if is_valid:
            return True, trans_step, rot_step_deg, reproj_rmse

        strong = (
            track_result.inlier_count >= 60
            and track_result.inlier_ratio >= 0.35
            and reproj_rmse < 4.0
        )
        moderate_motion = trans_step < 4.0 and rot_step_deg < 35.0
        if strong and moderate_motion:
            return True, trans_step, rot_step_deg, reproj_rmse

        return False, trans_step, rot_step_deg, reproj_rmse

    def _choose_better_track_result(self, local_result, global_result):
        candidates = []

        if local_result is not None and local_result.success:
            candidates.append(("local", local_result))
        if global_result is not None and global_result.success:
            candidates.append(("global", global_result))

        if not candidates:
            return None, None

        candidates.sort(
            key=lambda x: (
                x[1].inlier_count,
                x[1].num_matches_before_ransac,
                x[1].inlier_ratio,
            ),
            reverse=True,
        )

        return candidates[0]

    def _mark_tracking_lost_event(self, idx: int) -> None:
        self.stats["tracking_lost_events"] += 1
        if self.stats["first_tracking_lost_frame"] is None:
            self.stats["first_tracking_lost_frame"] = idx
        self.stats["current_tracked_streak"] = 0

    def _mark_pose_accepted(self) -> None:
        self.stats["accepted_pose_count"] += 1
        self.stats["current_tracked_streak"] += 1
        if self.stats["current_tracked_streak"] > self.stats["longest_tracked_streak"]:
            self.stats["longest_tracked_streak"] = self.stats["current_tracked_streak"]

    def _build_run_summary(
        self,
        final_frame_idx: int,
        final_keyframe_count: int,
        runtime_seconds: float | None = None,
    ) -> dict:
        frames_processed = self.stats["frames_processed"]
        summary = {
            "frames_processed": frames_processed,
            "final_frame_idx": final_frame_idx,
            "accepted_pose_count": self.stats["accepted_pose_count"],
            "tracking_lost_events": self.stats["tracking_lost_events"],
            "relocalization_attempts": self.stats["relocalization_attempts"],
            "relocalization_successes": self.stats["relocalization_successes"],
            "cv_fallback_count": self.stats["cv_fallback_count"],
            "first_tracking_lost_frame": self.stats["first_tracking_lost_frame"],
            "longest_tracked_streak": self.stats["longest_tracked_streak"],
            "final_map_point_count": len(self.map_manager.get_all_valid_points()),
            "final_keyframe_count": final_keyframe_count,
        }

        if runtime_seconds is not None:
            summary["runtime_seconds"] = float(runtime_seconds)
            avg_ms = (runtime_seconds / max(frames_processed, 1)) * 1000.0
            summary["avg_ms_per_frame"] = float(avg_ms)
            summary["runtime_fps_frames_processed"] = float(
                frames_processed / max(runtime_seconds, 1e-9)
            )
            summary["runtime_fps_accepted_poses"] = float(
                self.stats["accepted_pose_count"] / max(runtime_seconds, 1e-9)
            )

        return summary

    def _rebuild_local_map_from_pair(
        self,
        feat_prev,
        feat_curr,
        T_prev_wc: np.ndarray,
        T_curr_wc: np.ndarray,
        current_dataset_idx: int,
    ) -> int:
        match_result = self.matcher.match(
            feat_prev.keypoints,
            feat_prev.descriptors,
            feat_curr.keypoints,
            feat_curr.descriptors,
        )

        if len(match_result.good_matches) < 40:
            return 0

        T_c1_w = invert_transform(T_prev_wc)
        T_c2_w = invert_transform(T_curr_wc)

        P1 = self.K @ T_c1_w[:3, :]
        P2 = self.K @ T_c2_w[:3, :]

        pts_w = triangulate_points(
            P1,
            P2,
            match_result.pts1,
            match_result.pts2,
        )

        C1_w = T_prev_wc[:3, 3]
        C2_w = T_curr_wc[:3, 3]

        tri_result = filter_triangulated_points_world(
            points_w=pts_w,
            pts1=match_result.pts1,
            pts2=match_result.pts2,
            P1=P1,
            P2=P2,
            T_c1_w=T_c1_w,
            T_c2_w=T_c2_w,
            C1_w=C1_w,
            C2_w=C2_w,
            max_depth=60.0,
            max_reproj_error=2.0,
            min_parallax_deg=0.5,
        )

        train_indices = self._filtered_match_indices(
            match_indices=match_result.train_indices,
            tri_valid_mask=tri_result.valid_mask,
        )
        descs_for_points = self._descriptors_from_match_train_indices(feat_curr, train_indices)

        inserted = 0
        for p3d, desc in zip(tri_result.points_3d, descs_for_points):
            if not self.map_manager.has_nearby_point(p3d, distance_thresh=0.05):
                self.map_manager.add_map_point(
                    position=p3d,
                    descriptor=desc,
                    created_in_keyframe=current_dataset_idx,
                    last_seen_frame=current_dataset_idx,
                )
                inserted += 1

        return inserted

    def _is_extreme_motion(self, trans_step: float, rot_step_deg: float) -> bool:
        return trans_step > 5.0 or rot_step_deg > 45.0

    @staticmethod
    def _enforce_so3(R: np.ndarray) -> np.ndarray:
        U, _, Vt = np.linalg.svd(R)
        R_clean = U @ Vt
        if np.linalg.det(R_clean) < 0:
            U[:, 2] *= -1
            R_clean = U @ Vt
        return R_clean

    def _update_cv_delta(self, T_prev_wc: np.ndarray, T_curr_wc: np.ndarray) -> None:
        """Store incremental pose change as a constant-velocity anchor.
        T_prev_wc and T_curr_wc are camera-to-world transforms."""
        self._cv_delta_T = np.linalg.inv(T_prev_wc) @ T_curr_wc
        self._cv_consecutive = 0

    def _get_cv_pose(self, T_prev_wc: np.ndarray) -> np.ndarray | None:
        """Predict current pose using dampened constant velocity.
        Returns a camera-to-world transform, or None if no velocity available."""
        if self._cv_delta_T is None:
            return None
        damp = self._cv_damp ** max(0, self._cv_consecutive)
        T_delta = self._cv_delta_T.copy()
        T_delta[:3, 3] = T_delta[:3, 3] * damp
        T_delta[:3, :3] = self._enforce_so3(T_delta[:3, :3])
        T_pred = T_prev_wc @ T_delta
        T_pred[:3, :3] = self._enforce_so3(T_pred[:3, :3])
        return T_pred

    def _initialize_from_two_frames(self, dataset, image_loader, idx0=0, idx1=4):
        frame0 = dataset[idx0]
        frame1 = dataset[idx1]

        img0, feat0 = self._load_left_frame(frame0, image_loader)
        img1, feat1 = self._load_left_frame(frame1, image_loader)

        match_result = self.matcher.match(
            feat0.keypoints, feat0.descriptors,
            feat1.keypoints, feat1.descriptors
        )

        if len(match_result.good_matches) < 8:
            print(f"[INIT] rejected pair ({idx0}, {idx1}): only {len(match_result.good_matches)} matches")
            return False

        try:
            essential_result = estimate_essential_and_pose(
                pts1=match_result.pts1,
                pts2=match_result.pts2,
                K=self.K,
                ransac_prob=0.999,
                ransac_threshold=1.0,
            )
        except Exception as exc:
            print(f"[INIT] rejected pair ({idx0}, {idx1}): essential estimation failed: {exc}")
            return False

        P0 = make_projection_matrix(self.K, np.eye(3), np.zeros((3, 1)))
        P1 = make_projection_matrix(self.K, essential_result.R, essential_result.t)

        points_3d = triangulate_points(
            P0,
            P1,
            essential_result.pts1_inliers,
            essential_result.pts2_inliers,
        )

        T_c1_w = np.eye(4, dtype=np.float64)
        T_c2_w = make_transform(essential_result.R, essential_result.t)
        C1_w = np.zeros(3, dtype=np.float64)
        C2_w = (-essential_result.R.T @ essential_result.t.reshape(3)).reshape(3)

        tri_result = filter_triangulated_points_world(
            points_w=points_3d,
            pts1=essential_result.pts1_inliers,
            pts2=essential_result.pts2_inliers,
            P1=P0,
            P2=P1,
            T_c1_w=T_c1_w,
            T_c2_w=T_c2_w,
            C1_w=C1_w,
            C2_w=C2_w,
            max_depth=100.0,
            max_reproj_error=3.0,
            min_parallax_deg=0.5,
        )

        if len(tri_result.points_3d) < 120:
            print(f"[INIT] rejected: only {len(tri_result.points_3d)} valid triangulated points")
            return False

        train_indices = self._filtered_match_indices(
            match_indices=match_result.train_indices,
            prefilter_mask=essential_result.inlier_mask,
            tri_valid_mask=tri_result.valid_mask,
        )
        inlier_descs = self._descriptors_from_train_indices(feat1, train_indices)

        if len(inlier_descs) < 100:
            print(f"[INIT] rejected: only {len(inlier_descs)} descriptors for map points")
            return False

        for p3d, desc in zip(tri_result.points_3d, inlier_descs):
            if not self.map_manager.has_nearby_point(p3d, distance_thresh=0.05):
                self.map_manager.add_map_point(
                    position=p3d,
                    descriptor=desc,
                    created_in_keyframe=idx1,
                    last_seen_frame=idx1,
                )

        T_wc0 = np.eye(4, dtype=np.float64)
        T_wc1 = self._pose_from_camera_extrinsics(
            essential_result.R,
            essential_result.t,
        )

        return {
            "frame0_idx": idx0,
            "frame1_idx": idx1,
            "img0": img0,
            "img1": img1,
            "feat0": feat0,
            "feat1": feat1,
            "T_wc0": T_wc0,
            "T_wc1": T_wc1,
        }

    def run(
        self,
        dataset,
        image_loader,
        visualizer=None,
        gt_positions=None,
        end_idx=None,
        debug_image_path=None,
        summary_path=None,
        log_path=None,
    ):
        run_start_time = time.perf_counter()
        if end_idx is None:
            end_idx = len(dataset)
        self.last_canvas = None
        self.recent_metric_steps = []
        self.last_relocalization_frame = -100000
        self.recovery_mode = False
        self.recovery_frames_remaining = 0
        self.pending_relocalization = False
        self.keyframes = []
        self._cv_delta_T = None
        self._cv_consecutive = 0
        self.stats = {
            "frames_processed": 0,
            "accepted_pose_count": 0,
            "tracking_lost_events": 0,
            "relocalization_attempts": 0,
            "relocalization_successes": 0,
            "first_tracking_lost_frame": None,
            "current_tracked_streak": 0,
            "longest_tracked_streak": 0,
            "cv_fallback_count": 0,
        }
        frame_log: list[dict] = []

        def append_frame_log(
            frame_idx: int,
            timestamp_ns: int,
            accepted: bool,
            source: str,
            pose_count: int,
            keyframe_count: int,
            map_point_count: int,
            pnp_inliers: int | None = None,
            reprojection_rmse_px: float | None = None,
            tracking_lost: bool | None = None,
        ) -> None:
            frame_log.append(
                {
                    "frame_idx": int(frame_idx),
                    "timestamp_s": float(timestamp_ns) * 1e-9,
                    "accepted": int(bool(accepted)),
                    "source": source,
                    "pose_count": int(pose_count),
                    "keyframe_count": int(keyframe_count),
                    "map_point_count": int(map_point_count),
                    "pnp_inliers": "" if pnp_inliers is None else int(pnp_inliers),
                    "reprojection_rmse_px": "" if reprojection_rmse_px is None else float(reprojection_rmse_px),
                    "tracking_lost": int(bool(self.fallback_state.tracking_lost if tracking_lost is None else tracking_lost)),
                }
            )

        best_init = self._find_best_initialization_pair(
            dataset,
            image_loader,
            start_idx=0,
            min_offset=2,
            max_offset=20,
            min_matches=80,
            min_essential_inliers=50,
            min_triangulated=120,
            min_median_parallax_deg=1.0,
        )
        if best_init is None:
            print(f"[INIT] no valid pair in offsets 2..20, widening search to {self.init_max_offset}")
            best_init = self._find_best_initialization_pair(
                dataset,
                image_loader,
                start_idx=0,
                min_offset=2,
                max_offset=self.init_max_offset,
                min_matches=80,
                min_essential_inliers=50,
                min_triangulated=120,
                min_median_parallax_deg=1.0,
            )
        if best_init is None:
            raise RuntimeError("Failed to find a valid initialization pair")

        init = self._initialize_from_two_frames(
            dataset,
            image_loader,
            idx0=best_init["idx0"],
            idx1=best_init["idx1"],
        )
        if not init:
            raise RuntimeError("Initialization failed after selecting a candidate pair")

        init_idx0 = init["frame0_idx"]
        init_idx1 = init["frame1_idx"]

        timestamps_ns = [
            dataset[init_idx0].timestamp_ns,
            dataset[init_idx1].timestamp_ns,
        ]
        poses_wc = [
            init["T_wc0"],
            init["T_wc1"],
        ]
        keyframe_indices = [0, 1]

        last_keyframe_feat = init["feat1"]
        last_keyframe_img = init["img1"]
        last_keyframe_pose = init["T_wc1"]
        last_keyframe_dataset_idx = init_idx1
        self.prev_feat_for_fallback = init["feat1"]
        self.prev_pose_for_fallback = init["T_wc1"]
        self.prev_dataset_idx_for_fallback = init_idx1
        self.last_strong_feat = init["feat1"]
        self.last_strong_pose = init["T_wc1"]
        self.last_strong_dataset_idx = init_idx1
        _, _, init_map_ids = self.map_manager.get_positions_and_descriptors()
        self.keyframes.append(
            self._make_keyframe_record(
                frame_idx=init_idx0,
                T_wc=init["T_wc0"],
                feat=init["feat0"],
                map_point_ids=[],
            )
        )
        append_frame_log(
            frame_idx=init_idx0,
            timestamp_ns=dataset[init_idx0].timestamp_ns,
            accepted=True,
            source="init",
            pose_count=len(poses_wc),
            keyframe_count=len(keyframe_indices),
            map_point_count=len(self.map_manager.get_all_valid_points()),
            tracking_lost=False,
        )
        append_frame_log(
            frame_idx=init_idx1,
            timestamp_ns=dataset[init_idx1].timestamp_ns,
            accepted=True,
            source="init",
            pose_count=len(poses_wc),
            keyframe_count=len(keyframe_indices),
            map_point_count=len(self.map_manager.get_all_valid_points()),
            tracking_lost=False,
        )
        self.keyframes.append(
            self._make_keyframe_record(
                frame_idx=init_idx1,
                T_wc=init["T_wc1"],
                feat=init["feat1"],
                map_point_ids=init_map_ids,
            )
        )

        final_frame_idx = init_idx1
        for idx in range(init_idx1 + 1, end_idx):
            final_frame_idx = idx
            self.stats["frames_processed"] += 1
            frame = dataset[idx]
            img_u, feat = self._load_left_frame(frame, image_loader)

            if self.fallback_state.tracking_lost:
                self.stats["relocalization_attempts"] += 1
                accepted = False
                relocalized = False
                best_source, relocal_track_result, relocal_meta = self._best_relocalization_candidate(
                    feat,
                    idx,
                    img_u.shape,
                )

                if relocal_track_result is not None:
                    T_relocal = self._pose_from_camera_extrinsics(
                        relocal_track_result.R,
                        relocal_track_result.t,
                    )
                    reference_pose_wc = None
                    if relocal_meta is not None:
                        reference_pose_wc = relocal_meta.get("reference_pose")
                    if reference_pose_wc is None:
                        reference_pose_wc = self.last_strong_pose

                    accept_relocal, trans_step_relocal, rot_step_relocal, reproj_rmse, coverage = (
                        self._accept_lost_mode_relocalization(
                            T_prev_wc=poses_wc[-1],
                            T_candidate_wc=T_relocal,
                            track_result=relocal_track_result,
                            image_shape=img_u.shape,
                        )
                    )
                    accepted_as_absolute_reset = False
                    abs_reproj = reproj_rmse
                    abs_coverage = coverage
                    abs_cells = 0
                    abs_max_cell_frac = 0.0
                    ref_trans = float("inf")
                    ref_rot = float("inf")

                    if not accept_relocal:
                        (
                            accept_relocal,
                            abs_reproj,
                            abs_coverage,
                            abs_cells,
                            abs_max_cell_frac,
                            ref_trans,
                            ref_rot,
                        ) = self._accept_absolute_relocalization_reset(
                            T_candidate_wc=T_relocal,
                            track_result=relocal_track_result,
                            image_shape=img_u.shape,
                            reference_pose_wc=reference_pose_wc,
                        )
                        accepted_as_absolute_reset = accept_relocal

                    if accept_relocal:
                        T_wc = T_relocal
                        track_result = relocal_track_result
                        source_name = (
                            "lost-local-relocal" if best_source == "local" else "lost-global-relocal"
                        )
                        accepted = True
                        relocalized = True
                        self.fallback_state.tracking_lost = False
                        self.fallback_state.consecutive_pnp_failures = 0
                        self.fallback_state.using_vo_fallback = False
                        self.last_relocalization_frame = idx
                        self.recovery_mode = True
                        self.recovery_frames_remaining = self.recovery_probation_length
                        self.pending_relocalization = True

                        kf_info = ""
                        if relocal_meta is not None and best_source == "global":
                            kf_info = f", kf={relocal_meta['keyframe_idx']}"
                        reset_info = ""
                        if accepted_as_absolute_reset:
                            reset_info = ", absolute_reset=1"
                        print(
                            f"[Frame {idx}] lost-mode relocalization accepted -> "
                            f"source={best_source}{kf_info}, "
                            f"inliers={relocal_track_result.inlier_count}, "
                            f"inlier_ratio={relocal_track_result.inlier_ratio:.3f}, "
                            f"matches={relocal_track_result.num_matches_before_ransac}, "
                            f"reproj_rmse={abs_reproj:.3f}, "
                            f"coverage={abs_coverage:.3f}"
                            f"{reset_info}"
                        )
                    else:
                        abs_info = ""
                        if np.isfinite(ref_trans) or np.isfinite(ref_rot):
                            abs_info = (
                                f", ref_trans={ref_trans:.3f}, "
                                f"ref_rot_deg={ref_rot:.2f}, "
                                f"abs_reproj_rmse={abs_reproj:.3f}, "
                                f"abs_coverage={abs_coverage:.3f}, "
                                f"abs_cells={abs_cells}, "
                                f"abs_max_cell_frac={abs_max_cell_frac:.3f}"
                            )
                        print(
                            f"[Frame {idx}] lost-mode relocalization rejected by motion gate: "
                            f"trans_step={trans_step_relocal:.3f}, "
                            f"rot_step_deg={rot_step_relocal:.2f}, "
                            f"reproj_rmse={reproj_rmse:.3f}, "
                            f"coverage={coverage:.3f}"
                            f"{abs_info}"
                        )

                if not accepted:
                    fallback_scale = self._get_fallback_translation_scale()
                    T_fallback, essential_inliers = self._try_essential_fallback(
                        prev_feat=self.prev_feat_for_fallback,
                        curr_feat=feat,
                        prev_pose_wc=self.prev_pose_for_fallback,
                        translation_scale_m=fallback_scale,
                    )

                    if T_fallback is not None:
                        is_valid_fb_motion, trans_step_fb, rot_step_fb = self.motion_validator.is_pose_valid(
                            poses_wc[-1], T_fallback
                        )

                        if (is_valid_fb_motion or essential_inliers >= 80) and not self._is_extreme_motion(
                            trans_step_fb,
                            rot_step_fb,
                        ):
                            poses_wc.append(T_fallback)
                            timestamps_ns.append(frame.timestamp_ns)
                            self._mark_pose_accepted()

                            self.prev_feat_for_fallback = feat
                            self.prev_pose_for_fallback = T_fallback
                            self.prev_dataset_idx_for_fallback = idx

                            self.fallback_state.using_vo_fallback = True
                            self.fallback_state.consecutive_pnp_failures += 1
                            self._cv_consecutive = 0  # refresh CV budget on geometric recovery

                            print(
                                f"[Frame {idx}] tracking_lost -> essential-fallback accepted, "
                                f"essential_inliers={essential_inliers}, "
                                f"scale={fallback_scale:.3f}, "
                                f"trans_step={trans_step_fb:.3f}, "
                                f"rot_step_deg={rot_step_fb:.2f}"
                            )

                            # Trigger a map rebuild if we've accumulated enough consecutive
                            # fallback frames — the rebuild check at the end of the normal
                            # tracking path is bypassed by `continue`, so we do it inline here.
                            if (
                                not self.fallback_state.pending_rebuild
                                and self.fallback_state.consecutive_pnp_failures
                                >= self.max_fallback_frames_before_rebuild
                            ):
                                self.fallback_state.pending_rebuild = True
                                # Reset the rebuild anchor to the current fallback pose so
                                # the baseline accumulates from HERE.  Without this, the
                                # strong anchor may be hundreds of frames old, leaving
                                # almost no feature overlap when the rebuild fires.
                                self.last_strong_feat = feat
                                self.last_strong_pose = T_fallback
                                self.last_strong_dataset_idx = idx
                                print(
                                    f"[Frame {idx}] fallback limit reached in tracking_lost "
                                    f"(consecutive={self.fallback_state.consecutive_pnp_failures})"
                                    f" -> pending rebuild, resetting anchor"
                                )

                            if (
                                self.fallback_state.pending_rebuild
                                and self.last_strong_feat is not None
                                and self.last_strong_pose is not None
                                and self._has_sufficient_rebuild_baseline(
                                    self.last_strong_pose, T_fallback
                                )
                            ):
                                match_count = self._count_feature_matches(
                                    self.last_strong_feat, feat
                                )
                                if match_count >= self.min_rebuild_match_count:
                                    inserted = self._rebuild_local_map_from_pair(
                                        feat_prev=self.last_strong_feat,
                                        feat_curr=feat,
                                        T_prev_wc=self.last_strong_pose,
                                        T_curr_wc=T_fallback,
                                        current_dataset_idx=idx,
                                    )
                                    print(
                                        f"[Frame {idx}] fallback-triggered rebuild "
                                        f"(anchor_idx={self.last_strong_dataset_idx}, "
                                        f"matches={match_count}) inserted={inserted}"
                                    )
                                    if self._is_rebuild_successful(inserted):
                                        self.fallback_state.pending_rebuild = False
                                        self.fallback_state.tracking_lost = False
                                        self.fallback_state.consecutive_pnp_failures = 0
                                        print(
                                            f"[Frame {idx}] rebuild in tracking_lost succeeded "
                                            f"-> exiting tracking_lost"
                                        )
                                    else:
                                        # Rebuild failed — reset anchor to here so baseline
                                        # accumulates from this point for the next attempt.
                                        self.last_strong_feat = feat
                                        self.last_strong_pose = T_fallback
                                        self.last_strong_dataset_idx = idx

                            append_frame_log(
                                frame_idx=idx,
                                timestamp_ns=frame.timestamp_ns,
                                accepted=True,
                                source="tracking-lost-essential-fallback",
                                pose_count=len(poses_wc),
                                keyframe_count=len(keyframe_indices),
                                map_point_count=len(self.map_manager.get_all_valid_points()),
                                pnp_inliers=essential_inliers,
                            )
                            continue

                    # Constant-velocity fallback when both relocalization and essential-fallback fail
                    # Inspired by the optical-flow pipeline's dampened motion model
                    if self._cv_consecutive < self._cv_max_consecutive:
                        T_cv = self._get_cv_pose(poses_wc[-1])
                        if T_cv is not None:
                            poses_wc.append(T_cv)
                            timestamps_ns.append(frame.timestamp_ns)
                            self._mark_pose_accepted()
                            self._cv_consecutive += 1
                            self.stats["cv_fallback_count"] += 1
                            self.prev_feat_for_fallback = feat
                            self.prev_pose_for_fallback = T_cv
                            self.prev_dataset_idx_for_fallback = idx
                            print(
                                f"[Frame {idx}] const-vel fallback (lost mode, "
                                f"consecutive={self._cv_consecutive}, "
                                f"damp={self._cv_damp ** self._cv_consecutive:.3f})"
                            )
                            append_frame_log(
                                frame_idx=idx,
                                timestamp_ns=frame.timestamp_ns,
                                accepted=True,
                                source="tracking-lost-const-vel",
                                pose_count=len(poses_wc),
                                keyframe_count=len(keyframe_indices),
                                map_point_count=len(self.map_manager.get_all_valid_points()),
                            )
                            continue

                    if relocal_track_result is not None and relocal_track_result.success:
                        print(
                            f"[Frame {idx}] tracking_lost, relocalization rejected: "
                            f"inliers={relocal_track_result.inlier_count}, "
                            f"inlier_ratio={relocal_track_result.inlier_ratio:.3f}, "
                            f"matches={relocal_track_result.num_matches_before_ransac}"
                        )
                    else:
                        print(f"[Frame {idx}] tracking_lost, waiting for relocalization")
                    continue
            else:
                accepted = False
                relocalized = False
                source_name = "unknown"

                if self.recovery_mode:
                    local_track_result, _ = self._try_track_with_map_source(
                        feat=feat,
                        idx=idx,
                        use_local_map=True,
                    )

                    if (
                        local_track_result is not None
                        and local_track_result.success
                        and local_track_result.inlier_count >= self.recovery_min_local_inliers
                    ):
                        T_candidate = self._pose_from_camera_extrinsics(
                            local_track_result.R,
                            local_track_result.t,
                        )
                        accept_recovery, trans_step, rot_step_deg, reproj_rmse = self._accept_tracking_candidate(
                            T_prev_wc=poses_wc[-1],
                            T_candidate_wc=T_candidate,
                            track_result=local_track_result,
                            image_shape=img_u.shape,
                        )

                        if accept_recovery:
                            T_wc = T_candidate
                            track_result = local_track_result
                            accepted = True
                            source_name = "recovery-local"
                            self.recovery_frames_remaining -= 1

                            if self.recovery_frames_remaining <= 0:
                                self.recovery_mode = False
                                if self.pending_relocalization:
                                    self.stats["relocalization_successes"] += 1
                                    self.pending_relocalization = False
                                    print(f"[Frame {idx}] relocalization probation passed")
                        else:
                            print(
                                f"[Frame {idx}] recovery probation failed: "
                                f"trans_step={trans_step:.3f}, "
                                f"rot_step_deg={rot_step_deg:.2f}, "
                                f"reproj_rmse={reproj_rmse:.3f}"
                            )
                            self.recovery_mode = False
                            self.pending_relocalization = False
                            self.fallback_state.tracking_lost = True
                            self.fallback_state.consecutive_pnp_failures = max(
                                self.fallback_state.consecutive_pnp_failures,
                                2,
                            )
                            self._mark_tracking_lost_event(idx)
                            continue
                    else:
                        inliers = (
                            local_track_result.inlier_count
                            if local_track_result is not None and local_track_result.success
                            else 0
                        )
                        print(
                            f"[Frame {idx}] recovery probation failed: "
                            f"local_inliers={inliers}"
                        )
                        self.recovery_mode = False
                        self.pending_relocalization = False
                        self.fallback_state.tracking_lost = True
                        self.fallback_state.consecutive_pnp_failures = max(
                            self.fallback_state.consecutive_pnp_failures,
                            2,
                        )
                        self._mark_tracking_lost_event(idx)
                        continue

                if accepted:
                    pass
                else:
                    local_track_result, _ = self._try_track_with_map_source(
                        feat=feat,
                        idx=idx,
                        use_local_map=True,
                    )
                    global_track_result, _ = self._try_track_with_map_source(
                        feat=feat,
                        idx=idx,
                        use_local_map=False,
                    )

                    best_source, best_result = self._choose_better_track_result(
                        local_track_result,
                        global_track_result,
                    )
                    track_result = best_result if best_result is not None else local_track_result
                    best_trans = None
                    best_rot = None
                    alt_source = None
                    alt_result = None
                    alt_trans = None
                    alt_rot = None

                    if best_result is not None and best_result.inlier_count >= self.min_pnp_inliers:
                        T_candidate = self._pose_from_camera_extrinsics(
                            best_result.R,
                            best_result.t,
                        )
                        accept_best, best_trans, best_rot, best_reproj = self._accept_tracking_candidate(
                            T_prev_wc=poses_wc[-1],
                            T_candidate_wc=T_candidate,
                            track_result=best_result,
                            image_shape=img_u.shape,
                        )

                        if accept_best:
                            T_wc = T_candidate
                            track_result = best_result
                            source_name = "local" if best_source == "local" else "global-relocal"
                            accepted = True
                            relocalized = best_source == "global"
                        else:
                            if best_source == "local" and global_track_result.success:
                                alt_source, alt_result = "global", global_track_result
                            elif best_source == "global" and local_track_result.success:
                                alt_source, alt_result = "local", local_track_result

                            if alt_result is not None and alt_result.inlier_count >= self.min_pnp_inliers:
                                T_alt = self._pose_from_camera_extrinsics(
                                    alt_result.R,
                                    alt_result.t,
                                )
                                alt_valid, alt_trans, alt_rot, alt_reproj = self._accept_tracking_candidate(
                                    T_prev_wc=poses_wc[-1],
                                    T_candidate_wc=T_alt,
                                    track_result=alt_result,
                                    image_shape=img_u.shape,
                                )

                                if alt_valid:
                                    T_wc = T_alt
                                    track_result = alt_result
                                    source_name = "local" if alt_source == "local" else "global-relocal"
                                    accepted = True
                                    relocalized = alt_source == "global"

                    if not accepted:
                        T_fallback, essential_inliers = self._try_essential_fallback(
                            prev_feat=self.prev_feat_for_fallback,
                            curr_feat=feat,
                            prev_pose_wc=self.prev_pose_for_fallback,
                            translation_scale_m=self._get_fallback_translation_scale(),
                        )

                        if T_fallback is not None and self.fallback_state.consecutive_pnp_failures < self.max_fallback_frames_before_rebuild:
                            T_wc = T_fallback
                            accepted = True
                            source_name = "essential-fallback"
                            relocalized = False
                            self.fallback_state.consecutive_pnp_failures += 1
                            self.fallback_state.using_vo_fallback = True
                            print(
                                f"[Frame {idx}] source=essential-fallback, "
                                f"essential_inliers={essential_inliers}, "
                                f"consecutive_fallback={self.fallback_state.consecutive_pnp_failures}"
                            )
                        else:
                            local_inliers = local_track_result.inlier_count if local_track_result is not None else 0
                            global_inliers = global_track_result.inlier_count if global_track_result is not None else 0
                            if best_trans is not None:
                                if alt_trans is not None:
                                    print(
                                        f"[Frame {idx}] rejected pose: "
                                        f"{best_source} trans_step={best_trans:.3f}, rot_step_deg={best_rot:.2f}, "
                                        f"reproj_rmse={best_reproj:.2f}; "
                                        f"{alt_source} trans_step={alt_trans:.3f}, rot_step_deg={alt_rot:.2f}, "
                                        f"reproj_rmse={alt_reproj:.2f}"
                                    )
                                else:
                                    print(
                                        f"[Frame {idx}] rejected pose: "
                                        f"{best_source} trans_step={best_trans:.3f}, rot_step_deg={best_rot:.2f}, "
                                        f"reproj_rmse={best_reproj:.2f}; "
                                        f"local_inliers={local_inliers}, global_inliers={global_inliers}"
                                    )
                            else:
                                print(
                                    f"[Frame {idx}] PnP failed / too few inliers "
                                    f"(local={local_inliers}, global={global_inliers})"
                                )
                            if self._cv_consecutive < self._cv_max_consecutive:
                                T_cv = self._get_cv_pose(poses_wc[-1])
                                if T_cv is not None:
                                    poses_wc.append(T_cv)
                                    timestamps_ns.append(frame.timestamp_ns)
                                    self._mark_pose_accepted()
                                    self._cv_consecutive += 1
                                    self.stats["cv_fallback_count"] += 1
                                    self.prev_feat_for_fallback = feat
                                    self.prev_pose_for_fallback = T_cv
                                    self.prev_dataset_idx_for_fallback = idx
                                    print(
                                        f"[Frame {idx}] const-vel fallback (normal mode, "
                                        f"consecutive={self._cv_consecutive}, "
                                        f"damp={self._cv_damp ** self._cv_consecutive:.3f})"
                                    )
                                    append_frame_log(
                                        frame_idx=idx,
                                        timestamp_ns=frame.timestamp_ns,
                                        accepted=True,
                                        source="const-vel",
                                        pose_count=len(poses_wc),
                                        keyframe_count=len(keyframe_indices),
                                        map_point_count=len(self.map_manager.get_all_valid_points()),
                                    )
                                    continue
                            self.fallback_state.tracking_lost = True
                            self._mark_tracking_lost_event(idx)
                            continue

            if not accepted:
                continue

            if source_name in ["local", "global-relocal", "lost-local-relocal", "lost-global-relocal", "recovery-local"]:
                self.fallback_state.consecutive_pnp_failures = 0
                self.fallback_state.using_vo_fallback = False

            is_valid_motion, trans_step, rot_step_deg = self.motion_validator.is_pose_valid(
                poses_wc[-1], T_wc
            )

            if source_name == "essential-fallback":
                if self._is_extreme_motion(trans_step, rot_step_deg):
                    print(
                        f"[Frame {idx}] essential fallback produced extreme motion: "
                        f"trans_step={trans_step:.3f}, rot_step_deg={rot_step_deg:.2f}; "
                        f"entering tracking_lost"
                    )
                    self.fallback_state.tracking_lost = True
                    self._mark_tracking_lost_event(idx)
                    continue

            poses_wc.append(T_wc)
            self._mark_pose_accepted()
            timestamps_ns.append(frame.timestamp_ns)
            self.prev_feat_for_fallback = feat
            self.prev_pose_for_fallback = T_wc
            self.prev_dataset_idx_for_fallback = idx
            if len(poses_wc) >= 2 and source_name in ["local", "global-relocal", "lost-local-relocal", "lost-global-relocal", "recovery-local"]:
                self._record_metric_step(poses_wc[-2], poses_wc[-1])

            if source_name in ["local", "global-relocal", "lost-local-relocal", "lost-global-relocal"] and len(poses_wc) >= 2:
                self._update_cv_delta(poses_wc[-2], poses_wc[-1])
            elif source_name == "essential-fallback":
                # Essential-fallback accepted = geometric continuity maintained;
                # reset the CV counter so future tracking_lost sections get fresh capacity.
                self._cv_consecutive = 0

            if (
                self.fallback_state.pending_rebuild
                and source_name in [
                    "local",
                    "global-relocal",
                    "lost-local-relocal",
                    "lost-global-relocal",
                    "essential-fallback",
                ]
                and self.last_strong_feat is not None
                and self.last_strong_pose is not None
                and self.last_strong_dataset_idx is not None
                and self._has_sufficient_rebuild_baseline(self.last_strong_pose, T_wc)
            ):
                match_count = self._count_feature_matches(self.last_strong_feat, feat)

                if match_count >= self.min_rebuild_match_count:
                    inserted = self._rebuild_local_map_from_pair(
                        feat_prev=self.last_strong_feat,
                        feat_curr=feat,
                        T_prev_wc=self.last_strong_pose,
                        T_curr_wc=T_wc,
                        current_dataset_idx=idx,
                    )
                    rebuild_context = (
                        "post-relocalization"
                        if source_name != "essential-fallback"
                        else "fallback-assisted"
                    )
                    print(
                        f"[Frame {idx}] {rebuild_context} rebuild from strong anchor "
                        f"(anchor_idx={self.last_strong_dataset_idx}, matches={match_count}) "
                        f"inserted={inserted}"
                    )

                    if self._is_rebuild_successful(inserted):
                        self.fallback_state.pending_rebuild = False
                        self.fallback_state.tracking_lost = False
                        self.fallback_state.consecutive_pnp_failures = 0
                        if source_name == "essential-fallback":
                            print(f"[Frame {idx}] rebuild during fallback succeeded -> exiting tracking_lost")
                    else:
                        print(f"[Frame {idx}] post-relocalization rebuild insufficient")
                else:
                    print(
                        f"[Frame {idx}] rebuild skipped after recovery: "
                        f"insufficient matches ({match_count})"
                    )

            if source_name in ["local", "global-relocal", "lost-local-relocal", "lost-global-relocal"]:
                if self._should_update_strong_anchor(idx, T_wc, track_result.inlier_count):
                    self.last_strong_feat = feat
                    self.last_strong_pose = T_wc
                    self.last_strong_dataset_idx = idx
                    print(
                        f"[Frame {idx}] updated strong anchor "
                        f"(inliers={track_result.inlier_count})"
                    )

            removed = 0
            if source_name in ["local", "global-relocal", "lost-local-relocal", "lost-global-relocal", "recovery-local"]:
                self.map_manager.increment_seen(track_result.matched_map_point_ids, idx)
                self.map_manager.increment_misses_for_unseen(set(track_result.matched_map_point_ids))
                current_map_size = len(self.map_manager.get_all_valid_points())
                recently_relocalized = (
                    idx - self.last_relocalization_frame
                ) <= self.post_relocalization_grace_frames

                if current_map_size < 250:
                    prune_min_observations = 1
                    prune_max_misses = 80
                else:
                    prune_min_observations = 2
                    prune_max_misses = 60

                if recently_relocalized:
                    prune_max_misses += 20

                removed = self.map_manager.prune_points(
                    min_observations=prune_min_observations,
                    max_consecutive_misses=prune_max_misses,
                )

                if current_map_size < 100 and not self.fallback_state.pending_rebuild:
                    self.fallback_state.pending_rebuild = True
                    print(
                        f"[Frame {idx}] emergency: map_size={current_map_size} < 100, "
                        f"triggering rebuild"
                    )

            if self.fallback_state.consecutive_pnp_failures >= self.max_fallback_frames_before_rebuild:
                self.fallback_state.pending_rebuild = True
                self.fallback_state.tracking_lost = True
                self._mark_tracking_lost_event(idx)
                self.fallback_state.consecutive_pnp_failures = 0
                # Reset rebuild anchor to the current frame so baseline accumulates from HERE.
                # Without this reset the strong anchor may be many frames old (high rotation →
                # almost no feature overlap when the rebuild fires in tracking_lost).
                self.last_strong_feat = feat
                self.last_strong_pose = T_wc
                self.last_strong_dataset_idx = idx
                print(f"[Frame {idx}] fallback limit reached -> pending rebuild, entering tracking_lost")

            current_kf_index = len(poses_wc) - 1
            translation_from_last_kf = np.linalg.norm(
                T_wc[:3, 3] - last_keyframe_pose[:3, 3]
            )
            frames_since_last_kf = idx - last_keyframe_dataset_idx
            rotation_from_last_kf_deg = self._rotation_step_deg(last_keyframe_pose, T_wc)

            new_points_world = np.empty((0, 3), dtype=np.float64)

            should_add_keyframe = (
                not self.pending_relocalization
                and
                frames_since_last_kf >= self.min_keyframe_gap
                and (
                    translation_from_last_kf > self.keyframe_translation_thresh
                    or rotation_from_last_kf_deg > self.keyframe_rotation_thresh_deg
                    or track_result.inlier_count < 40
                )
            )

            if should_add_keyframe:
                match_result = self.matcher.match(
                    last_keyframe_feat.keypoints, last_keyframe_feat.descriptors,
                    feat.keypoints, feat.descriptors
                )

                try:
                    if len(match_result.good_matches) < 25:
                        raise RuntimeError(
                            f"too few matches for keyframe triangulation: {len(match_result.good_matches)}"
                        )

                    T_c1_w = invert_transform(last_keyframe_pose)
                    T_c2_w = invert_transform(T_wc)

                    P1 = self.K @ T_c1_w[:3, :]
                    P2 = self.K @ T_c2_w[:3, :]

                    pts_w = triangulate_points(
                        P1,
                        P2,
                        match_result.pts1,
                        match_result.pts2,
                    )

                    C1_w = last_keyframe_pose[:3, 3]
                    C2_w = T_wc[:3, 3]

                    tri_result = filter_triangulated_points_world(
                        points_w=pts_w,
                        pts1=match_result.pts1,
                        pts2=match_result.pts2,
                        P1=P1,
                        P2=P2,
                        T_c1_w=T_c1_w,
                        T_c2_w=T_c2_w,
                        C1_w=C1_w,
                        C2_w=C2_w,
                        max_depth=60.0,
                        max_reproj_error=2.0,
                        min_parallax_deg=0.35,
                    )

                    train_indices = self._filtered_match_indices(
                        match_indices=match_result.train_indices,
                        tri_valid_mask=tri_result.valid_mask,
                    )
                    descs_for_points = self._descriptors_from_match_train_indices(feat, train_indices)

                    inserted_points = []
                    inserted_point_ids = []
                    for p3d, desc in zip(tri_result.points_3d, descs_for_points):
                        if not self.map_manager.has_nearby_point(p3d, distance_thresh=0.05):
                            point_id = self.map_manager.add_map_point(
                                position=p3d,
                                descriptor=desc,
                                created_in_keyframe=idx,
                                last_seen_frame=idx,
                            )
                            inserted_points.append(p3d)
                            inserted_point_ids.append(point_id)

                    num_new_points = len(inserted_points)

                    if source_name == "essential-fallback":
                        accept_keyframe = False
                    else:
                        force_keyframe_due_to_weak_tracking = (
                            source_name in ["local", "global-relocal"]
                            and track_result.inlier_count < 35
                            and num_new_points >= 3
                        )
                        accept_keyframe = (
                            num_new_points >= 6 or force_keyframe_due_to_weak_tracking
                        )

                    if accept_keyframe:
                        new_points_world = (
                            np.array(inserted_points, dtype=np.float64)
                            if inserted_points
                            else np.empty((0, 3), dtype=np.float64)
                        )

                        keyframe_indices.append(current_kf_index)
                        last_keyframe_feat = feat
                        last_keyframe_img = img_u
                        last_keyframe_pose = T_wc
                        last_keyframe_dataset_idx = idx
                        keyframe_map_ids = list(dict.fromkeys(
                            list(track_result.matched_map_point_ids) + inserted_point_ids
                        ))
                        self.keyframes.append(
                            self._make_keyframe_record(
                                frame_idx=idx,
                                T_wc=T_wc,
                                feat=feat,
                                map_point_ids=keyframe_map_ids,
                            )
                        )
                        print(f"[Frame {idx}] keyframe added, new points={len(new_points_world)}")
                    else:
                        print(
                            f"[Frame {idx}] keyframe rejected after triangulation: "
                            f"new_points={num_new_points}, inliers={track_result.inlier_count}"
                        )

                except Exception as e:
                    print(f"[Frame {idx}] triangulation skipped: {e}")
                    # Advance keyframe anchor even without new points so future attempts
                    # match against a closer frame rather than staling the reference.
                    if source_name not in ("essential-fallback",):
                        last_keyframe_feat = feat
                        last_keyframe_pose = T_wc
                        last_keyframe_dataset_idx = idx

            all_map_positions, _, _ = self.map_manager.get_positions_and_descriptors()
            reprojection_rmse_px = None
            if source_name in ["local", "global-relocal", "lost-local-relocal", "lost-global-relocal", "recovery-local"]:
                reprojection_rmse_px = self._compute_reprojection_rmse(
                    track_result.object_points,
                    track_result.image_points,
                    T_wc,
                )
            append_frame_log(
                frame_idx=idx,
                timestamp_ns=frame.timestamp_ns,
                accepted=True,
                source=source_name,
                pose_count=len(poses_wc),
                keyframe_count=len(keyframe_indices),
                map_point_count=len(all_map_positions),
                pnp_inliers=track_result.inlier_count if source_name != "essential-fallback" else None,
                reprojection_rmse_px=reprojection_rmse_px,
            )

            if visualizer is not None:
                canvas = visualizer.draw(
                    image=img_u,
                    keypoints=feat.keypoints,
                    poses_wc=poses_wc,
                    keyframe_indices=keyframe_indices,
                    map_points=all_map_positions,
                    new_map_points=new_points_world,
                    gt_positions=gt_positions,
                    frame_idx=idx,
                    pnp_inliers=track_result.inlier_count,
                )
                self.last_canvas = canvas.copy()
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            self.map_manager.mark_all_points_old()

            if source_name == "essential-fallback":
                print(
                    f"[Frame {idx}] source={source_name}, "
                    f"map={len(self.map_manager.get_all_valid_points())}, "
                    f"removed={removed}"
                )
            elif relocalized:
                print(
                    f"[Frame {idx}] source={source_name}, RELOCALIZED, "
                    f"PnP inliers={track_result.inlier_count}, "
                    f"map={len(self.map_manager.get_all_valid_points())}, "
                    f"removed={removed}"
                )
            else:
                print(
                    f"[Frame {idx}] source={source_name}, "
                    f"PnP inliers={track_result.inlier_count}, "
                    f"map={len(self.map_manager.get_all_valid_points())}, "
                    f"removed={removed}"
                )

        if visualizer is not None:
            visualizer.release()

        if debug_image_path is not None and self.last_canvas is not None:
            debug_image_path = Path(debug_image_path)
            debug_image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_image_path), self.last_canvas)
            print(f"Saved debug image: {debug_image_path}")

        runtime_seconds = time.perf_counter() - run_start_time
        summary = self._build_run_summary(
            final_frame_idx=final_frame_idx,
            final_keyframe_count=len(keyframe_indices),
            runtime_seconds=runtime_seconds,
        )

        if summary_path is None:
            summary_path = Path("outputs") / "metrics" / "room2_monocular_vo_run_summary.json"
        else:
            summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        import json
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(frame_log[0].keys()))
                writer.writeheader()
                writer.writerows(frame_log)
            print(f"Saved monocular frame log: {log_path}")

        print("\n================ Run Summary ================")
        for k, v in summary.items():
            print(f"{k}: {v}")
        print(f"Saved run summary: {summary_path}")

        cv2.destroyAllWindows()

        return MonocularVOMapResult(
            timestamps_ns=timestamps_ns,
            poses_wc=poses_wc,
            keyframe_indices=keyframe_indices,
        )
