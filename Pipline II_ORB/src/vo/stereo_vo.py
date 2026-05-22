from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import csv
import json
import time

import cv2
import numpy as np

from src.geometry.rectification import compute_stereo_rectification_fisheye, rectify_stereo_pair
from src.io.image_loader import load_image
from src.io.trajectory_io import save_trajectory_tum
from src.stereo.stereo_frame import StereoFrameProcessor
from src.stereo.disparity import StereoSGBMConfig
from src.stereo.stereo_matcher import StereoFeatureExtractor
from src.stereo.stereo_tracker import StereoPnPTracker
from src.visualization.vo_visualizer import VOVisualizer


@dataclass
class StereoVOState:
    frame_idx: int
    timestamp_ns: int
    T_w_c: np.ndarray
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray | None
    points_3d: np.ndarray
    dense_keypoints: List[cv2.KeyPoint] | None = None
    dense_descriptors: np.ndarray | None = None
    dense_points_3d: np.ndarray | None = None
    left_vis: np.ndarray | None = None
    disparity_vis: np.ndarray | None = None


class StereoVisualOdometry:
    def __init__(
        self,
        dataset,
        calibration,
        output_path: Path,
        max_depth_m: float = 15.0,
        min_depth_m: float = 0.3,
        enable_visualization: bool = True,
        summary_path: Path | None = None,
        log_path: Path | None = None,
        feature_window_name: str = "VIBOT - Visual Odometry Live",
        disparity_window_name: str = "Stereo VO - Disparity",
        trajectory_window_name: str = "Stereo VO - Trajectory",
        # Dataset-tunable parameters
        min_disparity: float = 1.0,
        orb_nfeatures: int = 2000,
        min_tracker_inliers: int = 25,
        reprojection_error_px: float = 2.0,
        min_recovery_inliers: int = 15,
        recovery_reprojection_px: float = 2.0,
        min_3d3d_inliers: int = 12,
        num_disparities: int = 64,
        block_size: int = 5,
        uniqueness_ratio: int = 12,
        speckle_window_size: int = 100,
        speckle_range: int = 2,
        video_path: Path | None = None,
    ):
        self.dataset = dataset
        self.calibration = calibration
        self.output_path = output_path
        self.enable_visualization = enable_visualization
        self.summary_path = summary_path
        self.log_path = log_path
        self.min_depth_m = min_depth_m

        self.feature_window_name = feature_window_name
        self.disparity_window_name = disparity_window_name
        self.trajectory_window_name = trajectory_window_name
        self.visualizer = (
            VOVisualizer(window_name=feature_window_name, video_path=video_path)
            if self.enable_visualization else None
        )

        self.rectification = compute_stereo_rectification_fisheye(
            K1=calibration.cam0.intrinsics,
            d1=calibration.cam0.distortion,
            K2=calibration.cam1.intrinsics,
            d2=calibration.cam1.distortion,
            image_size=calibration.cam0.resolution,
            T_cam1_cam0=calibration.T_cam1_cam0,
        )

        self.K_rect_left = self.rectification.P1[:3, :3].astype(np.float64)
        self.baseline_rect_m = abs(float(self.rectification.P2[0, 3]) / float(self.rectification.P2[0, 0]))

        self.frame_processor = StereoFrameProcessor(
            K_left=self.K_rect_left,
            baseline_m=self.baseline_rect_m,
            sgbm_config=StereoSGBMConfig(
                min_disparity=0,
                num_disparities=num_disparities,
                block_size=block_size,
                uniqueness_ratio=uniqueness_ratio,
                speckle_window_size=speckle_window_size,
                speckle_range=speckle_range,
                disp12_max_diff=1,
            ),
            max_depth_m=max_depth_m,
        )

        self.feature_extractor = StereoFeatureExtractor(
            K_left=self.K_rect_left,
            baseline_m=self.baseline_rect_m,
            orb_nfeatures=orb_nfeatures,
            min_disparity=min_disparity,
            max_depth_m=max_depth_m,
            min_depth_m=min_depth_m,
        )

        self.tracker = StereoPnPTracker(
            K_left=self.K_rect_left,
            reprojection_error_px=reprojection_error_px,
            confidence=0.999,
            min_inliers=min_tracker_inliers,
            max_tracking_depth_m=max_depth_m,
        )
        self.recovery_tracker = StereoPnPTracker(
            K_left=self.K_rect_left,
            reprojection_error_px=recovery_reprojection_px,
            confidence=0.999,
            min_inliers=min_recovery_inliers,
            ratio_thresh=0.75,
            mutual_check=True,
            max_tracking_depth_m=max_depth_m,
            min_3d3d_inliers=min_3d3d_inliers,
        )

        self.poses: list[tuple[int, np.ndarray]] = []
        self.traj_points_xz: list[tuple[float, float]] = []
        self.stats = {
            "frames_attempted": 0,
            "frames_succeeded": 0,
            "tracking_failures": 0,
            "loose_retry_count": 0,
            "cv_fallback_count": 0,
        }
        self.frame_log: list[dict] = []

        # Constant-velocity fallback state
        self._last_T_prev_curr: np.ndarray | None = None
        self._cv_consecutive: int = 0
        self._cv_damp: float = 0.88
        self._cv_max_consecutive: int = 8

    def _normalize_disparity_for_display(self, disparity: np.ndarray) -> np.ndarray:
        disp = disparity.copy().astype(np.float32)
        valid = disp > 0.0

        if np.count_nonzero(valid) == 0:
            return np.zeros((disp.shape[0], disp.shape[1], 3), dtype=np.uint8)

        valid_disp = disp[valid]
        disp_min = np.percentile(valid_disp, 5)
        disp_max = np.percentile(valid_disp, 95)

        if disp_max <= disp_min:
            disp_max = disp_min + 1.0

        disp = np.clip((disp - disp_min) / (disp_max - disp_min), 0.0, 1.0)
        disp_u8 = (disp * 255.0).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_u8, cv2.COLORMAP_JET)
        disp_color[~valid] = (0, 0, 0)
        return disp_color

    def _draw_init_features(self, left_bgr: np.ndarray, keypoints: List[cv2.KeyPoint]) -> np.ndarray:
        vis = left_bgr.copy()
        for kp in keypoints:
            x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
        return vis

    def _draw_tracking_inliers(
        self,
        curr_left_bgr: np.ndarray,
        curr_keypoints: List[cv2.KeyPoint],
        inlier_matches: List[cv2.DMatch],
    ) -> np.ndarray:
        vis = curr_left_bgr.copy()
        for m in inlier_matches:
            pt = curr_keypoints[m.trainIdx].pt
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
        return vis

    def _needs_dense_features(self, points_3d: np.ndarray) -> bool:
        return (
            self.feature_extractor.orb_fallback is not None
            and len(points_3d) < self.feature_extractor.min_valid_features_for_fallback
        )

    def _prepare_dense_features(
        self,
        left_gray: np.ndarray,
        disparity: np.ndarray,
        points_3d: np.ndarray,
    ) -> tuple[List[cv2.KeyPoint] | None, np.ndarray | None, np.ndarray | None]:
        if not self._needs_dense_features(points_3d):
            return None, None, None

        dense_keypoints, dense_descriptors, dense_points_3d, _ = self.feature_extractor.extract_dense(
            left_gray,
            disparity,
        )
        if dense_descriptors is None or len(dense_keypoints) == 0 or len(dense_points_3d) == 0:
            return None, None, None

        if len(dense_points_3d) <= len(points_3d):
            return None, None, None

        return dense_keypoints, dense_descriptors, dense_points_3d

    def _retry_tracking_with_dense_features(
        self,
        prev_state: StereoVOState,
        curr_keypoints: List[cv2.KeyPoint],
        curr_descriptors: np.ndarray | None,
        curr_points_3d: np.ndarray,
        left_gray: np.ndarray,
        disparity: np.ndarray,
    ) -> tuple[object | None, tuple[List[cv2.KeyPoint] | None, np.ndarray | None, np.ndarray | None], bool]:
        curr_dense = self._prepare_dense_features(left_gray, disparity, curr_points_3d)
        prev_dense_available = (
            prev_state.dense_descriptors is not None
            and prev_state.dense_points_3d is not None
            and prev_state.dense_keypoints is not None
            and len(prev_state.dense_points_3d) > len(prev_state.points_3d)
        )
        curr_dense_available = curr_dense[1] is not None and curr_dense[2] is not None

        candidates = []

        def try_candidate(
            label: str,
            prev_keypoints: List[cv2.KeyPoint],
            prev_descriptors: np.ndarray | None,
            prev_points_3d: np.ndarray,
            candidate_keypoints: List[cv2.KeyPoint],
            candidate_descriptors: np.ndarray | None,
            candidate_points_3d: np.ndarray,
            used_curr_dense: bool,
        ) -> None:
            result = self.recovery_tracker.track(
                prev_keypoints=prev_keypoints,
                prev_descriptors=prev_descriptors,
                prev_points_3d=prev_points_3d,
                curr_keypoints=candidate_keypoints,
                curr_descriptors=candidate_descriptors,
                curr_points_3d=candidate_points_3d,
            )
            if result.success:
                candidates.append(
                    (
                        len(result.inlier_matches),
                        int(result.method == "pnp"),
                        len(result.matches),
                        label,
                        result,
                        used_curr_dense,
                    )
                )

        if curr_dense_available:
            try_candidate(
                label="primary-dense",
                prev_keypoints=prev_state.keypoints,
                prev_descriptors=prev_state.descriptors,
                prev_points_3d=prev_state.points_3d,
                candidate_keypoints=curr_dense[0],
                candidate_descriptors=curr_dense[1],
                candidate_points_3d=curr_dense[2],
                used_curr_dense=True,
            )

        if prev_dense_available:
            try_candidate(
                label="dense-primary",
                prev_keypoints=prev_state.dense_keypoints,
                prev_descriptors=prev_state.dense_descriptors,
                prev_points_3d=prev_state.dense_points_3d,
                candidate_keypoints=curr_keypoints,
                candidate_descriptors=curr_descriptors,
                candidate_points_3d=curr_points_3d,
                used_curr_dense=False,
            )

        if prev_dense_available and curr_dense_available:
            try_candidate(
                label="dense-dense",
                prev_keypoints=prev_state.dense_keypoints,
                prev_descriptors=prev_state.dense_descriptors,
                prev_points_3d=prev_state.dense_points_3d,
                candidate_keypoints=curr_dense[0],
                candidate_descriptors=curr_dense[1],
                candidate_points_3d=curr_dense[2],
                used_curr_dense=True,
            )

        if not candidates:
            return None, curr_dense, False

        candidates.sort(reverse=True)
        _, _, _, label, best_result, used_curr_dense = candidates[0]
        best_result.method = f"{best_result.method}+{label}"
        return best_result, curr_dense, used_curr_dense

    @staticmethod
    def _enforce_so3(R: np.ndarray) -> np.ndarray:
        U, _, Vt = np.linalg.svd(R)
        R_clean = U @ Vt
        if np.linalg.det(R_clean) < 0:
            U[:, 2] *= -1
            R_clean = U @ Vt
        return R_clean

    def _update_velocity(self, T_prev_curr: np.ndarray) -> None:
        self._last_T_prev_curr = T_prev_curr.copy()
        self._cv_consecutive = 0

    def _get_const_vel_T_prev_curr(self) -> np.ndarray | None:
        if self._last_T_prev_curr is None:
            return None
        damp = self._cv_damp ** self._cv_consecutive
        T = self._last_T_prev_curr.copy()
        T[:3, 3] = T[:3, 3] * damp
        # Re-enforce rotation validity after dampening
        T[:3, :3] = self._enforce_so3(T[:3, :3])
        return T

    def _retry_tracking_loose(
        self,
        prev_state: StereoVOState,
        curr_keypoints: List[cv2.KeyPoint],
        curr_descriptors: np.ndarray | None,
        curr_points_3d: np.ndarray,
    ) -> object | None:
        """Last-resort retry: no mutual check, wider reprojection error, lower min inliers."""
        result = self.recovery_tracker.track(
            prev_keypoints=prev_state.keypoints,
            prev_descriptors=prev_state.descriptors,
            prev_points_3d=prev_state.points_3d,
            curr_keypoints=curr_keypoints,
            curr_descriptors=curr_descriptors,
            curr_points_3d=curr_points_3d,
            reprojection_error_override=3.5,
            mutual_check_override=False,
            min_inliers_override=8,
        )
        if result.success:
            result.method = f"{result.method}+loose"
            return result
        return None

    def _show(
        self,
        left_vis: np.ndarray,
        frame_idx: int,
        match_count: int,
        inlier_count: int,
        valid_3d_count: int,
    ) -> None:
        if self.visualizer is None:
            return

        poses_wc = [T for _, T in self.poses]
        map_points = np.zeros((valid_3d_count, 3), dtype=np.float64)
        canvas = self.visualizer.draw(
            image=left_vis,
            keypoints=[],
            poses_wc=poses_wc,
            keyframe_indices=[],
            map_points=map_points,
            new_map_points=np.empty((0, 3), dtype=np.float64),
            frame_idx=frame_idx,
            pnp_inliers=inlier_count,
        )

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            print("Visualization stop requested, stopping stereo VO.")
            raise KeyboardInterrupt

    def _reprojection_rmse(self, tracking) -> float | None:
        if tracking.rvec is None or tracking.tvec is None:
            return None
        if tracking.object_points is None or tracking.image_points is None:
            return None
        if len(tracking.object_points) == 0 or len(tracking.image_points) == 0:
            return None
        projected, _ = cv2.projectPoints(
            tracking.object_points.astype(np.float32),
            tracking.rvec,
            tracking.tvec,
            self.K_rect_left,
            np.zeros((4, 1), dtype=np.float32),
        )
        err = projected.reshape(-1, 2) - tracking.image_points.astype(np.float32)
        return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))

    def _append_frame_log(
        self,
        frame_idx: int,
        timestamp_ns: int,
        accepted: bool,
        status: str,
        pose_count: int,
        valid_3d_count: int,
        match_count: int = 0,
        inlier_count: int = 0,
        reprojection_rmse_px: float | None = None,
    ) -> None:
        self.frame_log.append(
            {
                "frame_idx": int(frame_idx),
                "timestamp_s": float(timestamp_ns) * 1e-9,
                "accepted": int(bool(accepted)),
                "status": status,
                "pose_count": int(pose_count),
                "valid_3d_count": int(valid_3d_count),
                "match_count": int(match_count),
                "inlier_count": int(inlier_count),
                "reprojection_rmse_px": "" if reprojection_rmse_px is None else float(reprojection_rmse_px),
            }
        )

    def initialize(self, frame_idx: int = 0) -> StereoVOState:
        frame = self.dataset[frame_idx]

        left_img = load_image(frame.left_image_path, grayscale=False)
        right_img = load_image(frame.right_image_path, grayscale=False)
        left_rect, right_rect = rectify_stereo_pair(left_img, right_img, self.rectification)

        stereo = self.frame_processor.process(left_rect, right_rect)

        keypoints, descriptors, points_3d, _ = self.feature_extractor.extract(
            stereo.left_gray,
            stereo.disparity,
        )
        dense_keypoints, dense_descriptors, dense_points_3d = self._prepare_dense_features(
            stereo.left_gray,
            stereo.disparity,
            points_3d,
        )

        T_w_c = np.eye(4, dtype=np.float64)

        self.poses = [(frame.timestamp_ns, T_w_c.copy())]
        self.traj_points_xz = [(0.0, 0.0)]
        self.stats = {
            "frames_attempted": 1,
            "frames_succeeded": 1,
            "tracking_failures": 0,
            "loose_retry_count": 0,
            "cv_fallback_count": 0,
        }
        self.frame_log = []
        self._last_T_prev_curr = None
        self._cv_consecutive = 0

        left_vis = self._draw_init_features(left_rect, keypoints)
        disparity_vis = self._normalize_disparity_for_display(stereo.disparity)

        print(
            f"[Init] frame={frame_idx}, stereo_features={len(keypoints)}, "
            f"valid_3d={len(points_3d)}, baseline_rect_m={self.baseline_rect_m:.4f}"
        )

        if self.enable_visualization:
            self._show(
                left_vis=left_vis,
                frame_idx=frame_idx,
                match_count=0,
                inlier_count=0,
                valid_3d_count=len(points_3d),
            )
        self._append_frame_log(
            frame_idx=frame_idx,
            timestamp_ns=frame.timestamp_ns,
            accepted=True,
            status="init",
            pose_count=len(self.poses),
            valid_3d_count=len(points_3d),
        )

        return StereoVOState(
            frame_idx=frame_idx,
            timestamp_ns=frame.timestamp_ns,
            T_w_c=T_w_c,
            keypoints=keypoints,
            descriptors=descriptors,
            points_3d=points_3d,
            dense_keypoints=dense_keypoints,
            dense_descriptors=dense_descriptors,
            dense_points_3d=dense_points_3d,
            left_vis=left_vis,
            disparity_vis=disparity_vis,
        )

    def step(self, prev_state: StereoVOState, frame_idx: int) -> StereoVOState | None:
        frame = self.dataset[frame_idx]

        left_img = load_image(frame.left_image_path, grayscale=False)
        right_img = load_image(frame.right_image_path, grayscale=False)
        left_rect, right_rect = rectify_stereo_pair(left_img, right_img, self.rectification)

        stereo = self.frame_processor.process(left_rect, right_rect)

        curr_keypoints, curr_descriptors, curr_points_3d, _ = self.feature_extractor.extract(
            stereo.left_gray,
            stereo.disparity,
        )

        self.stats["frames_attempted"] += 1

        tracking = self.tracker.track(
            prev_keypoints=prev_state.keypoints,
            prev_descriptors=prev_state.descriptors,
            prev_points_3d=prev_state.points_3d,
            curr_keypoints=curr_keypoints,
            curr_descriptors=curr_descriptors,
            curr_points_3d=curr_points_3d,
        )
        curr_dense_keypoints = None
        curr_dense_descriptors = None
        curr_dense_points_3d = None

        disparity_vis = self._normalize_disparity_for_display(stereo.disparity)

        if not tracking.success:
            retry_tracking, curr_dense, used_curr_dense = self._retry_tracking_with_dense_features(
                prev_state=prev_state,
                curr_keypoints=curr_keypoints,
                curr_descriptors=curr_descriptors,
                curr_points_3d=curr_points_3d,
                left_gray=stereo.left_gray,
                disparity=stereo.disparity,
            )
            if retry_tracking is not None:
                tracking = retry_tracking
                curr_dense_keypoints, curr_dense_descriptors, curr_dense_points_3d = curr_dense
                if used_curr_dense and curr_dense_keypoints is not None:
                    curr_keypoints = curr_dense_keypoints
                    curr_descriptors = curr_dense_descriptors
                    curr_points_3d = curr_dense_points_3d
            else:
                # Loose retry: wider reprojection error, no mutual check, lower min inliers
                loose_tracking = self._retry_tracking_loose(
                    prev_state=prev_state,
                    curr_keypoints=curr_keypoints,
                    curr_descriptors=curr_descriptors,
                    curr_points_3d=curr_points_3d,
                )
                if loose_tracking is not None:
                    tracking = loose_tracking
                    self.stats["loose_retry_count"] += 1
                    print(
                        f"[Frame {frame_idx}] loose retry succeeded: "
                        f"method={tracking.method}, inliers={len(tracking.inlier_matches)}"
                    )
                else:
                    # Constant-velocity fallback: extrapolate pose from last known motion
                    self.stats["tracking_failures"] += 1
                    fail_vis = self._draw_init_features(left_rect, curr_keypoints)
                    print(
                        f"[Frame {frame_idx}] all tracking failed "
                        f"(matches={len(tracking.matches)}, cv_consecutive={self._cv_consecutive})"
                    )
                    cv_T = self._get_const_vel_T_prev_curr()
                    if cv_T is not None and self._cv_consecutive < self._cv_max_consecutive:
                        T_w_curr_cv = prev_state.T_w_c @ np.linalg.inv(cv_T)
                        self._cv_consecutive += 1
                        self.stats["cv_fallback_count"] += 1
                        self.poses.append((frame.timestamp_ns, T_w_curr_cv.copy()))
                        x = float(T_w_curr_cv[0, 3])
                        z = float(T_w_curr_cv[2, 3])
                        self.traj_points_xz.append((x, z))
                        print(
                            f"[Frame {frame_idx}] const-vel fallback applied "
                            f"(consecutive={self._cv_consecutive})"
                        )
                        if self.enable_visualization:
                            self._show(
                                left_vis=fail_vis,
                                frame_idx=frame_idx,
                                match_count=0,
                                inlier_count=0,
                                valid_3d_count=len(curr_points_3d),
                            )
                        self._append_frame_log(
                            frame_idx=frame_idx,
                            timestamp_ns=frame.timestamp_ns,
                            accepted=True,
                            status="const-vel",
                            pose_count=len(self.poses),
                            valid_3d_count=len(curr_points_3d),
                        )
                        # Return state using current frame features + CV-predicted pose
                        # so the next frame can attempt tracking from current features
                        return StereoVOState(
                            frame_idx=frame_idx,
                            timestamp_ns=frame.timestamp_ns,
                            T_w_c=T_w_curr_cv,
                            keypoints=curr_keypoints,
                            descriptors=curr_descriptors,
                            points_3d=curr_points_3d,
                            dense_keypoints=None,
                            dense_descriptors=None,
                            dense_points_3d=None,
                            left_vis=fail_vis,
                            disparity_vis=disparity_vis,
                        )

                    if self.enable_visualization:
                        self._show(
                            left_vis=fail_vis,
                            frame_idx=frame_idx,
                            match_count=len(tracking.matches),
                            inlier_count=0,
                            valid_3d_count=len(curr_points_3d),
                        )
                    self._append_frame_log(
                        frame_idx=frame_idx,
                        timestamp_ns=frame.timestamp_ns,
                        accepted=False,
                        status="failed",
                        pose_count=len(self.poses),
                        valid_3d_count=len(curr_points_3d),
                        match_count=len(tracking.matches),
                    )
                    return None
        elif self._needs_dense_features(curr_points_3d):
            curr_dense_keypoints, curr_dense_descriptors, curr_dense_points_3d = self._prepare_dense_features(
                stereo.left_gray,
                stereo.disparity,
                curr_points_3d,
            )

        T_prev_curr = np.eye(4, dtype=np.float64)
        T_prev_curr[:3, :3] = tracking.R
        T_prev_curr[:3, 3] = tracking.t

        T_w_curr = prev_state.T_w_c @ np.linalg.inv(T_prev_curr)

        # Update constant-velocity state on any real tracking success
        self._update_velocity(T_prev_curr)

        self.poses.append((frame.timestamp_ns, T_w_curr.copy()))
        self.stats["frames_succeeded"] += 1

        x = float(T_w_curr[0, 3])
        z = float(T_w_curr[2, 3])
        self.traj_points_xz.append((x, z))

        left_vis = self._draw_tracking_inliers(
            left_rect,
            curr_keypoints,
            tracking.inlier_matches,
        )

        print(
            f"[Frame {frame_idx}] method={tracking.method}, matches={len(tracking.matches)}, "
            f"inliers={len(tracking.inlier_matches)}, "
            f"curr_valid_3d={len(curr_points_3d)}"
        )
        self._append_frame_log(
            frame_idx=frame_idx,
            timestamp_ns=frame.timestamp_ns,
            accepted=True,
            status=tracking.method,
            pose_count=len(self.poses),
            valid_3d_count=len(curr_points_3d),
            match_count=len(tracking.matches),
            inlier_count=len(tracking.inlier_matches),
            reprojection_rmse_px=self._reprojection_rmse(tracking),
        )

        if self.enable_visualization:
            self._show(
                left_vis=left_vis,
                frame_idx=frame_idx,
                match_count=len(tracking.matches),
                inlier_count=len(tracking.inlier_matches),
                valid_3d_count=len(curr_points_3d),
            )

        return StereoVOState(
            frame_idx=frame_idx,
            timestamp_ns=frame.timestamp_ns,
            T_w_c=T_w_curr,
            keypoints=curr_keypoints,
            descriptors=curr_descriptors,
            points_3d=curr_points_3d,
            dense_keypoints=curr_dense_keypoints,
            dense_descriptors=curr_dense_descriptors,
            dense_points_3d=curr_dense_points_3d,
            left_vis=left_vis,
            disparity_vis=disparity_vis,
        )

    def _build_run_summary(
        self,
        start_idx: int,
        end_idx_exclusive: int,
        runtime_seconds: float,
    ) -> dict:
        frames_attempted = int(self.stats["frames_attempted"])
        frames_succeeded = int(self.stats["frames_succeeded"])
        tracking_failures = int(self.stats["tracking_failures"])
        cv_fallback_count = int(self.stats["cv_fallback_count"])
        loose_retry_count = int(self.stats["loose_retry_count"])
        avg_ms = (runtime_seconds / max(frames_attempted, 1)) * 1000.0
        return {
            "start_idx": int(start_idx),
            "end_idx_exclusive": int(end_idx_exclusive),
            "frames_attempted": frames_attempted,
            "frames_succeeded": frames_succeeded,
            "tracking_failures": tracking_failures,
            "loose_retry_count": loose_retry_count,
            "cv_fallback_count": cv_fallback_count,
            "saved_pose_count": int(len(self.poses)),
            "runtime_seconds": float(runtime_seconds),
            "avg_ms_per_frame": float(avg_ms),
            "runtime_fps_attempted": float(frames_attempted / max(runtime_seconds, 1e-9)),
            "runtime_fps_succeeded": float(frames_succeeded / max(runtime_seconds, 1e-9)),
            "rectified_baseline_m": float(self.baseline_rect_m),
            "max_depth_m": float(self.feature_extractor.max_depth_m),
            "min_depth_m": float(self.feature_extractor.min_depth_m),
        }

    def run(self, start_idx: int = 0, max_frames: int | None = None) -> list[tuple[int, np.ndarray]]:
        run_start = time.perf_counter()
        state = self.initialize(start_idx)

        end_idx = len(self.dataset) if max_frames is None else min(len(self.dataset), start_idx + max_frames)

        try:
            for frame_idx in range(start_idx + 1, end_idx):
                new_state = self.step(state, frame_idx)
                if new_state is None:
                    print(f"[Frame {frame_idx}] skipped")
                    continue
                state = new_state
        except KeyboardInterrupt:
            print("Stereo VO interrupted by user.")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        timestamps_ns = [ts for ts, _ in self.poses]
        poses_wc = [T for _, T in self.poses]
        save_trajectory_tum(self.output_path, timestamps_ns, poses_wc)
        print(f"Saved stereo trajectory to: {self.output_path}")

        runtime_seconds = time.perf_counter() - run_start
        summary = self._build_run_summary(
            start_idx=start_idx,
            end_idx_exclusive=end_idx,
            runtime_seconds=runtime_seconds,
        )
        if self.summary_path is not None:
            self.summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"Saved stereo run summary to: {self.summary_path}")

        if self.log_path is not None and self.frame_log:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.frame_log[0].keys()))
                writer.writeheader()
                writer.writerows(self.frame_log)
            print(f"Saved stereo frame log to: {self.log_path}")

        print("\n" + "=" * 60)
        print("Stereo VO Run Summary")
        print("=" * 60)
        print(f"Frames attempted:      {summary['frames_attempted']}")
        print(f"Frames succeeded:      {summary['frames_succeeded']}")
        print(f"Tracking failures:     {summary['tracking_failures']}")
        print(f"Loose retry saves:     {summary['loose_retry_count']}")
        print(f"Const-vel fallbacks:   {summary['cv_fallback_count']}")
        print(f"Saved poses:           {summary['saved_pose_count']}")
        print(f"Total runtime:         {summary['runtime_seconds']:.2f} s")
        print(f"Avg ms/frame:          {summary['avg_ms_per_frame']:.1f} ms")
        print(f"FPS (attempted):       {summary['runtime_fps_attempted']:.2f}")
        print(f"FPS (succeeded):       {summary['runtime_fps_succeeded']:.2f}")
        print(f"Rectified baseline:    {summary['rectified_baseline_m']:.4f} m")
        print("=" * 60)

        if self.visualizer is not None:
            self.visualizer.release()

        return self.poses
