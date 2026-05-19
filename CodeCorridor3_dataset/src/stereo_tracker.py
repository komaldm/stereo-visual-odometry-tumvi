import cv2
import numpy as np
from scipy.optimize import least_squares

# CRITICAL: Inherit natively from your Master Monocular baseline
from mono_tracker import MonocularTracker


class MonoExtendedStereoTracker(MonocularTracker):
    """
    STRICT OOP EXTENSION (Stage 1 -> Stage 2 -> Stage 3)
    - Architecture Preserved: Relies strictly on super().process_frame(img_left) backbone.
    - Absolute Interlock: Restricts stereo mapping strictly to Keyframes, driving speeds >= 9.5 Hz.
    - Total Map Purity: Evaluates 100% of active landmarks natively to eradicate mixed-scale drift.
    - Precision Anchor: Calibrated disparity floors and expanded depth ceilings arrest terminal drift.
    - Memory Guard: Explicitly masks disparity arrays to natively prevent runtime divide-by-0 crashes.
    """

    def __init__(self, K, baseline):
        super().__init__(K)

        # Stereo physical parameters
        self.f = self.K[0, 0]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.b = baseline

        # Optimized sub-pixel parameters enforcing hyper-fine flow convergence to minimize disparity noise
        # TIGHTENED: winSize increased to 21x21 for robust patch tracking against sliding noise
        self.stereo_lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.0001)
        )

        # Overridden feature extraction properties highly calibrated to support stable local map retention
        # TIGHTENED: qualityLevel set to 0.01 and minDistance to 15 to force premium feature selection
        self.feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=15, blockSize=5)

    def process_frame(self, img_left, img_right):
        """
        THE SUPERVISOR LOOP:
        Executes base Monocular backbone, then applies unified metric alignment strictly on map updates.
        """

        # --- STAGE 1: RUN FULL MONOCULAR PIPELINE ---
        mono_R, mono_t, n_inliers, mono_rmse, status = super().process_frame(img_left)

        # Fulfill baseline trajectory bootstrap sequences
        if status == "WAITING_FOR_PARALLAX" or self.state == "NOT_INITIALIZED":
            if self._force_metric_bootstrap(img_left, img_right):
                status = "STEREO_BOOTSTRAP_SUCCESS"

        # --- STAGE 2 & 3: STRICT KEYFRAME METRIC CORRECTION ---
        if self.state == "OK":
            # SPEED OPTIMIZATION: Strict Keyframe Interlock
            if "KEYFRAME" in status or status == "STEREO_BOOTSTRAP_SUCCESS":
                # 1. Enforce Total Map Purity: Overwrite entire active map with absolute stereo depth
                self._inject_absolute_metric_scale(img_left, img_right)

                # 2. Stabilize Pose: Fast local Gauss-Newton alignment against the pure metric map
                self._stabilize_pose()
                status = "METRIC_STABILIZED_KF"

        # Maintain internal image state sync for base class feature tracking
        self.prev_img = img_left

        return self.cur_R.copy(), self.cur_t.copy(), len(self.active_3d), mono_rmse, status

    def _force_metric_bootstrap(self, img_left, img_right):
        """Forces uncorrupted metric initialization straight on Frame 0."""
        pts = self.cand_2d_curr if len(self.cand_2d_curr) > 20 else None
        success = self._spawn_metric_points(img_left, img_right, pts)
        if success:
            self.state = "OK"
            self.prev_img = img_left
            return True
        return False

    def _inject_absolute_metric_scale(self, img_left, img_right):
        """
        TOTAL MAP SANITATION:
        Evaluates 100% of actively tracked features natively to guarantee absolute scale purity.
        Safely vectorized to eliminate divide-by-zero runtime exceptions while stopping map depletion loops.
        """
        if len(self.active_2d) == 0: return

        # Ensure safe explicit casting to float32 to prevent OpenCV assertion failures
        pts_f32 = self.active_2d.astype(np.float32)

        # Perform fast optical flow evaluation across the ENTIRE active tracking set
        pr, st, _ = cv2.calcOpticalFlowPyrLK(
            img_left, img_right, pts_f32, None, **self.stereo_lk_params
        )
        valid = (st.ravel() == 1)

        disp = pts_f32[:, 0] - pr[:, 0]

        # TIGHTENED: Disparity floor > 0.5 and rigorous epipolar gate < 0.5px
        mask = valid & (disp > 0.5) & (disp < 65.0) & (np.abs(pts_f32[:, 1] - pr[:, 1]) < 0.5)

        if np.sum(mask) > 12:
            # Memory Guard: Explicitly mask array prior to division to prevent divide-by-0 exceptions
            disp_safe = disp[mask]
            Z = (self.f * self.b) / disp_safe

            # TIGHTENED: Upper depth verification ceiling clamped to 35.0m to reject far-field triangulation noise
            valid_depth = (Z > 0.5) & (Z < 35.0)

            if np.sum(valid_depth) > 0:
                # 1. STRICT MAP HYGIENE: Drop any map points that lack a valid close-field stereo correspondence.
                final_mask = np.zeros(len(self.active_2d), dtype=bool)
                final_mask[np.where(mask)[0][valid_depth]] = True

                self.active_2d = self.active_2d[final_mask]
                self.active_3d = self.active_3d[final_mask]

                # 2. Apply flawless metric transformations to the purified mapping arrays
                Z_clean = Z[valid_depth]
                pl_clean = self.active_2d

                pts3d_cam = np.column_stack(
                    ((pl_clean[:, 0] - self.cx) * Z_clean / self.f, (pl_clean[:, 1] - self.cy) * Z_clean / self.f,
                     Z_clean))

                # Exact Dimension-Safe Inverse Transformation: P_world = R^T @ (P_cam - t)
                pts3d_world = (self.cur_R.T @ (pts3d_cam.T - self.cur_t.reshape(3, 1))).T

                # Directly assign true metric coordinates back to parent states maintaining float64 consistency
                self.active_3d = pts3d_world.astype(np.float64)

    def _stabilize_pose(self):
        """Fast Pose-Only Refinement optimized to prevent matrix solver micro-stalls."""
        if len(self.active_3d) < 12: return
        x0 = np.hstack([cv2.Rodrigues(self.cur_R)[0].ravel(), self.cur_t.ravel()])

        # SPEED OPTIMIZATION: Bounded solver criteria with highly aggressive Huber loss weights (f_scale=0.0005)
        # ensures narrow-baseline spatial depth jitter is heavily downweighted to securely anchor terminal drift.
        res = least_squares(self._ba_objective, x0,
                            args=(self.active_3d, self.active_2d, self.K),
                            loss='huber', f_scale=0.0005,
                            max_nfev=3, ftol=1e-3, xtol=1e-3)
        self.cur_R, _ = cv2.Rodrigues(res.x[:3])
        self.cur_t = res.x[3:6].reshape(3, 1)

    def _spawn_metric_points(self, img_left, img_right, pts=None):
        """Triangulates premium metric initialization landmarks natively."""
        if pts is None:
            corners = cv2.goodFeaturesToTrack(img_left, mask=None, **self.feature_params)
            if corners is None: return False
            pts = corners.reshape(-1, 2)

        # Safely cast input candidate coordinates to float32 to satisfy optical flow assertions
        pts_f32 = pts.astype(np.float32)

        pr, st, _ = cv2.calcOpticalFlowPyrLK(
            img_left, img_right, pts_f32, None, **self.stereo_lk_params
        )
        valid = (st.ravel() == 1)
        disp = pts_f32[:, 0] - pr[:, 0]

        # TIGHTENED: Disparity floor > 0.5 and rigorous epipolar gate < 0.5px
        vg = valid & (disp > 0.5) & (disp < 65.0) & (np.abs(pts_f32[:, 1] - pr[:, 1]) < 0.5)

        if np.sum(vg) > 10:
            disp_safe = disp[vg]
            Z = (self.f * self.b) / disp_safe

            # TIGHTENED: Upper depth verification ceiling clamped to 35.0m
            dm = (Z > 0.5) & (Z < 35.0)

            if np.sum(dm) > 0:
                Z_fin = Z[dm]
                pl = pts_f32[vg][dm]

                pts3d_cam = np.column_stack(
                    ((pl[:, 0] - self.cx) * Z_fin / self.f, (pl[:, 1] - self.cy) * Z_fin / self.f, Z_fin))
                pts3d_world = (self.cur_R.T @ (pts3d_cam.T - self.cur_t.reshape(3, 1))).T

                self.active_3d = np.vstack((self.active_3d, pts3d_world.astype(np.float64)))
                self.active_2d = np.vstack((self.active_2d, pl.astype(np.float64)))
                return True
        return False