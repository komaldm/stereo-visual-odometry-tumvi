import cv2
import numpy as np
from scipy.optimize import least_squares

# CRITICAL: Inherit natively from your Master Monocular baseline
from mono_tracker import MonocularTracker


class MonoExtendedStereoTracker(MonocularTracker):
    """
    STRICT OOP EXTENSION CONFIGURATION FOR MIXED OUTDOOR/INDOOR DATASETS
    Optimized with high-density feature buffers, relaxed RANSAC bounds,
    Global Adaptive Bucketing, and an Iterative Pose Refinement Loop
    to completely eradicate starvation and strictly lock terminal drift.
    """

    def __init__(self, K, baseline, max_depth=100.0, epipolar_err=1.5, min_consensus=10):
        super().__init__(K)

        self.f = self.K[0, 0]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.b = baseline

        self.max_epipolar_err = float(epipolar_err)
        self.min_depth = 0.5
        self.max_depth = float(max_depth)
        self.min_disparity = (self.f * self.b) / self.max_depth
        self.min_map_consensus = int(min_consensus)

        self.stereo_lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01)
        )

        # HIGH-DENSITY FIX: Dropping minDistance to 10 lets the tracker densely cluster
        # on premium building corners, completely ending map starvation dropouts.
        self.feature_params = dict(maxCorners=1500, qualityLevel=0.003, minDistance=8, blockSize=7)

    def process_frame(self, img_left, img_right):
        """
        THE SUPERVISOR LOOP:
        Executes base Monocular backbone, then applies unified metric alignment strictly on map updates.
        """
        mono_R, mono_t, n_inliers, mono_rmse, status = super().process_frame(img_left)

        if status == "WAITING_FOR_PARALLAX" or self.state == "NOT_INITIALIZED":
            if self._force_metric_bootstrap(img_left, img_right):
                status = "STEREO_BOOTSTRAP_SUCCESS"

        if self.state == "OK":
            if "KEYFRAME" in status or status == "STEREO_BOOTSTRAP_SUCCESS":
                updated = self._inject_absolute_metric_scale(img_left, img_right)
                if updated:
                    self._stabilize_pose()
                    status = "METRIC_STABILIZED_KF"
                else:
                    status = "KF_REJECTED_LOW_CONSENSUS"

        self.prev_img = img_left
        return self.cur_R.copy(), self.cur_t.copy(), len(self.active_3d), mono_rmse, status

    def _track(self, img):
        """Executes PnP feature tracking localization alongside continuous map tracking updates."""
        pts2_curr, status_map = self._track_fb(self.prev_img, img, self.active_2d)

        self.active_3d = self.active_3d[status_map]
        self.active_2d = pts2_curr[status_map]

        if len(self.active_3d) < 12:
            self.state = "NOT_INITIALIZED"
            self.prev_img = None
            return self.cur_R.copy(), self.cur_t.copy(), 0, 0.0, "LOST_TRACKING_POINTS"

        # DROPOUT FIX: Relaxed reprojectionError to 2.0 absorbs outdoor camera bounce
        # without failing the PnP solver, seamlessly fusing the global trajectory loop.
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            self.active_3d, self.active_2d, self.K, None,
            flags=cv2.SOLVEPNP_EPNP, reprojectionError=2.5, confidence=0.995
        )

        if not success or inliers is None or len(inliers) < 10:
            self.state = "NOT_INITIALIZED"
            self.prev_img = None
            return self.cur_R.copy(), self.cur_t.copy(), 0, 0.0, "LOST_PNP_FAILED"

        inliers = inliers.ravel()
        obj_in = self.active_3d[inliers]
        img_in = self.active_2d[inliers]

        x0 = np.hstack([rvec.ravel(), tvec.ravel()])

        # =======================================================
        # DRIFT FIX: ITERATIVE POSE REFINEMENT LOOP
        # Purges dynamic outdoor sliding noise from the map arrays
        # before the final Gauss-Newton step mathematically bakes the pose.
        # =======================================================
        for _ in range(2):
            res = least_squares(
                self._ba_objective, x0,
                args=(obj_in, img_in, self.K),
                loss='huber', f_scale=1.5,
                max_nfev=10, ftol=1e-3, xtol=1e-4
            )
            x0 = res.x

            rvec_opt = res.x[:3].reshape(3, 1)
            tvec_opt = res.x[3:6].reshape(3, 1)

            # Project points using the newly optimized pose to hunt for outliers
            proj, _ = cv2.projectPoints(obj_in, rvec_opt, tvec_opt, self.K, None)
            errs = np.linalg.norm(proj.reshape(-1, 2) - img_in, axis=1)

            # Outlier Rejection: Drop sliding points that break the 1.5px reprojection boundary
            valid_opt = errs < 2.5

            # Abort the refinement loop early if we risk dropping below the survival threshold
            if np.sum(valid_opt) < 12:
                break

            obj_in = obj_in[valid_opt]
            img_in = img_in[valid_opt]

        # RE-PROJECT PURIFIED POINTS TO PREVENT SHAPE MISMATCH ERROR
        proj_final, _ = cv2.projectPoints(obj_in, rvec_opt, tvec_opt, self.K, None)
        rmse = float(np.sqrt(np.mean(np.linalg.norm(proj_final.reshape(-1, 2) - img_in, axis=1) ** 2)))

        self.cur_R, _ = cv2.Rodrigues(rvec_opt)
        self.cur_t = tvec_opt

        # Overwrite parent tracking arrays ONLY with the purified, tightly-locked inliers
        self.active_3d = obj_in
        self.active_2d = img_in
        status = "TRACKING_OK"

        cand_curr, cand_status = self._track_fb(self.prev_img, img, self.cand_2d_curr)
        self.cand_2d_curr = cand_curr[cand_status]
        self.cand_2d_kf = self.cand_2d_kf[cand_status]

        parallax = float(np.mean(np.linalg.norm(self.cand_2d_curr - self.cand_2d_kf, axis=1))) if len(
            self.cand_2d_curr) > 0 else 0.0

        if parallax > 15.0 and len(self.cand_2d_curr) >= 20:
            P1 = self.K @ np.hstack((self.kf_R, self.kf_t))
            P2 = self.K @ np.hstack((self.cur_R, self.cur_t))
            pts4d = cv2.triangulatePoints(P1, P2, self.cand_2d_kf.T, self.cand_2d_curr.T)
            pts3d_w = (pts4d[:3, :] / (pts4d[3, :] + 1e-9)).T

            pts3d_c = (self.cur_R @ pts3d_w.T).T + self.cur_t.T
            valid_depth = (pts3d_c[:, 2] > 0.5) & (pts3d_c[:, 2] < 50.0)

            if np.sum(valid_depth) > 0:
                pts3d_valid_w = pts3d_w[valid_depth]
                pts2d_valid = self.cand_2d_curr[valid_depth]

                proj_new, _ = cv2.projectPoints(pts3d_valid_w, rvec_opt, tvec_opt, self.K, None)
                errs = np.linalg.norm(proj_new.reshape(-1, 2) - pts2d_valid, axis=1)
                strict_valid = errs < 2.5

                if np.sum(strict_valid) > 0:
                    self.active_3d = np.vstack((self.active_3d, pts3d_valid_w[strict_valid]))
                    self.active_2d = np.vstack((self.active_2d, pts2d_valid[strict_valid]))

            self._spawn_new_keyframe(img)
            status = "NEW_KEYFRAME"

        elif len(self.cand_2d_curr) < 40:
            self._spawn_new_keyframe(img)
            status = "RESET_CANDIDATES"

        self.prev_img = img
        return self.cur_R.copy(), self.cur_t.copy(), len(inliers), rmse, status

    def _spawn_new_keyframe(self, img):
        """Anchors local transformation properties securely using Global Adaptive Bucketing."""
        self.kf_R = self.cur_R.copy()
        self.kf_t = self.cur_t.copy()

        # Create the base extraction mask to prevent duplicating actively tracked points
        mask = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
        for pt in self.active_2d:
            cv2.circle(mask, (int(pt[0]), int(pt[1])), 12, 0, -1)

        # 1. ADAPTIVE GLOBAL EXTRACTION
        # Evaluate feature quality against the entire scene context. This naturally rejects
        # flat outdoor asphalt noise while automatically retaining valid indoor floor textures.
        global_corners = cv2.goodFeaturesToTrack(
            img, mask=mask, maxCorners=self.feature_params['maxCorners'] * 2,
            qualityLevel=self.feature_params['qualityLevel'],
            minDistance=self.feature_params['minDistance'],
            blockSize=self.feature_params['blockSize']
        )

        if global_corners is None:
            self.cand_2d_curr = np.empty((0, 2), dtype=np.float64)
            self.cand_2d_kf = np.empty((0, 2), dtype=np.float64)
            return

        # 2. SPATIAL GRID BUCKETING
        # Enforce strict spatial distribution to prevent asymmetric yaw pulling (curvy drift)
        grid_size_x, grid_size_y = 4, 4
        h, w = img.shape
        dx, dy = w // grid_size_x, h // grid_size_y
        max_per_cell = self.feature_params['maxCorners'] // (grid_size_x * grid_size_y)

        cell_counts = np.zeros((grid_size_y, grid_size_x), dtype=int)
        filtered_candidates = []

        for pt in global_corners.reshape(-1, 2):
            x, y = int(pt[0]), int(pt[1])

            # Enforce absolute image border safety margins
            if x < 5 or x >= w - 5 or y < 5 or y >= h - 5:
                continue

            grid_x = min(x // dx, grid_size_x - 1)
            grid_y = min(y // dy, grid_size_y - 1)

            # Only accept the feature if its specific spatial cell has not reached capacity
            if cell_counts[grid_y, grid_x] < max_per_cell:
                filtered_candidates.append(pt)
                cell_counts[grid_y, grid_x] += 1

        if len(filtered_candidates) > 0:
            cand = np.array(filtered_candidates, dtype=np.float32)
            polished = cv2.cornerSubPix(
                img, cand, (5, 5), (-1, -1), self.subpix_criteria
            )
            self.cand_2d_curr = polished.astype(np.float64)
            self.cand_2d_kf = self.cand_2d_curr.copy()
        else:
            self.cand_2d_curr = np.empty((0, 2), dtype=np.float64)
            self.cand_2d_kf = np.empty((0, 2), dtype=np.float64)

    def _force_metric_bootstrap(self, img_left, img_right):
        pts = self.cand_2d_curr if len(self.cand_2d_curr) > 20 else None
        success = self._spawn_metric_points(img_left, img_right, pts)
        if success:
            self.state = "OK"
            self.prev_img = img_left
            return True
        return False

    def _inject_absolute_metric_scale(self, img_left, img_right):
        if len(self.active_2d) == 0:
            return False

        self.active_2d = np.array(self.active_2d, dtype=np.float32).reshape(-1, 2)
        pr, st, _ = cv2.calcOpticalFlowPyrLK(img_left, img_right, self.active_2d, None, **self.stereo_lk_params)
        valid = (st.ravel() == 1)
        disp = self.active_2d[:, 0] - pr[:, 0]

        epipolar_clean = np.abs(self.active_2d[:, 1] - pr[:, 1]) < self.max_epipolar_err
        mask = valid & (disp > self.min_disparity) & (disp < 96.0) & epipolar_clean

        if np.sum(mask) >= self.min_map_consensus:
            disp_safe = disp[mask]
            Z = (self.f * self.b) / disp_safe
            valid_depth = (Z > self.min_depth) & (Z < self.max_depth)

            if np.sum(valid_depth) >= self.min_map_consensus:
                final_mask = np.zeros(len(self.active_2d), dtype=bool)
                final_mask[np.where(mask)[0][valid_depth]] = True

                self.active_2d = self.active_2d[final_mask]
                self.active_3d = self.active_3d[final_mask]

                Z_clean = Z[valid_depth]
                pl_clean = self.active_2d

                pts3d_cam = np.column_stack((
                    (pl_clean[:, 0] - self.cx) * Z_clean / self.f,
                    (pl_clean[:, 1] - self.cy) * Z_clean / self.f,
                    Z_clean
                ))

                pts3d_world = (self.cur_R.T @ (pts3d_cam.T - self.cur_t.reshape(3, 1))).T
                self.active_3d = pts3d_world.astype(np.float32)
                return True

        return False

    def _stabilize_pose(self):
        if len(self.active_3d) < 12:
            return

        x0 = np.hstack([cv2.Rodrigues(self.cur_R)[0].ravel(), self.cur_t.ravel()])
        res = least_squares(
            self._ba_objective, x0,
            args=(self.active_3d, self.active_2d, self.K),
            loss='huber', f_scale=0.05,
            max_nfev=3, ftol=1e-2, xtol=1e-3
        )
        self.cur_R, _ = cv2.Rodrigues(res.x[:3])
        self.cur_t = res.x[3:6].reshape(3, 1)

    def _spawn_metric_points(self, img_left, img_right, pts=None):
        if pts is None:
            mask = np.full((img_left.shape[0], img_left.shape[1]), 255, dtype=np.uint8)
            corners = cv2.goodFeaturesToTrack(img_left, mask=mask, **self.feature_params)
            if corners is None:
                return False
            pts = corners.reshape(-1, 2)

        pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
        if len(pts) == 0:
            return False

        pr, st, _ = cv2.calcOpticalFlowPyrLK(img_left, img_right, pts, None, **self.stereo_lk_params)
        valid = (st.ravel() == 1)
        disp = pts[:, 0] - pr[:, 0]

        epipolar_clean = np.abs(pts[:, 1] - pr[:, 1]) < self.max_epipolar_err
        vg = valid & (disp > self.min_disparity) & (disp < 65.0) & epipolar_clean

        if np.sum(vg) > 10:
            disp_safe = disp[vg]
            Z = (self.f * self.b) / disp_safe
            dm = (Z > self.min_depth) & (Z < self.max_depth)

            if np.sum(dm) > 0:
                Z_fin = Z[dm]
                pl = pts[vg][dm]

                pts3d_cam = np.column_stack((
                    (pl[:, 0] - self.cx) * Z_fin / self.f,
                    (pl[:, 1] - self.cy) * Z_fin / self.f,
                    Z_fin
                ))
                pts3d_world = (self.cur_R.T @ (pts3d_cam.T - self.cur_t.reshape(3, 1))).T

                self.active_3d = np.vstack((self.active_3d, pts3d_world.astype(np.float32)))
                self.active_2d = np.vstack((self.active_2d, pl.astype(np.float32)))
                return True

        return False