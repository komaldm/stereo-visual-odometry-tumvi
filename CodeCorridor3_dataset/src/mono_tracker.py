import cv2
import numpy as np
from scipy.optimize import least_squares


class MonocularTracker:
    def __init__(self, K):
        self.K = K.astype(np.float64)
        self.state = "NOT_INITIALIZED"

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        self.subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.005)

        self.feature_params = dict(maxCorners=1000, qualityLevel=0.005, minDistance=12, blockSize=3)

        self.cur_R = np.eye(3, dtype=np.float64)
        self.cur_t = np.zeros((3, 1), dtype=np.float64)

        self.active_3d = np.empty((0, 3), dtype=np.float64)
        self.active_2d = np.empty((0, 2), dtype=np.float64)
        self.prev_img = None

        self.kf_R = np.eye(3, dtype=np.float64)
        self.kf_t = np.zeros((3, 1), dtype=np.float64)
        self.cand_2d_curr = np.empty((0, 2), dtype=np.float64)
        self.cand_2d_kf = np.empty((0, 2), dtype=np.float64)

    def _ba_objective(self, opt_vars, pts3d, pts2d, K):
        rvec = opt_vars[:3].reshape(3, 1)
        tvec = opt_vars[3:6].reshape(3, 1)
        proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, None)
        return (proj.reshape(-1, 2) - pts2d).ravel()

    def _track_fb(self, img1, img2, pts1):
        if len(pts1) == 0:
            return np.empty((0, 2), dtype=np.float64), np.zeros(0, dtype=bool)

        pts1_reshaped = pts1.reshape(-1, 1, 2).astype(np.float32)

        pts2, st, _ = cv2.calcOpticalFlowPyrLK(img1, img2, pts1_reshaped, None, **self.lk_params)
        pts1_back, st_back, _ = cv2.calcOpticalFlowPyrLK(img2, img1, pts2, None, **self.lk_params)

        d = abs(pts1_reshaped - pts1_back).reshape(-1, 2).max(-1)
        good = d < 1.5

        status = (st.ravel() == 1) & (st_back.ravel() == 1) & good
        pts2_clean = pts2.reshape(-1, 2).astype(np.float64)

        h, w = img2.shape
        margin = 5
        inside = (pts2_clean[:, 0] >= margin) & (pts2_clean[:, 0] < w - margin) & \
                 (pts2_clean[:, 1] >= margin) & (pts2_clean[:, 1] < h - margin)

        status = status & inside

        if np.sum(status) > 0:
            polished = cv2.cornerSubPix(
                img2, pts2_clean[status].astype(np.float32), (5, 5), (-1, -1), self.subpix_criteria
            )
            pts2_clean[status] = polished.astype(np.float64)

        return pts2_clean, status

    def get_camera_center(self):
        return -self.cur_R.T @ self.cur_t

    def process_frame(self, img):
        if self.state == "NOT_INITIALIZED":
            return self._initialize(img)
        else:
            return self._track(img)

    def _initialize(self, img):
        if self.prev_img is None:
            corners = cv2.goodFeaturesToTrack(img, mask=None, **self.feature_params)
            if corners is not None:
                cand = corners.reshape(-1, 2)
                h, w = img.shape
                margin = 5
                inside = (cand[:, 0] >= margin) & (cand[:, 0] < w - margin) & \
                         (cand[:, 1] >= margin) & (cand[:, 1] < h - margin)
                cand = cand[inside]

                if len(cand) > 0:
                    polished = cv2.cornerSubPix(
                        img, cand.astype(np.float32), (5, 5), (-1, -1), self.subpix_criteria
                    )
                    self.cand_2d_curr = polished.astype(np.float64)
                    self.cand_2d_kf = self.cand_2d_curr.copy()
            self.prev_img = img
            return self.cur_R.copy(), self.cur_t.copy(), 0, 0.0, "INITIALIZING_SEEDS"

        pts2, status = self._track_fb(self.prev_img, img, self.cand_2d_curr)
        self.cand_2d_curr = pts2[status]
        self.cand_2d_kf = self.cand_2d_kf[status]

        if len(self.cand_2d_curr) < 30:
            self.prev_img = None
            return self.cur_R.copy(), self.cur_t.copy(), 0, 0.0, "INIT_FAIL_LOST_SEEDS"

        parallax = float(np.mean(np.linalg.norm(self.cand_2d_curr - self.cand_2d_kf, axis=1)))

        if parallax < 25.0:
            self.prev_img = img
            return self.cur_R.copy(), self.cur_t.copy(), len(self.cand_2d_curr), parallax, "WAITING_FOR_PARALLAX"

        E, mask = cv2.findEssentialMat(
            self.cand_2d_kf, self.cand_2d_curr, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None or mask is None:
            self.prev_img = None
            return self.cur_R.copy(), self.cur_t.copy(), 0, 0.0, "INIT_FAIL_EPI"

        _, R, t, mask_pose = cv2.recoverPose(
            E, self.cand_2d_kf, self.cand_2d_curr, self.K, mask=mask.copy()
        )

        inliers = mask_pose.ravel() > 0
        if np.sum(inliers) < 30:
            self.prev_img = None
            return self.cur_R.copy(), self.cur_t.copy(), 0, 0.0, "INIT_FAIL_RECOVER"

        pts1_in = self.cand_2d_kf[inliers]
        pts2_in = self.cand_2d_curr[inliers]

        P1 = self.K @ np.eye(3, 4, dtype=np.float64)
        P2 = self.K @ np.hstack((R, t))

        pts4d = cv2.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T)
        pts3d_w = (pts4d[:3, :] / (pts4d[3, :] + 1e-9)).T

        pts3d_c = (R @ pts3d_w.T).T + t.T
        valid = (pts3d_c[:, 2] > 0.5) & (pts3d_c[:, 2] < 45.0)

        if np.sum(valid) < 30:
            self.prev_img = None
            return self.cur_R.copy(), self.cur_t.copy(), 0, 0.0, "INIT_FAIL_CHEIRALITY"

        self.cur_R = R
        self.cur_t = t
        self.active_3d = pts3d_w[valid]
        self.active_2d = pts2_in[valid]

        self._spawn_new_keyframe(img)
        self.prev_img = img
        self.state = "OK"

        return self.cur_R.copy(), self.cur_t.copy(), int(np.sum(valid)), parallax, "EPI_SUCCESS"

    def _track(self, img):
        pts2_curr, status_map = self._track_fb(self.prev_img, img, self.active_2d)

        self.active_3d = self.active_3d[status_map]
        self.active_2d = pts2_curr[status_map]

        if len(self.active_3d) < 12:
            self.state = "NOT_INITIALIZED"
            self.prev_img = None
            return self.cur_R.copy(), self.cur_t.copy(), 0, 0.0, "LOST_TRACKING_POINTS"

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            self.active_3d, self.active_2d, self.K, None,
            flags=cv2.SOLVEPNP_EPNP, reprojectionError=1.5, confidence=0.99
        )

        if not success or inliers is None or len(inliers) < 10:
            self.state = "NOT_INITIALIZED"
            self.prev_img = None
            return self.cur_R.copy(), self.cur_t.copy(), 0, 0.0, "LOST_PNP_FAILED"

        inliers = inliers.ravel()
        obj_in = self.active_3d[inliers]
        img_in = self.active_2d[inliers]

        x0 = np.hstack([rvec.ravel(), tvec.ravel()])
        res = least_squares(self._ba_objective, x0, args=(obj_in, img_in, self.K), loss='huber', f_scale=0.1)

        rvec_opt = res.x[:3].reshape(3, 1)
        tvec_opt = res.x[3:6].reshape(3, 1)

        proj, _ = cv2.projectPoints(obj_in, rvec_opt, tvec_opt, self.K, None)
        rmse = float(np.sqrt(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_in, axis=1) ** 2)))

        self.cur_R, _ = cv2.Rodrigues(rvec_opt)
        self.cur_t = tvec_opt

        self.active_3d = obj_in
        self.active_2d = img_in
        status = "TRACKING_OK"

        cand_curr, cand_status = self._track_fb(self.prev_img, img, self.cand_2d_curr)
        self.cand_2d_curr = cand_curr[cand_status]
        self.cand_2d_kf = self.cand_2d_kf[cand_status]

        parallax = float(np.mean(np.linalg.norm(self.cand_2d_curr - self.cand_2d_kf, axis=1))) if len(
            self.cand_2d_curr) > 0 else 0.0

        if len(self.active_3d) < 80 or parallax > 10.0:
            if len(self.cand_2d_curr) >= 20:
                P1 = self.K @ np.hstack((self.kf_R, self.kf_t))
                P2 = self.K @ np.hstack((self.cur_R, self.cur_t))

                pts4d = cv2.triangulatePoints(P1, P2, self.cand_2d_kf.T, self.cand_2d_curr.T)
                pts3d_w = (pts4d[:3, :] / (pts4d[3, :] + 1e-9)).T

                pts3d_c = (self.cur_R @ pts3d_w.T).T + self.cur_t.T
                valid_depth = (pts3d_c[:, 2] > 0.5) & (pts3d_c[:, 2] < 45.0)

                if np.sum(valid_depth) > 0:
                    pts3d_valid_w = pts3d_w[valid_depth]
                    pts2d_valid = self.cand_2d_curr[valid_depth]

                    proj_new, _ = cv2.projectPoints(pts3d_valid_w, rvec_opt, tvec_opt, self.K, None)
                    errs = np.linalg.norm(proj_new.reshape(-1, 2) - pts2d_valid, axis=1)

                    strict_valid = errs < 1.5

                    if np.sum(strict_valid) > 0:
                        self.active_3d = np.vstack((self.active_3d, pts3d_valid_w[strict_valid]))
                        self.active_2d = np.vstack((self.active_2d, pts2d_valid[strict_valid]))

            self._spawn_new_keyframe(img)
            status = "NEW_KEYFRAME"

        self.prev_img = img
        return self.cur_R.copy(), self.cur_t.copy(), len(inliers), rmse, status

    def _spawn_new_keyframe(self, img):
        self.kf_R = self.cur_R.copy()
        self.kf_t = self.cur_t.copy()

        mask = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
        for pt in self.active_2d:
            cv2.circle(mask, (int(pt[0]), int(pt[1])), 10, 0, -1)

        grid_size_x, grid_size_y = 4, 4
        h, w = img.shape
        dx, dy = w // grid_size_x, h // grid_size_y

        corners_per_cell = self.feature_params['maxCorners'] // (grid_size_x * grid_size_y)
        all_candidates = []

        for i in range(grid_size_x):
            for j in range(grid_size_y):
                x1, x2 = i * dx, (i + 1) * dx
                y1, y2 = j * dy, (j + 1) * dy

                cell_mask = mask[y1:y2, x1:x2]
                cell_img = img[y1:y2, x1:x2]

                cell_corners = cv2.goodFeaturesToTrack(
                    cell_img, mask=cell_mask, maxCorners=corners_per_cell,
                    qualityLevel=self.feature_params['qualityLevel'],
                    minDistance=self.feature_params['minDistance'],
                    blockSize=self.feature_params['blockSize']
                )

                if cell_corners is not None:
                    cell_corners[:, 0, 0] += x1
                    cell_corners[:, 0, 1] += y1
                    all_candidates.extend(cell_corners)

        if len(all_candidates) > 0:
            cand = np.array(all_candidates).reshape(-1, 2)

            margin = 5
            inside = (cand[:, 0] >= margin) & (cand[:, 0] < w - margin) & \
                     (cand[:, 1] >= margin) & (cand[:, 1] < h - margin)
            cand = cand[inside]

            if len(cand) > 0:
                polished = cv2.cornerSubPix(
                    img, cand.astype(np.float32), (5, 5), (-1, -1), self.subpix_criteria
                )
                self.cand_2d_curr = polished.astype(np.float64)
                self.cand_2d_kf = self.cand_2d_curr.copy()
                return

        self.cand_2d_curr = np.empty((0, 2), dtype=np.float64)
        self.cand_2d_kf = np.empty((0, 2), dtype=np.float64)