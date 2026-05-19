import cv2
import numpy as np
import torch
import pypose as pp
from scipy.spatial.transform import Rotation as R_scipy

# Keep deterministic seed for testing
cv2.setRNGSeed(0)
torch.manual_seed(0)


class PyPoseBA(torch.nn.Module):
    def __init__(self, init_pose, K):
        super().__init__()
        # FORCE FLOAT64 (Double Precision) for stable Cholesky decomposition
        self.pose = pp.Parameter(pp.SE3(init_pose).to(torch.float64))
        self.K = torch.tensor(K, dtype=torch.float64)

    def forward(self, pts3d, pts2d):
        # 1. Transform 3D points using PyPose SE3 matrix multiplication
        pts3d_trans = self.pose @ pts3d

        # 2. Project to 2D image plane
        z = pts3d_trans[:, 2] + 1e-7
        u = (pts3d_trans[:, 0] / z) * self.K[0, 0] + self.K[0, 2]
        v = (pts3d_trans[:, 1] / z) * self.K[1, 1] + self.K[1, 2]

        proj = torch.stack([u, v], dim=1)

        # 3. Return the residuals directly
        return proj - pts2d


class MonocularTracker:
    def __init__(self, K):
        self.K = K
        self.state = "NOT_INITIALIZED"

        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.feature_params = dict(maxCorners=1000, qualityLevel=0.005, minDistance=10, blockSize=3)

        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))

        self.active_3d = np.empty((0, 3), dtype=np.float32)
        self.active_2d = np.empty((0, 2), dtype=np.float32)
        self.prev_img = None

        self.kf_R = np.eye(3)
        self.kf_t = np.zeros((3, 1))
        self.cand_2d_curr = np.empty((0, 2), dtype=np.float32)
        self.cand_2d_kf = np.empty((0, 2), dtype=np.float32)

    def _track_fb(self, img1, img2, pts1):
        if len(pts1) == 0: return np.empty((0, 2)), np.zeros(0, dtype=bool)
        pts1_reshaped = pts1.reshape(-1, 1, 2).astype(np.float32)
        pts2, st, _ = cv2.calcOpticalFlowPyrLK(img1, img2, pts1_reshaped, None, **self.lk_params)
        pts1_back, st_back, _ = cv2.calcOpticalFlowPyrLK(img2, img1, pts2, None, **self.lk_params)
        d = abs(pts1_reshaped - pts1_back).reshape(-1, 2).max(-1)
        good = d < 1.0
        status = (st.ravel() == 1) & (st_back.ravel() == 1) & good
        pts2_clean = pts2.reshape(-1, 2)

        h, w = img2.shape
        margin = 5
        inside = (pts2_clean[:, 0] >= margin) & (pts2_clean[:, 0] < w - margin) & \
                 (pts2_clean[:, 1] >= margin) & (pts2_clean[:, 1] < h - margin)
        status = status & inside

        if np.sum(status) > 0:
            pts2_clean[status] = cv2.cornerSubPix(img2, pts2_clean[status].astype(np.float32), (5, 5), (-1, -1),
                                                  self.subpix_criteria)
        return pts2_clean, status

    def process_frame(self, img):
        if self.state == "NOT_INITIALIZED":
            return self._initialize(img)
        else:
            return self._track(img)

    def _initialize(self, img):
        if self.prev_img is None:
            self._spawn_new_keyframe(img)
            self.prev_img = img
            return self.cur_R.copy(), self.cur_t.copy(), 0, 0.0, "INITIALIZING_SEEDS"

        pts2, status = self._track_fb(self.prev_img, img, self.cand_2d_curr)
        self.cand_2d_curr = pts2[status]
        self.cand_2d_kf = self.cand_2d_kf[status]

        if len(self.cand_2d_curr) < 30:
            self.prev_img = None
            return self.cur_R.copy(), self.cur_t.copy(), 0, 0.0, "INIT_FAIL_LOST_SEEDS"

        parallax = float(np.mean(np.linalg.norm(self.cand_2d_curr - self.cand_2d_kf, axis=1)))
        if parallax < 15.0:
            self.prev_img = img
            return self.cur_R.copy(), self.cur_t.copy(), len(self.cand_2d_curr), parallax, "WAITING_FOR_PARALLAX"

        E, mask = cv2.findEssentialMat(self.cand_2d_curr, self.cand_2d_kf, self.K, cv2.RANSAC, 0.999, 1.0)
        if E is None or mask is None:
            self.prev_img = None
            return self.cur_R.copy(), self.cur_t.copy(), 0, 0.0, "INIT_FAIL_EPI"

        _, R, t, mask_pose = cv2.recoverPose(E, self.cand_2d_curr, self.cand_2d_kf, self.K, mask=mask.copy())
        inliers = mask_pose.ravel() > 0
        if np.sum(inliers) < 30:
            self.prev_img = None
            return self.cur_R.copy(), self.cur_t.copy(), 0, 0.0, "INIT_FAIL_RECOVER"

        P1 = self.K @ np.eye(3, 4)
        P2 = self.K @ np.hstack((R, t))
        pts4d = cv2.triangulatePoints(P1, P2, self.cand_2d_kf[inliers].T, self.cand_2d_curr[inliers].T)
        pts3d = (pts4d[:3, :] / (pts4d[3, :] + 1e-9)).T

        valid = (pts3d[:, 2] > 0.1) & (pts3d[:, 2] < 20.0)
        if np.sum(valid) < 30:
            self.prev_img = None
            return self.cur_R.copy(), self.cur_t.copy(), 0, 0.0, "INIT_FAIL_CHEIRALITY"

        self.cur_R, self.cur_t = R, t
        self.active_3d, self.active_2d = pts3d[valid], self.cand_2d_curr[inliers][valid]
        self._spawn_new_keyframe(img)
        self.prev_img = img
        self.state = "OK"
        return self.cur_R.copy(), self.cur_t.copy(), np.sum(valid), parallax, "EPI_SUCCESS"

    def _track(self, img):
        pts2_curr, status_map = self._track_fb(self.prev_img, img, self.active_2d)
        self.active_3d, self.active_2d = self.active_3d[status_map], pts2_curr[status_map]

        if len(self.active_3d) < 12:
            self.state, self.prev_img = "NOT_INITIALIZED", None
            return self.cur_R.copy(), self.cur_t.copy(), 0, 0.0, "LOST_TRACKING_POINTS"

        success, rvec, tvec, inliers = cv2.solvePnPRansac(self.active_3d, self.active_2d, self.K, None,
                                                          flags=cv2.SOLVEPNP_EPNP, reprojectionError=2.0,
                                                          confidence=0.99)
        if not success or inliers is None or len(inliers) < 10:
            self.state, self.prev_img = "NOT_INITIALIZED", None
            return self.cur_R.copy(), self.cur_t.copy(), 0, 0.0, "LOST_PNP_FAILED"

        inliers = inliers.ravel()
        obj_in, img_in = self.active_3d[inliers], self.active_2d[inliers]

        # ==========================================
        # FIXED PYPOSE OPTIMIZATION BLOCK (FLOAT64)
        # ==========================================
        r_mat, _ = cv2.Rodrigues(rvec)
        quat = R_scipy.from_matrix(r_mat).as_quat()  # [x, y, z, w]

        # Use float64 tensors
        init_pose = torch.tensor([tvec[0, 0], tvec[1, 0], tvec[2, 0], quat[0], quat[1], quat[2], quat[3]],
                                 dtype=torch.float64)
        pts3d_tensor = torch.tensor(obj_in, dtype=torch.float64)
        pts2d_tensor = torch.tensor(img_in, dtype=torch.float64)

        # 1. Initialize the module WITH the initial pose
        pypose_model = PyPoseBA(init_pose, self.K)

        # 2. Add damping to TrustRegion to prevent singular matrices
        strategy = pp.optim.strategy.TrustRegion(radius=1e4)
        optimizer = pp.optim.LM(pypose_model,
                                strategy=strategy,
                                kernel=pp.optim.kernel.Huber(delta=1.0))

        # 3. Optimization Loop
        for _ in range(5):  # Reduced to 5 to save some compute time
            loss = optimizer.step(input=(pts3d_tensor, pts2d_tensor))

        # 4. Extract optimized pose
        opt_pose_array = pypose_model.pose.data.numpy()
        self.cur_t = opt_pose_array[:3].reshape(3, 1)
        self.cur_R = R_scipy.from_quat(opt_pose_array[3:]).as_matrix()

        # Calculate final RMSE
        final_res = pypose_model(pts3d_tensor, pts2d_tensor).detach().numpy()
        rmse = float(np.sqrt(np.mean(np.linalg.norm(final_res, axis=1) ** 2)))
        # ==========================================

        self.active_3d, self.active_2d = obj_in, img_in
        status = "TRACKING_OK"

        cand_curr, cand_status = self._track_fb(self.prev_img, img, self.cand_2d_curr)
        self.cand_2d_curr, self.cand_2d_kf = cand_curr[cand_status], self.cand_2d_kf[cand_status]

        parallax = np.mean(np.linalg.norm(self.cand_2d_curr - self.cand_2d_kf, axis=1)) if len(
            self.cand_2d_curr) > 0 else 0

        if len(self.active_3d) < 100 or parallax > 15.0:
            if len(self.cand_2d_curr) >= 20:
                P1 = self.K @ np.hstack((self.kf_R, self.kf_t))
                P2 = self.K @ np.hstack((self.cur_R, self.cur_t))
                pts4d = cv2.triangulatePoints(P1, P2, self.cand_2d_kf.T, self.cand_2d_curr.T)
                pts3d = (pts4d[:3, :] / (pts4d[3, :] + 1e-9)).T
                pts3d_c = (self.cur_R @ pts3d.T).T + self.cur_t.T
                valid_depth = (pts3d_c[:, 2] > 0.1) & (pts3d_c[:, 2] < 20.0)

                if np.sum(valid_depth) > 0:
                    pts3d_v, pts2d_v = pts3d[valid_depth], self.cand_2d_curr[valid_depth]

                    p3_t = torch.tensor(pts3d_v, dtype=torch.float64)
                    p2_t = torch.tensor(pts2d_v, dtype=torch.float64)
                    errs = torch.norm(pypose_model(p3_t, p2_t), dim=1).detach().numpy()

                    strict_valid = errs < 2.0
                    if np.sum(strict_valid) > 0:
                        self.active_3d = np.vstack((self.active_3d, pts3d_v[strict_valid]))
                        self.active_2d = np.vstack((self.active_2d, pts2d_v[strict_valid]))

            self._spawn_new_keyframe(img)
            status = "NEW_KEYFRAME"

        self.prev_img = img
        return self.cur_R.copy(), self.cur_t.copy(), len(inliers), rmse, status

    def _spawn_new_keyframe(self, img):
        self.kf_R, self.kf_t = self.cur_R.copy(), self.cur_t.copy()
        mask = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
        for pt in self.active_2d: cv2.circle(mask, (int(pt[0]), int(pt[1])), 10, 0, -1)

        gx, gy = 4, 4
        h, w = img.shape
        dx, dy = w // gx, h // gy
        c_per_cell = self.feature_params['maxCorners'] // (gx * gy)
        all_cand = []

        for i in range(gx):
            for j in range(gy):
                x1, x2, y1, y2 = i * dx, (i + 1) * dx, j * dy, (j + 1) * dy
                c = cv2.goodFeaturesToTrack(img[y1:y2, x1:x2], mask=mask[y1:y2, x1:x2], maxCorners=c_per_cell,
                                            qualityLevel=self.feature_params['qualityLevel'],
                                            minDistance=self.feature_params['minDistance'],
                                            blockSize=self.feature_params['blockSize'])
                if c is not None:
                    c[:, 0, 0] += x1
                    c[:, 0, 1] += y1
                    all_cand.extend(c)

        if len(all_cand) > 0:
            cand = np.array(all_cand).reshape(-1, 2)
            inside = (cand[:, 0] >= 5) & (cand[:, 0] < w - 5) & (cand[:, 1] >= 5) & (cand[:, 1] < h - 5)
            cand = cand[inside]
            if len(cand) > 0:
                self.cand_2d_curr = cv2.cornerSubPix(img, cand.astype(np.float32), (5, 5), (-1, -1),
                                                     self.subpix_criteria)
                self.cand_2d_kf = self.cand_2d_curr.copy()
            else:
                self.cand_2d_curr, self.cand_2d_kf = np.empty((0, 2), dtype=np.float32), np.empty((0, 2),
                                                                                                  dtype=np.float32)
        else:
            self.cand_2d_curr, self.cand_2d_kf = np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)