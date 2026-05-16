import cv2
import numpy as np
import os
import time
from scipy.spatial.transform import Rotation as R_scipy
from dataset_loader import TumDataset


class StereoVO:
    def __init__(self, dataset_path):
        self.dataset = TumDataset(dataset_path)
        self.K = self.dataset.K
        self.f = self.dataset.focal_length
        self.B = self.dataset.baseline
        self.cx, self.cy = self.K[0, 2], self.K[1, 2]

        # V.B: Disparity map via StereoSGBM (Using your stable 64 disparities)
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=64, blockSize=5,
            P1=200, P2=800, disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=32, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def compute_3d_points(self, pts2d_old, pts2d_new, disparity):
        """V.A: Stereo Geometry (Equations 4 and 5) - YOUR STABLE VERSION"""
        pts3d, valid_pts2d_new = [], []
        pts2d_old = pts2d_old.reshape(-1, 2)
        pts2d_new = pts2d_new.reshape(-1, 2)

        for pt_old, pt_new in zip(pts2d_old, pts2d_new):
            u, v = int(pt_old[0]), int(pt_old[1])
            if 0 <= v < disparity.shape[0] and 0 <= u < disparity.shape[1]:
                d = disparity[v, u]
                if d > 0.0:
                    Z = (self.f * self.B) / d
                    # Your specific stable depth filter
                    if 0.2 < Z < 4.5:
                        B_over_d = self.B / d
                        X = B_over_d * (u - self.cx)
                        Y = B_over_d * (v - self.cy)
                        pts3d.append([X, Y, Z])
                        valid_pts2d_new.append(pt_new)

        return np.array(pts3d, dtype=np.float32), np.array(valid_pts2d_new, dtype=np.float32)

    def run(self):
        print("\n" + "=" * 80)
        print("--- RUNNING STABLE STEREO VO: MISSION CONTROL MODE ---")
        print("=" * 80)

        cur_R, cur_t = np.eye(3), np.zeros((3, 1))
        last_rel_R, last_rel_t = np.eye(3), np.zeros((3, 1))

        trajectory, reprojection_errors, runtimes = [], [], []
        failure_count = 0

        timestamp0 = self.dataset.get_timestamp(0)
        trajectory.append(f"{timestamp0:.9f} 0.0 0.0 0.0 0.0 0.0 0.0 1.0")

        imgL_prev, imgR_prev = self.dataset.get_stereo_frame(0)
        disp_prev = self.stereo.compute(imgL_prev, imgR_prev).astype(np.float32) / 16.0
        p0 = cv2.goodFeaturesToTrack(imgL_prev, mask=None, maxCorners=500, qualityLevel=0.01, minDistance=10)

        for i in range(self.dataset.length - 1):
            start_time = time.time()
            imgL_curr, imgR_curr = self.dataset.get_stereo_frame(i + 1)
            timestamp = self.dataset.get_timestamp(i + 1)

            use_fallback, mean_error, inliers_count = False, 0.0, 0

            if p0 is not None and len(p0) >= 20:
                p1, st, err = cv2.calcOpticalFlowPyrLK(imgL_prev, imgL_curr, p0, None, **self.lk_params)
                good_old = p0[st.ravel() == 1]
                good_new = p1[st.ravel() == 1]

                if len(good_new) >= 15:
                    E, mask_e = cv2.findEssentialMat(good_new, good_old, self.K, cv2.RANSAC, 0.999, 1.0)
                    if mask_e is not None:
                        good_old = good_old[mask_e.ravel() == 1]
                        good_new = good_new[mask_e.ravel() == 1]

                    pts3d, valid_new_2d = self.compute_3d_points(good_old, good_new, disp_prev)

                    if len(pts3d) >= 10:
                        success, rvec, tvec, inliers = cv2.solvePnPRansac(
                            pts3d, valid_new_2d, self.K, None,
                            iterationsCount=1000, reprojectionError=1.5, flags=cv2.SOLVEPNP_ITERATIVE
                        )

                        if success and inliers is not None and len(inliers) >= 10:
                            rel_R, _ = cv2.Rodrigues(rvec)
                            rel_t = tvec
                            inliers_count = len(inliers)

                            if np.linalg.norm(rel_t) < 0.3:
                                rel_R_inv = rel_R.T
                                rel_t_inv = -rel_R_inv.dot(rel_t)
                                cur_t = cur_t + cur_R.dot(rel_t_inv)
                                cur_R = cur_R.dot(rel_R_inv)

                                # RMSE Calculation
                                inlier_3d = pts3d[inliers.flatten()]
                                inlier_2d = valid_new_2d[inliers.flatten()]
                                proj_2d, _ = cv2.projectPoints(inlier_3d, rvec, tvec, self.K, None)
                                mean_error = np.mean(np.linalg.norm(inlier_2d - proj_2d.reshape(-1, 2), axis=1))

                                p0 = good_new[inliers.flatten()].reshape(-1, 1, 2)
                                if len(p0) < 100:
                                    p0 = cv2.goodFeaturesToTrack(imgL_curr, mask=None, maxCorners=500,
                                                                 qualityLevel=0.01, minDistance=10)
                            else:
                                use_fallback = True
                        else:
                            use_fallback = True
                    else:
                        use_fallback = True
                else:
                    use_fallback = True
            else:
                use_fallback = True

            if use_fallback:
                failure_count += 1
                status = "FALLBACK"
                reprojection_errors.append(f"{timestamp:.9f} 0.0000")
                p0 = cv2.goodFeaturesToTrack(imgL_curr, mask=None, maxCorners=500, qualityLevel=0.01, minDistance=10)
            else:
                status = "SUCCESS"
                reprojection_errors.append(f"{timestamp:.9f} {mean_error:.4f}")

            # Timing and Output
            frame_time = (time.time() - start_time) * 1000
            runtimes.append(frame_time)
            print(
                f"Frame {i:04d}->{i + 1:04d} | Status: {status:8s} | Inliers: {inliers_count:3d} | RMSE: {mean_error:6.3f}px | Time: {frame_time:6.2f}ms")

            imgL_prev, imgR_prev = imgL_curr, imgR_curr
            disp_prev = self.stereo.compute(imgL_curr, imgR_curr).astype(np.float32) / 16.0
            quat = R_scipy.from_matrix(cur_R).as_quat()
            trajectory.append(
                f"{timestamp:.9f} {cur_t[0][0]:.6f} {cur_t[1][0]:.6f} {cur_t[2][0]:.6f} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}")

        # Final Summary
        print("\n" + "=" * 80)
        print("--- STEREO VO FINAL SUMMARY ---")
        print(f"Mean Runtime:           {np.mean(runtimes):.2f} ms")
        print(f"Total Tracking Failures: {failure_count}")
        print("=" * 80)

        with open("stereo_trajectory.txt", 'w') as f:
            for line in trajectory: f.write(line + "\n")
        with open("stereo_reprojection_errors.txt", 'w') as f:
            for line in reprojection_errors: f.write(line + "\n")
        with open("stereo_stats.txt", "w") as f:
            f.write(f"{np.mean(runtimes):.4f}\n{failure_count}")


if __name__ == "__main__":
    dataset_path = r"C:\Users\RoboticsLab\PycharmProjects\stereo_vo_project\dataset\dataset-room2_512_16"
    vo = StereoVO(dataset_path)
    vo.run()