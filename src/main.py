import cv2
import numpy as np
import os
import time
from scipy.spatial.transform import Rotation as R_scipy
from scipy.optimize import least_squares
from dataset_loader_mono import TumDataset
from feature_matching import FeatureMatcher


def reprojection_error(opt_vars, pts_3d, pts_2d, K):
    rvec = opt_vars[:3]
    tvec = opt_vars[3:6].reshape(3, 1)
    proj_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, None)
    proj_2d = proj_2d.reshape(-1, 2)
    return (proj_2d - pts_2d).ravel()


if __name__ == "__main__":
    dataset_path = r"C:\Users\RoboticsLab\PycharmProjects\stereo_vo_project\dataset\dataset-room2_512_16"
    dataset = TumDataset(dataset_path)
    matcher = FeatureMatcher()
    K = dataset.K

    cur_R, cur_t = np.eye(3), np.zeros((3, 1))
    last_rel_R, last_rel_t = np.eye(3), np.zeros((3, 1))

    MIN_INLIERS = 50
    trajectory_export = []
    runtimes = []
    failure_count = 0

    timestamp0 = dataset.get_timestamp(0)
    trajectory_export.append(f"{timestamp0:.9f} 0.0 0.0 0.0 0.0 0.0 0.0 1.0")

    if os.path.exists("reprojection_errors.txt"): os.remove("reprojection_errors.txt")

    print("\n" + "=" * 80)
    print("--- STARTING MONOCULAR VO: FULL DIAGNOSTICS MODE ---")
    print("=" * 80)

    for i in range(0, dataset.length - 1):
        start_time = time.time()

        img1, _ = dataset.get_stereo_frame(i)
        img2, _ = dataset.get_stereo_frame(i + 1)
        timestamp = dataset.get_timestamp(i + 1)

        kp1, kp2, good, pts1, pts2 = matcher.detect_and_match(img1, img2)

        use_fallback = False
        rmse_pixels = 0.0
        inliers_count = 0

        if good is None or len(good) < MIN_INLIERS:
            use_fallback = True
        else:
            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is None or mask is None:
                use_fallback = True
            else:
                mask_bool = mask.ravel().astype(bool)
                inliers_count = np.sum(mask_bool)
                if inliers_count < MIN_INLIERS:
                    use_fallback = True
                else:
                    pts1_in = pts1[mask_bool].reshape(-1, 2)
                    pts2_in = pts2[mask_bool].reshape(-1, 2)
                    _, rel_R, rel_t, _ = cv2.recoverPose(E, pts1_in, pts2_in, K)

                    P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
                    P2 = np.dot(K, np.hstack((rel_R, rel_t)))
                    pts4D = cv2.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T)
                    pts3D = (pts4D[:3, :] / (pts4D[3, :] + 1e-9)).T

                    rvec_guess, _ = cv2.Rodrigues(rel_R)
                    initial_guess = np.hstack((rvec_guess.ravel(), rel_t.ravel()))
                    opt_res = least_squares(reprojection_error, initial_guess, args=(pts3D, pts2_in, K),
                                            method='trf', loss='huber', max_nfev=50)

                    rel_R, _ = cv2.Rodrigues(opt_res.x[:3])
                    rel_t = opt_res.x[3:6].reshape(3, 1)
                    last_rel_R, last_rel_t = rel_R, rel_t
                    rmse_pixels = np.sqrt(np.mean(opt_res.fun ** 2))

                    with open("reprojection_errors.txt", "a") as f:
                        f.write(f"{timestamp:.9f} {rmse_pixels:.4f}\n")

        if use_fallback:
            failure_count += 1
            status = "FALLBACK"
            rel_R, rel_t = last_rel_R, last_rel_t
            with open("reprojection_errors.txt", "a") as f:
                f.write(f"{timestamp:.9f} 0.0000\n")
        else:
            status = "SUCCESS"

        # Global Pose Update
        rel_R_inv = rel_R.T;
        rel_t_inv = -rel_R_inv.dot(rel_t)
        cur_t = cur_t + cur_R.dot(rel_t_inv);
        cur_R = cur_R.dot(rel_R_inv)

        frame_time = (time.time() - start_time) * 1000
        runtimes.append(frame_time)

        # DETAILED PER-FRAME OUTPUT
        print(
            f"Frame {i:04d}->{i + 1:04d} | Status: {status:8s} | Inliers: {inliers_count:3d} | RMSE: {rmse_pixels:6.3f}px | Time: {frame_time:6.2f}ms")

        quat = R_scipy.from_matrix(cur_R).as_quat()
        trajectory_export.append(
            f"{timestamp:.9f} {cur_t[0][0]:.6f} {cur_t[1][0]:.6f} {cur_t[2][0]:.6f} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}")

    # FINAL SUMMARY OUTPUT
    print("\n" + "=" * 80)
    print("--- MONOCULAR VO FINAL SUMMARY ---")
    print(f"Total Frames Processed: {dataset.length}")
    print(f"Mean Runtime:           {np.mean(runtimes):.2f} ms")
    print(f"Total Tracking Failures: {failure_count} ({(failure_count / dataset.length) * 100:.2f}%)")
    print(f"Success Rate:            {100 - (failure_count / dataset.length) * 100:.2f}%")
    print("=" * 80)

    with open("monocular_stats.txt", "w") as f:
        f.write(f"{np.mean(runtimes):.4f}\n{failure_count}")
    with open("monocular_trajectory.txt", 'w') as f:
        for line in trajectory_export: f.write(line + "\n")