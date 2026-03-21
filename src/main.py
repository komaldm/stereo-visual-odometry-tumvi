import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation as R_scipy
from scipy.optimize import least_squares

# Import your classes from the other files
from dataset_loader import TumDataset
from feature_matching import FeatureMatcher


# ---------------------------------------------------------
# MOTION-ONLY BUNDLE ADJUSTMENT (RUBRIC SECTION C)
# ---------------------------------------------------------
def reprojection_error(opt_vars, pts_3d, pts_2d, K):
    """
    Calculates the difference between the observed 2D points
    and the 3D points projected using the current pose guess.
    minimizes: x = pi(TX)
    """
    rvec = opt_vars[:3]
    tvec = opt_vars[3:6].reshape(3, 1)

    proj_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, None)
    proj_2d = proj_2d.reshape(-1, 2)

    return (proj_2d - pts_2d).ravel()


if __name__ == "__main__":
    # 1. Initialize Paths and Classes
    dataset_path = "C:/Users/RoboticsLab/PycharmProjects/stereo_vo_project/dataset/dataset-room2_512_16"

    dataset = TumDataset(dataset_path)
    matcher = FeatureMatcher()
    K = dataset.K

    # --- POSE TRACKING VARIABLES ---
    cur_R = np.eye(3)
    cur_t = np.zeros((3, 1))

    last_rel_R = np.eye(3)
    last_rel_t = np.zeros((3, 1))

    MIN_INLIERS = 50
    trajectory_export = []

    # Get frame 0 timestamp from filename
    frame0_name = os.path.basename(dataset.image_files[0]).replace('.png', '')
    timestamp0 = float(frame0_name) / 1e9
    trajectory_export.append(f"{timestamp0:.9f} 0.0 0.0 0.0 0.0 0.0 0.0 1.0")

    # Clear out the old error file if it exists so we start fresh
    if os.path.exists("reprojection_errors.txt"):
        os.remove("reprojection_errors.txt")

    print("\n--- STARTING MONOCULAR VO WITH OPTIMIZATION ---")
    print(f"Total Frames to process: {dataset.length}\n")

    for i in range(0, dataset.length - 1):
        img1 = dataset.get_frame(i)
        img2 = dataset.get_frame(i + 1)

        # FIX: Extract the timestamp immediately so the whole loop can use it
        img2_filename = os.path.basename(dataset.image_files[i + 1]).replace('.png', '')
        timestamp = float(img2_filename) / 1e9

        kp1, kp2, good, pts1, pts2 = matcher.detect_and_match(img1, img2)

        use_fallback = False
        inliers_count = 0
        inlier_ratio = 0.0
        mask = None

        if good is None or len(good) < MIN_INLIERS:
            use_fallback = True
            good_len = len(good) if good is not None else 0
        else:
            good_len = len(good)

            # A. Calculate Essential Matrix
            E, mask = cv2.findEssentialMat(
                pts1, pts2, K,
                method=cv2.RANSAC, prob=0.999, threshold=1.0
            )

            if E is None or mask is None:
                use_fallback = True
            else:
                inliers_count = int(mask.ravel().sum())
                inlier_ratio = (inliers_count / good_len) * 100

                if inliers_count < MIN_INLIERS:
                    use_fallback = True
                else:
                    # Filter points based on RANSAC mask
                    mask_bool = mask.ravel().astype(bool)
                    pts1_in = pts1[mask_bool]
                    pts2_in = pts2[mask_bool]

                    # B. Recover Pose & Cheirality Check
                    _, rel_R, rel_t, _ = cv2.recoverPose(E, pts1_in, pts2_in, K)

                    # --- TRIANGULATION (RUBRIC SECTION B) ---
                    P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
                    P2 = np.dot(K, np.hstack((rel_R, rel_t)))

                    pts4D = cv2.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T)
                    pts3D = (pts4D[:3, :] / pts4D[3, :]).T

                    # --- MOTION-ONLY BUNDLE ADJUSTMENT (RUBRIC SECTION C) ---
                    rvec_guess, _ = cv2.Rodrigues(rel_R)
                    initial_guess = np.hstack((rvec_guess.ravel(), rel_t.ravel()))

                    opt_res = least_squares(
                        reprojection_error,
                        initial_guess,
                        args=(pts3D, pts2_in, K),
                        method='lm',
                        max_nfev=50
                    )

                    opt_rvec = opt_res.x[:3]
                    opt_tvec = opt_res.x[3:6].reshape(3, 1)
                    rel_R_opt, _ = cv2.Rodrigues(opt_rvec)

                    rel_R = rel_R_opt
                    rel_t = opt_tvec

                    last_rel_R = rel_R
                    last_rel_t = rel_t

                    # --- SAVE REPROJECTION ERROR ---
                    # Calculate RMSE in pixels and write to file
                    rmse_pixels = np.sqrt((2 * opt_res.cost) / len(pts2_in))
                    with open("reprojection_errors.txt", "a") as f:
                        f.write(f"{timestamp:.9f} {rmse_pixels:.4f}\n")

        print(f"Frame {i} -> {i + 1}")

        if use_fallback:
            print("  [WARNING]: Low inlier count! Applying Constant Velocity Fallback.")
            rel_R = last_rel_R
            rel_t = last_rel_t
        else:
            print(f"  Good matches: {good_len}")
            print(f"  RANSAC Inliers: {inliers_count} ({inlier_ratio:.1f}%)")
            print("  [OPTIMIZED]: Motion-Only BA Applied.")

        print("-" * 30)

                # C. Update the absolute global pose (Inverted for Camera Tracking)
        rel_R_inv = rel_R.T
        rel_t_inv = -rel_R_inv.dot(rel_t)

        cur_t = cur_t + cur_R.dot(rel_t_inv)
        cur_R = cur_R.dot(rel_R_inv)

        # --- SAVE POSE FOR EXPORT ---
        quat = R_scipy.from_matrix(cur_R).as_quat()
        pose_str = f"{timestamp:.9f} {cur_t[0][0]:.6f} {cur_t[1][0]:.6f} {cur_t[2][0]:.6f} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"
        trajectory_export.append(pose_str)

        # --- VISUALIZATION ---
        if not use_fallback and good is not None and mask is not None:
            draw_matches = [good[j] for j in range(len(good)) if mask[j]]
            img_visual = cv2.drawMatches(
                img1, kp1, img2, kp2, draw_matches[:150], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.imshow("Filtered Inliers (RANSAC)", img_visual)
            if cv2.waitKey(1) == 27:
                break

    cv2.destroyAllWindows()

    output_file = "monocular_trajectory.txt"
    with open(output_file, 'w') as f:
        for line in trajectory_export:
            f.write(line + "\n")

    print(f"\nProcessing Complete. Trajectory saved to {output_file}")