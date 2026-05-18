import cv2
import numpy as np
import time
import os
from scipy.spatial.transform import Rotation as R_scipy

# Your existing dataset and tracker imports
from dataset_loader_mono import TumDataset
from mono_tracker import MonocularTracker
from dashboard import RealTimeVODashboard


def enforce_orthogonality(R):
    """Ensures a rotation matrix stays a valid mathematical rotation."""
    U, _, Vt = np.linalg.svd(R)
    R_clean = U @ Vt
    if np.linalg.det(R_clean) < 0:
        U[:, 2] *= -1
        R_clean = U @ Vt
    return R_clean


if __name__ == "__main__":
    # --- PATH SETTINGS ---
    dataset_path = r"C:\Users\RoboticsLab\PycharmProjects\stereo_vo_projectR\dataset\dataset-room2_512_16"
    gt_file_path = os.path.join(dataset_path, "mav0", "mocap0", "data.csv")

    # --- INITIALIZE DATASET & TRACKER ---
    dataset = TumDataset(dataset_path)
    tracker = MonocularTracker(dataset.K)
    dashboard = RealTimeVODashboard()

    # --- LOAD GROUND TRUTH FOR REAL-TIME OVERLAY ---
    print("Loading Ground Truth for Dashboard...")
    data_gt = np.loadtxt(gt_file_path, delimiter=',', comments='#')
    gt_times = data_gt[:, 0]
    if len(gt_times) > 0 and gt_times[0] > 1e12:
        gt_times = gt_times / 1e9
    gt_xyz = data_gt[:, 1:4]
    gt_start_xyz = gt_xyz[0]

    # --- STATE VARIABLES ---
    last_valid_global_R = np.eye(3)
    last_valid_global_t = np.zeros((3, 1))
    delta_R = np.eye(3)
    delta_t_local = np.zeros((3, 1))

    trajectory_export = []
    runtimes_s = []
    failure_count = 0

    # Initialize Export with Start Frame
    ts0 = dataset.get_timestamp(0)
    trajectory_export.append(f"{ts0:.9f} 0.0 0.0 0.0 0.0 0.0 0.0 1.0")

    # Setup Results Directory & Add Tracking Files
    os.makedirs("../Results_Room2_Dataset", exist_ok=True)
    err_file = "../Results_Room2_Dataset/monocular_reprojection_errors.txt"
    seeds_file = "../Results_Room2_Dataset/monocular_seeds.txt"
    kf_file = "../Results_Room2_Dataset/monocular_keyframes.txt" # <--- ADDED KEYFRAME FILE

    # Clean old files to prevent appending to previous runs
    if os.path.exists(err_file): os.remove(err_file)
    if os.path.exists(seeds_file): os.remove(seeds_file)
    if os.path.exists(kf_file): os.remove(kf_file) # <--- ADDED REMOVAL

    print("\n" + "=" * 80)
    print("--- SECTION C: MONOCULAR VO (STATE-OF-THE-ART LOCAL MAP) ---")
    print("=" * 80)

    for i in range(dataset.length - 1):
        start_time = time.time()

        # Load Frame
        try:
            img_curr, _ = dataset.get_stereo_frame(i + 1)
        except AttributeError:
            img_curr = dataset.get_frame(i + 1)

        ts = dataset.get_timestamp(i + 1)

        # --- PROCESS FRAME ---
        cur_R, cur_t, inliers, rmse, status = tracker.process_frame(img_curr)

        # --- SAFE CONSTANT VELOCITY FALLBACK ---
        if "FAIL" in status or "LOST" in status or "WAITING" in status:
            failure_count += 1
            # Apply dampened delta to prevent trajectory "explosion"
            cur_R = enforce_orthogonality(last_valid_global_R @ delta_R)
            cur_t = last_valid_global_t + last_valid_global_R @ delta_t_local
            delta_t_local *= 0.8  # Friction/Dampening
            tracker.cur_R = cur_R
            tracker.cur_t = cur_t
        else:
            # Update velocity delta for smooth motion estimation
            if "SUCCESS" in status or "OK" in status or "KEYFRAME" in status:
                delta_R = last_valid_global_R.T @ cur_R
                delta_R = enforce_orthogonality(delta_R)
                delta_t_local = last_valid_global_R.T @ (cur_t - last_valid_global_t)

                # Cap speed (0.1m per frame limit)
                if np.linalg.norm(delta_t_local) > 0.1:
                    delta_t_local = (delta_t_local / np.linalg.norm(delta_t_local)) * 0.1

                last_valid_global_R = cur_R.copy()
                last_valid_global_t = cur_t.copy()

        # --- DASHBOARD SYNC & UPDATE ---
        # Sync Ground Truth for real-time light-green path
        time_diffs = np.abs(gt_times - ts)
        closest_idx = np.argmin(time_diffs)
        curr_gt_xyz = gt_xyz[closest_idx] - gt_start_xyz

        # Update the 5-panel professional dashboard
        dashboard.update(i, img_curr, tracker.active_2d, tracker.cand_2d_curr, cur_t, rmse, curr_gt_xyz=curr_gt_xyz)

        # --- LOGGING & STATS ---
        total_points = len(tracker.active_3d) if tracker.state == "OK" else len(tracker.cand_2d_curr)
        inlier_ratio = (inliers / total_points * 100.0) if total_points > 0 else 0.0

        # Log Errors
        with open(err_file, "a") as f:
            f.write(f"{ts:.9f} {rmse:.4f}\n")

        # Log Seeds
        if "INITIALIZING" in status or "SUCCESS" in status or "NEW_KEYFRAME" in status:
            with open(seeds_file, "a") as f:
                f.write(f"{ts:.9f} {total_points}\n")

        # <--- ADDED KEYFRAME LOGGING LOGIC --->
        # This writes the frame index and the exact text status for the plotting script
        with open(kf_file, "a") as f:
            f.write(f"{i} {status}\n")

        frame_time_s = time.time() - start_time
        runtimes_s.append(frame_time_s)
        frame_hz = 1.0 / frame_time_s if frame_time_s > 0 else 0.0

        if i % 10 == 0:
            print(
                f"Frame {i:04d} | Status: {status:20s} | Inliers: {inliers:3d}/{total_points:<3d} ({inlier_ratio:5.1f}%) | RMSE: {rmse:5.3f}px | {frame_hz:.2f} Hz")

        # Prepare TUM Export
        quat = R_scipy.from_matrix(cur_R).as_quat()
        trajectory_export.append(
            f"{ts:.9f} {cur_t[0][0]:.6f} {cur_t[1][0]:.6f} {cur_t[2][0]:.6f} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}")

    # --- FINAL EXPORT ---
    overall_hz = 1.0 / np.mean(runtimes_s) if len(runtimes_s) > 0 else 0.0
    with open("../Results_Room2_Dataset/monocular_trajectory.txt", "w") as f:
        for line in trajectory_export: f.write(line + "\n")
    with open("../Results_Room2_Dataset/monocular_stats.txt", "w") as f:
        f.write(f"{overall_hz:.2f}\n{failure_count}")

    print("\n" + "=" * 80)
    print(f"VO COMPLETE. Fallback Frames: {failure_count}")
    print(f"OVERALL RUNTIME: {overall_hz:.2f} Hz")
    print("=" * 80)
    print("\nSequence Complete.")
    dashboard.close()  # This releases the video file so it actually saves!
    cv2.destroyAllWindows()