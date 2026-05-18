import cv2
import numpy as np
import time
import os
from scipy.spatial.transform import Rotation as R_scipy

from dataset_loader import UniversalTumDataset
from stereo_tracker import MonoExtendedStereoTracker
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
    dataset_path = r"C:\Users\RoboticsLab\PycharmProjects\stereo_vo_projectR\dataset\dataset-room2_512_16"
    gt_file_path = os.path.join(dataset_path, "mav0", "mocap0", "data.csv")

    print("Initializing Universal Dataset...", flush=True)
    dataset = UniversalTumDataset(dataset_path)

    # =====================================================================
    # 100% DYNAMIC CALIBRATION EXTRACTION
    # Explicitly pull the rectified baseline and pinhole matrix from camchain.yaml
    # =====================================================================
    dynamic_K = dataset.K
    dynamic_baseline = dataset.baseline

    print(f"-> Inherited Dynamic Focal Length : {dynamic_K[0, 0]:.2f} px", flush=True)
    print(f"-> Inherited Dynamic Baseline     : {dynamic_baseline:.5f} m", flush=True)

    # Inject the data-driven parameters strictly into the tracker
    tracker = MonoExtendedStereoTracker(dynamic_K, baseline=dynamic_baseline)
    dashboard = RealTimeVODashboard()

    print("Loading Ground Truth for Dashboard...", flush=True)
    data_gt = np.loadtxt(gt_file_path, delimiter=',', comments='#')
    gt_times = data_gt[:, 0]
    if len(gt_times) > 0 and gt_times[0] > 1e12:
        gt_times = gt_times / 1e9
    gt_xyz = data_gt[:, 1:4]
    gt_start_xyz = gt_xyz[0]

    last_valid_R_world = np.eye(3)
    last_valid_C_world = np.zeros((3, 1))
    delta_R_world = np.eye(3)
    delta_C_world = np.zeros((3, 1))

    trajectory_export = []
    runtimes_s = []
    failure_count = 0

    ts0 = dataset.get_timestamp(0)
    trajectory_export.append(f"{ts0:.9f} 0.0 0.0 0.0 0.0 0.0 0.0 1.0")

    # Setup Results Directory & Track Logging Targets
    os.makedirs("../Results_Room2_Dataset", exist_ok=True)
    err_file = "../Results_Room2_Dataset/stereo_reprojection_errors.txt"
    seeds_file = "../Results_Room2_Dataset/stereo_seeds.txt"
    kf_file = "../Results_Room2_Dataset/stereo_keyframes.txt"

    # Clear old files
    if os.path.exists(err_file): os.remove(err_file)
    if os.path.exists(seeds_file): os.remove(seeds_file)
    if os.path.exists(kf_file): os.remove(kf_file)

    print("\n" + "=" * 80, flush=True)
    print("--- SECTION D: PURE METRIC STEREO VO (DYNAMICALLY ANCHORED) ---", flush=True)
    print("=" * 80, flush=True)

    if dataset.length <= 1:
        print(f"CRITICAL ERROR: Dataset length is {dataset.length}. Check your path!", flush=True)
        exit(1)

    print("Entering main processing loop...", flush=True)

    for i in range(dataset.length - 1):
        start_time = time.time()

        img_curr_left, img_curr_right = dataset.get_stereo_frame(i + 1)
        ts = dataset.get_timestamp(i + 1)

        cur_R, cur_t, inliers, rmse, status = tracker.process_frame(img_curr_left, img_curr_right)

        # Extract absolute World Camera Center and orientation natively
        R_world = cur_R.T
        C_world = -R_world @ cur_t

        if "FAIL" in status or "LOST" in status or "WAITING" in status:
            failure_count += 1
            R_world = enforce_orthogonality(last_valid_R_world @ delta_R_world)
            C_world = last_valid_C_world + delta_C_world
            delta_C_world *= 0.8

            tracker.cur_R = R_world.T
            tracker.cur_t = -tracker.cur_R @ C_world
        else:
            if "SUCCESS" in status or "OK" in status or "KEYFRAME" in status or "STABILIZED" in status:
                delta_R_world = enforce_orthogonality(last_valid_R_world.T @ R_world)
                delta_C_world = C_world - last_valid_C_world

                disp_norm = np.linalg.norm(delta_C_world)
                if disp_norm > 0.1:
                    delta_C_world = (delta_C_world / disp_norm) * 0.1

                last_valid_R_world = R_world.copy()
                last_valid_C_world = C_world.copy()

        # Map top-down plotting axes directly to Ground Truth convention
        aligned_t = np.array([
            [C_world[2, 0]],  # X_plot follows GT tx
            [-C_world[0, 0]],  # Y_plot follows GT ty
            [-C_world[1, 0]]  # Z_plot matches GT tz
        ], dtype=np.float32)

        time_diffs = np.abs(gt_times - ts)
        closest_idx = np.argmin(time_diffs)
        curr_gt_xyz = gt_xyz[closest_idx] - gt_start_xyz

        empty_candidates = np.empty((0, 2), dtype=np.float32)
        dashboard.update(i, img_curr_left, tracker.active_2d, empty_candidates, aligned_t, rmse,
                         curr_gt_xyz=curr_gt_xyz)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Process interrupted by user.", flush=True)
            break

        total_points = len(tracker.active_3d)
        inlier_ratio = (inliers / total_points * 100.0) if total_points > 0 else 0.0

        # --- ALL LOGGING HANDLED HERE ---
        with open(err_file, "a") as f:
            f.write(f"{ts:.9f} {rmse:.4f}\n")

        # Fixed logic condition added here to match seed text formatting criteria
        if "INITIALIZING" in status or "SUCCESS" in status or "NEW_KEYFRAME" in status or "STABILIZED" in status or "BOOTSTRAP" in status:
            with open(seeds_file, "a") as f:
                f.write(f"{ts:.9f} {total_points}\n")

        with open(kf_file, "a") as f:
            f.write(f"{i} {status}\n")

        frame_time_s = time.time() - start_time
        runtimes_s.append(frame_time_s)
        frame_hz = 1.0 / frame_time_s if frame_time_s > 0 else 0.0

        if i % 10 == 0:
            print(
                f"Frame {i:04d} | Status: {status:20s} | Inliers: {inliers:3d}/{total_points:<3d} ({inlier_ratio:5.1f}%) | RMSE: {rmse:5.3f}px | {frame_hz:.2f} Hz",
                flush=True
            )

        # Log pure, unaligned absolute World Center coordinates for clean downstream SE3 evaluation
        quat_world = R_scipy.from_matrix(R_world).as_quat()
        trajectory_export.append(
            f"{ts:.9f} {C_world[0, 0]:.6f} {C_world[1, 0]:.6f} {C_world[2, 0]:.6f} "
            f"{quat_world[0]:.6f} {quat_world[1]:.6f} {quat_world[2]:.6f} {quat_world[3]:.6f}"
        )

    overall_hz = 1.0 / np.mean(runtimes_s) if len(runtimes_s) > 0 else 0.0
    with open("../Results_Room2_Dataset/stereo_trajectory.txt", "w") as f:
        for line in trajectory_export: f.write(line + "\n")
    with open("../Results_Room2_Dataset/stereo_stats.txt", "w") as f:
        f.write(f"{overall_hz:.2f}\n{failure_count}")

    print("\n" + "=" * 80, flush=True)
    print(f"HYBRID VO COMPLETE. Fallback Frames: {failure_count}", flush=True)
    print(f"OVERALL RUNTIME: {overall_hz:.2f} Hz", flush=True)
    print("=" * 80, flush=True)

    cv2.waitKey(100)
    dashboard.close()
    cv2.destroyAllWindows()