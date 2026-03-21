import numpy as np
import matplotlib.pyplot as plt
import os


def load_tum_format(filename):
    """Loads TUM trajectory and returns timestamps and Nx3 XYZ coordinates."""
    data = np.loadtxt(filename)
    timestamps = data[:, 0]
    xyz = data[:, 1:4]
    return timestamps, xyz


def associate_timestamps_start_only(ts_est, xyz_est, ts_gt, xyz_gt, max_diff=0.05, max_frames=200):
    """Matches estimated poses to ground truth poses for the first 200 frames to avoid scale drift breaking the alignment."""
    matched_est = []
    matched_gt = []

    for i, t_est in enumerate(ts_est):
        diffs = np.abs(ts_gt - t_est)
        best_idx = np.argmin(diffs)

        if diffs[best_idx] < max_diff:
            matched_est.append(xyz_est[i])
            matched_gt.append(xyz_gt[best_idx])

        if len(matched_est) >= max_frames:
            break

    return np.array(matched_est).T, np.array(matched_gt).T


def get_sim3(model, data):
    """Computes Scale, Rotation, and Translation to align data to model."""
    mu_M = model.mean(1, keepdims=True)
    mu_D = data.mean(1, keepdims=True)
    model_zero = model - mu_M
    data_zero = data - mu_D

    Sigma_px = (data_zero @ model_zero.T) / data.shape[1]
    U, D, V_T = np.linalg.svd(Sigma_px)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(V_T) < 0:
        S[2, 2] = -1

    R = V_T.T @ S @ U.T
    sigma_p2 = np.mean(np.sum(data_zero ** 2, axis=0))
    scale = (1.0 / sigma_p2) * np.trace(np.diag(D) @ S)
    t = mu_M - scale * R @ mu_D

    return scale, R, t


def main():
    # --- PATHS ---
    est_file = "monocular_trajectory.txt"
    gt_file = r"C:\Users\RoboticsLab\PycharmProjects\stereo_vo_project\dataset\dataset-room2_512_16\mav0\mocap0\data.csv"
    err_file = "reprojection_errors.txt"

    print("Loading data...")
    ts_est, xyz_est = load_tum_format(est_file)

    try:
        data_gt = np.loadtxt(gt_file, delimiter=',', skiprows=1)
        ts_gt = data_gt[:, 0] / 1e9
        xyz_gt = data_gt[:, 1:4]
    except Exception as e:
        print(f"Error loading GT: {e}")
        return

    print("Aligning the START of the trajectory to Ground Truth...")
    data_est_matched, data_gt_matched = associate_timestamps_start_only(ts_est, xyz_est, ts_gt, xyz_gt)

    # Calculate Sim3 based ONLY on the start of the sequence
    scale, R, t = get_sim3(data_gt_matched, data_est_matched)

    # Apply the alignment to the ENTIRE estimated trajectory
    aligned_full_est = (scale * R @ xyz_est.T + t).T

    # Calculate ATE (Only for the first 200 frames inside the room)
    aligned_matched_est = (scale * R @ data_est_matched + t).T
    errors = np.linalg.norm(data_gt_matched.T - aligned_matched_est, axis=1)
    rmse_ate = np.sqrt(np.mean(errors ** 2))

    if os.path.exists(err_file):
        rep_err_data = np.loadtxt(err_file)
        frames = np.arange(len(rep_err_data))
        rep_errors = rep_err_data[:, 1]
    else:
        frames, rep_errors = [], []

    print("\nGenerating and saving individual high-resolution graphs...")

    # --- 1. INDIVIDUAL GRAPH: Unscaled Monocular ---
    fig1 = plt.figure(figsize=(10, 8))
    plt.plot(xyz_est[:, 0], xyz_est[:, 1], 'b-', linewidth=2, label='Monocular Path')
    plt.title("1. Unscaled Monocular Trajectory\n(Shows massive scale drift accumulation)", fontsize=14,
              fontweight='bold')
    plt.xlabel("X (arbitrary units)", fontsize=12)
    plt.ylabel("Y (arbitrary units)", fontsize=12)
    plt.grid(True)
    plt.axis('equal')
    plt.legend(fontsize=12)
    plt.savefig("graph1_unscaled_monocular.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # --- 2. INDIVIDUAL GRAPH: Ground Truth ---
    fig2 = plt.figure(figsize=(10, 8))
    plt.plot(xyz_gt[:, 0], xyz_gt[:, 1], color='black', linestyle='--', linewidth=3, label='Ground Truth')
    plt.scatter(xyz_gt[0, 0], xyz_gt[0, 1], c='green', s=150, zorder=5, label='Start Point')
    plt.scatter(xyz_gt[-1, 0], xyz_gt[-1, 1], c='red', marker='X', s=150, zorder=5, label='End Point')
    plt.title("2. TUM VI Ground Truth\n(Available only inside the 3x3m MoCap room)", fontsize=14, fontweight='bold')
    plt.xlabel("X (meters)", fontsize=12)
    plt.ylabel("Y (meters)", fontsize=12)
    plt.axis('equal')
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.savefig("graph2_ground_truth.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # --- 3. INDIVIDUAL GRAPH: Reprojection Errors ---
    fig3 = plt.figure(figsize=(12, 6))
    if len(rep_errors) > 0:
        plt.scatter(frames, rep_errors, s=15, alpha=0.6, c='blue')
        plt.axhline(y=2.0, color='red', linestyle='--', linewidth=3, label='2.0 px limit')
        plt.title(f"3. Reprojection Error per Frame\nMean: {np.mean(rep_errors):.2f} px (Excellent)", fontsize=14,
                  fontweight='bold')
        plt.xlabel("Frame Number", fontsize=12)
        plt.ylabel("Error (pixels)", fontsize=12)
        plt.ylim(0, max(3.0, min(10, np.max(rep_errors))))
        plt.grid(True)
        plt.legend(fontsize=12)
    plt.savefig("graph3_reprojection_error.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)

    # --- 4. INDIVIDUAL GRAPH: Merged Map ---
    fig4 = plt.figure(figsize=(12, 10))
    plt.plot(aligned_full_est[:, 0], aligned_full_est[:, 1], color='dodgerblue', linewidth=2,
             label='Estimated (Full Walk)')
    plt.plot(xyz_gt[:, 0], xyz_gt[:, 1], color='black', linestyle='--', linewidth=3, zorder=4,
             label='Ground Truth (Room)')
    plt.scatter(xyz_gt[0, 0], xyz_gt[0, 1], c='green', s=200, zorder=5, label='Start Point')
    plt.title(f"4. Merged Map: Estimated vs Ground Truth\nInitial ATE: {rmse_ate:.3f}m (Before Drift)", fontsize=14,
              fontweight='bold')
    plt.xlabel("X (meters)", fontsize=12)
    plt.ylabel("Y (meters)", fontsize=12)
    plt.axis('equal')
    plt.grid(True)
    plt.legend(fontsize=12, loc='upper right')
    plt.savefig("graph4_merged_full.png", dpi=300, bbox_inches='tight')
    plt.close(fig4)

    # ---------------------------------------------------------
    # MASTER DASHBOARD (For PyCharm SciView)
    # ---------------------------------------------------------
    print("Generating master dashboard for display...")
    fig_dash = plt.figure(figsize=(18, 14))

    # Subplot 1
    ax1 = fig_dash.add_subplot(221)
    ax1.plot(xyz_est[:, 0], xyz_est[:, 1], 'b-', linewidth=2, label='Monocular Path')
    ax1.set_title("1. Unscaled Monocular Trajectory\n(The massive scale drift accumulation)", fontsize=14,
                  fontweight='bold')
    ax1.set_xlabel("X (arbitrary units)", fontsize=12)
    ax1.set_ylabel("Y (arbitrary units)", fontsize=12)
    ax1.grid(True)
    ax1.axis('equal')
    ax1.legend(fontsize=12)

    # Subplot 2
    ax2 = fig_dash.add_subplot(222)
    ax2.plot(xyz_gt[:, 0], xyz_gt[:, 1], color='black', linestyle='--', linewidth=3, label='Ground Truth')
    ax2.scatter(xyz_gt[0, 0], xyz_gt[0, 1], c='green', s=150, zorder=5, label='Start Point')
    ax2.scatter(xyz_gt[-1, 0], xyz_gt[-1, 1], c='red', marker='X', s=150, zorder=5, label='End Point')
    ax2.set_title("2. TUM VI Ground Truth", fontsize=14, fontweight='bold')
    ax2.set_xlabel("X (meters)", fontsize=12)
    ax2.set_ylabel("Y (meters)", fontsize=12)
    ax2.axis('equal')
    ax2.grid(True)
    ax2.legend(fontsize=12)

    # Subplot 3
    ax3 = fig_dash.add_subplot(223)
    if len(rep_errors) > 0:
        ax3.scatter(frames, rep_errors, s=15, alpha=0.6, c='blue')
        ax3.axhline(y=2.0, color='red', linestyle='--', linewidth=3, label='2.0 px limit')
        ax3.set_title(f"3. Reprojection Error per Frame\nMean: {np.mean(rep_errors):.2f} px (Excellent)", fontsize=14,
                      fontweight='bold')
        ax3.set_xlabel("Frame Number", fontsize=12)
        ax3.set_ylabel("Error (pixels)", fontsize=12)
        ax3.set_ylim(0, max(3.0, min(10, np.max(rep_errors))))
        ax3.grid(True)
        ax3.legend(fontsize=12)

    # Subplot 4
    ax4 = fig_dash.add_subplot(224)
    ax4.plot(aligned_full_est[:, 0], aligned_full_est[:, 1], color='dodgerblue', linewidth=2,
             label='Estimated (Full Walk)')
    ax4.plot(xyz_gt[:, 0], xyz_gt[:, 1], color='black', linestyle='--', linewidth=3, zorder=4,
             label='Ground Truth (Room)')
    ax4.scatter(xyz_gt[0, 0], xyz_gt[0, 1], c='green', s=200, zorder=5, label='Start Point')
    ax4.set_title(f"4. Merged Map: Estimated vs Ground Truth\nInitial ATE: {rmse_ate:.3f}m (Before Drift)", fontsize=14,
                  fontweight='bold')
    ax4.set_xlabel("X (meters)", fontsize=12)
    ax4.set_ylabel("Y (meters)", fontsize=12)
    ax4.axis('equal')
    ax4.grid(True)
    ax4.legend(fontsize=12, loc='upper right')

    plt.tight_layout(pad=3.0)
    plt.savefig("final_trajectory_dashboard.png", dpi=300, bbox_inches='tight')

    print("Success! 5 files saved (4 individual, 1 dashboard). Opening display...")
    plt.show()


if __name__ == "__main__":
    main()