import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def load_tum(filename):
    """Loads TUM trajectory: timestamp tx ty tz qx qy qz qw"""
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1:4]


def associate(ts_est, xyz_est, ts_gt, xyz_gt, max_diff=0.05):
    """Matches estimated timestamps to Ground Truth timestamps."""
    matched_est, matched_gt = [], []
    for i, t_est in enumerate(ts_est):
        diffs = np.abs(ts_gt - t_est)
        idx = np.argmin(diffs)
        if diffs[idx] < max_diff:
            matched_est.append(xyz_est[i])
            matched_gt.append(xyz_gt[idx])
    return np.array(matched_est).T, np.array(matched_gt).T


def align_se3(model, data):
    """
    RUBRIC VI: SE(3) Alignment for Stereo VO (Metric).
    Finds optimal Rotation and Translation (Rigid Body Alignment).
    """
    mu_M = model.mean(1, keepdims=True)
    mu_D = data.mean(1, keepdims=True)

    model_zero = model - mu_M
    data_zero = data - mu_D

    W = (data_zero @ model_zero.T) / data.shape[1]
    U, S, V_T = np.linalg.svd(W)

    # Ensure a right-handed coordinate system
    S_mat = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(V_T) < 0:
        S_mat[2, 2] = -1

    R = V_T.T @ S_mat @ U.T
    t = mu_M - R @ mu_D
    return R, t


def main():
    # File Paths (Matches the stereo_main.py outputs)
    est_file = "stereo_trajectory.txt"
    err_file = "stereo_reprojection_errors.txt"
    stats_file = "stereo_stats.txt"
    gt_file = r"C:\Users\RoboticsLab\PycharmProjects\stereo_vo_project\dataset\dataset-room2_512_16\mav0\mocap0\data.csv"

    if not os.path.exists(est_file):
        print(f"Error: {est_file} not found! Run stereo_main.py first.")
        return

    # Load Data
    ts_s, xyz_s = load_tum(est_file)
    data_gt = np.loadtxt(gt_file, delimiter=',', skiprows=1)
    ts_g, xyz_g = data_gt[:, 0] / 1e9, data_gt[:, 1:4]

    # SE(3) Alignment (Rotation + Translation)
    mat_s_raw, mat_g_raw = associate(ts_s, xyz_s, ts_g, xyz_g)
    R_align, t_align = align_se3(mat_g_raw, mat_s_raw)
    xyz_s_aligned = (R_align @ xyz_s.T + t_align).T
    est_f, gt_f = associate(ts_s, xyz_s_aligned, ts_g, xyz_g)

    # Compute Metrics
    ate_rmse = np.sqrt(np.mean(np.linalg.norm(gt_f - est_f, axis=0) ** 2))
    step = 10
    rpe_list = [np.abs(np.linalg.norm(gt_f[:, i + step] - gt_f[:, i]) -
                       np.linalg.norm(est_f[:, i + step] - est_f[:, i]))
                for i in range(0, gt_f.shape[1] - step)]
    rpe_trans = np.sqrt(np.mean(np.array(rpe_list) ** 2))

    # Load Stats (Runtime/Failures)
    mean_runtime, failures = (np.loadtxt(stats_file) if os.path.exists(stats_file) else [0, 0])

    # --- THE "CLEAN" PALETTE ---
    c_est = '#1F77B4'  # Vibrant Blue (Thin Path)
    c_gt = '#000000'  # Pure Black (Ground Truth)
    c_mean = '#FF7F0E'  # Bright Orange (Stability Line)
    c_start = '#27AE60'  # Emerald Green
    c_end = '#E74C3C'  # Pomegranate Red

    # 1. INDIVIDUAL: 3D TRAJECTORY MERGED
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(xyz_g[:, 0], xyz_g[:, 1], xyz_g[:, 2], color=c_gt, linestyle='--', linewidth=0.8, alpha=0.5,
             label='GT (Black)')
    ax1.plot(xyz_s_aligned[:, 0], xyz_s_aligned[:, 1], xyz_s_aligned[:, 2], color=c_est, linewidth=1.0,
             label='Stereo VO (Blue)')
    ax1.scatter(xyz_g[0, 0], xyz_g[0, 1], xyz_g[0, 2], color=c_start, s=40, label='Start Point')
    ax1.scatter(xyz_g[-1, 0], xyz_g[-1, 1], xyz_g[-1, 2], color=c_end, marker='X', s=40, label='End Point')
    ax1.set_title("Stereo Trajectory: Metric Reconstruction", fontsize=12, fontweight='bold')
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=9)
    plt.savefig("Stereo_1_Trajectory.png", dpi=300, bbox_inches='tight')

    # 2. INDIVIDUAL: REPROJECTION ERROR
    fig2 = plt.figure(figsize=(10, 5))
    if os.path.exists(err_file):
        err_data = np.loadtxt(err_file)[:, 1]
        valid = err_data > 0
        plt.scatter(range(len(err_data[valid])), err_data[valid], s=5, alpha=0.35, color=c_est, label='Residuals')
        m_err = np.mean(err_data[valid])
        plt.axhline(y=m_err, color=c_mean, linewidth=2.0, label=f'Mean Stability: {m_err:.3f} px')
        plt.ylim(0, 1.5);
        plt.title("Stereo Tracking Consistency", fontsize=12, fontweight='bold')
        plt.ylabel("Error (pixels)");
        plt.xlabel("Frame Sequence")
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False, fontsize=9)
    plt.savefig("Stereo_2_Reprojection.png", dpi=300, bbox_inches='tight')

    # 3. INDIVIDUAL: METRICS TABLE
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.axis('off')
    table_data = [
        ["Stereo Metric", "Computed Value"],
        ["ATE RMSE (Metric)", f"{ate_rmse:.4f} m"],
        ["RPE (Local Drift)", f"{rpe_trans:.4f} m"],
        ["Mean Runtime", f"{mean_runtime:.2f} ms"],
        ["Tracking Failures", f"{int(failures)}"],
        ["Alignment Mode", "SE(3) Rigid"]
    ]
    tbl = ax3.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.5, 0.4])
    tbl.auto_set_font_size(False);
    tbl.set_fontsize(11);
    tbl.scale(1.0, 2.5)
    plt.savefig("Stereo_3_Metrics_Table.png", dpi=300, bbox_inches='tight')

    # 4. MASTER DASHBOARD
    fig_dash = plt.figure(figsize=(20, 10), facecolor='white')
    # Dashboard Left
    ax_d1 = fig_dash.add_subplot(121, projection='3d')
    ax_d1.plot(xyz_g[:, 0], xyz_g[:, 1], xyz_g[:, 2], color=c_gt, linestyle='--', linewidth=0.7, alpha=0.5, label='GT')
    ax_d1.plot(xyz_s_aligned[:, 0], xyz_s_aligned[:, 1], xyz_s_aligned[:, 2], color=c_est, linewidth=0.9,
               label='Stereo VO')
    ax_d1.scatter(xyz_g[0, 0], xyz_g[0, 1], xyz_g[0, 2], color=c_start, s=35, label='Start')
    ax_d1.scatter(xyz_g[-1, 0], xyz_g[-1, 1], xyz_g[-1, 2], color=c_end, marker='X', s=35, label='End')
    ax_d1.set_title("Stereo Trajectory Evaluation", fontsize=13, fontweight='bold')
    ax_d1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=8)

    # Dashboard Right
    ax_d2 = fig_dash.add_subplot(122)
    ax_d2.scatter(range(len(err_data[valid])), err_data[valid], s=4, alpha=0.3, color=c_est)
    ax_d2.axhline(y=m_err, color=c_mean, linewidth=2.0, label=f'Mean: {m_err:.3f} px')
    ax_d2.set_ylim(0, 1.2);
    ax_d2.set_title("Stereo Reprojection Stability", fontsize=13, fontweight='bold')
    ax_d2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=8)

    plt.tight_layout(pad=6.0)
    plt.savefig("STEREO_MASTER_DASHBOARD.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()