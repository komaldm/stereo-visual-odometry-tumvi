import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def load_tum(filename):
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1:4]


def associate(ts_est, xyz_est, ts_gt, xyz_gt, max_diff=0.05):
    matched_est, matched_gt = [], []
    for i, t_est in enumerate(ts_est):
        diffs = np.abs(ts_gt - t_est)
        idx = np.argmin(diffs)
        if diffs[idx] < max_diff:
            matched_est.append(xyz_est[i])
            matched_gt.append(xyz_gt[idx])
    return np.array(matched_est).T, np.array(matched_gt).T


def align_sim3(model, data):
    mu_M = model.mean(1, keepdims=True);
    mu_D = data.mean(1, keepdims=True)
    model_zero = model - mu_M;
    data_zero = data - mu_D
    C = (data_zero @ model_zero.T) / data.shape[1]
    U, S, V_T = np.linalg.svd(C)
    R_mat = V_T.T @ U.T
    if np.linalg.det(R_mat) < 0: V_T[2, :] *= -1; R_mat = V_T.T @ U.T
    var_data = np.mean(np.sum(data_zero ** 2, axis=0))
    scale = (1.0 / var_data) * np.trace(np.diag(S))
    t_vec = mu_M - scale * (R_mat @ mu_D)
    return scale, R_mat, t_vec


def main():
    est_file, err_file = "monocular_trajectory.txt", "reprojection_errors.txt"
    gt_file = r"C:\Users\RoboticsLab\PycharmProjects\stereo_vo_project\dataset\dataset-room2_512_16\mav0\mocap0\data.csv"
    stats_file = "monocular_stats.txt"

    # Load Data
    ts_s, xyz_s = load_tum(est_file)
    data_gt = np.loadtxt(gt_file, delimiter=',', skiprows=1)
    ts_g, xyz_g = data_gt[:, 0] / 1e9, data_gt[:, 1:4]

    scale, R, t = align_sim3(*associate(ts_s, xyz_s, ts_g, xyz_g)[::-1])
    xyz_s_aligned = (scale * R @ xyz_s.T + t).T
    est_f, gt_f = associate(ts_s, xyz_s_aligned, ts_g, xyz_g)

    # Metrics
    ate_rmse = np.sqrt(np.mean(np.linalg.norm(gt_f - est_f, axis=0) ** 2))
    mean_runtime, failures = (np.loadtxt(stats_file) if os.path.exists(stats_file) else [0, 0])

    # --- THE "CLEAN" PALETTE ---
    c_est = '#1F77B4'  # Vibrant Blue (Visible & Bright)
    c_gt = '#000000'  # Pure Black
    c_mean = '#FF7F0E'  # Bright Orange
    c_start = '#27AE60'  # Green
    c_end = '#E74C3C'  # Red

    # 1. INDIVIDUAL: 3D TRAJECTORY
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(xyz_g[:, 0], xyz_g[:, 1], xyz_g[:, 2], color=c_gt, linestyle='--', linewidth=0.8, alpha=0.6,
             label='Ground Truth (TUM)')
    ax1.plot(xyz_s_aligned[:, 0], xyz_s_aligned[:, 1], xyz_s_aligned[:, 2], color=c_est, linewidth=1.0,
             label='Estimated VO Path')
    ax1.scatter(xyz_g[0, 0], xyz_g[0, 1], xyz_g[0, 2], color=c_start, s=40, label='Start Point', zorder=10)
    ax1.scatter(xyz_g[-1, 0], xyz_g[-1, 1], xyz_g[-1, 2], color=c_end, marker='X', s=40, label='End Point', zorder=10)
    ax1.set_title("3D Trajectory Reconstruction", fontsize=12, fontweight='bold')
    ax1.set_xlabel("X (m)");
    ax1.set_ylabel("Y (m)");
    ax1.set_zlabel("Z (m)")
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False, fontsize=9)
    plt.savefig("1_Trajectory_3D.png", dpi=300, bbox_inches='tight')

    # 2. INDIVIDUAL: REPROJECTION ERROR
    fig2 = plt.figure(figsize=(10, 5))
    if os.path.exists(err_file):
        err_data = np.loadtxt(err_file)[:, 1]
        valid = err_data > 0
        plt.scatter(range(len(err_data[valid])), err_data[valid], s=5, alpha=0.4, color=c_est,
                    label='Per-Frame Residuals')
        m_err = np.mean(err_data[valid])
        plt.axhline(y=m_err, color=c_mean, linewidth=2.0, label=f'Mean Stability: {m_err:.3f} px')
        plt.ylim(0, 1.2);
        plt.title("Visual Tracking Consistency", fontsize=12, fontweight='bold')
        plt.xlabel("Frame Sequence");
        plt.ylabel("Error (pixels)")
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False, fontsize=9)
    plt.savefig("2_Reprojection_Error.png", dpi=300, bbox_inches='tight')

    # 3. INDIVIDUAL: METRICS TABLE
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.axis('off')
    table_data = [
        ["Metric Category", "Computed Value"],
        ["ATE RMSE (Global)", f"{ate_rmse:.4f} m"],
        ["Mean Runtime", f"{mean_runtime:.2f} ms"],
        ["Tracking Failures", f"{int(failures)}"],
        ["Sim3 Scale Factor", f"{scale:.6f}"]
    ]
    tbl = ax3.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.5, 0.4])
    tbl.auto_set_font_size(False);
    tbl.set_fontsize(11);
    tbl.scale(1.0, 2.5)
    plt.savefig("3_Metrics_Table.png", dpi=300, bbox_inches='tight')

    # 4. MASTER DASHBOARD
    fig_dash = plt.figure(figsize=(20, 10), facecolor='white')

    # Left: 3D
    ax_d1 = fig_dash.add_subplot(121, projection='3d')
    ax_d1.plot(xyz_g[:, 0], xyz_g[:, 1], xyz_g[:, 2], color=c_gt, linestyle='--', linewidth=0.7, alpha=0.5,
               label='GT (Black)')
    ax_d1.plot(xyz_s_aligned[:, 0], xyz_s_aligned[:, 1], xyz_s_aligned[:, 2], color=c_est, linewidth=0.9,
               label='VO Path (Blue)')
    ax_d1.scatter(xyz_g[0, 0], xyz_g[0, 1], xyz_g[0, 2], color=c_start, s=35, label='Start')
    ax_d1.scatter(xyz_g[-1, 0], xyz_g[-1, 1], xyz_g[-1, 2], color=c_end, marker='X', s=35, label='End')
    ax_d1.set_title("Trajectory Evaluation", fontsize=13, fontweight='bold')
    ax_d1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False, fontsize=8)

    # Right: Error
    ax_d2 = fig_dash.add_subplot(122)
    ax_d2.scatter(range(len(err_data[valid])), err_data[valid], s=4, alpha=0.35, color=c_est, label='Frame Residuals')
    ax_d2.axhline(y=m_err, color=c_mean, linewidth=2.0, label=f'Mean: {m_err:.3f} px')
    ax_d2.set_ylim(0, 1.0);
    ax_d2.set_title("Reprojection Stability", fontsize=13, fontweight='bold')
    ax_d2.set_xlabel("Time (Frame Sequence)");
    ax_d2.set_ylabel("Residual (px)")
    ax_d2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=8)

    plt.tight_layout(pad=6.0)
    plt.savefig("MASTER_DASHBOARD.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__": main()