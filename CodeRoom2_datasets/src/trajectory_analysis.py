import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11


# ==========================================
# DATA LOADING & ALIGNMENT MATH
# ==========================================
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
    mu_M = model.mean(1, keepdims=True)
    mu_D = data.mean(1, keepdims=True)
    model_zero = model - mu_M
    data_zero = data - mu_D
    C = (data_zero @ model_zero.T) / data.shape[1]
    U, S, V_T = np.linalg.svd(C)
    R_mat = V_T.T @ U.T
    if np.linalg.det(R_mat) < 0:
        V_T[2, :] *= -1
        R_mat = V_T.T @ U.T
    var_data = np.mean(np.sum(data_zero ** 2, axis=0))
    scale = (1.0 / var_data) * np.trace(np.diag(S))
    t_vec = mu_M - scale * (R_mat @ mu_D)
    return scale, R_mat, t_vec


def compute_rpe(xyz_gt, xyz_est, delta=1):
    errors = []
    for i in range(len(xyz_gt) - delta):
        delta_gt = xyz_gt[i + delta] - xyz_gt[i]
        delta_est = xyz_est[i + delta] - xyz_est[i]
        errors.append(np.linalg.norm(delta_gt - delta_est))
    return np.sqrt(np.mean(np.array(errors) ** 2)) if errors else 0.0


# ==========================================
# MAIN EXECUTION & PLOTTING
# ==========================================
def main():
    # --- DYNAMIC RELATIVE PATHS (GitHub Ready) ---
    # Dynamically locate the main project root folder relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Construct the path to your results folder
    base_dir = os.path.join(project_root, "Results_Room2_Dataset")

    est_file = os.path.join(base_dir, "monocular_trajectory.txt")
    err_file = os.path.join(base_dir, "monocular_reprojection_errors.txt")
    kf_file = os.path.join(base_dir, "monocular_keyframes.txt")
    stats_file = os.path.join(base_dir, "monocular_stats.txt")
    seeds_file = os.path.join(base_dir, "monocular_seeds.txt")

    # Construct the path to the Ground Truth dataset CSV
    gt_file = os.path.join(project_root, "dataset", "dataset-room2_512_16", "mav0", "mocap0", "data.csv")
    # 1. Load Data
    ts_s, xyz_s = load_tum(est_file)
    data_gt = np.loadtxt(gt_file, delimiter=',', comments='#')
    ts_g = data_gt[:, 0]
    if len(ts_g) > 0 and ts_g[0] > 1e12:
        ts_g = ts_g / 1e9
    xyz_g = data_gt[:, 1:4]

    err_data = None
    mean_rep_error_val = 0.0
    if os.path.exists(err_file):
        err_raw = np.loadtxt(err_file)
        if len(err_raw) > 0:
            err_data = err_raw[:, 1]
            valid_err = err_data[err_data > 0]
            if len(valid_err) > 0:
                mean_rep_error_val = np.mean(valid_err)

    # 2. Align & Compute Rubric Metrics
    xyz_s_matched, xyz_g_matched = associate(ts_s, xyz_s, ts_g, xyz_g)
    scale, R, t = align_sim3(xyz_g_matched, xyz_s_matched)
    xyz_s_aligned = (scale * R @ xyz_s_matched + t).T
    xyz_g_aligned = xyz_g_matched.T

    ate_rmse = np.sqrt(np.mean(np.linalg.norm(xyz_g_aligned - xyz_s_aligned, axis=1) ** 2))
    rpe_rmse = compute_rpe(xyz_g_aligned, xyz_s_aligned, delta=1)
    start_end_drift = np.linalg.norm(xyz_g_aligned[-1] - xyz_s_aligned[-1])
    mean_runtime, failures = (np.loadtxt(stats_file) if os.path.exists(stats_file) else [0, 0])

    # Calculate Success Rate
    total_frames = len(ts_s)
    success_rate = ((total_frames - failures) / total_frames * 100.0) if total_frames > 0 else 0.0

    # ---------------------------------------------------------
    # NEW METRICS COMPUTATION BLOCK
    # ---------------------------------------------------------
    # A. Total Path Length (Calculated natively along the Ground Truth path)
    delta_gt = np.diff(xyz_g_aligned, axis=0)
    total_path_length = np.sum(np.linalg.norm(delta_gt, axis=1))

    # B. Drift Percentage
    drift_percentage = (start_end_drift / total_path_length * 100.0) if total_path_length > 0 else 0.0

    # C. MTTR (Mean Time To Recovery) & Max Time Blind
    recovery_times = []
    current_fail_start_ts = None

    if os.path.exists(kf_file):
        with open(kf_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        frame_idx = int(parts[0])
                        status = parts[1]
                    except ValueError:
                        continue

                    # Classify if tracking was dropped or is currently recovering raw parallax state
                    is_fail = any(x in status for x in ["FAIL", "LOST", "WAITING", "REJECTED"])

                    if is_fail:
                        if current_fail_start_ts is None and frame_idx < len(ts_s):
                            current_fail_start_ts = ts_s[frame_idx]
                    else:
                        if current_fail_start_ts is not None and frame_idx < len(ts_s):
                            fail_end_ts = ts_s[frame_idx]
                            recovery_times.append(fail_end_ts - current_fail_start_ts)
                            current_fail_start_ts = None

        if current_fail_start_ts is not None and len(ts_s) > 0:
            recovery_times.append(ts_s[-1] - current_fail_start_ts)

    mttr_val = np.mean(recovery_times) if len(recovery_times) > 0 else 0.0
    max_time_blind_val = np.max(recovery_times) if len(recovery_times) > 0 else 0.0

    # D. Seed Tracking Statistics (Parsed from monocular_seeds.txt)
    seed_counts = []
    if os.path.exists(seeds_file):
        try:
            seed_data = np.loadtxt(seeds_file)
            if seed_data.ndim == 2 and seed_data.shape[1] >= 2:
                seed_counts = seed_data[:, 1]
            elif seed_data.ndim == 1 and len(seed_data) >= 2:
                seed_counts = [seed_data[1]]
        except Exception:
            pass

    total_seeds = int(np.sum(seed_counts)) if len(seed_counts) > 0 else 0
    mean_seeds = float(np.mean(seed_counts)) if len(seed_counts) > 0 else 0.0
    std_seeds = float(np.std(seed_counts)) if len(seed_counts) > 0 else 0.0

    # ==========================================
    # COLOR PALETTE (Clean, Professional)
    # ==========================================
    c_est = '#2980B9'  # Strong Blue for Estimated Path
    c_gt = '#7F8C8D'  # Smooth Slate/Gray for Ground Truth
    c_mean = '#27AE60'  # Emerald Green for Mean line
    c_limit = '#E74C3C'  # Crimson Red for QC Threshold Limit

    # ---------------------------------------------------------
    # Figure 1: METRICS TABLE (Fully Combined Layout)
    # ---------------------------------------------------------
    fig_table, ax_table = plt.subplots(figsize=(9, 6.5))
    ax_table.axis('off')

    table_data = [
        ["Absolute Trajectory Error (ATE)", f"{ate_rmse:.4f} m"],
        ["Relative Pose Error (RPE)", f"{rpe_rmse:.4f} m/frame"],
        ["Start-End Drift", f"{start_end_drift:.4f} m"],
        ["Total Path Length", f"{total_path_length:.4f} m"],
        ["Drift Percentage", f"{drift_percentage:.2f} %"],
        ["Mean Time To Recovery (MTTR)", f"{mttr_val:.3f} s"],
        ["Max Time Blind", f"{max_time_blind_val:.3f} s"],
        ["Total Count of Seeds", f"{total_seeds}"],
        ["Mean Seed Count", f"{mean_seeds:.2f}"],
        ["Seed Standard Deviation", f"{std_seeds:.2f}"],
        ["Mean per-frame Runtime", f"{mean_runtime:.2f} Hz"],
        ["Tracking Failures", f"{int(failures)}"],
        ["Tracking Success Rate", f"{success_rate:.2f} %"],
        ["Mean Reprojection Error", f"{mean_rep_error_val:.3f} px"],
        ["Sim(3) Scale Factor", f"{scale:.6f}"]
    ]

    headers = ["Evaluation Metric", "Computed Outcome"]

    tbl = ax_table.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='left',
        colWidths=[0.65, 0.35]
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.8)

    # Styled cleanly to perfectly mirror a LaTeX publication template output
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#D1D5DB')
        cell.PAD = 0.05
        if row == 0:
            cell.set_facecolor('#1F2937')
            cell.set_text_props(color='white', weight='bold', size=12, ha='center')
        else:
            if row % 2 == 0:
                cell.set_facecolor('#F9FAFB')
            else:
                cell.set_facecolor('#FFFFFF')
            if col == 1:
                cell.set_text_props(weight='bold', color='#111827')

    ax_table.set_title("Table 1: Monocular VO Comprehensive Evaluation Metrics", fontweight='bold', pad=10)
    plt.savefig(os.path.join(base_dir, "1_Metrics_Table1.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # Figure 2: REPROJECTION ERROR
    # ---------------------------------------------------------
    if err_data is not None:
        valid_err = err_data[err_data > 0]
        mean_err = np.mean(valid_err)
        QC_LIMIT = 5.0

        fig_err, ax_err = plt.subplots(figsize=(10, 6))
        ax_err.scatter(range(len(valid_err)), valid_err, s=15, alpha=0.5, color=c_est, edgecolor='none',
                       label='Per-Frame RMSE')
        ax_err.axhline(y=mean_err, color=c_mean, linewidth=2.5, linestyle='-', label=f'Mean Error ({mean_err:.3f} px)')
        ax_err.axhline(y=QC_LIMIT, color=c_limit, linewidth=2.0, linestyle='--',
                       label=f'QC Threshold Limit ({QC_LIMIT} px)')

        ax_err.set_title("Visual Tracking Consistency (Reprojection Error)", fontweight='bold', pad=15)
        ax_err.set_xlabel("Time Sequence (Tracked Frames)")
        ax_err.set_ylabel("Reprojection RMSE (pixels)")
        ax_err.grid(True, linestyle='--', alpha=0.4)
        max_y = max(np.max(valid_err), QC_LIMIT) * 1.2
        ax_err.set_ylim(0, max_y)
        ax_err.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, shadow=True)
        plt.subplots_adjust(right=0.7)
        plt.savefig(os.path.join(base_dir, "2_Reprojection_Error1.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # ---------------------------------------------------------
    # Figure 3: GROUND TRUTH ONLY
    # ---------------------------------------------------------
    fig_gt = plt.figure(figsize=(10, 8))
    ax_gt = fig_gt.add_subplot(111, projection='3d')
    ax_gt.plot(xyz_g_aligned[:, 0], xyz_g_aligned[:, 1], xyz_g_aligned[:, 2], color=c_gt, linewidth=2.5,
               label='Ground Truth')
    ax_gt.scatter(xyz_g_aligned[0, 0], xyz_g_aligned[0, 1], xyz_g_aligned[0, 2], color='green', s=100, zorder=5,
                  label='Start')
    ax_gt.scatter(xyz_g_aligned[-1, 0], xyz_g_aligned[-1, 1], xyz_g_aligned[-1, 2], color='red', marker='X', s=100,
                  zorder=5, label='End')
    ax_gt.set_title("Mocap Ground Truth Reference", fontweight='bold', pad=20)
    ax_gt.set_xlabel("X Position (m)")
    ax_gt.set_ylabel("Y Position (m)")
    ax_gt.set_zlabel("Z Position (m)")
    ax_gt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), frameon=True, shadow=True)
    plt.subplots_adjust(right=0.75)
    plt.savefig(os.path.join(base_dir, "3_Ground_Truth_Trajectory1.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # Figure 4: ESTIMATED VS GROUND TRUTH (Static Comparison)
    # ---------------------------------------------------------
    fig_comp = plt.figure(figsize=(10, 8))
    ax_comp = fig_comp.add_subplot(111, projection='3d')
    ax_comp.plot(xyz_g_aligned[:, 0], xyz_g_aligned[:, 1], xyz_g_aligned[:, 2], color=c_gt, linestyle='--',
                 linewidth=1.5, alpha=0.8, label='Ground Truth')
    ax_comp.plot(xyz_s_aligned[:, 0], xyz_s_aligned[:, 1], xyz_s_aligned[:, 2], color=c_est, linewidth=2.5,
                 label='Estimated VO Path')
    ax_comp.scatter(xyz_g_aligned[0, 0], xyz_g_aligned[0, 1], xyz_g_aligned[0, 2], color='green', s=100, zorder=5,
                    label='Start')
    ax_comp.scatter(xyz_g_aligned[-1, 0], xyz_g_aligned[-1, 1], xyz_g_aligned[-1, 2], color='red', marker='X', s=100,
                    zorder=5, label='End')
    ax_comp.set_title("Trajectory Comparison (Sim3 Aligned)", fontweight='bold', pad=20)
    ax_comp.set_xlabel("X (m)")
    ax_comp.set_ylabel("Y (m)")
    ax_comp.set_zlabel("Z (m)")
    ax_comp.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), frameon=True, shadow=True)
    plt.subplots_adjust(right=0.75)
    plt.savefig(os.path.join(base_dir, "4_Estimated_vs_GT1.png"), dpi=300, bbox_inches='tight')

    # =========================================================
    # UPGRADE 4a: AUTOMATED ROTATING ANIMATION (GIF EXPORT)
    # =========================================================
    print("Generating 360-degree rotating GIF animation... (This may take a moment)")

    def update_view(frame):
        ax_comp.view_init(elev=25, azim=frame * 2)
        return fig_comp,

    anim = FuncAnimation(fig_comp, update_view, frames=180, interval=50, blit=False)
    anim.save(os.path.join(base_dir, "4a_Rotating_Trajectory.gif"), dpi=150, writer='pillow')
    plt.close(fig_comp)

    # =========================================================
    # UPGRADE 4b: FULLY INTERACTIVE HTML GRAPH (PLOTLY EXPORT)
    # =========================================================
    print("Exporting fully interactive 3D HTML trajectory visualizer...")
    fig_plotly = go.Figure()
    fig_plotly.add_trace(go.Scatter3d(
        x=xyz_g_aligned[:, 0], y=xyz_g_aligned[:, 1], z=xyz_g_aligned[:, 2],
        mode='lines', line=dict(color=c_gt, width=3, dash='dash'), name='Ground Truth'
    ))
    fig_plotly.add_trace(go.Scatter3d(
        x=xyz_s_aligned[:, 0], y=xyz_s_aligned[:, 1], z=xyz_s_aligned[:, 2],
        mode='lines', line=dict(color=c_est, width=5), name='Estimated VO Path'
    ))
    fig_plotly.add_trace(go.Scatter3d(
        x=[xyz_g_aligned[0, 0]], y=[xyz_g_aligned[0, 1]], z=[xyz_g_aligned[0, 2]],
        mode='markers', marker=dict(color='green', size=8), name='Start Node'
    ))
    fig_plotly.add_trace(go.Scatter3d(
        x=[xyz_g_aligned[-1, 0]], y=[xyz_g_aligned[-1, 1]], z=[xyz_g_aligned[-1, 2]],
        mode='markers', marker=dict(color='red', size=8, symbol='x'), name='End Node'
    ))
    fig_plotly.update_layout(
        title=dict(text="Interactive 3D Trajectory Debugger (Sim3 Aligned)", font=dict(size=18)),
        scene=dict(xaxis_title='X Axis (m)', yaxis_title='Y Axis (m)', zaxis_title='Z Axis (m)', aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=50), legend=dict(x=0.02, y=0.98)
    )
    fig_plotly.write_html(os.path.join(base_dir, "4b_Interactive_Trajectory.html"))

    # ---------------------------------------------------------
    # Figure 5: ESTIMATED TRAJECTORY ONLY
    # ---------------------------------------------------------
    fig_est = plt.figure(figsize=(10, 8))
    ax_est = fig_est.add_subplot(111, projection='3d')
    ax_est.plot(xyz_s_aligned[:, 0], xyz_s_aligned[:, 1], xyz_s_aligned[:, 2], color=c_est, linewidth=2.5,
                label='Estimated VO Path')
    ax_est.scatter(xyz_s_aligned[0, 0], xyz_s_aligned[0, 1], xyz_s_aligned[0, 2], color='green', s=100, zorder=5,
                   label='Start Point')
    ax_est.scatter(xyz_s_aligned[-1, 0], xyz_s_aligned[-1, 1], xyz_s_aligned[-1, 2], color='red', marker='X', s=100,
                   zorder=5, label='End Point')

    ax_est.set_title("Standalone Estimated Trajectory (Aligned)", fontweight='bold', pad=20)
    ax_est.set_xlabel("X Position (m)", labelpad=10)
    ax_est.set_ylabel("Y Position (m)", labelpad=10)
    ax_est.set_zlabel("Z Position (m)", labelpad=10)
    ax_est.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), frameon=True, shadow=True)
    plt.subplots_adjust(right=0.75)
    plt.savefig(os.path.join(base_dir, "5_Estimated_Only1.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # Figure 6: KEYFRAME EVOLUTION
    # ---------------------------------------------------------
    if os.path.exists(kf_file):
        print("Generating Keyframe Evolution graph...")
        frames = []
        cumulative_keyframes = []
        current_kf_count = 0

        with open(kf_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    frame_id = int(parts[0])
                    status = parts[1]
                except ValueError:
                    continue

                if "KEYFRAME" in status or "METRIC_STABILIZED_KF" in status:
                    current_kf_count += 1

                frames.append(frame_id)
                cumulative_keyframes.append(current_kf_count)

        if frames:
            plt.figure(figsize=(8, 5), dpi=300)
            plt.plot(frames, cumulative_keyframes, label="Proposed Pipeline", color=c_est, linewidth=2.0)
            plt.title("Evolution of Keyframes in Local Map", fontsize=14, fontweight='bold', pad=15)
            plt.xlabel("Time Sequence (Frames)", fontsize=12)
            plt.ylabel("Cumulative Keyframes", fontsize=12)
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            plt.legend(loc="lower right", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(base_dir, "6_Keyframe_Evolution1.png"), bbox_inches='tight')
            plt.close()
        else:
            print("Keyframe log was found, but it contained no valid data points.")
    else:
        print(f"Keyframe log not found at {kf_file}. Skipping Figure 6.")

    print("Successfully generated all static outputs, animated GIF, and full HTML interactivity.")


if __name__ == "__main__":
    main()