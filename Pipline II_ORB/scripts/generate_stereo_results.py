from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


def apply_reference_style() -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11


def load_tum_trajectory(file_path: Path):
    data = np.loadtxt(file_path, delimiter=None, comments="#")
    times = data[:, 0]
    positions = data[:, 1:4]
    quats = data[:, 4:8]
    return times, positions, quats


def align_se3(model_pts: np.ndarray, data_pts: np.ndarray):
    model_centroid = np.mean(model_pts, axis=0)
    data_centroid = np.mean(data_pts, axis=0)
    model_centered = model_pts - model_centroid
    data_centered = data_pts - data_centroid
    H = model_centered.T @ data_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = data_centroid - R @ model_centroid
    aligned_model = (R @ model_pts.T).T + t
    return aligned_model, R, t


def compute_comprehensive_metrics(aligned_vo: np.ndarray, gt_pts: np.ndarray):
    errors = np.linalg.norm(aligned_vo - gt_pts, axis=1)
    ate_rmse = np.sqrt(np.mean(errors**2))
    start_to_end_drift = errors[-1]
    delta_vo = np.diff(aligned_vo, axis=0)
    delta_gt = np.diff(gt_pts, axis=0)
    rpe_errors = np.linalg.norm(delta_vo - delta_gt, axis=1)
    rpe_rmse = np.sqrt(np.mean(rpe_errors**2)) if len(rpe_errors) else 0.0
    return ate_rmse, rpe_rmse, start_to_end_drift, errors


def load_gt(gt_path: Path):
    data_gt = np.loadtxt(gt_path, delimiter=",", comments="#")
    t_gt = data_gt[:, 0]
    if len(t_gt) > 0 and t_gt[0] > 1e12:
        t_gt = t_gt / 1e9
    poses_gt = data_gt[:, 1:4]
    return t_gt, poses_gt


def synchronize(t_vo: np.ndarray, poses_vo: np.ndarray, t_gt: np.ndarray, poses_gt: np.ndarray, max_diff: float):
    aligned_gt_poses = []
    valid_vo_poses = []
    valid_times = []
    for i, t in enumerate(t_vo):
        idx = np.argmin(np.abs(t_gt - t))
        if np.abs(t_gt[idx] - t) < max_diff:
            aligned_gt_poses.append(poses_gt[idx])
            valid_vo_poses.append(poses_vo[i])
            valid_times.append(t)
    return np.array(valid_times), np.array(valid_vo_poses), np.array(aligned_gt_poses)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_frame_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def float_or_nan(value) -> float:
    try:
        if value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def compute_path_length(poses: np.ndarray) -> float:
    if len(poses) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(poses, axis=0), axis=1)))


def compute_seed_metrics(frame_log: list[dict], count_col: str, n_total_frames: int):
    """Returns (total_seeds, mean_seeds, std_seeds) from positive deltas in count_col."""
    if not frame_log:
        return 0, 0.0, 0.0
    counts = np.array([float_or_nan(r.get(count_col, "")) for r in frame_log])
    finite = counts[np.isfinite(counts)]
    if len(finite) == 0:
        return 0, 0.0, 0.0
    deltas = np.diff(finite)
    total = int(finite[0]) + int(np.sum(deltas[deltas > 0]))
    mean_val = total / max(n_total_frames, 1)
    std_val = float(np.std(finite))
    return total, mean_val, std_val


def resolve_gt_path(dataset: str) -> Path:
    if dataset == "corridor3":
        return (
            PROJECT_ROOT
            / "dataset-corridor3_512_16"
            / "mav0" / "mocap0" / "data.csv"
        )
    if dataset == "outdoors5":
        return (
            PROJECT_ROOT
            / "dataset-outdoors5_512_16"
            / "mav0" / "mocap0" / "data.csv"
        )
    return (
        PROJECT_ROOT
        / "dataset-room2_512_16"
        / "mav0" / "mocap0" / "data.csv"
    )


def infer_dataset(tag: str, traj: Path) -> str:
    combined = (tag + str(traj)).lower()
    if "corridor3" in combined:
        return "corridor3"
    if "outdoors5" in combined:
        return "outdoors5"
    return "room2"


def render_11_metric_table(metrics_dict: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.0))
    ax.axis("off")
    table_data = [
        ["Start-to-End Drift (E_drift)", f"{metrics_dict['E_drift']:.4f} m"],
        ["Total Path Length", f"{metrics_dict['Path Length']:.2f} m"],
        ["Drift Percentage", f"{metrics_dict['Drift %']:.2f}%"],
        ["Total Count of Seeds", f"{metrics_dict['Total Seeds']}"],
        ["Mean Seed Count", f"{metrics_dict['Mean Seeds']:.2f}"],
        ["Seed Standard Deviation", f"{metrics_dict['Seed Std']:.2f}"],
        ["Tracking Failures (Recoveries)", f"{metrics_dict['Tracking Failures']}"],
        ["Mean Reprojection Error", f"{metrics_dict['Mean Reproj']:.4f} px"],
        ["Mean Per-Frame Runtime", metrics_dict["Runtime"]],
    ]
    tbl = ax.table(
        cellText=table_data,
        colLabels=["Evaluation Metric", "Recorded Outcome"],
        cellLoc="center",
        loc="center",
        colWidths=[0.62, 0.38],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.0, 1.9)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        if row == 0:
            cell.set_facecolor("#1F2937")
            cell.set_text_props(color="white", weight="bold", size=12)
        elif row % 2 == 0:
            cell.set_facecolor("#F9FAFB")
    ax.set_title(
        f"Stereo VO Evaluation Metrics — {metrics_dict['Sequence Name']}",
        fontweight="bold", pad=20,
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def render_keyframe_evolution(frame_log: list[dict], output_path: Path) -> None:
    if frame_log:
        frames = np.array([int(r["frame_idx"]) for r in frame_log])
        cumulative = np.arange(1, len(frames) + 1)
    else:
        frames = np.arange(1)
        cumulative = np.ones(1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(frames, cumulative, where="post", color="#2980B9", linewidth=2.5)
    ax.set_title("Evolution of Keyframes in Local Metric Map", fontweight="bold", pad=15)
    ax.set_xlabel("Time Sequence Frame")
    ax.set_ylabel("Cumulative Keyframes")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def render_four_panel(
    aligned_vo: np.ndarray,
    aligned_gt: np.ndarray,
    valid_times: np.ndarray,
    point_errors: np.ndarray,
    rep_times: np.ndarray,
    rep_errors: np.ndarray,
    output_path: Path,
    full_aligned_vo: np.ndarray | None = None,
    full_gt: np.ndarray | None = None,
) -> None:
    relative_time = valid_times - valid_times[0]
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    # Full trajectories as faint background — critical for datasets where GT covers only a
    # fraction of the sequence (e.g. corridor3 has a 230-second mocap gap: 1200 / 5802 matches).
    if full_gt is not None:
        plt.plot(full_gt[:, 0], full_gt[:, 1], "g-", linewidth=0.8, alpha=0.30, zorder=1)
    if full_aligned_vo is not None:
        plt.plot(full_aligned_vo[:, 0], full_aligned_vo[:, 1], "b--", linewidth=0.8, alpha=0.30, zorder=1)
    # Matched pairs (used for metric computation) drawn on top in solid style
    plt.plot(aligned_gt[:, 0], aligned_gt[:, 1], "g-", label="Ground Truth", linewidth=2, zorder=2)
    plt.plot(aligned_vo[:, 0], aligned_vo[:, 1], "b--", label="Stereo VO", linewidth=1.5, zorder=2)
    plt.scatter(aligned_gt[0, 0], aligned_gt[0, 1], c="green", marker="o", s=100, label="GT start", zorder=3)
    plt.scatter(aligned_gt[-1, 0], aligned_gt[-1, 1], c="red", marker="D", s=100, label="GT end", zorder=3)
    plt.scatter(aligned_vo[0, 0], aligned_vo[0, 1], c="blue", marker="x", s=120, linewidths=2, label="VO start", zorder=3)
    plt.scatter(aligned_vo[-1, 0], aligned_vo[-1, 1], c="magenta", marker="x", s=100, label="VO end", zorder=3)
    plt.title("Top-Down Trajectory Overlay (X-Y Plane)")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    plt.subplot(2, 2, 2)
    plt.plot(relative_time, aligned_gt[:, 2], "g-", label="Ground Truth Z")
    plt.plot(relative_time, aligned_vo[:, 2], "b--", label="Stereo VO Z")
    plt.title("Vertical Drift Evaluation (Altitude over Time)")
    plt.xlabel("Time (s)")
    plt.ylabel("Z Altitude (m)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    ate_rmse = float(np.sqrt(np.mean(point_errors**2)))
    plt.plot(relative_time, point_errors, "r-", linewidth=1.5)
    plt.axhline(ate_rmse, color="black", linestyle=":", label=f"ATE RMSE ({ate_rmse:.3f} m)")
    plt.title("Absolute Trajectory Drift (ATE) per Frame")
    plt.xlabel("Time (s)")
    plt.ylabel("Positional Error (m)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    if len(rep_errors) > 0:
        rel_t_err = rep_times - rep_times[0]
        plt.plot(rel_t_err, rep_errors, color="orange", linewidth=1)
        plt.title("Frame-by-Frame Reprojection Error")
        plt.xlabel("Time (s)")
        plt.ylabel("Error (pixel)")
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, "Reprojection Error File Not Found", ha="center", va="center")
        plt.title("Reprojection Stability")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def render_animated_trajectory_gif(aligned_vo: np.ndarray, aligned_gt: np.ndarray, output_gif_path: Path) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    max_range = np.array(
        [
            aligned_gt[:, 0].max() - aligned_gt[:, 0].min(),
            aligned_gt[:, 1].max() - aligned_gt[:, 1].min(),
            aligned_gt[:, 2].max() - aligned_gt[:, 2].min(),
        ]
    ).max() / 2.0
    max_range = max(max_range, 0.5)
    mid_x = (aligned_gt[:, 0].max() + aligned_gt[:, 0].min()) / 2.0
    mid_y = (aligned_gt[:, 1].max() + aligned_gt[:, 1].min()) / 2.0
    mid_z = (aligned_gt[:, 2].max() + aligned_gt[:, 2].min()) / 2.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_title("Stereo VO vs Ground Truth Animation")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z Altitude (m)")
    line_gt, = ax.plot([], [], [], "g-", label="Ground Truth", linewidth=2)
    line_vo, = ax.plot([], [], [], "b--", label="Stereo VO", linewidth=1.5)
    ax.legend(loc="upper left")
    total_frames = len(aligned_vo)
    step = max(1, total_frames // 150)
    indices = list(range(0, total_frames, step))
    if indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)

    def update(frame_idx):
        idx = indices[frame_idx]
        line_gt.set_data(aligned_gt[:idx, 0], aligned_gt[:idx, 1])
        line_gt.set_3d_properties(aligned_gt[:idx, 2])
        line_vo.set_data(aligned_vo[:idx, 0], aligned_vo[:idx, 1])
        line_vo.set_3d_properties(aligned_vo[:idx, 2])
        ax.view_init(elev=25, azim=frame_idx * (360 / len(indices)))
        return line_gt, line_vo

    anim = animation.FuncAnimation(fig, update, frames=len(indices), interval=50, blit=False)
    anim.save(output_gif_path, writer=animation.PillowWriter(fps=20))
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=Path, default=PROJECT_ROOT / "outputs" / "trajectories" / "room2_stereo_vo_final.txt")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None, help="room2 | corridor3 (auto-inferred from tag/path if omitted)")
    parser.add_argument("--gt", type=Path, default=None, help="GT CSV path; resolved from --dataset if omitted")
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--frame-log", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_png" / "stereo")
    parser.add_argument("--max-dt", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    apply_reference_style()
    args = parse_args()
    traj_path = args.traj if args.traj.is_absolute() else PROJECT_ROOT / args.traj
    tag = args.tag or traj_path.stem
    dataset = args.dataset or infer_dataset(tag, traj_path)
    gt_path = args.gt or resolve_gt_path(dataset)
    metrics_path = args.metrics or PROJECT_ROOT / "outputs" / "metrics" / f"{tag}_metrics.json"
    summary_path = args.summary or PROJECT_ROOT / "outputs" / "metrics" / f"{tag}_run_summary.json"
    frame_log_path = args.frame_log or PROJECT_ROOT / "outputs" / "logs" / f"{tag}_frame_log.csv"
    out_dir = args.output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    t_vo, poses_vo, _ = load_tum_trajectory(traj_path)
    t_gt, poses_gt = load_gt(gt_path)
    valid_times, valid_vo_poses, aligned_gt_poses = synchronize(t_vo, poses_vo, t_gt, poses_gt, args.max_dt)
    if len(valid_vo_poses) < 2:
        raise RuntimeError("Insufficient synchronized frames found.")

    aligned_vo, R_mat, t_vec = align_se3(valid_vo_poses, aligned_gt_poses)
    # Full VO trajectory aligned to GT frame (used for visualization only, not metrics)
    full_aligned_vo = (R_mat @ poses_vo.T).T + t_vec
    ate_rmse, rpe_rmse, start_to_end_drift, point_errors = compute_comprehensive_metrics(aligned_vo, aligned_gt_poses)
    metrics_json = load_json(metrics_path)
    summary = metrics_json.get("run_summary") or load_json(summary_path)
    frame_log = load_frame_log(frame_log_path)
    rep_rows = [r for r in frame_log if r.get("reprojection_rmse_px", "") != ""]
    rep_times = np.array([float(r["timestamp_s"]) for r in rep_rows], dtype=float)
    rep_errors = np.array([float_or_nan(r["reprojection_rmse_px"]) for r in rep_rows], dtype=float)
    finite = np.isfinite(rep_errors)
    rep_times = rep_times[finite]
    rep_errors = rep_errors[finite]

    path_length = compute_path_length(poses_vo)
    e_drift = float(point_errors[-1])
    total_seeds, mean_seeds, std_seeds = compute_seed_metrics(frame_log, "valid_3d_count", len(t_vo))
    mean_reproj = float(np.mean(rep_errors)) if len(rep_errors) > 0 else 0.0
    avg_ms = float(summary.get("avg_ms_per_frame", 0.0))
    runtime_hz_ms = 1000.0 / max(avg_ms, 1e-9) if avg_ms > 0 else 0.0
    runtime_str = f"{avg_ms:.2f} ms ({runtime_hz_ms:.2f} Hz)"
    metrics_summary = {
        "Sequence Name": tag,
        "E_drift": e_drift,
        "Path Length": path_length,
        "Drift %": 100.0 * e_drift / max(path_length, 1e-9),
        "Total Seeds": total_seeds,
        "Mean Seeds": mean_seeds,
        "Seed Std": std_seeds,
        "Tracking Failures": int(summary.get("tracking_failures", 0)),
        "Mean Reproj": mean_reproj,
        "Runtime": runtime_str,
    }

    render_keyframe_evolution(frame_log, out_dir / "01_keyframe_evolution.png")
    render_11_metric_table(metrics_summary, out_dir / "02_metrics_table.png")
    render_four_panel(aligned_vo, aligned_gt_poses, valid_times, point_errors, rep_times, rep_errors,
                      out_dir / "03_vo_analysis_4panel.png",
                      full_aligned_vo=full_aligned_vo, full_gt=poses_gt)
    render_animated_trajectory_gif(aligned_vo, aligned_gt_poses, out_dir / "04_vo_vs_gt.gif")
    print(f"Stereo result assets saved to: {out_dir}")


if __name__ == "__main__":
    main()
