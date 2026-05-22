from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def load_tum(filename: Path):
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1:4]


def associate(ts_est, xyz_est, ts_gt, xyz_gt, max_diff=0.05):
    matched_est, matched_gt, matched_ts = [], [], []
    for i, t_est in enumerate(ts_est):
        diffs = np.abs(ts_gt - t_est)
        idx = np.argmin(diffs)
        if diffs[idx] < max_diff:
            matched_est.append(xyz_est[i])
            matched_gt.append(xyz_gt[idx])
            matched_ts.append(t_est)
    return np.array(matched_ts), np.array(matched_est).T, np.array(matched_gt).T


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
    var_data = np.mean(np.sum(data_zero**2, axis=0))
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


def load_gt(gt_path: Path):
    data_gt = np.loadtxt(gt_path, delimiter=",", comments="#")
    ts_g = data_gt[:, 0]
    if len(ts_g) > 0 and ts_g[0] > 1e12:
        ts_g = ts_g / 1e9
    xyz_g = data_gt[:, 1:4]
    return ts_g, xyz_g


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


def compute_tracking_episodes(frame_log: list[dict]) -> tuple[float, float]:
    """Returns (mttr_seconds, max_blind_seconds) from 'tracking_lost' column."""
    if not frame_log or "tracking_lost" not in frame_log[0]:
        return 0.0, 0.0
    sorted_log = sorted(frame_log, key=lambda r: int(r["frame_idx"]))
    episode_durations: list[float] = []
    in_episode = False
    ep_start_ts = 0.0
    prev_ts = 0.0
    for row in sorted_log:
        lost = row.get("tracking_lost", "0").strip() == "1"
        ts = float(row["timestamp_s"])
        if lost and not in_episode:
            in_episode = True
            ep_start_ts = ts
        elif not lost and in_episode:
            in_episode = False
            episode_durations.append(ts - ep_start_ts)
        prev_ts = ts
    if in_episode and prev_ts > ep_start_ts:
        episode_durations.append(prev_ts - ep_start_ts)
    if not episode_durations:
        return 0.0, 0.0
    return float(np.mean(episode_durations)), float(np.max(episode_durations))


def resolve_gt_path(dataset: str) -> Path:
    if dataset == "corridor3":
        return PROJECT_ROOT / "dataset-corridor3_512_16" / "mav0" / "mocap0" / "data.csv"
    if dataset == "outdoors5":
        return PROJECT_ROOT / "dataset-outdoors5_512_16" / "mav0" / "mocap0" / "data.csv"
    return PROJECT_ROOT / "dataset-room2_512_16" / "mav0" / "mocap0" / "data.csv"


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
        ["Mean Time to Recovery (MTTR)", f"{metrics_dict['MTTR']:.2f} s"],
        ["Max Time Blind", f"{metrics_dict['Max Blind']:.2f} s"],
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
        f"Monocular VO Evaluation Metrics — {metrics_dict['Sequence Name']}",
        fontweight="bold", pad=20,
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def render_keyframe_evolution(frame_log: list[dict], summary: dict, output_path: Path) -> None:
    if frame_log and "keyframe_count" in frame_log[0]:
        frames = np.array([int(r["frame_idx"]) for r in frame_log])
        cumulative = np.array([int(r["keyframe_count"]) for r in frame_log])
    else:
        final_count = int(summary.get("final_keyframe_count", 1))
        total = int(summary.get("final_frame_idx", final_count))
        frames = np.linspace(0, total, final_count)
        cumulative = np.arange(1, final_count + 1)
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
    xyz_s_aligned: np.ndarray,
    xyz_g_aligned: np.ndarray,
    matched_times: np.ndarray,
    point_errors: np.ndarray,
    rep_times: np.ndarray,
    rep_errors: np.ndarray,
    output_path: Path,
) -> None:
    relative_time = matched_times - matched_times[0]
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(xyz_g_aligned[:, 0], xyz_g_aligned[:, 1], "g-", linewidth=2, label="Ground Truth")
    plt.plot(xyz_s_aligned[:, 0], xyz_s_aligned[:, 1], "b:", linewidth=2, label="VO")
    plt.scatter(xyz_g_aligned[0, 0], xyz_g_aligned[0, 1], c="green", marker="o", s=100, label="GT start")
    plt.scatter(xyz_s_aligned[0, 0], xyz_s_aligned[0, 1], c="blue", marker="x", s=120, label="VO start")
    plt.title("Top-Down Trajectory Overlay")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.axis("equal")

    plt.subplot(2, 2, 2)
    plt.plot(relative_time, xyz_g_aligned[:, 2], "g-", label="Ground Truth")
    plt.plot(relative_time, xyz_s_aligned[:, 2], "b--", label="VO")
    plt.title("Vertical Drift Evolution")
    plt.xlabel("Time (s)")
    plt.ylabel("Z Altitude (m)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.subplot(2, 2, 3)
    ate_rmse = float(np.sqrt(np.mean(point_errors**2)))
    plt.plot(relative_time, point_errors, "r-", linewidth=1.5)
    plt.axhline(ate_rmse, color="black", linestyle=":", label=f"ATE RMSE ({ate_rmse:.3f} m)")
    plt.title("Absolute Trajectory Drift (ATE) per Frame")
    plt.xlabel("Time (s)")
    plt.ylabel("Positional Error (m)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.subplot(2, 2, 4)
    if len(rep_errors) > 0:
        rel_t_err = rep_times - rep_times[0]
        plt.plot(rel_t_err, rep_errors, color="orange", linewidth=1)
        plt.title("Frame-by-Frame Reprojection Error")
        plt.xlabel("Time (s)")
        plt.ylabel("Error (pixel)")
        plt.grid(True, linestyle="--", alpha=0.4)
    else:
        plt.text(0.5, 0.5, "Reprojection Error File Not Found", ha="center", va="center")
        plt.title("Reprojection Stability")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=Path, default=PROJECT_ROOT / "outputs" / "trajectories" / "room2_monocular_vo_map_final.txt")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None, help="room2 | corridor3 (auto-inferred from tag/path if omitted)")
    parser.add_argument("--gt", type=Path, default=None, help="GT CSV path; resolved from --dataset if omitted")
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--frame-log", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_png" / "monocular")
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

    ts_s, xyz_s = load_tum(traj_path)
    ts_g, xyz_g = load_gt(gt_path)
    matched_times, xyz_s_matched, xyz_g_matched = associate(ts_s, xyz_s, ts_g, xyz_g, max_diff=args.max_dt)
    if xyz_s_matched.shape[1] < 2:
        raise RuntimeError("Insufficient synchronized frames found.")

    scale, R, t = align_sim3(xyz_g_matched, xyz_s_matched)
    xyz_s_aligned = (scale * R @ xyz_s_matched + t).T
    xyz_g_aligned = xyz_g_matched.T
    point_errors = np.linalg.norm(xyz_g_aligned - xyz_s_aligned, axis=1)
    ate_rmse = np.sqrt(np.mean(point_errors**2))
    rpe_rmse = compute_rpe(xyz_g_aligned, xyz_s_aligned, delta=1)

    metrics_json = load_json(metrics_path)
    summary = metrics_json.get("run_summary") or load_json(summary_path)
    frame_log = load_frame_log(frame_log_path)
    rep_rows = [r for r in frame_log if r.get("reprojection_rmse_px", "") != ""]
    rep_times = np.array([float(r["timestamp_s"]) for r in rep_rows], dtype=float)
    rep_errors = np.array([float_or_nan(r["reprojection_rmse_px"]) for r in rep_rows], dtype=float)
    finite = np.isfinite(rep_errors)
    rep_times = rep_times[finite]
    rep_errors = rep_errors[finite]

    # Apply Sim3 transform to full trajectory for metric path length (xyz_s is in VO scale units)
    xyz_s_metric = (scale * R @ xyz_s.T + t).T
    path_length = compute_path_length(xyz_s_metric)
    e_drift = float(point_errors[-1])
    total_seeds, mean_seeds, std_seeds = compute_seed_metrics(frame_log, "map_point_count", len(ts_s))
    mttr, max_blind = compute_tracking_episodes(frame_log)
    mean_reproj = float(np.mean(rep_errors)) if len(rep_errors) > 0 else 0.0
    avg_ms = float(summary.get("avg_ms_per_frame", 0.0))
    runtime_hz_ms = 1000.0 / max(avg_ms, 1e-9) if avg_ms > 0 else 0.0
    runtime_str = f"{avg_ms:.2f} ms ({runtime_hz_ms:.2f} Hz)"
    metrics_summary = {
        "Sequence Name": tag,
        "E_drift": e_drift,
        "Path Length": path_length,
        "Drift %": 100.0 * e_drift / max(path_length, 1e-9),
        "MTTR": mttr,
        "Max Blind": max_blind,
        "Total Seeds": total_seeds,
        "Mean Seeds": mean_seeds,
        "Seed Std": std_seeds,
        "Tracking Failures": int(summary.get("tracking_lost_events", 0)),
        "Mean Reproj": mean_reproj,
        "Runtime": runtime_str,
    }

    render_keyframe_evolution(frame_log, summary, out_dir / "01_keyframe_evolution.png")
    render_11_metric_table(metrics_summary, out_dir / "02_metrics_table.png")
    render_four_panel(xyz_s_aligned, xyz_g_aligned, matched_times, point_errors, rep_times, rep_errors, out_dir / "03_vo_analysis_4panel.png")
    print(f"Monocular result assets saved to: {out_dir}")


if __name__ == "__main__":
    main()
