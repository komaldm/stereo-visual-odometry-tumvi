from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.evaluation.alignment import (
    apply_se3,
    apply_sim3,
    umeyama_alignment_se3,
    umeyama_alignment_sim3,
)
from src.evaluation.association import associate_by_timestamp
from src.evaluation.metrics import (
    compute_ate,
    compute_coverage,
    compute_end_point_drift,
    compute_rpe_translation,
)
from src.evaluation.plots import save_ate_plot, save_rpe_plot, save_trajectory_plot
from src.evaluation.trajectory_loader import load_tum_trajectory
from src.io.groundtruth_loader import TumVIGroundTruthLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traj",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "trajectories" / "room2_stereo_vo.txt",
        help="TUM stereo trajectory file to evaluate.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional stereo run summary JSON.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Output tag. Defaults to the trajectory stem.",
    )
    parser.add_argument(
        "--rpe-delta",
        type=int,
        default=10,
        help="Frame gap used for translation RPE.",
    )
    parser.add_argument(
        "--max-dt",
        type=float,
        default=0.02,
        help="Maximum timestamp association gap in seconds.",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="Ground truth CSV path. Defaults to room2 mocap data.csv.",
    )
    return parser.parse_args()


def load_optional_summary(summary_path: Path | None) -> dict | None:
    if summary_path is None or not summary_path.exists():
        return None

    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metrics_table(output_path: Path, row: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def save_scale_check_plot(
    gt_positions: np.ndarray,
    est_positions_rigid: np.ndarray,
    est_positions_sim3: np.ndarray,
    output_path: str | Path,
    title: str = "Room2 Stereo VO Scale Check",
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(gt_positions[:, 0], gt_positions[:, 2], label="Ground Truth", linewidth=2)
    plt.plot(est_positions_rigid[:, 0], est_positions_rigid[:, 2], label="Stereo rigid align", linewidth=2)
    plt.plot(est_positions_sim3[:, 0], est_positions_sim3[:, 2], label="Stereo Sim(3) align", linewidth=2, linestyle="--")
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()

    traj_path = args.traj
    if not traj_path.is_absolute():
        traj_path = (PROJECT_ROOT / traj_path).resolve()

    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    tag = args.tag or traj_path.stem

    summary_path = args.summary
    if summary_path is not None and not summary_path.is_absolute():
        summary_path = (PROJECT_ROOT / summary_path).resolve()

    gt_csv = args.gt if args.gt is not None else (
        PROJECT_ROOT / "dataset-room2_512_16" / "dataset-room2_512_16" / "mav0" / "mocap0" / "data.csv"
    )
    gt_loader = TumVIGroundTruthLoader(gt_csv)
    gt_poses = gt_loader.load()

    gt_timestamps = np.array([p.timestamp_ns * 1e-9 for p in gt_poses], dtype=np.float64)
    gt_positions = np.array([[p.tx, p.ty, p.tz] for p in gt_poses], dtype=np.float64)

    est_traj = load_tum_trajectory(traj_path)

    est_idx, gt_idx = associate_by_timestamp(
        est_timestamps=est_traj.timestamps,
        gt_timestamps=gt_timestamps,
        max_dt=args.max_dt,
    )

    if len(est_idx) < 20:
        raise RuntimeError(f"Too few associated trajectory points: {len(est_idx)}")

    est_assoc = est_traj.positions[est_idx]
    gt_assoc = gt_positions[gt_idx]

    R_rigid, t_rigid = umeyama_alignment_se3(est_assoc, gt_assoc)
    est_rigid = apply_se3(est_assoc, R_rigid, t_rigid)

    scale_sim3, R_sim3, t_sim3 = umeyama_alignment_sim3(est_assoc, gt_assoc)
    est_sim3 = apply_sim3(est_assoc, scale_sim3, R_sim3, t_sim3)

    ate_rigid = compute_ate(gt_assoc, est_rigid)
    rpe_rigid = compute_rpe_translation(gt_assoc, est_rigid, delta=args.rpe_delta)
    drift_rigid = compute_end_point_drift(gt_assoc, est_rigid)

    ate_sim3 = compute_ate(gt_assoc, est_sim3)
    rpe_sim3 = compute_rpe_translation(gt_assoc, est_sim3, delta=args.rpe_delta)
    drift_sim3 = compute_end_point_drift(gt_assoc, est_sim3)

    coverage = compute_coverage(
        num_est_total=len(est_traj.timestamps),
        num_associated=len(est_idx),
        num_gt_total=len(gt_timestamps),
    )

    run_summary = load_optional_summary(summary_path)

    metrics = {
        "trajectory_path": str(traj_path),
        "summary_path": str(summary_path) if summary_path is not None else None,
        "coverage": coverage,
        "rigid_alignment": {
            "R": R_rigid.tolist(),
            "t": t_rigid.tolist(),
            "ate": {
                "rmse": ate_rigid.rmse,
                "mean": ate_rigid.mean,
                "median": ate_rigid.median,
                "std": ate_rigid.std,
                "min": ate_rigid.min,
                "max": ate_rigid.max,
            },
            "rpe_translation": {
                "delta_frames": args.rpe_delta,
                "rmse": rpe_rigid.rmse,
                "mean": rpe_rigid.mean,
                "median": rpe_rigid.median,
                "std": rpe_rigid.std,
                "min": rpe_rigid.min,
                "max": rpe_rigid.max,
            },
            "end_point_drift": drift_rigid,
        },
        "sim3_diagnostic": {
            "scale": float(scale_sim3),
            "scale_error_abs": float(abs(scale_sim3 - 1.0)),
            "R": R_sim3.tolist(),
            "t": t_sim3.tolist(),
            "ate": {
                "rmse": ate_sim3.rmse,
                "mean": ate_sim3.mean,
                "median": ate_sim3.median,
                "std": ate_sim3.std,
                "min": ate_sim3.min,
                "max": ate_sim3.max,
            },
            "rpe_translation": {
                "delta_frames": args.rpe_delta,
                "rmse": rpe_sim3.rmse,
                "mean": rpe_sim3.mean,
                "median": rpe_sim3.median,
                "std": rpe_sim3.std,
                "min": rpe_sim3.min,
                "max": rpe_sim3.max,
            },
            "end_point_drift": drift_sim3,
        },
        "run_summary": run_summary,
    }

    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    plots_dir = PROJECT_ROOT / "outputs" / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / f"{tag}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    table_row = {
        "tag": tag,
        "trajectory_path": str(traj_path),
        "num_est_total": len(est_traj.timestamps),
        "num_associated": len(est_idx),
        "association_ratio_vs_est": coverage["association_ratio_vs_est"],
        "association_ratio_vs_gt": coverage["association_ratio_vs_gt"],
        "rigid_ate_rmse_m": ate_rigid.rmse,
        "rigid_rpe_translation_rmse_m": rpe_rigid.rmse,
        "rigid_end_point_drift_m": drift_rigid,
        "sim3_scale": float(scale_sim3),
        "sim3_scale_error_abs": float(abs(scale_sim3 - 1.0)),
        "sim3_ate_rmse_m": ate_sim3.rmse,
        "sim3_rpe_translation_rmse_m": rpe_sim3.rmse,
        "sim3_end_point_drift_m": drift_sim3,
    }

    metrics_table_path = metrics_dir / f"{tag}_metrics_table.csv"
    save_metrics_table(metrics_table_path, table_row)

    save_trajectory_plot(
        gt_positions=gt_assoc,
        est_positions_aligned=est_rigid,
        output_path=plots_dir / f"{tag}_rigid_aligned_trajectory.png",
        title="Room2 Stereo VO vs Ground Truth (Rigid Alignment)",
    )
    save_trajectory_plot(
        gt_positions=gt_assoc,
        est_positions_aligned=est_sim3,
        output_path=plots_dir / f"{tag}_sim3_aligned_trajectory.png",
        title="Room2 Stereo VO vs Ground Truth (Sim(3) Diagnostic)",
    )
    save_scale_check_plot(
        gt_positions=gt_assoc,
        est_positions_rigid=est_rigid,
        est_positions_sim3=est_sim3,
        output_path=plots_dir / f"{tag}_scale_check.png",
    )
    save_ate_plot(
        ate_errors=ate_rigid.per_frame_errors,
        output_path=plots_dir / f"{tag}_rigid_ate.png",
        title="Stereo ATE per Associated Frame (Rigid Alignment)",
    )
    save_rpe_plot(
        rpe_errors=rpe_rigid.per_segment_errors,
        output_path=plots_dir / f"{tag}_rigid_rpe_translation.png",
        title="Stereo RPE Translation per Segment (Rigid Alignment)",
    )

    print("=" * 60)
    print("Room2 Stereo Evaluation")
    print("=" * 60)
    print(f"Trajectory:                    {traj_path}")
    print(f"Estimated total poses:         {len(est_traj.timestamps)}")
    print(f"GT total poses:                {len(gt_timestamps)}")
    print(f"Associated poses:              {len(est_idx)}")
    print(f"Association ratio / est:       {coverage['association_ratio_vs_est']:.3f}")
    print(f"Association ratio / gt:        {coverage['association_ratio_vs_gt']:.3f}")
    print()
    print("Rigid alignment stereo metrics")
    print(f"ATE RMSE [m]:                  {ate_rigid.rmse:.6f}")
    print(f"ATE mean [m]:                  {ate_rigid.mean:.6f}")
    print(f"RPE trans RMSE [m]:            {rpe_rigid.rmse:.6f}")
    print(f"RPE trans mean [m]:            {rpe_rigid.mean:.6f}")
    print(f"End-point drift [m]:           {drift_rigid:.6f}")
    print()
    print("Sim(3) scale diagnostic")
    print(f"Estimated scale:               {scale_sim3:.6f}")
    print(f"|scale - 1|:                   {abs(scale_sim3 - 1.0):.6f}")
    print(f"Sim(3) ATE RMSE [m]:           {ate_sim3.rmse:.6f}")
    print(f"Sim(3) RPE trans RMSE [m]:     {rpe_sim3.rmse:.6f}")
    print(f"Sim(3) end-point drift [m]:    {drift_sim3:.6f}")
    print()
    print("Interpretation")
    print("If the Sim(3) scale is close to 1.0 and rigid-align metrics are close to Sim(3), stereo scale is healthy.")
    print("If Sim(3) is much better and the scale is far from 1.0, that indicates a scale problem.")
    print()
    print(f"Saved metrics:                 {metrics_path}")
    print(f"Saved metrics table:           {metrics_table_path}")
    print(f"Saved plots dir:               {plots_dir}")


if __name__ == "__main__":
    main()
