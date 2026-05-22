from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.evaluation.alignment import apply_sim3, umeyama_alignment_sim3
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
        default=PROJECT_ROOT / "outputs" / "trajectories" / "room2_monocular_vo_map_1500_absreset_rebuild.txt",
        help="TUM trajectory file to evaluate.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional run summary JSON. Defaults to outputs/metrics/<traj_stem>_run_summary.json if it exists.",
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


def main():
    args = parse_args()

    traj_path = args.traj
    if not traj_path.is_absolute():
        traj_path = (PROJECT_ROOT / traj_path).resolve()

    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    tag = args.tag or traj_path.stem

    summary_path = args.summary
    if summary_path is None:
        candidate = PROJECT_ROOT / "outputs" / "metrics" / f"{traj_path.stem}_run_summary.json"
        summary_path = candidate if candidate.exists() else None
    elif not summary_path.is_absolute():
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

    scale, R, t = umeyama_alignment_sim3(est_assoc, gt_assoc)
    est_aligned = apply_sim3(est_assoc, scale, R, t)

    ate = compute_ate(gt_assoc, est_aligned)
    rpe = compute_rpe_translation(gt_assoc, est_aligned, delta=args.rpe_delta)
    end_drift = compute_end_point_drift(gt_assoc, est_aligned)
    coverage = compute_coverage(
        num_est_total=len(est_traj.timestamps),
        num_associated=len(est_idx),
        num_gt_total=len(gt_timestamps),
    )

    run_summary = load_optional_summary(summary_path)

    metrics = {
        "trajectory_path": str(traj_path),
        "summary_path": str(summary_path) if summary_path is not None else None,
        "alignment": {
            "scale": float(scale),
            "R": R.tolist(),
            "t": t.tolist(),
        },
        "coverage": coverage,
        "ate": {
            "rmse": ate.rmse,
            "mean": ate.mean,
            "median": ate.median,
            "std": ate.std,
            "min": ate.min,
            "max": ate.max,
        },
        "rpe_translation": {
            "delta_frames": args.rpe_delta,
            "rmse": rpe.rmse,
            "mean": rpe.mean,
            "median": rpe.median,
            "std": rpe.std,
            "min": rpe.min,
            "max": rpe.max,
        },
        "end_point_drift": end_drift,
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
        "scale": float(scale),
        "ate_rmse_m": ate.rmse,
        "ate_mean_m": ate.mean,
        "rpe_translation_delta_frames": args.rpe_delta,
        "rpe_translation_rmse_m": rpe.rmse,
        "rpe_translation_mean_m": rpe.mean,
        "end_point_drift_m": end_drift,
        "runtime_seconds": None if run_summary is None else run_summary.get("runtime_seconds"),
        "runtime_fps_frames_processed": None if run_summary is None else run_summary.get("runtime_fps_frames_processed"),
        "runtime_fps_accepted_poses": None if run_summary is None else run_summary.get("runtime_fps_accepted_poses"),
        "tracking_lost_events": None if run_summary is None else run_summary.get("tracking_lost_events"),
        "relocalization_successes": None if run_summary is None else run_summary.get("relocalization_successes"),
        "first_tracking_lost_frame": None if run_summary is None else run_summary.get("first_tracking_lost_frame"),
        "longest_tracked_streak": None if run_summary is None else run_summary.get("longest_tracked_streak"),
        "final_map_point_count": None if run_summary is None else run_summary.get("final_map_point_count"),
        "final_keyframe_count": None if run_summary is None else run_summary.get("final_keyframe_count"),
    }

    metrics_table_path = metrics_dir / f"{tag}_metrics_table.csv"
    save_metrics_table(metrics_table_path, table_row)

    save_trajectory_plot(
        gt_positions=gt_assoc,
        est_positions_aligned=est_aligned,
        output_path=plots_dir / f"{tag}_aligned_trajectory.png",
    )

    save_ate_plot(
        ate_errors=ate.per_frame_errors,
        output_path=plots_dir / f"{tag}_ate.png",
    )

    save_rpe_plot(
        rpe_errors=rpe.per_segment_errors,
        output_path=plots_dir / f"{tag}_rpe_translation.png",
    )

    print("=" * 60)
    print("Room2 Monocular Evaluation")
    print("=" * 60)
    print(f"Trajectory:              {traj_path}")
    if summary_path is not None:
        print(f"Run summary:             {summary_path}")
    print(f"Estimated total poses:   {len(est_traj.timestamps)}")
    print(f"GT total poses:          {len(gt_timestamps)}")
    print(f"Associated poses:        {len(est_idx)}")
    print(f"Association ratio / est: {coverage['association_ratio_vs_est']:.3f}")
    print(f"Association ratio / gt:  {coverage['association_ratio_vs_gt']:.3f}")
    print()
    print(f"Sim(3) scale:            {scale:.6f}")
    print()
    print(f"ATE RMSE [m]:            {ate.rmse:.6f}")
    print(f"ATE mean [m]:            {ate.mean:.6f}")
    print(f"ATE median [m]:          {ate.median:.6f}")
    print()
    print(f"RPE trans RMSE [m]:      {rpe.rmse:.6f}")
    print(f"RPE trans mean [m]:      {rpe.mean:.6f}")
    print(f"RPE delta frames:        {args.rpe_delta}")
    print()
    print(f"End-point drift [m]:     {end_drift:.6f}")
    if run_summary is not None and run_summary.get("runtime_seconds") is not None:
        print()
        print(f"Runtime [s]:             {run_summary['runtime_seconds']:.3f}")
        print(f"FPS processed:           {run_summary['runtime_fps_frames_processed']:.3f}")
        print(f"FPS accepted poses:      {run_summary['runtime_fps_accepted_poses']:.3f}")
    print()
    print(f"Saved metrics:           {metrics_path}")
    print(f"Saved metrics table:     {metrics_table_path}")
    print(f"Saved plots dir:         {plots_dir}")


if __name__ == "__main__":
    main()
