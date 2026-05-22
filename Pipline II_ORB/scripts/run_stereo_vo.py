from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.io.calibration_loader import TumVICalibrationLoader
from src.io.dataset_loader import TumVIDatasetLoader
from src.vo.stereo_vo import StereoVisualOdometry
from src.config.dataset_profiles import get_profile


def _run_evaluation(traj_path: Path, summary_path: Path, tag: str, gt_path: Path) -> None:
    import importlib.util

    eval_script = PROJECT_ROOT / "scripts" / "evaluate_room2_stereo.py"
    if not eval_script.exists():
        print("[eval] evaluate_room2_stereo.py not found, skipping auto-evaluation")
        return

    spec = importlib.util.spec_from_file_location("eval_stereo", eval_script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_stereo"] = mod

    old_argv = sys.argv
    sys.argv = [
        str(eval_script),
        "--traj", str(traj_path),
        "--summary", str(summary_path),
        "--tag", tag,
        "--gt", str(gt_path),
    ]
    try:
        spec.loader.exec_module(mod)
        mod.main()
    finally:
        sys.argv = old_argv
        del sys.modules["eval_stereo"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="room2",
                        help="Dataset profile name: room2 | corridor3")
    parser.add_argument("--start-idx", type=int, default=0, help="Start frame index")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Maximum frames to process (0 = full sequence)")
    parser.add_argument("--tag", type=str, default=None, help="Output tag suffix")
    parser.add_argument("--no-vis", action="store_true", help="Disable OpenCV visualization")
    parser.add_argument("--no-eval", action="store_true", help="Skip ATE/RPE evaluation after run")
    parser.add_argument("--save-video", action="store_true",
                        help="Save visualization as MP4 to outputs/videos/")
    args = parser.parse_args()

    profile = get_profile(args.dataset)
    sp = profile.stereo

    dataset = TumVIDatasetLoader(profile.dataset_root)
    calibration = TumVICalibrationLoader(profile.dataset_root / "dso" / "camchain.yaml").load()

    max_frames = None if args.max_frames <= 0 else args.max_frames
    tag = args.tag or f"{profile.name}_stereo_vo_final"

    output_path = PROJECT_ROOT / "outputs" / "trajectories" / f"{tag}.txt"
    summary_path = PROJECT_ROOT / "outputs" / "metrics" / f"{tag}_run_summary.json"
    log_path = PROJECT_ROOT / "outputs" / "logs" / f"{tag}_frame_log.csv"
    video_path = (
        PROJECT_ROOT / "outputs" / "videos" / f"{tag}_vis.mp4"
        if args.save_video and not args.no_vis else None
    )

    vo = StereoVisualOdometry(
        dataset=dataset,
        calibration=calibration,
        output_path=output_path,
        summary_path=summary_path,
        min_depth_m=sp.min_depth_m,
        max_depth_m=sp.max_depth_m,
        min_disparity=sp.min_disparity,
        enable_visualization=not args.no_vis,
        log_path=log_path,
        video_path=video_path,
        orb_nfeatures=sp.orb_nfeatures,
        min_tracker_inliers=sp.min_tracker_inliers,
        reprojection_error_px=sp.reprojection_error_px,
        min_recovery_inliers=sp.min_recovery_inliers,
        recovery_reprojection_px=sp.recovery_reprojection_px,
        min_3d3d_inliers=sp.min_3d3d_inliers,
        num_disparities=sp.num_disparities,
        block_size=sp.block_size,
        uniqueness_ratio=sp.uniqueness_ratio,
        speckle_window_size=sp.speckle_window_size,
        speckle_range=sp.speckle_range,
    )

    vo.run(start_idx=args.start_idx, max_frames=max_frames)

    if not args.no_eval and output_path.exists():
        print("\n" + "=" * 60)
        print("Auto-evaluating trajectory ...")
        print("=" * 60)
        _run_evaluation(output_path, summary_path, tag, profile.gt_path)


if __name__ == "__main__":
    main()
