from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.io.dataset_loader import TumVIDatasetLoader
from src.io.calibration_loader import TumVICalibrationLoader
from src.io.groundtruth_loader import TumVIGroundTruthLoader
from src.io.groundtruth_utils import gt_poses_to_xyz_array
from src.io.image_loader import load_image
from src.io.trajectory_io import save_trajectory_tum
from src.geometry.undistortion import FisheyeUndistorter
from src.features.orb_frontend import ORBFrontend
from src.features.matcher import ORBMatcher
from src.visualization.vo_map_visualizer import VOMapVisualizer
from src.vo.monocular_vo_map import MonocularVOMap
from src.config.dataset_profiles import get_profile


def _run_evaluation(traj_path: Path, summary_path: Path, tag: str, gt_path: Path) -> None:
    import importlib.util

    eval_script = PROJECT_ROOT / "scripts" / "evaluate_room2_monocular.py"
    if not eval_script.exists():
        print("[eval] evaluate_room2_monocular.py not found, skipping auto-evaluation")
        return

    spec = importlib.util.spec_from_file_location("eval_mono", eval_script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_mono"] = mod

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
        del sys.modules["eval_mono"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="room2",
                        help="Dataset profile name: room2 | corridor3")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="Exclusive end frame index (default: full sequence)")
    parser.add_argument("--tag", type=str, default=None, help="Output tag suffix")
    parser.add_argument("--no-vis", action="store_true", help="Disable OpenCV visualization")
    parser.add_argument("--no-eval", action="store_true", help="Skip ATE/RPE evaluation after run")
    parser.add_argument("--save-debug-image", action="store_true",
                        help="Save the final visualization frame")
    parser.add_argument("--save-video", action="store_true",
                        help="Save visualization as MP4 to outputs/videos/")
    args = parser.parse_args()

    profile = get_profile(args.dataset)
    tag = args.tag or f"{profile.name}_monocular_vo_map_final"
    mp = profile.mono

    dataset_root = profile.dataset_root
    dataset = TumVIDatasetLoader(dataset_root)
    stereo_calib = TumVICalibrationLoader(dataset_root / "dso" / "camchain.yaml").load()

    gt_loader = TumVIGroundTruthLoader(profile.gt_path)
    gt_positions = gt_poses_to_xyz_array(gt_loader.load())

    undistorter = FisheyeUndistorter(
        K=stereo_calib.cam0.intrinsics,
        D=stereo_calib.cam0.distortion,
        image_size=stereo_calib.cam0.resolution,
        balance=0.0,
    )

    frontend = ORBFrontend(nfeatures=mp.orb_nfeatures)
    matcher = ORBMatcher(ratio_thresh=mp.ratio_thresh, mutual_check=True)
    if args.no_vis:
        visualizer = None
    else:
        video_path_for_vis = (
            PROJECT_ROOT / "outputs" / "videos" / f"{tag}_vis.mp4"
            if args.save_video else None
        )
        visualizer = VOMapVisualizer(
            traj_size=(700, 700),
            scale=15.0,
            video_path=video_path_for_vis,
            show_window=not args.save_video,
        )

    vo = MonocularVOMap(
        K=undistorter.new_K,
        undistorter=undistorter,
        frontend=frontend,
        matcher=matcher,
        min_pnp_inliers=mp.min_pnp_inliers,
        keyframe_translation_thresh=mp.keyframe_translation_thresh,
        min_essential_inliers=mp.min_essential_inliers,
        max_fallback_frames_before_rebuild=mp.max_fallback_frames_before_rebuild,
        keyframe_rotation_thresh_deg=mp.keyframe_rotation_thresh_deg,
        min_relocalization_inliers=mp.min_relocalization_inliers,
        min_lost_relocalization_inliers=mp.min_lost_relocalization_inliers,
        min_lost_relocalization_inlier_ratio=mp.min_lost_relocalization_inlier_ratio,
        local_map_window=mp.local_map_window,
        post_relocalization_grace_frames=mp.post_relocalization_grace_frames,
        recovery_probation_length=mp.recovery_probation_length,
        recovery_min_local_inliers=mp.recovery_min_local_inliers,
        min_strong_inliers_for_rebuild_anchor=mp.min_strong_inliers_for_rebuild_anchor,
        min_rebuild_baseline=mp.min_rebuild_baseline,
        min_rebuild_match_count=mp.min_rebuild_match_count,
        max_cv_fallback_frames=mp.max_cv_fallback_frames,
        max_translation_step=mp.max_translation_step,
        max_rotation_step_deg=mp.max_rotation_step_deg,
        init_max_offset=mp.init_max_offset,
    )

    debug_image_path = (
        PROJECT_ROOT / "outputs" / "plots" / f"{tag}_debug.png"
        if args.save_debug_image else None
    )
    summary_path = PROJECT_ROOT / "outputs" / "metrics" / f"{tag}_run_summary.json"
    log_path = PROJECT_ROOT / "outputs" / "logs" / f"{tag}_frame_log.csv"

    result = vo.run(
        dataset=dataset,
        image_loader=load_image,
        visualizer=visualizer,
        gt_positions=gt_positions,
        end_idx=args.end_idx,
        debug_image_path=debug_image_path,
        summary_path=summary_path,
        log_path=log_path,
    )

    output_path = PROJECT_ROOT / "outputs" / "trajectories" / f"{tag}.txt"
    save_trajectory_tum(output_path, result.timestamps_ns, result.poses_wc)
    print(f"Saved trajectory: {output_path}")

    if not args.no_eval and output_path.exists():
        print("\n" + "=" * 60)
        print("Auto-evaluating trajectory ...")
        print("=" * 60)
        _run_evaluation(output_path, summary_path, tag, profile.gt_path)


if __name__ == "__main__":
    main()
