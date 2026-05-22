from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.io.dataset_loader import TumVIDatasetLoader
from src.io.calibration_loader import TumVICalibrationLoader
from src.vo.stereo_vo import StereoVisualOdometry


def main():
    dataset_root = Path(r"D:\New folder\project\dataset-room2_512_16\dataset-room2_512_16")

    dataset = TumVIDatasetLoader(dataset_root).load()
    calibration = TumVICalibrationLoader(dataset_root / "dso" / "camchain.yaml").load()

    output_path = PROJECT_ROOT / "outputs" / "trajectories" / "room2_stereo_test.txt"

    vo = StereoVisualOdometry(
        dataset=dataset,
        calibration=calibration,
        output_path=output_path,
        max_depth_m=15.0,
        enable_visualization=True,
        feature_window_name="Room2 Stereo Features",
        disparity_window_name="Room2 Stereo Disparity",
        trajectory_window_name="Room2 Stereo Trajectory",
    )

    vo.run(start_idx=0, max_frames=200)


if __name__ == "__main__":
    main()
