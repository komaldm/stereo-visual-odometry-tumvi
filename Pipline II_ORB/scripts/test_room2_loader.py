from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.io.dataset_loader import TumVIDatasetLoader
from src.io.calibration_loader import TumVICalibrationLoader
from src.io.groundtruth_loader import TumVIGroundTruthLoader


def main():
    dataset_root = Path(r"D:\New folder\project\dataset-room2_512_16\dataset-room2_512_16")

    print("=" * 60)
    print("1. Testing dataset loader")
    print("=" * 60)
    dataset = TumVIDatasetLoader(dataset_root)
    print(dataset.summary())

    first_frame = dataset[0]
    print("\nFirst stereo frame:")
    print(f"Timestamp: {first_frame.timestamp_ns}")
    print(f"Left image:  {first_frame.left_image_path}")
    print(f"Right image: {first_frame.right_image_path}")

    print("\n" + "=" * 60)
    print("2. Testing calibration loader")
    print("=" * 60)
    camchain_path = dataset_root / "dso" / "camchain.yaml"
    calib_loader = TumVICalibrationLoader(camchain_path)
    stereo_calib = calib_loader.load()

    print("Cam0 K:")
    print(stereo_calib.cam0.intrinsics)
    print("\nCam1 K:")
    print(stereo_calib.cam1.intrinsics)
    print("\nT_cam1_cam0:")
    print(stereo_calib.T_cam1_cam0)
    print(f"\nBaseline [m]: {stereo_calib.baseline_m:.6f}")

    print("\n" + "=" * 60)
    print("3. Testing ground truth loader")
    print("=" * 60)
    gt_path = dataset_root / "mav0" / "mocap0" / "data.csv"
    gt_loader = TumVIGroundTruthLoader(gt_path)
    gt_summary = gt_loader.summary()
    print(gt_summary)

    poses = gt_loader.load()
    if poses:
        print("\nFirst GT pose:")
        print(poses[0])

    print("\nAll loaders are working.")


if __name__ == "__main__":
    main()