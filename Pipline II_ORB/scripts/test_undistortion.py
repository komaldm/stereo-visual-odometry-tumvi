from pathlib import Path
import sys
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.io.dataset_loader import TumVIDatasetLoader
from src.io.calibration_loader import TumVICalibrationLoader
from src.io.image_loader import load_image
from src.geometry.undistortion import FisheyeUndistorter


def main():
    dataset_root = Path(r"D:\New folder\project\dataset-room2_512_16\dataset-room2_512_16")

    dataset = TumVIDatasetLoader(dataset_root)
    frame = dataset[0]
    left_img = load_image(frame.left_image_path, grayscale=True)

    stereo_calib = TumVICalibrationLoader(dataset_root / "dso" / "camchain.yaml").load()

    undistorter = FisheyeUndistorter(
        K=stereo_calib.cam0.intrinsics,
        D=stereo_calib.cam0.distortion,
        image_size=stereo_calib.cam0.resolution,
        balance=0.0,
    )

    undistorted = undistorter.undistort(left_img)

    comparison = np.hstack([left_img, undistorted])

    cv2.imshow("Original | Undistorted", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()