from pathlib import Path
import sys
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.io.dataset_loader import TumVIDatasetLoader
from src.io.calibration_loader import TumVICalibrationLoader
from src.io.image_loader import load_image
from src.geometry.rectification import compute_stereo_rectification_fisheye, rectify_stereo_pair
from src.utils.visualization import draw_horizontal_epipolar_lines


def main():
    dataset_root = Path(r"D:\New folder\project\dataset-room2_512_16\dataset-room2_512_16")

    dataset = TumVIDatasetLoader(dataset_root)
    frame = dataset[0]

    left_img = load_image(frame.left_image_path, grayscale=True)
    right_img = load_image(frame.right_image_path, grayscale=True)

    stereo_calib = TumVICalibrationLoader(dataset_root / "dso" / "camchain.yaml").load()

    print("cam0 distortion model:", stereo_calib.cam0.distortion_model)
    print("cam1 distortion model:", stereo_calib.cam1.distortion_model)

    rectification = compute_stereo_rectification_fisheye(
        K1=stereo_calib.cam0.intrinsics,
        d1=stereo_calib.cam0.distortion,
        K2=stereo_calib.cam1.intrinsics,
        d2=stereo_calib.cam1.distortion,
        image_size=stereo_calib.cam0.resolution,
        T_cam1_cam0=stereo_calib.T_cam1_cam0,
    )

    left_rect, right_rect = rectify_stereo_pair(left_img, right_img, rectification)

    vis_before = draw_horizontal_epipolar_lines(left_img, right_img)
    vis_after = draw_horizontal_epipolar_lines(left_rect, right_rect)

    cv2.imshow("Before Rectification", vis_before)
    cv2.imshow("After Rectification", vis_after)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()