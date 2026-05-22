from pathlib import Path
import sys
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.io.dataset_loader import TumVIDatasetLoader
from src.io.calibration_loader import TumVICalibrationLoader
from src.io.image_loader import load_image
from src.geometry.undistortion import FisheyeUndistorter
from src.features.orb_frontend import ORBFrontend
from src.features.matcher import ORBMatcher
from src.vo.monocular_initializer import MonocularInitializer
from src.utils.feature_vis import draw_matches
from src.utils.plot_3d import plot_point_cloud_3d


def main():
    dataset_root = Path(r"D:\New folder\project\dataset-room2_512_16\dataset-room2_512_16")

    dataset = TumVIDatasetLoader(dataset_root)
    stereo_calib = TumVICalibrationLoader(dataset_root / "dso" / "camchain.yaml").load()

    undistorter = FisheyeUndistorter(
        K=stereo_calib.cam0.intrinsics,
        D=stereo_calib.cam0.distortion,
        image_size=stereo_calib.cam0.resolution,
        balance=0.0,
    )

    frame1 = dataset[0]
    frame2 = dataset[4]

    img1 = load_image(frame1.left_image_path, grayscale=True)
    img2 = load_image(frame2.left_image_path, grayscale=True)

    img1_u = undistorter.undistort(img1)
    img2_u = undistorter.undistort(img2)

    frontend = ORBFrontend(nfeatures=2000)
    matcher = ORBMatcher(ratio_thresh=0.70, mutual_check=True)

    feat1 = frontend.detect_and_compute(img1_u)
    feat2 = frontend.detect_and_compute(img2_u)

    match_result = matcher.match(
        feat1.keypoints, feat1.descriptors,
        feat2.keypoints, feat2.descriptors
    )

    initializer = MonocularInitializer(K=undistorter.new_K)
    init_result = initializer.initialize(match_result.pts1, match_result.pts2)

    print(f"Good matches:        {len(match_result.good_matches)}")
    print(f"Essential inliers:   {init_result.num_inliers}")
    print(f"Valid 3D points:     {init_result.num_valid_3d}")
    if init_result.num_valid_3d > 0:
        print(f"Mean reproj err img1: {np.mean(init_result.reproj_err1):.3f}")
        print(f"Mean reproj err img2: {np.mean(init_result.reproj_err2):.3f}")
        print(f"Mean parallax [deg]:  {np.mean(init_result.parallax_deg):.3f}")
        print(f"Median parallax [deg]: {np.median(init_result.parallax_deg):.3f}")

    if init_result.num_inliers > 0:
        ratio = init_result.num_valid_3d / init_result.num_inliers
        print(f"3D validity ratio:   {ratio:.3f}")

    # visualize only the matches that survived both essential filtering and triangulation filtering
    # for a quick display we take the first num_valid_3d essential matches approximately
    # better bookkeeping can be added later
    print("Rotation:")
    print(init_result.R)
    print("Translation direction:")
    print(init_result.t.ravel())

    plot_point_cloud_3d(init_result.triangulated_points_3d, title="Initial Monocular 3D Map")


if __name__ == "__main__":
    main()
