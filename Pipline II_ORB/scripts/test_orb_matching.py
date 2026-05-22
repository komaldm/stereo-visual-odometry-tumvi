from pathlib import Path
import sys
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.io.dataset_loader import TumVIDatasetLoader
from src.io.calibration_loader import TumVICalibrationLoader
from src.io.image_loader import load_image
from src.geometry.undistortion import FisheyeUndistorter
from src.features.orb_frontend import ORBFrontend
from src.features.matcher import ORBMatcher
from src.utils.feature_vis import draw_keypoints, draw_matches


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
    frame2 = dataset[1]

    img1 = load_image(frame1.left_image_path, grayscale=True)
    img2 = load_image(frame2.left_image_path, grayscale=True)

    img1_u = undistorter.undistort(img1)
    img2_u = undistorter.undistort(img2)

    frontend = ORBFrontend(nfeatures=2000)
    matcher = ORBMatcher(ratio_thresh=0.70, mutual_check=True)

    feat1 = frontend.detect_and_compute(img1_u)
    feat2 = frontend.detect_and_compute(img2_u)

    result = matcher.match(feat1.keypoints, feat1.descriptors, feat2.keypoints, feat2.descriptors)

    print(f"Frame 1 keypoints: {len(feat1.keypoints)}")
    print(f"Frame 2 keypoints: {len(feat2.keypoints)}")
    print(f"Good matches:      {len(result.good_matches)}")

    kp_vis1 = draw_keypoints(img1_u, feat1.keypoints)
    kp_vis2 = draw_keypoints(img2_u, feat2.keypoints)
    match_vis = draw_matches(img1_u, feat1.keypoints, img2_u, feat2.keypoints, result.good_matches, max_matches=150)

    cv2.imshow("Undistorted Left Frame 1 Keypoints", kp_vis1)
    cv2.imshow("Undistorted Left Frame 2 Keypoints", kp_vis2)
    cv2.imshow("ORB Matches", match_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
