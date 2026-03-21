import os
import cv2
import numpy as np
import yaml


# -------------------------
# CLASS DEFINITION
# -------------------------
class TumDataset:
    def __init__(self, dataset_path):

        self.dataset_path = dataset_path

        # -------------------------
        # Load calibration
        # -------------------------
        calib_path = os.path.join(dataset_path, "dso", "camchain.yaml")

        with open(calib_path, 'r') as f:
            data = yaml.safe_load(f)

        cam0 = data["cam0"]

        fx, fy, cx, cy = cam0["intrinsics"]

        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        self.D = np.array(cam0["distortion_coeffs"])

        self.image_size = tuple(cam0["resolution"])

        print("K:\n", self.K)
        print("D:\n", self.D)

        # -------------------------
        # Precompute undistortion maps
        # -------------------------
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K,
            self.D,
            np.eye(3),
            self.K,
            self.image_size,
            cv2.CV_16SC2
        )

        # -------------------------
        # Load image paths
        # -------------------------
        img_folder = os.path.join(dataset_path, "mav0", "cam0", "data")

        self.image_files = sorted([
            os.path.join(img_folder, f)
            for f in os.listdir(img_folder)
            if f.endswith(".png")
        ])
        self.length = len(self.image_files)

        print("Total images:", len(self.image_files))

    def get_frame(self, idx):

        img = cv2.imread(self.image_files[idx], cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("Image not found!")

        # Undistort
        img = cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

        return img


# -------------------------
# MAIN TEST BLOCK (OUTSIDE CLASS)
# -------------------------
if __name__ == "__main__":

    dataset_path = "C:/Users/RoboticsLab/PycharmProjects/stereo_vo_project/dataset/dataset-room2_512_16"

    dataset = TumDataset(dataset_path)

    print("\n--- TESTING FRAME LOADING ---")

    img = dataset.get_frame(0)

    print("Image shape:", img.shape)
    print("Pixel min/max:", img.min(), img.max())

    cv2.imshow("Undistorted Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()