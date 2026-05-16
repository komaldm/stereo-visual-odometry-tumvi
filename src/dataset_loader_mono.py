import os
import cv2
import numpy as np
import yaml


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

        print("Total images:", self.length)

    def get_timestamp(self, idx):
        """
        FIX: Extracts the timestamp from the filename (nanoseconds)
        and converts it to decimal seconds for TUM format.
        """
        filename = os.path.basename(self.image_files[idx])
        timestamp_ns = int(filename.replace(".png", ""))
        return timestamp_ns / 1e9

    def get_stereo_frame(self, idx):
        """
        Required for main.py to fetch the image.
        Returns (Left_Image, None) to maintain compatibility with
        monocular logic.
        """
        img = cv2.imread(self.image_files[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image not found at index {idx}")

        # Undistort using the fish-eye maps
        undistorted_img = cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

        return undistorted_img, None

    def get_frame(self, idx):
        """Legacy support for older versions of the script."""
        img, _ = self.get_stereo_frame(idx)
        return img


# -------------------------
# MAIN TEST BLOCK
# -------------------------
if __name__ == "__main__":
    dataset_path = r"C:\Users\RoboticsLab\PycharmProjects\stereo_vo_project\dataset\dataset-room2_512_16"
    dataset = TumDataset(dataset_path)
    print("Test Timestamp for frame 0:", dataset.get_timestamp(0))