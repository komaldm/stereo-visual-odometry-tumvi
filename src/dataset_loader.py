import os
import cv2
import numpy as np
import yaml


class TumDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        # Paths for Left (cam0) and Right (cam1) cameras
        self.cam0_path = os.path.join(dataset_path, "mav0", "cam0", "data")
        self.cam1_path = os.path.join(dataset_path, "mav0", "cam1", "data")

        self.left_image_files = sorted(
            [os.path.join(self.cam0_path, f) for f in os.listdir(self.cam0_path) if f.endswith('.png')])
        self.right_image_files = sorted(
            [os.path.join(self.cam1_path, f) for f in os.listdir(self.cam1_path) if f.endswith('.png')])

        self.length = min(len(self.left_image_files), len(self.right_image_files))
        self.image_size = (512, 512)

        # ==========================================
        # DYNAMIC CALIBRATION PARSING
        # ==========================================
        yaml_path = os.path.join(dataset_path, "dso", "camchain.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Could not find camchain.yaml at {yaml_path}")

        with open(yaml_path, 'r') as f:
            calib = yaml.safe_load(f)

        # Left Camera (cam0) - Ensure dtype is float64
        intr0 = calib['cam0']['intrinsics']
        self.K1 = np.array([[intr0[0], 0, intr0[2]], [0, intr0[1], intr0[3]], [0, 0, 1]], dtype=np.float64)
        self.D1 = np.array(calib['cam0']['distortion_coeffs'], dtype=np.float64).reshape(4, 1)

        # Right Camera (cam1) - Ensure dtype is float64
        intr1 = calib['cam1']['intrinsics']
        self.K2 = np.array([[intr1[0], 0, intr1[2]], [0, intr1[1], intr1[3]], [0, 0, 1]], dtype=np.float64)
        self.D2 = np.array(calib['cam1']['distortion_coeffs'], dtype=np.float64).reshape(4, 1)

        # Extrinsics (Transformation from cam0 to cam1)
        T_1_0 = np.array(calib['cam1']['T_cn_cnm1'], dtype=np.float64)

        # CRITICAL FIX: Use .copy() to ensure the matrix is continuous in memory for OpenCV C++
        self.R = T_1_0[0:3, 0:3].copy()
        self.T = T_1_0[0:3, 3:4].copy()

        # ==========================================
        # PERFORM FISHEYE STEREO RECTIFICATION
        # ==========================================
        # balance=0.0 crops out the black warped edges to maximize valid pixels
        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
            self.K1, self.D1, self.K2, self.D2,
            self.image_size, self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            newImageSize=self.image_size,
            balance=0.0,
            fov_scale=1.0
        )

        self.map1x, self.map1y = cv2.fisheye.initUndistortRectifyMap(
            self.K1, self.D1, R1, P1, self.image_size, cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.fisheye.initUndistortRectifyMap(
            self.K2, self.D2, R2, P2, self.image_size, cv2.CV_32FC1)

        # --- UPDATE METRIC RULERS BASED ON RECTIFICATION ---
        self.K = P1[:, :3]
        self.focal_length = self.K[0, 0]
        self.baseline = abs(P2[0, 3] / P2[0, 0])

    def get_stereo_frame(self, index):
        if index >= self.length:
            raise IndexError("Index out of bounds.")

        img_left = cv2.imread(self.left_image_files[index], cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread(self.right_image_files[index], cv2.IMREAD_GRAYSCALE)

        img_left_rect = cv2.remap(img_left, self.map1x, self.map1y, interpolation=cv2.INTER_LINEAR)
        img_right_rect = cv2.remap(img_right, self.map2x, self.map2y, interpolation=cv2.INTER_LINEAR)

        return img_left_rect, img_right_rect

    def get_timestamp(self, index):
        filename = os.path.basename(self.left_image_files[index])
        timestamp_ns = float(filename.replace('.png', ''))
        return timestamp_ns / 1e9


# ==========================================
# UNIT TEST
# ==========================================
if __name__ == "__main__":
    print("--- Testing Dynamic Fisheye Stereo Rectification ---")

    # UPDATE THIS PATH to match your actual dataset location
    test_path = r"C:\Users\RoboticsLab\PycharmProjects\stereo_vo_project\dataset\dataset-room2_512_16"

    try:
        dataset = TumDataset(test_path)
        print(f"Successfully loaded dataset with {dataset.length} stereo pairs.")
        print(f"Original Left Focal Length: {dataset.K1[0, 0]:.2f}")
        print(f"Rectified Pinhole Focal Length: {dataset.focal_length:.2f} pixels")
        print(f"Rectified Baseline: {dataset.baseline:.4f} meters")

        frame_idx = 0
        imgL, imgR = dataset.get_stereo_frame(frame_idx)

        imgL_color = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
        imgR_color = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)

        stereo_view = cv2.hconcat([imgL_color, imgR_color])

        for y in range(0, stereo_view.shape[0], 25):
            cv2.line(stereo_view, (0, y), (stereo_view.shape[1], y), (0, 255, 0), 1)

        cv2.imshow("Stereo Rectification Test (Press ESC to close)", stereo_view)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Test Failed: {e}")