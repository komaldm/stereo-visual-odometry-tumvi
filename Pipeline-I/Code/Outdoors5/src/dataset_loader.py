import os
import cv2
import numpy as np
import yaml


class UniversalTumDataset:
    """
    Unified Dataset Loader for Hybrid Visual Odometry.
    Handles both pure monocular undistortion and fisheye stereo rectification.
    """

    def __init__(self, dataset_path, fov_scale=0.324):
        self.dataset_path = dataset_path
        self.image_size = (512, 512)

        # Paths for Left (cam0) and Right (cam1) cameras
        self.cam0_path = os.path.join(dataset_path, "mav0", "cam0", "data")
        self.cam1_path = os.path.join(dataset_path, "mav0", "cam1", "data")

        if not os.path.exists(self.cam0_path):
            raise FileNotFoundError(f"Left image directory not found: {self.cam0_path}")

        self.left_image_files = sorted(
            [os.path.join(self.cam0_path, f) for f in os.listdir(self.cam0_path) if f.lower().endswith('.png')])

        # Check if right camera exists (for stereo capability)
        self.is_stereo = os.path.exists(self.cam1_path)
        if self.is_stereo:
            self.right_image_files = sorted(
                [os.path.join(self.cam1_path, f) for f in os.listdir(self.cam1_path) if f.lower().endswith('.png')])
            self.length = min(len(self.left_image_files), len(self.right_image_files))
        else:
            self.length = len(self.left_image_files)

        if self.length == 0:
            raise RuntimeError(f"No PNG images found in dataset path: {dataset_path}")

        # ==========================================
        # DYNAMIC CALIBRATION PARSING
        # ==========================================
        yaml_path = os.path.join(dataset_path, "dso", "camchain.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Could not find camchain.yaml at {yaml_path}")

        with open(yaml_path, 'r') as f:
            calib = yaml.safe_load(f)

        # Left Camera (cam0)
        intr0 = calib['cam0']['intrinsics']
        self.K1 = np.array([[intr0[0], 0, intr0[2]], [0, intr0[1], intr0[3]], [0, 0, 1]], dtype=np.float64)
        self.D1 = np.array(calib['cam0']['distortion_coeffs'], dtype=np.float64).reshape(-1)

        # Default Monocular Projection Matrix (will be overwritten if stereo rectifying)
        self.K = self.K1.copy()

        # ==========================================
        # UNDISTORTION / RECTIFICATION MAPS
        # ==========================================
        if self.is_stereo and calib.get('cam1'):
            # Right Camera (cam1)
            intr1 = calib['cam1']['intrinsics']
            self.K2 = np.array([[intr1[0], 0, intr1[2]], [0, intr1[1], intr1[3]], [0, 0, 1]], dtype=np.float64)
            self.D2 = np.array(calib['cam1']['distortion_coeffs'], dtype=np.float64).reshape(-1)

            # Extrinsics (Transformation from cam0 to cam1)
            T_1_0 = np.array(calib['cam1']['T_cn_cnm1'], dtype=np.float64)
            self.R = T_1_0[0:3, 0:3].copy()
            self.T = T_1_0[0:3, 3:4].copy()

            # --- PERFORM FISHEYE STEREO RECTIFICATION ---
            # Inherited exact fov_scale from your stereo loader
            R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
                self.K1, self.D1.reshape(4, 1), self.K2, self.D2.reshape(4, 1),
                self.image_size, self.R, self.T,
                flags=cv2.CALIB_ZERO_DISPARITY,
                newImageSize=self.image_size,
                balance=0.0,
                fov_scale=fov_scale
            )

            self.map1x, self.map1y = cv2.fisheye.initUndistortRectifyMap(
                self.K1, self.D1.reshape(4, 1), R1, P1, self.image_size, cv2.CV_32FC1)
            self.map2x, self.map2y = cv2.fisheye.initUndistortRectifyMap(
                self.K2, self.D2.reshape(4, 1), R2, P2, self.image_size, cv2.CV_32FC1)

            # Update metric rulers based on stereo rectification
            self.K = P1[:, :3].copy()
            self.focal_length = self.K[0, 0]
            self.baseline = abs(P2[0, 3] / P2[0, 0])
        else:
            # --- STANDARD MONOCULAR UNDISTORTION ---
            w, h = self.image_size
            self.new_K, _ = cv2.getOptimalNewCameraMatrix(self.K1, self.D1, (w, h), 0)

            if len(self.D1) == 4:  # Fisheye
                self.map1x, self.map1y = cv2.fisheye.initUndistortRectifyMap(
                    self.K1, self.D1.reshape(4, 1), np.eye(3), self.new_K, (w, h), cv2.CV_32FC1)
            else:
                self.map1x, self.map1y = cv2.initUndistortRectifyMap(
                    self.K1, self.D1, None, self.new_K, (w, h), cv2.CV_32FC1)

            self.K = self.new_K.copy()
            self.focal_length = self.K[0, 0]
            self.baseline = 0.0

    def __len__(self):
        return self.length

    def get_timestamp(self, index):
        if index < 0 or index >= self.length:
            raise IndexError("Index out of bounds.")
        filename = os.path.basename(self.left_image_files[index])
        return float(filename.replace('.png', '')) / 1e9

    def get_frame(self, index):
        """Returns the undistorted/rectified left frame for Monocular VO."""
        if index < 0 or index >= self.length:
            raise IndexError("Index out of bounds.")
        img_left = cv2.imread(self.left_image_files[index], cv2.IMREAD_GRAYSCALE)
        return cv2.remap(img_left, self.map1x, self.map1y, interpolation=cv2.INTER_LINEAR)

    def get_stereo_frame(self, index):
        """Returns perfectly rectified Left and Right frames for Hybrid/Stereo VO."""
        if not self.is_stereo:
            raise RuntimeError("Requested stereo frame, but no right camera (cam1) data exists.")

        if index < 0 or index >= self.length:
            raise IndexError("Index out of bounds.")

        img_left = cv2.imread(self.left_image_files[index], cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread(self.right_image_files[index], cv2.IMREAD_GRAYSCALE)

        img_left_rect = cv2.remap(img_left, self.map1x, self.map1y, interpolation=cv2.INTER_LINEAR)
        img_right_rect = cv2.remap(img_right, self.map2x, self.map2y, interpolation=cv2.INTER_LINEAR)

        return img_left_rect, img_right_rect