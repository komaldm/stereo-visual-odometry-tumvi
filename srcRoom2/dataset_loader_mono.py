import os
import cv2
import numpy as np
import yaml


class TumDataset:
    def __init__(self, dataset_path, cam_name="cam0", use_undistort=True):
        self.dataset_path = dataset_path
        self.cam_name = cam_name
        self.use_undistort = use_undistort

        self.calib_path = os.path.join(dataset_path, "dso", "camchain.yaml")
        if not os.path.exists(self.calib_path):
            raise FileNotFoundError(f"Calibration file not found: {self.calib_path}")

        with open(self.calib_path, "r") as f:
            data = yaml.safe_load(f)

        if cam_name not in data:
            raise KeyError(f"Camera '{cam_name}' not found in camchain.yaml")

        cam = data[cam_name]

        intrinsics = cam.get("intrinsics", None)
        if intrinsics is None or len(intrinsics) != 4:
            raise ValueError("Expected 4 intrinsics: fx, fy, cx, cy")

        fx, fy, cx, cy = map(float, intrinsics)

        self.K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        self.D = np.array(cam.get("distortion_coeffs", []), dtype=np.float64).reshape(-1)
        self.image_size = tuple(cam.get("resolution", [0, 0]))
        if len(self.image_size) != 2:
            raise ValueError("Invalid image resolution in calibration file")

        self.model_type = cam.get("camera_model", cam.get("model_type", "")).lower()

        img_folder = os.path.join(dataset_path, "mav0", cam_name, "data")
        if not os.path.isdir(img_folder):
            raise FileNotFoundError(f"Image folder not found: {img_folder}")

        self.image_files = sorted(
            os.path.join(img_folder, f)
            for f in os.listdir(img_folder)
            if f.lower().endswith(".png")
        )

        if len(self.image_files) == 0:
            raise RuntimeError(f"No PNG images found in {img_folder}")

        self.length = len(self.image_files)

        self.map1 = None
        self.map2 = None
        self.new_K = self.K.copy()

        if use_undistort:
            self._build_undistort_maps()

    def _build_undistort_maps(self):
        w, h = self.image_size

        if "fisheye" in self.model_type or len(self.D) == 4:
            self.new_K, _ = cv2.getOptimalNewCameraMatrix(
                self.K, np.zeros(5), (w, h), 0
            )
            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K,
                self.D.reshape(4, 1),
                np.eye(3),
                self.new_K,
                (w, h),
                cv2.CV_16SC2
            )
        else:
            self.new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w, h), 0)
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.K,
                self.D,
                None,
                self.new_K,
                (w, h),
                cv2.CV_16SC2
            )

    def __len__(self):
        return self.length

    def get_timestamp(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError("Frame index out of range")
        filename = os.path.basename(self.image_files[idx])
        return int(filename.replace(".png", "")) / 1e9

    def get_frame(self, idx, undistort=None):
        if idx < 0 or idx >= self.length:
            raise IndexError("Frame index out of range")

        if undistort is None:
            undistort = self.use_undistort

        img = cv2.imread(self.image_files[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {self.image_files[idx]}")

        if undistort:
            if self.map1 is None or self.map2 is None:
                self._build_undistort_maps()
            img = cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

        return img

    def get_stereo_frame(self, idx):
        img = self.get_frame(idx)
        return img, None

    def get_K(self, undistorted=True):
        return self.new_K.copy() if undistorted and self.use_undistort else self.K.copy()

    def info(self):
        print("Dataset path:", self.dataset_path)
        print("Camera:", self.cam_name)
        print("Model:", self.model_type if self.model_type else "unknown")
        print("Image size:", self.image_size)
        print("Frames:", self.length)
        print("K:\n", self.K)
        print("D:", self.D)
        print("Using undistortion:", self.use_undistort)


if __name__ == "__main__":
    dataset_path = r"C:\Users\RoboticsLab\PycharmProjects\stereo_vo_projectR\dataset\dataset-room2_512_16"
    dataset = TumDataset(dataset_path, cam_name="cam0", use_undistort=True)
    dataset.info()

    print("\n--- FPS Check ---")
    if len(dataset) > 1:
        # Get timestamps in seconds
        t0 = dataset.get_timestamp(0)
        t1 = dataset.get_timestamp(1)

        # Calculate time difference
        dt = t1 - t0

        # FPS is 1 divided by the time difference
        fps = 1.0 / dt if dt > 0 else 0

        print(f"Frame 0 Timestamp: {t0:.6f} s")
        print(f"Frame 1 Timestamp: {t1:.6f} s")
        print(f"Delta T (\u0394t):      {dt:.6f} s")
        print(f"Exact Frame Rate:  {fps:.2f} FPS (Hz)")

        # For an even better measure, let's average the whole dataset
        t_last = dataset.get_timestamp(len(dataset) - 1)
        avg_dt = (t_last - t0) / (len(dataset) - 1)
        avg_fps = 1.0 / avg_dt if avg_dt > 0 else 0
        print(f"Average Frame Rate (All Frames): {avg_fps:.2f} FPS (Hz)")
    else:
        print("Not enough frames to calculate FPS.")