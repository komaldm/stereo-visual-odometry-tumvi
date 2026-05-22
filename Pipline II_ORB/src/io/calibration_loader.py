from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml
import numpy as np


@dataclass
class CameraCalibration:
    intrinsics: np.ndarray
    distortion: np.ndarray
    resolution: tuple[int, int]   # (width, height)
    camera_model: str
    distortion_model: str
    T_cam_imu: np.ndarray | None = None


@dataclass
class StereoCalibration:
    cam0: CameraCalibration
    cam1: CameraCalibration
    T_cam1_cam0: np.ndarray
    baseline_m: float


class TumVICalibrationLoader:
    def __init__(self, camchain_path: str | Path):
        self.camchain_path = Path(camchain_path)
        if not self.camchain_path.exists():
            raise FileNotFoundError(f"camchain.yaml not found: {self.camchain_path}")

        with open(self.camchain_path, "r") as f:
            self.data = yaml.safe_load(f)

    @staticmethod
    def _to_matrix_4x4(T_list) -> np.ndarray:
        T = np.array(T_list, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"Expected 4x4 transform, got {T.shape}")
        return T

    @staticmethod
    def _build_K(intrinsics) -> np.ndarray:
        fx, fy, cx, cy = intrinsics
        return np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _parse_camera(self, cam_key: str) -> CameraCalibration:
        cam = self.data[cam_key]

        K = self._build_K(cam["intrinsics"])
        distortion = np.array(cam["distortion_coeffs"], dtype=np.float64)
        resolution = tuple(cam["resolution"])
        camera_model = cam["camera_model"]
        distortion_model = cam["distortion_model"]

        T_cam_imu = None
        if "T_cam_imu" in cam:
            T_cam_imu = self._to_matrix_4x4(cam["T_cam_imu"])

        return CameraCalibration(
            intrinsics=K,
            distortion=distortion,
            resolution=resolution,
            camera_model=camera_model,
            distortion_model=distortion_model,
            T_cam_imu=T_cam_imu,
        )

    def load(self) -> StereoCalibration:
        cam0 = self._parse_camera("cam0")
        cam1 = self._parse_camera("cam1")

        if "T_cn_cnm1" not in self.data["cam1"]:
            raise KeyError("cam1 does not contain T_cn_cnm1 in camchain.yaml")

        T_cam1_cam0 = self._to_matrix_4x4(self.data["cam1"]["T_cn_cnm1"])
        baseline_m = float(np.linalg.norm(T_cam1_cam0[:3, 3]))

        return StereoCalibration(
            cam0=cam0,
            cam1=cam1,
            T_cam1_cam0=T_cam1_cam0,
            baseline_m=baseline_m,
        )