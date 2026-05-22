from __future__ import annotations

import cv2
import numpy as np


class FisheyeUndistorter:
    def __init__(
        self,
        K: np.ndarray,
        D: np.ndarray,
        image_size: tuple[int, int],
        balance: float = 0.0,
    ):
        """
        image_size: (width, height)
        """
        self.K = K
        self.D = D
        self.image_size = image_size
        self.balance = balance

        identity_R = np.eye(3, dtype=np.float64)

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K=self.K,
            D=self.D,
            image_size=self.image_size,
            R=identity_R,
            balance=self.balance,
            new_size=self.image_size,
            fov_scale=1.0,
        )

        self.new_K = new_K

        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            K=self.K,
            D=self.D,
            R=identity_R,
            P=self.new_K,
            size=self.image_size,
            m1type=cv2.CV_32FC1,
        )

    def undistort(self, image: np.ndarray) -> np.ndarray:
        return cv2.remap(image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)