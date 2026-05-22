from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class RectificationResult:
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    map1x: np.ndarray
    map1y: np.ndarray
    map2x: np.ndarray
    map2y: np.ndarray


def compute_stereo_rectification_fisheye(
    K1: np.ndarray,
    d1: np.ndarray,
    K2: np.ndarray,
    d2: np.ndarray,
    image_size: tuple[int, int],
    T_cam1_cam0: np.ndarray,
) -> RectificationResult:
    """
    Rectification for pinhole + equidistant (fisheye) cameras.

    image_size must be (width, height)
    T_cam1_cam0 is the transform from cam0 to cam1.
    """
    R = T_cam1_cam0[:3, :3]
    t = T_cam1_cam0[:3, 3]

    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K1=K1,
        D1=d1,
        K2=K2,
        D2=d2,
        imageSize=image_size,
        R=R,
        tvec=t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=image_size,
        balance=0.0,
        fov_scale=1.0,
    )

    map1x, map1y = cv2.fisheye.initUndistortRectifyMap(
        K=K1,
        D=d1,
        R=R1,
        P=P1,
        size=image_size,
        m1type=cv2.CV_32FC1,
    )

    map2x, map2y = cv2.fisheye.initUndistortRectifyMap(
        K=K2,
        D=d2,
        R=R2,
        P=P2,
        size=image_size,
        m1type=cv2.CV_32FC1,
    )

    return RectificationResult(
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        map1x=map1x,
        map1y=map1y,
        map2x=map2x,
        map2y=map2y,
    )


def rectify_stereo_pair(
    left_img: np.ndarray,
    right_img: np.ndarray,
    rectification: RectificationResult,
) -> tuple[np.ndarray, np.ndarray]:
    left_rect = cv2.remap(left_img, rectification.map1x, rectification.map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_img, rectification.map2x, rectification.map2y, cv2.INTER_LINEAR)
    return left_rect, right_rect