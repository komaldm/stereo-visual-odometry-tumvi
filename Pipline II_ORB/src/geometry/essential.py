from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class EssentialResult:
    E: np.ndarray
    R: np.ndarray
    t: np.ndarray
    inlier_mask: np.ndarray
    pts1_inliers: np.ndarray
    pts2_inliers: np.ndarray
    num_inliers: int


def estimate_essential_and_pose(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    ransac_prob: float = 0.999,
    ransac_threshold: float = 1.0,
) -> EssentialResult:
    """
    Estimate essential matrix and recover relative pose.

    pts1, pts2: matched 2D points in pixel coordinates, shape (N, 2)
    K: camera intrinsic matrix
    """
    if pts1.shape[0] < 8 or pts2.shape[0] < 8:
        raise ValueError("At least 8 correspondences are required.")

    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        cameraMatrix=K,
        method=cv2.RANSAC,
        prob=ransac_prob,
        threshold=ransac_threshold,
    )

    if E is None or mask is None:
        raise RuntimeError("Essential matrix estimation failed.")

    _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, cameraMatrix=K, mask=mask)

    inlier_mask = pose_mask.ravel().astype(bool)
    pts1_inliers = pts1[inlier_mask]
    pts2_inliers = pts2[inlier_mask]

    return EssentialResult(
        E=E,
        R=R,
        t=t,
        inlier_mask=inlier_mask,
        pts1_inliers=pts1_inliers,
        pts2_inliers=pts2_inliers,
        num_inliers=int(np.sum(inlier_mask)),
    )