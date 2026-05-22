from __future__ import annotations

import numpy as np


def make_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    if t.ndim == 1:
        t = t.reshape(3, 1)

    Rt = np.hstack([R, t])
    return K @ Rt


def project_points(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
    Project 3D points into image using projection matrix P.
    points_3d: (N, 3)
    returns: (N, 2)
    """
    num_points = points_3d.shape[0]
    pts_h = np.hstack([points_3d, np.ones((num_points, 1), dtype=np.float64)])  # (N, 4)
    proj = (P @ pts_h.T).T  # (N, 3)
    proj = proj[:, :2] / proj[:, 2:3]
    return proj


def compute_reprojection_errors(P: np.ndarray, points_3d: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
    """
    Euclidean reprojection error per point in pixels.
    """
    projected = project_points(P, points_3d)
    errors = np.linalg.norm(projected - points_2d, axis=1)
    return errors