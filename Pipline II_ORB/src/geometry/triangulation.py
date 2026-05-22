from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from src.geometry.projection import compute_reprojection_errors


@dataclass
class TriangulationResult:
    points_3d: np.ndarray
    pts1: np.ndarray
    pts2: np.ndarray
    valid_mask: np.ndarray
    reproj_err1: np.ndarray
    reproj_err2: np.ndarray
    parallax_deg: np.ndarray
    num_valid: int


def triangulate_points(
    P1: np.ndarray,
    P2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    pts4d_h = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (pts4d_h[:3, :] / pts4d_h[3, :]).T
    return pts3d


def transform_world_to_camera(points_w: np.ndarray, T_c_w: np.ndarray) -> np.ndarray:
    pts_h = np.hstack([points_w, np.ones((points_w.shape[0], 1), dtype=np.float64)])
    pts_c = (T_c_w @ pts_h.T).T
    return pts_c[:, :3]


def compute_parallax_angles_deg_from_centers(points_w: np.ndarray, C1_w: np.ndarray, C2_w: np.ndarray) -> np.ndarray:
    v1 = points_w - C1_w.reshape(1, 3)
    v2 = points_w - C2_w.reshape(1, 3)

    n1 = np.linalg.norm(v1, axis=1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=1, keepdims=True)

    valid = (n1[:, 0] > 1e-12) & (n2[:, 0] > 1e-12)

    cosang = np.ones(points_w.shape[0], dtype=np.float64)
    cosang[valid] = np.sum(v1[valid] * v2[valid], axis=1) / (n1[valid, 0] * n2[valid, 0])
    cosang = np.clip(cosang, -1.0, 1.0)

    return np.degrees(np.arccos(cosang))


def filter_triangulated_points_world(
    points_w: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    T_c1_w: np.ndarray,
    T_c2_w: np.ndarray,
    C1_w: np.ndarray,
    C2_w: np.ndarray,
    max_depth: float = 100.0,
    max_reproj_error: float = 3.0,
    min_parallax_deg: float = 0.5,
) -> TriangulationResult:
    pts_c1 = transform_world_to_camera(points_w, T_c1_w)
    pts_c2 = transform_world_to_camera(points_w, T_c2_w)

    z1 = pts_c1[:, 2]
    z2 = pts_c2[:, 2]

    finite_mask = np.isfinite(points_w).all(axis=1)
    positive_depth_mask = (z1 > 0.0) & (z2 > 0.0)
    depth_range_mask = (z1 < max_depth) & (z2 < max_depth)

    reproj_err1 = compute_reprojection_errors(P1, points_w, pts1)
    reproj_err2 = compute_reprojection_errors(P2, points_w, pts2)
    reproj_mask = (reproj_err1 < max_reproj_error) & (reproj_err2 < max_reproj_error)

    parallax_deg = compute_parallax_angles_deg_from_centers(points_w, C1_w, C2_w)
    parallax_mask = parallax_deg > min_parallax_deg

    valid_mask = finite_mask & positive_depth_mask & depth_range_mask & reproj_mask & parallax_mask

    return TriangulationResult(
        points_3d=points_w[valid_mask],
        pts1=pts1[valid_mask],
        pts2=pts2[valid_mask],
        valid_mask=valid_mask,
        reproj_err1=reproj_err1[valid_mask],
        reproj_err2=reproj_err2[valid_mask],
        parallax_deg=parallax_deg[valid_mask],
        num_valid=int(np.sum(valid_mask)),
    )


def filter_triangulated_points(
    points_3d: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    max_depth: float = 100.0,
    max_reproj_error: float = 3.0,
    min_parallax_deg: float = 0.5,
) -> TriangulationResult:
    """
    Compatibility wrapper for two-view monocular initialization.

    Assumes the first camera is the world frame:
      T_c1_w = I
      T_c2_w = [R2 | t2]
    and points_3d are in that world/camera-1 frame.
    """
    if t2.ndim == 1:
        t2 = t2.reshape(3, 1)

    P1 = K @ np.hstack([np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)])
    P2 = K @ np.hstack([R2, t2])

    T_c1_w = np.eye(4, dtype=np.float64)
    T_c2_w = np.eye(4, dtype=np.float64)
    T_c2_w[:3, :3] = R2
    T_c2_w[:3, 3] = t2.reshape(3)

    C1_w = np.zeros(3, dtype=np.float64)
    C2_w = (-R2.T @ t2.reshape(3)).reshape(3)

    return filter_triangulated_points_world(
        points_w=points_3d,
        pts1=pts1,
        pts2=pts2,
        P1=P1,
        P2=P2,
        T_c1_w=T_c1_w,
        T_c2_w=T_c2_w,
        C1_w=C1_w,
        C2_w=C2_w,
        max_depth=max_depth,
        max_reproj_error=max_reproj_error,
        min_parallax_deg=min_parallax_deg,
    )
