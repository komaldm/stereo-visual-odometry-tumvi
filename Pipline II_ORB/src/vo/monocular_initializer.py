from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.geometry.essential import estimate_essential_and_pose
from src.geometry.projection import make_projection_matrix
from src.geometry.triangulation import triangulate_points, filter_triangulated_points


@dataclass
class MonocularInitializationResult:
    R: np.ndarray
    t: np.ndarray
    pts1_inliers: np.ndarray
    pts2_inliers: np.ndarray
    triangulated_points_3d: np.ndarray
    reproj_err1: np.ndarray
    reproj_err2: np.ndarray
    parallax_deg: np.ndarray
    num_inliers: int
    num_valid_3d: int


class MonocularInitializer:
    def __init__(self, K: np.ndarray):
        self.K = K

    def initialize(self, pts1: np.ndarray, pts2: np.ndarray) -> MonocularInitializationResult:
        essential_result = estimate_essential_and_pose(
            pts1=pts1,
            pts2=pts2,
            K=self.K,
            ransac_prob=0.999,
            ransac_threshold=1.0,
        )

        R1 = np.eye(3, dtype=np.float64)
        t1 = np.zeros((3, 1), dtype=np.float64)

        R2 = essential_result.R
        t2 = essential_result.t

        P1 = make_projection_matrix(self.K, R1, t1)
        P2 = make_projection_matrix(self.K, R2, t2)

        points_3d = triangulate_points(
            P1=P1,
            P2=P2,
            pts1=essential_result.pts1_inliers,
            pts2=essential_result.pts2_inliers,
        )

        tri_result = filter_triangulated_points(
            points_3d=points_3d,
            pts1=essential_result.pts1_inliers,
            pts2=essential_result.pts2_inliers,
            K=self.K,
            R2=R2,
            t2=t2,
            max_depth=100.0,
            max_reproj_error=3.0,
            min_parallax_deg=0.5,
        )

        return MonocularInitializationResult(
            R=R2,
            t=t2,
            pts1_inliers=tri_result.pts1,
            pts2_inliers=tri_result.pts2,
            triangulated_points_3d=tri_result.points_3d,
            reproj_err1=tri_result.reproj_err1,
            reproj_err2=tri_result.reproj_err2,
            parallax_deg=tri_result.parallax_deg,
            num_inliers=essential_result.num_inliers,
            num_valid_3d=tri_result.num_valid,
        )