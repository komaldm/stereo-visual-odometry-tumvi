from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class StereoReconstructionResult:
    points_3d: np.ndarray      # (H, W, 3), left camera frame
    depth: np.ndarray          # (H, W)
    valid_mask: np.ndarray     # (H, W) bool


def reconstruct_from_disparity(
    disparity: np.ndarray,
    K_left: np.ndarray,
    baseline_m: float,
    min_disparity: float = 0.5,
    max_depth_m: float = 20.0,
) -> StereoReconstructionResult:
    """
    Reconstruct dense 3D points from rectified disparity using:
        Z = f * B / d
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
    """
    if disparity.ndim != 2:
        raise ValueError("disparity must be a 2D array")

    fx = float(K_left[0, 0])
    fy = float(K_left[1, 1])
    cx = float(K_left[0, 2])
    cy = float(K_left[1, 2])

    h, w = disparity.shape
    u_coords, v_coords = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    valid_disp = disparity > min_disparity
    depth = np.zeros_like(disparity, dtype=np.float32)
    depth[valid_disp] = (fx * baseline_m) / disparity[valid_disp]

    valid_mask = valid_disp & np.isfinite(depth) & (depth > 0.0) & (depth < max_depth_m)

    X = np.zeros_like(depth, dtype=np.float32)
    Y = np.zeros_like(depth, dtype=np.float32)
    Z = depth.copy()

    X[valid_mask] = (u_coords[valid_mask] - cx) * Z[valid_mask] / fx
    Y[valid_mask] = (v_coords[valid_mask] - cy) * Z[valid_mask] / fy

    points_3d = np.stack([X, Y, Z], axis=-1)
    return StereoReconstructionResult(points_3d=points_3d, depth=depth, valid_mask=valid_mask)


def sample_valid_points(
    points_3d: np.ndarray,
    valid_mask: np.ndarray,
    colors_bgr: np.ndarray | None = None,
    stride: int = 4,
) -> tuple[np.ndarray, np.ndarray | None]:
    ys, xs = np.where(valid_mask)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32), None

    keep = np.arange(0, len(xs), stride)
    ys = ys[keep]
    xs = xs[keep]

    pts = points_3d[ys, xs].astype(np.float32)

    cols = None
    if colors_bgr is not None:
        cols = colors_bgr[ys, xs].astype(np.uint8)

    return pts, cols