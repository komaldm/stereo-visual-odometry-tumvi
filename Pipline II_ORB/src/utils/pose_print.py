from __future__ import annotations

import numpy as np


def rotation_matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """
    Returns approximate XYZ Euler angles in radians.
    Mainly for debugging / interpretation.
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0

    return np.array([x, y, z], dtype=np.float64)