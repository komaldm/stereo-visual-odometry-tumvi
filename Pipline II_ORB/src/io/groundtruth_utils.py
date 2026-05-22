from __future__ import annotations

import numpy as np


def gt_poses_to_xyz_array(poses) -> np.ndarray:
    if not poses:
        return np.empty((0, 3), dtype=np.float64)

    arr = np.array([[p.tx, p.ty, p.tz] for p in poses], dtype=np.float64)
    return arr