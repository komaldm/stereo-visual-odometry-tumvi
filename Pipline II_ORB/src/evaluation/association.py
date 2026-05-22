from __future__ import annotations

import numpy as np


def associate_by_timestamp(
    est_timestamps: np.ndarray,
    gt_timestamps: np.ndarray,
    max_dt: float = 0.02,
):
    """
    Associate estimated timestamps with GT timestamps by nearest neighbor.

    Returns:
        est_indices, gt_indices
    """
    est_indices = []
    gt_indices = []

    gt_idx = 0
    for i, ts_est in enumerate(est_timestamps):
        while gt_idx + 1 < len(gt_timestamps) and gt_timestamps[gt_idx + 1] < ts_est:
            gt_idx += 1

        candidates = [gt_idx]
        if gt_idx + 1 < len(gt_timestamps):
            candidates.append(gt_idx + 1)

        best_j = None
        best_dt = float("inf")

        for j in candidates:
            dt = abs(gt_timestamps[j] - ts_est)
            if dt < best_dt:
                best_dt = dt
                best_j = j

        if best_j is not None and best_dt <= max_dt:
            est_indices.append(i)
            gt_indices.append(best_j)

    return np.array(est_indices, dtype=int), np.array(gt_indices, dtype=int)