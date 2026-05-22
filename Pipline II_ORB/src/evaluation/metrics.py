from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ATEResult:
    rmse: float
    mean: float
    median: float
    std: float
    min: float
    max: float
    per_frame_errors: np.ndarray


@dataclass
class RPETranslationResult:
    rmse: float
    mean: float
    median: float
    std: float
    min: float
    max: float
    per_segment_errors: np.ndarray


def compute_ate(gt_positions: np.ndarray, est_positions_aligned: np.ndarray) -> ATEResult:
    errors = np.linalg.norm(gt_positions - est_positions_aligned, axis=1)

    return ATEResult(
        rmse=float(np.sqrt(np.mean(errors**2))),
        mean=float(np.mean(errors)),
        median=float(np.median(errors)),
        std=float(np.std(errors)),
        min=float(np.min(errors)),
        max=float(np.max(errors)),
        per_frame_errors=errors,
    )


def compute_rpe_translation(
    gt_positions: np.ndarray,
    est_positions_aligned: np.ndarray,
    delta: int = 10,
) -> RPETranslationResult:
    """
    Translation-only RPE over a fixed frame gap.
    """
    if len(gt_positions) <= delta:
        raise ValueError("Not enough frames to compute RPE.")

    errors = []

    for i in range(len(gt_positions) - delta):
        gt_rel = gt_positions[i + delta] - gt_positions[i]
        est_rel = est_positions_aligned[i + delta] - est_positions_aligned[i]
        err = np.linalg.norm(gt_rel - est_rel)
        errors.append(err)

    errors = np.array(errors, dtype=np.float64)

    return RPETranslationResult(
        rmse=float(np.sqrt(np.mean(errors**2))),
        mean=float(np.mean(errors)),
        median=float(np.median(errors)),
        std=float(np.std(errors)),
        min=float(np.min(errors)),
        max=float(np.max(errors)),
        per_segment_errors=errors,
    )


def compute_end_point_drift(gt_positions: np.ndarray, est_positions_aligned: np.ndarray) -> float:
    return float(np.linalg.norm(gt_positions[-1] - est_positions_aligned[-1]))


def compute_coverage(num_est_total: int, num_associated: int, num_gt_total: int) -> dict:
    return {
        "num_est_total": int(num_est_total),
        "num_gt_total": int(num_gt_total),
        "num_associated": int(num_associated),
        "association_ratio_vs_est": float(num_associated / max(num_est_total, 1)),
        "association_ratio_vs_gt": float(num_associated / max(num_gt_total, 1)),
    }