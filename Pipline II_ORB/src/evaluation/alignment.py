from __future__ import annotations

import numpy as np


def umeyama_alignment_sim3(X: np.ndarray, Y: np.ndarray):
    """
    Align X to Y using Sim(3) with Umeyama.
    X, Y: shape (N, 3)

    Returns:
        s, R, t
    such that:
        Y ≈ s * R @ X + t
    """
    assert X.shape == Y.shape
    n = X.shape[0]

    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)

    Xc = X - mu_X
    Yc = Y - mu_Y

    cov = (Yc.T @ Xc) / n

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)

    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt

    var_X = np.mean(np.sum(Xc**2, axis=1))
    scale = np.trace(np.diag(D) @ S) / var_X

    t = mu_Y - scale * (R @ mu_X)

    return scale, R, t


def apply_sim3(positions: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (scale * (R @ positions.T)).T + t.reshape(1, 3)


def umeyama_alignment_se3(X: np.ndarray, Y: np.ndarray):
    """
    Align X to Y using a rigid transform with fixed scale 1.
    X, Y: shape (N, 3)

    Returns:
        R, t
    such that:
        Y ~= R @ X + t
    """
    assert X.shape == Y.shape

    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)

    Xc = X - mu_X
    Yc = Y - mu_Y

    cov = Yc.T @ Xc
    U, _, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt
    t = mu_Y - R @ mu_X
    return R, t


def apply_se3(positions: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ positions.T).T + t.reshape(1, 3)
