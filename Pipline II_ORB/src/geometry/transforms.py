from __future__ import annotations

import numpy as np


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def camera_center_from_T_wc(T_wc: np.ndarray) -> np.ndarray:
    return T_wc[:3, 3].copy()


def world_to_camera_from_T_wc(T_wc: np.ndarray) -> np.ndarray:
    return invert_transform(T_wc)


def compose_transforms(T_a_b: np.ndarray, T_b_c: np.ndarray) -> np.ndarray:
    return T_a_b @ T_b_c