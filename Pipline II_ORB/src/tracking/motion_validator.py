from __future__ import annotations

import numpy as np


def rotation_angle_deg(R: np.ndarray) -> float:
    trace = np.trace(R)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


class MotionValidator:
    def __init__(
        self,
        max_translation_step: float = 2.0,
        max_rotation_step_deg: float = 25.0,
    ):
        self.max_translation_step = max_translation_step
        self.max_rotation_step_deg = max_rotation_step_deg

    def is_pose_valid(self, T_prev_wc: np.ndarray, T_curr_wc: np.ndarray) -> tuple[bool, float, float]:
        t_prev = T_prev_wc[:3, 3]
        t_curr = T_curr_wc[:3, 3]

        translation_step = float(np.linalg.norm(t_curr - t_prev))

        R_prev = T_prev_wc[:3, :3]
        R_curr = T_curr_wc[:3, :3]
        R_rel = R_prev.T @ R_curr
        rotation_step_deg = rotation_angle_deg(R_rel)

        valid = (
            translation_step <= self.max_translation_step
            and rotation_step_deg <= self.max_rotation_step_deg
        )

        return valid, translation_step, rotation_step_deg