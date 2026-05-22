from __future__ import annotations

import cv2
import numpy as np


def draw_horizontal_epipolar_lines(
    left_img: np.ndarray,
    right_img: np.ndarray,
    line_spacing: int = 40
) -> np.ndarray:
    if len(left_img.shape) == 2:
        left_vis = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
    else:
        left_vis = left_img.copy()

    if len(right_img.shape) == 2:
        right_vis = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
    else:
        right_vis = right_img.copy()

    combined = np.hstack([left_vis, right_vis])
    h, w = combined.shape[:2]

    for y in range(0, h, line_spacing):
        cv2.line(combined, (0, y), (w - 1, y), (0, 255, 0), 1)

    return combined