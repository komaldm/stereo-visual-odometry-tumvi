from __future__ import annotations

import cv2
import numpy as np


def draw_keypoints(image: np.ndarray, keypoints: list) -> np.ndarray:
    return cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def draw_matches(image1, kp1, image2, kp2, matches, max_matches: int = 100) -> np.ndarray:
    matches_to_draw = sorted(matches, key=lambda m: m.distance)[:max_matches]
    vis = cv2.drawMatches(
        image1, kp1,
        image2, kp2,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return vis