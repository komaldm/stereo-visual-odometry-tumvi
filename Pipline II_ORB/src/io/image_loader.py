from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def load_image(image_path: str | Path, grayscale: bool = True) -> np.ndarray:
    image_path = str(image_path)

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(image_path, flag)

    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    return image