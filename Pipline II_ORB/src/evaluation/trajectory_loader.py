from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Trajectory:
    timestamps: np.ndarray   # shape (N,)
    positions: np.ndarray    # shape (N, 3)
    quaternions: np.ndarray  # shape (N, 4) [qx, qy, qz, qw]


def load_tum_trajectory(path: str | Path) -> Trajectory:
    path = Path(path)
    data = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            vals = line.split()
            if len(vals) != 8:
                continue

            ts = float(vals[0])
            tx, ty, tz = map(float, vals[1:4])
            qx, qy, qz, qw = map(float, vals[4:8])

            data.append([ts, tx, ty, tz, qx, qy, qz, qw])

    if len(data) == 0:
        raise ValueError(f"No valid trajectory entries found in: {path}")

    arr = np.array(data, dtype=np.float64)

    return Trajectory(
        timestamps=arr[:, 0],
        positions=arr[:, 1:4],
        quaternions=arr[:, 4:8],
    )