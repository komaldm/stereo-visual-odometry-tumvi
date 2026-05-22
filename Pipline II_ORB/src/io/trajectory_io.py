from __future__ import annotations

from pathlib import Path
import numpy as np


def rotation_matrix_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion [qx, qy, qz, qw].
    """
    q = np.empty(4, dtype=np.float64)

    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[3] = 0.25 / s
        q[0] = (R[2, 1] - R[1, 2]) * s
        q[1] = (R[0, 2] - R[2, 0]) * s
        q[2] = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[3] = (R[2, 1] - R[1, 2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[3] = (R[0, 2] - R[2, 0]) / s
            q[0] = (R[0, 1] + R[1, 0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[3] = (R[1, 0] - R[0, 1]) / s
            q[0] = (R[0, 2] + R[2, 0]) / s
            q[1] = (R[1, 2] + R[2, 1]) / s
            q[2] = 0.25 * s

    q = q / np.linalg.norm(q)
    return q  # [qx, qy, qz, qw]


def save_trajectory_tum(output_path: str | Path, timestamps_ns: list[int], poses_wc: list[np.ndarray]) -> None:
    """
    Save trajectory in TUM format:
    timestamp tx ty tz qx qy qz qw

    poses_wc are 4x4 camera poses in world frame.
    timestamp is written in seconds.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for ts_ns, T_wc in zip(timestamps_ns, poses_wc):
            t = T_wc[:3, 3]
            q = rotation_matrix_to_quaternion_xyzw(T_wc[:3, :3])

            ts_sec = ts_ns * 1e-9
            line = f"{ts_sec:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n"
            f.write(line)