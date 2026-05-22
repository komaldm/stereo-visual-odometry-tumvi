from __future__ import annotations

import matplotlib
import numpy as np

# Use a non-interactive backend so plotting works in headless/runtime-only setups.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_point_cloud_3d(points_3d: np.ndarray, title: str = "Triangulated 3D Points") -> None:
    if points_3d.shape[0] == 0:
        print("No 3D points to plot.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=4)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()
