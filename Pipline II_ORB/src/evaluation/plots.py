from __future__ import annotations

from pathlib import Path
import matplotlib
import numpy as np

# Use a non-interactive backend so plot generation works without Tk.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_trajectory_plot(
    gt_positions: np.ndarray,
    est_positions_aligned: np.ndarray,
    output_path: str | Path,
    title: str = "Room2 Monocular VO vs Ground Truth",
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(gt_positions[:, 0], gt_positions[:, 2], label="Ground Truth", linewidth=2)
    plt.plot(est_positions_aligned[:, 0], est_positions_aligned[:, 2], label="Estimated (aligned)", linewidth=2)
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_ate_plot(
    ate_errors: np.ndarray,
    output_path: str | Path,
    title: str = "ATE per Associated Frame",
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(ate_errors)
    plt.xlabel("Associated frame index")
    plt.ylabel("ATE [m]")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_rpe_plot(
    rpe_errors: np.ndarray,
    output_path: str | Path,
    title: str = "RPE Translation per Segment",
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(rpe_errors)
    plt.xlabel("Segment index")
    plt.ylabel("RPE translation [m]")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
