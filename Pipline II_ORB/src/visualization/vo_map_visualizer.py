from __future__ import annotations

from pathlib import Path

from src.visualization.vo_visualizer import VOVisualizer


class VOMapVisualizer(VOVisualizer):
    def __init__(
        self,
        traj_size: tuple[int, int] = (700, 700),
        scale: float = 15.0,
        window_name: str = "VIBOT - Visual Odometry Live",
        video_path: str | Path | None = None,
        fps: float = 20.0,
        show_window: bool = True,
    ):
        super().__init__(
            traj_size=traj_size,
            scale=scale,
            window_name=window_name,
            video_path=video_path,
            fps=fps,
            show_window=show_window,
        )
