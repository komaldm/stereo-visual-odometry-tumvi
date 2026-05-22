from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VOVisualizer:
    def __init__(
        self,
        traj_size: tuple[int, int] = (700, 700),
        scale: float = 15.0,
        window_name: str = "VIBOT - Visual Odometry Live",
        video_path: str | Path | None = None,
        fps: float = 20.0,
        show_window: bool = True,
    ):
        self.traj_w, self.traj_h = traj_size
        self.scale = scale
        self.window_name = window_name

        self.canvas_w = 1250
        self.canvas_h = 750
        self.font = cv2.FONT_HERSHEY_COMPLEX

        self.frames: list[int] = []
        self.active_counts: list[int] = []
        self.inlier_counts: list[int] = []
        self.map_counts: list[int] = []
        self.trajectory_x: list[float] = []
        self.trajectory_z: list[float] = []
        self.gt_x: list[float] = []
        self.gt_z: list[float] = []

        self._show_window = show_window
        if self._show_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self._video_path = Path(video_path) if video_path is not None else None
        self._fps = fps
        self._video_writer: cv2.VideoWriter | None = None

    @staticmethod
    def _points_from_keypoints(keypoints: list) -> np.ndarray:
        if not keypoints:
            return np.empty((0, 2), dtype=np.float32)
        return np.array([kp.pt for kp in keypoints], dtype=np.float32)

    def _draw_graph(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        x_data: list[float],
        y_data: list[float],
        title: str,
        color: tuple[int, int, int],
        is_scatter: bool = False,
        gt_x_data: list[float] | None = None,
        gt_z_data: list[float] | None = None,
    ) -> None:
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 0), 1)
        cv2.putText(canvas, title, (x + 12, y + 25), self.font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        if len(x_data) < 2:
            return

        gt_x_data = gt_x_data or []
        gt_z_data = gt_z_data or []
        pl, pr, pt, pb = 60, 20, 40, 45

        all_x = list(x_data) + list(gt_x_data)
        all_z = list(y_data) + list(gt_z_data)
        min_x, max_x = min(all_x), max(all_x)
        min_z, max_z = min(all_z), max(all_z)

        if max_x == min_x:
            max_x += 0.1
        if max_z == min_z:
            max_z += 0.1

        bx = (max_x - min_x) * 0.1
        bz = (max_z - min_z) * 0.1
        min_x -= bx
        max_x += bx
        min_z -= bz
        max_z += bz

        def get_pt(vx: float, vz: float) -> tuple[int, int]:
            px = int(x + pl + (vx - min_x) / (max_x - min_x) * (w - pl - pr))
            py = int(y + h - pb - (vz - min_z) / (max_z - min_z) * (h - pt - pb))
            return px, py

        num_ticks = 6
        for i in range(num_ticks):
            val_x = min_x + i * (max_x - min_x) / (num_ticks - 1)
            px, _ = get_pt(val_x, min_z)
            cv2.line(canvas, (px, y + h - pb), (px, y + h - pb + 5), (0, 0, 0), 1)
            cv2.putText(canvas, f"{val_x:.1f}", (px - 18, y + h - 15), self.font, 0.35, (60, 60, 60), 1, cv2.LINE_AA)

            val_z = min_z + i * (max_z - min_z) / (num_ticks - 1)
            _, py = get_pt(min_x, val_z)
            cv2.line(canvas, (x + pl - 5, py), (x + pl, py), (0, 0, 0), 1)
            cv2.putText(canvas, f"{val_z:.1f}", (x + 8, py + 5), self.font, 0.35, (60, 60, 60), 1, cv2.LINE_AA)

        if gt_x_data:
            pts_gt = [get_pt(vx, vz) for vx, vz in zip(gt_x_data, gt_z_data)]
            for i in range(1, len(pts_gt)):
                cv2.line(canvas, pts_gt[i - 1], pts_gt[i], (210, 235, 210), 2)

        pts = [get_pt(vx, vz) for vx, vz in zip(x_data, y_data)]
        if is_scatter:
            for pt in pts:
                cv2.circle(canvas, pt, 2, color, -1)
        else:
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], color, 1)

    def draw(
        self,
        image: np.ndarray,
        keypoints: list,
        poses_wc: list[np.ndarray],
        keyframe_indices: list[int],
        map_points: np.ndarray,
        new_map_points: np.ndarray,
        gt_positions: np.ndarray | None = None,
        frame_idx: int = 0,
        pnp_inliers: int | None = None,
        candidate_points: np.ndarray | None = None,
    ) -> np.ndarray:
        if len(image.shape) == 2:
            left_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            left_vis = image.copy()

        if candidate_points is not None:
            for pt in candidate_points:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(left_vis, (x, y), 2, (0, 100, 255), -1)

        active_points = self._points_from_keypoints(keypoints)
        for pt in active_points:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(left_vis, (x, y), 2, (0, 255, 0), -1)

        if len(poses_wc) > 0:
            p = poses_wc[-1][:3, 3]
            self.trajectory_x.append(float(p[0]))
            self.trajectory_z.append(float(p[2]))

        if gt_positions is not None and len(gt_positions) > 0:
            gt_idx = min(frame_idx, len(gt_positions) - 1)
            self.gt_x.append(float(gt_positions[gt_idx, 0]))
            self.gt_z.append(float(gt_positions[gt_idx, 2]))

        self.frames.append(frame_idx)
        self.active_counts.append(len(active_points))
        self.inlier_counts.append(0 if pnp_inliers is None else int(pnp_inliers))
        self.map_counts.append(0 if map_points is None else int(len(map_points)))

        canvas = np.full((self.canvas_h, self.canvas_w, 3), 255, dtype=np.uint8)

        m_side, m_top, gap = 25, 25, 20
        top_h, bot_h = 260, 320
        cam_w = 700
        local_w = self.canvas_w - cam_w - (2 * m_side) - gap
        bot_panel_w = (self.canvas_w - 2 * m_side - 2 * gap) // 3

        cv2.rectangle(canvas, (m_side, m_top), (m_side + cam_w, m_top + top_h), (0, 0, 0), 1)
        cv2.putText(canvas, "Landmarks & Candidates", (m_side + 12, m_top + 22), self.font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        img_draw = cv2.resize(left_vis, (cam_w - 20, top_h - 45))
        canvas[m_top + 35:m_top + 35 + img_draw.shape[0], m_side + 10:m_side + 10 + img_draw.shape[1]] = img_draw

        self._draw_graph(
            canvas,
            (m_side + cam_w + gap, m_top, local_w, top_h),
            self.trajectory_x[-40:],
            self.trajectory_z[-40:],
            "Local Trajectory (40f)",
            color=(200, 0, 0),
            is_scatter=True,
        )

        by = m_top + top_h + gap
        self._draw_graph(
            canvas,
            (m_side, by, bot_panel_w, bot_h),
            self.frames,
            self.map_counts,
            "Active Landmarks",
            color=(150, 75, 0),
        )
        self._draw_graph(
            canvas,
            (m_side + bot_panel_w + gap, by, bot_panel_w, bot_h),
            self.frames,
            self.inlier_counts,
            "PnP Inliers",
            color=(0, 100, 150),
        )
        self._draw_graph(
            canvas,
            (m_side + 2 * bot_panel_w + 2 * gap, by, bot_panel_w, bot_h),
            self.trajectory_x,
            self.trajectory_z,
            "Global (VO vs GT)",
            color=(0, 0, 150),
            is_scatter=True,
            gt_x_data=self.gt_x,
            gt_z_data=self.gt_z,
        )

        if self._show_window:
            cv2.imshow(self.window_name, canvas)

        if self._video_path is not None:
            if self._video_writer is None:
                self._video_path.parent.mkdir(parents=True, exist_ok=True)
                self._video_writer = cv2.VideoWriter(
                    str(self._video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self._fps,
                    (self.canvas_w, self.canvas_h),
                )
            self._video_writer.write(canvas)

        return canvas

    def release(self) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            print(f"Saved visualization video: {self._video_path}")
        if self._show_window:
            cv2.destroyWindow(self.window_name)
