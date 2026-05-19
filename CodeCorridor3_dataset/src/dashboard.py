import cv2
import numpy as np


class RealTimeVODashboard:
    def __init__(self, render_skip=2, save_video=True, video_name="VO_Final_Result1.mp4"):
        self.canvas_w = 1250
        self.canvas_h = 750
        self.bg_color = (255, 255, 255)

        self.render_skip = render_skip
        self.counter = 0
        self.font = cv2.FONT_HERSHEY_COMPLEX  # Academic Serif Font

        self.frames, self.num_landmarks, self.repro_errs = [], [], []
        self.trajectory_x, self.trajectory_z = [], []
        self.gt_x, self.gt_z = [], []

        # --- VIDEO WRITER SETUP ---
        self.save_video = save_video
        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Set to 15.0 or 20.0 depending on your desired playback speed
            self.video_writer = cv2.VideoWriter(video_name, fourcc, 15.0, (self.canvas_w, self.canvas_h))
            print(f"Video Recording Started: {video_name}")

    def _draw_graph(self, canvas, rect, x_data, y_data, title, color=(200, 0, 0), is_scatter=False, gt_x_data=None,
                    gt_z_data=None):
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 0), 1)
        cv2.putText(canvas, title, (x + 12, y + 25), self.font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        if len(x_data) < 2: return

        pl, pr, pt, pb = 60, 20, 40, 45

        all_x = list(x_data) + (list(gt_x_data) if gt_x_data else [])
        all_z = list(y_data) + (list(gt_z_data) if gt_z_data else [])
        min_x, max_x = min(all_x), max(all_x)
        min_z, max_z = min(all_z), max(all_z)

        if max_x == min_x: max_x += 0.1
        if max_z == min_z: max_z += 0.1

        bx, bz = (max_x - min_x) * 0.1, (max_z - min_z) * 0.1
        min_x -= bx;
        max_x += bx;
        min_z -= bz;
        max_z += bz

        def get_pt(vx, vz):
            px = int(x + pl + (vx - min_x) / (max_x - min_x) * (w - pl - pr))
            py = int(y + h - pb - (vz - min_z) / (max_z - min_z) * (h - pt - pb))
            return (px, py)

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

        if is_scatter:
            for vx, vz in zip(x_data, y_data):
                cv2.circle(canvas, get_pt(vx, vz), 2, color, -1)
        else:
            pts = [get_pt(vx, vy) for vx, vy in zip(x_data, y_data)]
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], color, 1)

    def update(self, frame_idx, img_gray, active_2d, cand_2d, tvec, rmse, curr_gt_xyz=None):
        self.frames.append(frame_idx)
        self.num_landmarks.append(len(active_2d))
        self.repro_errs.append(rmse)
        self.trajectory_x.append(float(tvec[0, 0]))
        self.trajectory_z.append(float(tvec[2, 0]))
        if curr_gt_xyz is not None:
            self.gt_x.append(float(curr_gt_xyz[0]))
            self.gt_z.append(float(curr_gt_xyz[2]))

        self.counter += 1
        if self.counter % self.render_skip != 0: return

        canvas = np.full((self.canvas_h, self.canvas_w, 3), 255, dtype=np.uint8)

        m_side, m_top, gap = 25, 25, 20
        top_h, bot_h = 260, 320
        cam_w = 700
        local_w = self.canvas_w - cam_w - (2 * m_side) - gap
        bot_panel_w = (self.canvas_w - 2 * m_side - 2 * gap) // 3

        # Panel 1: Camera
        cv2.rectangle(canvas, (m_side, m_top), (m_side + cam_w, m_top + top_h), (0, 0, 0), 1)
        cv2.putText(canvas, "Landmarks & Candidates", (m_side + 12, m_top + 22), self.font, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)
        img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        for p in cand_2d: cv2.circle(img_color, (int(p[0]), int(p[1])), 2, (0, 100, 255), -1)
        for p in active_2d: cv2.circle(img_color, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
        img_draw = cv2.resize(img_color, (cam_w - 20, top_h - 45))
        canvas[m_top + 35:m_top + 35 + img_draw.shape[0], m_side + 10:m_side + 10 + img_draw.shape[1]] = img_draw

        # Panel 2: Local Trajectory
        self._draw_graph(canvas, (m_side + cam_w + gap, m_top, local_w, top_h),
                         self.trajectory_x[-40:], self.trajectory_z[-40:], "Local Trajectory (40f)", color=(200, 0, 0),
                         is_scatter=True)

        by = m_top + top_h + gap
        # Panel 3, 4, 5: Bottom Row
        self._draw_graph(canvas, (m_side, by, bot_panel_w, bot_h), self.frames, self.num_landmarks, "Active Landmarks",
                         color=(150, 75, 0))
        self._draw_graph(canvas, (m_side + bot_panel_w + gap, by, bot_panel_w, bot_h), self.frames, self.repro_errs,
                         "Reprojection Error (px)", color=(0, 100, 150))
        self._draw_graph(canvas, (m_side + 2 * bot_panel_w + 2 * gap, by, bot_panel_w, bot_h),
                         self.trajectory_x, self.trajectory_z, "Global (VO vs GT)",
                         color=(0, 0, 150), is_scatter=True, gt_x_data=self.gt_x, gt_z_data=self.gt_z)

        # Write to Video
        if self.save_video:
            self.video_writer.write(canvas)

        cv2.imshow("VIBOT - Visual Odometry Live", canvas)
        cv2.waitKey(1)

    def close(self):
        if self.save_video:
            self.video_writer.release()
            print("Video Saved Successfully.")