import cv2
import numpy as np


class StereoDepthEstimator:
    def __init__(self, focal_length, baseline):
        """
        Initializes the Stereo SGBM algorithm and stores the metric rulers.
        """
        self.f = focal_length
        self.B = baseline

        # --- StereoSGBM Parameters ---
        # These are highly tuned for indoor environments like TUM VI
        window_size = 5
        min_disp = 0
        # numDisparities must be divisible by 16. Higher means closer objects can be seen.
        num_disp = 16 * 6

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=window_size,
            P1=8 * 1 * window_size ** 2,  # Penalty for disparity smoothness (small)
            P2=32 * 1 * window_size ** 2,  # Penalty for disparity smoothness (large)
            disp12MaxDiff=1,
            uniquenessRatio=10,  # Prevents noisy matches
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    def compute_disparity(self, img_left_rect, img_right_rect):
        """
        Calculates the dense disparity map from the rectified image pair.
        """
        # SGBM computes disparity scaled by 16, so we divide by 16.0 to get actual pixels
        disparity = self.stereo.compute(img_left_rect, img_right_rect).astype(np.float32) / 16.0
        return disparity

    def get_3d_point(self, u, v, disparity, cx, cy):
        """
        Implements Rubric Equation (4) and (5):
        Extracts the 3D metric coordinate (X, Y, Z) for a specific pixel (u, v).
        """
        d = disparity[int(v), int(u)]

        # If disparity is 0 or negative, it's invalid (too far away or mismatched)
        if d <= 0.0:
            return None

        # Equation 4: Z = (f * B) / d
        Z = (self.f * self.B) / d

        # Equation 5: X and Y
        X = Z * (u - cx) / self.f
        Y = Z * (v - cy) / self.f

        return np.array([X, Y, Z])


# ==========================================
# UNIT TEST
# ==========================================
if __name__ == "__main__":
    from dataset_loader import TumDataset

    print("--- Testing Stereo Disparity & Depth ---")
    test_path = r"C:\Users\RoboticsLab\PycharmProjects\stereo_vo_project\dataset\dataset-room2_512_16"

    try:
        # 1. Load Dataset
        dataset = TumDataset(test_path)
        imgL, imgR = dataset.get_stereo_frame(0)  # Get first frame

        # We need the principal point (cx, cy) from the rectified matrix K
        cx = dataset.K[0, 2]
        cy = dataset.K[1, 2]

        # 2. Initialize Depth Estimator
        depth_estimator = StereoDepthEstimator(dataset.focal_length, dataset.baseline)

        # 3. Compute Disparity Map
        print("Computing Disparity Map (SGBM)...")
        disparity_map = depth_estimator.compute_disparity(imgL, imgR)

        # 4. Normalize disparity for visualization (makes it look like a depth map)
        disp_vis = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

        # 5. Test a specific point (Let's pick the center of the image)
        u_center, v_center = 256, 256
        point_3d = depth_estimator.get_3d_point(u_center, v_center, disparity_map, cx, cy)

        if point_3d is not None:
            print(f"\n3D Coordinate at Image Center (u=256, v=256):")
            print(f"X: {point_3d[0]:.3f} meters (Right/Left)")
            print(f"Y: {point_3d[1]:.3f} meters (Down/Up)")
            print(f"Z: {point_3d[2]:.3f} meters (Depth/Forward)")

        # Show the Depth Map
        cv2.imshow("Left Camera", imgL)
        cv2.imshow("Disparity Map (Red=Close, Blue=Far)", disp_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Test Failed: {e}")