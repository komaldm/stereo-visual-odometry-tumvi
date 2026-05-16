import cv2
import numpy as np
from dataset_loader import TumDataset


class FeatureMatcher:
    def __init__(self):
        # Increased features for better coverage in indoor rooms
        self.detector = cv2.ORB_create(
            nfeatures=3000,
            scaleFactor=1.2,
            nlevels=8
        )
        # BFMatcher for ORB (NORM_HAMMING is mandatory)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_and_match(self, img1, img2):
        # 1. Detect and Compute
        kp1, des1 = self.detector.detectAndCompute(img1, None)
        kp2, des2 = self.detector.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return None, None, None, None, None

        # 2. KNN Matching
        matches = self.matcher.knnMatch(des1, des2, k=2)

        # 3. Lowe's Ratio Test (Fixed at 0.75 for ORB)
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 20:
            return None, None, None, None, None

        # 4. Extract points for RANSAC
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        return kp1, kp2, good_matches, pts1, pts2


# -------------------------
# MAIN EXECUTION LOOP
# -------------------------
if __name__ == "__main__":
    # Path from your previous code
    dataset_path = "C:/Users/RoboticsLab/PycharmProjects/stereo_vo_project/dataset/dataset-room2_512_16"

    dataset = TumDataset(dataset_path)
    matcher = FeatureMatcher()
    K = dataset.K

    print("\n--- STARTING MONOCULAR FEATURE TRACKING ---")
    print(f"Total Frames to process: {dataset.length}\n")

    for i in range(0, dataset.length - 1):
        img1 = dataset.get_frame(i)
        img2 = dataset.get_frame(i + 1)

        # Get matches using the improved logic
        kp1, kp2, good, pts1, pts2 = matcher.detect_and_match(img1, img2)

        if good is None:
            print(f"Frame {i}: [FAILED] - Not enough matches found")
            continue

        # 5. Essential Matrix with Robust RANSAC
        # Increased threshold to 1.0 to account for ORB noise
        E, mask = cv2.findEssentialMat(
            pts1, pts2, K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None or mask is None:
            print(f"Frame {i}: [FAILED] - Essential Matrix calculation failed")
            continue

        # Calculate Inliers
        inliers_count = int(mask.ravel().sum())
        inlier_ratio = (inliers_count / len(good)) * 100

        # Print output in your requested format
        print(f"Frame {i} -> {i + 1}")
        print(f"  Good matches: {len(good)}")
        print(f"  RANSAC Inliers: {inliers_count} ({inlier_ratio:.1f}%)")

        # Logic check: If inliers are too low, the trajectory will drift
        if inliers_count < 50:
            print("  [WARNING]: Low inlier count! Possible trajectory jump.")
        print("-" * 30)

        # Visualization
        # We only draw the inliers (mask == 1) to see the "true" matching
        draw_matches = [good[j] for j in range(len(good)) if mask[j]]

        img_visual = cv2.drawMatches(
            img1, kp1,
            img2, kp2,
            draw_matches[:150], None,  # Show top 150 inliers
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        cv2.imshow("Filtered Inliers (RANSAC)", img_visual)

        if cv2.waitKey(1) == 27:  # Press ESC to exit
            break

    cv2.destroyAllWindows()
    print("\nProcessing Complete.")