import cv2
import numpy as np

class FeatureDetector:
    def __init__(self):
        # Tuned specifically for indoor environments like room2
        self.detector = cv2.ORB_create(
            nfeatures=3000,       # Increased to ensure enough points survive RANSAC
            scaleFactor=1.2,
            nlevels=8,
            fastThreshold=20      # Lower threshold to find features on blank walls
        )

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_and_match(self, img1, img2):
        kp1, des1 = self.detector.detectAndCompute(img1, None)
        kp2, des2 = self.detector.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return None, None, [], [], []

        matches = self.matcher.knnMatch(des1, des2, k=2)

        good_matches = []
        pts1 = []
        pts2 = []

        # FIXED: Safely unpack only if exactly 2 neighbors are found
        for match in matches:
            if len(match) == 2:
                m, n = match
                # Lowe's Ratio Test
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
                    pts1.append(kp1[m.queryIdx].pt)
                    pts2.append(kp2[m.trainIdx].pt)

        # Convert to numpy arrays immediately for OpenCV functions
        return kp1, kp2, good_matches, np.float32(pts1), np.float32(pts2)