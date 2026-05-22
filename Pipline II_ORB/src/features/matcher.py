from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class MatchResult:
    good_matches: list
    pts1: np.ndarray          # shape (N, 2)
    pts2: np.ndarray          # shape (N, 2)
    query_indices: np.ndarray # indices into kp1/des1
    train_indices: np.ndarray # indices into kp2/des2


class ORBMatcher:
    def __init__(
        self,
        ratio_thresh: float = 0.70,
        mutual_check: bool = True,
    ):
        self.ratio_thresh = ratio_thresh
        self.mutual_check = mutual_check
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def _empty_result(self) -> MatchResult:
        return MatchResult(
            good_matches=[],
            pts1=np.empty((0, 2), dtype=np.float32),
            pts2=np.empty((0, 2), dtype=np.float32),
            query_indices=np.empty((0,), dtype=np.int32),
            train_indices=np.empty((0,), dtype=np.int32),
        )

    def _ratio_filter(self, knn_matches: list[list[cv2.DMatch]]) -> list[cv2.DMatch]:
        good = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_thresh * n.distance:
                good.append(m)
        return good

    def match(self, kp1, des1, kp2, des2) -> MatchResult:
        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return self._empty_result()

        knn_12 = self.matcher.knnMatch(des1, des2, k=2)
        good_12 = self._ratio_filter(knn_12)

        if len(good_12) == 0:
            return self._empty_result()

        if self.mutual_check:
            knn_21 = self.matcher.knnMatch(des2, des1, k=2)
            good_21 = self._ratio_filter(knn_21)
            reverse_pairs = {(m.queryIdx, m.trainIdx) for m in good_21}

            good_12 = [
                m for m in good_12
                if (m.trainIdx, m.queryIdx) in reverse_pairs
            ]

            if len(good_12) == 0:
                return self._empty_result()

        best_for_train = {}
        for m in good_12:
            prev = best_for_train.get(m.trainIdx)
            if prev is None or m.distance < prev.distance:
                best_for_train[m.trainIdx] = m

        good_matches = list(best_for_train.values())
        good_matches.sort(key=lambda m: (m.queryIdx, m.trainIdx, m.distance))

        if len(good_matches) == 0:
            return self._empty_result()

        query_indices = np.array([m.queryIdx for m in good_matches], dtype=np.int32)
        train_indices = np.array([m.trainIdx for m in good_matches], dtype=np.int32)
        pts1 = np.array([kp1[i].pt for i in query_indices], dtype=np.float32)
        pts2 = np.array([kp2[i].pt for i in train_indices], dtype=np.float32)

        return MatchResult(
            good_matches=good_matches,
            pts1=pts1,
            pts2=pts2,
            query_indices=query_indices,
            train_indices=train_indices,
        )
