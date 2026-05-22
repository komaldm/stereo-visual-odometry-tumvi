from __future__ import annotations

import numpy as np
from src.map.map_point import MapPoint


class MapManager:
    def __init__(self):
        self.map_points: dict[int, MapPoint] = {}
        self.next_point_id = 0

    def add_map_point(
        self,
        position: np.ndarray,
        descriptor: np.ndarray,
        created_in_keyframe: int,
        last_seen_frame: int,
    ) -> int:
        point_id = self.next_point_id
        self.next_point_id += 1

        self.map_points[point_id] = MapPoint(
            id=point_id,
            position=position.astype(np.float64),
            descriptor=descriptor.copy(),
            created_in_keyframe=created_in_keyframe,
            last_seen_frame=last_seen_frame,
        )
        return point_id

    def get_all_valid_points(self) -> list[MapPoint]:
        return [p for p in self.map_points.values() if not p.is_bad]

    def get_positions_and_descriptors(self):
        valid_points = self.get_all_valid_points()

        if not valid_points:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 32), dtype=np.uint8),
                [],
            )

        positions = np.array([p.position for p in valid_points], dtype=np.float64)
        descriptors = np.array([p.descriptor for p in valid_points], dtype=np.uint8)
        ids = [p.id for p in valid_points]

        return positions, descriptors, ids

    def get_local_positions_and_descriptors(
        self,
        current_frame_idx: int,
        recent_frame_window: int = 20,
    ):
        local_points = [
            p for p in self.get_all_valid_points()
            if (current_frame_idx - p.last_seen_frame) <= recent_frame_window
        ]

        if not local_points:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 32), dtype=np.uint8),
                [],
            )

        positions = np.array([p.position for p in local_points], dtype=np.float64)
        descriptors = np.array([p.descriptor for p in local_points], dtype=np.uint8)
        ids = [p.id for p in local_points]

        return positions, descriptors, ids

    def get_relocalization_positions_and_descriptors(
        self,
        max_points: int = 400,
        min_observations: int = 2,
        max_consecutive_misses: int = 40,
        max_distance: float = 80.0,
    ):
        candidates = []

        for pid, mp in self.map_points.items():
            if mp.is_bad:
                continue
            if mp.descriptor is None:
                continue
            if mp.observation_count < min_observations:
                continue
            if mp.consecutive_misses > max_consecutive_misses:
                continue

            dist = float(np.linalg.norm(mp.position))
            if dist > max_distance:
                continue

            score = (
                10.0 * float(mp.observation_count)
                - 1.5 * float(mp.consecutive_misses)
                + 0.01 * float(mp.last_seen_frame)
            )
            candidates.append((score, pid, mp))

        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:max_points]

        if not candidates:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 32), dtype=np.uint8),
                [],
            )

        positions = np.array([c[2].position for c in candidates], dtype=np.float64)
        descriptors = np.array([c[2].descriptor for c in candidates], dtype=np.uint8)
        ids = [c[1] for c in candidates]

        return positions, descriptors, ids

    def get_points_for_keyframe_hypothesis(
        self,
        keyframe_idx: int,
        frame_window: int = 40,
        max_points: int = 400,
        min_observations: int = 1,
        max_consecutive_misses: int = 80,
        max_distance: float = 80.0,
        anchor_point_ids: list[int] | None = None,
    ):
        anchor_ids = set(anchor_point_ids or [])
        candidates = []

        for pid, mp in self.map_points.items():
            if mp.is_bad:
                continue
            if mp.descriptor is None:
                continue
            if mp.observation_count < min_observations:
                continue
            if mp.consecutive_misses > max_consecutive_misses:
                continue

            dist = float(np.linalg.norm(mp.position))
            if dist > max_distance:
                continue

            created_delta = abs(mp.created_in_keyframe - keyframe_idx) if mp.created_in_keyframe >= 0 else 10**9
            seen_delta = abs(mp.last_seen_frame - keyframe_idx) if mp.last_seen_frame >= 0 else 10**9
            near_keyframe = created_delta <= frame_window or seen_delta <= frame_window
            anchored = pid in anchor_ids

            if not near_keyframe and not anchored:
                continue

            score = (
                10.0 * float(mp.observation_count)
                - 1.5 * float(mp.consecutive_misses)
                - 0.05 * float(min(created_delta, seen_delta))
            )
            if anchored:
                score += 25.0

            candidates.append((score, pid, mp))

        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:max_points]

        if not candidates:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 32), dtype=np.uint8),
                [],
            )

        positions = np.array([c[2].position for c in candidates], dtype=np.float64)
        descriptors = np.array([c[2].descriptor for c in candidates], dtype=np.uint8)
        ids = [c[1] for c in candidates]

        return positions, descriptors, ids

    def mark_all_points_old(self) -> None:
        for p in self.map_points.values():
            p.is_new = False

    def mark_seen(self, point_ids: list[int], frame_idx: int) -> None:
        for pid in point_ids:
            if pid in self.map_points:
                self.map_points[pid].last_seen_frame = frame_idx

    def increment_seen(self, point_ids: list[int], frame_idx: int) -> None:
        for pid in point_ids:
            if pid in self.map_points:
                p = self.map_points[pid]
                p.last_seen_frame = frame_idx
                p.observation_count += 1
                p.consecutive_misses = 0

    def increment_misses_for_unseen(self, seen_ids: set[int]) -> None:
        for pid, p in self.map_points.items():
            if p.is_bad:
                continue
            if pid not in seen_ids:
                p.consecutive_misses += 1

    def prune_points(
        self,
        min_observations: int = 2,
        max_consecutive_misses: int = 20,
    ) -> int:
        removed = 0
        for p in self.map_points.values():
            if p.is_bad:
                continue
            if p.observation_count < min_observations and p.consecutive_misses > 5:
                p.is_bad = True
                removed += 1
            elif p.consecutive_misses > max_consecutive_misses:
                p.is_bad = True
                removed += 1
        return removed

    def has_nearby_point(self, position: np.ndarray, distance_thresh: float = 0.10) -> bool:
        valid_points = self.get_all_valid_points()
        if not valid_points:
            return False

        pts = np.array([p.position for p in valid_points], dtype=np.float64)
        dists = np.linalg.norm(pts - position.reshape(1, 3), axis=1)
        return np.any(dists < distance_thresh)
