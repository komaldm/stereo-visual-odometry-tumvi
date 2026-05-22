from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import csv


@dataclass
class StereoFrame:
    timestamp_ns: int
    left_image_path: Path
    right_image_path: Path


class TumVIDatasetLoader:
    """
    Loader for TUM VI Room2 stereo image data.

    Expected structure:
        root/
            mav0/
                cam0/
                    data.csv
                    data/
                cam1/
                    data.csv
                    data/
            dso/
                camchain.yaml
    """

    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root)
        self.mav0_dir = self.dataset_root / "mav0"
        self.cam0_dir = self.mav0_dir / "cam0"
        self.cam1_dir = self.mav0_dir / "cam1"

        self.cam0_csv = self.cam0_dir / "data.csv"
        self.cam1_csv = self.cam1_dir / "data.csv"
        self.cam0_data_dir = self.cam0_dir / "data"
        self.cam1_data_dir = self.cam1_dir / "data"

        self._validate_paths()

        self.left_frames = self._read_image_csv(self.cam0_csv, self.cam0_data_dir)
        self.right_frames = self._read_image_csv(self.cam1_csv, self.cam1_data_dir)
        self.stereo_frames = self._build_stereo_pairs()

    def _validate_paths(self) -> None:
        required_paths = [
            self.dataset_root,
            self.mav0_dir,
            self.cam0_dir,
            self.cam1_dir,
            self.cam0_csv,
            self.cam1_csv,
            self.cam0_data_dir,
            self.cam1_data_dir,
        ]

        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"Required path does not exist: {path}")

    @staticmethod
    def _read_image_csv(csv_path: Path, image_dir: Path) -> dict[int, Path]:
        """
        Reads a TUM VI camera CSV file.
        Expected format:
            #timestamp [ns],filename
            1234567890,1234567890.png
        """
        frames = {}

        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if row[0].startswith("#"):
                    continue

                timestamp_ns = int(row[0].strip())
                filename = row[1].strip()
                image_path = image_dir / filename

                if not image_path.exists():
                    raise FileNotFoundError(f"Image file not found: {image_path}")

                frames[timestamp_ns] = image_path

        return frames

    def _build_stereo_pairs(self) -> List[StereoFrame]:
        """
        Build stereo pairs by exact timestamp matching.
        For TUM VI exported data, cam0 and cam1 are typically synchronized.
        """
        common_timestamps = sorted(set(self.left_frames.keys()) & set(self.right_frames.keys()))

        if not common_timestamps:
            raise RuntimeError("No common timestamps found between cam0 and cam1.")

        stereo_frames = [
            StereoFrame(
                timestamp_ns=ts,
                left_image_path=self.left_frames[ts],
                right_image_path=self.right_frames[ts],
            )
            for ts in common_timestamps
        ]

        return stereo_frames

    def __len__(self) -> int:
        return len(self.stereo_frames)

    def __getitem__(self, idx: int) -> StereoFrame:
        return self.stereo_frames[idx]

    def load(self) -> "TumVIDatasetLoader":
        return self

    def get_all_frames(self) -> List[StereoFrame]:
        return self.stereo_frames

    def summary(self) -> dict:
        return {
            "dataset_root": str(self.dataset_root),
            "left_frame_count": len(self.left_frames),
            "right_frame_count": len(self.right_frames),
            "stereo_pair_count": len(self.stereo_frames),
            "first_timestamp_ns": self.stereo_frames[0].timestamp_ns if self.stereo_frames else None,
            "last_timestamp_ns": self.stereo_frames[-1].timestamp_ns if self.stereo_frames else None,
        }
