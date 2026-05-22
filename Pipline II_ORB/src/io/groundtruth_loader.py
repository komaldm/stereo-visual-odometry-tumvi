from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import csv
import numpy as np


@dataclass
class GroundTruthPose:
    timestamp_ns: int
    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float


class TumVIGroundTruthLoader:
    """
    Loads Room2 mocap ground truth from mav0/mocap0/data.csv

    Typical expected format:
    #timestamp, tx, ty, tz, qx, qy, qz, qw
    """

    def __init__(self, gt_csv_path: str | Path):
        self.gt_csv_path = Path(gt_csv_path)
        if not self.gt_csv_path.exists():
            raise FileNotFoundError(f"Ground truth CSV not found: {self.gt_csv_path}")

    def load(self) -> List[GroundTruthPose]:
        poses = []

        with open(self.gt_csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if row[0].startswith("#"):
                    continue

                # Adjust this if your actual mocap0/data.csv has a different format
                timestamp_ns = int(row[0].strip())
                tx = float(row[1])
                ty = float(row[2])
                tz = float(row[3])
                qx = float(row[4])
                qy = float(row[5])
                qz = float(row[6])
                qw = float(row[7])

                poses.append(
                    GroundTruthPose(
                        timestamp_ns=timestamp_ns,
                        tx=tx,
                        ty=ty,
                        tz=tz,
                        qx=qx,
                        qy=qy,
                        qz=qz,
                        qw=qw
                    )
                )

        return poses

    def summary(self) -> dict:
        poses = self.load()
        return {
            "pose_count": len(poses),
            "first_timestamp_ns": poses[0].timestamp_ns if poses else None,
            "last_timestamp_ns": poses[-1].timestamp_ns if poses else None,
        }