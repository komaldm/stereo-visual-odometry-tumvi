from __future__ import annotations

from pathlib import Path
import sys
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.io.dataset_loader import TumVIDatasetLoader
from src.io.calibration_loader import TumVICalibrationLoader
from src.io.image_loader import load_image
from src.stereo.stereo_frame import StereoFrameProcessor
from src.stereo.disparity import StereoSGBMConfig, normalize_disparity_for_vis


def normalize_depth_for_vis(depth: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(depth, dtype=np.uint8)
    if np.count_nonzero(valid_mask) == 0:
        return out

    d = depth[valid_mask]
    vmin = float(np.min(d))
    vmax = float(np.max(d))

    if vmax - vmin < 1e-6:
        out[valid_mask] = 255
        return out

    scaled = (depth - vmin) / (vmax - vmin)
    out[valid_mask] = np.clip(255.0 * scaled[valid_mask], 0, 255).astype(np.uint8)
    return out


def main():
    dataset_root = Path(r"D:\New folder\project\dataset-room2_512_16\dataset-room2_512_16")

    dataset = TumVIDatasetLoader(dataset_root)
    calib = TumVICalibrationLoader(dataset_root / "dso" / "camchain.yaml").load()

    frame_idx = 100
    frame = dataset[frame_idx]

    left_img = load_image(frame.left_image_path, grayscale=False)
    right_img = load_image(frame.right_image_path, grayscale=False)

    if left_img is None or right_img is None:
        raise RuntimeError("Failed to load stereo images.")

    processor = StereoFrameProcessor(
        K_left=calib.cam0.intrinsics,
        baseline_m=calib.baseline_m,
        sgbm_config=StereoSGBMConfig(
            min_disparity=0,
            num_disparities=64,
            block_size=7,
            uniqueness_ratio=10,
            speckle_window_size=100,
            speckle_range=2,
        ),
        max_depth_m=15.0,
    )

    result = processor.process(left_img, right_img)

    valid_mask = result.reconstruction.valid_mask
    disparity_vis = normalize_disparity_for_vis(result.disparity, valid_mask)
    depth_vis = normalize_depth_for_vis(result.reconstruction.depth, valid_mask)

    disparity_color = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)

    output_dir = PROJECT_ROOT / "outputs" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    left_vis_path = output_dir / "room2_stereo_left.png"
    disparity_path = output_dir / "room2_stereo_disparity.png"
    depth_path = output_dir / "room2_stereo_depth.png"

    cv2.imwrite(str(left_vis_path), left_img)
    cv2.imwrite(str(disparity_path), disparity_color)
    cv2.imwrite(str(depth_path), depth_color)

    valid_depths = result.reconstruction.depth[valid_mask]

    print("=" * 60)
    print("Stereo depth test")
    print("=" * 60)
    print(f"Frame index:            {frame_idx}")
    print(f"Timestamp:              {frame.timestamp_ns}")
    print(f"Baseline [m]:           {calib.baseline_m:.6f}")
    print(f"Valid depth pixels:     {int(np.count_nonzero(valid_mask))}")
    if valid_depths.size > 0:
        print(f"Min depth [m]:          {float(np.min(valid_depths)):.3f}")
        print(f"Median depth [m]:       {float(np.median(valid_depths)):.3f}")
        print(f"Mean depth [m]:         {float(np.mean(valid_depths)):.3f}")
        print(f"Max depth [m]:          {float(np.max(valid_depths)):.3f}")
    print(f"Saved left image:       {left_vis_path}")
    print(f"Saved disparity image:  {disparity_path}")
    print(f"Saved depth image:      {depth_path}")


if __name__ == "__main__":
    main()
