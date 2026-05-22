"""
Dataset-specific configuration profiles.

Each profile contains parameters for both Stereo VO and Monocular VO
that are tuned for the characteristics of that dataset.

Allowed to tune (per user spec):
  - ORB feature count, matching ratio
  - Disparity filtering thresholds, depth min/max
  - Stereo/mono reprojection thresholds, inlier thresholds
  - Relocalization, triangulation, parallax, map pruning thresholds
  - Fallback trigger timing, retry tolerance
  - Motion validation thresholds
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class StereoVOProfile:
    # Depth range
    min_depth_m: float = 0.4
    max_depth_m: float = 12.0
    # Disparity filter (features with disparity <= min_disparity are discarded)
    # max usable depth = baseline * focal / min_disparity  (≈19.3m at 1.0px for this hardware)
    min_disparity: float = 1.0
    # ORB
    orb_nfeatures: int = 2000
    # Primary tracker
    min_tracker_inliers: int = 25
    reprojection_error_px: float = 2.0
    # Recovery tracker
    min_recovery_inliers: int = 15
    recovery_reprojection_px: float = 2.0
    # Loose retry (passed through StereoPnPTracker override)
    loose_reprojection_px: float = 3.5
    loose_min_inliers: int = 8
    # 3D-3D fallback
    min_3d3d_inliers: int = 12
    # SGBM
    num_disparities: int = 64
    block_size: int = 5
    uniqueness_ratio: int = 12
    speckle_window_size: int = 100
    speckle_range: int = 2


@dataclass
class MonoVOProfile:
    # ORB frontend
    orb_nfeatures: int = 2000
    # ORBMatcher
    ratio_thresh: float = 0.70
    # PnP
    min_pnp_inliers: int = 20
    # Essential matrix init
    min_essential_inliers: int = 40
    # Keyframe
    keyframe_translation_thresh: float = 0.25
    keyframe_rotation_thresh_deg: float = 5.0
    min_keyframe_gap: int = 4
    # Relocalization
    min_relocalization_inliers: int = 14
    min_lost_relocalization_inliers: int = 14
    min_lost_relocalization_inlier_ratio: float = 0.10
    # Recovery probation
    recovery_probation_length: int = 5
    recovery_min_local_inliers: int = 18
    # Rebuild
    min_rebuild_baseline: float = 0.75
    min_rebuild_match_count: int = 50
    min_strong_inliers_for_rebuild_anchor: int = 30
    max_fallback_frames_before_rebuild: int = 10
    # Fallback
    max_cv_fallback_frames: int = 8
    # Map / local window
    local_map_window: int = 80
    # Motion validator
    max_translation_step: float = 2.0
    max_rotation_step_deg: float = 25.0
    # Post-relocalization grace
    post_relocalization_grace_frames: int = 8
    # Initialization search range (wide search max offset)
    init_max_offset: int = 60


@dataclass
class DatasetProfile:
    name: str
    dataset_root: Path
    gt_path: Path
    stereo: StereoVOProfile = field(default_factory=StereoVOProfile)
    mono: MonoVOProfile = field(default_factory=MonoVOProfile)


# ─── Room2 (baseline, tuned for indoor room) ────────────────────────────────
ROOM2 = DatasetProfile(
    name="room2",
    dataset_root=PROJECT_ROOT / "dataset-room2_512_16",
    gt_path=PROJECT_ROOT / "dataset-room2_512_16" / "mav0" / "mocap0" / "data.csv",
    stereo=StereoVOProfile(
        min_depth_m=0.4,
        max_depth_m=12.0,
        orb_nfeatures=2000,
        min_tracker_inliers=25,
        reprojection_error_px=2.0,
        min_recovery_inliers=15,
        recovery_reprojection_px=2.0,
        loose_reprojection_px=3.5,
        loose_min_inliers=8,
        min_3d3d_inliers=12,
        num_disparities=64,
        block_size=5,
        uniqueness_ratio=12,
        speckle_window_size=100,
        speckle_range=2,
    ),
    mono=MonoVOProfile(
        orb_nfeatures=2000,
        ratio_thresh=0.70,
        min_pnp_inliers=20,
        min_essential_inliers=40,
        keyframe_translation_thresh=0.25,
        keyframe_rotation_thresh_deg=5.0,
        min_keyframe_gap=4,
        min_relocalization_inliers=14,
        min_lost_relocalization_inliers=14,
        min_lost_relocalization_inlier_ratio=0.10,
        recovery_probation_length=5,
        recovery_min_local_inliers=18,
        min_rebuild_baseline=0.75,
        min_rebuild_match_count=50,
        min_strong_inliers_for_rebuild_anchor=30,
        max_fallback_frames_before_rebuild=10,
        max_cv_fallback_frames=8,
        local_map_window=80,
        max_translation_step=2.0,
        max_rotation_step_deg=25.0,
        post_relocalization_grace_frames=8,
        init_max_offset=60,
    ),
)

# ─── Corridor3 tuned profile ──────────────────────────────────────────────────
#
# Corridor3 characteristics vs Room2:
#   - 5802 frames (2× longer), 20 FPS, 290s
#   - Starts near-stationary for first 7s → init pair at frame 137
#   - Severe texture variation: ORB kp drops to 210-365 in low-texture zones
#   - Dark frames (mean brightness ~15 in some sections)
#   - Higher inter-frame motion at some sections (mean abs diff ~21 px vs ~3 at start)
#   - Calibration nearly identical to room2 (same camera body, same baseline 0.1011m)
#   - GT has 230-second gap (mocap lost mid-sequence); evaluation on ~1200 matched frames
#
# Baseline failure analysis:
#   - Stereo: 100% tracking with baseline room2 params (no changes needed)
#   - Monocular: lost at frame 371 (PnP inliers 13-15 < threshold 20)
#               stuck after frame 377 (0 relocalization successes in 1000+ frames)
#     Root causes:
#       1. min_pnp_inliers=20 too strict for low-texture frames (measured 13-15)
#       2. Relocalization blocked: pnp_inliers 7-11 < min=14; drift → motion gate rejects
#       3. min_rebuild_baseline=0.75 too strict; map stays stale at 858 points
#
# Tuning rationale (one change per root cause):
#   min_pnp_inliers: 20→12     prevents loss at 13-15 inlier frames (primary fix)
#   ratio_thresh: 0.70→0.75   more matches in low-texture corridor sections
#   min_essential_inliers: 40→32 easier essential fallback in low-texture zones
#   min_relocalization_inliers: 14→10  allows recovery with 7-11 inlier frames
#   recovery_min_local_inliers: 18→12  consistent with lowered PnP threshold
#   min_rebuild_baseline: 0.75→0.25   corridor traversal builds baseline faster
#   min_rebuild_match_count: 50→28    low texture → fewer available matches
#   min_strong_inliers_for_rebuild_anchor: 30→18  consistent with PnP threshold
#   max_fallback_frames_before_rebuild: 10→15  more patience before declaring lost
#   max_cv_fallback_frames: 8→12      more CV frames before giving up
#   local_map_window: 80→120          larger local map helps in corridor (linear motion)
#   max_translation_step: 2.0→5.0    allows reloc to correct accumulated drift
#   max_rotation_step_deg: 25.0→40.0 allows larger rotational corrections on reloc
#   post_relocalization_grace_frames: 8→12  more grace after reloc
CORRIDOR3 = DatasetProfile(
    name="corridor3",
    dataset_root=PROJECT_ROOT / "dataset-corridor3_512_16",
    gt_path=PROJECT_ROOT / "dataset-corridor3_512_16" / "mav0" / "mocap0" / "data.csv",
    stereo=StereoVOProfile(
        # Stereo: room2 params work perfectly (100% tracking in baseline)
        min_depth_m=0.4,
        max_depth_m=12.0,
        orb_nfeatures=2000,
        min_tracker_inliers=25,
        reprojection_error_px=2.0,
        min_recovery_inliers=15,
        recovery_reprojection_px=2.0,
        loose_reprojection_px=3.5,
        loose_min_inliers=8,
        min_3d3d_inliers=12,
        num_disparities=64,
        block_size=5,
        uniqueness_ratio=12,
        speckle_window_size=100,
        speckle_range=2,
    ),
    mono=MonoVOProfile(
        orb_nfeatures=2000,
        ratio_thresh=0.75,            # 0.70→0.75: more matches in low-texture zones
        min_pnp_inliers=12,           # 20→12: CRITICAL — prevents loss at 13-15 inlier frames
        min_essential_inliers=32,     # 40→32: easier essential fallback
        keyframe_translation_thresh=0.25,
        keyframe_rotation_thresh_deg=5.0,
        min_keyframe_gap=4,
        min_relocalization_inliers=10,      # 14→10: allows recovery with 7-11 inlier frames
        min_lost_relocalization_inliers=10, # 14→10
        min_lost_relocalization_inlier_ratio=0.08,  # 0.10→0.08
        recovery_probation_length=5,
        recovery_min_local_inliers=12,       # 18→12: consistent with PnP threshold
        min_rebuild_baseline=0.25,           # 0.75→0.25: corridor builds baseline faster
        min_rebuild_match_count=28,          # 50→28: low texture → fewer matches
        min_strong_inliers_for_rebuild_anchor=12,  # 30→12: more frames qualify as anchors in low-texture zones
        max_fallback_frames_before_rebuild=6,      # 10→6: rebuild while anchor is fresh (~7 frames stale = ~21° rotation)
        max_cv_fallback_frames=12,           # 8→12: more CV before declaring lost
        local_map_window=120,               # 80→120: larger local map for linear corridor
        max_translation_step=5.0,           # 2.0→5.0: allow reloc drift correction
        max_rotation_step_deg=40.0,         # 25.0→40.0: allow larger rot corrections
        post_relocalization_grace_frames=12,# 8→12: more grace after reloc
        init_max_offset=300,               # corridor starts stationary — wide init search
    ),
)

# ─── Outdoors5 profile ───────────────────────────────────────────────────────
#
# Outdoors5 characteristics vs Room2 / Corridor3:
#   - 17 747 frames, 887s, 20 FPS — 6× longer than room2
#   - Excellent texture: ORB p5 = 454 kp, median ≈ 490 kp (no low-texture zones)
#   - No stationary start: 0.5 m motion in first 5s → init_max_offset=60 is fine
#   - High-motion zone at frames 800–1150: 20–40 px/frame inter-frame motion
#   - Outdoor brightness variation: mean brightness 36–76 across sequence
#   - Same camera hardware as room2/corridor3 (fisheye 512×512, 0.101 m baseline)
#   - Faster peak speed: max 2.16 m/s (vs ~1 m/s corridor3)
#   - GT coverage: only ~15.5% (749s mocap gap in the middle)
#   - Nearly closed loop (start–end distance 0.56 m)
#
# Stereo baseline expectation:
#   Excellent texture → room2 params should already achieve ~100% tracking.
#   Only change worth testing: slightly wider motion tolerance in high-motion zone.
#
# Monocular baseline expectation:
#   - Init fine (immediate motion, no stationary start)
#   - High-motion zone (frames 800–1150) is the primary risk:
#       max_translation_step / max_rotation_step_deg may reject valid large moves
#   - Very long sequence (17 747 frames): map management becomes important;
#       larger local_map_window helps maintain a rich local map over time
#   - Good texture throughout → PnP inlier counts should be healthy (>20)
#     so room2 thresholds for min_pnp_inliers are fine
#   - Outdoor: ratio_thresh can stay at 0.70; no low-texture starvation expected
#   - Rebuilds should work well given dense features
#
# Tuning vs room2 defaults:
#   max_translation_step:   2.0 → 3.0  allow fast outdoor motion without gate rejection
#   max_rotation_step_deg: 25.0 → 35.0 allow larger per-frame rotation in high-speed zone
#   local_map_window:        80 → 100  slightly larger window for long sequence
#   max_cv_fallback_frames:   8 → 10   extra patience before rebuild in outdoor variability
OUTDOORS5 = DatasetProfile(
    name="outdoors5",
    dataset_root=PROJECT_ROOT / "dataset-outdoors5_512_16",
    gt_path=PROJECT_ROOT / "dataset-outdoors5_512_16" / "mav0" / "mocap0" / "data.csv",
    stereo=StereoVOProfile(
        # Outdoor scenes have features at 15-50m; widen depth window accordingly.
        # min_disparity=0.5 → max usable depth ≈ 38.6m (vs 19.3m at default 1.0px)
        # max_depth_m=25.0 caps noisy very-far estimates while keeping 12-25m range
        min_depth_m=0.4,
        max_depth_m=25.0,
        min_disparity=0.5,
        orb_nfeatures=2000,
        min_tracker_inliers=25,
        reprojection_error_px=2.0,
        min_recovery_inliers=15,
        recovery_reprojection_px=2.0,
        loose_reprojection_px=3.5,
        loose_min_inliers=8,
        min_3d3d_inliers=12,
        num_disparities=64,
        block_size=5,
        uniqueness_ratio=12,
        speckle_window_size=100,
        speckle_range=2,
    ),
    mono=MonoVOProfile(
        orb_nfeatures=2000,
        ratio_thresh=0.70,               # good texture — default ratio is fine
        min_pnp_inliers=20,              # room2 default: good texture keeps inliers healthy
        min_essential_inliers=40,        # room2 default
        keyframe_translation_thresh=0.25,
        keyframe_rotation_thresh_deg=5.0,
        min_keyframe_gap=4,
        min_relocalization_inliers=14,
        min_lost_relocalization_inliers=14,
        min_lost_relocalization_inlier_ratio=0.10,
        recovery_probation_length=5,
        recovery_min_local_inliers=18,
        min_rebuild_baseline=0.75,
        min_rebuild_match_count=50,
        min_strong_inliers_for_rebuild_anchor=30,
        max_fallback_frames_before_rebuild=10,
        max_cv_fallback_frames=10,        # 8→10: extra patience for outdoor variability
        local_map_window=100,            # 80→100: larger window for long sequence
        max_translation_step=3.0,        # 2.0→3.0: accommodate fast outdoor motion
        max_rotation_step_deg=35.0,      # 25.0→35.0: allow bigger per-frame rotations
        post_relocalization_grace_frames=8,
        init_max_offset=60,              # no stationary start — default is fine
    ),
)

PROFILES: dict[str, DatasetProfile] = {
    "room2": ROOM2,
    "corridor3": CORRIDOR3,
    "outdoors5": OUTDOORS5,
}


def get_profile(name: str) -> DatasetProfile:
    if name not in PROFILES:
        raise ValueError(f"Unknown dataset profile '{name}'. Available: {list(PROFILES)}")
    return PROFILES[name]
