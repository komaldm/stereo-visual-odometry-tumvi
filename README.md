# Sparse Optical Flow Framework for Monocular and Metric Stereo Visual Odometry

This repository implements a classical, geometry-based sparse visual odometry (VO) pipeline evaluated on the TUM-VI dataset. The system features a robust Monocular VO backbone that is dynamically extended into a Metric Stereo VO framework via a **Strict Keyframe Interlock** mechanism. By processing the right stereo stream only at selected keyframes, the system effectively eliminates scale drift while preserving high runtime efficiency on low-power computational edge hardware.

## 🚀 Key Features

* **Purely Geometric & Lightweight:** Operates entirely on classical computer vision techniques without deep learning dependencies.
* **Robust Feature Management:** Utilizes grid bucketing with Shi-Tomasi corner extraction capped at 650 points to eliminate sorting micro-stalls and ensure even spatial coverage.
* **Active Masking:** Employs a 12-pixel radius mask around active map points to force feature re-seeding in unexplored image regions, preventing map starvation.
* **Temporal Tracking Validation:** Implements pyramidal Forward-Backward Lucas-Kanade (LK) optical flow with sub-pixel refinement and border containment filtering.
* **Optimized Pose Estimation:** Computes initial pose via EPnP + RANSAC, followed by non-linear motion refinement minimizing a robust Huber-loss reprojection error.
* **Strict Keyframe Interlock:** Tracks purely monocularly during standard frames and drops scale drift by activating the right camera stream exclusively at keyframes to estimate direct metric depth.
* **Map Hygiene Filters:** Rejects unstable far-field triangulations using adaptive disparity floors and custom depth-bounding windows.

## 🛠️ Pipeline Architecture

### 1. Monocular VO Backbone (Stage 1)
The baseline pipeline extracts undistorted pinhole features and establishes robust frame-to-frame temporal correspondences.

![Monocular Pipeline](assets/Monocular_pipeline.png)

### 2. Mono-Extended Stereo & Metric Correction (Stages 2 & 3)
When a keyframe is triggered by parallax thresholds, the right stereo frame is read to calculate horizontal disparity and inject absolute metric scale into the global map.

$$Z = \frac{fb}{d}$$

A localized, non-linear optimization step immediately stabilizes the updated metric structure with low computational overhead.

![Stereo Pipeline](assets/Stereo_pipeline.png)

## 📦 Dependencies & Packages

The project is built using a clean, standard Python configuration optimized for fast numerical execution on edge devices:

* **Python 3.x**
* **OpenCV (`opencv-python`):** Used for lens distortion mapping, Shi-Tomasi corner extraction, LK optical flow tracking, and initial EPnP RANSAC pose estimation.
* **SciPy (`scipy.optimize`):** Utilizes `least_squares` with robust loss configurations to handle non-linear pose refinement, local bundle adjustment, and Huber-loss optimization loops.
* **NumPy:** Handles underlying matrix operations, frame transformations, vector operations, and coordinate projections.
* **Matplotlib:** (Optional) For real-time 2D/3D trajectory plotting and error evaluation tracking.

## 💻 Hardware Verification Environment

To demonstrate computational efficiency under strict constraints, the pipeline's runtime benchmarks were recorded entirely on low-power consumer edge hardware without GPU acceleration:

* **Processor:** Intel Core i5-10210U CPU (1.60GHz base, ultra-low voltage mobile architecture)
* **Memory:** 8 GB RAM
* **Acceleration:** 100% CPU-bound (Execution of pyramidal LK flow and iterative optimization loops runs entirely without GPU threading)

## 📊 Dataset Evaluation Summary

The pipeline was validated against three distinct sequences from the **TUM Visual-Inertial (TUM-VI)** benchmark (recorded at 20 Hz in 16-bit HDR):

| Sequence | Environment Type | Primary Challenge | Stereo Impact |
| :--- | :--- | :--- | :--- |
| **Room 2** | Indoor, Feature-Dense | High-speed dynamics | ATE reduced by 56%; tracking failures cut to 0. |
| **Corridor 3** | Narrow Hallway | Low texture, repeating patterns | Drastically stabilized tracking; bounded local optimization. |
| **Outdoor 5** | Large-Scale Mixed | Far-field sub-pixel disparity | Eradicated reinitialization loops; highlights long-range degradation. |

> **Note:** For specific hyperparameter setups per sequence (such as adaptive KLT window sizes, Huber loss scaling thresholds, and custom depth ceilings), refer to the configuration matrices inside the project documentation.
> ## 📊 Results and Discussion

The proposed framework was evaluated across three distinct environments from the TUM-VI benchmark, exposing the behavioral trade-offs between a baseline monocular system and the mono-extended stereo pipeline.

* [cite_start]**Room 2 (Indoor, Feature-Dense):** The baseline Monocular VO pipeline struggled with significant tracking instability, registering 154 tracking failures, a high mean reprojection error of 1.148 pixels, and an Absolute Trajectory Error (ATE) of 0.8195 m[cite: 114]. [cite_start]Upgrading to the Mono-Extended Stereo architecture completely eradicated this tracking instability (0 failures), dropped the mean reprojection error to 0.630 pixels, and reduced the ATE by roughly 56% to 0.3615 m[cite: 117, 118, 119]. [cite_start]Relative Pose Error (RPE) similarly saw a 76% drop down to 0.0230 m/frame[cite: 117]. [cite_start]This vast improvement in spatial consistency incurred only a minor computational tax, dropping tracking frequencies slightly from 11.10 Hz to 9.63 Hz[cite: 115, 120].
* [cite_start]**Corridor 3 (Low-Texture Hallway):** Characterized by repeating wall patterns and restricted texture, the monocular system failed heavily, dropping tracking 451 times[cite: 123]. [cite_start]The stereo architecture successfully stabilized the pipeline, minimizing failures to just 5 and reducing the reprojection error to a highly precise 0.619 pixels[cite: 123, 124]. [cite_start]Due to the active left-right feature matching and validation at keyframes, the average runtime decreased from 12.09 Hz to 7.64 Hz[cite: 125, 132]. [cite_start]Interestingly, terminal drift slightly increased from 0.5566 m to 0.6293 m[cite: 126]. [cite_start]This behavior highlights a classic geometric constraint of narrow hallway scenes where features reside far down the corridor; the resulting low-disparity measurements degrade depth accuracy, allowing small forward-motion tracking discrepancies to accumulate over time[cite: 127, 139, 140].
* [cite_start]**Outdoor 5 (Large-Scale Environment):** The monocular baseline's deceptively low terminal drift of 0.4431 m was an artifact of its 1,411 tracking failures, where constant reinitializations prevented long trajectory tracking and artificially masked global drift accumulation[cite: 142, 143, 144]. [cite_start]The stereo pipeline achieved highly stable, continuous tracking (only 5 failures) and maintained a low reprojection error of 0.801 pixels, but accumulated a large terminal drift of 28.5646 m[cite: 145, 147]. [cite_start]This outcome directly exposes the physical boundaries of stereo triangulation [cite: 147][cite_start]: in expansive outdoor environments, far-field features yield sub-pixel disparities[cite: 148, 155]. [cite_start]Consequently, minute temporal matching errors generate massive depth estimation scaling errors that are continuously injected into the active map, compounding tracking drift over long sequences[cite: 149, 150].

## 🎬 Execution Videos

Below are the screen recordings showing the real-time tracking performance, feature bucketing, and trajectory maps across the indoor and outdoor datasets.

### Room 2 MONO VO Sequence
<video src="assest/Videos/Mono_video.mp4" width="100%" controls></video>

### ROOM 2 STEREO VO Sequence
<video src="assestsVideos/VO_Stereo_Room2.mp4" width="100%" controls></video>

### Outdoor 5 Sequence
<video src="Videos/Outdoor5_execution.mp4" width="100%" controls></video>
