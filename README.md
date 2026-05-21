# Monocular vs. Stereo Visual Odometry Evaluatio

This repository contains the source code, evaluation metrics, and trajectory analysis tools for a comparative study between Classical Monocular and Strict Keyframe-Interlock Stereo Visual Odometry pipelines, specifically evaluated against high-rotation, sparse-texture environments.

## Repository Architecture

The codebase is structured into three primary directories corresponding to the evaluation datasets. Each directory is entirely self-contained and houses the complete execution pipeline (Monocular VO, Stereo VO, and their respective trajectory evaluation scripts). The core logic remains consistent, but the entry points are isolated to cleanly manage the specific ground truth alignments and results for each unique dataset.

## Dataset Configuration

Due to size limitations (approximately 15GB combined), the raw image sequences and sensor data are not included in this repository. To execute the pipelines, you must download the datasets and place them in the correct relative locations.

**1. Download the Dataset**
Download the required sequence (e.g., `dataset-outdoors5_512_16` or `dataset-room2_512_16`) from the official TUM VI dataset provider.

**2. Place in the Root Directory**
Create a root-level directory named `dataset/` and extract your downloaded sequences into it. Ensure the extracted folders match the naming conventions expected by the scripts.

## 🛠️ Pipeline Architecture

### 1. Monocular VO Backbone (Stage 1)
The baseline pipeline extracts undistorted pinhole features and establishes  frame-to-frame temporal correspondences.

![Monocular Pipeline](asset/Monocular_pipeline.png)

### 2. Mono-Extended Stereo & Metric Correction (Stages 2 & 3)
When a keyframe is triggered by parallax thresholds, the right stereo frame is read to calculate horizontal disparity and inject absolute metric scale into the global map.

$$Z = \frac{fb}{d}$$

A localized, non-linear optimization step immediately stabilizes the updated metric structure with low computational overhead.

![Stereo Pipeline](asset/Stereo_pipeline.png)

## 📦 Dependencies & Packages

The project is built using a clean, standard Python configuration optimized for fast numerical execution on edge devices:

* **Python 3.x**
* **OpenCV (`opencv-python`):** Used for lens distortion mapping, Shi-Tomasi corner extraction, LK optical flow tracking, and initial EPnP RANSAC pose estimation.
* **SciPy (`scipy.optimize`):** Utilizes `least_squares` with robust loss configurations to handle non-linear pose refinement, local bundle adjustment, and Huber-loss optimization loops.
* **NumPy:** Handles underlying matrix operations, frame transformations, vector operations, and coordinate projections.
* **Matplotlib:** (Optional) For real-time 2D/3D trajectory plotting and error evaluation tracking.

## 💻 Hardware Verification Environment

------To be added later----

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


## 🎬 Execution Videos

Below are the screen recordings showing the real-time tracking performance, feature bucketing, and trajectory maps across the indoor and outdoor datasets.

### Room 2 MONO VO Sequence
<video src="assest/Videos/Mono_video.mp4" width="100%" controls></video>

### ROOM 2 STEREO VO Sequence
<video src="assestsVideos/VO_Stereo_Room2.mp4" width="100%" controls></video>

### Outdoor 5 Sequence
<video src="Videos/Outdoor5_execution.mp4" width="100%" controls></video>
