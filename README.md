# Monocular vs. Stereo Visual Odometry Evaluation

This repository contains the source code, evaluation metrics, and trajectory analysis tools for a comparative study between Classical Monocular and Strict Keyframe-Interlock Stereo Visual Odometry pipelines, specifically evaluated against high-rotation, sparse-texture environments.

## Repository Architecture

The codebase is structured into three primary directories corresponding to the evaluation datasets. Each directory is entirely self-contained and houses the complete execution pipeline (Monocular VO, Stereo VO, and their respective trajectory evaluation scripts). The core logic remains consistent, but the entry points are isolated to cleanly manage the specific ground truth alignments and results for each unique dataset.

## Dataset Configuration

Due to size limitations (approximately 15GB combined), the raw image sequences and sensor data are not included in this repository. To execute the pipelines, you must download the datasets and place them in the correct relative locations.

**1. Download the Dataset**
Download the required sequence (e.g., `dataset-outdoors5_512_16` or `dataset-room2_512_16`) from the official TUM VI dataset provider.

**2. Place in the Root Directory**
Create a root-level directory named `dataset/` and extract your downloaded sequences into it. Ensure the extracted folders match the naming conventions expected by the scripts.

## 📦 Dependencies & Packages

The project is built using a clean, standard Python configuration optimized for fast numerical execution on edge devices:

* **Python 3.x**
* **OpenCV (`opencv-python`):** Used for lens distortion mapping, Shi-Tomasi corner extraction, LK optical flow tracking, and initial EPnP RANSAC pose estimation.
* **SciPy (`scipy.optimize`):** Utilizes `least_squares` with robust loss configurations to handle non-linear pose refinement, local bundle adjustment, and Huber-loss optimization loops.
* **NumPy:** Handles underlying matrix operations, frame transformations, vector operations, and coordinate projections.
* **Matplotlib:** (Optional) For real-time 2D/3D trajectory plotting and error evaluation tracking.

## 💻 Hardware Verification Environment

------To be added later----

## 🛠️ Pipeline Architecture
The proposed framework evaluates two visual odometry paradigms: a baseline Monocular VO pipeline and a Mono-Extended Stereo VO pipeline with metric-scale correction.
### 1. Monocular VO Backbone (Stage 1)

The baseline pipeline extracts undistorted pinhole features and establishes frame-to-frame temporal correspondences.

<p align="center">
  <img src="asset/Monocular_pipeline.png" width="350" height="600">
</p>

---
The monocular backbone performs sparse feature tracking using the Pyramidal Lucas–Kanade (KLT) optical flow framework. Shi–Tomasi corners are extracted and refined to sub-pixel accuracy before temporal tracking.

To improve tracking reliability:

* Forward–backward optical flow verification removes inconsistent correspondences.
* Border filtering rejects unstable edge features.
* EPnP + RANSAC pose estimation computes the relative camera transformation.
* Triangulated landmarks are validated using reprojection error and positive depth constraints.

Keyframes are inserted dynamically using parallax magnitude and feature depletion thresholds. A 4×4 grid bucketing strategy maintains uniform spatial feature distribution across the image plane.

### 2. Mono-Extended Stereo & Metric Correction (Stages 2 & 3)

<p align="center">
  <img src="asset/Stereo_Pipeline.png" width="350" height="600">
</p>
When a keyframe is triggered by parallax thresholds, the right stereo frame is read to calculate horizontal disparity and inject absolute metric scale into the global map.
For every selected keyframe:

* Left-right stereo correspondences are computed using KLT tracking.
* Horizontal disparity is converted into metric depth using the stereo pinhole model:

$$
Z = \frac{fb}{d}
$$

where:

* \(f\) = focal length  
* \(b\) = stereo baseline  
* \(d\) = disparity

Recovered 3D landmarks overwrite the scale-ambiguous monocular map, eliminating scale drift accumulation.

A lightweight local optimization stage using `scipy.optimize.least_squares()` further refines the pose by minimizing reprojection residuals under robust Huber-loss constraints.

## 📊 Results and Discussion

* The pipelines were evaluated on Room2, Corridor3, and Outdoor5 sequences from the TUM-VI benchmark.

* Monocular VO achieved efficient real-time tracking but accumulated noticeable scale drift in long-range and low-texture environments.

* The stereo framework improved trajectory stability by injecting metric depth during keyframe updates.

* Stereo VO significantly reduced tracking failures, reprojection error, and long-term drift across all datasets.

* The largest improvement was observed in Outdoor5, where stereo depth correction stabilized large-scale trajectory estimation.

* Despite additional disparity computation, the stereo pipeline maintained real-time performance.

## 📈 Quantitative Evaluation

| Metric | Mono VO (Room2) | Stereo VO (Room2) |
| :--- | :---: | :---: |
| Absolute Trajectory Error (ATE) | 0.8696 m | 0.2697 m |
| Relative Pose Error (RPE) | 0.0789 m/frame | 0.0181 m/frame |
| Start-to-End Drift | 0.5130 m | 0.4342 m |
| Drift Percentage | 0.36% | 0.31% |
| Tracking Failures | 169 | 0 |
| Tracking Success Rate | 94.14% | 100% |
| Mean Reprojection Error | 1.203 px | 0.650 px |
| Runtime | 41.25 Hz | 32.97 Hz |

## 🖼️ Trajectory Comparison

| Dataset | Monocular VO | Stereo VO |
| :--- | :---: | :---: |
| **Room2** | <img src="asset/room2_mono.png" width="350"> | <img src="asset/room2_stereo.png" width="350"> |
| **Corridor3** | <img src="asset/corridor3_mono.png" width="350"> | <img src="asset/corridor3_stereo.png" width="350"> |
| **Outdoor5** | <img src="asset/outdoor5_mono.png" width="350"> | <img src="asset/outdoor5_stereo.png" width="350"> |


## ⚖️ Strengths and Limitations

| Pipeline | Strengths | Limitations |
| :--- | :--- | :--- |
| **Monocular VO** | • Low computational cost and fast runtime  <br> • Requires only a single camera  <br> • Can estimate depth using temporal triangulation | • No absolute metric scale  <br> • Accumulates scale drift over time  <br> • Tracking failure in low-texture scenes  <br> • Essential matrix becomes unstable during pure rotation |
| **Stereo VO** | • Direct metric depth from stereo disparity  <br> • Reduced trajectory drift  <br> • Improved tracking robustness  <br> • Better long-term trajectory stability | • Higher computational overhead  <br> • Limited reliable depth range  <br> • Noisy disparity near occlusions/boundaries  <br> • Still affected by long-term drift |

## 🔄 Pipeline Architecture Comparison (ORB vs. KLT)

| Aspect | ORB Pipeline | KLT Pipeline |
| :--- | :--- | :--- |
| Feature Extraction | FAST corners + BRIEF descriptors | Shi–Tomasi corners + sub-pixel refinement |
| Tracking Method | Descriptor matching using Hamming distance | Pyramidal Lucas–Kanade optical flow |
| Filtering Strategy | Lowe ratio test + cross-checking | Forward–backward error validation |
| Motion Handling | Better for sudden/large motion | Better for smooth continuous motion |
| Relocalization | Supports descriptor-based relocalization | Relies on temporal continuity |
| Precision | Robust but lower local precision | High sub-pixel tracking accuracy |
| Computation Cost | Higher descriptor matching overhead | Lightweight but requires re-seeding |
| Main Weakness | Computationally expensive | Sensitive to sudden motion loss |

## 📊 Dataset Evaluation Summary

The pipeline was validated against three distinct sequences from the **TUM Visual-Inertial (TUM-VI)** benchmark (recorded at 20 Hz in 16-bit HDR):

| Sequence | Environment Type | Primary Challenge | Stereo Impact |
| :--- | :--- | :--- | :--- |
| **Room 2** | Indoor, Feature-Dense | High-speed dynamics | ATE reduced by 56%; tracking failures cut to 0. |
| **Corridor 3** | Narrow Hallway | Low texture, repeating patterns | Drastically stabilized tracking; bounded local optimization. |
| **Outdoor 5** | Large-Scale Mixed | Far-field sub-pixel disparity | Eradicated reinitialization loops; highlights long-range degradation. |

> **Note:** For specific hyperparameter setups per sequence (such as adaptive KLT window sizes, Huber loss scaling thresholds, and custom depth ceilings), refer to the configuration matrices inside the project documentation.



## 🎬 Execution Videos

Below are the screen recordings showing the real-time tracking performance, feature bucketing, and trajectory maps across the indoor and outdoor datasets.

### Room2 Stereo VO Sequence
<video src="assest/Videos/room2.mp4" width="100%" controls></video>

### Outdoors5 STEREO VO Sequence
<video src="assestsVideos/outdoors5.mp4" width="100%" controls></video>

### Outdoor 5 Sequence
<video src="Videos/Outdoor5_execution.mp4" width="100%" controls></video>
