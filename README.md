# Semantic-Aware Dynamic Keypoint Filtering and Enhanced Description

This repository implements a semantic-aware keypoint detection and description framework for dynamic outdoor scenes. The method predicts keypoint repeatability and reliability in a single network, removes unreliable keypoints around dynamic objects and boundaries, and produces semantic-geometric-texture fused descriptors for robust matching under viewpoint and illumination changes.

## Contributions

- Reformulate multi-class semantic segmentation as a keypoint reliability binary classification problem and combine it with repeatability for robust keypoint selection.
- Introduce a Semantic-Geometric Fusion Module (SGFM) to fuse geometry, texture, and high-level semantics for discriminative descriptors.
- Propose a semantic-aware ranking loss to improve descriptor matching while preserving local texture diversity.

## Method Overview

The network contains three decoders over a shared encoder:

- Repeatability decoder: grid-based keypoint probability map.
- Reliability decoder: binary reliable/unreliable map, trained from semantic labels (dynamic vs. static).
- Descriptor decoder: produces fused descriptors via SGFM and cross-attention.

Keypoint selection multiplies repeatability and reliability maps, then applies a neighborhood difference filter to remove boundary keypoints before NMS.

### Repeatability

Input images are divided into 8x8 cells and classified into 64 pixel locations plus a no-keypoint class. Cross-entropy loss is used over the 65-way prediction to supervise repeatability.

### Reliability

Semantic labels are reduced to a binary reliable/unreliable mask. A lightweight reliability decoder outputs a 2-class probability map, trained with weighted cross-entropy to address class imbalance.

### Neighborhood Difference Filter

Boundary pixels between reliable and unreliable regions are marked unreliable to suppress unstable keypoints around semantic edges.

### SGFM Descriptor Fusion

Descriptors fuse local texture with keypoint geometry and high-level semantic features. A semantic-guided cross-attention mechanism aggregates global context across keypoints to strengthen semantic discriminability and geometric stability.

### Semantic-Aware Ranking Loss

Training optimizes a differentiable AP-based ranking loss where true geometric matches rank above same-semantic non-matches, which in turn rank above different-semantic negatives.

## Training Details

- Dataset: Cityscapes (240x480)
- Optimizer: Adam
- Batch size: 8
- Learning rate: 0.001
- Weight decay: 0.0005
- Loss weights: lambda_rel = 1.2, lambda_desc = 1.0
- Repeatability is trained separately; reliability and descriptor are trained jointly.

## Datasets and Evaluation

- Aachen Day-Night: visual localization under day/night changes.
- Oxford RobotCar-Seasons: outdoor localization with dynamic traffic and weather changes.
- HPatches: homography estimation and descriptor matching evaluation.

Evaluation follows HLoc protocol with the same keypoint count (4096) and image scaling (long side 1024). HPatches uses nearest-neighbor matching and RANSAC for homography estimation.

## Results

### Aachen Day-Night (localization success rate, %)

| Method | Day 0.25m/2deg | Day 0.5m/5deg | Day 5m/10deg | Night 0.25m/2deg | Night 0.5m/5deg | Night 5m/10deg |
| --- | --- | --- | --- | --- | --- | --- |
| SuperPoint | 82.6 | 90.3 | 97.0 | 77.6 | 85.7 | 95.5 |
| ALIKE | 85.7 | 92.4 | 97.0 | 81.6 | 85.7 | 99.0 |
| SFD2 | 84.2 | 92.0 | 96.7 | 77.8 | 90.9 | 98.3 |
| XFeat | 84.7 | 90.5 | 96.5 | 77.6 | 89.8 | 98.0 |
| LifeFeat | 86.6 | 90.1 | 97.1 | 82.1 | 89.9 | 99.1 |
| R2D2 | 85.3 | 86.7 | 92.2 | 74.8 | 80.5 | 97.8 |
| Ours | 85.4 | 91.0 | 97.2 | 83.6 | 91.7 | 99.3 |

### RobotCar-Seasons (localization success rate, %)

| Method | Day 0.25m/2deg | Day 0.5m/5deg | Day 5m/10deg | Night 0.25m/2deg | Night 0.5m/5deg | Night 5m/10deg |
| --- | --- | --- | --- | --- | --- | --- |
| SuperPoint | 56.5 | 81.5 | 97.1 | 16.9 | 41.6 | 71.5 |
| SFD2 | 55.6 | 80.2 | 96.5 | 17.0 | 42.1 | 70.0 |
| SIFT | 57.5 | 81.9 | 98.3 | 7.8 | 13.9 | 22.1 |
| LifeFeat | 53.5 | 82.1 | 97.6 | 18.0 | 43.0 | 70.9 |
| R2D2 | 57.4 | 81.9 | 97.9 | 18.3 | 43.4 | 72.8 |
| Ours | 57.5 | 82.1 | 98.0 | 18.5 | 44.3 | 73.0 |

### HPatches (homography estimation and descriptor metrics)

| Method | eps=3 | eps=5 | eps=7 | NN mAP | M.score |
| --- | --- | --- | --- | --- | --- |
| SuperPoint | 68.4 | 82.9 | 88.6 | 82.1 | 47.0 |
| XFeat | 67.2 | 83.5 | 87.8 | 81.1 | 48.7 |
| LifeFeat | 68.9 | 84.0 | 88.9 | 86.2 | 50.8 |
| Ours | 67.3 | 84.6 | 89.0 | 86.4 | 51.0 |

## Ablation (RobotCar-Seasons)

| Reli | SRL | SGFM | Day-all / Night-all (2deg/0.25m, 5deg/0.5m, 10deg/5m) |
| --- | --- | --- | --- |
| x | x | x | 54.6 / 80.5 / 94.3 , 16.5 / 41.6 / 71.4 |
| check | x | x | 55.8 / 81.3 / 95.0 , 17.2 / 42.8 / 72.0 |
| x | check | x | 55.2 / 81.0 / 94.8 , 17.0 / 42.0 / 71.8 |
| x | x | check | 55.0 / 80.8 / 94.7 , 16.8 / 41.9 / 71.7 |
| x | check | check | 56.8 / 82.0 / 97.0 , 18.2 / 44.0 / 72.9 |
| check | check | check | 57.5 / 82.1 / 98.0 , 18.5 / 44.3 / 73.0 |

## Runtime and Model Size

| Metric | SuperPoint | XFeat | Ours |
| --- | --- | --- | --- |
| Params (M) | 1.37 | 0.66 | 1.59 |
| FLOPs (G) | 19.65 | 1.33 | 23.79 |
| Desc dim | 256 | 64 | 256 |
| CPU ms | 254 | 35 | 304 |
| GPU ms | 32 | 4.9 | 41 |

## Dependencies

- Python 3 >= 3.6
- PyTorch >= 1.8
- OpenCV >= 3.4
- NumPy >= 1.18
- colmap
- pycolmap = 0.0.1
