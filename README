# TP — Object Detection, Tracking and Benchmarking

## Overview

This project implements a complete computer vision pipeline for:

- Object Detection
- Multi-Object Tracking
- Instance Segmentation
- Performance Benchmarking

using:

- YOLOv8
- DeepSORT
- Mask R-CNN

The application provides an interactive graphical interface for video analysis and comparison between different detection and tracking approaches.

---

# Features

## YOLOv8 Detection

- Real-time object detection
- Bounding box visualization
- Fast inference

---

## YOLOv8 + DeepSORT

- Multi-object tracking
- Unique object IDs
- Robust tracking under occlusion
- Real-time performance

---

## Mask R-CNN + DeepSORT

- Instance segmentation
- Object tracking with masks
- More precise object localization
- Better object separation in complex scenes

---

## Benchmark System

The benchmark module automatically compares:

- Execution time
- FPS
- Detection and tracking performance

between:

- YOLOv8 + DeepSORT
- Mask R-CNN + DeepSORT

Results are exported to CSV format.

---

# Project Structure

```txt
TP_YOLO_MaskRCNN_DeepSORT/
│
├── src/
│   ├── app_interface.py
│   ├── yolo_deepsort.py
│   ├── maskrcnn_deepsort.py
│   └── benchmark.py
│
├── videos/
│   └── test videos
│
├── outputs/
│   ├── benchmark/
│   └── result videos
│
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Create Virtual Environment

```powershell
python -m venv venv
```

Activate environment:

```powershell
venv\Scripts\activate
```

---

## 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

# Required Libraries

```txt
ultralytics
opencv-python
customtkinter
pillow
deep-sort-realtime
torch
torchvision
numpy
pandas
```

---

# Running the Application

Launch the graphical interface:

```powershell
python src/app_interface.py
```

---

# Interface Modes

## 1. YOLOv8 Detection

Simple object detection using YOLOv8.

---

## 2. YOLOv8 + DeepSORT

Detection and tracking with unique IDs.

---

## 3. Mask R-CNN + DeepSORT

Instance segmentation and tracking.

---

# Benchmark Usage

Run automatic benchmark:

```powershell
python src/benchmark.py --video videos/test.mp4 --output_dir outputs/benchmark
```

---

# Benchmark Outputs

The benchmark generates:

```txt
outputs/benchmark/
│
├── yolo_result.mp4
├── maskrcnn_result.mp4
└── benchmark_results.csv
```

Example CSV:

| method | total_time_seconds |
|---|---|
| YOLOv8 + DeepSORT | 18.2 |
| Mask R-CNN + DeepSORT | 64.5 |

---

# Experimental Evaluation

The system was tested on:

- Simple scenes
- Multiple object classes
- Occlusions
- Dense environments

---

# Performance Comparison

method               |  video,          |  status  | total_time_seconds | output_video
YOLOv8 + DeepSORT    |  videos/test.mp4 |  success | 167.52             | outputs/benchmark/yolo_result.mp4
Mask R-CNN + DeepSORT| videos/test.mp4  |  success  | 1343.17            | outputs/benchmark/maskrcnn_result.mp4

---

# Technologies Used

- Python
- OpenCV
- PyTorch
- YOLOv8
- DeepSORT
- Mask R-CNN
- CustomTkinter

---

# Future Improvements

Possible improvements include:

- Object speed estimation
- Object counting
- Trajectory visualization
- Heatmaps
- GPU optimization
- Streamlit dashboard
- Real-time webcam support

---

# Conclusion

This project demonstrates how modern computer vision techniques can be combined to build intelligent object detection and tracking systems.

YOLOv8 provides fast and efficient real-time detection, while Mask R-CNN offers more precise instance segmentation. DeepSORT enhances both approaches with robust multi-object tracking capabilities.