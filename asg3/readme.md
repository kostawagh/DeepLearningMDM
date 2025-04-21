# YOLOv11 Object Detection - Car & Bike via CCTV

## Course
**Deep Learning Lab** | **Student:** Kaustubh Wagh (202201070021)

## Objective
Implement YOLOv11 for real-time object detection on a custom car-bike dataset from CCTV footage.

## Tasks Overview

### 1. Setup
- Install Python, PyTorch, OpenCV, Ultralytics.
- Clone YOLOv11 repo and verify setup.

### 2. Dataset Prep
- Use Roboflow-labeled custom dataset.
- Export in YOLO format.
- Train/val/test split: 80/10/10.  
[Dataset Link](https://universe.roboflow.com/leo-ueno/people-detection-o4rdr/dataset/8/download)

### 3. Model Training
- Configure epochs, batch size, learning rate.
- Train model and save weights.

### 4. Inference & Evaluation
- Run detection on test images/videos.
- Evaluate: **mAP, Precision, Recall, F1-Score**.
- Visualize bounding boxes.

## Results
- Strong detection performance on vehicles.
- Metrics show balanced accuracy and low false positives.
- Dataset quality and tuning were key to performance.

**GitHub:** [DeepLearningMDM](https://github.com/kostawagh/DeepLearningMDM)
