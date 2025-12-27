# Real-Time Multi-Object Tracking and Action Recognition

A PyTorch implementation of a custom deep learning architecture for simultaneous multi-object tracking (MOT) and action recognition in video streams.

## 🎯 Project Overview

This project implements a novel end-to-end framework that:
- Detects multiple objects in video frames
- Tracks objects across frames with identity preservation
- Recognizes actions performed by each tracked object
- Operates in real-time with optimized inference

## 🏗️ Architecture

The model consists of:
- **Custom Feature Backbone**: Efficient spatial feature extraction
- **Temporal Encoder**: Captures motion and temporal dependencies
- **Multi-Task Heads**: 
  - Detection Head (bounding boxes + classification)
  - Tracking Head (re-identification features)
  - Action Head (action classification)
- **Association Module**: Links detections across frames

## 🚀 Features

- Built from scratch - no pretrained models
- Multi-task learning with custom loss functions
- Efficient temporal feature aggregation
- Real-time inference capability
- Comprehensive evaluation metrics (MOTA, IDF1, mAP)

## 📋 Requirements
```
Python >= 3.8
PyTorch >= 2.0
CUDA >= 11.8 (for GPU training)
```

See `requirements.txt` for full dependencies.

## 🔧 Installation
```bash
git clone https://github.com/yourusername/realtime-mot-action.git
cd realtime-mot-action
pip install -r requirements.txt
pip install -e .
```

## 📊 Dataset

Currently supports:
- MOT17/MOT20 (Multi-Object Tracking)
- AVA (Action Detection)
- Custom video datasets

## 🏃 Quick Start
```bash
# Download and prepare data
python scripts/download_data.py

# Train the model
python scripts/train.py --config config/experiment_configs/baseline.yaml

# Evaluate
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth

# Run inference
python scripts/inference.py --video path/to/video.mp4
```

## 📈 Training

Detailed training instructions coming soon...

## 🎓 Model Architecture Details

Documentation coming soon...

## 📝 Results

Performance benchmarks will be updated as training progresses.

## 🤝 Contributing

This is a research/learning project. Feel free to open issues or submit PRs!

## 📄 License

MIT License