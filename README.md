# CV-2: Crop Disease Detection

A computer vision project for identifying crop diseases from plant leaf images using transfer learning.

## Project Information

- Course: Computer Vision
- Project Code: CV-2
- Topic: Crop Disease Detection
- Task: Multi-class image classification
- Number of Classes: 38
- Model: EfficientNet-B0
- Framework: PyTorch

## Overview

This project builds a crop disease detection system that classifies plant leaf images into healthy or diseased crop categories. The system uses transfer learning with EfficientNet-B0 pretrained on ImageNet, then trains a custom classifier on a crop disease dataset.

The final system includes:

- Training pipeline
- Testing and evaluation pipeline
- Confusion matrix and classification report
- Command-line image inference
- Gradio web demo
- Saved trained model

## Dataset

The project uses the New Plant Diseases Dataset from Kaggle.

Dataset structure after processing:

```text
data/processed/
├── train/
├── val/
└── test/
