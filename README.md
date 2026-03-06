---
title: Polyp Detection Pro
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
short_description: Professional AI Polyp Detection & Segmentation
---

# 🩺 AI-Powered Polyp Detection

This application uses **YOLOv8** for object detection and **U-Net (ResNet34)** for semantic segmentation to identify polyps in colonoscopy images and videos.

## Features
- **Dual-Stage Pipeline**: Detects location and segments boundaries.
- **Video Support**: Process full video files with frame-by-frame analysis.
- **Interactive UI**: Adjust confidence thresholds and visualization settings.

## How to Use
1. **Upload** an image or video.
2. **Adjust** the settings if needed (Confidence, IoU).
3. **Analyze** to see the results.

## Disclaimer
This tool is for **research and educational purposes only**. It is not intended for clinical diagnosis.
