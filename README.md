This repository contains the code used in the paper **"[CAD: Two Cattle Datasets in Dorsal View for Cattle Recognition]"**, focusing on object detection and feature extraction for cattle identification in precision agriculture. The code is organized into separate modules for object detection and feature extraction, each housed in designated folders for clear navigation.

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Object Detection](#object-detection)
  - [Feature Extraction](#feature-extraction)
- [Acknowledgments](#acknowledgments)


## Overview
This repository provides:
- Object detection code based on **YOLOv7** for identifying cattle in various scenes.
- Feature extraction code utilizing **OSNet** for distinguishing individual cattle based on visual characteristics.

These tools were used to create and validate the CAD datasets. For details on dataset usage, results, and methodology, please refer to the paper.

## Repository Structure

- `object_detection/yolov7/`: Contains YOLOv7-based code for cattle detection.
  - `detect.py`: Main processing script for object detection, adapted for cattle identification.
  
- `feature_extraction/osnet/`: Houses OSNet-based code for cattle feature extraction.
  - `main.py`: Primary script for feature extraction, focused on distinguishing cattle based on unique characteristics.

## Setup and 

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo-link
   cd your-repo-name
   ```
2. **Install required dependencies** (YOLOv7 and OSNet dependencies):

- Ensure Python 3.7+ is installed.
- Install dependencies from `requirements.txt`:
   ```bash
	pip install -r requirements.txt
   ```
3. **Download Pretrained Weights** (if needed):

- For YOLOv7, download weights from [YOLOv7 repository](https://github.com/WongKinYiu/yolov7).
- For OSNet, pretrained models are available in the [OSNet repository](https://github.com/KaiyangZhou/deep-person-reid).

## Usage

### Object Detection

Navigate to the `object_detection/yolov7` directory and run `detect.py` to perform cattle detection.

### Feature Extraction

Navigate to the `feature_extraction/osnet` directory and run `main.py` to perform feature extraction

## Acknowledgments

This repository integrates components from the following open-source projects:
- [YOLOv7](https://github.com/WongKinYiu/yolov7) by WongKinYiu, for object detection.
- [LightMBN](https://github.com/jixunbo/LightMBN) by Jixunbo, for efficient feature extraction, which served as an alternative to OSNet in certain tasks.
- [OSNet](https://github.com/KaiyangZhou/deep-person-reid) by Kaiyang Zhou, originally developed for person re-identification and adapted here for cattle feature extraction.

We are grateful for the contributions of these projects, which have significantly supported advancements in our work on cattle identification and precision livestock applications.

