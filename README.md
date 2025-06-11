# BCI_development

## Overview

This repository contains MATLAB code and documentation for the development of a Motor Imagery (MI)-based Brain-Computer Interface (BCI) system using EEG signals. The project is divided into two main components:

- `Online_Dataset`: Processing, feature extraction, and classification using public EEG data from the BNCI Horizon 2020 dataset.
- `Real_Data`: Preprocessing, classification, and command generation for controlling a robotic exoskeleton and game interface using real EEG data.

---

## 1. Online_Dataset

This folder includes all the scripts for training and testing classifiers on the public BNCI-Horizon 2020 motor imagery dataset.

### Dataset
- Source: [BNCI Horizon 2020](https://bnci-horizon-2020.eu/database/data-sets)
- Tasks: Cue-based motor imagery of right hand vs. feet
- Sampling Rate: 512 Hz
- Channels: 15 (motor cortex – Graz-BCI montage)
- Trials: 20 per run, across 8 runs (5 for training, 3 for evaluation)

### Preprocessing
- Bandpass filter (4–56 Hz)
- Log-band power extraction for mu (8–12 Hz) and beta (14–30 Hz) bands
- Feature smoothing with moving average

### Feature Extraction
- Power Spectral Density (PSD)
- Common Spatial Patterns (CSP)
- Event-Related Desynchronization (ERD) and Synchronization (ERS)
Feature Selection:
- Fisher Score to rank features by discriminability between MI tasks

### Classifiers
- **Gaussian Classifier**: Trained on CSP features selected via Fisher Score; achieves ~79% accuracy
- **SVM**: Achieves higher accuracy (~85%) when paired with Fisher-ranked CSP features

## 2. Real_Data
This folder implements the full closed-loop BCI pipeline for real-time MI classification and control of a game-driven robotic exoskeleton for stroke rehabilitation.

Objectives:

Real-time motor imagery classification from OpenBCI cap data
Trigger robotic movement (elbow flexion) via classified intent
Provide engaging visual feedback via a Unity-based game
MI Tasks:

Right arm vs. right hand imagery (custom recorded dataset)

### Game Overview
The Unity game simulates a flower-picking task using elbow flexion, triggered by motor imagery (MI). The current implementation supports only:

Grabbing flowers thanks to right hand MI task and lifting them via right arm MI task.

### How to Run
Install Unity: Use Unity Hub to install Unity Editor (version 2021.3 LTS or later).
Open Project: Load the project folder in Unity.
Run: Open MainScene.unity and press Play in the Editor.
UDP Communication: The game listens for "grab" and "lift" commands via UDP. Ensure the BCI system sends messages to the IP/port specified in UdpHost.cs.
### Requirements

MATLAB (R2020b or newer recommended)
Signal Processing Toolbox
Statistics and Machine Learning Toolbox
Unity (for the game interface)
Python (on Raspberry Pi for exoskeleton control)
OpenBCI GUI and EEG headset (for new data)
### Setup Instructions

Clone the Repository
git clone https://github.com/243135-tech/BCI_development.git
Run Classifier on Online Dataset
cd Online_Dataset
run('Data_analysis.m')
Run Real-Time BCI Pipeline
Start OpenBCI GUI and begin EEG stream
Launch Unity game
Run classification script in MATLAB
UDP triggers will control both the LEGO arm and Unity interface

### References

BNCI Horizon 2020 Dataset: https://bnci-horizon-2020.eu/database/data-sets
"Random Forests vs. Regularized LDA – Non-linear Beats Linear"
"BCI Classification using Locally Generated CSP Features"
"BCI-Based Robotic System for Upper Limb Rehabilitation After Stroke"
"Transferring BCIs Beyond the Laboratory"
License

This project is licensed under the MIT License.

