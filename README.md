# BCI_development

## Overview

This repository contains MATLAB code and documentation for the development of a Motor Imagery (MI)-based Brain-Computer Interface (BCI) system using EEG signals. The project is divided into two main components:

- `Online_Dataset`: Processing, feature extraction, and classification using public EEG data from the BNCI Horizon 2020 dataset.
- `Real_Data`: Preprocessing, classification, and command generation for controlling a robotic exoskeleton and game interface using real EEG data.

---

## 1. Online_Dataset

This folder includes all the scripts for training and testing classifiers on the public BNCI-Horizon 2020 motor imagery dataset.

### Dataset
- **Source**: [BNCI Horizon 2020](https://bnci-horizon-2020.eu/database/data-sets)
- **Tasks**: Cue-based motor imagery of right hand vs. feet
- **Sampling Rate**: 512 Hz
- **Channels**: 15 (motor cortex – Graz-BCI montage)
- **Trials**: 20 per run, across 8 runs (5 for training, 3 for evaluation)

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

## 2.Real_Data
This folder implements the full closed-loop BCI pipeline for real-time MI classification and control of a game-driven robotic exoskeleton for stroke rehabilitation.

Objectives:

Real-time motor imagery classification from OpenBCI cap data
Trigger robotic movement (elbow flexion) via classified intent
Provide engaging visual feedback via a Unity-based game
MI Tasks:

Right arm vs. right hand imagery (custom recorded dataset)

Two game modes:
Mode 1: Full exoskeleton actuation upon MI detection
Mode 2: Patient-initiated movement + MI-triggered assistance
Communication handled via UDP sockets
### Game:

Developed in Unity
Simulates picking flowers with elbow flexion
Adjustable difficulty: range of motion, sequence length
Tracks progress via .csv logs (reaction times, voluntary movement)
Data Logging:

Session ID, reaction time, autonomous movement (degrees), timestamps
Integration:

UDP-based command architecture between:
MATLAB BCI backend
Raspberry Pi-controlled exoskeleton
Unity game engine
### Requirements

MATLAB (R2020b or newer recommended)
Signal Processing Toolbox
Statistics and Machine Learning Toolbox
Unity (for the game interface)
Python (on Raspberry Pi for exoskeleton control)
OpenBCI GUI and EEG headset
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
### Results & Insights

Fisher Score provided effective feature ranking
CSP + Random Forest outperformed linear models in several configurations
Game-based feedback improved participant engagement
Exponential accumulation improved classification robustness in real-time settings
### References

BNCI Horizon 2020 Dataset: https://bnci-horizon-2020.eu/database/data-sets
"Random Forests vs. Regularized LDA – Non-linear Beats Linear"
"BCI Classification using Locally Generated CSP Features"
"BCI-Based Robotic System for Upper Limb Rehabilitation After Stroke"
"Transferring BCIs Beyond the Laboratory"
License

This project is licensed under the MIT License.

