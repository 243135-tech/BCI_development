# BCI_development
Overview
This repository contains code and documentation for a motor imagery-based brain-computer interface (MI-BCI) using EEG signals. The project implements two different classification pipelines for motor imagery tasks:

Gaussian Classifier using Fisher Score for feature selection.
Random Forest Classifier using Common Spatial Patterns (CSP) for feature selection.
The dataset used is sourced from the BNCI Horizon 2020 repository, containing EEG recordings from motor imagery tasks.

Dataset

Source: https://bnci-horizon-2020.eu/database/data-sets
EEG Setup:
Sampling Rate: 512 Hz
Channels: 15 (Motor Cortex, Graz-BCI Setup)
Tasks: Right Hand vs Feet Motor Imagery
Experimental Protocol:
Cue-based MI task (Graz-BCI Paradigm)
5s trials with log-band power and CSP filtering

1. EEG Data Preprocessing
Band-pass filtering (4-56 Hz, Butterworth filter)
Log-band power extraction
Feature extraction in mu (8-12 Hz) & beta (14-30 Hz) bands
2. Feature Selection & Classification

A) Gaussian Classifier with Fisher Score
Feature Selection: Fisher Score
Classification: Gaussian Model
Workflow:
Compute Fisher Score for each feature
Select top features with highest discriminability
Train Gaussian classifier on selected features
Why Fisher Score?
Fisher Score evaluates feature separability between two classes (right hand vs feet). It ranks features based on their contribution to classification​.

B) Random Forest Classifier with CSP
Feature Selection: Common Spatial Patterns (CSP)
Classification: Random Forest
Workflow:
Apply CSP to extract discriminative EEG features
Train Random Forest classifier on CSP features
Evaluate model performance using accuracy & Youden Index
Why CSP?
CSP maximizes variance differences between two classes, making it ideal for motor imagery BCI​.

Why Random Forest?
RF classifiers provide non-linear decision boundaries, handle high-dimensional data well, and outperform linear classifiers for EEG tasks​.

3. Performance Metrics
   
Classification Accuracy
ERD/ERS Analysis (Event-Related Desynchronization/Synchronization)​
Installation & Dependencies

To run the MATLAB scripts:

Install MATLAB (recommended: 2020b or newer).
Clone this repository:
git clone https://github.com/243135-tech/BCI_development.git
Add the necessary toolboxes:
Signal Processing Toolbox
Statistics and Machine Learning Toolbox
Results & Insights

Gaussian Classifier with Fisher Score provides robust feature selection, ensuring high separability between MI classes.
Random Forest with CSP outperforms linear classifiers due to better feature utilization​.
Combining Fisher Score and CSP improves classification accuracy and ensures better generalization.
Future Work

Optimize real-time BCI feedback for practical applications.
Test on additional EEG datasets to validate generalization.
References

BNCI Horizon 2020 Database: https://bnci-horizon-2020.eu/database/data-sets
Motor Imagery BCI - Random Forest vs. Linear Models​
Locally Generated CSP Features for Classification​
EEG Data Analysis & Preprocessing Script​
License

This project is licensed under the MIT License.
