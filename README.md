# BCI_development
Overview
This repository contains code and documentation for a motor imagery-based brain-computer interface (MI-BCI) using EEG signals. The project implements two different classification pipelines for motor imagery tasks:

Gaussian Classifier using Fisher Score for CSP feature selection.
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
Band-pass filtering (4-30 Hz, Butterworth filter), 
Power Spectral Density (proc_spectogram.m), 
Feature extraction in mu (8-12 Hz) & beta (14-30 Hz) bands
2. Feature Selection & Classification

A) Gaussian Classifier with Fisher Score
Feature Extraction: Apply CSP to extract log-variance features.
Feature Selection: Rank CSP features by Fisher Score; select top ones.
Classification: Gaussian Model
Workflow:
Compute Fisher Score for each CSP feature,
Select top features with highest discriminability,
Train Gaussian classifier on selected features.
Why Fisher Score?
Fisher Score evaluates feature separability between two classes (right hand vs feet). It ranks features based on their contribution to classification​.
Accuracy overall on all the evaluation data: 79%

B) SVM Classifier with Fisher Score
Feature Extraction: Apply CSP to extract log-variance features.
Feature Selection: Rank CSP features by Fisher Score; select top ones.
Classification: SVM Model
Workflow:
Compute Fisher Score for each CSP feature,
Select top features with highest discriminability,
Train SVM classifier on selected features.
Accuracy overall on all the evaluation data: 85%

C) Random Forest Classifier with CSP
Feature Selection: Common Spatial Patterns (CSP).
Classification: Random Forest
Workflow:
Apply CSP to extract discriminative EEG features,
Train Random Forest classifier on CSP features,
Evaluate model performance using accuracy. 

3. Performance Metrics
   
Classification Accuracy

A)Accuracy overall on all the evaluation data: 79%

B)Accuracy overall on all the evaluation data: 85%

C)Accuracy overall on all the evaluation data: 52%

4. Exponential Accumulation Framework for Decision Making
To enhance the robustness and reliability of EEG-based motor imagery classification, we implement an Exponential Accumulation Framework for decision-making. This framework helps to smooth fluctuations in classification outputs, preventing erratic false positives and ensuring stable decision boundaries.

Key Concept: Smoothing Classification Probabilities

Instead of making decisions based on single-instance classifications, which can be noisy, we apply an exponential moving average to accumulate evidence over time. Given a probability output 
p
t
p 
t
​	
  at time 
t
t, the accumulated decision probability
P
t
P 
t
​	
  is computed as:
	
Pt =αP 
t−1
​	
 +(1−α)p 
t
​	
 
where:

P 
t
​	
  is the accumulated probability at time 
t,
p 
t
​	
  is the classifier’s probability output at time 
t,
α is the smoothing factor.
P 
0
​	
  is initialized to zero or a neutral state.
Decision Thresholding

A command is triggered when 
P 
t
​	
  exceeds a predefined threshold 
T
d
​	
Decision=arg 
i
max
​	
 (P 
t,i
​	
 )ifmax(P 
t
​	
 )>T 
d
​	
 
where:


i represents each possible class (e.g., right hand vs feet imagery),
T 
d
​	
  is an adaptive threshold tuned to minimize false positives.
This approach significantly improves BCI performance, especially in asynchronous (self-paced) BCIs, where decisions need to be stable rather than reacting to every small change in the EEG signal

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
