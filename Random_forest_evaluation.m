%% Load Trained CSP Classifier and Filters
clc; clear; close all;
load('Trained_CSP_RF_Classifier.mat'); % Load trained Random Forest model
load('CSP_Filters.mat'); % Load trained CSP filters

%% Parameters
fs = 512; % Sampling Rate
nchannel = 15;
ntrial = 20;
eval_folder = 'Evaluations'; % Folder containing evaluation .mat files
eval_files = dir(fullfile(eval_folder, '*.mat'));
num_eval_files = length(eval_files);

if num_eval_files == 0
    error('No evaluation files found in Evaluations/*.mat');
end

fprintf('Found %d evaluation files. Processing...\n', num_eval_files);

%% Load and Preprocess Evaluation Data
all_eval_trials = [];
all_eval_labels = [];

for f = 1:num_eval_files
    eval_file = fullfile(eval_folder, eval_files(f).name);
    fprintf('Processing evaluation file: %s\n', eval_file);
    
    load(eval_file);
    nrun = length(data);

    % Bandpass filter (4-56 Hz)
    [b,a] = butter(5, 2*[4 56]/fs, 'bandpass');
    for r = 1:nrun
        data{1,r}.X = filtfilt(b, a, data{1,r}.X);
    end
    
    % Extract trials and labels
    for r = 1:nrun
        for p = 1:ntrial
            start_idx = data{1,r}.trial(1,p);
            trial_data = data{1,r}.X(start_idx:start_idx+fs*5-1, :);
            all_eval_trials = cat(3, all_eval_trials, trial_data);
            all_eval_labels = [all_eval_labels; data{1,r}.y(p)];
        end
    end
end

fprintf('Total Evaluation Trials: %d\n', size(all_eval_trials,3));

%% Extract CSP Features from Evaluation Data
disp('Extracting CSP features for evaluation data...');
num_eval_trials = size(all_eval_trials,3);
nfilters = size(csp_filters, 2);
eval_csp_features = zeros(num_eval_trials, nfilters);

for t = 1:num_eval_trials
    projected = all_eval_trials(:,:,t) * csp_filters;
    eval_csp_features(t, :) = log(var(projected));
end

%% Predict Labels Using Trained Classifier
disp('Classifying evaluation data...');
predicted_eval_labels = str2double(predict(rf_model, eval_csp_features));

%% Compute Accuracy
eval_accuracy = sum(predicted_eval_labels == all_eval_labels) / num_eval_trials * 100;
fprintf('Evaluation Accuracy: %.2f%%\n', eval_accuracy);

%% Compute Confusion Matrix
conf_mat = confusionmat(all_eval_labels, predicted_eval_labels);

% Plot Confusion Matrix
figure;
confusionchart(conf_mat);
title('Confusion Matrix for Evaluation Data');
xlabel('Predicted Label');
ylabel('True Label');

