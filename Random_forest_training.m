%% Common Spatial Pattern Implementation for MI BCI
% Luca Boggiani 
clc; clear; close all;

%% Parameters
fs = 512; % Sampling Rate
nchannel = 15;
ntrial = 20;
nfilters = 5; % Number of CSP filters per class
train_folder = 'Trainings'; % Folder with all training files
data_files = dir(fullfile(train_folder, '*.mat'));
file_count = length(data_files);

% EEG Channel Labels
channel_labels = {'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', ...
                  'FC3', 'FCz', 'FC4', 'CP3', 'CPz', 'CP4'};

%% Preprocessing: Load and Filter All Training Data
all_trials = [];
all_labels = [];

for f = 1:file_count
    file_name = fullfile(train_folder, data_files(f).name);
    fprintf('Processing file: %s\n', file_name);
    
    load(file_name);
    nrun = length(data);
    
    % Bandpass filter (4-56 Hz)
    [b, a] = butter(5, 2*[4 56]/fs, 'bandpass');
    for r = 1:nrun
        data{1,r}.X = filtfilt(b, a, data{1,r}.X);
    end
    
    % Extract trials and labels
    for r = 1:nrun
        for p = 1:ntrial
            start_idx = data{1,r}.trial(1,p);
            trial_data = data{1,r}.X(start_idx:start_idx+fs*5-1, :);
            all_trials = cat(3, all_trials, trial_data);
            all_labels = [all_labels; data{1,r}.y(p)];
        end
    end
end

fprintf('Total Trials: %d\n', size(all_trials,3));

%% Compute Common Spatial Pattern (CSP) Filters
disp('Computing CSP filters...');
class1_trials = all_trials(:,:,all_labels==1);
class2_trials = all_trials(:,:,all_labels==2);

C1 = cov(reshape(class1_trials,[],nchannel));
C2 = cov(reshape(class2_trials,[],nchannel));
[W, D] = eig(C1, C1+C2);

% Select top and bottom nfilters CSP components
csp_filters = [W(:,1:nfilters) W(:,end-nfilters+1:end)];

%% Extract CSP Features for Classification
disp('Extracting CSP features...');
num_trials = size(all_trials,3);
csp_features = zeros(num_trials, nfilters * 2);

for t = 1:num_trials
    projected = all_trials(:,:,t) * csp_filters;
    csp_features(t, :) = log(var(projected));
end

%% Train a Random Forest Classifier
n_trees = 500;
disp('Training Random Forest Classifier...');
rf_model = TreeBagger(n_trees, csp_features, all_labels, 'OOBPrediction', 'on', 'Method', 'classification');
save('Trained_CSP_RF_Classifier.mat', 'rf_model');

%% Evaluate Training Accuracy
disp('Evaluating Training Accuracy...');
train_predicted_labels = str2double(predict(rf_model, csp_features));
train_accuracy = sum(train_predicted_labels == all_labels) / num_trials * 100;
fprintf('Training Accuracy: %.2f%%\n', train_accuracy);

%% Save CSP Filters for Future Use
save('CSP_Filters.mat', 'csp_filters');
