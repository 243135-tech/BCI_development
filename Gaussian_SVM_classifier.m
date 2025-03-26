%% Optimized Data Processing for Motor Imagery BCI
% Luca Boggiani 
clc; clear; close all;
%% Parameters
fs = 512; % Sampling Rate
T = 1/fs;
ntrial = 20;
nchannel = 15;
wlength = 0.5; % Spectrogram window length in seconds
pshift = 0.25; % Shift of the internal PSD windows
wshift = 0.0625; % External window shift in seconds
mlength = 1; % Moving average window length

% Load all .mat files in the directory
data_files = dir('Trainings/*.mat'); 
file_count = length(data_files);

% EEG Channel Labels
channel_labels = {'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', ...
                  'FC3', 'FCz', 'FC4', 'CP3', 'CPz', 'CP4'};

%% Preprocessing: Loop Through All Files
all_trials = {}; % Store extracted trials
all_labels = {}; % Store corresponding labels
all_psd_features = {}; % Store extracted PSD features
all_baseline_features = {}; % Store baseline PSD features
all_frequencies = {}; % Store frequency bins
all_csp_features = {}; % Store CSP features across all runs
all_csp_labels = {};   % Corresponding labels


for f = 1:file_count
    file_name = data_files(f).name;
    file_name = fullfile('Trainings', file_name);
    fprintf('Processing file: %s\n', file_name);
    
    load(file_name);
    data_copy = data; % Keep a copy of original data
    nrun = length(data);

    trials_logband = cell(nrun, ntrial);
    labels = cell(nrun, ntrial);
    psd_features = cell(nrun, ntrial);
    baseline_features = cell(nrun, 1); % One baseline per run

    % Bandpass filter (4-56 Hz)
    [b,a] = butter(5, 2*[4 30]/fs, 'bandpass');
    for r = 1:nrun
        data{1,r}.X = filtfilt(b, a, data{1,r}.X);
    end
    
    for r = 1:nrun
        % === Extract Baseline Data (First 5s of Recording - Before Trials) ===
        baseline_data = data{1,r}.X(1:fs*5, :); 
        [baseline_psd, freq_bins] = proc_spectrogram(baseline_data, wlength, wshift, pshift, fs, mlength);

        for p = 1:ntrial
            start_idx = data{1,r}.trial(1,p);
            trial_data = data{1,r}.X(start_idx:start_idx+fs*5-1,:);
            baseline_features{r,p} = mean(baseline_psd, 1); % Store per trial
            
            % Compute Log-Band Power
            trials_logband{r,p} = log(abs(trial_data).^2);
            
            % Compute Power Spectral Density using proc_spectrogram
            [psd_features{r,p}, ~] = proc_spectrogram(trial_data, wlength, wshift, pshift, fs, mlength);
            
            
            % Store Labels (1 = right hand, 2 = feet)
            labels{r,p} = data{1,r}.y(p);
        end
        % === CSP: Compute Features for Each Run ===
        fprintf('Computing CSP features for run %d\n', r);
        
        % Build 3D EEG matrix: [channels x samples x trials]
        X_run = zeros(nchannel, fs*5, ntrial);
        y_run = zeros(ntrial, 1);
        
        for p = 1:ntrial
            X_run(:, :, p) = trials_logband{r, p}';  % transpose: [channels x samples]
            y_run(p) = labels{r, p};
        end
        
        % Run your CSP function
        W = csp(X_run, y_run); % W: [channels x channels]
        
        % Project trials and extract CSP features (log-variance)
        csp_features = zeros(ntrial, nchannel); % one feature vector per trial
        for p = 1:ntrial
            trial = X_run(:, :, p);
            Z = W' * trial;                  % Apply spatial filter
            var_csp = var(Z, 0, 2);          % Variance over time
            csp_features(p, :) = log(var_csp / sum(var_csp)); % Normalized log-variance
        end
        
        % Store CSP features and labels for this run
        all_csp_features = [all_csp_features; num2cell(csp_features, 2)]; % append row-wise
        all_csp_labels = [all_csp_labels; num2cell(y_run)];
    end

    % Store all processed data
    all_trials = [all_trials; trials_logband];
    all_labels = [all_labels; labels];
    all_psd_features = [all_psd_features; psd_features];
    all_baseline_features = [all_baseline_features; baseline_features];
    all_frequencies = freq_bins;
end

%% CSP Fisher score
fprintf('Computing Fisher Scores on CSP features...\n');

% Convert to matrices
CSP_FeatureMatrix = cell2mat(all_csp_features);
CSP_Labels = cell2mat(all_csp_labels);

% Compute Fisher score
class1_idx = CSP_Labels == 1;
class2_idx = CSP_Labels == 2;

mean1 = mean(CSP_FeatureMatrix(class1_idx, :), 1);
mean2 = mean(CSP_FeatureMatrix(class2_idx, :), 1);
std1 = std(CSP_FeatureMatrix(class1_idx, :), [], 1);
std2 = std(CSP_FeatureMatrix(class2_idx, :), [], 1);

fisher_scores_csp = abs(mean1 - mean2) ./ sqrt(std1.^2 + std2.^2);

% CSP has 1D components instead of freq/channel pairs, so we plot a bar chart
figure;
bar(fisher_scores_csp);
xlabel('CSP Component Index');
ylabel('Fisher Score');
title('Fisher Score of CSP Components (Click to Select)');
xticks(1:nchannel);

[x, ~] = ginput(3); %

% Process selected CSP components
selected_features_manual = cell(length(x), 2);
selected_idx = zeros(length(x), 1);
channel_idx = zeros(length(x), 1);
freq_idx = zeros(length(x), 1);

for i = 1:length(x)
    component_idx = round(x(i));
    selected_idx(i) = component_idx;
    selected_features_manual{i, 1} = sprintf('CSP %d', component_idx);
    selected_features_manual{i, 2} = fisher_scores_csp(component_idx);
end

% Display Selected CSP Features
fprintf('\nSelected CSP Features:\n');
for i = 1:length(selected_idx)
    fprintf('Component: %s, Fisher Score: %.3f\n', ...
        selected_features_manual{i, 1}, selected_features_manual{i, 2});
end


%% Train Classifiers CSP
fprintf('Training Gaussian Classifier...\n');

% Extract corresponding features from dataset
%selected_features = Features(:, selected_idx);
selected_features = CSP_FeatureMatrix(:, selected_idx);

Labels = CSP_Labels;  % Ensure label variable is in use

% Shuffle before splitting to avoid any ordering bias
rng(42); % for reproducibility
shuffled_idx = randperm(length(Labels));
selected_features = selected_features(shuffled_idx, :);
Labels = Labels(shuffled_idx);

% Split into training and testing sets (80/20)
num_trials = length(Labels);
num_train = round(0.8 * num_trials);

train_data = selected_features(1:num_train, :);
train_labels = Labels(1:num_train);

test_data = selected_features(num_train+1:end, :);
test_labels = Labels(num_train+1:end);

% Train Gaussian Naive Bayes Classifier
gaussian_model = fitcnb(train_data, train_labels);

% Train SVM with RBF Kernel
svm_model = fitcsvm(train_data, train_labels, 'KernelFunction', 'rbf', 'BoxConstraint', 1, 'Standardize', true);

% Predict on test set
predicted_labels = predict(gaussian_model, test_data);
accuracy = sum(predicted_labels == test_labels) / length(test_labels) * 100;

% Report performance
fprintf('Gaussian Classifier Accuracy (CSP features): %.2f%%\n', accuracy);

% Save model for evaluation step
save('Gaussian_CSP_Classifier.mat', 'gaussian_model', 'selected_idx');

% Predict on test set using SVM
predicted_labels_SVM = predict(svm_model, test_data);
accuracy_SVM = sum(predicted_labels_SVM == test_labels) / length(test_labels) * 100;

% Report performance
fprintf('SVM Classifier Accuracy (CSP features): %.2f%%\n', accuracy_SVM);

% Save model for evaluation step
save('SVM_CSP_Classifier.mat', 'svm_model', 'selected_idx');


%% Load and Preprocess Evaluation Data
eval_files = dir(fullfile('Evaluations/', '*.mat')); % Get all evaluation files
num_eval_files = length(eval_files);

if num_eval_files == 0
    error('No evaluation files found in Evaluations/*.mat');
end

fprintf('Found %d evaluation files. Loading and processing...\n', num_eval_files);

all_eval_features = {}; % Store extracted features
all_eval_labels = {}; % Store true labels
all_csp_features_eval = {}; % Store CSP features across all runs
all_csp_labels_eval = {};   % Corresponding labels

for f = 1:num_eval_files
    eval_file = eval_files(f).name;
    eval_path = fullfile('Evaluations', eval_file);
    fprintf('Processing evaluation file: %s\n', eval_path);
    
    load(eval_path); % Load the data
    nrun = length(data);

    eval_psd_features = cell(nrun, ntrial);
    eval_labels = cell(nrun, ntrial);

    % Apply same bandpass filter as training data
    [b,a] = butter(5, 2*[4 30]/fs, 'bandpass');
    for r = 1:nrun
        data{1,r}.X = filtfilt(b, a, data{1,r}.X);
    end
    
    for r = 1:nrun
        for p = 1:ntrial
            start_idx = data{1,r}.trial(1,p);
            trial_data = data{1,r}.X(start_idx:start_idx+fs*5-1,:);

            % Compute Power Spectral Density using proc_spectrogram
            [eval_psd_features{r,p}, ~] = proc_spectrogram(trial_data, wlength, wshift, pshift, fs, mlength);
            
            % Store true labels (1 = right hand, 2 = both feet)
            eval_labels{r,p} = data{1,r}.y(p);
        end
        % === CSP: Compute Features for Each Run ===
        fprintf('Computing CSP features for run %d\n', r);
        
        % Build 3D EEG matrix: [channels x samples x trials]
        X_run = zeros(nchannel, fs*5, ntrial);
        y_run = zeros(ntrial, 1);
        
        for p = 1:ntrial
            X_run(:, :, p) = trials_logband{r, p}';  % transpose: [channels x samples]
            y_run(p) = labels{r, p};
        end
        
        % Run your CSP function
        W = csp(X_run, y_run); % W: [channels x channels]
        
        % Project trials and extract CSP features (log-variance)
        csp_features = zeros(ntrial, nchannel); % one feature vector per trial
        for p = 1:ntrial
            trial = X_run(:, :, p);
            Z = W' * trial;                  % Apply spatial filter
            var_csp = var(Z, 0, 2);          % Variance over time
            csp_features(p, :) = log(var_csp / sum(var_csp)); % Normalized log-variance
        end
        
        % Store CSP features and labels for this run
        all_csp_features_eval = [all_csp_features; num2cell(csp_features, 2)]; % append row-wise
        all_csp_labels_eval = [all_csp_labels; num2cell(y_run)];
    end

    % Store extracted features and labels
    all_eval_features = [all_eval_features; eval_psd_features];
    all_eval_labels = [all_eval_labels; eval_labels];
end

%% Extract feature CSP from eval
fprintf('Extracting CSP features from evaluation data...\n');

num_eval_trials = length(all_csp_features_eval); 
eval_selected_features = zeros(num_eval_trials, length(selected_idx));
eval_labels_vector = zeros(num_eval_trials, 1);

for t = 1:num_eval_trials
    % trial_data should be [channels x samples]
    trial_psd = all_csp_features_eval{t}; % should be [samples x channels] or [channels x samples]
    if size(trial_psd, 1) < size(trial_psd, 2)
        trial_psd = trial_psd'; % ensure [channels x samples]
    end
    
    % Select the same components used in training
    eval_selected_features(t, :) = trial_psd(selected_idx);

    % Store true label
    eval_labels_vector(t) = all_csp_labels_eval{t};
end

fprintf('Final Feature Matrix Size: [%d x %d]\n', size(eval_selected_features, 1), size(eval_selected_features, 2));

%% Evaluate Classifier Performance for Each Evaluation File**
fprintf('Evaluating Gaussian Classifier on Each Evaluation File...\n');

% Predict labels on entire evaluation dataset
predicted_eval_labels = predict(gaussian_model, eval_selected_features);
predicted_eval_labels_svm = predict(svm_model, eval_selected_features);

% Initialize accuracy storage
file_accuracies = zeros(num_eval_files, 1);
file_accuracies_svm = zeros(num_eval_files, 1);
start_idx = 1;
trials_per_file = ntrial * 3; % 3 runs of 20 trials per file

for f = 1:num_eval_files
    end_idx = start_idx + trials_per_file - 1;

    % Extract file-specific predicted labels and ground truth
    predicted_labels_file = predicted_eval_labels(start_idx:end_idx);
    predicted_labels_file_svm = predicted_eval_labels_svm(start_idx:end_idx);
    true_labels_file = eval_labels_vector(start_idx:end_idx);

    % Compute accuracy
    file_accuracies(f) = sum(predicted_labels_file == true_labels_file) / trials_per_file * 100;
    file_accuracies_svm(f) = sum(predicted_labels_file_svm == true_labels_file) / trials_per_file * 100;

    fprintf('Accuracy Gaussian for %s: %.2f%%\n', eval_files(f).name, file_accuracies(f));
    fprintf('Accuracy SVM for %s: %.2f%%\n', eval_files(f).name, file_accuracies_svm(f));

    % Update start index for next file
    start_idx = end_idx + 1;
end

% Display overall summary
fprintf('Overall Gaussian Mean Accuracy Across Files: %.2f%%\n', mean(file_accuracies));
fprintf('Overall SVM Mean Accuracy Across Files: %.2f%%\n', mean(file_accuracies_svm));
