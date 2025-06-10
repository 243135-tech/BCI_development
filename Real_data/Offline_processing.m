% Luca Boggiani 
clc; clear; close all;
%% Parameters
fs = 128; % Sampling Rate
T = 1/fs;
ntrial = 10;
nchannel = 16;
wlength = 0.5; % Spectrogram window length in seconds
pshift = 0.25; % Shift of the internal PSD windows
wshift = 0.0625; % External window shift in seconds
mlength = 1; % Moving average window length

% Load all .mat files in the directory
data_files = dir('Trainings/*.mat'); 
file_count = length(data_files);

% EEG Channel Labels
channel_labels = {'FP1', 'FP2', 'F3', 'F4', 'FZ', 'C3', 'C4', 'CZ', ...
                  'P3', 'P4', 'PZ', 'O1', 'O2', 'T5', 'T6', 'T3'};

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

    % Bandpass filter (4-30 Hz)
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
            
            
            % Store Labels (1 = right hand, 2 = right arm)
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

%% Plot before CSP

ch1 = 2; % Fz
ch2 = 5; % Cz

X_raw = [];
y_raw = [];

for i = 1:numel(all_trials)
    trial_data = all_trials{i}; % [samples x channels]
    avg_power = mean(trial_data, 1); % mean over time
    X_raw(end+1, :) = [avg_power(ch1), avg_power(ch2)];
    y_raw(end+1) = all_labels{i};
end

figure;
gscatter(X_raw(:,1), X_raw(:,2), y_raw, 'rb', 'ox');
xlabel(sprintf('Mean Log-Power - %s', channel_labels{ch1}));
ylabel(sprintf('Mean Log-Power - %s', channel_labels{ch2}));
title('Scatter Plot Before CSP');
legend('Right Hand', 'Right Arm');

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
title('Fisher Score of CSP Components');
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

%% Plot after CSP
fprintf('Plotting scatter AFTER CSP...\n');

% Choose 2 best CSP components based on Fisher score
[~, idx_sorted] = sort(fisher_scores_csp, 'descend');
comp1 = idx_sorted(1);
comp2 = idx_sorted(2);

X_csp = CSP_FeatureMatrix(:, [comp1, comp2]);

figure;
gscatter(X_csp(:,1), X_csp(:,2), CSP_Labels, 'rb', 'ox');
xlabel(sprintf('CSP Component %d', comp1));
ylabel(sprintf('CSP Component %d', comp2));
title('Scatter Plot After CSP');
legend('Right Hand', 'Right Arm');
% Train a linear classifier (LDA) on the two selected CSP components
lda_model = fitcdiscr(X_csp, CSP_Labels);

% Create a grid of points for contour plotting
x_range = linspace(min(X_csp(:,1))-0.5, max(X_csp(:,1))+0.5, 100);
y_range = linspace(min(X_csp(:,2))-0.5, max(X_csp(:,2))+0.5, 100);
[x1Grid, x2Grid] = meshgrid(x_range, y_range);
gridX = [x1Grid(:), x2Grid(:)];

% Predict class scores for the grid
[~, score] = predict(lda_model, gridX);

% Plot the decision boundary (where score for class 1 = class 2 → score(:,2) - score(:,1) = 0)
hold on;
contour(x1Grid, x2Grid, reshape(score(:,2) - score(:,1), size(x1Grid)), [0 0], 'k--', 'LineWidth', 2);
legend('Class 1', 'Class 2', 'Decision Boundary');

%% === Prepare Selected CSP Features ===
CSP_FeatureMatrix = cell2mat(all_csp_features);
CSP_Labels = cell2mat(all_csp_labels);

% Select only chosen CSP components
features_selected = CSP_FeatureMatrix(:, selected_idx);

%  Train/Test Split 
rng(42); % for reproducibility
cv = cvpartition(CSP_Labels, 'HoldOut', 0.3); % 70% train, 30% test

X_train = features_selected(training(cv), :);
y_train = CSP_Labels(training(cv));
X_test = features_selected(test(cv), :);
y_test = CSP_Labels(test(cv));

% we have 14 training samples, each with 3 features.
% we have 6 test samples, each with 3 features.

%% === Remove Zero-Variance Predictors 
var_train = var(X_train, 0, 1);
nonzero_var_idx = var_train > 0;

X_train = X_train(:, nonzero_var_idx);
X_test = X_test(:, nonzero_var_idx);

%% === Classifier: Gaussian Naive Bayes ===
try
    gnb_model = fitcnb(X_train, y_train);
    y_pred_gnb = predict(gnb_model, X_test);
    acc_gnb = mean(y_pred_gnb == y_test) * 100;
catch ME
    acc_gnb = NaN;
    fprintf('[GNB] ERROR: %s\n', ME.message);
end

%% === Classifier: SVM ===
svm_model = fitcsvm(X_train, y_train, 'KernelFunction', 'linear');
y_pred_svm = predict(svm_model, X_test);
acc_svm = mean(y_pred_svm == y_test) * 100;

%% === Classifier: LDA ===
lda_model = fitcdiscr(X_train, y_train);
y_pred_lda = predict(lda_model, X_test);
acc_lda = mean(y_pred_lda == y_test) * 100;

%% === Classifier: Random Forest ===
rf_model = TreeBagger(100, X_train, y_train, 'OOBPrediction', 'on', 'Method', 'classification');
y_pred_rf = str2double(predict(rf_model, X_test));
acc_rf = mean(y_pred_rf == y_test) * 100;

%% === Display Results ===
fprintf('\n=== Classification Accuracy ===\n');
fprintf('Gaussian NB:     %.2f%%\n', acc_gnb);
fprintf('SVM (linear):    %.2f%%\n', acc_svm);
fprintf('LDA:             %.2f%%\n', acc_lda);
fprintf('Random Forest:   %.2f%%\n', acc_rf);

%% === Save Trained Classifier and Feature Info ===
model_filename = 'trained_svm_model.mat';

% Save the trained model, selected CSP component indices, and mean/std info
save(model_filename, 'svm_model', 'selected_idx', 'nonzero_var_idx');

fprintf('Saved trained model and selected features to %s\n', model_filename);


% % === PREPARE TEST DATA ===
% CSP_FeatureMatrix = cell2mat(all_csp_features);
% CSP_Labels = cell2mat(all_csp_labels);
% 
% LOAD MODEL 
% fprintf('\nLoading pre-trained SVM classifier...\n');
% load('SVM_CSP_Classifier.mat', 'svm_model', 'selected_idx');
% 
% Select correct features
% test_features = CSP_FeatureMatrix(:, selected_idx);
% test_labels = CSP_Labels;
% 
% === EVALUATE ===
% fprintf('Evaluating classifier...\n');
% predicted_labels = predict(svm_model, test_features);
% accuracy = mean(predicted_labels == test_labels) * 100;
% 
% fprintf('\nSVM Classifier Accuracy on Current Data: %.2f%%\n', accuracy);
% 
% === CONFUSION MATRIX (Optional) ===
% figure;
% confusionchart(test_labels, predicted_labels);
% title('Confusion Matrix - Loaded SVM Classifier');xs
%% === Visualize Fisher Score Map for PSD Features ===
fprintf('Computing and Plotting Fisher Score Map for Channel x Frequency...\n');

% Get frequency bins and dimensions
freq_bins = all_frequencies(10:6:100); 
n_freq = length(freq_bins);
n_chan = nchannel;

% Preallocate matrices for classwise means and stds
mean_right = zeros(n_chan, n_freq);
std_right = zeros(n_chan, n_freq);
mean_arm = zeros(n_chan, n_freq);
std_arm = zeros(n_chan, n_freq);

% Separate trials by class
psd_right = {}; psd_arm = {};
for i = 1:numel(all_psd_features)
    trial = all_psd_features{i}; % size: [windows x frequencies x channels]
    trial_avg = squeeze(mean(trial,1)); % [frequencies x channels]
    if all_labels{i} == 1
        psd_right{end+1} = trial_avg;
    elseif all_labels{i} == 2
        psd_arm{end+1} = trial_avg;
    end
end

% Stack trials into 3D matrices: [trials x frequencies x channels]
psd_right_mat = cat(3, psd_right{:});
psd_arm_mat = cat(3, psd_arm{:});

% Transpose to [channels x frequencies x trials]
psd_right_mat = permute(psd_right_mat, [3 2 1]);
psd_arm_mat = permute(psd_arm_mat, [3 2 1]);

% Compute Fisher score per channel and frequency
F_map = zeros(n_chan, n_freq);
for ch = 1:n_chan
    for f = 1:n_freq
        r_vals = squeeze(psd_right_mat(ch,f,:));
        a_vals = squeeze(psd_arm_mat(ch,f,:));
        m1 = mean(r_vals); s1 = std(r_vals);
        m2 = mean(a_vals); s2 = std(a_vals);
        F_map(ch,f) = abs(m1 - m2) / sqrt(s1^2 + s2^2);
    end
end

% Plot Fisher Score Map
figure;
imagesc(freq_bins, 1:n_chan, F_map);
colorbar;
xlabel('Frequency (Hz)');
ylabel('Channel');
yticks(1:n_chan); yticklabels(channel_labels);
title('Fisher Score Map - Channel × Frequency');
axis xy;



